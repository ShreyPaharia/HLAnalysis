"""Probe an arrow→numpy→vectorised event-array build vs the current path.

Runs both implementations on real HL HIP-4 data for one leg and compares the
*final hftbacktest depth state at each snapshot ts* (since CLEAR+SET and
per-level qty=0 produce semantically equivalent depth state but different
intermediate event sequences).

Used to verify safety of the fast path before plumbing it into the data source.
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np

from hftbacktest.types import (
    BUY_EVENT, DEPTH_CLEAR_EVENT, DEPTH_EVENT, EXCH_EVENT, LOCAL_EVENT,
    SELL_EVENT, TRADE_EVENT, event_dtype,
)

from hlanalysis.backtest.core.events import BookSnapshot, TradeEvent
from hlanalysis.backtest.runner.hftbt_runner import _build_leg_event_array


def _read_columns(con, glob: str, date_list, start_ns: int, end_ns: int):
    tbl = con.sql(f"""
        SELECT exchange_ts, bid_px, bid_sz, ask_px, ask_sz
        FROM read_parquet('{glob}', hive_partitioning=1)
        WHERE date IN ({','.join(repr(d) for d in date_list)})
          AND exchange_ts >= {start_ns} AND exchange_ts < {end_ns}
        ORDER BY exchange_ts
    """).to_arrow_table()
    bid_px = tbl["bid_px"].combine_chunks()
    bid_sz = tbl["bid_sz"].combine_chunks()
    ask_px = tbl["ask_px"].combine_chunks()
    ask_sz = tbl["ask_sz"].combine_chunks()
    return dict(
        ts=tbl["exchange_ts"].to_numpy(),
        bid_px=bid_px.flatten().to_numpy(),
        bid_sz=bid_sz.flatten().to_numpy(),
        bid_offsets=bid_px.offsets.to_numpy(),
        ask_px=ask_px.flatten().to_numpy(),
        ask_sz=ask_sz.flatten().to_numpy(),
        ask_offsets=ask_px.offsets.to_numpy(),
    )


def _read_trade_columns(con, glob: str, date_list, start_ns: int, end_ns: int):
    tbl = con.sql(f"""
        SELECT exchange_ts, price, size, side
        FROM read_parquet('{glob}', hive_partitioning=1)
        WHERE date IN ({','.join(repr(d) for d in date_list)})
          AND exchange_ts >= {start_ns} AND exchange_ts < {end_ns}
        ORDER BY exchange_ts
    """).to_arrow_table()
    return dict(
        ts=tbl["exchange_ts"].to_numpy(),
        px=tbl["price"].to_numpy(),
        sz=tbl["size"].to_numpy(),
        side=np.asarray(tbl["side"].to_pylist()),
    )


def _build_from_columns(book_cols: dict, trade_cols: dict) -> np.ndarray:
    """Fast-path builder: CLEAR + SET semantics, fully vectorised."""
    flag = EXCH_EVENT | LOCAL_EVENT
    ts = book_cols["ts"]
    bid_px = book_cols["bid_px"]
    bid_sz = book_cols["bid_sz"]
    bid_off = book_cols["bid_offsets"].astype(np.int64)
    ask_px = book_cols["ask_px"]
    ask_sz = book_cols["ask_sz"]
    ask_off = book_cols["ask_offsets"].astype(np.int64)

    n_snaps = len(ts)
    n_bid = len(bid_px)
    n_ask = len(ask_px)

    # Clears: 1 bid clear + 1 ask clear per snapshot, both at snap ts.
    clear_n = n_snaps * 2
    clear_ts = np.repeat(ts, 2)
    clear_ev = np.empty(clear_n, dtype=np.uint64)
    clear_ev[0::2] = DEPTH_CLEAR_EVENT | flag | BUY_EVENT
    clear_ev[1::2] = DEPTH_CLEAR_EVENT | flag | SELL_EVENT

    bid_lens = np.diff(bid_off)
    ask_lens = np.diff(ask_off)
    bid_set_ts = np.repeat(ts, bid_lens)
    ask_set_ts = np.repeat(ts, ask_lens)
    bid_set_ev = np.full(n_bid, DEPTH_EVENT | flag | BUY_EVENT, dtype=np.uint64)
    ask_set_ev = np.full(n_ask, DEPTH_EVENT | flag | SELL_EVENT, dtype=np.uint64)

    trade_ts = trade_cols["ts"]
    trade_px = trade_cols["px"]
    trade_sz = trade_cols["sz"]
    trade_side = trade_cols["side"]
    n_trades = len(trade_ts)
    if n_trades > 0:
        trade_ev = np.where(
            trade_side == "sell",
            np.uint64(TRADE_EVENT | flag | SELL_EVENT),
            np.uint64(TRADE_EVENT | flag | BUY_EVENT),
        ).astype(np.uint64)
    else:
        trade_ev = np.zeros(0, dtype=np.uint64)

    total = clear_n + n_bid + n_ask + n_trades
    if total == 0:
        return np.zeros(0, dtype=event_dtype)

    out_ts = np.concatenate([clear_ts, bid_set_ts, ask_set_ts, trade_ts])
    out_ev = np.concatenate([clear_ev, bid_set_ev, ask_set_ev, trade_ev])
    out_px = np.concatenate([
        np.zeros(clear_n, dtype=np.float64), bid_px, ask_px, trade_px,
    ])
    out_qty = np.concatenate([
        np.zeros(clear_n, dtype=np.float64), bid_sz, ask_sz, trade_sz,
    ])

    arr = np.zeros(total, dtype=event_dtype)
    arr["exch_ts"] = out_ts
    arr["local_ts"] = out_ts
    arr["ev"] = out_ev
    arr["px"] = out_px
    arr["qty"] = out_qty
    # Stable sort by exch_ts: at equal ts, input order (clears, bids, asks, trades) is preserved.
    arr = arr[np.argsort(arr["exch_ts"], kind="stable")]
    return arr


def _depth_state_at_each_snapshot(events: np.ndarray, ts_to_check: np.ndarray) -> list[dict]:
    """Replay events through a Python depth model; record state at each ts_to_check.

    Returns a list of {"ts", "bids": {px: qty}, "asks": {px: qty}} dicts.
    """
    bid_state: dict[float, float] = {}
    ask_state: dict[float, float] = {}
    out = []
    idx = 0
    n_ev = len(events)
    for target in ts_to_check:
        while idx < n_ev and events[idx]["exch_ts"] <= target:
            ev = int(events[idx]["ev"])
            ts = int(events[idx]["exch_ts"])
            px = float(events[idx]["px"])
            qty = float(events[idx]["qty"])
            is_clear = (ev & DEPTH_CLEAR_EVENT) == DEPTH_CLEAR_EVENT
            is_depth = (ev & DEPTH_EVENT) == DEPTH_EVENT
            is_trade = (ev & TRADE_EVENT) == TRADE_EVENT
            is_buy = (ev & BUY_EVENT) == BUY_EVENT
            if is_clear:
                if is_buy:
                    bid_state.clear()
                else:
                    ask_state.clear()
            elif is_depth and not is_trade:
                state = bid_state if is_buy else ask_state
                if qty == 0.0:
                    state.pop(px, None)
                else:
                    state[px] = qty
            # Trades don't update the book.
            idx += 1
        out.append({
            "ts": int(target),
            "bids": dict(bid_state),
            "asks": dict(ask_state),
        })
    return out


def main():
    root = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data")
    leg = "#300"
    glob_book = str(root / f"venue=hyperliquid/product_type=prediction_binary/mechanism=clob/event=book_snapshot/symbol={leg}/**/*.parquet")
    glob_trade = str(root / f"venue=hyperliquid/product_type=prediction_binary/mechanism=clob/event=trade/symbol={leg}/**/*.parquet")
    date_list = ["2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15", "2026-05-16"]
    start_ns = 1778565618000000000
    end_ns = 1778652000000000000

    con = duckdb.connect()
    # Read with both methods.
    print("Reading columns...")
    book_cols = _read_columns(con, glob_book, date_list, start_ns, end_ns)
    trade_cols = _read_trade_columns(con, glob_trade, date_list, start_ns, end_ns)
    print(f"  snaps={len(book_cols['ts'])}, bid_levels={len(book_cols['bid_px'])}, trades={len(trade_cols['ts'])}")

    # Read for legacy path.
    print("Reading via fetchall + dataclass...")
    rows_book = con.sql(f"""SELECT exchange_ts, bid_px, bid_sz, ask_px, ask_sz
        FROM read_parquet('{glob_book}', hive_partitioning=1)
        WHERE date IN ({','.join(repr(d) for d in date_list)})
          AND exchange_ts >= {start_ns} AND exchange_ts < {end_ns}
        ORDER BY exchange_ts""").fetchall()
    snaps = [BookSnapshot(int(r[0]), leg,
                          tuple(zip(r[1] or [], r[2] or [], strict=False)),
                          tuple(zip(r[3] or [], r[4] or [], strict=False))) for r in rows_book]
    rows_tr = con.sql(f"""SELECT exchange_ts, price, size, side
        FROM read_parquet('{glob_trade}', hive_partitioning=1)
        WHERE date IN ({','.join(repr(d) for d in date_list)})
          AND exchange_ts >= {start_ns} AND exchange_ts < {end_ns}
        ORDER BY exchange_ts""").fetchall()
    trades = [TradeEvent(int(r[0]), leg, "buy" if r[3] != "sell" else "sell", float(r[1]), float(r[2])) for r in rows_tr]

    t0 = time.perf_counter()
    legacy = _build_leg_event_array(snaps, trades)
    t_legacy = time.perf_counter() - t0
    print(f"\nLegacy builder: {t_legacy:.3f}s, n_events={len(legacy)}")

    t0 = time.perf_counter()
    fast = _build_from_columns(book_cols, trade_cols)
    t_fast = time.perf_counter() - t0
    print(f"Fast builder:   {t_fast:.3f}s, n_events={len(fast)}")
    print(f"Speedup: {t_legacy/t_fast:.2f}x")

    # Sample 100 timestamps to verify depth state equivalence.
    rng = np.random.default_rng(0)
    snap_ts = book_cols["ts"]
    sample_ts = snap_ts[rng.choice(len(snap_ts), size=min(100, len(snap_ts)), replace=False)]
    sample_ts = np.sort(sample_ts)

    state_legacy = _depth_state_at_each_snapshot(legacy, sample_ts)
    state_fast = _depth_state_at_each_snapshot(fast, sample_ts)

    matches = 0
    diffs = []
    for sl, sf in zip(state_legacy, state_fast):
        if sl["bids"] == sf["bids"] and sl["asks"] == sf["asks"]:
            matches += 1
        else:
            diffs.append((sl, sf))
    print(f"\nDepth-state matches: {matches}/{len(state_legacy)}")
    if diffs:
        print("First mismatch:")
        sl, sf = diffs[0]
        print(f"  ts={sl['ts']}")
        print(f"  legacy bids={sl['bids']}")
        print(f"  fast   bids={sf['bids']}")
        print(f"  legacy asks={sl['asks']}")
        print(f"  fast   asks={sf['asks']}")


if __name__ == "__main__":
    main()
