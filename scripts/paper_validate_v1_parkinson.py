"""Paper-validation: replay recorded HL HIP-4 ticks through the LIVE engine
code path (ReplayRunner → engine MarketState/Scanner-equivalent) with the
EXACT live v1 config, comparing the newly-activated Parkinson σ vs the old
(dormant) stdev behaviour.

This exercises the code changed on feat/engine-bbo-sigma-source-parkinson-fix:
the live engine MarketState now tracks per-bucket OHLC + recent_hl_bars, and
that is threaded into strategy.evaluate — so vol_estimator=parkinson actually
fires (it previously fell back to stdev live).

Run from the repo root that holds data/ :
    uv run --project .worktrees/feat/engine-bbo-sigma-source-parkinson-fix \
        python .worktrees/feat/engine-bbo-sigma-source-parkinson-fix/scripts/paper_validate_v1_parkinson.py
"""
from __future__ import annotations

import dataclasses
import glob
from collections import Counter
from pathlib import Path

import duckdb

from hlanalysis.engine.config import load_strategy_config
from hlanalysis.engine.replay import ReplayRunner
from hlanalysis.engine.runtime import (
    build_late_resolution_config,
    reference_sampling_dt_seconds,
    reference_vol_lookback_seconds,
)
from hlanalysis.events import (
    BboEvent, MarkEvent, Mechanism, ProductType, QuestionMetaEvent, TradeEvent,
)
from hlanalysis.strategy.late_resolution import LateResolutionStrategy
from hlanalysis.strategy.types import Action

HIP4 = "data/venue=hyperliquid/product_type=prediction_binary/mechanism=clob"
PERP_MARK = "data/venue=hyperliquid/product_type=perp/mechanism=clob/event=mark"
NS = 1_000_000_000

# Recent BTC priceBinary HIP-4 questions with full leg coverage (settled within
# the recorded window). (question_idx, outcome_idx, expiry 'YYYYMMDD-HHMM', strike)
QUESTIONS = [
    (1000085, 85, "20260524-0600", 74560.0),
    (1000090, 90, "20260525-0600", 76772.0),
    (1000095, 95, "20260526-0600", 77363.0),
    (1000105, 105, "20260527-0600", 76877.0),
    (1000111, 111, "20260528-0600", 75668.0),
    (1000116, 116, "20260529-0600", 72951.0),
    (1000121, 121, "20260530-0600", 73674.0),
]


def _expiry_ns(expiry: str) -> int:
    from datetime import datetime, timezone
    dt = datetime.strptime(expiry, "%Y%m%d-%H%M").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * NS)


def _read(con, glob_pat: str, cols: str, start: int, end: int) -> list:
    files = glob.glob(glob_pat, recursive=True)
    if not files:
        return []
    return con.sql(
        f"SELECT {cols} FROM read_parquet({files!r}) "
        f"WHERE exchange_ts >= {start} AND exchange_ts <= {end} ORDER BY exchange_ts"
    ).fetchall()


def _build_events(con, qidx: int, oi: int, expiry: str, strike: float):
    exp_ns = _expiry_ns(expiry)
    # Window: entry window (tte_max=7200s) + σ lookback (3600s) warmup + margin.
    start = exp_ns - 4 * 3600 * NS
    end = exp_ns
    yes, no = f"#{10 * oi}", f"#{10 * oi + 1}"

    evs: list = []
    # Reference σ feed — HL perp BTC mark (sub-second).
    for ts, px in _read(con, f"{PERP_MARK}/**/*.parquet", "exchange_ts, mark_px", start, end):
        evs.append(MarkEvent(
            venue="hyperliquid", product_type=ProductType.PERP,
            mechanism=Mechanism.CLOB, symbol="BTC",
            exchange_ts=int(ts), local_recv_ts=int(ts), mark_px=float(px),
        ))
    # Leg books + trades.
    for leg in (yes, no):
        for ts, bpx, bsz, apx, asz in _read(
            con, f"{HIP4}/event=bbo/symbol={leg}/**/*.parquet",
            "exchange_ts, bid_px, bid_sz, ask_px, ask_sz", start, end,
        ):
            if bpx is None or apx is None:
                continue
            evs.append(BboEvent(
                venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
                mechanism=Mechanism.CLOB, symbol=leg,
                exchange_ts=int(ts), local_recv_ts=int(ts),
                bid_px=float(bpx), bid_sz=float(bsz or 0.0),
                ask_px=float(apx), ask_sz=float(asz or 0.0),
            ))
        for ts, px, sz, side in _read(
            con, f"{HIP4}/event=trade/symbol={leg}/**/*.parquet",
            "exchange_ts, price, size, side", start, end,
        ):
            evs.append(TradeEvent(
                venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
                mechanism=Mechanism.CLOB, symbol=leg,
                exchange_ts=int(ts), local_recv_ts=int(ts),
                price=float(px), size=float(sz), side=str(side or "buy"),
            ))
    # Question registration up front.
    evs.append(QuestionMetaEvent(
        venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol=f"Q{qidx}",
        exchange_ts=start, local_recv_ts=start,
        question_idx=qidx, named_outcome_idxs=[oi],
        keys=["class", "underlying", "expiry", "strike"],
        values=["priceBinary", "BTC", expiry, str(strike)],
    ))
    evs.sort(key=lambda e: e.exchange_ts or e.local_recv_ts)
    return evs


def _run(events, base_cfg, estimator: str, dt: int, lookback: int):
    cfg = dataclasses.replace(base_cfg, vol_estimator=estimator)
    runner = ReplayRunner(
        strategy=LateResolutionStrategy(cfg),
        reference_symbol="BTC", sampling_dt_seconds=dt,
    )
    # Match live-Scanner fidelity exactly: grow the σ history deque to cover the
    # lookback at this cadence, and request the same number of bars the live
    # Scanner._required_returns_n would.
    runner._market.set_reference_cadence("BTC", sampling_dt_seconds=dt, lookback_seconds=lookback)
    runner._recent_returns_n = max(32, (lookback + dt - 1) // dt)
    reasons: Counter = Counter()
    enters = 0
    for d in runner.run_iter(events):
        reasons[d.diagnostics[0].message if d.diagnostics else "?"] += 1
        if d.action is Action.ENTER:
            enters += 1
    # Concrete σ at the last evaluated state (range vs close-to-close).
    n = runner._recent_returns_n
    rets = runner._market.recent_returns("BTC", n=n)
    hl = runner._market.recent_hl_bars("BTC", n=n)
    strat = runner._strategy
    n_keep = max(2, lookback // dt)
    rw = rets[-n_keep:] if len(rets) > n_keep else rets
    hlw = hl[-n_keep:] if len(hl) > n_keep else hl
    sigma = strat._sigma(rw, hlw) if rw else float("nan")
    return enters, reasons, sigma, len(hl)


def main() -> None:
    cfg = load_strategy_config(Path("config/strategy.yaml"))
    assert cfg.name == "late_resolution", f"expected v1, got {cfg.name}"
    base_cfg = build_late_resolution_config(cfg)
    assert base_cfg.vol_estimator == "parkinson", base_cfg.vol_estimator
    dt = reference_sampling_dt_seconds(cfg)
    lookback = reference_vol_lookback_seconds(cfg)
    print(f"v1 live config: vol_estimator={base_cfg.vol_estimator} dt={dt}s "
          f"lookback={lookback}s min_safety_d={base_cfg.min_safety_d} "
          f"ewma_lambda={base_cfg.vol_ewma_lambda} "
          f"thr=[{base_cfg.price_extreme_threshold},{base_cfg.price_extreme_max}] "
          f"use_bid_gate={base_cfg.use_bid_for_entry_gate}")
    print(f"recent_returns_n (live-equiv) = {max(32, (lookback + dt - 1)//dt)}\n")

    con = duckdb.connect()
    tot_p = tot_s = 0
    for qidx, oi, expiry, strike in QUESTIONS:
        evs = _build_events(con, qidx, oi, expiry, strike)
        n_marks = sum(1 for e in evs if isinstance(e, MarkEvent))
        n_bbo = sum(1 for e in evs if isinstance(e, BboEvent))
        ep, rp, sp, nbars = _run(evs, base_cfg, "parkinson", dt, lookback)
        es, rs, ss, _ = _run(evs, base_cfg, "stdev", dt, lookback)
        tot_p += ep
        tot_s += es
        print(f"Q{qidx} exp={expiry} strike={strike:.0f} "
              f"events={len(evs)} (marks={n_marks} bbo={n_bbo}) bars={nbars}")
        print(f"   σ_parkinson={sp:.6f}  σ_stdev={ss:.6f}  ratio={sp/ss if ss else float('nan'):.2f}×")
        print(f"   ENTER: parkinson={ep}  stdev={es}")
        print(f"   parkinson reasons: {dict(rp)}")
        print(f"   stdev     reasons: {dict(rs)}")
        print()
    print("=" * 70)
    print(f"TOTAL ENTER decisions  parkinson={tot_p}  stdev={tot_s}")
    print("Parkinson is ACTIVE on the live path iff σ_parkinson ≠ σ_stdev above\n"
          "(would have been identical under the dormant-stdev bug).")


if __name__ == "__main__":
    main()
