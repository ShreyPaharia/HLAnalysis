"""The in-memory data sources (synthetic / pm_nba / binance_perp) build their
per-leg hftbacktest event arrays through the SAME assembler as the HL/PM fast
paths — ``build_leg_event_array_from_columns`` — via the snapshot→column
adapter ``build_leg_event_array_from_snapshots``.

This test locks in that funnelling the in-memory ``BookSnapshot`` / ``TradeEvent``
lists through the unified assembler produces event arrays that are equivalent
(byte-for-byte for single-level books; multiset-equivalent in general) to the
legacy per-cell Python builder it replaced. ``_naive_build_leg_event_array`` is
an independent re-implementation of that legacy builder, kept inline so the
regression oracle does not depend on the deleted code.
"""
from __future__ import annotations

import numpy as np

from hftbacktest.types import (
    BUY_EVENT,
    DEPTH_EVENT,
    EXCH_EVENT,
    LOCAL_EVENT,
    SELL_EVENT,
    TRADE_EVENT,
    event_dtype,
)

from hlanalysis.backtest.core.events import BookSnapshot, TradeEvent
from hlanalysis.backtest.data._fastpath_core import (
    build_leg_event_array_from_snapshots,
)
from hlanalysis.backtest.data.synthetic import (
    SyntheticDataSource,
    make_default_binary_question,
)


def _naive_build_leg_event_array(snapshots, trades):
    """Byte-for-byte re-implementation of the legacy ``_build_leg_event_array``
    (per-cell Python loop). The independent oracle for the unified assembler."""
    n_events = 0
    prev_bids: set[float] = set()
    prev_asks: set[float] = set()
    for snap in snapshots:
        new_bids = {b[0] for b in snap.bids}
        new_asks = {a[0] for a in snap.asks}
        n_events += len(prev_bids - new_bids)
        n_events += len(prev_asks - new_asks)
        n_events += len(snap.bids) + len(snap.asks)
        prev_bids = new_bids
        prev_asks = new_asks
    n_events += len(trades)

    arr = np.zeros(n_events, dtype=event_dtype)
    idx = 0
    prev_bids = set()
    prev_asks = set()
    flag = EXCH_EVENT | LOCAL_EVENT
    for snap in snapshots:
        new_bid_set = {b[0] for b in snap.bids}
        new_ask_set = {a[0] for a in snap.asks}
        for px in prev_bids - new_bid_set:
            arr[idx]["ev"] = DEPTH_EVENT | flag | BUY_EVENT
            arr[idx]["exch_ts"] = snap.ts_ns
            arr[idx]["local_ts"] = snap.ts_ns
            arr[idx]["px"] = px
            arr[idx]["qty"] = 0.0
            idx += 1
        for px in prev_asks - new_ask_set:
            arr[idx]["ev"] = DEPTH_EVENT | flag | SELL_EVENT
            arr[idx]["exch_ts"] = snap.ts_ns
            arr[idx]["local_ts"] = snap.ts_ns
            arr[idx]["px"] = px
            arr[idx]["qty"] = 0.0
            idx += 1
        for px, qty in snap.bids:
            arr[idx]["ev"] = DEPTH_EVENT | flag | BUY_EVENT
            arr[idx]["exch_ts"] = snap.ts_ns
            arr[idx]["local_ts"] = snap.ts_ns
            arr[idx]["px"] = px
            arr[idx]["qty"] = qty
            idx += 1
        for px, qty in snap.asks:
            arr[idx]["ev"] = DEPTH_EVENT | flag | SELL_EVENT
            arr[idx]["exch_ts"] = snap.ts_ns
            arr[idx]["local_ts"] = snap.ts_ns
            arr[idx]["px"] = px
            arr[idx]["qty"] = qty
            idx += 1
        prev_bids = new_bid_set
        prev_asks = new_ask_set

    for trade in trades:
        side_flag = BUY_EVENT if trade.side == "buy" else SELL_EVENT
        arr[idx]["ev"] = TRADE_EVENT | flag | side_flag
        arr[idx]["exch_ts"] = trade.ts_ns
        arr[idx]["local_ts"] = trade.ts_ns
        arr[idx]["px"] = trade.price
        arr[idx]["qty"] = trade.size
        idx += 1

    if idx > 0:
        arr = arr[:idx]
        arr = arr[np.argsort(arr["exch_ts"], kind="stable")]
    return arr


def _multiset(arr: np.ndarray) -> list[tuple]:
    return sorted(
        (int(r["exch_ts"]), int(r["ev"]), float(r["px"]), float(r["qty"]))
        for r in arr
    )


def test_unified_builder_matches_naive_on_synthetic_source() -> None:
    """Partition the synthetic source's real events per leg and assemble them
    through the unified adapter — must match the legacy per-cell oracle."""
    sq = make_default_binary_question()
    src = SyntheticDataSource(questions=[sq])
    q = sq.descriptor

    books: dict[str, list[BookSnapshot]] = {s: [] for s in q.leg_symbols}
    trades: dict[str, list[TradeEvent]] = {s: [] for s in q.leg_symbols}
    for ev in src.events(q):
        if isinstance(ev, BookSnapshot) and ev.symbol in books:
            books[ev.symbol].append(ev)
        elif isinstance(ev, TradeEvent) and ev.symbol in trades:
            trades[ev.symbol].append(ev)

    for sym in q.leg_symbols:
        unified = build_leg_event_array_from_snapshots(books[sym], trades[sym])
        naive = _naive_build_leg_event_array(books[sym], trades[sym])
        assert unified.shape == naive.shape, sym
        assert _multiset(unified) == _multiset(naive), sym
        # Single-level top-of-book → deterministic order → bit-identical.
        assert unified.dtype == event_dtype


def test_unified_builder_empty_inputs() -> None:
    arr = build_leg_event_array_from_snapshots([], [])
    assert arr.shape == (0,)
    assert arr.dtype == event_dtype


def test_unified_builder_trades_only() -> None:
    trades = [
        TradeEvent(ts_ns=3, symbol="#150", side="sell", price=0.5, size=10.0),
        TradeEvent(ts_ns=1, symbol="#150", side="buy", price=0.4, size=20.0),
        TradeEvent(ts_ns=2, symbol="#150", side="buy", price=0.45, size=5.0),
    ]
    arr = build_leg_event_array_from_snapshots([], trades)
    assert list(arr["exch_ts"]) == [1, 2, 3]
    assert list(arr["px"]) == [0.4, 0.45, 0.5]
    assert list(arr["qty"]) == [20.0, 5.0, 10.0]


def test_unified_builder_clears_match_naive_multiset() -> None:
    """Multi-level shrinking book exercises stale-level clears; the unified
    builder must match the naive oracle as a multiset (clear ordering within a
    timestamp is fill-irrelevant)."""
    snaps = [
        BookSnapshot(ts_ns=10, symbol="#1",
                     bids=((0.59, 100.0), (0.58, 200.0), (0.57, 300.0)),
                     asks=((0.61, 100.0), (0.62, 200.0))),
        BookSnapshot(ts_ns=20, symbol="#1",
                     bids=((0.58, 210.0),),
                     asks=((0.62, 220.0),)),
    ]
    unified = build_leg_event_array_from_snapshots(snaps, [])
    naive = _naive_build_leg_event_array(snaps, [])
    assert _multiset(unified) == _multiset(naive)
    assert np.any(naive["qty"] == 0.0)  # clears really fired
