"""Regression tests for the HL HIP-4 perf rework (perf/backtest-hl-hip4-speedup).

Locks in the shared-connection ``events()`` contract and the vectorised
``_build_leg_event_array`` output, so future edits can't silently regress fill
equivalence.

The fixture is the 2-hour committed slice of Q1000015 — see
``tests/fixtures/hl_hip4/README.md``.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from hlanalysis.backtest.core.events import BookSnapshot, TradeEvent
from hlanalysis.backtest.data.hl_hip4 import HLHip4DataSource
from hlanalysis.backtest.runner.hftbt_runner import _build_leg_event_array


FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "hl_hip4"


@pytest.fixture(scope="module")
def source() -> HLHip4DataSource:
    return HLHip4DataSource(data_root=FIXTURE_ROOT)


def test_events_opens_single_connection(source: HLHip4DataSource) -> None:
    """One `events()` call must open exactly one DuckDB connection — the perf
    rework's central invariant. Reverting to per-iter `duckdb.connect()` would
    silently re-introduce 14-23 connections per bucket question (one of the two
    headline regressions this branch fixed)."""
    q = source.discover(start="2026-05-09", end="2026-05-11", underlying="BTC")[0]
    count = [0]
    real_connect = duckdb.connect

    def counting_connect(*a, **kw):
        count[0] += 1
        return real_connect(*a, **kw)

    with patch.object(duckdb, "connect", counting_connect):
        # Consume the iterator fully so all underlying queries fire.
        for _ in source.events(q):
            pass

    assert count[0] == 1, (
        f"events() opened {count[0]} duckdb connections; expected exactly 1 "
        "(perf invariant — share one connection across leg iterators)."
    )


def test_resolved_outcome_opens_single_connection(source: HLHip4DataSource) -> None:
    """`resolved_outcome(q)` must also share one connection across its per-leg
    settlement probes plus the `_last_btc_ref_at_or_before` fallback. The
    fixture has no settlement rows so this exercises the fallback path."""
    q = source.discover(start="2026-05-09", end="2026-05-11", underlying="BTC")[0]
    count = [0]
    real_connect = duckdb.connect

    def counting_connect(*a, **kw):
        count[0] += 1
        return real_connect(*a, **kw)

    with patch.object(duckdb, "connect", counting_connect):
        outcome = source.resolved_outcome(q)
    # 1 for settlement probe + 1 for _last_btc_ref_at_or_before fallback.
    assert outcome in ("yes", "no")
    assert count[0] <= 2, (
        f"resolved_outcome() opened {count[0]} duckdb connections; "
        "expected ≤ 2 (one for settlement scan, optionally one for BTC ref fallback)."
    )


def test_build_leg_event_array_empty_inputs() -> None:
    arr = _build_leg_event_array([], [])
    assert arr.shape == (0,)


def test_build_leg_event_array_trades_only_ordering() -> None:
    """Trades-only input: every trade must appear in the output array, in
    monotone exch_ts order regardless of input order."""
    trades = [
        TradeEvent(ts_ns=3, symbol="#150", side="sell", price=0.5, size=10.0),
        TradeEvent(ts_ns=1, symbol="#150", side="buy", price=0.4, size=20.0),
        TradeEvent(ts_ns=2, symbol="#150", side="buy", price=0.45, size=5.0),
    ]
    arr = _build_leg_event_array([], trades)
    assert arr.shape == (3,)
    assert list(arr["exch_ts"]) == [1, 2, 3]
    assert list(arr["px"]) == [0.4, 0.45, 0.5]
    assert list(arr["qty"]) == [20.0, 5.0, 10.0]


def test_build_leg_event_array_book_clear_semantics() -> None:
    """Snapshot 1 puts a single bid level at 0.42. Snapshot 2 changes the bid
    to 0.43. The vectorised builder must emit (a) a qty=0 clear at 0.42 then
    (b) a qty=N set at 0.43 — semantically identical to the legacy per-cell
    loop."""
    snaps = [
        BookSnapshot(ts_ns=10, symbol="#150", bids=((0.42, 100.0),), asks=()),
        BookSnapshot(ts_ns=20, symbol="#150", bids=((0.43, 50.0),), asks=()),
    ]
    arr = _build_leg_event_array(snaps, [])
    # Expect: snap1 set@0.42, snap2 clear@0.42 (qty=0), snap2 set@0.43.
    assert arr.shape == (3,)
    by_ts = {int(r["exch_ts"]): (float(r["px"]), float(r["qty"])) for r in arr}
    # ts=10 contributes only the initial set
    assert by_ts[10] == (0.42, 100.0)
    # ts=20 has TWO entries: clear + set. Take the np.where slice.
    ts20_rows = arr[arr["exch_ts"] == 20]
    assert ts20_rows.shape == (2,)
    pxs = sorted((float(r["px"]), float(r["qty"])) for r in ts20_rows)
    assert pxs == [(0.42, 0.0), (0.43, 50.0)]


def test_build_leg_event_array_matches_naive_reference() -> None:
    """The vectorised implementation must produce row-for-row identical output
    to a naive per-cell reference implementation on a representative input
    (multiple snapshots, multiple levels, interleaved trades)."""
    rng = np.random.default_rng(42)
    snaps: list[BookSnapshot] = []
    for ts in range(100, 200, 5):
        n_bids = int(rng.integers(0, 4))
        n_asks = int(rng.integers(0, 4))
        bids = tuple(
            (float(0.4 - 0.001 * i), float(rng.uniform(1, 100))) for i in range(n_bids)
        )
        asks = tuple(
            (float(0.5 + 0.001 * i), float(rng.uniform(1, 100))) for i in range(n_asks)
        )
        snaps.append(BookSnapshot(ts_ns=ts, symbol="#150", bids=bids, asks=asks))
    trades = [
        TradeEvent(ts_ns=int(ts), symbol="#150", side="buy" if i % 2 else "sell",
                   price=float(rng.uniform(0.3, 0.6)), size=float(rng.uniform(1, 50)))
        for i, ts in enumerate(rng.integers(100, 200, size=20))
    ]

    arr = _build_leg_event_array(snaps, trades)
    naive = _naive_build_leg_event_array(snaps, trades)
    # Compare per-field after re-sorting (stable sort on identical keys can
    # interleave bid vs ask qty-0 clears in either order; compare as a sorted
    # multiset of records).
    arr_recs = sorted(
        (int(r["ev"]), int(r["exch_ts"]), float(r["px"]), float(r["qty"]))
        for r in arr
    )
    naive_recs = sorted(
        (int(r["ev"]), int(r["exch_ts"]), float(r["px"]), float(r["qty"]))
        for r in naive
    )
    assert arr_recs == naive_recs


def test_events_emits_consistent_order_with_shared_connection(source: HLHip4DataSource) -> None:
    """Sanity: the shared-connection events() still yields a strictly
    non-decreasing ts_ns stream and the correct event-type mix. Pre-rework this
    came for free from per-leg heapq.merge; the rework changes ONLY the
    connection sharing, but the test guards against accidentally collapsing
    the merge call."""
    q = source.discover(start="2026-05-09", end="2026-05-11", underlying="BTC")[0]
    n_book = n_trade = n_ref = 0
    prev_ts = -1
    for ev in source.events(q):
        assert ev.ts_ns >= prev_ts, (prev_ts, ev)
        prev_ts = ev.ts_ns
        from hlanalysis.backtest.core.events import (
            BookSnapshot as BS, TradeEvent as TE, ReferenceEvent as RE,
        )
        if isinstance(ev, BS):
            n_book += 1
        elif isinstance(ev, TE):
            n_trade += 1
        elif isinstance(ev, RE):
            n_ref += 1
    assert n_book > 1000 and n_trade > 100 and n_ref > 1000


# ---------------------------------------------------------------------------
# Naive reference for parity testing — intentionally per-cell to match the
# legacy implementation byte-for-byte. Kept inline so the regression test is
# self-contained.
# ---------------------------------------------------------------------------


def _naive_build_leg_event_array(snapshots, trades):
    from hftbacktest.types import (
        BUY_EVENT, DEPTH_EVENT, EXCH_EVENT, LOCAL_EVENT,
        SELL_EVENT, TRADE_EVENT, event_dtype,
    )

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
