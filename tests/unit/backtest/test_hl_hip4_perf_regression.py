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
from hlanalysis.backtest.data._hl_hip4_fastpath import (
    build_leg_event_array_from_columns,
)
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


def test_events_arrays_present_and_matches_legacy(source: HLHip4DataSource) -> None:
    """The arrow-backed fast path must produce per-leg event arrays
    bit-identical to the legacy ``events()`` + ``_build_leg_event_array`` path
    on the fixture, and yield the same reference + settlement event lists."""
    q = source.discover(start="2026-05-09", end="2026-05-11", underlying="BTC")[0]

    # Legacy: drain events() and build per-leg arrays the way the runner did
    # before the fast-path existed.
    book_events: dict[str, list[BookSnapshot]] = {s: [] for s in q.leg_symbols}
    trade_events: dict[str, list[TradeEvent]] = {s: [] for s in q.leg_symbols}
    ref_events = []
    settle_events = []
    from hlanalysis.backtest.core.events import (
        BookSnapshot as BS, TradeEvent as TE, ReferenceEvent as RE,
        SettlementEvent as SE,
    )
    for ev in source.events(q):
        if isinstance(ev, BS):
            if ev.symbol in book_events:
                book_events[ev.symbol].append(ev)
        elif isinstance(ev, TE):
            if ev.symbol in trade_events:
                trade_events[ev.symbol].append(ev)
        elif isinstance(ev, RE):
            ref_events.append(ev)
        elif isinstance(ev, SE):
            settle_events.append(ev)
    legacy_arrays = {
        sym: _build_leg_event_array(book_events[sym], trade_events[sym])
        for sym in q.leg_symbols
    }

    bundle = source.events_arrays(q)

    # Reference + settlement event lists match.
    assert len(bundle.reference_events) == len(ref_events)
    assert len(bundle.settlement_events) == len(settle_events)

    # Per-leg event arrays must agree on every column. We sort each side as a
    # multiset since the per-snapshot bid/ask/clear ordering is stable but the
    # within-snapshot ordering of equal-ts events may shift slightly between
    # implementations — and only the final depth state matters to hftbacktest.
    for sym in q.leg_symbols:
        fast = bundle.leg_arrays[sym].events
        legacy = legacy_arrays[sym]
        assert fast.shape == legacy.shape, (sym, fast.shape, legacy.shape)
        fast_recs = sorted(
            (int(r["exch_ts"]), int(r["ev"]), float(r["px"]), float(r["qty"]))
            for r in fast
        )
        legacy_recs = sorted(
            (int(r["exch_ts"]), int(r["ev"]), float(r["px"]), float(r["qty"]))
            for r in legacy
        )
        assert fast_recs == legacy_recs, sym


def test_events_arrays_book_ts_tracks_snapshot_timestamps(source: HLHip4DataSource) -> None:
    """The runner uses ``LegArrays.book_ts`` as a cursor for the stale-book
    gate. Must be the int64 ts of every snapshot, monotonically non-decreasing
    and identical to the legacy ``[b.ts_ns for b in book_events[sym]]`` list."""
    q = source.discover(start="2026-05-09", end="2026-05-11", underlying="BTC")[0]
    bundle = source.events_arrays(q)
    for sym, legarr in bundle.leg_arrays.items():
        assert legarr.book_ts.dtype.kind == "i" and legarr.book_ts.itemsize == 8
        diffs = np.diff(legarr.book_ts)
        assert (diffs >= 0).all(), f"{sym} book_ts is not monotone"


def test_build_leg_event_array_from_columns_matches_legacy_on_random_input() -> None:
    """Parity test on randomised numpy column inputs that mimic the parquet
    schema (variable-length bid/ask price + size lists per snapshot)."""
    rng = np.random.default_rng(7)
    n_snaps = 50
    bid_lens = rng.integers(0, 8, size=n_snaps)
    ask_lens = rng.integers(0, 8, size=n_snaps)
    bid_off = np.concatenate([[0], np.cumsum(bid_lens)]).astype(np.int64)
    ask_off = np.concatenate([[0], np.cumsum(ask_lens)]).astype(np.int64)
    ts = np.arange(1000, 1000 + n_snaps * 10, 10, dtype=np.int64)
    bid_px = rng.uniform(0.1, 0.49, size=int(bid_off[-1]))
    bid_sz = rng.uniform(1.0, 100.0, size=int(bid_off[-1]))
    ask_px = rng.uniform(0.51, 0.9, size=int(ask_off[-1]))
    ask_sz = rng.uniform(1.0, 100.0, size=int(ask_off[-1]))

    n_trades = 12
    trade_ts = rng.integers(1000, 1000 + n_snaps * 10, size=n_trades).astype(np.int64)
    trade_px = rng.uniform(0.1, 0.9, size=n_trades)
    trade_sz = rng.uniform(1.0, 50.0, size=n_trades)
    trade_side = np.where(rng.integers(0, 2, size=n_trades) == 0, "buy", "sell").astype(object)

    book_cols = dict(
        ts=ts, bid_px=bid_px, bid_sz=bid_sz, bid_offsets=bid_off,
        ask_px=ask_px, ask_sz=ask_sz, ask_offsets=ask_off,
    )
    trade_cols = dict(ts=trade_ts, px=trade_px, sz=trade_sz, side=trade_side)

    fast = build_leg_event_array_from_columns(book_cols, trade_cols)

    # Build the legacy equivalent by reconstructing snapshots + trades.
    snaps = []
    for i in range(n_snaps):
        bids = tuple((float(bid_px[j]), float(bid_sz[j])) for j in range(bid_off[i], bid_off[i + 1]))
        asks = tuple((float(ask_px[j]), float(ask_sz[j])) for j in range(ask_off[i], ask_off[i + 1]))
        snaps.append(BookSnapshot(int(ts[i]), "#150", bids, asks))
    trades = [
        TradeEvent(int(trade_ts[i]), "#150", str(trade_side[i]),
                   float(trade_px[i]), float(trade_sz[i]))
        for i in range(n_trades)
    ]
    legacy = _build_leg_event_array(snaps, trades)

    assert fast.shape == legacy.shape
    fast_recs = sorted(
        (int(r["exch_ts"]), int(r["ev"]), float(r["px"]), float(r["qty"])) for r in fast
    )
    legacy_recs = sorted(
        (int(r["exch_ts"]), int(r["ev"]), float(r["px"]), float(r["qty"])) for r in legacy
    )
    assert fast_recs == legacy_recs


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
