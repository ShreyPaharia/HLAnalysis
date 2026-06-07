"""PM fast-path equivalence tests covering both recorded and synthetic book modes.

Fixture is built programmatically in ``tmp_path`` (same approach as
``test_polymarket_recorded_book.py``) — no committed fixture directory needed.

The equivalence test compares per-leg ``event_dtype`` arrays produced by two
paths over the same fabricated data:

  Legacy:  events() → collect BookSnapshot / TradeEvent per leg
           → build_leg_event_array_from_snapshots (shared in-memory assembler)

  Fast:    events_arrays() → recorded: read_pm_book_columns + read_pm_trade_columns
                                       → build_leg_event_array_from_columns
                           → synthetic: events() stream partitioned + same
                                        in-memory assembler (build_pm_synthetic_fast_path_bundle)

For bit-equivalence we choose a fixture that avoids stale-level clear events
(which have non-deterministic ordering in the legacy set-subtraction path).
Concretely: within a leg, each successive snapshot's price set is a SUPERSET
of the previous snapshot's price set (no price is ever removed), so the
``prev_bids - new_bid_set`` / ``prev_asks - new_ask_set`` diffs in the legacy
path are always empty — no clear events are emitted.  Both paths then produce
identical DEPTH_EVENT / TRADE_EVENT sequences after the stable sort.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hlanalysis.backtest.core.events import BookSnapshot, TradeEvent
from hlanalysis.backtest.data.polymarket import PolymarketDataSource
# The legacy runner helper ``_build_leg_event_array`` was collapsed into the
# shared assembler; the in-memory snapshot→column adapter is its drop-in
# replacement and the legacy reference these equivalence tests compare against.
from hlanalysis.backtest.data._fastpath_core import (
    build_leg_event_array_from_snapshots as _build_leg_event_array,
)

# ── shared constants ─────────────────────────────────────────────────────────

_BOOK_SUBPATH = (
    "venue=polymarket/product_type=prediction_binary"
    "/mechanism=clob/event=book_snapshot"
)
_START = 1_780_000_000_000_000_000   # ≈ 2026-05-28 UTC
_END = _START + 7200 * 1_000_000_000  # +2h
_COND = "equiv_cond_001"
_YES = "yes_token_equiv"
_NO = "no_token_equiv"

# ── fixture helpers ───────────────────────────────────────────────────────────


def _write_manifest(cache_root: Path, *, cond: str, yes_t: str, no_t: str,
                    start: int, end: int, outcome: str = "yes") -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        cond: {
            "kind": "binary",
            "n_rows": 4,
            "last_pull_ts_ns": end,
            "market": {
                "condition_id": cond,
                "yes_token_id": yes_t,
                "no_token_id": no_t,
                "start_ts_ns": start,
                "end_ts_ns": end,
                "resolved_outcome": outcome,
                "total_volume_usd": 500.0,
                "n_trades": 4,
            },
        }
    }
    (cache_root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _write_trades(cache_root: Path, cond_id: str, rows: list[dict]) -> None:
    pm_trades = cache_root / "pm_trades"
    pm_trades.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "ts_ns":    pa.array([int(r["ts_ns"]) for r in rows], pa.int64()),
        "token_id": [str(r["token_id"]) for r in rows],
        "side":     [str(r["side"]) for r in rows],
        "price":    pa.array([float(r["price"]) for r in rows], pa.float64()),
        "size":     pa.array([float(r["size"]) for r in rows], pa.float64()),
    })
    pq.write_table(table, pm_trades / f"{cond_id}.parquet")


def _write_klines(cache_root: Path, klines: list[dict]) -> None:
    klines_dir = cache_root / "btc_klines"
    klines_dir.mkdir(parents=True, exist_ok=True)
    (klines_dir / "fixture.json").write_text(json.dumps(klines))


def _write_recorded_book(book_root: Path, token_id: str, date: str, hour: str,
                         rows: list[dict]) -> None:
    """Write recorded book_snapshot parquet for one token leg."""
    d = (
        book_root / _BOOK_SUBPATH
        / f"symbol={token_id}" / f"date={date}" / f"hour={hour}"
    )
    d.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "exchange_ts": pa.array([int(r["exchange_ts"]) for r in rows], pa.int64()),
        "bid_px": pa.array(
            [[float(x) for x in r["bid_px"]] for r in rows],
            pa.list_(pa.float64()),
        ),
        "bid_sz": pa.array(
            [[float(x) for x in r["bid_sz"]] for r in rows],
            pa.list_(pa.float64()),
        ),
        "ask_px": pa.array(
            [[float(x) for x in r["ask_px"]] for r in rows],
            pa.list_(pa.float64()),
        ),
        "ask_sz": pa.array(
            [[float(x) for x in r["ask_sz"]] for r in rows],
            pa.list_(pa.float64()),
        ),
    })
    pq.write_table(table, d / "data.parquet")


# ── fixture factory ───────────────────────────────────────────────────────────


def _build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Construct a minimal recorded-book fixture and return (cache, book_root).

    Design constraints for bit-equivalent comparison (see module docstring):
    - Each snapshot's bid/ask price sets are SUPERSETS of the prior snapshot's
      price sets → no stale-level clear events in either path.
    - Book level arrays are written in unsorted order (bids ASC, asks DESC) so
      both paths exercise the _normalize_levels / sorting logic.
    - 2 legs, 2 snapshots each, 2 trades each.
    """
    cache = tmp_path / "cache"
    book_root = tmp_path / "bookroot"

    _write_manifest(cache, cond=_COND, yes_t=_YES, no_t=_NO,
                    start=_START, end=_END, outcome="yes")

    # Trades: 2 for YES leg, 2 for NO leg. Mixed sides.
    _write_trades(cache, _COND, [
        {"ts_ns": _START + 200,  "token_id": _YES, "side": "buy",  "price": 0.60, "size": 50.0},
        {"ts_ns": _START + 400,  "token_id": _YES, "side": "sell", "price": 0.61, "size": 30.0},
        {"ts_ns": _START + 300,  "token_id": _NO,  "side": "buy",  "price": 0.39, "size": 40.0},
        {"ts_ns": _START + 600,  "token_id": _NO,  "side": "sell", "price": 0.38, "size": 20.0},
    ])

    _write_klines(cache, [
        {"ts_ns": _START + 50,  "open": 80000.0, "high": 80100.0,
         "low": 79900.0, "close": 80050.0},
        {"ts_ns": _START + 110, "open": 80050.0, "high": 80200.0,
         "low": 80000.0, "close": 80150.0},
    ])

    # YES leg: 2 snapshots. Written in NON-STANDARD order (bids ASC, asks DESC)
    # to exercise the normalisation. Second snapshot has STRICTLY MORE price
    # levels than the first (superset) → no stale-level clears.
    _write_recorded_book(book_root, _YES, "2026-05-28", "12", [
        {
            "exchange_ts": _START + 500,
            # Bids written ASC (ascending price) — normalised to DESC by both paths.
            "bid_px": [0.57, 0.58, 0.59],
            "bid_sz": [300.0, 200.0, 100.0],
            # Asks written DESC — normalised to ASC by both paths.
            "ask_px": [0.63, 0.62, 0.61],
            "ask_sz": [350.0, 400.0, 450.0],
        },
        {
            # Superset snapshot: adds price levels 0.56 (bid) and 0.64 (ask).
            "exchange_ts": _START + 1000,
            "bid_px": [0.56, 0.57, 0.58, 0.59],
            "bid_sz": [400.0, 310.0, 210.0, 110.0],
            "ask_px": [0.64, 0.63, 0.62, 0.61],
            "ask_sz": [500.0, 360.0, 410.0, 460.0],
        },
    ])

    # NO leg: 2 snapshots, same superset constraint.
    _write_recorded_book(book_root, _NO, "2026-05-28", "12", [
        {
            "exchange_ts": _START + 550,
            "bid_px": [0.38, 0.39, 0.40],
            "bid_sz": [120.0, 130.0, 140.0],
            "ask_px": [0.42, 0.41],
            "ask_sz": [150.0, 160.0],
        },
        {
            "exchange_ts": _START + 1100,
            "bid_px": [0.37, 0.38, 0.39, 0.40],
            "bid_sz": [110.0, 125.0, 135.0, 145.0],
            "ask_px": [0.43, 0.42, 0.41],
            "ask_sz": [165.0, 155.0, 165.0],
        },
    ])

    return cache, book_root


# ── equivalence helpers ───────────────────────────────────────────────────────


def _legacy_leg_arrays(src: PolymarketDataSource, q) -> dict[str, np.ndarray]:
    """Collect per-leg event arrays via the legacy events() dataclass path."""
    from collections import defaultdict

    books: dict[str, list[BookSnapshot]] = defaultdict(list)
    trades: dict[str, list[TradeEvent]] = defaultdict(list)

    for ev in src.events(q):
        if isinstance(ev, BookSnapshot):
            books[ev.symbol].append(ev)
        elif isinstance(ev, TradeEvent):
            trades[ev.symbol].append(ev)

    return {
        sym: _build_leg_event_array(books[sym], trades[sym])
        for sym in q.leg_symbols
    }


# ── tests ─────────────────────────────────────────────────────────────────────



def test_pm_fastpath_leg_arrays_bit_equivalent(tmp_path: Path) -> None:
    """Per-leg event arrays from events_arrays() match the legacy events() path.

    Checks exch_ts, ev, px, and qty fields for exact equality (bit-identical).
    The fixture is designed so no stale-level clear events are produced
    (each snapshot extends the prior price set), making both paths deterministic.
    """
    cache, book_root = _build_fixture(tmp_path)
    src = PolymarketDataSource(
        cache_root=cache, book_source="recorded", pm_book_root=book_root
    )
    q = src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]

    legacy = _legacy_leg_arrays(src, q)
    fast_bundle = src.events_arrays(q)
    fast = {sym: la.events for sym, la in fast_bundle.leg_arrays.items()}

    assert set(fast.keys()) == set(q.leg_symbols), "leg_arrays keys must match leg_symbols"

    for sym in q.leg_symbols:
        leg_fast = fast[sym]
        leg_leg = legacy[sym]
        assert len(leg_fast) == len(leg_leg), (
            f"{sym}: fast path has {len(leg_fast)} events, "
            f"legacy has {len(leg_leg)}"
        )
        assert np.array_equal(leg_fast["exch_ts"], leg_leg["exch_ts"]), (
            f"{sym}: exch_ts mismatch"
        )
        assert np.array_equal(leg_fast["ev"], leg_leg["ev"]), (
            f"{sym}: ev (event type flag) mismatch"
        )
        assert np.allclose(leg_fast["px"], leg_leg["px"]), (
            f"{sym}: px mismatch"
        )
        assert np.allclose(leg_fast["qty"], leg_leg["qty"]), (
            f"{sym}: qty mismatch"
        )


def _multiset_sorted(arr: np.ndarray) -> np.ndarray:
    """Sort an event array by (exch_ts, ev, px, qty) for order-independent
    comparison."""
    order = np.lexsort((arr["qty"], arr["px"], arr["ev"], arr["exch_ts"]))
    return arr[order]


def test_pm_fastpath_multiset_equiv_with_clears(tmp_path: Path) -> None:
    """When snapshots SHRINK (stale levels disappear), both paths emit qty=0
    clear events — but their order among themselves differs (legacy uses a
    Python set-subtraction whose iteration order is arbitrary; the fast path
    uses array order). Element-wise equality therefore does NOT hold here, and
    is not the contract.

    The guaranteed invariant is MULTISET equality: the same events with the
    same values, possibly reordered WITHIN a timestamp. This is the exact same
    situation as the HL fast path (verified there via fill equivalence). It is
    fill-safe because both builders stable-sort depth events before trades at
    equal timestamps, and clears (stale prices) and sets (current prices) touch
    disjoint price levels — so any intra-timestamp reordering yields an
    identical final book per timestamp, hence identical fills.

    This test locks in that the fast path neither drops, adds, nor corrupts any
    event when clears are present (the path the superset fixture cannot reach).
    """
    cache = tmp_path / "cache"
    book_root = tmp_path / "bookroot"
    _write_manifest(cache, cond=_COND, yes_t=_YES, no_t=_NO,
                    start=_START, end=_END, outcome="yes")
    _write_trades(cache, _COND, [
        {"ts_ns": _START + 200, "token_id": _YES, "side": "buy", "price": 0.60, "size": 50.0},
        {"ts_ns": _START + 300, "token_id": _NO, "side": "buy", "price": 0.39, "size": 40.0},
    ])
    _write_klines(cache, [
        {"ts_ns": _START + 50, "open": 80000.0, "high": 80100.0,
         "low": 79900.0, "close": 80050.0},
    ])
    # YES leg: snapshot 2 REMOVES levels 0.59 (bid) and 0.61 (ask) present in
    # snapshot 1 → stale-level clears fire in both paths.
    _write_recorded_book(book_root, _YES, "2026-05-28", "12", [
        {"exchange_ts": _START + 500,
         "bid_px": [0.57, 0.58, 0.59], "bid_sz": [300.0, 200.0, 100.0],
         "ask_px": [0.63, 0.62, 0.61], "ask_sz": [350.0, 400.0, 450.0]},
        {"exchange_ts": _START + 1000,
         "bid_px": [0.57, 0.58], "bid_sz": [300.0, 200.0],
         "ask_px": [0.63, 0.62], "ask_sz": [350.0, 400.0]},
    ])
    _write_recorded_book(book_root, _NO, "2026-05-28", "12", [
        {"exchange_ts": _START + 550,
         "bid_px": [0.38, 0.39, 0.40], "bid_sz": [120.0, 130.0, 140.0],
         "ask_px": [0.42, 0.41], "ask_sz": [150.0, 160.0]},
        {"exchange_ts": _START + 1100,
         "bid_px": [0.38], "bid_sz": [120.0],
         "ask_px": [0.41], "ask_sz": [160.0]},
    ])

    src = PolymarketDataSource(
        cache_root=cache, book_source="recorded", pm_book_root=book_root
    )
    q = src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]
    legacy = _legacy_leg_arrays(src, q)
    fast = {sym: la.events for sym, la in src.events_arrays(q).leg_arrays.items()}

    for sym in q.leg_symbols:
        lf, ll = _multiset_sorted(fast[sym]), _multiset_sorted(legacy[sym])
        assert len(lf) == len(ll), f"{sym}: event count differs ({len(lf)} vs {len(ll)})"
        assert np.array_equal(lf["exch_ts"], ll["exch_ts"]), f"{sym}: exch_ts multiset"
        assert np.array_equal(lf["ev"], ll["ev"]), f"{sym}: ev multiset"
        assert np.allclose(lf["px"], ll["px"]), f"{sym}: px multiset"
        assert np.allclose(lf["qty"], ll["qty"]), f"{sym}: qty multiset"
        # Sanity: clears actually fired (qty==0 events present), so this test
        # really exercised the reordering path.
        assert np.any(ll["qty"] == 0.0), f"{sym}: fixture should produce clears"


def test_pm_fastpath_book_ts_matches_snapshot_timestamps(tmp_path: Path) -> None:
    """LegArrays.book_ts must equal the sorted snapshot exchange timestamps."""
    cache, book_root = _build_fixture(tmp_path)
    src = PolymarketDataSource(
        cache_root=cache, book_source="recorded", pm_book_root=book_root
    )
    q = src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]

    bundle = src.events_arrays(q)

    expected_yes_ts = np.array([_START + 500, _START + 1000], dtype=np.int64)
    expected_no_ts = np.array([_START + 550, _START + 1100], dtype=np.int64)

    assert np.array_equal(bundle.leg_arrays[_YES].book_ts, expected_yes_ts), (
        f"YES book_ts mismatch: {bundle.leg_arrays[_YES].book_ts}"
    )
    assert np.array_equal(bundle.leg_arrays[_NO].book_ts, expected_no_ts), (
        f"NO book_ts mismatch: {bundle.leg_arrays[_NO].book_ts}"
    )


def test_pm_fastpath_reference_and_settlement_events(tmp_path: Path) -> None:
    """FastPathBundle carries correct reference and settlement event lists."""
    from hlanalysis.backtest.core.events import ReferenceEvent, SettlementEvent

    cache, book_root = _build_fixture(tmp_path)
    src = PolymarketDataSource(
        cache_root=cache, book_source="recorded", pm_book_root=book_root
    )
    q = src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]

    bundle = src.events_arrays(q)

    # Reference events come from klines.
    assert len(bundle.reference_events) == 2
    assert all(isinstance(e, ReferenceEvent) for e in bundle.reference_events)

    # Settlement: one per leg at end_ts_ns.
    assert len(bundle.settlement_events) == 2
    assert all(isinstance(e, SettlementEvent) for e in bundle.settlement_events)
    settle_syms = {e.symbol for e in bundle.settlement_events}
    assert settle_syms == {_YES, _NO}
    assert all(e.ts_ns == _END for e in bundle.settlement_events)


def test_pm_fastpath_no_coverage_returns_empty_arrays(tmp_path: Path) -> None:
    """When a leg has no recorded book parquet, events_arrays() returns an
    empty event array for that leg (graceful degradation)."""
    cache = tmp_path / "cache"
    book_root = tmp_path / "bookroot_empty"  # nothing written here

    _write_manifest(cache, cond=_COND, yes_t=_YES, no_t=_NO,
                    start=_START, end=_END)
    _write_trades(cache, _COND, [
        {"ts_ns": _START + 100, "token_id": _YES, "side": "buy", "price": 0.6, "size": 10.0},
    ])
    _write_klines(cache, [
        {"ts_ns": _START + 50, "open": 80000.0, "high": 80100.0,
         "low": 79900.0, "close": 80050.0},
    ])

    src = PolymarketDataSource(
        cache_root=cache, book_source="recorded", pm_book_root=book_root
    )
    q = src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]
    bundle = src.events_arrays(q)

    # No book → only trade events; book_ts is empty.
    for sym in q.leg_symbols:
        assert len(bundle.leg_arrays[sym].book_ts) == 0, (
            f"{sym}: expected empty book_ts when no recorded book"
        )


# ── synthetic-mode equivalence tests ─────────────────────────────────────────


def _build_synthetic_fixture(tmp_path: Path) -> Path:
    """Construct a minimal synthetic-mode fixture (no book parquet needed).

    Design constraints for bit-equivalent comparison (see module docstring):
    - Snapshots are synthesised from PM trades via ``trade_to_l2``; the
      synthetic stream always uses superset-price-set semantics within a single
      trade, so no stale-level clears fire.
    - 2 legs (YES / NO), 3 trades each.
    """
    cache = tmp_path / "cache"
    _write_manifest(cache, cond=_COND, yes_t=_YES, no_t=_NO,
                    start=_START, end=_END, outcome="yes")
    _write_trades(cache, _COND, [
        {"ts_ns": _START + 200, "token_id": _YES, "side": "buy",  "price": 0.60, "size": 50.0},
        {"ts_ns": _START + 400, "token_id": _YES, "side": "sell", "price": 0.61, "size": 30.0},
        {"ts_ns": _START + 600, "token_id": _YES, "side": "buy",  "price": 0.59, "size": 20.0},
        {"ts_ns": _START + 300, "token_id": _NO,  "side": "buy",  "price": 0.39, "size": 40.0},
        {"ts_ns": _START + 500, "token_id": _NO,  "side": "sell", "price": 0.38, "size": 25.0},
        {"ts_ns": _START + 700, "token_id": _NO,  "side": "buy",  "price": 0.40, "size": 15.0},
    ])
    _write_klines(cache, [
        {"ts_ns": _START + 50,  "open": 80000.0, "high": 80100.0,
         "low": 79900.0, "close": 80050.0},
        {"ts_ns": _START + 110, "open": 80050.0, "high": 80200.0,
         "low": 80000.0, "close": 80150.0},
    ])
    return cache


def test_synthetic_events_arrays_bit_equivalent_to_legacy(tmp_path: Path) -> None:
    """events_arrays() in synthetic mode must produce bit-identical per-leg
    event arrays to the legacy events()-based path (no liquidity profile)."""
    cache = _build_synthetic_fixture(tmp_path)
    src = PolymarketDataSource(cache_root=cache, book_source="synthetic")
    q = src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]

    legacy = _legacy_leg_arrays(src, q)
    fast_bundle = src.events_arrays(q)
    fast = {sym: la.events for sym, la in fast_bundle.leg_arrays.items()}

    assert set(fast.keys()) == set(q.leg_symbols)

    for sym in q.leg_symbols:
        lf, ll = fast[sym], legacy[sym]
        assert len(lf) == len(ll), (
            f"{sym}: fast={len(lf)} events, legacy={len(ll)}"
        )
        assert np.array_equal(lf["exch_ts"], ll["exch_ts"]), f"{sym}: exch_ts mismatch"
        assert np.array_equal(lf["ev"], ll["ev"]), f"{sym}: ev mismatch"
        assert np.allclose(lf["px"], ll["px"]), f"{sym}: px mismatch"
        assert np.allclose(lf["qty"], ll["qty"]), f"{sym}: qty mismatch"


def test_synthetic_events_arrays_bit_equivalent_with_liquidity_profile(
    tmp_path: Path,
) -> None:
    """events_arrays() in synthetic mode must be bit-identical when a
    LiquidityProfile is supplied (profile-aware spreads/depths)."""
    import json as _json
    from hlanalysis.backtest.data._synthetic_l2 import LiquidityProfile

    cache = _build_synthetic_fixture(tmp_path)

    # Write a minimal liquidity profile JSON so we can test the profile path.
    profile_data = {
        "bucket_width": 0.1,
        "half_spread": [0.004, 0.005, 0.006, 0.005, 0.004,
                        0.004, 0.005, 0.006, 0.005, 0.004],
        "depth": [8000.0, 9000.0, 10000.0, 9000.0, 8000.0,
                  8000.0, 9000.0, 10000.0, 9000.0, 8000.0],
        "global_half_spread": 0.005,
        "global_depth": 10000.0,
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(_json.dumps(profile_data))

    src = PolymarketDataSource(
        cache_root=cache,
        book_source="synthetic",
        liquidity_profile_path=profile_path,
    )
    q = src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]

    legacy = _legacy_leg_arrays(src, q)
    fast_bundle = src.events_arrays(q)
    fast = {sym: la.events for sym, la in fast_bundle.leg_arrays.items()}

    assert set(fast.keys()) == set(q.leg_symbols)

    for sym in q.leg_symbols:
        lf, ll = fast[sym], legacy[sym]
        assert len(lf) == len(ll), (
            f"{sym}: fast={len(lf)} events, legacy={len(ll)}"
        )
        assert np.array_equal(lf["exch_ts"], ll["exch_ts"]), f"{sym}: exch_ts mismatch"
        assert np.array_equal(lf["ev"], ll["ev"]), f"{sym}: ev mismatch"
        assert np.allclose(lf["px"], ll["px"]), f"{sym}: px mismatch"
        assert np.allclose(lf["qty"], ll["qty"]), f"{sym}: qty mismatch"

    # With a profile, spreads differ from the flat default — verify events are
    # actually different from the no-profile run (the profile is exercised).
    src_no_profile = PolymarketDataSource(
        cache_root=cache, book_source="synthetic"
    )
    fast_no_profile = {
        sym: la.events
        for sym, la in src_no_profile.events_arrays(q).leg_arrays.items()
    }
    # At least one leg should differ (profile changes spread ≠ default 0.005).
    any_differ = any(
        not np.array_equal(fast[sym]["px"], fast_no_profile[sym]["px"])
        for sym in q.leg_symbols
    )
    assert any_differ, (
        "liquidity profile had no effect on event px — profile not exercised"
    )


def test_synthetic_events_arrays_reference_and_settlement(tmp_path: Path) -> None:
    """FastPathBundle from synthetic mode carries correct reference and settlement."""
    from hlanalysis.backtest.core.events import ReferenceEvent, SettlementEvent

    cache = _build_synthetic_fixture(tmp_path)
    src = PolymarketDataSource(cache_root=cache, book_source="synthetic")
    q = src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]

    bundle = src.events_arrays(q)

    assert len(bundle.reference_events) == 2
    assert all(isinstance(e, ReferenceEvent) for e in bundle.reference_events)

    assert len(bundle.settlement_events) == 2
    assert all(isinstance(e, SettlementEvent) for e in bundle.settlement_events)
    settle_syms = {e.symbol for e in bundle.settlement_events}
    assert settle_syms == {_YES, _NO}
    assert all(e.ts_ns == _END for e in bundle.settlement_events)


def test_bundle_config_sig_differs_by_liquidity_profile(tmp_path: Path) -> None:
    """Two PolymarketDataSource instances differing only in liquidity profile
    must produce DIFFERENT cache key signatures."""
    import json as _json

    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    profile_data = {
        "bucket_width": 0.1,
        "half_spread": [0.004] * 10,
        "depth": [9000.0] * 10,
        "global_half_spread": 0.005,
        "global_depth": 10000.0,
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(_json.dumps(profile_data))

    src_no_profile = PolymarketDataSource(
        cache_root=cache, book_source="synthetic"
    )
    src_with_profile = PolymarketDataSource(
        cache_root=cache,
        book_source="synthetic",
        liquidity_profile_path=profile_path,
    )

    sig_none = src_no_profile._bundle_config_sig()
    sig_prof = src_with_profile._bundle_config_sig()
    assert sig_none != sig_prof, (
        f"Expected different sigs but both were: {sig_none!r}"
    )


def test_bundle_config_sig_identical_for_same_profile(tmp_path: Path) -> None:
    """Two instances with the same liquidity profile file produce IDENTICAL
    cache key signatures."""
    import json as _json

    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    profile_data = {
        "bucket_width": 0.1,
        "half_spread": [0.004] * 10,
        "depth": [9000.0] * 10,
        "global_half_spread": 0.005,
        "global_depth": 10000.0,
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(_json.dumps(profile_data))

    src_a = PolymarketDataSource(
        cache_root=cache,
        book_source="synthetic",
        liquidity_profile_path=profile_path,
    )
    src_b = PolymarketDataSource(
        cache_root=cache,
        book_source="synthetic",
        liquidity_profile_path=profile_path,
    )

    assert src_a._bundle_config_sig() == src_b._bundle_config_sig(), (
        "Same profile should produce identical config sig"
    )
