"""Unit tests for the recorded-book fill source on `PolymarketDataSource`.

These cover the `book_source="recorded"` mode added for HL-parity PM fills:
the real recorded multi-level L2 book replaces the synthetic 1-level book.

Verifies:
- level normalization (best bid = max(bid_px), best ask = min(ask_px)) from
  arbitrarily-ordered recorded arrays;
- per-leg join (each leg reads its own token's book partition);
- multi-level emission (a recorded snapshot with N levels → N bid/ask tuples);
- recorded mode drops the synthetic `1−p` parity synthesis;
- the default (`synthetic`) event stream is bit-identical to explicit synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hlanalysis.backtest.core.events import BookSnapshot
from hlanalysis.backtest.data.polymarket import PolymarketDataSource

_BOOK_SUBPATH = "venue=polymarket/product_type=prediction_binary/mechanism=clob/event=book_snapshot"
_SETTLE_SUBPATH = "venue=polymarket/product_type=prediction_binary/mechanism=clob/event=settlement"


# ---- fixture helpers --------------------------------------------------------


def _write_manifest(cache_root: Path, manifest: dict) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _write_trades(cache_root: Path, condition_id: str, rows: list[dict]) -> None:
    pm_trades = cache_root / "pm_trades"
    pm_trades.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "ts_ns": [int(r["ts_ns"]) for r in rows],
            "token_id": [str(r["token_id"]) for r in rows],
            "side": [str(r["side"]) for r in rows],
            "price": [float(r["price"]) for r in rows],
            "size": [float(r["size"]) for r in rows],
        }
    )
    pq.write_table(table, pm_trades / f"{condition_id}.parquet")


def _write_klines(cache_root: Path, klines: list[dict]) -> None:
    klines_dir = cache_root / "btc_klines"
    klines_dir.mkdir(parents=True, exist_ok=True)
    (klines_dir / "fixture.json").write_text(json.dumps(klines))


def _write_recorded_book(book_root: Path, token_id: str, date: str, hour: str, rows: list[dict]) -> None:
    """Write a recorded `book_snapshot` parquet for one token leg.

    `rows`: list of dicts with keys exchange_ts (int ns), bid_px, bid_sz,
    ask_px, ask_sz (each a list[float]).
    """
    d = book_root / _BOOK_SUBPATH / f"symbol={token_id}" / f"date={date}" / f"hour={hour}"
    d.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "exchange_ts": pa.array([int(r["exchange_ts"]) for r in rows], pa.int64()),
            "bid_px": pa.array([[float(x) for x in r["bid_px"]] for r in rows], pa.list_(pa.float64())),
            "bid_sz": pa.array([[float(x) for x in r["bid_sz"]] for r in rows], pa.list_(pa.float64())),
            "ask_px": pa.array([[float(x) for x in r["ask_px"]] for r in rows], pa.list_(pa.float64())),
            "ask_sz": pa.array([[float(x) for x in r["ask_sz"]] for r in rows], pa.list_(pa.float64())),
        }
    )
    pq.write_table(table, d / "data.parquet")


def _write_settlement(
    book_root: Path,
    token_id: str,
    *,
    settled_side_idx: int,
    settle_price: float,
    settle_ts: int,
    date: str = "2026-05-28",
    hour: str = "16",
) -> None:
    """Write a recorder `settlement` parquet for one token leg."""
    d = book_root / _SETTLE_SUBPATH / f"symbol={token_id}" / f"date={date}" / f"hour={hour}"
    d.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "settled_side_idx": pa.array([int(settled_side_idx)], pa.int64()),
            "settle_price": pa.array([float(settle_price)], pa.float64()),
            "settle_ts": pa.array([int(settle_ts)], pa.int64()),
        }
    )
    pq.write_table(table, d / "data.parquet")


def _binary_manifest(cond: str, yes_t: str, no_t: str, start: int, end: int, outcome: str = "yes") -> dict:
    return {
        cond: {
            "kind": "binary",
            "n_rows": 1,
            "last_pull_ts_ns": end,
            "market": {
                "condition_id": cond,
                "yes_token_id": yes_t,
                "no_token_id": no_t,
                "start_ts_ns": start,
                "end_ts_ns": end,
                "resolved_outcome": outcome,
                "total_volume_usd": 100.0,
                "n_trades": 1,
            },
        }
    }


# Timestamps land in 2026-05 so default discover windows include them.
_START = 1_780_000_000_000_000_000  # ~2026-05-28
_END = _START + 3600 * 1_000_000_000  # +1h
_COND = "cond_rec_1"
_YES = "11111yes"
_NO = "22222no"


def _build_recorded_cache(tmp_path: Path) -> tuple[Path, Path]:
    """Build a PM cache + recorded book root. Returns (cache_root, book_root)."""
    cache = tmp_path / "cache"
    book_root = tmp_path / "bookroot"
    _write_manifest(cache, _binary_manifest(_COND, _YES, _NO, _START, _END))
    _write_trades(
        cache,
        _COND,
        [
            {"ts_ns": _START + 100, "token_id": _YES, "side": "buy", "price": 0.6, "size": 10.0},
        ],
    )
    _write_klines(
        cache,
        [
            {"ts_ns": _START + 50, "open": 80000.0, "high": 80100.0, "low": 79900.0, "close": 80050.0},
        ],
    )
    return cache, book_root


def _src_recorded(cache: Path, book_root: Path) -> PolymarketDataSource:
    return PolymarketDataSource(cache_root=cache, book_source="recorded", pm_book_root=book_root)


# ---- tests ------------------------------------------------------------------


def test_recorded_book_level_normalization(tmp_path: Path) -> None:
    """Unsorted recorded arrays normalize to best bid = max(bid_px),
    best ask = min(ask_px); bids sorted px DESC, asks sorted px ASC."""
    cache, book_root = _build_recorded_cache(tmp_path)
    # Deliberately unsorted: bids ascending, asks descending (PM convention).
    _write_recorded_book(
        book_root,
        _YES,
        "2026-05-28",
        "12",
        [
            {
                "exchange_ts": _START + 500,
                "bid_px": [0.01, 0.58, 0.30],
                "bid_sz": [100.0, 200.0, 150.0],
                "ask_px": [0.99, 0.62, 0.70],
                "ask_sz": [300.0, 400.0, 350.0],
            },
        ],
    )
    _write_recorded_book(
        book_root,
        _NO,
        "2026-05-28",
        "12",
        [
            {"exchange_ts": _START + 500, "bid_px": [0.38], "bid_sz": [120.0], "ask_px": [0.42], "ask_sz": [130.0]},
        ],
    )
    src = _src_recorded(cache, book_root)
    d = src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]
    books = [e for e in src.events(d) if isinstance(e, BookSnapshot) and e.symbol == _YES]
    assert len(books) == 1
    b = books[0]
    # Best bid = max(bid_px) = 0.58, sorted DESC.
    assert b.bids[0][0] == 0.58
    assert [px for px, _ in b.bids] == sorted([px for px, _ in b.bids], reverse=True)
    # Best ask = min(ask_px) = 0.62, sorted ASC.
    assert b.asks[0][0] == 0.62
    assert [px for px, _ in b.asks] == sorted([px for px, _ in b.asks])
    # Size travels with its price level.
    assert dict(b.bids)[0.58] == 200.0
    assert dict(b.asks)[0.62] == 400.0


def test_recorded_book_per_leg_join(tmp_path: Path) -> None:
    """Each leg reads its own token's recorded book partition."""
    cache, book_root = _build_recorded_cache(tmp_path)
    _write_recorded_book(
        book_root,
        _YES,
        "2026-05-28",
        "12",
        [
            {"exchange_ts": _START + 500, "bid_px": [0.6], "bid_sz": [10.0], "ask_px": [0.61], "ask_sz": [11.0]},
        ],
    )
    _write_recorded_book(
        book_root,
        _NO,
        "2026-05-28",
        "12",
        [
            {"exchange_ts": _START + 600, "bid_px": [0.39], "bid_sz": [20.0], "ask_px": [0.40], "ask_sz": [21.0]},
        ],
    )
    src = _src_recorded(cache, book_root)
    d = src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]
    books = [e for e in src.events(d) if isinstance(e, BookSnapshot)]
    yes_books = [b for b in books if b.symbol == _YES]
    no_books = [b for b in books if b.symbol == _NO]
    assert len(yes_books) == 1 and len(no_books) == 1
    assert yes_books[0].bids[0][0] == 0.6
    assert no_books[0].bids[0][0] == 0.39


def test_recorded_book_multilevel_emission(tmp_path: Path) -> None:
    """A recorded snapshot with N levels emits N bid/ask tuples."""
    cache, book_root = _build_recorded_cache(tmp_path)
    _write_recorded_book(
        book_root,
        _YES,
        "2026-05-28",
        "12",
        [
            {
                "exchange_ts": _START + 500,
                "bid_px": [0.58, 0.57, 0.56],
                "bid_sz": [100.0, 200.0, 300.0],
                "ask_px": [0.62, 0.63],
                "ask_sz": [150.0, 250.0],
            },
        ],
    )
    src = _src_recorded(cache, book_root)
    d = src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]
    books = [e for e in src.events(d) if isinstance(e, BookSnapshot) and e.symbol == _YES]
    assert len(books) == 1
    assert len(books[0].bids) == 3
    assert len(books[0].asks) == 2


def test_recorded_mode_drops_synthetic_parity(tmp_path: Path) -> None:
    """Recorded mode emits real per-leg books only; no synthetic `1−p`
    parity snapshot is synthesized from trades."""
    cache, book_root = _build_recorded_cache(tmp_path)
    _write_recorded_book(
        book_root,
        _YES,
        "2026-05-28",
        "12",
        [
            {"exchange_ts": _START + 500, "bid_px": [0.6], "bid_sz": [10.0], "ask_px": [0.61], "ask_sz": [11.0]},
        ],
    )
    _write_recorded_book(
        book_root,
        _NO,
        "2026-05-28",
        "12",
        [
            {"exchange_ts": _START + 700, "bid_px": [0.39], "bid_sz": [20.0], "ask_px": [0.40], "ask_sz": [21.0]},
        ],
    )
    src = _src_recorded(cache, book_root)
    d = src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]
    evs = list(src.events(d))
    # The trade is at _START+100 on YES. In synthetic mode this would emit a
    # NO parity book at the same ts; in recorded mode it must NOT.
    trade_ts = _START + 100
    no_books_at_trade = [e for e in evs if isinstance(e, BookSnapshot) and e.ts_ns == trade_ts and e.symbol == _NO]
    assert no_books_at_trade == []
    # Books present are exactly the two recorded snapshots.
    book_ts_syms = {(e.ts_ns, e.symbol) for e in evs if isinstance(e, BookSnapshot)}
    assert book_ts_syms == {(_START + 500, _YES), (_START + 700, _NO)}
    # Monotone non-decreasing ts preserved across the merged stream.
    assert all(evs[i].ts_ns <= evs[i + 1].ts_ns for i in range(len(evs) - 1))


def test_missing_coverage_skips_cleanly(tmp_path: Path) -> None:
    """A market with no recorded book partition yields no BookSnapshots and
    does not crash (other event kinds still flow)."""
    cache, book_root = _build_recorded_cache(tmp_path)
    # No _write_recorded_book calls → no coverage.
    src = _src_recorded(cache, book_root)
    d = src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]
    evs = list(src.events(d))
    assert [e for e in evs if isinstance(e, BookSnapshot)] == []
    # Trades + reference + settlement still present.
    kinds = {type(e).__name__ for e in evs}
    assert {"TradeEvent", "ReferenceEvent", "SettlementEvent"}.issubset(kinds)


def test_settlement_event_determines_winner(tmp_path: Path) -> None:
    """The recorder `settlement` event is authoritative for the winner and
    OVERRIDES the manifest `resolved_outcome` when present."""
    cache = tmp_path / "cache"
    book_root = tmp_path / "bookroot"
    # Manifest deliberately says "yes" — settlement must override to "no".
    _write_manifest(cache, _binary_manifest(_COND, _YES, _NO, _START, _END, outcome="yes"))
    _write_trades(
        cache,
        _COND,
        [
            {"ts_ns": _START + 100, "token_id": _YES, "side": "buy", "price": 0.6, "size": 10.0},
        ],
    )
    _write_klines(
        cache,
        [
            {"ts_ns": _START + 50, "open": 80000.0, "high": 80100.0, "low": 79900.0, "close": 80050.0},
        ],
    )
    # NO leg redeems at 1.0 (it won); winning side only is recorded.
    _write_settlement(book_root, _NO, settled_side_idx=1, settle_price=1.0, settle_ts=_END + 100)
    src = _src_recorded(cache, book_root)
    d = src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]
    assert src.resolved_outcome(d) == "no"


def test_settlement_falls_back_to_manifest_when_absent(tmp_path: Path) -> None:
    """With no recorder settlement coverage, the winner falls back to the
    manifest `resolved_outcome` (bit-identical to prior behavior)."""
    cache, book_root = _build_recorded_cache(tmp_path)  # outcome="yes", no settlement
    src = _src_recorded(cache, book_root)
    d = src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]
    assert src.resolved_outcome(d) == "yes"


def test_manifest_fallback_emits_warning(tmp_path: Path) -> None:
    """Falling back to the manifest outcome (no recorder settlement) logs a
    WARNING so silent reliance on the weaker source is visible."""
    from loguru import logger

    cache, book_root = _build_recorded_cache(tmp_path)  # no settlement coverage
    src = _src_recorded(cache, book_root)
    d = src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]
    msgs: list[str] = []
    sink_id = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        assert src.resolved_outcome(d) == "yes"
    finally:
        logger.remove(sink_id)
    assert any("settlement" in m.lower() for m in msgs), msgs


def test_settlement_present_does_not_warn(tmp_path: Path) -> None:
    """When recorder settlement IS present, no fallback warning is emitted."""
    from loguru import logger

    cache = tmp_path / "cache"
    book_root = tmp_path / "bookroot"
    _write_manifest(cache, _binary_manifest(_COND, _YES, _NO, _START, _END, outcome="no"))
    _write_trades(
        cache,
        _COND,
        [
            {"ts_ns": _START + 100, "token_id": _YES, "side": "buy", "price": 0.6, "size": 10.0},
        ],
    )
    _write_klines(
        cache,
        [
            {"ts_ns": _START + 50, "open": 80000.0, "high": 80100.0, "low": 79900.0, "close": 80050.0},
        ],
    )
    _write_settlement(book_root, _NO, settled_side_idx=1, settle_price=1.0, settle_ts=_END + 100)
    src = _src_recorded(cache, book_root)
    d = src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]
    msgs: list[str] = []
    sink_id = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        assert src.resolved_outcome(d) == "no"
    finally:
        logger.remove(sink_id)
    assert not any("settlement" in m.lower() for m in msgs), msgs


def test_settlement_yes_winner(tmp_path: Path) -> None:
    """settle_price=1.0 on the YES leg → outcome 'yes'."""
    cache = tmp_path / "cache"
    book_root = tmp_path / "bookroot"
    _write_manifest(cache, _binary_manifest(_COND, _YES, _NO, _START, _END, outcome="no"))
    _write_trades(
        cache,
        _COND,
        [
            {"ts_ns": _START + 100, "token_id": _YES, "side": "buy", "price": 0.6, "size": 10.0},
        ],
    )
    _write_klines(
        cache,
        [
            {"ts_ns": _START + 50, "open": 80000.0, "high": 80100.0, "low": 79900.0, "close": 80050.0},
        ],
    )
    _write_settlement(book_root, _YES, settled_side_idx=0, settle_price=1.0, settle_ts=_END + 100)
    src = _src_recorded(cache, book_root)
    d = src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]
    assert src.resolved_outcome(d) == "yes"


def test_settlement_used_even_in_synthetic_mode(tmp_path: Path) -> None:
    """Settlement is authoritative regardless of book_source (it governs payoff,
    not fills)."""
    cache = tmp_path / "cache"
    book_root = tmp_path / "bookroot"
    _write_manifest(cache, _binary_manifest(_COND, _YES, _NO, _START, _END, outcome="yes"))
    _write_trades(
        cache,
        _COND,
        [
            {"ts_ns": _START + 100, "token_id": _YES, "side": "buy", "price": 0.6, "size": 10.0},
        ],
    )
    _write_klines(
        cache,
        [
            {"ts_ns": _START + 50, "open": 80000.0, "high": 80100.0, "low": 79900.0, "close": 80050.0},
        ],
    )
    _write_settlement(book_root, _NO, settled_side_idx=1, settle_price=1.0, settle_ts=_END + 100)
    src = PolymarketDataSource(cache_root=cache, pm_book_root=book_root)  # synthetic default
    d = src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]
    assert src.resolved_outcome(d) == "no"


def test_synthetic_default_bit_identical(tmp_path: Path) -> None:
    """The default (`synthetic`) event stream equals explicit synthetic, and
    differs from recorded (which carries no `1−p` parity)."""
    cache, book_root = _build_recorded_cache(tmp_path)
    default_src = PolymarketDataSource(cache_root=cache)
    explicit_synth = PolymarketDataSource(cache_root=cache, book_source="synthetic")
    d_default = default_src.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]
    d_synth = explicit_synth.discover(start="2026-01-01", end="2026-12-31", kind="binary")[0]
    evs_default = list(default_src.events(d_default))
    evs_synth = list(explicit_synth.events(d_synth))
    # Frozen dataclasses compare by value → bit-identical stream.
    assert evs_default == evs_synth
    # Synthetic emits a NO parity book at the YES trade ts; sanity-check it's there.
    trade_ts = _START + 100
    assert any(isinstance(e, BookSnapshot) and e.ts_ns == trade_ts and e.symbol == _NO for e in evs_default)
