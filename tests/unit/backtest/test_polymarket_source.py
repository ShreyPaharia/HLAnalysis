"""Unit tests for `hlanalysis.backtest.data.polymarket.PolymarketDataSource`.

Covers the §3 DataSource contract on a cache populated by hand — no live
PM API hits. Bucket-path tests cover leg ordering, threshold parsing,
within-pair parity, no cross-pair parity, and per-leg settlement (§4 Task C
acceptance).
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hlanalysis.backtest.core.events import (
    BookSnapshot,
    ReferenceEvent,
    SettlementEvent,
    TradeEvent,
)
from hlanalysis.backtest.data.polymarket import PolymarketDataSource


# ---- Cache fixture helpers --------------------------------------------------


def _write_trades(cache_root: Path, condition_id: str, rows: list[dict]) -> None:
    """Write a trades parquet file shaped like the legacy `sim.data.cache`
    output. `rows` is a list of dicts with keys ts_ns, token_id, side, price,
    size — same order as the legacy `PMTrade`.
    """
    pm_trades = cache_root / "pm_trades"
    pm_trades.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "ts_ns":    [int(r["ts_ns"]) for r in rows],
        "token_id": [str(r["token_id"]) for r in rows],
        "side":     [str(r["side"]) for r in rows],
        "price":    [float(r["price"]) for r in rows],
        "size":     [float(r["size"]) for r in rows],
    })
    pq.write_table(table, pm_trades / f"{condition_id}.parquet")


def _write_klines(cache_root: Path, klines: list[dict]) -> None:
    klines_dir = cache_root / "btc_klines"
    klines_dir.mkdir(parents=True, exist_ok=True)
    (klines_dir / "fixture.json").write_text(json.dumps(klines))


def _write_manifest(cache_root: Path, manifest: dict) -> None:
    (cache_root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _binary_manifest_entry(
    *,
    condition_id: str,
    yes_token: str,
    no_token: str,
    start_ts_ns: int,
    end_ts_ns: int,
    resolved_outcome: str = "yes",
    n_rows: int = 1,
) -> dict:
    return {
        "kind": "binary",
        "n_rows": n_rows,
        "last_pull_ts_ns": end_ts_ns,
        "market": {
            "condition_id": condition_id,
            "yes_token_id": yes_token,
            "no_token_id": no_token,
            "start_ts_ns": start_ts_ns,
            "end_ts_ns": end_ts_ns,
            "resolved_outcome": resolved_outcome,
            "total_volume_usd": 100.0,
            "n_trades": n_rows,
        },
    }


def _bucket_manifest_entry(
    *,
    event_slug: str,
    start_ts_ns: int,
    end_ts_ns: int,
    thresholds: list[float],
    leg_tokens: list[tuple[str, str]],
    leg_condition_ids: list[str],
    leg_resolutions: list[str],
    n_rows: int = 1,
) -> dict:
    return {
        "kind": "bucket",
        "n_rows": n_rows,
        "last_pull_ts_ns": end_ts_ns,
        "bucket": {
            "event_slug": event_slug,
            "start_ts_ns": start_ts_ns,
            "end_ts_ns": end_ts_ns,
            "thresholds": thresholds,
            "leg_tokens": [list(p) for p in leg_tokens],
            "leg_condition_ids": leg_condition_ids,
            "leg_resolutions": leg_resolutions,
        },
    }


# ---- Binary path tests ------------------------------------------------------


@pytest.fixture
def binary_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache_binary"
    cache.mkdir()
    cond = "cond_binary_1"
    yes_t = "tok_yes"
    no_t = "tok_no"
    start = 1_000_000_000_000_000_000
    end = 2_000_000_000_000_000_000
    _write_manifest(cache, {
        cond: _binary_manifest_entry(
            condition_id=cond, yes_token=yes_t, no_token=no_t,
            start_ts_ns=start, end_ts_ns=end, resolved_outcome="yes", n_rows=2,
        ),
    })
    _write_trades(cache, cond, [
        {"ts_ns": start + 100, "token_id": yes_t, "side": "buy", "price": 0.6, "size": 10.0},
        {"ts_ns": start + 200, "token_id": no_t,  "side": "sell", "price": 0.45, "size": 20.0},
    ])
    _write_klines(cache, [
        {"ts_ns": start + 50,  "open": 80000.0, "high": 80100.0, "low": 79900.0, "close": 80050.0, "volume": 1.0},
        {"ts_ns": start + 150, "open": 80050.0, "high": 80200.0, "low": 79950.0, "close": 80100.0, "volume": 1.2},
    ])
    return cache


def test_descriptor_binary_basic(binary_cache: Path) -> None:
    src = PolymarketDataSource(cache_root=binary_cache)
    descs = src.discover(start="1970-01-01", end="2999-12-31", kind="binary")
    assert len(descs) == 1
    d = descs[0]
    assert d.question_id == "cond_binary_1"
    assert d.klass == "priceBinary"
    assert d.underlying == "BTC"
    assert d.leg_symbols == ("tok_yes", "tok_no")
    # question_idx is hash(question_id) & 0x7FFFFFFF and must match QuestionView.
    qv = src.question_view(d, now_ns=d.start_ts_ns + 1, settled=False)
    assert qv.question_idx == d.question_idx
    assert qv.klass == "priceBinary"
    assert qv.yes_symbol == "tok_yes"
    assert qv.no_symbol == "tok_no"


def test_events_monotone_ts_binary(binary_cache: Path) -> None:
    src = PolymarketDataSource(cache_root=binary_cache)
    d = src.discover(start="1970-01-01", end="2999-12-31", kind="binary")[0]
    evs = list(src.events(d))
    # Monotone non-decreasing ts.
    assert all(evs[i].ts_ns <= evs[i + 1].ts_ns for i in range(len(evs) - 1))
    # Should contain all four event kinds.
    by_kind = {type(e).__name__ for e in evs}
    assert {"BookSnapshot", "TradeEvent", "ReferenceEvent", "SettlementEvent"}.issubset(by_kind)
    # Two settlement events (one per leg) at end_ts_ns.
    settles = [e for e in evs if isinstance(e, SettlementEvent)]
    assert len(settles) == 2
    assert {s.symbol for s in settles} == {"tok_yes", "tok_no"}
    assert all(s.ts_ns == d.end_ts_ns for s in settles)


def test_binary_within_pair_parity(binary_cache: Path) -> None:
    """A trade on YES at 0.6 emits a synthetic BookSnapshot for NO at price ≈ 0.4."""
    src = PolymarketDataSource(cache_root=binary_cache)
    d = src.discover(start="1970-01-01", end="2999-12-31", kind="binary")[0]
    evs = list(src.events(d))
    # First synthetic book pair (from yes trade at price 0.6).
    yes_trade_ts = d.start_ts_ns + 100
    yes_books = [e for e in evs
                 if isinstance(e, BookSnapshot) and e.ts_ns == yes_trade_ts and e.symbol == "tok_yes"]
    no_books_synthetic = [e for e in evs
                          if isinstance(e, BookSnapshot) and e.ts_ns == yes_trade_ts and e.symbol == "tok_no"]
    assert len(yes_books) == 1
    assert len(no_books_synthetic) == 1
    # NO leg synthetic price ≈ 1 - 0.6 = 0.4 (mid).
    no_bid, _no_bid_sz = no_books_synthetic[0].bids[0]
    no_ask, _no_ask_sz = no_books_synthetic[0].asks[0]
    no_mid = (no_bid + no_ask) / 2
    assert abs(no_mid - 0.4) < 1e-9


def test_resolved_outcome_binary(binary_cache: Path) -> None:
    src = PolymarketDataSource(cache_root=binary_cache)
    d = src.discover(start="1970-01-01", end="2999-12-31", kind="binary")[0]
    assert src.resolved_outcome(d) == "yes"


# ---- Bucket path tests ------------------------------------------------------


@pytest.fixture
def bucket_cache(tmp_path: Path) -> Path:
    """A bucket event with 3 strike markets (80k, 81k, 82k). BTC closes at
    81.5k, so leg pair 0 won (above 80k = YES), pair 1 won (above 81k = YES),
    pair 2 lost (above 82k = NO).
    """
    cache = tmp_path / "cache_bucket"
    cache.mkdir()
    event_slug = "bitcoin-above-on-test-event"
    start = 1_000_000_000_000_000_000
    end = 2_000_000_000_000_000_000
    leg_tokens = [
        ("tok_y0", "tok_n0"),   # strike 80k
        ("tok_y1", "tok_n1"),   # strike 81k
        ("tok_y2", "tok_n2"),   # strike 82k
    ]
    leg_condition_ids = ["cond_o0", "cond_o1", "cond_o2"]
    leg_resolutions = ["yes", "yes", "no"]
    _write_manifest(cache, {
        event_slug: _bucket_manifest_entry(
            event_slug=event_slug,
            start_ts_ns=start, end_ts_ns=end,
            thresholds=[80000.0, 81000.0, 82000.0],
            leg_tokens=leg_tokens,
            leg_condition_ids=leg_condition_ids,
            leg_resolutions=leg_resolutions,
            n_rows=2,
        ),
    })
    # One trade each on tok_y0 (above-80k YES) and tok_y2 (above-82k YES).
    _write_trades(cache, "cond_o0", [
        {"ts_ns": start + 100, "token_id": "tok_y0", "side": "buy", "price": 0.7, "size": 5.0},
    ])
    _write_trades(cache, "cond_o1", [])
    _write_trades(cache, "cond_o2", [
        {"ts_ns": start + 150, "token_id": "tok_y2", "side": "sell", "price": 0.2, "size": 8.0},
    ])
    _write_klines(cache, [
        {"ts_ns": start + 50, "open": 80000.0, "high": 81600.0, "low": 80000.0, "close": 81500.0, "volume": 1.0},
    ])
    return cache


def test_descriptor_bucket_ordering(bucket_cache: Path) -> None:
    src = PolymarketDataSource(cache_root=bucket_cache)
    descs = src.discover(start="1970-01-01", end="2999-12-31", kind="bucket")
    assert len(descs) == 1
    d = descs[0]
    assert d.klass == "priceBucket"
    assert d.underlying == "BTC"
    assert d.leg_symbols == (
        "tok_y0", "tok_n0",
        "tok_y1", "tok_n1",
        "tok_y2", "tok_n2",
    )


def test_threshold_parsing_and_sidecar(bucket_cache: Path) -> None:
    """Thresholds stored in cache manifest as explicit float list, round-trip
    through QuestionView.kv["priceThresholds"]."""
    raw = json.loads((bucket_cache / "manifest.json").read_text())
    bucket = raw["bitcoin-above-on-test-event"]["bucket"]
    assert bucket["thresholds"] == [80000.0, 81000.0, 82000.0]

    src = PolymarketDataSource(cache_root=bucket_cache)
    d = src.discover(start="1970-01-01", end="2999-12-31", kind="bucket")[0]
    qv = src.question_view(d, now_ns=d.start_ts_ns + 1, settled=False)
    kv_map = dict(qv.kv)
    assert "priceThresholds" in kv_map
    parts = [p.strip() for p in kv_map["priceThresholds"].split(",")]
    assert [float(p) for p in parts] == [80000.0, 81000.0, 82000.0]


def test_bucket_no_cross_pair_parity(bucket_cache: Path) -> None:
    """A trade on `tok_y0` should emit a within-pair synthetic snapshot on
    `tok_n0`, but NO synthetic snapshot on any other pair's tokens.
    """
    src = PolymarketDataSource(cache_root=bucket_cache)
    d = src.discover(start="1970-01-01", end="2999-12-31", kind="bucket")[0]
    evs = list(src.events(d))
    trade_ts = d.start_ts_ns + 100
    snapshots_at_ts = {e.symbol for e in evs if isinstance(e, BookSnapshot) and e.ts_ns == trade_ts}
    # Only pair 0 legs.
    assert snapshots_at_ts == {"tok_y0", "tok_n0"}
    # Confirm there's no parity inference to pair-1 / pair-2 legs at this ts.
    for sym in ("tok_y1", "tok_n1", "tok_y2", "tok_n2"):
        assert sym not in snapshots_at_ts


def test_bucket_per_leg_settlement(bucket_cache: Path) -> None:
    """2N settlement events at end_ts_ns, each keyed to its leg symbol with
    per-leg outcome reflecting the binary YES/NO of that strike market.
    """
    src = PolymarketDataSource(cache_root=bucket_cache)
    d = src.discover(start="1970-01-01", end="2999-12-31", kind="bucket")[0]
    evs = list(src.events(d))
    settles = [e for e in evs if isinstance(e, SettlementEvent)]
    assert len(settles) == 6  # 2 legs * 3 strike-pairs
    by_sym = {s.symbol: s.outcome for s in settles}
    # Pair 0 won (above 80k): yes=yes, no=no.
    assert by_sym["tok_y0"] == "yes" and by_sym["tok_n0"] == "no"
    # Pair 1 won (above 81k): yes=yes, no=no.
    assert by_sym["tok_y1"] == "yes" and by_sym["tok_n1"] == "no"
    # Pair 2 lost (not above 82k): yes=no, no=yes.
    assert by_sym["tok_y2"] == "no" and by_sym["tok_n2"] == "yes"
    # All at end_ts_ns.
    assert all(s.ts_ns == d.end_ts_ns for s in settles)


def test_bucket_events_monotone_ts(bucket_cache: Path) -> None:
    src = PolymarketDataSource(cache_root=bucket_cache)
    d = src.discover(start="1970-01-01", end="2999-12-31", kind="bucket")[0]
    evs = list(src.events(d))
    assert all(evs[i].ts_ns <= evs[i + 1].ts_ns for i in range(len(evs) - 1))


def test_discover_kind_both(binary_cache: Path, bucket_cache: Path, tmp_path: Path) -> None:
    """`kind='both'` returns binaries and buckets together."""
    # Build a combined cache.
    combined = tmp_path / "combined"
    combined.mkdir()
    bin_manifest = json.loads((binary_cache / "manifest.json").read_text())
    bk_manifest = json.loads((bucket_cache / "manifest.json").read_text())
    merged = {**bin_manifest, **bk_manifest}
    _write_manifest(combined, merged)
    # Copy trades from both caches.
    pm_trades = combined / "pm_trades"
    pm_trades.mkdir(parents=True)
    for src_cache in (binary_cache, bucket_cache):
        for f in (src_cache / "pm_trades").glob("*.parquet"):
            (pm_trades / f.name).write_bytes(f.read_bytes())
    # Klines: copy from one.
    _write_klines(combined, json.loads((binary_cache / "btc_klines" / "fixture.json").read_text()))

    src = PolymarketDataSource(cache_root=combined)
    descs = src.discover(start="1970-01-01", end="2999-12-31", kind="both")
    klasses = sorted([d.klass for d in descs])
    assert klasses == ["priceBinary", "priceBucket"]
