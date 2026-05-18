from __future__ import annotations

import json
from pathlib import Path

import pytest

from hlanalysis.backtest.data.kalshi import (
    KalshiDataSource,
    _thresholds_from_markets,
    ContiguityError,
)


def _write_bucket_manifest(root: Path, *, thresholds: list[float],
                           leg_markets: list[str],
                           leg_settlements: list[str]) -> None:
    (root / "manifest.json").write_text(json.dumps({
        "KXBTCD-26MAY18": {
            "n_rows": 0,
            "last_pull_ts_ns": 0,
            "kind": "bucket",
            "bucket": {
                "event_ticker": "KXBTCD-26MAY18",
                "series_ticker": "KXBTCD",
                "start_ts_ns": 1_700_000_000_000_000_000,
                "end_ts_ns":   1_700_086_400_000_000_000,
                "thresholds": thresholds,
                "leg_markets": leg_markets,
                "leg_strike_ranges": [
                    [None, thresholds[0]],
                    *[[thresholds[i], thresholds[i + 1]] for i in range(len(thresholds) - 1)],
                    [thresholds[-1], None],
                ],
                "leg_settlements": leg_settlements,
                "mutex_verified": True,
                "settlement_close_price": 80123.45,
            },
        },
    }))


def test_descriptor_from_bucket_builds_leg_symbols_in_strike_order(tmp_path):
    _write_bucket_manifest(
        tmp_path,
        thresholds=[79000.0, 80000.0],
        leg_markets=["KXBTCD-26MAY18-B79000", "KXBTCD-26MAY18-B79500",
                     "KXBTCD-26MAY18-B80000"],
        leg_settlements=["no", "yes", "no"],
    )
    ds = KalshiDataSource(cache_root=tmp_path)
    descs = ds.discover(start="2023-11-14", end="2023-11-16")
    assert len(descs) == 1
    d = descs[0]
    assert d.klass == "priceBucket"
    assert d.underlying == "BTC"
    assert d.leg_symbols == (
        "KXBTCD-26MAY18-B79000|yes", "KXBTCD-26MAY18-B79000|no",
        "KXBTCD-26MAY18-B79500|yes", "KXBTCD-26MAY18-B79500|no",
        "KXBTCD-26MAY18-B80000|yes", "KXBTCD-26MAY18-B80000|no",
    )


def test_question_view_emits_priceThresholds_kv(tmp_path):
    _write_bucket_manifest(
        tmp_path,
        thresholds=[79000.0, 80000.0],
        leg_markets=["A", "B", "C"],
        leg_settlements=["yes", "no", "no"],
    )
    ds = KalshiDataSource(cache_root=tmp_path)
    d = ds.discover(start="2023-11-14", end="2023-11-16")[0]
    qv = ds.question_view(d, now_ns=d.end_ts_ns + 1, settled=True)
    assert dict(qv.kv)["priceThresholds"] == "79000,80000"
    assert qv.klass == "priceBucket"
    assert qv.period == "1d"
    assert qv.underlying == "BTC"
    assert qv.settled is True


def test_thresholds_from_contiguous_markets():
    markets = [
        {"ticker": "M0", "floor_strike": None,    "cap_strike": 79000.0},
        {"ticker": "M1", "floor_strike": 79000.0, "cap_strike": 80000.0},
        {"ticker": "M2", "floor_strike": 80000.0, "cap_strike": None},
    ]
    thresholds, ordered = _thresholds_from_markets(markets)
    assert thresholds == [79000.0, 80000.0]
    assert [m["ticker"] for m in ordered] == ["M0", "M1", "M2"]


def test_thresholds_rejects_gap():
    markets = [
        {"ticker": "M0", "floor_strike": None,    "cap_strike": 79000.0},
        {"ticker": "M1", "floor_strike": 79500.0, "cap_strike": 80000.0},
        {"ticker": "M2", "floor_strike": 80000.0, "cap_strike": None},
    ]
    with pytest.raises(ContiguityError, match="gap"):
        _thresholds_from_markets(markets)


def test_thresholds_rejects_overlap():
    markets = [
        {"ticker": "M0", "floor_strike": None,    "cap_strike": 79500.0},
        {"ticker": "M1", "floor_strike": 79000.0, "cap_strike": 80000.0},
        {"ticker": "M2", "floor_strike": 80000.0, "cap_strike": None},
    ]
    with pytest.raises(ContiguityError, match="overlap"):
        _thresholds_from_markets(markets)


def test_thresholds_rejects_missing_boundary():
    markets = [
        {"ticker": "M0", "floor_strike": 78000.0, "cap_strike": 79000.0},
        {"ticker": "M1", "floor_strike": 79000.0, "cap_strike": None},
    ]
    with pytest.raises(ContiguityError, match="boundary"):
        _thresholds_from_markets(markets)


def _make_manifest_entry(
    *, event_ticker: str, thresholds: list[float], leg_markets: list[str],
    leg_settlements: list[str], leg_strike_ranges: list[list],
) -> dict:
    return {
        "n_rows": 0,
        "last_pull_ts_ns": 0,
        "kind": "bucket",
        "bucket": {
            "event_ticker": event_ticker,
            "series_ticker": "KXBTCD",
            "start_ts_ns": 1_700_000_000_000_000_000,
            "end_ts_ns":   1_700_086_400_000_000_000,
            "thresholds": thresholds,
            "leg_markets": leg_markets,
            "leg_strike_ranges": leg_strike_ranges,
            "leg_settlements": leg_settlements,
            "mutex_verified": True,
            "settlement_close_price": None,
        },
    }


def test_audit_passes_on_clean_corpus(tmp_path):
    manifest = {
        "KXBTCD-26MAY18": _make_manifest_entry(
            event_ticker="KXBTCD-26MAY18",
            thresholds=[79000.0, 80000.0],
            leg_markets=["M0", "M1", "M2"],
            leg_settlements=["no", "yes", "no"],
            leg_strike_ranges=[[None, 79000.0], [79000.0, 80000.0], [80000.0, None]],
        ),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    ds = KalshiDataSource(cache_root=tmp_path)
    summary = ds.audit()
    assert summary["mutex_pass"] == 1
    assert summary["mutex_fail_zero_yes"] == 0
    assert summary["mutex_fail_multi_yes"] == 0
    assert summary["contiguity_fail"] == 0
    assert summary["mutex_rate"] == 1.0
    assert summary["failing_event_tickers"] == []
    written = json.loads((tmp_path / "fetch_summary.json").read_text())
    assert written["mutex_rate"] == 1.0


def test_audit_flags_zero_yes_and_multi_yes(tmp_path):
    manifest = {
        "OK": _make_manifest_entry(
            event_ticker="OK",
            thresholds=[79000.0], leg_markets=["A", "B"],
            leg_settlements=["yes", "no"],
            leg_strike_ranges=[[None, 79000.0], [79000.0, None]],
        ),
        "ZERO": _make_manifest_entry(
            event_ticker="ZERO",
            thresholds=[79000.0], leg_markets=["A", "B"],
            leg_settlements=["no", "no"],
            leg_strike_ranges=[[None, 79000.0], [79000.0, None]],
        ),
        "MULTI": _make_manifest_entry(
            event_ticker="MULTI",
            thresholds=[79000.0], leg_markets=["A", "B"],
            leg_settlements=["yes", "yes"],
            leg_strike_ranges=[[None, 79000.0], [79000.0, None]],
        ),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    ds = KalshiDataSource(cache_root=tmp_path)
    summary = ds.audit()
    assert summary["mutex_pass"] == 1
    assert summary["mutex_fail_zero_yes"] == 1
    assert summary["mutex_fail_multi_yes"] == 1
    assert set(summary["failing_event_tickers"]) == {"ZERO", "MULTI"}
    assert summary["mutex_rate"] == pytest.approx(1.0 / 3)


def test_audit_flags_contiguity_failure(tmp_path):
    manifest = {
        "BAD": _make_manifest_entry(
            event_ticker="BAD",
            thresholds=[79000.0], leg_markets=["A", "B"],
            leg_settlements=["yes", "no"],
            leg_strike_ranges=[[None, 79000.0], [79500.0, None]],  # gap
        ),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    ds = KalshiDataSource(cache_root=tmp_path)
    summary = ds.audit()
    assert summary["contiguity_fail"] == 1
    assert summary["failing_event_tickers"] == ["BAD"]


import pyarrow as pa
import pyarrow.parquet as pq

from hlanalysis.backtest.core.events import (
    BookSnapshot,
    SettlementEvent,
    TradeEvent,
    ReferenceEvent,
)


def _write_trades(cache_root: Path, market: str, rows: list[dict]) -> None:
    out_dir = cache_root / "kalshi_trades"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows) if rows else pa.table({
        "ts_ns": pa.array([], type=pa.int64()),
        "yes_price": pa.array([], type=pa.float64()),
        "size": pa.array([], type=pa.float64()),
        "taker_side": pa.array([], type=pa.string()),
    })
    pq.write_table(table, out_dir / f"{market}.parquet")


def _write_klines(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))


def test_events_emits_within_market_parity(tmp_path, monkeypatch):
    # one event, two markets (M0, M1); one trade on M0 YES at price 0.42.
    manifest = {
        "EVT": _make_manifest_entry(
            event_ticker="EVT", thresholds=[80000.0],
            leg_markets=["M0", "M1"],
            leg_settlements=["yes", "no"],
            leg_strike_ranges=[[None, 80000.0], [80000.0, None]],
        ),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    _write_trades(tmp_path, "M0", [{"ts_ns": 1_700_000_001_000_000_000,
                                    "yes_price": 0.42, "size": 5.0,
                                    "taker_side": "yes"}])
    _write_trades(tmp_path, "M1", [])
    # Empty kline cache.
    monkeypatch.setenv("HLBT_BINANCE_KLINES", str(tmp_path / "empty_klines"))
    _write_klines([], tmp_path / "empty_klines" / "BTCUSDT-1m.json")

    ds = KalshiDataSource(cache_root=tmp_path)
    d = ds.discover(start="2023-11-14", end="2023-11-16")[0]
    evs = list(ds.events(d))
    snaps = [e for e in evs if isinstance(e, BookSnapshot)]
    yes_snap = [s for s in snaps if s.symbol == "M0|yes"][0]
    no_snap = [s for s in snaps if s.symbol == "M0|no"][0]
    # Half-spread default 0.005: YES book bid=0.415 ask=0.425.
    assert yes_snap.bids[0][0] == pytest.approx(0.415, abs=1e-6) or \
           yes_snap.asks[0][0] == pytest.approx(0.425, abs=1e-6)
    # NO leg parity at 1 - 0.42 = 0.58: bid=0.575 ask=0.585.
    assert no_snap.bids[0][0] == pytest.approx(0.575, abs=1e-6) or \
           no_snap.asks[0][0] == pytest.approx(0.585, abs=1e-6)


def test_events_emits_settlement_per_leg(tmp_path, monkeypatch):
    manifest = {
        "EVT": _make_manifest_entry(
            event_ticker="EVT", thresholds=[80000.0],
            leg_markets=["M0", "M1"],
            leg_settlements=["yes", "no"],
            leg_strike_ranges=[[None, 80000.0], [80000.0, None]],
        ),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    _write_trades(tmp_path, "M0", [])
    _write_trades(tmp_path, "M1", [])
    monkeypatch.setenv("HLBT_BINANCE_KLINES", str(tmp_path / "empty_klines"))
    _write_klines([], tmp_path / "empty_klines" / "BTCUSDT-1m.json")

    ds = KalshiDataSource(cache_root=tmp_path)
    d = ds.discover(start="2023-11-14", end="2023-11-16")[0]
    evs = list(ds.events(d))
    settles = [e for e in evs if isinstance(e, SettlementEvent)]
    by_sym = {s.symbol: s for s in settles}
    assert by_sym["M0|yes"].outcome == "yes"
    assert by_sym["M0|no"].outcome == "no"
    assert by_sym["M1|yes"].outcome == "no"
    assert by_sym["M1|no"].outcome == "yes"
    for s in settles:
        assert s.ts_ns == d.end_ts_ns
