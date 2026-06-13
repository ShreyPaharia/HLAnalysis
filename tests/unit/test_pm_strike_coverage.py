# tests/unit/test_pm_strike_coverage.py
"""SHR-54: PM strike must never silently freeze on a stale kline cache.

Two behaviours are pinned here:
  1. Guard — `_binary_strike` raises `StrikeCoverageError` when the strike
     reference ts falls past the end of the cached kline series (instead of
     silently returning the last candle's close as a frozen strike). Interior
     gaps (a missing minute with later candles present, e.g. weekend) still
     resolve to the nearest-preceding close.
  2. Coupled fetch — `_ensure_kline_coverage` pulls the missing forward range
     of Binance spot klines so the kline cache advances in lockstep with the PM
     market cache. It is a no-op when already covered, and for non-Binance
     underlyings (WTI uses Pyth, not Binance).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hlanalysis.backtest.core.data_source import QuestionDescriptor
from hlanalysis.backtest.data import polymarket as pmmod
from hlanalysis.backtest.data.binance_klines import Kline
from hlanalysis.backtest.data.polymarket import PolymarketDataSource, StrikeCoverageError

_MIN_NS = 60 * 1_000_000_000
_DAY_NS = 24 * 3600 * 1_000_000_000
_T0 = 1_700_000_000_000_000_000  # fixed arbitrary epoch-ns, minute-aligned


def _write_klines(cache_root: Path, n_minutes: int) -> None:
    """Write `n_minutes` contiguous 1m candles starting at _T0; close = 100+i."""
    d = cache_root / "btc_klines"
    d.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "ts_ns": _T0 + i * _MIN_NS,
            "open": 100.0 + i,
            "high": 100.0 + i,
            "low": 100.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
        }
        for i in range(n_minutes)
    ]
    (d / "k.json").write_text(json.dumps(rows))


def _q(end_ts_ns: int) -> QuestionDescriptor:
    return QuestionDescriptor(
        question_id="c1",
        question_idx=1,
        start_ts_ns=_T0,
        end_ts_ns=end_ts_ns,
        leg_symbols=("yes", "no"),
        klass="priceBinary",
        underlying="BTC",
    )


def test_binary_strike_raises_when_strike_past_cache(tmp_path: Path) -> None:
    _write_klines(tmp_path, 101)  # covers _T0 .. _T0+100min
    src = PolymarketDataSource(cache_root=tmp_path)
    # strike_ts = end - 24h = _T0 + 110min → 10 minutes past the last candle.
    q = _q(_T0 + 110 * _MIN_NS + _DAY_NS)
    with pytest.raises(StrikeCoverageError):
        src._binary_strike(q)


def test_binary_strike_returns_close_for_interior_gap(tmp_path: Path) -> None:
    _write_klines(tmp_path, 101)
    src = PolymarketDataSource(cache_root=tmp_path)
    # strike_ts = _T0 + 50min — interior, candle present.
    q = _q(_T0 + 50 * _MIN_NS + _DAY_NS)
    assert src._binary_strike(q) == 150.0  # close of candle index 50


def test_ensure_kline_coverage_fetches_missing_forward_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_klines(tmp_path, 101)  # covered up to _T0+100min
    src = PolymarketDataSource(cache_root=tmp_path)
    calls: dict[str, object] = {}

    def fake_fetch(start_ts_ms, end_ts_ms, symbol="BTCUSDT", interval="1m"):
        calls["args"] = (start_ts_ms, end_ts_ms, symbol)
        out = []
        t_ms = start_ts_ms
        while t_ms < end_ts_ms:
            out.append(Kline(ts_ns=t_ms * 1_000_000, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0))
            t_ms += 60_000
        return out

    monkeypatch.setattr(pmmod, "fetch_klines", fake_fetch)
    src._ensure_kline_coverage(_T0, _T0 + 200 * _MIN_NS)

    assert "args" in calls, "expected a forward fetch for the uncovered range"
    assert calls["args"][2] == "BTCUSDT"
    cov = src._kline_coverage()
    assert cov is not None and cov[1] >= _T0 + 199 * _MIN_NS


def test_ensure_kline_coverage_noop_when_already_covered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_klines(tmp_path, 101)
    src = PolymarketDataSource(cache_root=tmp_path)
    n = {"calls": 0}
    monkeypatch.setattr(
        pmmod,
        "fetch_klines",
        lambda *a, **k: n.__setitem__("calls", n["calls"] + 1) or [],
    )
    src._ensure_kline_coverage(_T0, _T0 + 50 * _MIN_NS)  # within coverage
    assert n["calls"] == 0


def test_ensure_kline_coverage_noop_for_non_binance_symbol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = PolymarketDataSource(
        cache_root=tmp_path,
        reference_symbol="WTI",
        klines_subdir="wti_klines",
    )
    n = {"calls": 0}
    monkeypatch.setattr(
        pmmod,
        "fetch_klines",
        lambda *a, **k: n.__setitem__("calls", n["calls"] + 1) or [],
    )
    src._ensure_kline_coverage(_T0, _T0 + 200 * _MIN_NS)
    assert n["calls"] == 0


def _binary_entry(cond_id: str, end_ts_ns: int) -> dict:
    return {
        "kind": "binary",
        "market": {
            "start_ts_ns": _T0,
            "end_ts_ns": end_ts_ns,
            "condition_id": cond_id,
            "yes_token_id": "y",
            "no_token_id": "n",
        },
    }


def test_fetch_and_cache_extends_kline_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fetch_and_cache must drag the kline series forward to cover the markets
    it just cached (coupled fetch) — not leave klines frozen behind the PM data."""
    src = PolymarketDataSource(cache_root=tmp_path)
    end_ts = _T0 + 200 * _MIN_NS

    def fake_binary(manifest, **kw):
        manifest["c1"] = _binary_entry("c1", end_ts)

    monkeypatch.setattr(src, "_fetch_and_cache_binary", fake_binary)
    monkeypatch.setattr(src, "_fetch_and_cache_bucket", lambda manifest, **kw: None)
    seen: dict[str, tuple[int, int]] = {}
    monkeypatch.setattr(
        src,
        "_ensure_kline_coverage",
        lambda s, e: seen.__setitem__("win", (s, e)),
    )
    src.fetch_and_cache(start="2020-01-01", end="2099-01-01", kind="binary")
    assert seen.get("win") == (_T0, end_ts)


def test_question_view_raises_on_stale_cache(tmp_path: Path) -> None:
    """The runtime consumption point: building a QuestionView for a market whose
    strike ts is past the cached klines must surface StrikeCoverageError, not a
    silently-frozen strike (fail loud beats a fabricated number)."""
    _write_klines(tmp_path, 101)  # covers _T0 .. _T0+100min
    src = PolymarketDataSource(cache_root=tmp_path)
    end_ts = _T0 + 300 * _MIN_NS + _DAY_NS  # strike _T0+300min, past cache
    (tmp_path / "manifest.json").write_text(json.dumps({"c1": _binary_entry("c1", end_ts)}))
    q = _q(end_ts)
    with pytest.raises(StrikeCoverageError):
        src.question_view(q, now_ns=_T0, settled=False)
