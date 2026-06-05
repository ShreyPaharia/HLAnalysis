"""``reference_source="klines_1s"``: pull genuine Binance 1s klines and bucket
them to ``vol_sampling_dt_seconds`` OHLC ReferenceEvents.

This is the on-demand-klines counterpart to the recorded ``binance_bbo`` spot
feed — the BTC-ref-equivalence experiment compares the two at dt=5. The 1s
klines live in their own cache subdir (``btc_klines_1s``) so the canonical 1m
``btc_klines`` (used for PM strike resolution) is untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hlanalysis.backtest.data.polymarket import PolymarketDataSource

_S = 1_000_000_000  # 1 second in ns


def _write_1s_klines(cache_root: Path, klines: list[dict]) -> None:
    d = cache_root / "btc_klines_1s"
    d.mkdir(parents=True, exist_ok=True)
    (d / "fixture.json").write_text(json.dumps(klines))


def _kline(ts_ns: int, o: float, h: float, l: float, c: float) -> dict:
    return {"ts_ns": ts_ns, "open": o, "high": h, "low": l, "close": c, "volume": 1.0}


def test_klines_1s_is_a_valid_reference_source(tmp_path: Path) -> None:
    ds = PolymarketDataSource(cache_root=tmp_path, reference_source="klines_1s")
    assert ds._reference_source == "klines_1s"


def test_klines_1s_buckets_to_resample_seconds_ohlc(tmp_path: Path) -> None:
    # Five 1s bars in one 5s bucket, two in the next. dt=5 → two ReferenceEvents.
    base = 10 * _S  # ts aligned so floor(ts/5s) groups [10,11,12,13,14] then [15,16]
    klines = [
        _kline(base + 0 * _S, 100, 110, 99, 101),
        _kline(base + 1 * _S, 101, 112, 98, 105),   # bucket-0 high=112, low=98
        _kline(base + 2 * _S, 105, 108, 100, 106),
        _kline(base + 3 * _S, 106, 109, 101, 107),
        _kline(base + 4 * _S, 107, 111, 103, 108),  # bucket-0 close=108
        _kline(base + 5 * _S, 108, 115, 107, 113),  # bucket-1
        _kline(base + 6 * _S, 113, 116, 109, 114),  # bucket-1 close=114, high=116, low=107
    ]
    _write_1s_klines(tmp_path, klines)
    ds = PolymarketDataSource(
        cache_root=tmp_path, reference_source="klines_1s",
        reference_resample_seconds=5,
    )
    refs = ds._load_klines_1s_reference(base, base + 7 * _S)
    assert len(refs) == 2
    b0, b1 = refs
    assert b0.symbol == "BTC"
    assert b0.open == 100 and b0.high == 112 and b0.low == 98 and b0.close == 108
    assert b1.open == 108 and b1.high == 116 and b1.low == 107 and b1.close == 114
    # Monotone, bucket ts is the last bar in the bucket.
    assert b0.ts_ns < b1.ts_ns
    assert b1.ts_ns == base + 6 * _S


def test_klines_1s_window_filter(tmp_path: Path) -> None:
    base = 10 * _S
    klines = [_kline(base + i * _S, 100, 100, 100, 100) for i in range(10)]
    _write_1s_klines(tmp_path, klines)
    ds = PolymarketDataSource(
        cache_root=tmp_path, reference_source="klines_1s",
        reference_resample_seconds=1,
    )
    # Half-open [start, end): bars at base+2..base+4 only.
    refs = ds._load_klines_1s_reference(base + 2 * _S, base + 5 * _S)
    assert [r.ts_ns for r in refs] == [base + 2 * _S, base + 3 * _S, base + 4 * _S]


def test_klines_1s_strike_still_uses_1m_klines(tmp_path: Path) -> None:
    # The 1s subdir feeds σ only; the canonical 1m btc_klines drives strikes.
    ds = PolymarketDataSource(
        cache_root=tmp_path, reference_source="klines_1s",
        reference_resample_seconds=5,
    )
    assert ds._klines_subdir == "btc_klines"          # strike source unchanged
    assert ds._klines_1s_subdir == "btc_klines_1s"    # σ source is separate


def test_klines_1s_changes_bundle_config_sig(tmp_path: Path) -> None:
    base = dict(cache_root=tmp_path, reference_resample_seconds=5)
    sig_klines = PolymarketDataSource(reference_source="klines", **base)._bundle_config_sig()
    sig_1s = PolymarketDataSource(reference_source="klines_1s", **base)._bundle_config_sig()
    sig_bbo = PolymarketDataSource(reference_source="binance_bbo", **base)._bundle_config_sig()
    assert len({sig_klines, sig_1s, sig_bbo}) == 3  # all distinct → no cache aliasing


def test_invalid_reference_source_still_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reference_source"):
        PolymarketDataSource(cache_root=tmp_path, reference_source="nonsense")  # type: ignore[arg-type]
