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
        _kline(base + 1 * _S, 101, 112, 98, 105),  # bucket-0 high=112, low=98
        _kline(base + 2 * _S, 105, 108, 100, 106),
        _kline(base + 3 * _S, 106, 109, 101, 107),
        _kline(base + 4 * _S, 107, 111, 103, 108),  # bucket-0 close=108
        _kline(base + 5 * _S, 108, 115, 107, 113),  # bucket-1
        _kline(base + 6 * _S, 113, 116, 109, 114),  # bucket-1 close=114, high=116, low=107
    ]
    _write_1s_klines(tmp_path, klines)
    ds = PolymarketDataSource(
        cache_root=tmp_path,
        reference_source="klines_1s",
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
        cache_root=tmp_path,
        reference_source="klines_1s",
        reference_resample_seconds=1,
    )
    # Half-open [start, end): bars at base+2..base+4 only.
    refs = ds._load_klines_1s_reference(base + 2 * _S, base + 5 * _S)
    assert [r.ts_ns for r in refs] == [base + 2 * _S, base + 3 * _S, base + 4 * _S]


def test_klines_1s_strike_still_uses_1m_klines(tmp_path: Path) -> None:
    # The 1s subdir feeds σ only; the canonical 1m btc_klines drives strikes.
    ds = PolymarketDataSource(
        cache_root=tmp_path,
        reference_source="klines_1s",
        reference_resample_seconds=5,
    )
    assert ds._klines_subdir == "btc_klines"  # strike source unchanged
    assert ds._klines_1s_subdir == "btc_klines_1s"  # σ source is separate


def test_klines_1s_changes_bundle_config_sig(tmp_path: Path) -> None:
    base = dict(cache_root=tmp_path, reference_resample_seconds=5)
    sig_klines = PolymarketDataSource(reference_source="klines", **base)._bundle_config_sig()
    sig_1s = PolymarketDataSource(reference_source="klines_1s", **base)._bundle_config_sig()
    sig_bbo = PolymarketDataSource(reference_source="binance_bbo", **base)._bundle_config_sig()
    assert len({sig_klines, sig_1s, sig_bbo}) == 3  # all distinct → no cache aliasing


def test_invalid_reference_source_still_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reference_source"):
        PolymarketDataSource(cache_root=tmp_path, reference_source="nonsense")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fix-1: _load_klines_1s_reference routes through resample_ohlc — bit-identical
# ---------------------------------------------------------------------------


def _old_load_klines_1s_reference_inline(
    klines_rows: list[dict],
    *,
    start_ns: int,
    end_ns: int,
    resample_ns: int,
    symbol: str,
) -> list[tuple]:
    """Reference copy of the OLD inline bucketing logic (pre-refactor).

    Kept here to prove the new ``_bucket_to_ref_events``-based path produces
    bit-identical output.  Do NOT change this function — it is the baseline.
    """
    from hlanalysis.backtest.core.events import ReferenceEvent

    out: list[ReferenceEvent] = []
    cur_bucket = None
    o = h = l = c = 0.0
    last_ts = 0
    for k in klines_rows:
        ts = int(k["ts_ns"])
        if ts < start_ns or ts >= end_ns:
            continue
        kh, kl, kc, ko = float(k["high"]), float(k["low"]), float(k["close"]), float(k["open"])
        bucket = ts // resample_ns
        if cur_bucket is None:
            cur_bucket = bucket
            o, h, l, c = ko, kh, kl, kc
            last_ts = ts
        elif bucket != cur_bucket:
            out.append(ReferenceEvent(ts_ns=last_ts, symbol=symbol, high=h, low=l, close=c, open=o))
            cur_bucket = bucket
            o, h, l, c = ko, kh, kl, kc
            last_ts = ts
        else:
            if kh > h:
                h = kh
            if kl < l:
                l = kl
            c = kc
            last_ts = ts
    if cur_bucket is not None:
        out.append(ReferenceEvent(ts_ns=last_ts, symbol=symbol, high=h, low=l, close=c, open=o))
    return [(r.ts_ns, r.high, r.low, r.close, r.open) for r in out]


def _old_load_binance_bbo_reference_inline(
    bbo_rows: list[tuple],
    *,
    resample_ns: int,
    symbol: str,
) -> list[tuple]:
    """Reference copy of the OLD inline bucketing logic for the BBO path.

    ``bbo_rows`` is a list of ``(ts, bid_px, ask_px)`` as returned by duckdb.
    Kept here as baseline; do NOT change.
    """
    from hlanalysis.backtest.core.events import ReferenceEvent

    out: list[ReferenceEvent] = []
    cur_bucket = None
    h = l = c = o = 0.0
    last_ts = 0
    for ts, bid, ask in bbo_rows:
        mid = (float(bid) + float(ask)) / 2.0
        bucket = int(ts) // resample_ns
        if cur_bucket is None:
            cur_bucket = bucket
            h = l = c = o = mid
            last_ts = int(ts)
        elif bucket != cur_bucket:
            out.append(ReferenceEvent(ts_ns=last_ts, symbol=symbol, high=h, low=l, close=c, open=o))
            cur_bucket = bucket
            h = l = c = o = mid
            last_ts = int(ts)
        else:
            if mid > h:
                h = mid
            if mid < l:
                l = mid
            c = mid
            last_ts = int(ts)
    out.append(ReferenceEvent(ts_ns=last_ts, symbol=symbol, high=h, low=l, close=c, open=o))
    return [(r.ts_ns, r.high, r.low, r.close, r.open) for r in out]


def test_klines_1s_resample_ohlc_path_bit_identical_to_old_inline(tmp_path: Path) -> None:
    """``_load_klines_1s_reference`` routed through ``resample_ohlc`` must produce
    bit-identical (ts_ns, high, low, close, open) tuples vs the old inline loop.

    Uses a representative input: multiple buckets, irregular spacing, distinct
    per-kline OHLC (not just scalar prices) so the high/low merge logic is
    exercised.
    """
    base = 100 * _S
    klines_data = [
        _kline(base + 0 * _S, 100.0, 110.0, 99.0, 101.0),
        _kline(base + 1 * _S, 101.0, 112.0, 98.0, 105.0),
        _kline(base + 2 * _S, 105.0, 108.0, 100.0, 106.0),
        _kline(base + 3 * _S, 106.0, 109.0, 101.0, 107.0),
        _kline(base + 4 * _S, 107.0, 111.0, 103.0, 108.0),
        # bucket boundary at base+5s (5s buckets)
        _kline(base + 5 * _S, 108.0, 115.0, 107.0, 113.0),
        _kline(base + 6 * _S, 113.0, 116.0, 109.0, 114.0),
        _kline(base + 7 * _S, 114.0, 117.0, 111.0, 115.0),
        _kline(base + 8 * _S, 115.0, 118.0, 112.0, 116.0),
        _kline(base + 9 * _S, 116.0, 119.0, 113.0, 117.0),
        # partial third bucket
        _kline(base + 10 * _S, 117.0, 120.0, 114.0, 118.0),
        _kline(base + 11 * _S, 118.0, 121.0, 115.0, 119.0),
    ]
    _write_1s_klines(tmp_path, klines_data)

    resample_ns = 5 * _S
    start_ns = base
    end_ns = base + 12 * _S

    # New path: routed through resample_ohlc via _bucket_to_ref_events.
    ds = PolymarketDataSource(
        cache_root=tmp_path,
        reference_source="klines_1s",
        reference_resample_seconds=5,
    )
    new_refs = ds._load_klines_1s_reference(start_ns, end_ns)
    new_tuples = [(r.ts_ns, r.high, r.low, r.close, r.open) for r in new_refs]

    # Old path: inline copy above.
    old_tuples = _old_load_klines_1s_reference_inline(
        klines_data,
        start_ns=start_ns,
        end_ns=end_ns,
        resample_ns=resample_ns,
        symbol="BTC",
    )

    assert len(new_tuples) == len(old_tuples), f"bar count differs: new={len(new_tuples)} old={len(old_tuples)}"
    for i, (new, old) in enumerate(zip(new_tuples, old_tuples)):
        assert new == old, f"bar {i} differs: new={new} old={old} — resample_ohlc path is NOT bit-identical"


def test_bbo_resample_ohlc_path_bit_identical_to_old_inline() -> None:
    """``_bucket_to_ref_events`` on scalar BBO ticks (high=low=close=open=mid)
    must be bit-identical to the old BBO inline loop.

    This covers the ``_load_binance_bbo_reference`` refactor: each BBO tick
    contributes a scalar mid price, and the bucketer tracks high=max(mid),
    low=min(mid), close=last(mid), open=first(mid).
    """
    from hlanalysis.backtest.data.polymarket import _bucket_to_ref_events

    _NS = 1_000_000_000
    bbo_rows = [
        (0 * _NS, 99.9, 100.1),  # mid=100.0   bucket 0
        (1 * _NS, 101.9, 102.1),  # mid=102.0   bucket 0
        (2 * _NS, 97.9, 98.1),  # mid= 98.0   bucket 0  ← new low
        (3 * _NS, 103.9, 104.1),  # mid=104.0   bucket 0  ← new high
        (4 * _NS, 99.9, 100.1),  # mid=100.0   bucket 0  close=100.0
        (5 * _NS, 105.9, 106.1),  # mid=106.0   bucket 1
        (7 * _NS, 103.9, 104.1),  # mid=104.0   bucket 1
        (9 * _NS, 107.9, 108.1),  # mid=108.0   bucket 1  close=108.0
    ]
    resample_ns = 5 * _NS
    symbol = "BTC"

    # Old inline path (reference baseline).
    old_tuples = _old_load_binance_bbo_reference_inline(bbo_rows, resample_ns=resample_ns, symbol=symbol)

    # New path: build samples the way _load_binance_bbo_reference does.
    samples = [
        (
            int(ts),
            (float(bid) + float(ask)) / 2.0,
            (float(bid) + float(ask)) / 2.0,
            (float(bid) + float(ask)) / 2.0,
            (float(bid) + float(ask)) / 2.0,
        )
        for ts, bid, ask in bbo_rows
    ]
    new_refs = _bucket_to_ref_events(samples, symbol=symbol, bucket_ns=resample_ns)
    new_tuples = [(r.ts_ns, r.high, r.low, r.close, r.open) for r in new_refs]

    assert len(new_tuples) == len(old_tuples), f"bar count differs: new={len(new_tuples)} old={len(old_tuples)}"
    for i, (new, old) in enumerate(zip(new_tuples, old_tuples)):
        assert new == old, f"BBO bar {i} differs: new={new} old={old} — not bit-identical"
