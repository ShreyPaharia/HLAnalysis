"""Canonical OHLC bucketer — single source of truth for dt-wide bar resampling.

Three independent implementations of "bucket reference-price ticks into dt-wide
OHLC bars" previously lived in:
  - engine/market_state.py        (streaming, the LIVE path)
  - backtest/data/hl_hip4.py      (generator)
  - backtest/data/_fastpath_core.py (list; explicit twin of hl_hip4)

They must produce bit-identical bars or the live engine and the backtest see a
different σ input (the train/serve skew this collapse eliminates). These tests
pin the canonical bucketer's exact semantics: floor(ts/dt) bucket assignment,
high=max / low=min / close=last selection, last-tick timestamp per bar.
"""

from __future__ import annotations

from hlanalysis.marketdata.ohlc import bucket_index, resample_ohlc, update_bar


def test_bucket_index_is_floor_div() -> None:
    assert bucket_index(0, 5) == 0
    assert bucket_index(4, 5) == 0
    assert bucket_index(5, 5) == 1
    assert bucket_index(11, 5) == 2


def test_update_bar_high_max_low_min_close_last() -> None:
    # high = max (tie keeps existing), low = min (tie keeps existing), close = last
    assert update_bar((102.0, 99.0, 99.0), 101.0, 101.0, 101.0) == (102.0, 99.0, 101.0)
    assert update_bar((100.0, 100.0, 100.0), 105.0, 95.0, 102.0) == (105.0, 95.0, 102.0)
    # tie on high/low keeps the existing extreme value object
    assert update_bar((102.0, 99.0, 100.0), 102.0, 99.0, 101.0) == (102.0, 99.0, 101.0)


def test_resample_ohlc_scalar_stream() -> None:
    """Scalar (high=low=close=price) ticks bucketed at dt=5s. Each output bar is
    (last_tick_ts_ns, high, low, close)."""
    s = 1_000_000_000  # 1s in ns
    ticks = [
        (0 * s, 100.0, 100.0, 100.0),
        (1 * s, 102.0, 102.0, 102.0),
        (2 * s, 99.0, 99.0, 99.0),
        (5 * s, 101.0, 101.0, 101.0),
        (7 * s, 103.0, 103.0, 103.0),
        (12 * s, 98.0, 98.0, 98.0),
    ]
    bars = list(resample_ohlc(ticks, bucket_ns=5 * s))
    assert bars == [
        (2 * s, 102.0, 99.0, 99.0),
        (7 * s, 103.0, 101.0, 103.0),
        (12 * s, 98.0, 98.0, 98.0),
    ]


def test_resample_ohlc_merges_ranged_bars() -> None:
    """Ranged input bars (distinct high/low/close) merge within a bucket via
    update_bar — high=max across the bucket, low=min, close=last."""
    s = 1_000_000_000
    bars_in = [
        (0 * s, 105.0, 95.0, 100.0),
        (3 * s, 108.0, 99.0, 107.0),
        (4 * s, 106.0, 90.0, 92.0),
        (6 * s, 101.0, 100.0, 100.5),
    ]
    bars = list(resample_ohlc(bars_in, bucket_ns=5 * s))
    assert bars == [
        (4 * s, 108.0, 90.0, 92.0),  # bucket 0: max(105,108,106)=108, min(95,99,90)=90, close=92
        (6 * s, 101.0, 100.0, 100.5),  # bucket 1
    ]


def test_resample_ohlc_empty() -> None:
    assert list(resample_ohlc([], bucket_ns=5)) == []


def test_all_three_loaders_produce_identical_bars() -> None:
    """The generator (hl_hip4), the list twin (_fastpath_core), and the
    canonical resampler must emit byte-identical bars from one tick stream —
    the cross-implementation parity that the collapse guarantees."""
    from hlanalysis.backtest.core.events import ReferenceEvent
    from hlanalysis.backtest.data._fastpath_core import _resample_reference_rows
    from hlanalysis.backtest.data.hl_hip4 import _resample_reference

    s = 1_000_000_000
    resample_ns = 5 * s
    # Dense, irregular tick stream spanning several buckets (mirrors a real
    # bbo-mid feed: scalar high=low=close per tick).
    prices = [100.0, 101.5, 99.2, 103.7, 98.0, 100.1, 102.2, 97.5, 101.0, 100.0]
    ticks_ns = [0, 1, 2, 5, 7, 9, 12, 13, 20, 26]
    raw = [ReferenceEvent(t * s, "BTC", p, p, p) for t, p in zip(ticks_ns, prices)]

    gen_bars = [ev for _ts, ev in _resample_reference(iter((r.ts_ns, r) for r in raw), resample_ns=resample_ns)]
    list_bars = _resample_reference_rows(raw, resample_ns=resample_ns)
    canon = list(resample_ohlc(((r.ts_ns, r.high, r.low, r.close) for r in raw), bucket_ns=resample_ns))

    def as_tuples(bars: list[ReferenceEvent]) -> list[tuple]:
        return [(b.ts_ns, b.high, b.low, b.close) for b in bars]

    assert as_tuples(gen_bars) == as_tuples(list_bars)
    assert as_tuples(gen_bars) == canon
