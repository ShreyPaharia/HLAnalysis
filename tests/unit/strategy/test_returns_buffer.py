from __future__ import annotations

import math

import pytest

from hlanalysis.strategy._numba.returns_buffer import KlineRingBuffer

NS_PER_MIN = 60_000_000_000


def test_empty_buffer_returns_empty_window():
    buf = KlineRingBuffer()
    rets, hls = buf.slice_window(now_ns=0, lookback_seconds=120)
    assert rets == ()
    assert hls == ()


def test_single_kline_yields_no_returns_but_one_hl():
    buf = KlineRingBuffer()
    buf.append(ts_ns=0, high=100.5, low=99.5, close=100.0)
    rets, hls = buf.slice_window(now_ns=0, lookback_seconds=120)
    assert rets == ()
    assert hls == ((100.5, 99.5),)


def test_two_klines_yield_one_log_return():
    buf = KlineRingBuffer()
    buf.append(ts_ns=0, high=100.0, low=100.0, close=100.0)
    buf.append(ts_ns=NS_PER_MIN, high=110.0, low=109.0, close=110.0)
    rets, hls = buf.slice_window(now_ns=NS_PER_MIN, lookback_seconds=120)
    assert len(rets) == 1
    assert math.isclose(rets[0], math.log(110 / 100), rel_tol=1e-12)
    # Both bars in window.
    assert hls == ((100.0, 100.0), (110.0, 109.0))


def test_window_filter_only_keeps_in_window_returns():
    # Exact mirror of legacy test_recent_returns_window_filter in
    # tests/unit/test_sim_market_state.py.
    buf = KlineRingBuffer()
    for i in range(5):
        buf.append(ts_ns=i * NS_PER_MIN, high=100.0 + i, low=100.0 + i,
                   close=100.0 + i + 1)
    rets, _hls = buf.slice_window(now_ns=4 * NS_PER_MIN, lookback_seconds=120)
    # Cutoff = 4 - 2 = 2 minutes. Kept closes: ts=2,3,4 → 2 returns.
    assert len(rets) == 2


def test_skip_returns_for_nonpositive_prices():
    buf = KlineRingBuffer()
    buf.append(ts_ns=0, high=1.0, low=1.0, close=0.0)
    buf.append(ts_ns=NS_PER_MIN, high=1.0, low=1.0, close=100.0)
    buf.append(ts_ns=2 * NS_PER_MIN, high=1.0, low=1.0, close=110.0)
    rets, _hls = buf.slice_window(now_ns=2 * NS_PER_MIN, lookback_seconds=3600)
    # 0→100 has prev_close==0 → no return. 100→110 emits one.
    assert len(rets) == 1
    assert math.isclose(rets[0], math.log(110 / 100), rel_tol=1e-12)


def test_grows_past_initial_capacity():
    buf = KlineRingBuffer(initial_capacity=4)
    for i in range(20):
        buf.append(ts_ns=i * NS_PER_MIN, high=1.0, low=1.0, close=1.0 + i)
    rets, hls = buf.slice_window(now_ns=19 * NS_PER_MIN, lookback_seconds=1_000_000)
    # All 20 bars in window → 19 returns.
    assert len(rets) == 19
    assert len(hls) == 20


def test_latest_close():
    buf = KlineRingBuffer()
    assert buf.latest_close() is None
    buf.append(ts_ns=0, high=1.0, low=1.0, close=42.0)
    assert buf.latest_close() == 42.0
    buf.append(ts_ns=NS_PER_MIN, high=1.0, low=1.0, close=43.5)
    assert buf.latest_close() == 43.5


def test_hl_excludes_degenerate_bars():
    buf = KlineRingBuffer()
    buf.append(ts_ns=0, high=0.0, low=1.0, close=1.0)  # h=0 excluded
    buf.append(ts_ns=NS_PER_MIN, high=1.0, low=0.0, close=1.0)  # l=0 excluded
    buf.append(ts_ns=2 * NS_PER_MIN, high=1.5, low=0.5, close=1.0)  # kept
    _rets, hls = buf.slice_window(now_ns=2 * NS_PER_MIN, lookback_seconds=3600)
    assert hls == ((1.5, 0.5),)


@pytest.mark.parametrize("now_min,lookback,expected_rets",
                         [(0, 120, 0), (4, 120, 2), (4, 60, 1), (4, 0, 0)])
def test_window_boundaries(now_min, lookback, expected_rets):
    buf = KlineRingBuffer()
    for i in range(5):
        buf.append(ts_ns=i * NS_PER_MIN, high=100.0, low=100.0,
                   close=100.0 + i + 1)
    rets, _ = buf.slice_window(now_ns=now_min * NS_PER_MIN,
                               lookback_seconds=lookback)
    assert len(rets) == expected_rets
