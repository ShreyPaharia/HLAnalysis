# tests/unit/test_sim_market_state.py
from __future__ import annotations

import math

from hlanalysis.sim.data.binance_klines import Kline
from hlanalysis.sim.market_state import SimMarketState
from hlanalysis.sim.synthetic_l2 import L2Snapshot


def test_book_update_then_read():
    s = SimMarketState(vol_sampling_dt_seconds=60)
    s.apply_l2(L2Snapshot(1, "t1", 0.49, 100, 0.51, 100))
    book = s.book("t1")
    assert book is not None and book.ask_px == 0.51 and book.bid_px == 0.49


def test_recent_returns_logreturns_from_klines():
    s = SimMarketState(vol_sampling_dt_seconds=60)
    s.apply_kline(Kline(ts_ns=0, open=100, high=100, low=100, close=100, volume=0))
    s.apply_kline(Kline(ts_ns=60_000_000_000, open=100, high=100, low=100, close=110, volume=0))
    rets = s.recent_returns(now_ns=60_000_000_000, lookback_seconds=120)
    assert len(rets) == 1
    assert abs(rets[0] - math.log(110 / 100)) < 1e-9


def test_recent_returns_window_filter():
    s = SimMarketState(vol_sampling_dt_seconds=60)
    for i in range(5):
        s.apply_kline(Kline(ts_ns=i * 60_000_000_000, open=100 + i, high=100 + i,
                            low=100 + i, close=100 + i + 1, volume=0))
    # only the last 2 returns should be in window
    rets = s.recent_returns(now_ns=4 * 60_000_000_000, lookback_seconds=120)
    assert len(rets) == 2
