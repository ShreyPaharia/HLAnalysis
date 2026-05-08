# tests/unit/test_sim_hftbt_adapter.py
from __future__ import annotations

from hlanalysis.sim.data.binance_klines import Kline
from hlanalysis.sim.data.schemas import PMTrade
from hlanalysis.sim.hftbt_adapter import EventKind, build_event_stream


def test_events_emitted_in_chronological_order():
    trades = [
        PMTrade(ts_ns=2, token_id="t", side="buy", price=0.5, size=1),
        PMTrade(ts_ns=4, token_id="t", side="sell", price=0.51, size=1),
    ]
    klines = [
        Kline(ts_ns=1, open=1, high=1, low=1, close=1, volume=0),
        Kline(ts_ns=3, open=1, high=1, low=1, close=2, volume=0),
        Kline(ts_ns=5, open=1, high=1, low=1, close=3, volume=0),
    ]
    events = list(build_event_stream(trades=trades, klines=klines, half_spread=0.005, depth=10.0))
    ts = [e.ts_ns for e in events]
    assert ts == sorted(ts)
    kinds = [e.kind for e in events]
    assert EventKind.KLINE in kinds
    assert EventKind.L2 in kinds
