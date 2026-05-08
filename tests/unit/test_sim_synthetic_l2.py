# tests/unit/test_sim_synthetic_l2.py
from __future__ import annotations

from hlanalysis.sim.data.schemas import PMTrade
from hlanalysis.sim.synthetic_l2 import L2Snapshot, trade_to_l2


def test_trade_yields_symmetric_l2_around_price():
    t = PMTrade(ts_ns=1, token_id="t1", side="buy", price=0.50, size=100)
    snap = trade_to_l2(t, half_spread=0.005, depth=500.0)
    assert isinstance(snap, L2Snapshot)
    assert snap.token_id == "t1"
    assert abs(snap.bid_px - 0.495) < 1e-9
    assert abs(snap.ask_px - 0.505) < 1e-9
    assert snap.bid_sz == 500.0
    assert snap.ask_sz == 500.0


def test_trade_clamps_l2_into_unit_interval():
    t = PMTrade(ts_ns=1, token_id="t1", side="buy", price=0.999, size=1)
    snap = trade_to_l2(t, half_spread=0.05, depth=10.0)
    assert snap.bid_px >= 0.0 and snap.ask_px <= 1.0
