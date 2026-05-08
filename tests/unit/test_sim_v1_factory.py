from __future__ import annotations

from hlanalysis.sim.v1_factory import build_v1_strategy_from_params


def test_factory_produces_v1_with_overridable_params():
    s = build_v1_strategy_from_params({
        "tte_min_seconds": 3600,
        "tte_max_seconds": 86400,
        "price_extreme_threshold": 0.92,
        "distance_from_strike_usd_min": 500,
        "vol_max": 0.02,
        "stop_loss_pct": 10,
    })
    assert s.cfg.tte_max_seconds == 86400
    assert s.cfg.stop_loss_pct == 10
