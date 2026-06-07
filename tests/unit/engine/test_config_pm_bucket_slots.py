# tests/unit/engine/test_config_pm_bucket_slots.py
"""Config smoke tests for the two paper-mode PM multi-strike bucket slots.

Verifies that v31_pm_btc_ms and v31_pm_eth_ms load cleanly from
config/strategy.yaml, carry the correct reference symbols, paper_mode=True,
and that the priceBucket theta override carries the tuned cell values
(fav0.75/eb0.02/vlb1800/esd0.5).
"""
from __future__ import annotations

from pathlib import Path

from hlanalysis.engine.config import load_strategies_config
from hlanalysis.engine.config_builders import build_theta_harvester_configs_by_class


def test_bucket_slots_load_with_tuned_cell_and_refs():
    cfg = load_strategies_config(Path("config/strategy.yaml"))
    by_alias = {s.account_alias: s for s in cfg.strategies}
    assert "v31_pm_btc_ms" in by_alias, "v31_pm_btc_ms slot missing from strategy.yaml"
    assert "v31_pm_eth_ms" in by_alias, "v31_pm_eth_ms slot missing from strategy.yaml"
    btc = by_alias["v31_pm_btc_ms"]
    eth = by_alias["v31_pm_eth_ms"]
    assert btc.paper_mode is True
    assert eth.paper_mode is True
    assert btc.reference_symbol == "BTCUSDT_SPOT"
    assert eth.reference_symbol == "ETHUSDT_SPOT"
    for s in (btc, eth):
        overrides = build_theta_harvester_configs_by_class(s)
        assert "priceBucket" in overrides, (
            f"slot {s.account_alias}: theta_overrides missing priceBucket key"
        )
        bucket = overrides["priceBucket"]
        assert bucket.favorite_threshold == 0.75, (
            f"slot {s.account_alias}: favorite_threshold={bucket.favorite_threshold}"
        )
        assert bucket.edge_buffer == 0.02, (
            f"slot {s.account_alias}: edge_buffer={bucket.edge_buffer}"
        )
        assert bucket.vol_lookback_seconds == 1800, (
            f"slot {s.account_alias}: vol_lookback_seconds={bucket.vol_lookback_seconds}"
        )
        assert bucket.exit_safety_d == 0.5, (
            f"slot {s.account_alias}: exit_safety_d={bucket.exit_safety_d}"
        )
