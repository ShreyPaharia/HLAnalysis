# tests/unit/engine/test_config_pm_bucket_slots.py
"""Config smoke tests for the two paper-mode PM multi-strike bucket slots.

Verifies that v31_pm_btc_ms and v31_pm_eth_ms load cleanly from
config/strategy.yaml, carry the correct reference symbols, paper_mode=True,
and that the (bucket-only) base theta config carries the tuned cell
(fav0.75/eb0.02/vlb1800). exit_safety_d is held at the safety-guardrail floor
(1.0), NOT PR #12's tuned 0.5 — see config/strategy.yaml and
test_bucket_override_risk_gates_not_below_binary.

These slots allowlist ONLY priceBucket and carry no theta_overrides, so the
base theta block IS the effective bucket config.
"""
from __future__ import annotations

from pathlib import Path

from hlanalysis.engine.config import load_strategies_config
from hlanalysis.engine.config_builders import build_theta_harvester_configs_by_class
from hlanalysis.engine.runtime import build_theta_harvester_config


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
        # Bucket-only slot: no per-class override; base theta is the config.
        assert build_theta_harvester_configs_by_class(s) == {}, (
            f"slot {s.account_alias}: bucket-only slot should carry no theta_overrides"
        )
        base = build_theta_harvester_config(s)
        assert base.favorite_threshold == 0.75, (
            f"slot {s.account_alias}: favorite_threshold={base.favorite_threshold}"
        )
        assert base.edge_buffer == 0.02, (
            f"slot {s.account_alias}: edge_buffer={base.edge_buffer}"
        )
        assert base.vol_lookback_seconds == 1800, (
            f"slot {s.account_alias}: vol_lookback_seconds={base.vol_lookback_seconds}"
        )
        # esd held at the guardrail floor (1.0), not PR #12's 0.5.
        assert base.exit_safety_d == 1.0, (
            f"slot {s.account_alias}: exit_safety_d={base.exit_safety_d}"
        )
        assert base.vol_estimator == "bipower", (
            f"slot {s.account_alias}: vol_estimator={base.vol_estimator}"
        )
