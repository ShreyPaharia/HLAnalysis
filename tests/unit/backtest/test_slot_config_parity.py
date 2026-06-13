"""SHR-99: a live StrategyConfig round-trips to byte-identical backtest params.

The backtest must run from the SAME decision config the live engine builds, not
a hand-written params JSON (reconstructed-config drift caused an $80 sim-vs-live
swing). ``backtest_params_from_slot`` reuses the engine's config builders; the
registry strategy built from its emitted params must reproduce the live decision
config field-for-field, for the slot default AND every per-class override.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from hlanalysis.backtest.runner.parallel import build_strategy_for_run
from hlanalysis.backtest.slot_config import backtest_params_from_slot
from hlanalysis.engine.config import load_strategies_config
from hlanalysis.engine.config_builders import (
    build_late_resolution_config,
    build_late_resolution_configs_by_class,
    build_theta_harvester_config,
    build_theta_harvester_configs_by_class,
)

LIVE = Path("config/strategy.yaml")


def _slots():
    return load_strategies_config(LIVE).strategies


def _live_default_cfg(cfg):
    if cfg.strategy_type == "theta_harvester":
        return build_theta_harvester_config(cfg)
    return build_late_resolution_config(cfg)


def _live_by_class(cfg):
    if cfg.strategy_type == "theta_harvester":
        return build_theta_harvester_configs_by_class(cfg)
    return build_late_resolution_configs_by_class(cfg)


@pytest.mark.parametrize("alias", [c.account_alias for c in _slots()])
def test_slot_default_roundtrips_byte_identical(alias):
    cfg = next(c for c in _slots() if c.account_alias == alias)
    strategy_id, params = backtest_params_from_slot(cfg)
    sim = build_strategy_for_run(strategy_id, params)
    assert asdict(sim.cfg) == asdict(_live_default_cfg(cfg))


@pytest.mark.parametrize("alias", [c.account_alias for c in _slots()])
def test_slot_per_class_roundtrips_byte_identical(alias):
    cfg = next(c for c in _slots() if c.account_alias == alias)
    for klass, live_cfg in _live_by_class(cfg).items():
        strategy_id, params = backtest_params_from_slot(cfg, klass=klass)
        sim = build_strategy_for_run(strategy_id, params)
        assert asdict(sim.cfg) == asdict(live_cfg), f"{alias}/{klass}"


def test_unknown_class_falls_back_to_default():
    """An unmatched class mirrors the live engine: fall through to the slot
    default config (the strategy does this at evaluation time)."""
    cfg = next(c for c in _slots() if c.strategy_type == "theta_harvester")
    _, params_default = backtest_params_from_slot(cfg)
    _, params_unknown = backtest_params_from_slot(cfg, klass="not_a_real_class")
    assert params_unknown == params_default


def test_vol_estimator_pinned_not_silently_bipower():
    """SHR-99 vol_estimator pin: a slot whose estimator resolves to sample_std
    must NOT silently become bipower in the sim. Building the slot's params
    through the registry reproduces the live estimator exactly."""
    cfg = next(
        c
        for c in _slots()
        if c.strategy_type == "theta_harvester" and build_theta_harvester_config(c).vol_estimator == "sample_std"
    )
    strategy_id, params = backtest_params_from_slot(cfg)
    sim = build_strategy_for_run(strategy_id, params)
    assert sim.cfg.vol_estimator == "sample_std"
