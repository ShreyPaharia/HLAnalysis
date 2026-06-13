"""R7 payoff test — registry-driven strategy build.

Verifies that:
1. A DUMMY live strategy registered in the test builds via the engine path
   (_build_strategy_for_slot) without any edit to config.py, config_builders.py,
   or slot_config.py.
2. StrategyConfig validation rejects an unregistered strategy_type.
3. The live registry exposes the registered types and registry_id correctly.
4. The backtest slot path (backtest_params_from_slot) resolves registry_id from
   the live registry (not from a hardcoded dict).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from hlanalysis.engine.config import (
    AllowlistEntry,
    GlobalRiskConfig,
    StrategyConfig,
    ThetaParams,
)
from hlanalysis.strategy.base import Strategy
from hlanalysis.strategy.live_registry import (
    _reset_for_tests,
    build_live_strategy,
    live_registry_id,
    live_strategy_types,
    register_live_strategy,
)


# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------


def _entry(**kw) -> AllowlistEntry:
    base = dict(
        match={"class": "priceBinary"},
        max_position_usd=100.0,
        stop_loss_pct=None,
        tte_min_seconds=0,
        tte_max_seconds=7200,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=0.0,
        vol_max=1.0,
    )
    base.update(kw)
    return AllowlistEntry(**base)


def _global() -> GlobalRiskConfig:
    return GlobalRiskConfig(
        max_total_inventory_usd=500.0,
        max_concurrent_positions=5,
        daily_loss_cap_usd=200.0,
        max_strike_distance_pct=10.0,
        min_recent_volume_usd=1000.0,
        stale_data_halt_seconds=30,
        reconcile_interval_seconds=60,
    )


def _make_cfg(strategy_type: str = "late_resolution") -> StrategyConfig:
    return StrategyConfig(
        name="test",
        paper_mode=True,
        allowlist=[_entry()],
        defaults=_entry(),
        **{"global": _global()},
        strategy_type=strategy_type,
    )


# ---------------------------------------------------------------------------
# Minimal dummy strategy for payoff test
# ---------------------------------------------------------------------------


@dataclass
class _DummyConfig:
    tag: str = "dummy"


class _DummyStrategy(Strategy):
    name = "dummy_live"

    def __init__(self, cfg: _DummyConfig) -> None:
        self.cfg = cfg

    def evaluate(self, **_: Any):  # type: ignore[override]
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Dummy strategy builds via engine path without touching the 3 dispatch points
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def _restore_live_registry():
    """Snapshot and restore the live registry after each test that mutates it."""
    from hlanalysis.strategy.live_registry import _LIVE_REGISTRY

    snapshot = dict(_LIVE_REGISTRY)
    yield
    _LIVE_REGISTRY.clear()
    _LIVE_REGISTRY.update(snapshot)


def test_dummy_strategy_builds_via_engine_path(_restore_live_registry):
    """Registering a DUMMY strategy in the live registry lets _build_strategy_for_slot
    construct it — no edit to config_builders.py, config.py, or slot_config.py required.
    """
    from hlanalysis.engine.config_builders import _build_strategy_for_slot

    # Register a dummy live strategy.
    def _dummy_builder(cfg: StrategyConfig) -> _DummyStrategy:
        return _DummyStrategy(_DummyConfig(tag=cfg.account_alias))

    register_live_strategy(
        "dummy_live",
        live_builder=_dummy_builder,
        registry_id="v_dummy_backtest",
    )

    # StrategyConfig must now accept strategy_type="dummy_live" (no edit to config.py).
    cfg = _make_cfg(strategy_type="dummy_live")

    # _build_strategy_for_slot must build it (no edit to config_builders.py).
    strategy = _build_strategy_for_slot(cfg)
    assert isinstance(strategy, _DummyStrategy)
    assert strategy.cfg.tag == "default"  # account_alias defaults to "default"

    # live_registry_id must resolve it (no edit to slot_config.py).
    assert live_registry_id("dummy_live") == "v_dummy_backtest"


# ---------------------------------------------------------------------------
# 2. Unknown strategy_type is still rejected
# ---------------------------------------------------------------------------


def test_unknown_strategy_type_rejected_by_config():
    """StrategyConfig must raise a ValidationError for an unregistered strategy_type."""
    with pytest.raises(Exception, match="unknown strategy_type"):
        _make_cfg(strategy_type="completely_unknown_strategy_xyz")


def test_build_live_strategy_raises_for_unknown():
    """build_live_strategy must raise ValueError for an unregistered type."""
    cfg = _make_cfg()  # valid config (late_resolution)
    with pytest.raises(ValueError, match="unknown strategy_type"):
        build_live_strategy("not_registered_xyz", cfg)


# ---------------------------------------------------------------------------
# 3. Live registry exposes correct types and registry_ids
# ---------------------------------------------------------------------------


def test_live_strategy_types_includes_builtins():
    """Both built-in types must be in the live registry."""
    types = live_strategy_types()
    assert "late_resolution" in types
    assert "theta_harvester" in types


def test_live_registry_id_for_late_resolution():
    assert live_registry_id("late_resolution") == "v1_late_resolution"


def test_live_registry_id_for_theta_harvester():
    assert live_registry_id("theta_harvester") == "v3_theta_harvester"


def test_live_registry_id_raises_for_unknown():
    with pytest.raises(KeyError, match="has no live registry entry"):
        live_registry_id("not_a_real_type_xyz")


# ---------------------------------------------------------------------------
# 4. Backtest slot path uses live registry_id
# ---------------------------------------------------------------------------


def test_slot_config_uses_live_registry_id():
    """backtest_params_from_slot must resolve its strategy_id from the live
    registry — not a hardcoded dict. Verify for both built-in live types."""
    from hlanalysis.backtest.slot_config import backtest_params_from_slot

    cfg_v1 = _make_cfg(strategy_type="late_resolution")
    strategy_id, _ = backtest_params_from_slot(cfg_v1)
    assert strategy_id == live_registry_id("late_resolution")

    cfg_v31 = StrategyConfig(
        name="theta",
        paper_mode=False,
        allowlist=[_entry()],
        defaults=_entry(),
        **{"global": _global()},
        strategy_type="theta_harvester",
        theta=ThetaParams(),
    )
    strategy_id, _ = backtest_params_from_slot(cfg_v31)
    assert strategy_id == live_registry_id("theta_harvester")
