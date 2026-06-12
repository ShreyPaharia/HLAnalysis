# tests/unit/test_config_validation_hardening.py
"""TDD tests for config validation hardening (Fix 1).

Covers:
- extra='forbid' on GlobalRiskConfig (and other leaf risk/money config models)
- Range bounds on numeric risk knobs
- Production config/strategy.yaml still loads cleanly (regression check)
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from hlanalysis.engine.config import (
    GlobalRiskConfig,
    StrategiesConfig,
    StrategyConfig,
    load_strategies_config,
)


# ---------------------------------------------------------------------------
# Helper: minimal valid GlobalRiskConfig dict (matches prod shape)
# ---------------------------------------------------------------------------
_VALID_GLOBAL = dict(
    max_total_inventory_usd=1000.0,
    max_concurrent_positions=5,
    daily_loss_cap_usd=100.0,
    max_strike_distance_pct=50.0,
    min_recent_volume_usd=100.0,
    stale_data_halt_seconds=30,
    reconcile_interval_seconds=15,
)


# ---------------------------------------------------------------------------
# A) Valid existing config still loads
# ---------------------------------------------------------------------------

def test_global_risk_config_valid_loads():
    """A well-formed GlobalRiskConfig (matching prod shape) must still load."""
    cfg = GlobalRiskConfig(**_VALID_GLOBAL)
    assert cfg.max_total_inventory_usd == 1000.0
    assert cfg.daily_loss_cap_usd == 100.0


# ---------------------------------------------------------------------------
# B) Unknown key raises on forbid-models
# ---------------------------------------------------------------------------

def test_global_risk_config_unknown_key_raises():
    """A typo'd key must raise ValidationError with extra='forbid'."""
    bad = {**_VALID_GLOBAL, "max_total_inventory_uusd": 999.0}  # typo: double 'u'
    with pytest.raises(ValidationError):
        GlobalRiskConfig(**bad)


# ---------------------------------------------------------------------------
# C) Out-of-range values raise
# ---------------------------------------------------------------------------

def test_global_risk_daily_window_hour_too_high():
    """daily_window_start_hour_utc must be in [0, 23]."""
    with pytest.raises(ValidationError):
        GlobalRiskConfig(**{**_VALID_GLOBAL, "daily_window_start_hour_utc": 25})


def test_global_risk_daily_window_hour_negative():
    """daily_window_start_hour_utc must be >= 0."""
    with pytest.raises(ValidationError):
        GlobalRiskConfig(**{**_VALID_GLOBAL, "daily_window_start_hour_utc": -1})


def test_global_risk_max_total_inventory_negative():
    """max_total_inventory_usd must be >= 0."""
    with pytest.raises(ValidationError):
        GlobalRiskConfig(**{**_VALID_GLOBAL, "max_total_inventory_usd": -1.0})


def test_global_risk_daily_loss_cap_negative():
    """daily_loss_cap_usd must be >= 0."""
    with pytest.raises(ValidationError):
        GlobalRiskConfig(**{**_VALID_GLOBAL, "daily_loss_cap_usd": -50.0})


def test_global_risk_max_slippage_pct_negative():
    """max_slippage_pct must be >= 0."""
    with pytest.raises(ValidationError):
        GlobalRiskConfig(**{**_VALID_GLOBAL, "max_slippage_pct": -0.01})


def test_global_risk_min_order_notional_negative():
    """min_order_notional_usd must be >= 0."""
    with pytest.raises(ValidationError):
        GlobalRiskConfig(**{**_VALID_GLOBAL, "min_order_notional_usd": -5.0})


def test_global_risk_reconcile_interval_negative():
    """reconcile_interval_seconds must be > 0."""
    with pytest.raises(ValidationError):
        GlobalRiskConfig(**{**_VALID_GLOBAL, "reconcile_interval_seconds": -1})


def test_global_risk_stale_data_halt_negative():
    """stale_data_halt_seconds must be > 0."""
    with pytest.raises(ValidationError):
        GlobalRiskConfig(**{**_VALID_GLOBAL, "stale_data_halt_seconds": -5})


# ---------------------------------------------------------------------------
# D) Production config/strategy.yaml still loads (CRITICAL regression check)
# ---------------------------------------------------------------------------

def test_production_strategy_yaml_still_loads():
    """The real production config must load without error after adding extra='forbid'.

    This is the most important regression check: if any field in the live YAML
    is not present in GlobalRiskConfig, this will raise ValidationError.
    """
    strategies = load_strategies_config(Path("config/strategy.yaml"))
    assert len(strategies.strategies) >= 2
    # All slots have positive money caps
    for s in strategies.strategies:
        assert s.global_.max_total_inventory_usd > 0
        assert s.global_.daily_loss_cap_usd > 0
        assert s.global_.reconcile_interval_seconds > 0
