# tests/unit/test_late_resolution_config_parity.py
"""Train/serve-skew guard for the v1 late_resolution live config path.

The same SHR-65 failure mode the theta path had: the live builder
``_late_resolution_config_from_entry`` was a hand-maintained ``getattr`` subset
that DROPPED fields the backtest builder ``build_v1_late_resolution`` forwards
(``drift_aware_d``, ``exit_bid_floor``, ``exit_safety_d_5m``,
``exit_vol_lookback_5m_seconds``, ``size_scaling``, ``size_min_fraction`` and the
``vol_scaled_tte_*`` family). A knob set in the live YAML silently fell back to
the dataclass default → the live engine ran a *different* strategy than the
backtest that justified it. These tests pin ``AllowlistEntry`` and
``LateResolutionConfig`` together and assert reflection-based forwarding.
"""
from __future__ import annotations

import dataclasses

import pytest
from pydantic import ValidationError

from hlanalysis.engine.config import AllowlistEntry, GlobalRiskConfig
from hlanalysis.engine.runtime import _late_resolution_config_from_entry
from hlanalysis.strategy.late_resolution import LateResolutionConfig

# Fields the engine sources from the strategy GLOBAL block, not the allowlist
# entry. Everything else on LateResolutionConfig must be settable per entry.
_GLOBAL_SOURCED = {
    "max_strike_distance_pct",
    "min_recent_volume_usd",
    "stale_data_halt_seconds",
}

# The exact set the live builder used to drop — the headline regression.
_PREVIOUSLY_DROPPED = {
    "drift_aware_d",
    "exit_bid_floor",
    "exit_safety_d_5m",
    "exit_vol_lookback_5m_seconds",
    "size_scaling",
    "size_min_fraction",
    "vol_scaled_tte_enabled",
    "vol_scaled_tte_ref_sigma",
    "vol_scaled_tte_exponent",
    "vol_scaled_tte_ceiling_seconds",
}


def _minimal_entry(**overrides) -> AllowlistEntry:
    base = dict(
        match={"class": "priceBinary"},
        max_position_usd=100.0,
        stop_loss_pct=None,
        tte_min_seconds=60,
        tte_max_seconds=3600,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=10.0,
        vol_max=2.0,
    )
    base.update(overrides)
    return AllowlistEntry(**base)


def _global() -> GlobalRiskConfig:
    return GlobalRiskConfig(
        max_total_inventory_usd=1000.0,
        max_concurrent_positions=5,
        daily_loss_cap_usd=500.0,
        max_strike_distance_pct=50.0,
        min_recent_volume_usd=0.0,
        stale_data_halt_seconds=86400,
        reconcile_interval_seconds=15,
    )


def test_allowlist_entry_declares_every_late_resolution_knob() -> None:
    """AllowlistEntry must declare every LateResolutionConfig field except the
    handful sourced from the GLOBAL block. A field on the dataclass but not on
    AllowlistEntry is unsettable live → silent train/serve skew."""
    dataclass_fields = {f.name for f in dataclasses.fields(LateResolutionConfig)}
    entry_fields = set(AllowlistEntry.model_fields.keys())
    must_be_settable = dataclass_fields - _GLOBAL_SOURCED

    missing = must_be_settable - entry_fields
    assert not missing, (
        f"AllowlistEntry is missing late_resolution knobs (silently unsettable "
        f"live): {sorted(missing)}"
    )


def test_allowlist_entry_rejects_unknown_keys() -> None:
    """A typo'd or unsupported knob must fail loudly at load, not be silently
    dropped (extra='forbid')."""
    with pytest.raises(ValidationError):
        _minimal_entry(definitely_not_a_real_knob=1.23)


def test_builder_forwards_previously_dropped_fields_when_set() -> None:
    """Regression for the headline bug: each field the old getattr-subset
    dropped must now reach LateResolutionConfig when set on the entry."""
    entry = _minimal_entry(
        drift_aware_d=True,
        exit_bid_floor=0.05,
        exit_safety_d_5m=0.7,
        exit_vol_lookback_5m_seconds=600,
        size_scaling="sqrt_safety",
        size_min_fraction=0.4,
        vol_scaled_tte_enabled=True,
        vol_scaled_tte_ref_sigma=0.01,
        vol_scaled_tte_exponent=2.0,
        vol_scaled_tte_ceiling_seconds=43200,
    )
    cfg = _late_resolution_config_from_entry(entry, global_=_global())
    assert cfg.drift_aware_d is True
    assert cfg.exit_bid_floor == 0.05
    assert cfg.exit_safety_d_5m == 0.7
    assert cfg.exit_vol_lookback_5m_seconds == 600
    assert cfg.size_scaling == "sqrt_safety"
    assert cfg.size_min_fraction == 0.4
    assert cfg.vol_scaled_tte_enabled is True
    assert cfg.vol_scaled_tte_ref_sigma == 0.01
    assert cfg.vol_scaled_tte_exponent == 2.0
    assert cfg.vol_scaled_tte_ceiling_seconds == 43200


def test_builder_forwards_all_shared_fields() -> None:
    """Every field shared between AllowlistEntry and LateResolutionConfig (minus
    the global-sourced and the stop_loss_pct sentinel mapping) must be forwarded
    verbatim — no silent default fallback. Uses non-default values to catch drops."""
    entry = _minimal_entry(
        price_extreme_max=0.99,
        min_safety_d=1.5,
        vol_lookback_seconds=900,
        exit_safety_d=0.8,
        vol_ewma_lambda=0.97,
        vol_estimator="parkinson",
        vol_sampling_dt_seconds=5,
        size_cap_near_strike_pct=0.5,
        use_bid_for_entry_gate=True,
        min_bid_notional_usd=15.0,
        topup_enabled=False,
        fee_model="pm_binary",
        fee_rate=0.07,
    )
    cfg = _late_resolution_config_from_entry(entry, global_=_global())
    shared = (
        set(AllowlistEntry.model_fields.keys())
        & {f.name for f in dataclasses.fields(LateResolutionConfig)}
    )
    for name in shared - _GLOBAL_SOURCED - {"stop_loss_pct"}:
        assert getattr(cfg, name) == getattr(entry, name), (
            f"field {name!r} not forwarded: entry={getattr(entry, name)!r} "
            f"cfg={getattr(cfg, name)!r}"
        )


def test_stop_loss_none_maps_to_disabled_sentinel() -> None:
    cfg = _late_resolution_config_from_entry(_minimal_entry(stop_loss_pct=None), global_=_global())
    assert cfg.stop_loss_pct >= 1e8  # disabled sentinel
    cfg2 = _late_resolution_config_from_entry(_minimal_entry(stop_loss_pct=30.0), global_=_global())
    assert cfg2.stop_loss_pct == 30.0


def test_global_sourced_fields_come_from_global() -> None:
    g = _global()
    cfg = _late_resolution_config_from_entry(_minimal_entry(), global_=g)
    assert cfg.max_strike_distance_pct == g.max_strike_distance_pct
    assert cfg.min_recent_volume_usd == g.min_recent_volume_usd
    assert cfg.stale_data_halt_seconds == g.stale_data_halt_seconds


def test_defaults_unchanged_when_knobs_unset() -> None:
    """Today's effective live behavior is preserved: an entry that sets none of
    the new knobs yields the dataclass defaults (all 'off')."""
    cfg = _late_resolution_config_from_entry(_minimal_entry(), global_=_global())
    assert cfg.drift_aware_d is False
    assert cfg.exit_bid_floor == 0.0
    assert cfg.exit_safety_d_5m == 0.0
    assert cfg.exit_vol_lookback_5m_seconds == 300
    assert cfg.size_scaling == "fixed"
    assert cfg.size_min_fraction == 0.25
    assert cfg.vol_scaled_tte_enabled is False
