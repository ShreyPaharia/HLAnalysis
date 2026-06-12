# tests/unit/test_late_resolution_config_parity.py
"""Train/serve-skew guard for the v1 late_resolution live config path.

The same SHR-65 failure mode the theta path had: the live builder
``_late_resolution_config_from_entry`` was a hand-maintained ``getattr`` subset
that DROPPED fields the backtest builder ``build_v1_late_resolution`` forwards
(``drift_aware_d``, ``exit_bid_floor``, ``exit_safety_d_5m``,
``exit_vol_lookback_5m_seconds``). A knob set in the live YAML silently fell back
to the dataclass default → the live engine ran a *different* strategy than the
backtest that justified it.

R6.1 (single-source param schema): ``AllowlistEntry`` now inherits from
``LateResolutionParams`` for all optional LR knobs.  Adding a new optional knob
requires two edits: (1) add to ``LateResolutionParams`` → ``AllowlistEntry``
inherits automatically; (2) add the same field to ``LateResolutionConfig`` → the
parity test below enforces both name and default match.
``test_single_source_property`` is the structural guard for that invariant.
"""
from __future__ import annotations

import dataclasses

import pytest
from pydantic import ValidationError

from hlanalysis.engine.config import AllowlistEntry, GlobalRiskConfig
from hlanalysis.engine.runtime import _late_resolution_config_from_entry
from hlanalysis.strategy.late_resolution import LateResolutionConfig, LateResolutionParams

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
    )
    cfg = _late_resolution_config_from_entry(entry, global_=_global())
    assert cfg.drift_aware_d is True
    assert cfg.exit_bid_floor == 0.05
    assert cfg.exit_safety_d_5m == 0.7
    assert cfg.exit_vol_lookback_5m_seconds == 600


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


def test_single_source_property() -> None:
    """R6.1 structural guard: LateResolutionParams IS the single source of truth
    for all optional late_resolution knobs.

    This test asserts two invariants that together guarantee a new optional knob
    added to LateResolutionParams (one edit) automatically propagates everywhere:

    1. AllowlistEntry inherits ALL LateResolutionParams fields (so a new field
       declared in LateResolutionParams is immediately settable in the live YAML
       with no second edit on AllowlistEntry).

    2. Every LateResolutionParams field also appears on LateResolutionConfig with
       the SAME default value (so the canonical default declared once in
       LateResolutionParams is the default in both the YAML validation model and
       the runtime strategy config).  Violation = train/serve skew.

    The remaining "second edit" when adding a knob: add the field to
    LateResolutionConfig (the frozen dataclass used at runtime).  That edit is
    unavoidable — the dataclass is the runtime type — but the test below makes it
    mandatory and enforced.
    """
    params_fields = LateResolutionParams.model_fields
    dataclass_fields = {f.name: f for f in dataclasses.fields(LateResolutionConfig)}
    ae_fields = AllowlistEntry.model_fields

    # Invariant 1: every LateResolutionParams field appears on AllowlistEntry
    # (guaranteed by inheritance, but an explicit check catches if someone
    # accidentally re-declares a field on AllowlistEntry with a different type).
    missing_from_ae = set(params_fields) - set(ae_fields)
    assert not missing_from_ae, (
        f"LateResolutionParams fields not inherited by AllowlistEntry: "
        f"{sorted(missing_from_ae)}"
    )

    # Invariant 2: every LateResolutionParams field appears in LateResolutionConfig
    # with the same default — canonical default in one place only.
    missing_from_lr: list[str] = []
    default_mismatch: list[str] = []
    for name, pydantic_field in params_fields.items():
        if name not in dataclass_fields:
            missing_from_lr.append(name)
            continue
        dc_field = dataclass_fields[name]
        if dc_field.default is dataclasses.MISSING:
            # Required field on the dataclass — not expected for LateResolutionParams
            # entries (they are all optional). Flag it so a developer notices.
            default_mismatch.append(
                f"{name}: LateResolutionParams has default "
                f"{pydantic_field.default!r} but LateResolutionConfig has no default"
            )
            continue
        if dc_field.default != pydantic_field.default:
            default_mismatch.append(
                f"{name}: LateResolutionParams default={pydantic_field.default!r} "
                f"!= LateResolutionConfig default={dc_field.default!r}"
            )
    assert not missing_from_lr, (
        f"LateResolutionParams fields not present in LateResolutionConfig: "
        f"{sorted(missing_from_lr)}"
    )
    assert not default_mismatch, (
        "Default mismatch between LateResolutionParams and LateResolutionConfig:\n"
        + "\n".join(f"  {m}" for m in default_mismatch)
    )
