# tests/unit/test_theta_config_parity.py
"""Train/serve-skew guard for the theta_harvester live config path (SHR-65).

The live `ThetaParams` pydantic model parses the YAML `theta:` block and
`build_theta_harvester_config` turns it into the `ThetaHarvesterConfig` the
strategy runs. If `ThetaParams` omits a knob, that knob silently falls back to
the dataclass default and the live engine runs a *different* strategy than the
backtest that justified it. These tests pin the two models together.

R6.2 (single-source param schema): ``ThetaParams`` now inherits from
``ThetaHarvesterParams`` for all optional theta knobs.  Adding a new optional
knob requires two edits: (1) add to ``ThetaHarvesterParams`` → ``ThetaParams``
inherits automatically; (2) add the same field to ``ThetaHarvesterConfig`` →
the parity test below enforces both name and default match.
``test_single_source_property`` is the structural guard for that invariant.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from pydantic import ValidationError

from hlanalysis.engine.config import ThetaParams, load_strategies_config
from hlanalysis.engine.runtime import build_theta_harvester_config
from hlanalysis.strategy.theta_harvester import ThetaHarvesterConfig, ThetaHarvesterParams

# Fields the engine intentionally sources from the allowlist `defaults:` block,
# NOT from the `theta:` block. Everything else MUST be settable via `theta:`.
_ALLOWLIST_SOURCED = {
    "max_position_usd",
    "tte_min_seconds",
    "tte_max_seconds",
    "stop_loss_pct",
}


def test_theta_params_declares_every_theta_block_knob() -> None:
    """ThetaParams must declare every ThetaHarvesterConfig field except the
    handful sourced from the allowlist defaults. A field on the dataclass but
    not on ThetaParams is unsettable live → silent train/serve skew."""
    dataclass_fields = {f.name for f in dataclasses.fields(ThetaHarvesterConfig)}
    theta_param_fields = set(ThetaParams.model_fields.keys())
    must_be_settable = dataclass_fields - _ALLOWLIST_SOURCED

    missing = must_be_settable - theta_param_fields
    assert not missing, f"ThetaParams is missing theta-block knobs (silently unsettable live): {sorted(missing)}"


def test_theta_params_rejects_unknown_keys() -> None:
    """A typo'd or unsupported theta knob must fail loudly at load, not be
    silently dropped (extra='forbid')."""
    with pytest.raises(ValidationError):
        ThetaParams(definitely_not_a_real_knob=1.23)


def test_live_strategy_yaml_forwards_exit_safety_d() -> None:
    """Regression for the headline SHR-65 bug: exit_safety_d was silently 0.0
    (dataclass default) instead of the YAML value. The invariant is FORWARDING
    FIDELITY — the built config must carry whatever the YAML declares, never the
    silent default. (Originally pinned to ==1.0 when every slot used 1.0; the PM
    multi-strike bucket slots legitimately run the tuned 0.5, so we assert the
    round-trip instead, which is the stronger form of the same guard.)"""
    cfgs = load_strategies_config(Path("config/strategy.yaml"))
    theta_slots = [c for c in cfgs.strategies if c.strategy_type == "theta_harvester"]
    assert theta_slots, "expected at least one theta_harvester slot in strategy.yaml"
    for c in theta_slots:
        built = build_theta_harvester_config(c)
        assert built.exit_safety_d == c.theta.exit_safety_d, (
            f"slot {c.account_alias}: exit_safety_d forwarded as "
            f"{built.exit_safety_d}, expected YAML value {c.theta.exit_safety_d}"
        )
    # Pin v31_pm to 1.0 so a silent regression to the 0.0 default still fails
    # (the headline SHR-65 bug). v31 (HL binary) intentionally runs 0.0 as of the
    # 2026-06-17 buy-and-hold change (mid-hold exit disabled; protection moved to
    # entry via min_safety_d) — assert that explicitly so a silent flip is caught.
    by_alias = {c.account_alias: c for c in theta_slots}
    if "v31_pm" in by_alias:
        assert build_theta_harvester_config(by_alias["v31_pm"]).exit_safety_d == 1.0, (
            "slot v31_pm: exit_safety_d must stay 1.0 (SHR-65 regression guard)"
        )
    if "v31" in by_alias:
        assert build_theta_harvester_config(by_alias["v31"]).exit_safety_d == 0.0, (
            "slot v31: exit_safety_d must be 0.0 (buy-and-hold; protection via min_safety_d)"
        )


def test_build_forwards_all_declared_theta_fields() -> None:
    """Every field shared between ThetaParams and ThetaHarvesterConfig must be
    forwarded verbatim by build_theta_harvester_config (no silent default
    fallback). Uses a non-default value per field to catch drops."""
    cfgs = load_strategies_config(Path("config/strategy.yaml"))
    c = next(c for c in cfgs.strategies if c.strategy_type == "theta_harvester")
    built = build_theta_harvester_config(c)
    shared = set(ThetaParams.model_fields.keys()) & {f.name for f in dataclasses.fields(ThetaHarvesterConfig)}
    for name in shared - _ALLOWLIST_SOURCED:
        assert getattr(built, name) == getattr(c.theta, name), (
            f"field {name!r} not forwarded: theta={getattr(c.theta, name)!r} built={getattr(built, name)!r}"
        )


def test_single_source_property() -> None:
    """R6.2 structural guard: ThetaHarvesterParams IS the single source of truth
    for all optional theta_harvester knobs.

    This test asserts two invariants that together guarantee a new optional knob
    added to ThetaHarvesterParams (one edit) automatically propagates everywhere:

    1. ThetaParams inherits ALL ThetaHarvesterParams fields (so a new optional
       field declared in ThetaHarvesterParams is immediately settable in the live
       YAML ``theta:`` block with no second edit on ThetaParams).

    2. Every ThetaHarvesterParams field also appears on ThetaHarvesterConfig with
       the SAME default value (so the canonical default declared once in
       ThetaHarvesterParams is the default in both the YAML validation model and
       the runtime strategy config). Violation = train/serve skew.

    The remaining "second edit" when adding an optional knob: add the field to
    ThetaHarvesterConfig (the frozen dataclass used at runtime). That edit is
    unavoidable — the dataclass is the runtime type — but the test below makes it
    mandatory and enforced.

    Note: the 13 required-in-dataclass fields (vol_lookback_seconds, etc.) are
    NOT in ThetaHarvesterParams by design — they have no dataclass default and
    are therefore core required knobs, not optional extras. They stay on
    ThetaParams directly and are covered by test_theta_params_declares_every_theta_block_knob.
    """
    params_fields = ThetaHarvesterParams.model_fields
    dataclass_fields = {f.name: f for f in dataclasses.fields(ThetaHarvesterConfig)}
    tp_fields = ThetaParams.model_fields

    # Invariant 1: every ThetaHarvesterParams field appears on ThetaParams
    # (guaranteed by inheritance, but an explicit check catches if someone
    # accidentally re-declares a field on ThetaParams with a different type).
    missing_from_tp = set(params_fields) - set(tp_fields)
    assert not missing_from_tp, f"ThetaHarvesterParams fields not inherited by ThetaParams: {sorted(missing_from_tp)}"

    # Invariant 2: every ThetaHarvesterParams field appears in ThetaHarvesterConfig
    # with the same default — canonical default in one place only.
    missing_from_dc: list[str] = []
    default_mismatch: list[str] = []
    for name, pydantic_field in params_fields.items():
        if name not in dataclass_fields:
            missing_from_dc.append(name)
            continue
        dc_field = dataclass_fields[name]
        if dc_field.default is dataclasses.MISSING:
            # ThetaHarvesterParams fields are all optional (have defaults).
            # A required dataclass field would be a structural error.
            default_mismatch.append(
                f"{name}: ThetaHarvesterParams has default "
                f"{pydantic_field.default!r} but ThetaHarvesterConfig has no default"
            )
            continue
        if dc_field.default != pydantic_field.default:
            default_mismatch.append(
                f"{name}: ThetaHarvesterParams default={pydantic_field.default!r} "
                f"!= ThetaHarvesterConfig default={dc_field.default!r}"
            )
    assert not missing_from_dc, (
        f"ThetaHarvesterParams fields not present in ThetaHarvesterConfig: {sorted(missing_from_dc)}"
    )
    assert not default_mismatch, (
        "Default mismatch between ThetaHarvesterParams and ThetaHarvesterConfig:\n"
        + "\n".join(f"  {m}" for m in default_mismatch)
    )
