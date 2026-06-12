"""Convert a live ``StrategyConfig`` slot into backtest decision params.

SHR-99: the backtest must run from the SAME decision config the live engine
builds — not a hand-reconstructed params JSON (reconstructed-config drift caused
an $80 sim-vs-live swing). This module is the single converter: it REUSES the
engine's config builders (``build_theta_harvester_config* /
build_late_resolution_config*`` in ``hlanalysis.engine.config_builders``) to
perform the full allowlist/defaults/theta/theta_overrides merge, then emits the
flat ``(strategy_id, params)`` the backtest registry consumes.

Because every builder reads each config field by its dataclass name and the
emitted params are exactly ``dataclasses.asdict(config)``, building the registry
strategy from these params reproduces the live config byte-for-byte (guarded by
``tests/unit/backtest/test_slot_config_parity.py``). This pins ``vol_estimator``
(and every other knob) end-to-end: the slot's resolved estimator is emitted
explicitly, so the sim can never silently flip it (e.g. to bipower) regardless of
registry-builder defaults.
"""
from __future__ import annotations

from dataclasses import asdict

from ..engine.config import StrategyConfig
from ..engine.config_builders import (
    build_late_resolution_config,
    build_late_resolution_configs_by_class,
    build_theta_harvester_config,
    build_theta_harvester_configs_by_class,
)
from ..strategy.live_registry import live_registry_id


def backtest_params_from_slot(
    cfg: StrategyConfig, *, klass: str | None = None,
) -> tuple[str, dict]:
    """Return ``(registry_strategy_id, params)`` for a live slot.

    ``klass`` selects the per-class override config (mirroring the engine's
    per-``question.klass`` resolution); ``None`` — or a class with no override —
    uses the slot default, exactly as the live strategy falls through to its
    default at evaluation time.
    """
    try:
        strategy_id = live_registry_id(cfg.strategy_type)
    except KeyError:
        from ..strategy.live_registry import live_strategy_types  # noqa: PLC0415
        raise ValueError(
            f"slot {cfg.account_alias!r}: unsupported strategy_type "
            f"{cfg.strategy_type!r} for backtest (supported: "
            f"{live_strategy_types()})"
        ) from None

    if cfg.strategy_type == "theta_harvester":
        by_class = build_theta_harvester_configs_by_class(cfg)
        default = build_theta_harvester_config(cfg)
    else:
        by_class = build_late_resolution_configs_by_class(cfg)
        default = build_late_resolution_config(cfg)

    chosen = by_class.get(klass, default) if klass is not None else default
    return strategy_id, asdict(chosen)


__all__ = ["backtest_params_from_slot"]
