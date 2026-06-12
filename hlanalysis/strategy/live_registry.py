"""Live-strategy registry: maps strategy_type → live-engine contract.

This is the single source of truth coupling the three dispatch points that
previously hardcoded ``"late_resolution"`` / ``"theta_harvester"``:

  1. ``config_builders._build_strategy_for_slot`` — builds the live Strategy.
  2. ``config.py``  — validates ``strategy_type`` at YAML load time.
  3. ``backtest/slot_config.py`` — maps strategy_type → backtest registry id.

Adding a new live-capable strategy means calling ``register_live_strategy``
here. The engine, config validator, and backtest slot path all pick it up
automatically — no edits to those three call sites.

Backtest-only strategies (v2_model_edge, v4_binary_statarb, v5_delta_hedged,
v31_pm_nba) are NOT registered here: they have no live builder and are not
valid ``strategy_type`` values in a StrategyConfig.

Import note: the live builder callables below import lazily from
``engine.config_builders`` (deferred to call time) to avoid a circular import
with ``engine/config.py`` which imports this module at the module level.
``engine/config.py`` → ``strategy/live_registry.py`` → (no top-level engine
imports) is safe. The heavy imports inside each builder function execute only
when the engine actually constructs a strategy, long after all modules have
been fully loaded.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, NamedTuple

if TYPE_CHECKING:
    from ..engine.config import StrategyConfig
    from .base import Strategy


class LiveStrategyEntry(NamedTuple):
    """Contract registered per live strategy_type.

    live_builder  — callable(StrategyConfig) → Strategy; used by the engine's
                    _build_strategy_for_slot.
    registry_id   — backtest registry id used by slot_config.backtest_params_from_slot
                    to look up the matching sim builder (e.g. "v1_late_resolution").
    """
    live_builder: Callable[["StrategyConfig"], "Strategy"]
    registry_id: str


_LIVE_REGISTRY: dict[str, LiveStrategyEntry] = {}


def register_live_strategy(
    strategy_type: str,
    *,
    live_builder: Callable[["StrategyConfig"], "Strategy"],
    registry_id: str,
) -> None:
    """Register a strategy_type for live trading.

    Raises ValueError if the type is already registered (guards against
    accidental double-registration during import).
    """
    if strategy_type in _LIVE_REGISTRY:
        raise ValueError(
            f"strategy_type {strategy_type!r} is already registered in the live registry"
        )
    _LIVE_REGISTRY[strategy_type] = LiveStrategyEntry(
        live_builder=live_builder,
        registry_id=registry_id,
    )


def live_strategy_types() -> list[str]:
    """Return all registered live strategy_types (sorted for stability)."""
    return sorted(_LIVE_REGISTRY)


def build_live_strategy(strategy_type: str, cfg: "StrategyConfig") -> "Strategy":
    """Build a live Strategy for the given strategy_type and StrategyConfig.

    Raises ValueError for unknown types (mirrors the old if/elif raise).
    """
    entry = _LIVE_REGISTRY.get(strategy_type)
    if entry is None:
        raise ValueError(
            f"unknown strategy_type: {strategy_type!r} "
            f"(registered: {live_strategy_types()})"
        )
    return entry.live_builder(cfg)


def live_registry_id(strategy_type: str) -> str:
    """Return the backtest registry_id for a live strategy_type.

    Raises KeyError for unknown types (mirrors the old dict lookup).
    """
    try:
        return _LIVE_REGISTRY[strategy_type].registry_id
    except KeyError:
        raise KeyError(
            f"strategy_type {strategy_type!r} has no live registry entry "
            f"(supported: {live_strategy_types()})"
        ) from None


def _reset_for_tests() -> None:
    """Test helper: clear all live-strategy registrations."""
    _LIVE_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Built-in live-strategy registrations.
#
# Builder callables import lazily from engine.config_builders to avoid a
# module-level circular import (config.py → live_registry.py is already in
# the chain). The real work happens only at engine construction time, well
# after all modules are fully loaded.
# ---------------------------------------------------------------------------

def _late_resolution_builder(cfg: "StrategyConfig") -> "Strategy":
    from ..engine.config_builders import (  # noqa: PLC0415
        build_late_resolution_config,
        build_late_resolution_configs_by_class,
    )
    from .late_resolution import LateResolutionStrategy  # noqa: PLC0415
    return LateResolutionStrategy(
        build_late_resolution_config(cfg),
        cfg_by_class=build_late_resolution_configs_by_class(cfg),
    )


def _theta_harvester_builder(cfg: "StrategyConfig") -> "Strategy":
    from ..engine.config_builders import (  # noqa: PLC0415
        build_theta_harvester_config,
        build_theta_harvester_configs_by_class,
    )
    from .theta_harvester import ThetaHarvesterStrategy  # noqa: PLC0415
    return ThetaHarvesterStrategy(
        build_theta_harvester_config(cfg),
        cfg_by_class=build_theta_harvester_configs_by_class(cfg),
    )


register_live_strategy(
    "late_resolution",
    live_builder=_late_resolution_builder,
    registry_id="v1_late_resolution",
)
register_live_strategy(
    "theta_harvester",
    live_builder=_theta_harvester_builder,
    registry_id="v3_theta_harvester",
)


__all__ = [
    "LiveStrategyEntry",
    "register_live_strategy",
    "live_strategy_types",
    "build_live_strategy",
    "live_registry_id",
]
