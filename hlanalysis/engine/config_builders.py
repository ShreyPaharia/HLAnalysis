"""Pure config-building helpers: translate a loaded ``StrategyConfig`` (YAML)
into the strategy-specific dataclass configs the live engine and the replay CLI
both consume.

Extracted from runtime.py so the build_* functions and the reference_* cadence
helpers live in one focused module, imported by EngineRuntime (live),
``replay.py`` (backtest replay), and the config-parity tests. No engine
orchestration state here — every function is pure (cfg in, config out).
"""
from __future__ import annotations

from dataclasses import (
    fields as dataclass_fields_of,
    replace as dataclass_replace,
)

from .config import StrategyConfig
from ..strategy.base import Strategy
from ..strategy.late_resolution import (
    LateResolutionConfig, LateResolutionStrategy,
)
from ..strategy.theta_harvester import (
    ThetaHarvesterConfig, ThetaHarvesterStrategy,
)


# LateResolutionConfig fields sourced from the strategy GLOBAL block, not the
# allowlist entry. Everything else on the dataclass that also exists on
# AllowlistEntry is forwarded by reflection (see below); stop_loss_pct is
# special-cased (None -> disabled sentinel).
_LR_GLOBAL_SOURCED = {
    "max_strike_distance_pct",
    "min_recent_volume_usd",
    "stale_data_halt_seconds",
}


def _late_resolution_config_from_entry(
    entry, *, global_,
) -> LateResolutionConfig:
    """Build a LateResolutionConfig from a single AllowlistEntry plus the
    strategy's global block.

    Reflection-based forwarding (the SHR-65 pattern, mirroring
    ``build_theta_harvester_config``): every LateResolutionConfig field that
    also exists on AllowlistEntry is forwarded straight through — no
    hand-maintained subset, so a tuned knob can never be silently dropped (the
    old getattr-subset dropped drift_aware_d / exit_bid_floor / exit_safety_d_5m
    / exit_vol_lookback_5m_seconds / size_scaling / size_min_fraction /
    vol_scaled_tte_*, diverging live from the backtest builder
    build_v1_late_resolution). The GLOBAL-sourced fields come from ``global_``;
    stop_loss_pct maps None -> the ≥1e8 "disabled" sentinel the strategy expects
    (matches build_v1_late_resolution). Defaults on AllowlistEntry mirror the
    dataclass, so an entry that sets none of the optional knobs reproduces
    today's effective live behavior exactly.
    ``tests/unit/test_late_resolution_config_parity.py`` guards the mirror.
    """
    dataclass_field_names = {f.name for f in dataclass_fields_of(LateResolutionConfig)}
    entry_field_names = set(type(entry).model_fields)
    forwarded = {
        name: getattr(entry, name)
        for name in dataclass_field_names & entry_field_names
        if name not in _LR_GLOBAL_SOURCED and name != "stop_loss_pct"
    }
    return LateResolutionConfig(
        max_strike_distance_pct=global_.max_strike_distance_pct,
        min_recent_volume_usd=global_.min_recent_volume_usd,
        stale_data_halt_seconds=global_.stale_data_halt_seconds,
        # LateResolutionConfig.stop_loss_pct is a non-Optional float; the
        # strategy treats values ≥1e8 as "disabled". Map None -> sentinel here.
        stop_loss_pct=1e9 if entry.stop_loss_pct is None else entry.stop_loss_pct,
        **forwarded,
    )


def build_late_resolution_config(cfg: StrategyConfig) -> LateResolutionConfig:
    """Build the default LateResolutionConfig from a loaded StrategyConfig.

    Shared by EngineRuntime (live) and replay CLI. Returns the config sourced
    from `cfg.defaults`; per-class overrides land via
    `build_late_resolution_configs_by_class`.
    """
    return _late_resolution_config_from_entry(cfg.defaults, global_=cfg.global_)


def build_late_resolution_configs_by_class(
    cfg: StrategyConfig,
) -> dict[str, LateResolutionConfig]:
    """Build per-question.klass LateResolutionConfig overrides from the
    strategy's allowlist. Each entry whose `match.class` is set produces one
    config; entries without a class match fall through to defaults at
    evaluation time. Multiple entries with the same class: last one wins.

    Plumbed into LateResolutionStrategy so allowlist match-specific gate
    fields (e.g. priceBucket `tte_max_seconds: 86400`) actually take effect
    at the strategy gate, not only at the risk-gate caps.
    """
    by_class: dict[str, LateResolutionConfig] = {}
    for entry in cfg.allowlist:
        klass = entry.match.get("class")
        if not klass:
            continue
        by_class[klass] = _late_resolution_config_from_entry(
            entry, global_=cfg.global_,
        )
    return by_class


def build_theta_harvester_config(cfg: StrategyConfig) -> ThetaHarvesterConfig:
    """Construct ThetaHarvesterConfig from the YAML `theta:` block.

    Falls back to allowlist-defaults for fields the theta block omits so the
    strategy always sees a fully-populated config.
    """
    d = cfg.defaults
    t = cfg.theta
    if t is None:
        raise ValueError(
            f"strategy '{cfg.name}' (alias={cfg.account_alias}) is "
            "strategy_type=theta_harvester but no `theta:` block was supplied",
        )
    # Forward EVERY field the `theta:` block declares straight through to the
    # dataclass — no hand-maintained subset, so a new tuned knob can never be
    # silently dropped (SHR-65). The four fields below come from the allowlist
    # `defaults:` block instead and are not part of the theta block.
    # test_theta_config_parity.py guards that ThetaParams stays a full mirror.
    _ALLOWLIST_SOURCED = {
        "max_position_usd", "tte_min_seconds", "tte_max_seconds", "stop_loss_pct",
    }
    dataclass_fields = {f.name for f in dataclass_fields_of(ThetaHarvesterConfig)}
    forwarded = {
        name: getattr(t, name)
        for name in dataclass_fields & set(type(t).model_fields)
        if name not in _ALLOWLIST_SOURCED
    }
    return ThetaHarvesterConfig(
        max_position_usd=d.max_position_usd,
        tte_min_seconds=d.tte_min_seconds,
        tte_max_seconds=d.tte_max_seconds,
        stop_loss_pct=d.stop_loss_pct,
        **forwarded,
    )


def build_theta_harvester_configs_by_class(
    cfg: StrategyConfig,
) -> dict[str, ThetaHarvesterConfig]:
    """Build per-question.klass ThetaHarvesterConfig overrides from the strategy's
    `theta_overrides:` block. Mirrors build_late_resolution_configs_by_class.

    Each class maps to a PARTIAL ThetaParams; only the fields the operator
    explicitly set (pydantic ``model_fields_set``) override the instance theta
    defaults built by ``build_theta_harvester_config``. Resolution order is
    per-class override > instance theta defaults. When `theta_overrides` is unset
    the map is empty and the strategy runs the single default config for every
    class — bit-identical to today.

    Using ``model_fields_set`` (not a value diff) is load-bearing: e.g.
    exit_safety_d=0.0 is both the dataclass default AND a meaningful bucket
    target, so an explicit 0.0 must win over the instance theta's 1.0.

    Per-class ``vol_sampling_dt_seconds`` IS supported: MarketState buckets the
    shared reference feed at each registered (symbol, dt) cadence independently,
    so a class can run a different σ sampling cadence. The engine registers each
    class's cadence so the dt divergence is realized at the σ-history read.
    """
    overrides = cfg.theta_overrides
    if not overrides:
        return {}
    base = build_theta_harvester_config(cfg)
    by_class: dict[str, ThetaHarvesterConfig] = {}
    for klass, override in overrides.items():
        set_fields = {name: getattr(override, name) for name in override.model_fields_set}
        by_class[klass] = dataclass_replace(base, **set_fields)
    return by_class


def reference_sampling_dt_seconds(cfg: StrategyConfig) -> int:
    """Effective ``vol_sampling_dt_seconds`` for a slot's reference feed.

    Single source of truth coupling MarketState's mark-bucket period to the
    cadence the strategy's σ formula assumes. theta_harvester carries it in the
    `theta:` block; late_resolution carries it on its allowlist/defaults
    (`AllowlistEntry.vol_sampling_dt_seconds`). Default 60 preserves legacy
    1m bucketing for both. This lets v1 + v31 move to dt=5 in lockstep on the
    shared BTC feed (see summeries/v1_cadence_validation_2026_05_30.md).
    """
    if cfg.theta is not None:
        return int(cfg.theta.vol_sampling_dt_seconds)
    return int(cfg.defaults.vol_sampling_dt_seconds)


def reference_vol_lookback_seconds(cfg: StrategyConfig) -> int:
    """Largest σ/drift lookback window the slot's strategy will request, across
    defaults, every allowlist entry, and (for theta) the theta block. Used to
    size MarketState's per-symbol mark history so sub-minute cadences don't
    truncate the σ window. Mirrors Scanner._required_returns_n's inputs."""
    secs = cfg.defaults.vol_lookback_seconds
    for entry in cfg.allowlist:
        secs = max(secs, entry.vol_lookback_seconds)
    if cfg.theta is not None:
        secs = max(
            secs, cfg.theta.vol_lookback_seconds, cfg.theta.drift_lookback_seconds,
        )
    # Per-class theta overrides may request a longer σ/drift window for one
    # class; size MarketState history for the largest across all of them so a
    # bucket-vs-binary divergence isn't truncated. Only explicitly-set fields
    # carry a meaningful value (model_fields_set); unset ones fall back to the
    # already-counted instance theta defaults above.
    for override in (cfg.theta_overrides or {}).values():
        set_fields = override.model_fields_set
        if "vol_lookback_seconds" in set_fields:
            secs = max(secs, override.vol_lookback_seconds)
        if "drift_lookback_seconds" in set_fields:
            secs = max(secs, override.drift_lookback_seconds)
    return secs


def _build_strategy_for_slot(cfg: StrategyConfig) -> Strategy:
    """Dispatch on strategy_type. Add new strategies here as they're surfaced
    for live trading."""
    if cfg.strategy_type == "late_resolution":
        return LateResolutionStrategy(
            build_late_resolution_config(cfg),
            cfg_by_class=build_late_resolution_configs_by_class(cfg),
        )
    if cfg.strategy_type == "theta_harvester":
        return ThetaHarvesterStrategy(
            build_theta_harvester_config(cfg),
            cfg_by_class=build_theta_harvester_configs_by_class(cfg),
        )
    raise ValueError(f"unknown strategy_type: {cfg.strategy_type!r}")
