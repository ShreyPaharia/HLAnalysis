from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")

# Allowed strategy types. Keep these stable — they map to runtime construction in
# runtime.py:_build_strategy_for_slot.
StrategyType = Literal["late_resolution", "theta_harvester"]


def _substitute_env(raw: Any) -> Any:
    if isinstance(raw, dict):
        return {k: _substitute_env(v) for k, v in raw.items()}
    if isinstance(raw, list):
        return [_substitute_env(v) for v in raw]
    if isinstance(raw, str):
        def repl(m: re.Match[str]) -> str:
            key = m.group(1)
            val = os.environ.get(key)
            if val is None:
                raise ValueError(f"missing env var: {key}")
            return val
        return _ENV_RE.sub(repl, raw)
    return raw


class AllowlistEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    match: dict[str, str | list[str]]
    max_position_usd: float
    # Nullable: when None, the position carries a sentinel stop_loss_price that
    # the risk gate never triggers — disabling per-trade stops at the engine
    # level. Calibration on 1y PM corpus showed every tested non-null value
    # (10/30/50/70) underperformed null because the stop fires on transient
    # noise and exits positions that would have ridden to settlement.
    stop_loss_pct: float | None
    tte_min_seconds: int
    tte_max_seconds: int
    price_extreme_threshold: float
    distance_from_strike_usd_min: float
    vol_max: float
    # Optional safety-gate params; defaults preserve pre-gate behavior so older
    # YAMLs continue to load. Calibration in v1-safety-best (May 2026) showed
    # these together lift full-year PnL on PM BTC daily Up/Down from $5 → $560.
    price_extreme_max: float = 1.0
    min_safety_d: float = 0.0
    vol_lookback_seconds: int = 1800
    # Mid-hold safety_d exit threshold (0 = disabled) and EWMA decay for σ
    # (0 = legacy sample stdev). Layered on top of the entry gates above.
    exit_safety_d: float = 0.0
    vol_ewma_lambda: float = 0.0
    # Targeted size cap (Plan: v1-buckets-and-sizing). Defaults preserve pre-cap
    # behavior (pct=0 disables). See LateResolutionConfig docstring.
    size_cap_near_strike_pct: float = 0.0
    size_cap_max_dist_pct: float = 1.5
    size_cap_min_ask: float = 0.88


class GlobalRiskConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_total_inventory_usd: float
    max_concurrent_positions: int
    daily_loss_cap_usd: float
    max_strike_distance_pct: float
    min_recent_volume_usd: float
    stale_data_halt_seconds: int
    reconcile_interval_seconds: int


class ThetaParams(BaseModel):
    """v3.1 theta_harvester knobs. Only consumed when strategy_type='theta_harvester'.

    Mirrors ThetaHarvesterConfig fields the engine needs to seed the strategy.
    Unsupplied fields fall back to the strategy's own defaults via getattr.
    """
    model_config = ConfigDict(frozen=True)
    vol_lookback_seconds: int = 3600
    vol_sampling_dt_seconds: int = 60
    vol_clip_min: float = 0.05
    vol_clip_max: float = 5.0
    edge_buffer: float = 0.0
    fee_taker: float = 0.00035
    half_spread_assumption: float = 0.005
    drift_lookback_seconds: int = 3600
    drift_blend: float = 0.0
    favorite_threshold: float = 0.5
    exit_edge_threshold: float = 0.0
    take_profit_price: float | None = None
    time_stop_seconds: int = 0
    edge_max: float | None = None


class StrategyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    name: str
    paper_mode: bool
    allowlist: list[AllowlistEntry]
    blocklist_question_idxs: list[int] = Field(default_factory=list)
    defaults: AllowlistEntry
    # `global` is a Python keyword; alias to keep YAML ergonomics.
    global_: GlobalRiskConfig = Field(alias="global")
    # --- multi-account / multi-strategy ---
    # Logical name keyed into deploy_cfg.hl_accounts. Default "default" preserves
    # legacy single-account YAMLs that omit this field.
    account_alias: str = "default"
    # Which Strategy subclass to instantiate. Defaults to the v1 live strategy.
    strategy_type: StrategyType = "late_resolution"
    # Optional theta_harvester params (ignored when strategy_type != theta_harvester).
    theta: ThetaParams | None = None


class StrategiesConfig(BaseModel):
    """Container for one or more StrategyConfig entries. The engine accepts a
    flat list and runs each entry against its own (HLClient, RiskGate, Router,
    Reconciler, StateDAL) — orders, fills, risk are isolated per (strategy,
    account)."""
    model_config = ConfigDict(frozen=True)
    strategies: list[StrategyConfig]


class HLConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    account_address: str
    api_secret_key: str
    base_url: str


class TelegramConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    bot_token: str
    chat_id: str


class AlertsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    telegram: TelegramConfig


class DeployConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    env: str
    # Multi-account: alias -> HLConfig. Legacy 'hl:' single-block YAMLs are
    # auto-wrapped into {"default": HLConfig(...)} by load_deploy_config.
    hl_accounts: dict[str, HLConfig]
    alerts: AlertsConfig
    state_db_path: str
    kill_switch_path: str

    # Convenience back-compat: callers that still reach for `.hl` get the
    # single account when there's only one, or the "default" entry. New code
    # should use hl_accounts[alias] directly.
    @property
    def hl(self) -> HLConfig:
        if "default" in self.hl_accounts:
            return self.hl_accounts["default"]
        if len(self.hl_accounts) == 1:
            return next(iter(self.hl_accounts.values()))
        raise KeyError(
            "DeployConfig.hl is ambiguous with multiple hl_accounts; "
            "use deploy_cfg.hl_accounts[alias] instead."
        )

    def state_db_path_for(self, alias: str) -> str:
        """Per-account state DB path. Multiple accounts → namespaced subdir;
        single 'default' account → legacy flat path so existing deployments
        don't see their state.db move."""
        base = Path(self.state_db_path)
        if len(self.hl_accounts) <= 1 and alias == "default":
            return str(base)
        return str(base.parent / alias / base.name)

    def kill_switch_path_for(self, alias: str) -> str:
        """Per-account kill switch (so an operator can halt one strategy
        without killing both). Same namespacing rule as state_db_path_for."""
        base = Path(self.kill_switch_path)
        if len(self.hl_accounts) <= 1 and alias == "default":
            return str(base)
        return str(base.parent / alias / base.name)


def load_strategy_config(path: Path) -> StrategyConfig:
    """Legacy single-strategy loader. Reads top-level `strategy:` block.

    Kept for back-compat with replay tools and older tests. Engine entrypoint
    uses `load_strategies_config` instead so it can run N strategies in one
    process.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if "strategies" in raw:
        # Pick the first entry — replay CLI is single-strategy only.
        return StrategyConfig(**raw["strategies"][0])
    return StrategyConfig(**raw["strategy"])


def load_strategies_config(path: Path) -> StrategiesConfig:
    """Multi-strategy loader.

    Two accepted shapes:
      strategies:                      # new: list of (strategy, account) pairs
        - name: late_resolution
          account_alias: v1
          ...
        - name: theta_harvester
          account_alias: v31
          strategy_type: theta_harvester
          ...
      strategy: { ... }                # legacy: single block (alias defaults to 'default')
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if "strategies" in raw:
        entries = [StrategyConfig(**e) for e in raw["strategies"]]
    elif "strategy" in raw:
        entries = [StrategyConfig(**raw["strategy"])]
    else:
        raise ValueError(f"{path}: missing 'strategies:' list or 'strategy:' block")
    aliases = [e.account_alias for e in entries]
    if len(set(aliases)) != len(aliases):
        raise ValueError(
            f"{path}: duplicate account_alias values {aliases}; each "
            "(strategy, account) pair must use a distinct alias",
        )
    return StrategiesConfig(strategies=entries)


def load_deploy_config(path: Path) -> DeployConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = _substitute_env(raw)
    deploy = raw["deploy"]
    # Accept both new `hl_accounts: {alias: {...}}` and legacy single `hl: {...}`.
    if "hl_accounts" not in deploy:
        if "hl" not in deploy:
            raise ValueError(f"{path}: deploy must define 'hl_accounts' or 'hl'")
        deploy = dict(deploy)
        deploy["hl_accounts"] = {"default": deploy.pop("hl")}
    return DeployConfig(**deploy)


def _entry_matches(match: dict[str, str | list[str]], fields: dict[str, str]) -> bool:
    for key, expected in match.items():
        actual = fields.get(key)
        if actual is None:
            return False
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def match_question(
    cfg: StrategyConfig, *, question_idx: int, fields: dict[str, str]
) -> AllowlistEntry | None:
    """Return the first allowlist entry matching `fields`, or None.

    Blocklist takes precedence: if `question_idx` is in `blocklist_question_idxs`,
    return None even if a pattern would match.
    """
    if question_idx in cfg.blocklist_question_idxs:
        return None
    for entry in cfg.allowlist:
        if _entry_matches(entry.match, fields):
            return entry
    return None
