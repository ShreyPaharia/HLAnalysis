from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


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
    stop_loss_pct: float
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


class GlobalRiskConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_total_inventory_usd: float
    max_concurrent_positions: int
    daily_loss_cap_usd: float
    max_strike_distance_pct: float
    min_recent_volume_usd: float
    stale_data_halt_seconds: int
    reconcile_interval_seconds: int


class StrategyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    name: str
    paper_mode: bool
    allowlist: list[AllowlistEntry]
    blocklist_question_idxs: list[int] = Field(default_factory=list)
    defaults: AllowlistEntry
    # `global` is a Python keyword; alias to keep YAML ergonomics.
    global_: GlobalRiskConfig = Field(alias="global")


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
    hl: HLConfig
    alerts: AlertsConfig
    state_db_path: str
    kill_switch_path: str


def load_strategy_config(path: Path) -> StrategyConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return StrategyConfig(**raw["strategy"])


def load_deploy_config(path: Path) -> DeployConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = _substitute_env(raw)
    return DeployConfig(**raw["deploy"])


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
