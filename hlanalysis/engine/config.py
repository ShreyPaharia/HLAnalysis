from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

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
    # When True, late_resolution gates entry on bid_px instead of ask_px.
    # Sizing and IOC limit still use ask. Recommended for HL HIP-4 only —
    # PM corpus has tight enough spreads that this is a no-op there. See
    # LateResolutionConfig.use_bid_for_entry_gate.
    use_bid_for_entry_gate: bool = False
    # Minimum bid notional (bid_px × bid_sz) in USD for the favourite-leg
    # filter to accept an entry. Catches single-share spoof bids that pass
    # a numeric bid_px threshold but represent no real buying interest. 0
    # disables (legacy behavior); set to ~$10–20 for a meaningful filter.
    min_bid_notional_usd: float = 0.0
    # Veto entries when |reference_price − question.strike| / reference_price
    # is below this fraction. Only meaningful on priceBinary questions.
    # PM corpus tuning showed v3.1 entries below 0.20% lose -$7.68/entry
    # across 57 entries; v1's existing min_safety_d filters most of this
    # zone so the gate is OFF by default for v1. None disables.
    min_distance_pct: float | None = None
    # Post-exit cooldown in seconds. Router refuses to enter on a question
    # for this long after a position close. 0 disables (legacy behavior).
    entry_cooldown_seconds: int = 0
    # Strategy-side position topup knobs. When a held position is under-filled
    # (e.g. IOC partial-fill on a thin HL HIP-4 book), the strategy emits a
    # second ENTER intent at the current ask to top up. Exit-eval runs first;
    # exits always win over topup. See strategy modules' _evaluate_topup for
    # the full gate flow.
    topup_enabled: bool = True
    topup_threshold_pct: float = 0.2          # trigger when shortfall ≥ 20% of target
    topup_min_notional_usd: float = 11.0      # HL per-order min is $10; buffer 11


class GlobalRiskConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_total_inventory_usd: float
    max_concurrent_positions: int
    daily_loss_cap_usd: float
    max_strike_distance_pct: float
    min_recent_volume_usd: float
    stale_data_halt_seconds: int
    reconcile_interval_seconds: int
    # Maximum tolerated realized-fill slippage as a fraction of intent
    # limit_price (depth-walk gate). 0 disables; PM ships ~0.005
    # (~0.5¢ at a 0.95-favorite leg). HL slots keep this at 0 because the
    # BboEvent path doesn't populate BookState.ask_levels and the gate is a
    # no-op without ladder data.
    max_slippage_pct: float = 0.0
    # After the depth-walk gate clamps an intent down to available at-limit
    # liquidity, reject if the resulting notional falls below this floor.
    # Prevents the engine from spamming the venue with micro-orders when only
    # 1–2 contracts are quoted at the inside ask. 0 disables. PM: ~$11 (one
    # 5-share clip of a 0.95-favorite); HL: 0 (legacy behaviour unchanged).
    min_order_notional_usd: float = 0.0
    # Hour-of-day in UTC when the "daily" PnL accounting window resets.
    # Default 0 = UTC midnight (legacy). Set to 6 to align with HL HIP-4
    # binary settlement (markets resolve at 06:00 UTC = 11:30 IST). Aligning
    # the window with settlement means losses from a market that just
    # settled don't carry into the next market cycle's cap.
    daily_window_start_hour_utc: int = 0


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
    # See ThetaHarvesterConfig.min_distance_pct — None disables.
    min_distance_pct: float | None = None
    # See ThetaHarvesterConfig.min_bid_notional_usd — 0 disables.
    min_bid_notional_usd: float = 0.0
    # See ThetaHarvesterConfig.gamma_lambda — None disables.
    gamma_lambda: float | None = None
    # See ThetaHarvesterConfig.topup_* — strategy-side topup of partial-fill
    # positions. Defaults match AllowlistEntry; tune via the YAML `theta:` block
    # when v3.1 needs different topup behavior than v1.
    topup_enabled: bool = True
    topup_threshold_pct: float = 0.2
    topup_min_notional_usd: float = 11.0
    # See ThetaHarvesterConfig.exit_take_profit_mode / exit_fee.
    exit_take_profit_mode: bool = False
    exit_fee: float = 0.0007
    # Fee model selector. Plumbed into ThetaHarvesterConfig in Phase 7; the
    # strategy ignores these fields today, but they MUST round-trip cleanly
    # through YAML now so the v31_pm slot config is stable across phases.
    #   "flat"      → existing behaviour: fee = fee_taker (per-leg fixed bps)
    #   "pm_binary" → Polymarket curve: fee = C · fee_rate · p · (1−p)
    # Default "flat" / 0.0 preserves HL v31 bit-identically.
    fee_model: Literal["flat", "pm_binary"] = "flat"
    fee_rate: float = 0.0


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
    # Symbol whose MarkEvent feeds σ + p_model. Default "BTC" = HL perp mark
    # (HL adapter emits MarkEvent with symbol="BTC"). For PM markets that
    # resolve on Binance, set "BTCUSDT" so the Scanner reads Binance perp mark
    # (Binance adapter emits with symbol="BTCUSDT").
    reference_symbol: str = "BTC"


class StrategiesConfig(BaseModel):
    """Container for one or more StrategyConfig entries. The engine accepts a
    flat list and runs each entry against its own (HLClient, RiskGate, Router,
    Reconciler, StateDAL) — orders, fills, risk are isolated per (strategy,
    account)."""
    model_config = ConfigDict(frozen=True)
    strategies: list[StrategyConfig]


class _AccountBase(BaseModel):
    model_config = ConfigDict(frozen=True)
    venue: str  # discriminator


class HyperliquidAccount(_AccountBase):
    venue: Literal["hyperliquid"] = "hyperliquid"
    account_address: str
    api_secret_key: str
    base_url: str


class PolymarketAccount(_AccountBase):
    venue: Literal["polymarket"] = "polymarket"
    clob_host: str = "https://clob.polymarket.com"
    chain_id: int = 137
    private_key: str
    clob_api_key: str
    clob_api_secret: str
    clob_api_passphrase: str
    # Set for polymarket.com-UI accounts (proxy/safe wallet pattern). The EOA
    # signs; this address is the on-chain maker. Leave None for direct-EOA
    # deployments. signature_type defaults to POLY_1271 when funder is set.
    funder_address: str | None = None
    signature_type: str | None = None


AccountConfig = Annotated[
    HyperliquidAccount | PolymarketAccount,
    Field(discriminator="venue"),
]


# Back-compat alias — many tests still import HLConfig by name. Drop in a
# follow-up release once call sites switch.
HLConfig = HyperliquidAccount


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
    accounts: dict[str, AccountConfig]
    alerts: AlertsConfig
    state_db_path: str
    kill_switch_path: str

    def state_db_path_for(self, alias: str) -> str:
        """Per-account state DB path. Multiple accounts → namespaced subdir;
        single 'default' account → legacy flat path so existing deployments
        don't see their state.db move."""
        base = Path(self.state_db_path)
        if len(self.accounts) <= 1 and alias == "default":
            return str(base)
        return str(base.parent / alias / base.name)

    def kill_switch_path_for(self, alias: str) -> str:
        """Per-account kill switch (so an operator can halt one strategy
        without killing both). Same namespacing rule as state_db_path_for."""
        base = Path(self.kill_switch_path)
        if len(self.accounts) <= 1 and alias == "default":
            return str(base)
        return str(base.parent / alias / base.name)


# Back-compat property: monitoring code reads `.hl_accounts` directly.
# Kept as a read-only view; remove after the call sites are migrated.
def _hl_accounts(self: "DeployConfig") -> dict[str, HyperliquidAccount]:
    return {a: c for a, c in self.accounts.items() if isinstance(c, HyperliquidAccount)}
DeployConfig.hl_accounts = property(_hl_accounts)  # type: ignore[assignment]


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
    if "accounts" not in deploy:
        raise ValueError(
            f"{path}: deploy must define `accounts:` (venue-typed). "
            "Legacy `hl_accounts:` / `hl:` are no longer supported; see "
            "DEPLOYMENT.md for migration."
        )
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
