from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..strategy.late_resolution import LateResolutionParams
from ..strategy.theta_harvester import ThetaHarvesterParams
from ..strategy.live_registry import live_strategy_types

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")

# ``StrategyType`` is now a plain str — validated dynamically against the
# live registry in StrategyConfig._check_strategy_type below. This removes
# the frozen Literal so a new live strategy is added by registering it in
# ``hlanalysis/strategy/live_registry.py`` with no edit here.
StrategyType = str


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


class AllowlistEntry(LateResolutionParams):
    # Inherits all optional late_resolution knobs from LateResolutionParams —
    # the single source of truth for those defaults. Adding a new late_resolution
    # knob to LateResolutionParams automatically makes it settable here without
    # any edit to this class.
    #
    # extra='forbid' makes a typo'd or unsupported knob fail loudly at load
    # instead of being silently dropped → train/serve skew (the SHR-65 pattern,
    # extended to v1). `tests/unit/test_late_resolution_config_parity.py` pins
    # LateResolutionParams to LateResolutionConfig so every strategy knob stays
    # settable and both sources agree on the canonical defaults.
    model_config = ConfigDict(frozen=True, extra="forbid")
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
    # Veto entries when |reference_price − question.strike| / reference_price
    # is below this fraction. Only meaningful on priceBinary questions.
    # PM corpus tuning showed v3.1 entries below 0.20% lose -$7.68/entry
    # across 57 entries; v1's existing min_safety_d filters most of this
    # zone so the gate is OFF by default for v1. None disables.
    min_distance_pct: float | None = None
    # Post-exit cooldown in seconds. Router refuses to enter on a question
    # for this long after a position close. 0 disables (legacy behavior).
    entry_cooldown_seconds: int = 0


class GlobalRiskConfig(BaseModel):
    # extra='forbid': a typo'd key (e.g. 'daily_loss_cap_uusd') raises
    # ValidationError at load time instead of being silently dropped — the
    # SHR-65 pattern extended to global risk params. These are leaf config
    # models (never used as open override mixins), so strict rejection is safe.
    model_config = ConfigDict(frozen=True, extra="forbid")
    max_total_inventory_usd: float = Field(ge=0)
    max_concurrent_positions: int = Field(ge=1)
    daily_loss_cap_usd: float = Field(ge=0)
    max_strike_distance_pct: float = Field(ge=0)
    min_recent_volume_usd: float = Field(ge=0)
    stale_data_halt_seconds: int = Field(ge=1)
    reconcile_interval_seconds: int = Field(ge=1)
    # Maximum tolerated realized-fill slippage as a fraction of intent
    # limit_price (depth-walk gate). 0 disables; PM ships ~0.005
    # (~0.5¢ at a 0.95-favorite leg). HL slots keep this at 0 because the
    # BboEvent path doesn't populate BookState.ask_levels and the gate is a
    # no-op without ladder data.
    max_slippage_pct: float = Field(default=0.0, ge=0)
    # After the depth-walk gate clamps an intent down to available at-limit
    # liquidity, reject if the resulting notional falls below this floor.
    # Prevents the engine from spamming the venue with micro-orders when only
    # 1–2 contracts are quoted at the inside ask. 0 disables. PM: ~$11 (one
    # 5-share clip of a 0.95-favorite); HL: 0 (legacy behaviour unchanged).
    min_order_notional_usd: float = Field(default=0.0, ge=0)
    # Hour-of-day in UTC when the "daily" PnL accounting window resets.
    # Default 0 = UTC midnight (legacy). Set to 6 to align with HL HIP-4
    # binary settlement (markets resolve at 06:00 UTC = 11:30 IST). Aligning
    # the window with settlement means losses from a market that just
    # settled don't carry into the next market cycle's cap.
    daily_window_start_hour_utc: int = Field(default=0, ge=0, le=23)
    # Event-driven loop cadence (P1). The scan + stop-loss loops wake on a
    # market-data signal but are bounded:
    #   * scan_min_interval_seconds — minimum gap between scans (coalesces a
    #     burst of ticks into one scan; protects DB/CPU). 1.0 = legacy 1 Hz.
    #   * scan_max_interval_seconds — maximum gap (idle floor; a scan still
    #     runs this often even with no ticks, preserving time-based logic like
    #     TTE windows). 1.0 = legacy 1 Hz.
    # Defaults keep today's exact 1 Hz behaviour; set min < max to go
    # event-driven (e.g. 0.05 / 1.0 → ≤50 ms reaction, ≤20 scans/s).
    scan_min_interval_seconds: float = Field(default=1.0, ge=0)
    scan_max_interval_seconds: float = Field(default=1.0, ge=0)
    # When True, stop-loss enforcement runs in its own event-driven loop (woken
    # by the same market signal, same min/max bounds) instead of the 1 Hz
    # continuous-checks loop — so a stop breach is acted on within
    # scan_min_interval_seconds rather than up to 1 s later. Default False keeps
    # stop-loss on the legacy continuous-checks cadence.
    stop_loss_loop_enabled: bool = False


class ThetaParams(ThetaHarvesterParams):
    """v3.1 theta_harvester knobs. Only consumed when strategy_type='theta_harvester'.

    Inherits all optional theta_harvester knobs from ThetaHarvesterParams —
    the single source of truth for those defaults. Adding a new optional theta
    knob to ThetaHarvesterParams automatically makes it settable in the live
    YAML ``theta:`` block with no edit to this class.

    Declares the 13 non-optional-but-defaulted-in-YAML fields directly here:
    these fields are required in ThetaHarvesterConfig (no dataclass default) but
    always populated from YAML with a sensible default. They do NOT belong in
    ThetaHarvesterParams because they are required knobs, not optional extras.

    The four allowlist-sourced fields (max_position_usd, tte_min_seconds,
    tte_max_seconds, stop_loss_pct) are excluded from both; the live builder
    reads them from the ``defaults:`` block.

    `extra='forbid'` makes a typo'd knob fail loudly at load instead of being
    dropped (SHR-65). `tests/unit/test_theta_config_parity.py` guards the
    mirror; `test_single_source_property` guards the inheritance structure.
    """
    # extra='forbid' stays on ThetaParams (not on ThetaHarvesterParams) so the
    # base model itself stays permissive — ThetaHarvesterParams is also used as a
    # mixin base for theta_overrides entries, which share the same extra='forbid'
    # via ThetaParams. The base's frozen=True is sufficient for its own use.
    model_config = ConfigDict(frozen=True, extra="forbid")
    # === required-in-dataclass fields (no ThetaHarvesterConfig default) ===
    # These always have a value in the YAML `theta:` block; defaults here are
    # the canonical live baseline values. Not in ThetaHarvesterParams because
    # they are required core knobs, not optional feature gates.
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
    # Validated dynamically against the live registry (live_registry.py) so that
    # adding a new live strategy requires no edit here.
    strategy_type: StrategyType = "late_resolution"

    @field_validator("strategy_type")
    @classmethod
    def _check_strategy_type(cls, v: str) -> str:
        valid = live_strategy_types()
        if v not in valid:
            raise ValueError(
                f"unknown strategy_type {v!r}; registered live types: {valid}"
            )
        return v
    # Optional theta_harvester params (ignored when strategy_type != theta_harvester).
    theta: ThetaParams | None = None
    # Optional per-question.klass theta overrides. Maps a question class (e.g.
    # "priceBucket") to a PARTIAL ThetaParams: only the fields the operator
    # explicitly sets override the instance `theta:` defaults (resolution order
    # per-class override > instance theta). Reuses ThetaParams so the same
    # extra='forbid' strictness applies — a typo'd knob fails loud (SHR-65). The
    # independent HL bucket tune (v31_bucket_independent_tune_2026_06_05) found
    # buckets want the OPPOSITE of binary on favorite_threshold /
    # vol_lookback_seconds / exit_safety_d / edge_buffer; this is the seam that
    # lets a class diverge within one strategy instance. No block → bit-identical
    # to today. vol_sampling_dt_seconds may also be overridden per class: the
    # engine buckets the shared reference feed at each class's cadence
    # independently (see MarketState (symbol, dt) bucketing + Scanner
    # cadence_by_class).
    theta_overrides: dict[str, ThetaParams] | None = None
    # Symbol whose MarkEvent feeds σ + p_model. Default "BTC" = HL perp mark
    # (HL adapter emits MarkEvent with symbol="BTC"). For PM markets that
    # resolve on Binance SPOT, set "BTCUSDT_SPOT" so the Scanner reads the
    # Binance SPOT bbo reference (remapped from "BTCUSDT" on ingest).
    reference_symbol: str = "BTC"
    # Which feed sources the σ/OHLC reference for ``reference_symbol``.
    #   "mark" (default) — the venue MarkEvent. HL perp mark is sub-second; the
    #     Binance perp mark is a 3s REST poll, which is too sparse for dt=5
    #     bars (High≈Low≈Close → degenerate σ).
    #   "bbo" — the dense BBO mid = (bid_px+ask_px)/2. For PM (BTCUSDT_SPOT)
    #     this is the sub-second bookTicker stream the engine subscribes to, so
    #     dt=5 bars carry real intra-bar range. Mirrors the backtest
    #     `_load_binance_bbo_reference`. Two slots sharing a reference_symbol
    #     must agree on the source (shared OHLC history). Keep "mark" for HL.
    reference_sigma_source: Literal["mark", "bbo"] = "mark"


class StrategiesConfig(BaseModel):
    """Container for one or more StrategyConfig entries. The engine accepts a
    flat list and runs each entry against its own (HLClient, RiskGate, Router,
    Reconciler, StateDAL) — orders, fills, risk are isolated per (strategy,
    account)."""
    model_config = ConfigDict(frozen=True)
    strategies: list[StrategyConfig]


def strategy_config_sig(strategy_cfg: StrategyConfig) -> str:
    """Return a stable 16-hex SHA-256 fingerprint of a StrategyConfig's effective params.

    Covers the full resolved params: strategy_type, paper_mode, reference_symbol,
    reference_sigma_source, defaults (all fields), allowlist (all entries),
    allowlist_count, global_ (all fields), and theta (all fields, when present).

    Stable across Python sessions: uses sorted-keys JSON with str default.

    **Location rationale:** placed here in ``hlanalysis/engine/config.py`` (where
    ``StrategyConfig`` is defined) so both sides — the engine's diag.py and the
    backtest's report.py — can import it from the single authoritative source
    without creating an import cycle.  The backtest already imports StrategyConfig
    from this module via slot_config.py, so adding the sig function here is the
    least-surprise location and avoids a new intermediary module.
    """
    params: dict[str, Any] = {
        "strategy_type": strategy_cfg.strategy_type,
        "paper_mode": strategy_cfg.paper_mode,
        "reference_symbol": strategy_cfg.reference_symbol,
        "reference_sigma_source": strategy_cfg.reference_sigma_source,
        "defaults": strategy_cfg.defaults.model_dump(),
        "allowlist_count": len(strategy_cfg.allowlist),
        "allowlist": [e.model_dump() for e in strategy_cfg.allowlist],
        "global_risk": strategy_cfg.global_.model_dump(),
    }
    if strategy_cfg.theta is not None:
        params["theta"] = strategy_cfg.theta.model_dump()
    stable_json = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(stable_json.encode()).hexdigest()[:16]


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
    # Retention bounds for the events table (Component 2 — engine observability).
    # Age is the primary prune (rows older than N days are dropped); max_rows is
    # a burst backstop so a reject storm can't grow the table between age-prunes.
    # Both default to conservative values safe for the 1 GiB EC2 box.
    events_retention_days: int = 14
    events_retention_max_rows: int = 1_000_000

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
