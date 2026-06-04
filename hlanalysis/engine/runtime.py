from __future__ import annotations

import asyncio
import signal
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass, field, fields as dataclass_fields_of
from pathlib import Path
from typing import Awaitable

import aiohttp
from loguru import logger

from ..adapters.base import VenueAdapter
from ..adapters.binance_klines import binance_1m_close_at
from ..config import Subscription
from ..events import (
    NormalizedEvent, ProductType,
    BboEvent, BookSnapshotEvent, BookDeltaEvent, MarkEvent, TradeEvent,
)

# Event classes that move a price/book and should wake the scan + stop-loss
# loops (P1). QuestionMetaEvent / SettlementEvent do not move prices, so they
# don't trigger an immediate re-scan (the idle max-interval floor still covers
# their time-based effects).
_PRICE_EVENT_TYPES = (
    BboEvent, BookSnapshotEvent, BookDeltaEvent, MarkEvent, TradeEvent,
)
from .config import (
    AccountConfig,
    DeployConfig,
    HLConfig,
    HyperliquidAccount,
    PolymarketAccount,
    StrategyConfig,
    match_question,
)
from .event_bus import EventBus
from .exec_client import ExecutionClient
from .exec_types import ClearinghouseState, OpenOrderRow, UserFillRow
from .hl_client import HLClient
from .market_state import MarketState
from .reconcile import Reconciler
from .restart_drift import RestartDriftGate
from .risk import RiskGate
from .risk_events import (
    BusEvent,
    DailyLossHalt, EngineHeartbeat, Exit, FeedDown, FeedRecovered, FeedStale,
    KillSwitchActivated, NewQuestion, OrderUnconfirmed, RedemptionTimeout,
    StaleDataHalt, StopLossTriggered,
)
from .router import Router
from .scanner import Scanner
from .state import CachedStateDAL, StateDAL
from ..alerts.rules import AlertRules
from ..alerts.telegram import TelegramClient
from ..strategy.base import Strategy
from ..strategy.late_resolution import (
    LateResolutionConfig, LateResolutionStrategy,
)
from ..strategy.theta_harvester import (
    ThetaHarvesterConfig, ThetaHarvesterStrategy,
)


# Internal symbol the Binance SPOT reference feed is remapped to (see
# _remap_reference_symbol). PM strategy slots reference this via
# `reference_symbol: BTCUSDT_SPOT` in config/strategy.yaml — that YAML string
# MUST match this constant exactly, or the slot reads an empty book.
_SPOT_REF_SYMBOL = "BTCUSDT_SPOT"


def _remap_reference_symbol(ev: NormalizedEvent) -> NormalizedEvent:
    """Rename Binance SPOT BTCUSDT events to BTCUSDT_SPOT so the PM slots'
    ``reference_symbol: BTCUSDT_SPOT`` resolves to the spot feed (not any
    future perp entry). No-op for every other event."""
    if (
        ev.venue == "binance"
        and ev.product_type == ProductType.SPOT
        and ev.symbol == "BTCUSDT"
    ):
        return ev.model_copy(update={"symbol": _SPOT_REF_SYMBOL})
    return ev


# PM unconfirmed-order watchdog: OrderUnconfirmed fires once a live PM order
# has sat in flight (status=open/pending/partially_filled) past this threshold
# without a status change. 30s gives PM CLOB plenty of room under heavy chain
# load while still surfacing stuck orders before the next scan tick would
# top up on top of stale state.
PM_UNCONFIRMED_THRESHOLD_S: float = 30.0

# PM redemption watchdog: RedemptionTimeout fires this long after the
# settlement Exit if the operator hasn't redeemed yet (we don't watch USDC
# on-chain). 6h is a generous window that catches genuinely-forgotten
# settlements without nagging through the normal redemption flow.
PM_REDEMPTION_TIMEOUT_S: float = 6 * 3600.0


def _pm_check_unconfirmed_orders(
    slot: "AccountSlot", now_ns: int, *,
    threshold_s: float = PM_UNCONFIRMED_THRESHOLD_S,
) -> list[OrderUnconfirmed]:
    """Pure detector: scan slot.dal.live_orders() and return one
    OrderUnconfirmed for each open order older than threshold_s that hasn't
    already alerted. Mutates `slot.pm_alerted_unconfirmed_cloids` to record
    new alerts and to evict cloids no longer live.
    """
    live = slot.dal.live_orders()
    live_cloids: set[str] = {o.cloid for o in live}
    # Garbage-collect alerted set so a re-placed order with the same cloid
    # would re-fire after its next stall. Without this, the set grows
    # unbounded over the process lifetime.
    slot.pm_alerted_unconfirmed_cloids &= live_cloids
    out: list[OrderUnconfirmed] = []
    for o in live:
        if o.status != "open":
            continue
        age_s = (now_ns - o.last_update_ts_ns) / 1e9
        if age_s < threshold_s:
            continue
        if o.cloid in slot.pm_alerted_unconfirmed_cloids:
            continue
        out.append(OrderUnconfirmed(
            ts_ns=now_ns, account_alias=slot.alias,
            cloid=o.cloid, symbol=o.symbol, side=o.side,  # type: ignore[arg-type]
            size=o.size, limit_price=o.price, age_seconds=age_s,
            venue_oid=o.venue_oid or "",
        ))
        slot.pm_alerted_unconfirmed_cloids.add(o.cloid)
    return out


def _pm_check_redemption_timeouts(
    slot: "AccountSlot", now_ns: int, *,
    threshold_s: float = PM_REDEMPTION_TIMEOUT_S,
) -> list[RedemptionTimeout]:
    """Pure detector: walk slot.pm_settlements and return one
    RedemptionTimeout per PM settlement older than threshold_s that hasn't
    alerted. Mutates `slot.pm_alerted_redemption_qidxs`.
    """
    out: list[RedemptionTimeout] = []
    for qidx, (settled_ts_ns, symbol, qty, realized_pnl) in slot.pm_settlements.items():
        if qidx in slot.pm_alerted_redemption_qidxs:
            continue
        age_s = (now_ns - settled_ts_ns) / 1e9
        if age_s < threshold_s:
            continue
        # No on-chain check (see RedemptionTimeout docstring): for a winning
        # leg the operator should see qty USDC arrive; for a loser, zero.
        # Winner heuristic: realized_pnl > 0 (PM binary payouts make this
        # equivalent under positive entry prices, which is always the case).
        expected_payout = qty if realized_pnl > 0 else 0.0
        out.append(RedemptionTimeout(
            ts_ns=now_ns, account_alias=slot.alias,
            question_idx=qidx, symbol=symbol, qty=qty,
            settled_ts_ns=settled_ts_ns, age_seconds=age_s,
            expected_payout_usd=expected_payout,
        ))
        slot.pm_alerted_redemption_qidxs.add(qidx)
    return out


def _late_resolution_config_from_entry(
    entry, *, global_,
) -> LateResolutionConfig:
    """Build a LateResolutionConfig from a single AllowlistEntry plus the
    strategy's global block. Cross-cutting fields (max_strike_distance_pct,
    min_recent_volume_usd, stale_data_halt_seconds) come from `global_`;
    everything else comes from `entry`. `getattr` defaults let older YAMLs
    without the safety-gate fields keep loading.
    """
    return LateResolutionConfig(
        tte_min_seconds=entry.tte_min_seconds, tte_max_seconds=entry.tte_max_seconds,
        price_extreme_threshold=entry.price_extreme_threshold,
        distance_from_strike_usd_min=entry.distance_from_strike_usd_min,
        vol_max=entry.vol_max, max_position_usd=entry.max_position_usd,
        # LateResolutionConfig.stop_loss_pct is a non-Optional float; the
        # strategy treats values ≥1e8 as "disabled" (matches build_v1_late_resolution
        # in strategy/late_resolution.py). Map None -> sentinel here.
        stop_loss_pct=1e9 if entry.stop_loss_pct is None else entry.stop_loss_pct,
        max_strike_distance_pct=global_.max_strike_distance_pct,
        min_recent_volume_usd=global_.min_recent_volume_usd,
        stale_data_halt_seconds=global_.stale_data_halt_seconds,
        price_extreme_max=getattr(entry, "price_extreme_max", 1.0),
        min_safety_d=getattr(entry, "min_safety_d", 0.0),
        vol_lookback_seconds=getattr(entry, "vol_lookback_seconds", 1800),
        exit_safety_d=getattr(entry, "exit_safety_d", 0.0),
        vol_ewma_lambda=getattr(entry, "vol_ewma_lambda", 0.0),
        vol_estimator=getattr(entry, "vol_estimator", "stdev"),
        vol_sampling_dt_seconds=getattr(entry, "vol_sampling_dt_seconds", 60),
        size_cap_near_strike_pct=getattr(entry, "size_cap_near_strike_pct", 0.0),
        size_cap_max_dist_pct=getattr(entry, "size_cap_max_dist_pct", 1.5),
        size_cap_min_ask=getattr(entry, "size_cap_min_ask", 0.88),
        use_bid_for_entry_gate=getattr(entry, "use_bid_for_entry_gate", False),
        min_bid_notional_usd=getattr(entry, "min_bid_notional_usd", 0.0),
        topup_enabled=getattr(entry, "topup_enabled", True),
        topup_threshold_pct=getattr(entry, "topup_threshold_pct", 0.2),
        topup_min_notional_usd=getattr(entry, "topup_min_notional_usd", 11.0),
        fee_model=getattr(entry, "fee_model", "flat"),
        fee_rate=getattr(entry, "fee_rate", 0.0),
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
        return ThetaHarvesterStrategy(build_theta_harvester_config(cfg))
    raise ValueError(f"unknown strategy_type: {cfg.strategy_type!r}")


@dataclass
class AccountSlot:
    """One (strategy, account) pair. Owns its own DAL, HL client, risk gate,
    router, reconciler, and strategy instance. The only thing shared with
    sibling slots is the engine's MarketState + WS feed.
    """
    cfg: StrategyConfig
    account_cfg: "HyperliquidAccount | PolymarketAccount"
    state_db_path: Path
    kill_switch_path: Path
    cloid_prefix: str           # e.g. "hla-v1-" or "hla-v31-"
    dal: StateDAL
    exec_client: ExecutionClient
    risk: RiskGate
    router: Router
    strategy: Strategy
    scanner: Scanner
    # Restart-drift gate result for this slot — if True, the scanner does NOT
    # run for this slot but other slots may still trade.
    blocked: bool = False
    last_reconcile_ns: int = 0
    scans_completed: int = 0
    decisions_emitted: int = 0
    halted: bool = False         # daily-loss / kill-switch latched
    # PM-only alert tracking. Populated by `_continuous_checks_loop` for slots
    # whose `account_cfg` is a PolymarketAccount; HL slots leave these empty.
    # `pm_alerted_unconfirmed_cloids` lists cloids that have already triggered
    # an OrderUnconfirmed alert. Cleared when the cloid drops out of
    # live_orders (status flipped to filled/cancelled/rejected) so a future
    # stall on a re-placed order with the same cloid would re-fire.
    pm_alerted_unconfirmed_cloids: set[str] = field(default_factory=set)
    # `pm_settlements` records (qidx -> (settled_ts_ns, symbol, qty,
    # realized_pnl)) for PM settlement Exits, captured via the bus subscription
    # in `_continuous_checks_loop`. Drives RedemptionTimeout.
    # `pm_alerted_redemption_qidxs` tracks qidxs that have already fired
    # RedemptionTimeout — prevents per-tick spam after the 6h threshold trips.
    pm_settlements: dict[int, tuple[int, str, float, float]] = field(default_factory=dict)
    pm_alerted_redemption_qidxs: set[int] = field(default_factory=set)
    # PM only: set True after the first reconcile pass that authoritatively
    # synced positions from venue truth (the data-api). Until then the live
    # reconcile APPLIES venue changes (to adopt positions missed while the
    # engine was down — the restart gate runs before market data loads, so it
    # can't); afterwards the live reconcile is alert-only and our fill ledger is
    # the source of truth (PM data-api lags/flaps). HL ignores this (always
    # applies).
    pm_startup_position_synced: bool = False
    # Symbols that have already fired a StaleDataHalt alert this stale episode.
    # The continuous-checks loop runs every ~1s; without this a held position
    # whose leg book goes quiet (PM books update in bursts, esp. near
    # resolution) would re-publish the same StaleDataHalt every second. Evicted
    # when the book recovers so a fresh episode re-alerts. Alert-only — the
    # per-trade stale veto in risk/scanner is unaffected.
    stale_alerted_symbols: set[str] = field(default_factory=set)

    @property
    def alias(self) -> str:
        return self.cfg.account_alias


@dataclass
class EngineRuntime:
    # New multi-strategy entrypoint. Pass one StrategyConfig per
    # (strategy, account) pair; the engine builds one AccountSlot for each.
    strategies: list[StrategyConfig]
    deploy_cfg: DeployConfig
    adapter_factory: Callable[[], VenueAdapter]
    subscriptions: list[Subscription]
    # Optional dependency injections so tests can swap real components for fakes.
    # The factory is called once per slot with (alias, account_cfg, paper_mode).
    # account_cfg is the venue-typed AccountConfig (HyperliquidAccount | PolymarketAccount).
    exec_client_factory: Callable[[str, AccountConfig, bool], ExecutionClient] | None = None
    telegram_factory: Callable[[aiohttp.ClientSession], TelegramClient] | None = None
    # Binance SPOT 1m candle close lookup for PM up/down strike capture (covers
    # both observed-open and missed-open — it's the single capture source).
    # Injectable so tests avoid network. Returns the close at the given epoch-ns, or None.
    klines_fetcher: Callable[[int], float | None] = binance_1m_close_at
    market_state: MarketState = field(default_factory=MarketState)
    bus: EventBus = field(default_factory=EventBus)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    # Set by the ingest loop on every price-moving event; awaited by the
    # event-driven scan + stop-loss loops so a market move triggers a re-scan
    # within scan_min_interval_seconds (P1). Idle behaviour is unchanged: the
    # loops also wake on their max-interval floor.
    _market_dirty: asyncio.Event = field(default_factory=asyncio.Event)
    heartbeat_interval_s: float = 30.0
    # Ingest reconnect/backoff (SHR-42). On a feed drop the loop reconnects with
    # exponential backoff between base and max; after this many consecutive
    # failed reconnects it latches all slots halted and stops the engine (a
    # prolonged dead feed means positions are unmanaged — fail safe, let the
    # supervisor restart).
    ingest_reconnect_base_s: float = 1.0
    ingest_reconnect_max_s: float = 30.0
    ingest_halt_after_failures: int = 5
    # Daily-loss venue-read fail-safe (SHR-49). After this many CONSECUTIVE
    # failed venue realized-PnL reads, latch the slot halted rather than keep
    # trading on a venue-blind PnL — a transient outage must not defeat the cap.
    daily_loss_venue_fail_halt: int = 3
    _venue_pnl_failures: dict[str, int] = field(default_factory=dict)
    # Process-wide counter for events ingested (one WS feed, shared).
    events_ingested: int = 0
    # Populated by run() so external observers (heartbeat consumers, tests)
    # can read live slot state without rebuilding clones.
    slots: list[AccountSlot] = field(default_factory=list)
    # Per-question-idx asyncio.Lock so concurrent first-sight + periodic-loop
    # callers can't double-fetch the same PM strike. Created lazily; bounded by
    # the number of PM questions (no eviction needed).
    _pm_strike_locks: dict[int, asyncio.Lock] = field(default_factory=dict)

    # ---------- legacy single-strategy compatibility ----------

    @classmethod
    def from_single(
        cls,
        *,
        strategy_cfg: StrategyConfig,
        deploy_cfg: DeployConfig,
        adapter_factory: Callable[[], VenueAdapter],
        subscriptions: list[Subscription],
        exec_client_factory: Callable[[bool], ExecutionClient] | None = None,
        telegram_factory: Callable[[aiohttp.ClientSession], TelegramClient] | None = None,
    ) -> EngineRuntime:
        """Convenience constructor for tests / single-strategy use.

        Wraps the per-slot ExecutionClient factory so callers that only care
        about paper_mode can keep passing a unary lambda.
        """
        wrapped_factory: Callable[[str, AccountConfig, bool], ExecutionClient] | None = None
        if exec_client_factory is not None:
            def wrapped_factory(_alias: str, _acct: AccountConfig, paper: bool) -> ExecutionClient:
                return exec_client_factory(paper)
        return cls(
            strategies=[strategy_cfg],
            deploy_cfg=deploy_cfg,
            adapter_factory=adapter_factory,
            subscriptions=subscriptions,
            exec_client_factory=wrapped_factory,
            telegram_factory=telegram_factory,
        )

    # ---------- main entrypoint ----------

    async def run(self) -> None:
        if not self.strategies:
            raise ValueError("EngineRuntime requires at least one strategy")
        # 1) Build slots — store on self so observers (tests, heartbeat) can
        # read live state.
        self.slots = [self._build_slot(s_cfg) for s_cfg in self.strategies]
        slots = self.slots
        if len({s.alias for s in slots}) != len(slots):
            raise ValueError(
                "Duplicate account_alias across slots — each (strategy, account) "
                "pair must use a distinct alias",
            )
        # Couple the shared MarketState's per-symbol mark-bucket period to each
        # slot's vol_sampling_dt_seconds before the ingest loop streams marks.
        self._register_reference_cadences(slots)

        async with aiohttp.ClientSession() as http:
            tg = self._make_telegram(http)
            # Short venue tags ride in the alert prefix so HL and PM slots
            # are visually distinct on Telegram (e.g. `[HL:v31]` vs
            # `[PM:v31_pm]`). Tag picked off the venue-typed AccountConfig
            # discriminator so adding a new venue requires no rules.py edit.
            venue_by_alias: dict[str, str] = {}
            for s in slots:
                if isinstance(s.account_cfg, PolymarketAccount):
                    venue_by_alias[s.alias] = "PM"
                elif isinstance(s.account_cfg, HyperliquidAccount):
                    venue_by_alias[s.alias] = "HL"
            rules = AlertRules(
                bus=self.bus, telegram=tg, venue_by_alias=venue_by_alias,
            )

            # 2) Per-slot restart-drift gate
            now_ns0 = self._now_ns()
            for slot in slots:
                slot.last_reconcile_ns = now_ns0
                gate = RestartDriftGate(
                    dal=slot.dal,
                    block_path=slot.kill_switch_path.parent / "restart_blocked",
                    account_alias=slot.alias,
                )
                venue_open, venue_state, all_fills = await self._venue_snapshot(slot)
                drift_res = gate.run(
                    venue_open=venue_open,
                    venue_state=venue_state,
                    fills_lookup=lambda c, _f=all_fills: _f,
                    now_ns=now_ns0,
                )
                for ev in drift_res.drift_events:
                    await self.bus.publish(ev)
                if drift_res.blocked:
                    slot.blocked = True
                    logger.warning(
                        "RESTART BLOCKED alias={} — scanner suspended\n{}",
                        slot.alias, drift_res.summary,
                    )
                    await tg.send(
                        f"*RESTART BLOCKED* (alias={slot.alias})\n"
                        f"```\n{drift_res.summary[:3500]}\n```"
                    )

            # 3) Wire signal handlers
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                with suppress(NotImplementedError):
                    loop.add_signal_handler(sig, self.stop_event.set)

            # 4) Spawn tasks. Market data is shared (one WS); scan / reconcile /
            # continuous-checks loops run per-slot.
            alerts_sub = self.bus.subscribe()
            tasks: list[asyncio.Task] = [
                asyncio.create_task(self._ingest_loop(slots)),
                asyncio.create_task(rules.run(alerts_sub)),
                asyncio.create_task(self._heartbeat_loop(slots)),
                asyncio.create_task(self._pm_strike_capture_loop(slots)),
                asyncio.create_task(self._events_persist_loop()),
            ]
            for slot in slots:
                tasks.append(asyncio.create_task(self._reconcile_loop(slot)))
                tasks.append(asyncio.create_task(self._continuous_checks_loop(slot)))
                if not slot.blocked:
                    tasks.append(asyncio.create_task(self._scan_loop(slot)))
                if slot.cfg.global_.stop_loss_loop_enabled and not slot.blocked:
                    tasks.append(asyncio.create_task(self._stop_loss_loop(slot)))

            await self.stop_event.wait()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def _is_pm_slot(self, slot: "AccountSlot") -> bool:
        """True for Polymarket slots (venue discriminator)."""
        acct = self.deploy_cfg.accounts.get(slot.alias)
        return getattr(acct, "venue", "") == "polymarket" if acct is not None else False

    # ---------- cadence registration ----------

    def _register_reference_cadences(self, slots: list[AccountSlot]) -> None:
        """Register each slot's (reference_symbol → vol_sampling_dt_seconds)
        on the shared MarketState so marks are bucketed at exactly the cadence
        the strategy's σ formula assumes (no train/serve skew). Raises on
        conflicting cadences for the same reference_symbol (see
        MarketState.set_reference_cadence)."""
        for slot in slots:
            self.market_state.set_reference_cadence(
                slot.cfg.reference_symbol,
                sampling_dt_seconds=reference_sampling_dt_seconds(slot.cfg),
                lookback_seconds=reference_vol_lookback_seconds(slot.cfg),
            )
            # Couple the σ/OHLC source (mark | bbo) per reference symbol. Same
            # fail-fast conflict guard as the cadence — slots sharing a symbol
            # must agree. Default "mark" preserves HL behaviour bit-identically.
            self.market_state.set_reference_source(
                slot.cfg.reference_symbol, slot.cfg.reference_sigma_source,
            )

    # ---------- slot construction ----------

    def _build_slot(self, s_cfg: StrategyConfig) -> AccountSlot:
        alias = s_cfg.account_alias
        if alias not in self.deploy_cfg.accounts:
            raise ValueError(
                f"strategy '{s_cfg.name}' references account_alias={alias!r} but "
                f"deploy.accounts has only {list(self.deploy_cfg.accounts)}",
            )
        acct = self.deploy_cfg.accounts[alias]
        state_db_path = Path(self.deploy_cfg.state_db_path_for(alias))
        kill_switch_path = Path(self.deploy_cfg.kill_switch_path_for(alias))
        cloid_prefix = f"hla-{alias}-"

        # Cached DAL: positions/orders are read every loop wake (esp. under the
        # event-driven scan, P1); serve those from memory, write through to the
        # DB. run_migrations FIRST so the lazy cache load sees existing tables.
        dal = CachedStateDAL(state_db_path)
        dal.run_migrations()

        if self.exec_client_factory is not None:
            exec_client = self.exec_client_factory(alias, acct, s_cfg.paper_mode)
        elif isinstance(acct, HyperliquidAccount):
            exec_client = HLClient(
                account_address=acct.account_address,
                api_secret_key=acct.api_secret_key,
                base_url=acct.base_url,
                paper_mode=s_cfg.paper_mode,
            )
        elif isinstance(acct, PolymarketAccount):
            from .pm_client import PMClient
            exec_client = PMClient(
                paper_mode=s_cfg.paper_mode,
                clob_host=acct.clob_host,
                chain_id=acct.chain_id,
                private_key=acct.private_key,
                clob_api_key=acct.clob_api_key,
                clob_api_secret=acct.clob_api_secret,
                clob_api_passphrase=acct.clob_api_passphrase,
                funder_address=acct.funder_address,
                signature_type=acct.signature_type,
            )
        else:
            raise ValueError(f"unsupported account type: {type(acct).__name__}")

        risk = RiskGate(s_cfg)
        router = Router(
            dal=dal, gate=risk, bus=self.bus, exec_client=exec_client,
            strategy_cfg=s_cfg, strategy_id=s_cfg.name,
            cloid_prefix=cloid_prefix,
        )
        strategy = _build_strategy_for_slot(s_cfg)
        # Gate-decision log sibling of state.db. Operators tail this during
        # forward-testing to see which gates are firing without combing
        # through journal heartbeats. State-change-debounced, so file size
        # stays small (one line per question per transition).
        gate_log_path = state_db_path.parent / "gate_decisions.jsonl"
        scanner = Scanner(
            strategy=strategy, cfg=s_cfg,
            market_state=self.market_state, dal=dal,
            kill_switch_path=kill_switch_path,
            reference_symbol=s_cfg.reference_symbol,
            last_reconcile_ns=0,
            # Daily-loss cap reads from HL (venue truth) rather than the local
            # DB. The DB's realized_pnl is structurally near-zero — fills aren't
            # persisted on the happy path and closed positions are deleted —
            # so without this the cap would never fire.
            pnl_provider=exec_client.realized_pnl_since,
            gate_log_path=gate_log_path,
        )
        return AccountSlot(
            cfg=s_cfg, account_cfg=acct,
            state_db_path=state_db_path, kill_switch_path=kill_switch_path,
            cloid_prefix=cloid_prefix,
            dal=dal, exec_client=exec_client, risk=risk, router=router,
            strategy=strategy, scanner=scanner,
        )

    # ---------- task bodies ----------

    async def _ingest_loop(self, slots: list[AccountSlot]) -> None:
        """Single shared WS subscription feeding MarketState, wrapped in a
        bounded-backoff reconnect loop (SHR-42). Previously a single exception
        killed the only market-data task with no reconnect and no signal, while
        scan/stop-loss/reconcile kept running on a frozen MarketState. Now a
        drop reconnects with backoff, alerts FeedDown/FeedRecovered, and a
        prolonged outage latches all slots halted (positions are unmanageable
        without a feed — fail safe and let the supervisor restart).

        The SeenQuestion-dedup table is replicated across slots (one row per
        slot), which is fine — each slot's DB is independent and the in-process
        cache short-circuits the DB hit after the first emit per question."""
        seen_questions: set[int] = set()
        backoff = self.ingest_reconnect_base_s
        consecutive_failures = 0
        feed_down = False
        while not self.stop_event.is_set():
            adapter = self.adapter_factory()
            try:
                async for ev in adapter.stream(self.subscriptions):
                    if self.stop_event.is_set():
                        return
                    if feed_down:
                        # First event after a reconnect — the feed is back.
                        feed_down = False
                        consecutive_failures = 0
                        backoff = self.ingest_reconnect_base_s
                        await self.bus.publish(FeedRecovered(ts_ns=self._now_ns()))
                    await self._handle_ingest_event(ev, slots, seen_questions)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("adapter ingest crashed; reconnecting")
            # Reached here = the stream raised or ended → the feed is gone.
            if self.stop_event.is_set():
                return
            consecutive_failures += 1
            if not feed_down:
                feed_down = True
                await self.bus.publish(FeedDown(
                    ts_ns=self._now_ns(),
                    consecutive_failures=consecutive_failures,
                ))
            if consecutive_failures >= self.ingest_halt_after_failures:
                logger.error(
                    "feed down for {} consecutive reconnects; latching all "
                    "slots halted and stopping", consecutive_failures,
                )
                for slot in slots:
                    slot.halted = True
                self.stop_event.set()
                return
            await self._sleep_or_stop(backoff)
            backoff = min(backoff * 2, self.ingest_reconnect_max_s)

    async def _handle_ingest_event(
        self, ev, slots: list[AccountSlot], seen_questions: set[int],
    ) -> None:
        from ..events import QuestionMetaEvent
        ev = _remap_reference_symbol(ev)
        self.market_state.apply(ev)
        self.events_ingested += 1
        if isinstance(ev, _PRICE_EVENT_TYPES):
            self._market_dirty.set()
        if not isinstance(ev, QuestionMetaEvent):
            return
        qidx = ev.question_idx
        if qidx in seen_questions:
            return
        seen_questions.add(qidx)
        qv = self.market_state.question(qidx)
        if qv is None:
            return
        from ..strategy.render import question_description
        now_ns = self._now_ns()
        new_q_event = NewQuestion(
            ts_ns=now_ns,
            question_idx=qidx,
            klass=qv.klass,
            description=question_description(qv),
            expiry_ns=qv.expiry_ns,
            leg_count=len(qv.leg_symbols),
        )
        # Mark seen in EVERY slot's DB so the alert doesn't re-fire after
        # restart, then emit one alert globally — but only if at least one
        # slot's allowlist matches the question. symbols.yaml now subscribes to
        # many PM series (ETH/NVDA/NBA/...) for recorder-side data ingestion,
        # and none of those have a corresponding strategy slot; alerting on them
        # would spam Telegram with markets no strategy will ever trade.
        fields = {
            "class": qv.klass,
            "underlying": qv.underlying,
            "period": qv.period,
            # Mirror Scanner.scan so venue/series-scoped slots (PM) correctly
            # count as "tradeable" for the new-question alert gate.
            "venue": qv.venue,
            "series_slug": dict(qv.kv).get("series_slug", ""),
        }
        any_unseen = False
        any_tradeable = False
        for slot in slots:
            if not slot.dal.has_seen_question(qidx):
                slot.dal.mark_question_seen(qidx, now_ns=now_ns)
                any_unseen = True
            if match_question(
                slot.cfg, question_idx=qidx, fields=fields,
            ) is not None:
                any_tradeable = True
        # PM up/down open-strike: single capture path. Fires once the reference
        # 1m candle has closed (now >= ref_ts + 60s) by fetching the Binance
        # spot 1m close. Covers both the observed-open case (market listed just
        # now with ref_ts already ≥60s past) and the missed-open case (engine
        # restarted after the open). The scan loop no longer captures via the
        # live mark — it only reloads a persisted strike.
        if any_tradeable:
            await self._maybe_capture_pm_strike(
                qv, slots, fields, now_ns=now_ns,
            )
        if any_unseen and any_tradeable:
            await self.bus.publish(new_q_event)

    async def _maybe_capture_pm_strike(
        self, qv, slots: list[AccountSlot], fields: dict[str, str], *, now_ns: int,
    ) -> None:
        """Capture a PM up/down strike from the Binance SPOT 1m candle close.

        Single capture path. PM resolves against the spot 1m candle CLOSE at
        strike_ref_ts_ns, so we wait until that minute has closed
        (now >= ref_ts + 60s) and fetch the close. Fires only when: the question
        is a PM up/down market (has strike_ref_ts_ns), its strike is still unset,
        the reference minute has closed, and no slot already has a persisted
        strike (a restart reloads instead). Fetched off the event loop.
        """
        _ONE_MINUTE_NS = 60 * 1_000_000_000
        _CANDLE_SETTLE_NS = 2 * 1_000_000_000  # let Binance finalize/publish the 1m close
        if qv is None or qv.venue != "polymarket":
            return
        if qv.strike == qv.strike:  # already non-NaN
            return
        raw = dict(qv.kv).get("strike_ref_ts_ns")
        if not raw:
            return
        try:
            ref_ts_ns = int(raw)
        except (TypeError, ValueError):
            return
        if now_ns < ref_ts_ns + _ONE_MINUTE_NS + _CANDLE_SETTLE_NS:
            return  # reference 1m candle not closed+published yet — retry next tick
        qidx = qv.question_idx
        # Per-question lock: prevents a concurrent first-sight call and a periodic
        # loop tick from both entering the fetch path simultaneously. The second
        # caller acquires the lock after the first has persisted the strike and
        # bails on the get_pm_strike check inside.
        lock = self._pm_strike_locks.setdefault(qidx, asyncio.Lock())
        async with lock:
            pm_slots = [
                s for s in slots
                if match_question(s.cfg, question_idx=qidx, fields=fields) is not None
            ]
            if any(s.dal.get_pm_strike(qidx) is not None for s in pm_slots):
                return  # a prior run captured it; scanner reloads from DB
            try:
                strike = await asyncio.to_thread(self.klines_fetcher, ref_ts_ns)
            except Exception:
                logger.exception("pm strike capture fetch crashed qidx={}", qidx)
                return
            if strike is None:
                logger.warning(
                    "pm strike capture failed qidx={} ref_ts_ns={} — market skipped",
                    qidx, ref_ts_ns,
                )
                return
            self.market_state.set_question_strike(qidx, strike)
            for s in pm_slots:
                s.dal.set_pm_strike(qidx, strike)
            logger.info(
                "pm strike captured qidx={} strike={} (binance spot 1m close)",
                qidx, strike,
            )
            _MISMATCH_TOL_BPS = 10.0
            mark = self.market_state.last_mark(_SPOT_REF_SYMBOL)
            if mark:
                bps = abs(strike - mark) / mark * 1e4
                if bps > _MISMATCH_TOL_BPS:
                    from .risk_events import PMStrikeMismatch
                    await self.bus.publish(PMStrikeMismatch(
                        ts_ns=now_ns, question_idx=qidx,
                        captured_strike=strike, reference_mark=mark,
                        divergence_bps=bps,
                    ))
                    logger.warning(
                        "pm strike/mark divergence qidx={} strike={} mark={} bps={:.1f} "
                        "(alert only)", qidx, strike, mark, bps,
                    )

    async def _pm_strike_capture_loop(self, slots: list[AccountSlot]) -> None:
        """Retry PM up/down strike capture each second. First-sight capture in
        _ingest_loop fires at discovery, but PM lists markets ~24h before open,
        so the strike can only be fetched once the reference 1m candle closes.
        This loop walks unresolved PM questions and retries until captured."""
        while not self.stop_event.is_set():
            try:
                now_ns = self._now_ns()
                for qv in self.market_state.all_questions():
                    # skip non-PM and already-captured (strike==strike is True
                    # for a real float, False for NaN = still unresolved)
                    if qv.venue != "polymarket" or qv.strike == qv.strike:
                        continue
                    fields = {
                        "class": qv.klass, "underlying": qv.underlying,
                        "period": qv.period, "venue": qv.venue,
                        "series_slug": dict(qv.kv).get("series_slug", ""),
                    }
                    if any(
                        match_question(s.cfg, question_idx=qv.question_idx, fields=fields) is not None
                        for s in slots
                    ):
                        await self._maybe_capture_pm_strike(qv, slots, fields, now_ns=now_ns)
            except Exception:
                logger.exception("pm strike capture loop tick crashed")
            await self._sleep_or_stop(1.0)

    async def _scan_loop(self, slot: AccountSlot) -> None:
        g = slot.cfg.global_
        min_iv = float(getattr(g, "scan_min_interval_seconds", 1.0))
        max_iv = float(getattr(g, "scan_max_interval_seconds", 1.0))
        while not self.stop_event.is_set():
            if slot.halted:
                await self._sleep_or_stop(1.0)
                continue
            try:
                now = self._now_ns()
                slot.scanner.last_reconcile_ns = slot.last_reconcile_ns
                realized_today = await self._realized_pnl_today(slot, now_ns=now)
                for sd in slot.scanner.scan(
                    now_ns=now, realized_pnl_today=realized_today,
                ):
                    slot.decisions_emitted += 1
                    await slot.router.handle(sd.decision, inputs=sd.inputs, now_ns=now)
                slot.scans_completed += 1
            except Exception:
                logger.exception("scan tick crashed alias={}", slot.alias)
            # Coalesce bursts (min floor), then wake on the next tick up to the
            # idle max. min==max reproduces the legacy fixed-interval poll.
            await self._sleep_or_stop(min_iv)
            if max_iv > min_iv:
                await self._wait_for_market_or_timeout(max_interval=max_iv - min_iv)

    async def _heartbeat_loop(self, slots: list[AccountSlot]) -> None:
        """Periodic visibility into engine health, one line per slot. Without
        this the engine is silent on calm markets (strategy gates filter out
        almost everything; adapter doesn't log INFO during normal flow)."""
        prev_events = 0
        prev_scans = {s.alias: 0 for s in slots}
        while not self.stop_event.is_set():
            await self._sleep_or_stop(self.heartbeat_interval_s)
            if self.stop_event.is_set():
                return
            try:
                btc_mark = self.market_state.last_mark("BTC")
                n_questions = len(self.market_state.all_questions())
                d_events = self.events_ingested - prev_events
                prev_events = self.events_ingested
                for slot in slots:
                    n_positions = len(slot.dal.all_positions())
                    n_live = len(slot.dal.live_orders())
                    d_scans = slot.scans_completed - prev_scans[slot.alias]
                    prev_scans[slot.alias] = slot.scans_completed
                    logger.info(
                        "heartbeat alias={} events={} (+{}) scans={} (+{}) "
                        "decisions={} | btc={} questions={} positions={} live_orders={}"
                        "{}",
                        slot.alias,
                        self.events_ingested, d_events,
                        slot.scans_completed, d_scans,
                        slot.decisions_emitted,
                        f"${btc_mark:.2f}" if btc_mark else "none",
                        n_questions, n_positions, n_live,
                        " HALTED" if slot.halted else "",
                    )
                await self._publish_heartbeat(
                    d_events=d_events, n_questions=n_questions,
                )
            except Exception:
                logger.exception("heartbeat crashed")

    # ---- events persistence (Component 2 — engine observability) -----------

    @staticmethod
    def _event_columns(ev: BusEvent) -> dict[str, Any]:
        """Extract the stable, queryable columns from a bus event.

        Single source of truth for how an event maps to the `events` table so
        the static and instance persist loops can't drift. ``reason`` is
        normalised across event types: ``.reason`` (RiskVeto/RiskHalt/
        StaleDataHalt/Exit/ReconcileDrift) or ``.error`` (OrderRejected),
        else None. The full event is kept as ``payload_json`` for fidelity.
        """
        return {
            "ts_ns": ev.ts_ns,
            "alias": getattr(ev, "account_alias", None) or None,
            "kind": ev.kind,
            "question_idx": getattr(ev, "question_idx", None),
            "reason": getattr(ev, "reason", None) or getattr(ev, "error", None) or None,
            "payload_json": ev.model_dump_json(),
        }

    @staticmethod
    async def _events_persist_loop_static(
        sub: asyncio.Queue[BusEvent],
        dal: StateDAL,
        *,
        max_age_ns: int,
        max_rows: int,
        prune_every_n: int = 500,
        # Kept for back-compat with any callers that pass stop_event; ignored
        # because the loop terminates via task cancellation (same pattern as
        # AlertRules.run) so CancelledError exits cleanly.
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Consume every BusEvent and write it to the events table.

        Symmetric to AlertRules.run(alerts_sub): loops on sub.get(), extracts
        the stable fields (alias, kind, question_idx, reason) into named columns
        for SQL queries, and writes the full event as payload_json for fidelity.
        Terminates via task cancellation (CancelledError from sub.get()).

        Prune runs every prune_every_n inserts (not per-insert) to avoid a
        steady per-row overhead while still bounding growth between long idle
        periods. Both age and row-count bounds are applied on each prune call.

        Exposed as a @staticmethod so tests can call it directly without
        constructing a full EngineRuntime, following the same pattern as
        the test suite for alert rules.
        """
        inserted = 0
        while True:
            ev = await sub.get()
            try:
                dal.append_event(**EngineRuntime._event_columns(ev))
                inserted += 1
                if inserted % prune_every_n == 0:
                    try:
                        dal.prune_events(max_age_ns=max_age_ns, max_rows=max_rows)
                    except Exception:
                        logger.exception("events_persist_loop: prune failed")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("events_persist_loop: failed to persist {}", ev.kind)

    async def _events_persist_loop(self) -> None:
        """Subscribe to the bus and persist every event to each slot's DB.

        Route events to every slot's DAL. All slots' DBs are independent
        (one state.db per alias). We write the same event to every slot so
        each slot has a full engine-wide event log — cheap (event volume is
        low) and avoids needing a cross-alias query.

        NOTE: self.slots is populated by run() before tasks are spawned, so
        it is non-empty when this coroutine is awaited. Terminates via task
        cancellation (CancelledError), matching the alerts loop pattern.
        """
        sub = self.bus.subscribe()
        max_age_ns = self.deploy_cfg.events_retention_days * 24 * 3600 * 10**9
        max_rows = self.deploy_cfg.events_retention_max_rows
        dals = [s.dal for s in self.slots]
        inserted = 0
        while True:
            ev = await sub.get()
            try:
                cols = EngineRuntime._event_columns(ev)
                for dal in dals:
                    dal.append_event(**cols)
                inserted += 1
                if inserted % 500 == 0:
                    try:
                        for dal in dals:
                            dal.prune_events(
                                max_age_ns=max_age_ns, max_rows=max_rows
                            )
                    except Exception:
                        logger.exception("events_persist_loop: prune failed")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "events_persist_loop: failed to persist {}", ev.kind
                )

    # ---- liveness / dead-man's-switch (SHR-43) -----------------------------

    def _heartbeat_file_path(self) -> Path:
        return Path(self.deploy_cfg.state_db_path).parent / "engine_heartbeat"

    def _touch_heartbeat_file(self, now_ns: int) -> None:
        """Server-side dead-man's-switch primitive: rewrite a file every
        interval so an external systemd timer / uptime monitor can alert on the
        ABSENCE of updates. The engine cannot alert on its own death (crash,
        OOM, hung loop), so liveness has to be observable from outside."""
        try:
            self._heartbeat_file_path().write_text(str(now_ns))
        except Exception:
            logger.warning("failed to write heartbeat file")

    async def _publish_heartbeat(self, *, d_events: int, n_questions: int) -> None:
        """Emit the per-interval liveness pulse and, when the feed has gone
        silent, a FeedStale alert (SHR-43). A live feed delivers a steady stream
        of mark/book updates, so zero ingested events over a full interval with
        active subscriptions means the feed/ingest is dead — not a calm market.
        EngineHeartbeat is alert-silent; FeedStale drives a Telegram alert."""
        now = self._now_ns()
        await self.bus.publish(EngineHeartbeat(
            ts_ns=now, events_ingested=self.events_ingested,
            d_events=d_events, n_questions=n_questions,
        ))
        self._touch_heartbeat_file(now)
        if d_events == 0 and self.subscriptions:
            await self.bus.publish(FeedStale(
                ts_ns=now, d_events=d_events,
                interval_seconds=self.heartbeat_interval_s,
            ))

    # ---- venue IO offload (SHR-41) -----------------------------------------
    # Every ExecutionClient method is a synchronous, requests-backed SDK call
    # (some wrapped in tenacity retries). The engine runs ONE asyncio loop
    # shared by WS ingest, heartbeat, the stop-loss enforcer, reconcile, and
    # every slot — so calling these inline parks the whole loop for the network
    # round-trip, exactly when the venue is congested. These helpers push the
    # blocking call onto a worker thread. We deliberately offload only the
    # venue calls, not the surrounding DAL/MarketState work: StateDAL opens a
    # fresh connection per call (thread-safe), but MarketState is mutated by the
    # ingest loop and must stay on the loop thread.

    async def _venue_snapshot(
        self, slot: AccountSlot,
    ) -> tuple[list[OpenOrderRow], ClearinghouseState, list[UserFillRow]]:
        """Fetch venue open-orders, clearinghouse state, and the full fills list
        off the event loop. Fills are fetched once and reused as the reconcile
        `fills_lookup` for every cloid — the live lambda ignores its cloid arg
        and returns all fills anyway, so this is behaviour-preserving."""
        open_orders = await asyncio.to_thread(slot.exec_client.open_orders)
        state = await asyncio.to_thread(slot.exec_client.clearinghouse_state)
        fills = await asyncio.to_thread(slot.exec_client.user_fills)
        return open_orders, state, fills

    async def _realized_pnl_today(self, slot: AccountSlot, *, now_ns: int) -> float:
        """Realized PnL since the slot's daily window start, for both the Scanner
        daily-loss read and the continuous-checks cap.

        Settlement-inclusive (SHR-49/53): HL `realized_pnl_since` returns
        closedPnl from fills only — HIP-4 binaries close via settlement payouts,
        which are NOT fills — so persisted settlement PnL is added on top. On a
        venue-read failure we fall back to the DAL value (now also
        settlement-inclusive, no longer structurally zero) and count the
        failure; after `daily_loss_venue_fail_halt` consecutive failures we latch
        the slot halted (fail-safe) rather than keep trading venue-blind."""
        window_start_ns = Scanner._daily_window_start_ns(
            now_ns, hour=slot.cfg.global_.daily_window_start_hour_utc,
        )
        settlement_pnl = await asyncio.to_thread(
            slot.dal.settlement_pnl_since, window_start_ns,
        )
        try:
            venue_pnl = await asyncio.to_thread(
                slot.exec_client.realized_pnl_since, window_start_ns,
            )
            self._venue_pnl_failures[slot.alias] = 0
            return venue_pnl + settlement_pnl
        except Exception:
            n = self._venue_pnl_failures.get(slot.alias, 0) + 1
            self._venue_pnl_failures[slot.alias] = n
            logger.warning(
                "realized_pnl_since failed alias={} (consecutive={}); using "
                "settlement-inclusive DAL", slot.alias, n,
            )
            if n >= self.daily_loss_venue_fail_halt:
                logger.error(
                    "venue PnL unreadable for {} consecutive checks; halting "
                    "slot {} (fail-safe — cap can't be trusted venue-blind)",
                    n, slot.alias,
                )
                slot.halted = True
            # DAL realized_pnl_since already includes settlement PnL (SHR-53).
            return await asyncio.to_thread(
                slot.dal.realized_pnl_since, window_start_ns,
            )

    async def _reconcile_loop(self, slot: AccountSlot) -> None:
        interval = slot.cfg.global_.reconcile_interval_seconds
        while not self.stop_event.is_set():
            await self._sleep_or_stop(float(interval))
            if self.stop_event.is_set():
                return
            try:
                now = self._now_ns()
                # Venue-truth application policy:
                #   HL  → always apply (clearinghouseState is instant +
                #         authoritative; needed for live HIP-4 settlement detect).
                #   PM  → apply only on the FIRST successful pass after startup
                #         (to adopt positions we missed while down — the restart
                #         gate runs before market data is loaded so it can't),
                #         then alert-only. PM's data-api lags/flaps, so live we
                #         trust our own fill ledger and only alert on diffs.
                is_pm = self._is_pm_slot(slot)
                apply_positions = (not is_pm) or (not slot.pm_startup_position_synced)
                # Fetch venue open-orders, clearinghouse state, and fills off the
                # event loop (SHR-41); the cached fills feed the Reconciler's
                # fills_lookup for every cloid.
                venue_open, venue_state, all_fills = await self._venue_snapshot(slot)
                rec = Reconciler(
                    slot.dal,
                    fills_lookup=lambda c, _f=all_fills: _f,
                    symbol_to_question=self.market_state.symbol_to_question_map(),
                    cloid_prefix=slot.cloid_prefix,
                    account_alias=slot.alias,
                    apply_position_changes=apply_positions,
                )
                res = rec.run(
                    venue_open=venue_open,
                    venue_state=venue_state,
                    now_ns=now,
                )
                # Mark the PM startup sync done only once we've had an
                # authoritative (positions_known) pass — so a data-api outage at
                # startup doesn't prematurely flip us to alert-only.
                if is_pm and apply_positions and venue_state.positions_known:
                    slot.pm_startup_position_synced = True
                # Translate "local position vanished from venue" into a
                # settlement event before publishing any other drift. On HL
                # HIP-4 the venue auto-closes positions at settlement and the
                # leg's L2 book goes silent before the polled SettlementEvent
                # catches up. Without this, operators see a generic DRIFT
                # alert (no 🏁 emoji, no PnL line) followed by a spurious
                # STALE DATA HALT a few seconds later. Marking the question
                # settled here suppresses the stale-halt and also stops the
                # strategy from re-entering on the now-closed market.
                from ..strategy.render import (
                    outcome_description, question_description,
                    settlement_pnl_usd,
                )
                for qidx, sym, lp in res.vanished_positions:
                    self.market_state.mark_question_settled(qidx)
                    qv = self.market_state.question(qidx)
                    # When the position vanished before the settle event was
                    # polled, `qv.settled_symbol` is still unset and
                    # settlement_pnl_usd falls back to lp.realized_pnl. Once
                    # the polled SettlementEvent lands the next vanish-style
                    # alert (a re-emit on the same qidx) will carry the real
                    # PnL.
                    realized = settlement_pnl_usd(
                        qv, sym, lp.qty, lp.avg_entry,
                        prior_realized=lp.realized_pnl,
                    )
                    # Persist settlement PnL for the daily-loss gate (SHR-53).
                    # Upsert keyed by qidx: if this is the pre-settle vanish
                    # (incomplete PnL) the later authoritative re-emit overwrites
                    # it; shared with router._close_settled, so no double-book.
                    slot.dal.record_settlement(
                        question_idx=qidx, symbol=sym,
                        realized_pnl=realized, ts_ns=now,
                    )
                    await self.bus.publish(Exit(
                        ts_ns=now, account_alias=slot.alias,
                        question_idx=qidx, symbol=sym,
                        qty=lp.qty, realized_pnl=realized,
                        reason="settlement",
                        question_description=question_description(qv) if qv else "",
                        outcome_description=outcome_description(qv, sym) if qv else "",
                    ))
                for ev in res.drift_events:
                    await self.bus.publish(ev)
                for cloid, symbol in res.orphans_to_cancel:
                    await asyncio.to_thread(
                        slot.exec_client.cancel, cloid=cloid, symbol=symbol,
                    )
                slot.last_reconcile_ns = now
            except Exception:
                logger.exception("reconcile crashed alias={}", slot.alias)

    async def _continuous_checks_loop(self, slot: AccountSlot) -> None:
        kill_path = slot.kill_switch_path
        # PM-only flag gating the OrderUnconfirmed + RedemptionTimeout
        # watchdogs so HL slots don't emit PM-specific alerts.
        is_pm = isinstance(slot.account_cfg, PolymarketAccount)
        # PM-only bus subscription. We capture settlement Exit events into
        # `slot.pm_settlements` so the RedemptionTimeout watchdog has a
        # ts→qty record to scan 6h later. HL slots don't subscribe.
        pm_bus_sub: asyncio.Queue | None = self.bus.subscribe() if is_pm else None
        while not self.stop_event.is_set():
            try:
                if slot.halted:
                    await self._sleep_or_stop(1.0)
                    continue
                now = self._now_ns()
                # Kill switch — per-slot path. Operator can halt one strategy
                # without killing the other; we only set self.stop_event if
                # ALL slots are halted (handled below in the loop).
                if slot.risk.kill_switch_active(kill_path):
                    await self.bus.publish(KillSwitchActivated(
                        ts_ns=now, account_alias=slot.alias, path=str(kill_path),
                    ))
                    slot.halted = True
                    self._maybe_stop_all_halted(slot)
                    continue

                # Stop-loss enforcement. When stop_loss_loop_enabled, a dedicated
                # event-driven loop owns this (acts within scan_min_interval);
                # otherwise enforce here at the continuous-checks 1 Hz cadence.
                if not slot.cfg.global_.stop_loss_loop_enabled:
                    await self._enforce_stop_losses(slot, now_ns=now)

                # Daily loss — per slot. Read from HL (venue truth) instead of
                # the local DB; same reasoning as Scanner._pnl_provider — the
                # local fills table is empty on the happy path and closed
                # Positions are deleted, so the DB-side calculation is
                # structurally near-zero and the cap would never fire. On HL
                # outage, fall back to the DAL so the engine doesn't halt
                # itself on a transient network blip.
                #
                # The window cutoff is `daily_window_start_hour_utc` (default
                # 0 = UTC midnight; set to 6 to align with HL HIP-4 binary
                # settlement at 06:00 UTC / 11:30 IST so the cap resets in
                # lockstep with the market cycle). Offloaded off the event loop
                # (SHR-41) via the shared helper, which also handles the DAL
                # fallback on venue outage.
                pnl = await self._realized_pnl_today(slot, now_ns=now)
                if pnl < -slot.cfg.global_.daily_loss_cap_usd:
                    await self.bus.publish(DailyLossHalt(
                        ts_ns=now, account_alias=slot.alias,
                        realized_pnl=pnl,
                        cap=slot.cfg.global_.daily_loss_cap_usd,
                    ))
                    slot.halted = True
                    self._maybe_stop_all_halted(slot)
                    continue

                # PM-only: drain bus subscription into pm_settlements so the
                # RedemptionTimeout watchdog can fire 6h after a settlement
                # Exit. We filter to settlement Exits matching this slot's
                # alias — other slots' settlements / non-settlement events are
                # ignored. `setdefault` means a re-published Exit (e.g.
                # reconciler racing the close path) doesn't reset the clock.
                if pm_bus_sub is not None:
                    while True:
                        try:
                            ev = pm_bus_sub.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if (
                            isinstance(ev, Exit)
                            and ev.reason == "settlement"
                            and ev.account_alias == slot.alias
                        ):
                            slot.pm_settlements.setdefault(
                                ev.question_idx,
                                (ev.ts_ns, ev.symbol, ev.qty, ev.realized_pnl),
                            )

                # PM-only: unconfirmed-order + redemption watchdogs. Surfaces
                # PM CLOB orders that have sat as `open` past the threshold
                # (chain congestion, dropped acks), and settled positions
                # that haven't been redeemed within 6h.
                if is_pm:
                    for unconf in _pm_check_unconfirmed_orders(slot, now):
                        await self.bus.publish(unconf)
                    for redempt in _pm_check_redemption_timeouts(slot, now):
                        await self.bus.publish(redempt)

                # Stale-data halt is per-trade; we also surface it as an alert here.
                if positions_db:
                    settled_qidxs = {
                        q.question_idx for q in self.market_state.all_questions() if q.settled
                    }
                    held_symbols = {
                        p.symbol for p in positions_db
                        if p.question_idx not in settled_qidxs
                    }
                    books_only_held = {sym: self.market_state.book(sym) for sym in held_symbols}
                    books_only_held = {s: b for s, b in books_only_held.items() if b is not None}
                    stale_now = set(slot.risk.stale_books(books_only_held, now_ns=now))
                    # Evict recovered symbols so a future stale episode re-alerts.
                    slot.stale_alerted_symbols &= stale_now
                    for sym in stale_now:
                        if sym in slot.stale_alerted_symbols:
                            continue  # already alerted this episode — don't spam every 1s
                        slot.stale_alerted_symbols.add(sym)
                        b = books_only_held[sym]
                        await self.bus.publish(StaleDataHalt(
                            ts_ns=now, account_alias=slot.alias, symbol=sym,
                            age_seconds=(now - b.last_l2_ts_ns) / 1e9,
                        ))
            except Exception:
                logger.exception("continuous checks crashed alias={}", slot.alias)
            await self._sleep_or_stop(1.0)

    # ---------- stop-loss enforcement (extracted for event-driven loop) --------

    async def _enforce_stop_losses(self, slot: AccountSlot, *, now_ns: int) -> None:
        """One stop-loss enforcement pass: find breached stops on held
        positions and fire reduce-only IOC exits. Extracted from
        _continuous_checks_loop so it can run in its own event-driven loop
        (P1). Behaviour-identical to the prior inline block."""
        now = now_ns
        positions_db = slot.dal.all_positions()
        if not positions_db:
            return
        books = {}
        from ..strategy.types import Position as SPos
        for p in positions_db:
            b = self.market_state.book(p.symbol)
            if b is not None:
                books[p.symbol] = b
        sps = [
            SPos(question_idx=p.question_idx, symbol=p.symbol, qty=p.qty,
                 avg_entry=p.avg_entry, stop_loss_price=p.stop_loss_price,
                 last_update_ts_ns=p.last_update_ts_ns)
            for p in positions_db
        ]
        breached = slot.risk.breached_stops(sps, books)
        for sp in breached:
            await self.bus.publish(StopLossTriggered(
                ts_ns=now, account_alias=slot.alias,
                question_idx=sp.question_idx, symbol=sp.symbol, qty=sp.qty,
                trigger_px=sp.stop_loss_price,
            ))
            from ..strategy.types import Action, Decision, OrderIntent
            b = books.get(sp.symbol)
            if b is None or b.bid_px is None:
                continue
            intent = OrderIntent(
                question_idx=sp.question_idx, symbol=sp.symbol,
                side="sell" if sp.qty > 0 else "buy",
                size=abs(sp.qty), limit_price=b.bid_px,
                cloid=f"{slot.cloid_prefix}{uuid.uuid4().hex}",
                time_in_force="ioc", reduce_only=True, exit_reason="stop_loss",
            )
            from .risk import RiskInputs
            inp = RiskInputs(
                question=self.market_state.question(sp.question_idx) or _stub_question(sp),
                question_fields={}, reference_price=0.0, book=b,
                recent_volume_usd=0.0, positions=sps,
                live_orders_total_notional=0.0, realized_pnl_today=0.0,
                kill_switch_active=False, last_reconcile_ns=slot.last_reconcile_ns,
                now_ns=now,
            )
            await slot.router.handle(
                Decision(action=Action.EXIT, intents=(intent,)), inputs=inp, now_ns=now,
            )

    async def _stop_loss_loop(self, slot: AccountSlot) -> None:
        """Event-driven stop-loss enforcement (P1). Active only when
        stop_loss_loop_enabled; wakes on the market-dirty signal, bounded by
        scan_min/max_interval_seconds, so a stop breach is acted on promptly
        without speeding up the venue-reading checks in _continuous_checks_loop."""
        g = slot.cfg.global_
        min_iv = float(getattr(g, "scan_min_interval_seconds", 1.0))
        max_iv = float(getattr(g, "scan_max_interval_seconds", 1.0))
        while not self.stop_event.is_set():
            if slot.halted:
                await self._sleep_or_stop(1.0)
                continue
            try:
                await self._enforce_stop_losses(slot, now_ns=self._now_ns())
            except Exception:
                logger.exception("stop-loss loop crashed alias={}", slot.alias)
            await self._sleep_or_stop(min_iv)
            if max_iv > min_iv:
                await self._wait_for_market_or_timeout(max_interval=max_iv - min_iv)

    # ---------- helpers ----------

    def _maybe_stop_all_halted(self, just_halted: AccountSlot) -> None:
        """If every slot has latched halted, drop the global stop event so
        the engine exits cleanly. A single-strategy halt no longer kills the
        whole engine (that's the point of multi-account isolation)."""
        # Tracked via attribute on self to avoid walking slots twice.
        if not hasattr(self, "_slot_halt_count"):
            self._slot_halt_count = 0
            self._slot_total = len(self.strategies)
        self._slot_halt_count += 1
        if self._slot_halt_count >= self._slot_total:
            self.stop_event.set()

    async def _wait_for_market_or_timeout(self, *, max_interval: float) -> None:
        """Block until the market-dirty signal fires OR max_interval elapses OR
        the engine stops. Clears the signal before returning so the next call
        waits for a fresh tick. Multi-consumer note: several loops await the one
        shared Event; a clear() by one consumer may make a sibling wait up to
        max_interval, which is the intended idle floor — benign and bounded."""
        if self.stop_event.is_set():
            return
        dirty = asyncio.ensure_future(self._market_dirty.wait())
        stop = asyncio.ensure_future(self.stop_event.wait())
        try:
            await asyncio.wait(
                {dirty, stop}, timeout=max_interval,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            dirty.cancel()
            stop.cancel()
        self._market_dirty.clear()

    async def _sleep_or_stop(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self.stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    def _make_telegram(self, http: aiohttp.ClientSession) -> TelegramClient:
        if self.telegram_factory is not None:
            return self.telegram_factory(http)
        return TelegramClient(
            bot_token=self.deploy_cfg.alerts.telegram.bot_token,
            chat_id=self.deploy_cfg.alerts.telegram.chat_id,
            session=http,
        )

    @staticmethod
    def _now_ns() -> int:
        import time as _t
        return _t.time_ns()


def _stub_question(p):
    from ..strategy.types import QuestionView
    return QuestionView(
        question_idx=p.question_idx, yes_symbol=p.symbol, no_symbol="",
        strike=0.0, expiry_ns=0, underlying="", klass="", period="",
    )
