from __future__ import annotations

import asyncio
import resource
import signal
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable

import aiohttp
import msgspec
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
from .market_state import MarketState
from .reconcile import Reconciler
from .restart_drift import RestartDriftGate
from .risk import RiskGate
from .risk_events import (
    DailyLossHalt, EngineHeartbeat, Exit, FeedDown, FeedRecovered, FeedStale,
    KillSwitchActivated, MemoryHalt, NewQuestion,
    StaleDataHalt, StopLossTriggered,
)
from .router import Router
from .scanner import Scanner
from .state import CachedStateDAL, StateDAL
from ..alerts.rules import AlertRules
from ..alerts.telegram import TelegramClient
from ..marketdata.position_math import DUST_QTY_ABS_TOL
from ..strategy.base import Strategy

# Config-building, watchdog, strike-capture, and event-sink helpers were split
# out of this module (see each module's docstring). They are re-exported here so
# the long-standing `from hlanalysis.engine.runtime import ...` paths (used by
# replay.py and the engine test suite) keep working.
from .config_builders import (  # noqa: F401  (re-exported for back-compat)
    _LR_GLOBAL_SOURCED,
    _build_strategy_for_slot,
    _late_resolution_config_from_entry,
    build_exec_client,
    build_late_resolution_config,
    build_late_resolution_configs_by_class,
    build_theta_harvester_config,
    build_theta_harvester_configs_by_class,
    reference_sampling_dt_seconds,
    reference_vol_lookback_seconds,
)
from .pm_watchdogs import (  # noqa: F401  (re-exported for back-compat)
    PM_REDEMPTION_TIMEOUT_S,
    PM_UNCONFIRMED_THRESHOLD_S,
    _pm_check_redemption_timeouts,
    _pm_check_unconfirmed_orders,
)
from .pm_strike import maybe_capture_pm_strike, pm_strike_capture_loop
from .events_sink import events_persist_loop


# Internal symbol the Binance SPOT reference feed is remapped to (see
# _remap_reference_symbol). PM strategy slots reference this via
# `reference_symbol: BTCUSDT_SPOT` in config/strategy.yaml — that YAML string
# MUST match this constant exactly, or the slot reads an empty book.
_SPOT_REF_SYMBOL = "BTCUSDT_SPOT"


def _is_transient_venue_error(exc: BaseException) -> bool:
    """True for expected, self-recovering venue read failures the client's
    read-retry already exhausted — HL 429 rate limits, connection blips, 5xx.

    The reconcile loop uses this to log such failures concisely and retry next
    cycle, rather than emitting a full ``reconcile crashed`` traceback for
    routine venue flakiness (e.g. HL 429 bursts when two slots poll /info from
    one IP). Genuine, non-transient errors still surface with a full traceback.

    Covers both venues: HL's RateLimitError plus the PM client's own
    transient classifier (requests/builtin conn+timeout, HTTP 5xx and 429)."""
    from .hl_client import RateLimitError
    from .pm_client import _pm_is_transient
    if isinstance(exc, (RateLimitError, ConnectionError, TimeoutError)):
        return True
    return _pm_is_transient(exc)


def _remap_reference_symbol(ev: NormalizedEvent) -> NormalizedEvent:
    """Rename Binance SPOT BTCUSDT events to BTCUSDT_SPOT so the PM slots'
    ``reference_symbol: BTCUSDT_SPOT`` resolves to the spot feed (not any
    future perp entry). No-op for every other event."""
    if (
        ev.venue == "binance"
        and ev.product_type == ProductType.SPOT
        and ev.symbol == "BTCUSDT"
    ):
        return msgspec.structs.replace(ev, symbol=_SPOT_REF_SYMBOL)
    return ev


@dataclass
class PmSlotState:
    """Polymarket-only mutable slot state. Present (non-None) on a slot iff its
    venue is ``polymarket``; HL slots leave ``AccountSlot.pm`` as None. Grouping
    these here keeps the PM-specific bookkeeping out of HL slots' way and gives
    the PM watchdogs/loops a single object to read and mutate.
    """
    # cloids that have already triggered an OrderUnconfirmed alert. Cleared when
    # the cloid drops out of live_orders (status flipped to
    # filled/cancelled/rejected) so a future stall on a re-placed order with the
    # same cloid would re-fire.
    alerted_unconfirmed_cloids: set[str] = field(default_factory=set)
    # (qidx -> (settled_ts_ns, symbol, qty, realized_pnl)) for PM settlement
    # Exits, captured via the bus subscription in `_continuous_checks_loop`.
    # Drives RedemptionTimeout.
    settlements: dict[int, tuple[int, str, float, float]] = field(default_factory=dict)
    # qidxs that have already fired RedemptionTimeout — prevents per-tick spam
    # after the 6h threshold trips.
    alerted_redemption_qidxs: set[int] = field(default_factory=set)
    # Set True after the first reconcile pass that authoritatively synced
    # positions from venue truth (the data-api). Until then the live reconcile
    # APPLIES venue changes (to adopt positions missed while the engine was down
    # — the restart gate runs before market data loads, so it can't); afterwards
    # the live reconcile is alert-only and our fill ledger is the source of
    # truth (PM data-api lags/flaps). HL ignores this (always applies).
    startup_position_synced: bool = False


@dataclass
class AccountSlot:
    """One (strategy, account) pair. Owns its own DAL, HL client, risk gate,
    router, reconciler, and strategy instance. The only thing shared with
    sibling slots is the engine's MarketState + WS feed.
    """
    cfg: StrategyConfig
    account_cfg: "HyperliquidAccount | PolymarketAccount"
    # Venue discriminator, set once at build time from account_cfg.venue
    # ("hyperliquid" | "polymarket"). The single source of truth for "is this a
    # PM slot?" — read via the `is_pm` predicate, never via ad-hoc isinstance /
    # deploy-lookup checks.
    venue: str
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
    # PM-only mutable state. Non-None iff this is a Polymarket slot (set at build
    # time); HL slots leave it None. Populated/mutated by the PM watchdogs and
    # `_continuous_checks_loop`.
    pm: PmSlotState | None = None
    # Symbols that have already fired a StaleDataHalt alert this stale episode.
    # NOT PM-only: the continuous-checks loop surfaces StaleDataHalt for any slot
    # (HL or PM) holding a position whose leg book goes quiet, and dedups via
    # this set so the ~1s loop doesn't re-publish every second. Evicted when the
    # book recovers so a fresh episode re-alerts. Alert-only — the per-trade
    # stale veto in risk/scanner is unaffected.
    stale_alerted_symbols: set[str] = field(default_factory=set)

    @property
    def alias(self) -> str:
        return self.cfg.account_alias

    @property
    def is_pm(self) -> bool:
        """The single PM-slot predicate (venue discriminator)."""
        return self.venue == "polymarket"


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
    # RSS self-halt guard (W1.9). If process RSS (in KB) exceeds this ceiling
    # the heartbeat loop halts all slots and writes their kill-switch flags
    # before the kernel OOM-killer fires. Default is ~85% of a 1 GB box.
    # Override on EngineRuntime for different box sizes.
    rss_halt_kb: int = 850_000

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
            venue_by_alias: dict[str, str] = {
                s.alias: ("PM" if s.is_pm else "HL") for s in slots
            }
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

    # ---------- cadence registration ----------

    def _register_reference_cadences(self, slots: list[AccountSlot]) -> None:
        """Register each slot's default reference cadence AND any per-class
        theta-override cadences on the shared MarketState, so every (symbol, dt)
        bar series exists and accumulates from the one shared feed. Multiple
        cadences per reference_symbol are supported (each bucketed independently).
        Conflicting σ sources for the same reference_symbol still raise (see
        MarketState.set_reference_source)."""
        for slot in slots:
            sym = slot.cfg.reference_symbol
            self.market_state.set_reference_cadence(
                sym,
                sampling_dt_seconds=reference_sampling_dt_seconds(slot.cfg),
                lookback_seconds=reference_vol_lookback_seconds(slot.cfg),
            )
            for dt_s, _n in Scanner.cadence_by_class(slot.cfg).values():
                self.market_state.set_reference_cadence(
                    sym,
                    sampling_dt_seconds=dt_s,
                    lookback_seconds=reference_vol_lookback_seconds(slot.cfg),
                )
            # Couple the σ/OHLC source (mark | bbo) per reference symbol. Unlike
            # the cadence (which now accepts multiple per symbol), the source is
            # fail-fast: slots sharing a symbol must agree on one σ source.
            # Default "mark" preserves HL behaviour bit-identically.
            self.market_state.set_reference_source(
                sym, slot.cfg.reference_sigma_source,
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
        else:
            exec_client = build_exec_client(alias, acct, s_cfg.paper_mode)

        risk = RiskGate(s_cfg)
        # PM market sells floor the share amount to 2dp, stranding sub-0.01 dust
        # that wedges the position open (2026-06-06 v31_pm). PM slots treat a
        # reduce landing within that dust of flat as a full close (and suppress
        # un-sellable dust sells). HL fills the exact size, so it stays ~exact.
        reduce_close_atol = (
            DUST_QTY_ABS_TOL if acct.venue == "polymarket" else 1e-9
        )
        router = Router(
            dal=dal, gate=risk, bus=self.bus, exec_client=exec_client,
            strategy_cfg=s_cfg, strategy_id=s_cfg.name,
            cloid_prefix=cloid_prefix,
            reduce_close_atol=reduce_close_atol,
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
            cfg=s_cfg, account_cfg=acct, venue=acct.venue,
            state_db_path=state_db_path, kill_switch_path=kill_switch_path,
            cloid_prefix=cloid_prefix,
            dal=dal, exec_client=exec_client, risk=risk, router=router,
            strategy=strategy, scanner=scanner,
            pm=PmSlotState() if acct.venue == "polymarket" else None,
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
        """Thin delegator to ``pm_strike.maybe_capture_pm_strike`` (the capture
        logic lives there; this preserves the method API)."""
        await maybe_capture_pm_strike(self, qv, slots, fields, now_ns=now_ns)

    async def _pm_strike_capture_loop(self, slots: list[AccountSlot]) -> None:
        """Thin delegator to ``pm_strike.pm_strike_capture_loop``."""
        await pm_strike_capture_loop(self, slots)

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
                # W1.9: RSS self-halt guard — check on every heartbeat so an
                # OOM trajectory is caught before the kernel fires. Runs
                # AFTER the info log so the log line is always emitted.
                await self._check_rss_halt(slots)
                await self._publish_heartbeat(
                    d_events=d_events, n_questions=n_questions,
                )
            except Exception:
                logger.exception("heartbeat crashed")

    # ---- events persistence (Component 2 — engine observability) -----------

    async def _events_persist_loop(self) -> None:
        """Subscribe to the bus and persist every event to each slot's DB via
        the unified ``events_sink.events_persist_loop``.

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
        await events_persist_loop(
            sub, [s.dal for s in self.slots],
            max_age_ns=max_age_ns, max_rows=max_rows,
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

        Venue-truth, no double-count: HL exposes HIP-4 / bucket settlement as a
        *fill* (`dir="Settlement"`, `closedPnl` populated), so `realized_pnl_since`
        ALREADY includes the settlement payout — adding the separately-persisted
        settlement PnL on top double-counts it (and historically mis-signed it
        for multi-leg buckets: a winning leg booked as a total loss tripped a
        spurious DAILY LOSS HALT). PM settles via on-chain redeem, which is NOT a
        CLOB fill, so PM's venue read misses it and the persisted settlement must
        be added back. On a venue-read failure we fall back to the DAL value (now
        also settlement-inclusive, no longer structurally zero) and count the
        failure; after `daily_loss_venue_fail_halt` consecutive failures we latch
        the slot halted (fail-safe) rather than keep trading venue-blind."""
        window_start_ns = Scanner._daily_window_start_ns(
            now_ns, hour=slot.cfg.global_.daily_window_start_hour_utc,
        )
        try:
            venue_pnl = await asyncio.to_thread(
                slot.exec_client.realized_pnl_since, window_start_ns,
            )
            self._venue_pnl_failures[slot.alias] = 0
            if slot.is_pm:
                settlement_pnl = await asyncio.to_thread(
                    slot.dal.settlement_pnl_since, window_start_ns,
                )
                return venue_pnl + settlement_pnl
            return venue_pnl  # HL: settlement already in venue fills' closedPnl
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
                is_pm = slot.is_pm
                apply_positions = (not is_pm) or (not slot.pm.startup_position_synced)
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
                    slot.pm.startup_position_synced = True
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
                is_pm = slot.is_pm
                window_start_ns = Scanner._daily_window_start_ns(
                    now, hour=slot.cfg.global_.daily_window_start_hour_utc,
                )
                for qidx, sym, lp in res.vanished_positions:
                    self.market_state.mark_question_settled(qidx)
                    qv = self.market_state.question(qidx)
                    if not is_pm:
                        # HL: the position vanished because the venue settled it
                        # — and HL books settlement as a fill (dir="Settlement",
                        # closedPnl set), already present in `all_fills`. Sum the
                        # venue's own per-leg closedPnl so our realized PnL ==
                        # HL by construction. This replaces the old re-derivation
                        # that hardcoded the YES leg as winner and booked winning
                        # bucket legs as total losses. Not persisted:
                        # realized_pnl_since already counts these fills.
                        realized = sum(
                            f.closed_pnl - f.fee
                            for f in all_fills
                            if f.symbol == sym and f.ts_ns >= window_start_ns
                        )
                    else:
                        # PM: the redeem is not a CLOB fill, so re-derive from
                        # the (price-sourced, venue-correct) winner and persist
                        # for the daily-loss gate. When the position vanished
                        # before the settle event was polled, `qv.settled_symbol`
                        # is unset and settlement_pnl_usd falls back to
                        # lp.realized_pnl; the later authoritative re-emit on the
                        # same qidx overwrites it.
                        realized = settlement_pnl_usd(
                            qv, sym, lp.qty, lp.avg_entry,
                            prior_realized=lp.realized_pnl,
                        )
                        slot.dal.record_settlement(
                            question_idx=qidx, symbol=sym,
                            realized_pnl=realized, ts_ns=now,
                        )
                    await self.bus.publish(Exit(
                        ts_ns=now, account_alias=slot.alias,
                        question_idx=qidx, symbol=sym,
                        qty=lp.closed_qty + lp.qty, realized_pnl=realized,
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
                # SHR-44: bound the question set on the 1 GB box. Retain a
                # generous window after settlement so late reconciles / settlement
                # Exits still find the question, then drop it.
                self.market_state.evict_settled_questions(
                    now_ns=self._now_ns(),
                    retain_after_settle_ns=6 * 3600 * 1_000_000_000,  # 6h
                )
                slot.last_reconcile_ns = now
            except Exception as exc:
                # An expected, self-recovering venue read failure (HL 429 burst,
                # connection blip, 5xx) that the client's read-retry already
                # exhausted is NOT a crash — the loop retries next cycle. Log it
                # concisely instead of dumping a multi-frame "reconcile crashed"
                # traceback that reads like a bug and buries real ones.
                if _is_transient_venue_error(exc):
                    logger.warning(
                        "reconcile skipped alias={} — transient venue error "
                        "(retried next cycle): {}", slot.alias, exc,
                    )
                else:
                    logger.exception("reconcile crashed alias={}", slot.alias)

    async def _continuous_checks_loop(self, slot: AccountSlot) -> None:
        kill_path = slot.kill_switch_path
        # PM-only flag gating the OrderUnconfirmed + RedemptionTimeout
        # watchdogs so HL slots don't emit PM-specific alerts.
        is_pm = slot.is_pm
        # PM-only bus subscription. We capture settlement Exit events into
        # `slot.pm.settlements` so the RedemptionTimeout watchdog has a
        # ts→qty record to scan 6h later. HL slots don't subscribe.
        pm_bus_sub: asyncio.Queue | None = self.bus.subscribe() if is_pm else None
        while not self.stop_event.is_set():
            try:
                if slot.halted:
                    # W1.9: even a halted slot must protect open positions.
                    # The dedicated stop-loss loop also runs enforcement (when
                    # stop_loss_loop_enabled); when it is disabled, this is
                    # the only path.  New entries remain blocked (scan_loop
                    # still skips when halted).
                    if not slot.cfg.global_.stop_loss_loop_enabled or slot.blocked:
                        await self._enforce_stop_losses(slot, now_ns=self._now_ns())
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
                # A `blocked` slot never gets the dedicated loop spawned (see
                # run(): `... and not slot.blocked`), so it MUST still be enforced
                # here even when the flag is on — else a blocked slot holding a
                # position would have no stop-loss path at all.
                if (not slot.cfg.global_.stop_loss_loop_enabled) or slot.blocked:
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
                    # W1.9: latch persistently — writes flag file so a restart
                    # re-reads it and stays halted (operator-clearable only).
                    self._latch_kill_switch(slot)
                    self._maybe_stop_all_halted(slot)
                    continue

                # PM-only: drain bus subscription into slot.pm.settlements so the
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
                            slot.pm.settlements.setdefault(
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
                # Re-read positions (cached, cheap): the P1.4 stop-loss extraction
                # moved the prior shared `positions_db` fetch into
                # _enforce_stop_losses, so this block must fetch its own — without
                # it the name is undefined, the broad `except` below swallows the
                # NameError, and StaleDataHalt alerting is silently dead for every
                # slot holding a position (a forbidden safety-gate regression).
                positions_db = slot.dal.all_positions()
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
        # SHR-48: in-flight exit guard — if an exit/stop IOC is already live
        # for a position's question_idx (pending/open/partially_filled), don't
        # stack another full-size IOC before the prior ACK resolves.  The ~1 Hz
        # loop would otherwise fire a fresh order every tick, walking a thin
        # book to zero.  We build the set once per enforcement pass.
        live_question_idxs = {o.question_idx for o in slot.dal.live_orders()}
        for sp in breached:
            if sp.question_idx in live_question_idxs:
                continue  # exit already in flight; skip until ACK clears it
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
        without speeding up the venue-reading checks in _continuous_checks_loop.

        Note: stop-loss enforcement intentionally runs even when slot.halted is
        True (W1.9). A daily-loss latch blocks NEW entries via the scan loop,
        but must not abandon open positions — those need protective exits."""
        g = slot.cfg.global_
        min_iv = float(getattr(g, "scan_min_interval_seconds", 1.0))
        max_iv = float(getattr(g, "scan_max_interval_seconds", 1.0))
        while not self.stop_event.is_set():
            try:
                await self._enforce_stop_losses(slot, now_ns=self._now_ns())
            except Exception:
                logger.exception("stop-loss loop crashed alias={}", slot.alias)
            await self._sleep_or_stop(min_iv)
            if max_iv > min_iv:
                await self._wait_for_market_or_timeout(max_interval=max_iv - min_iv)

    # ---------- helpers ----------

    def _latch_kill_switch(self, slot: AccountSlot) -> None:
        """Write the kill-switch flag file for a slot and set slot.halted.

        Called on a confirmed daily-loss breach (W1.9). Writing the flag file
        makes the halt PERSISTENT across engine restarts — the
        _continuous_checks_loop reads the flag on every iteration, so an
        operator must explicitly remove it to resume the slot (no auto-clear).

        Safe to call multiple times (touch is idempotent).
        """
        slot.halted = True
        try:
            slot.kill_switch_path.parent.mkdir(parents=True, exist_ok=True)
            slot.kill_switch_path.touch(exist_ok=True)
        except OSError:
            logger.error(
                "failed to write kill-switch flag alias={} path={}; slot is "
                "halted in memory but will NOT survive a restart — investigate "
                "immediately", slot.alias, slot.kill_switch_path,
            )

    @staticmethod
    def _read_rss_kb() -> int:
        """Current process RSS in KiB (NOT peak). Linux reads /proc/self/status
        VmRSS; ru_maxrss is peak and never decreases → false permanent halt.

        ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` is the PEAK (maximum)
        RSS for the process lifetime — it is monotonic and never decreases.
        On a 1 GB box where duckdb compaction transiently spikes to ~570 MB
        and then frees memory, ``ru_maxrss`` stays elevated permanently and
        the guard would latch-halt all slots even though current memory is
        fine.  This implementation reads the CURRENT resident set instead:

        * Linux primary: ``/proc/self/status`` ``VmRSS`` field (already KiB).
        * Linux fallback: ``/proc/self/statm`` resident-pages × page-size.
        * Non-Linux (macOS dev): ``ru_maxrss`` bytes ÷ 1024 (macOS reports
          bytes, not KiB); peak semantics are acceptable on dev — the guard
          is only safety-critical in production Linux.

        Tests monkeypatch this method to inject arbitrary RSS values without
        caring about platform units.
        """
        import sys
        if sys.platform == "linux":
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            return int(line.split()[1])  # already kB
            except OSError:
                pass
            try:
                import os
                pages = int(open("/proc/self/statm").read().split()[1])
                return pages * os.sysconf("SC_PAGE_SIZE") // 1024
            except (OSError, ValueError, IndexError):
                pass
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS: bytes → KiB; Linux (last-resort fallback): already KiB.
        return raw // 1024 if sys.platform == "darwin" else raw

    async def _check_rss_halt(self, slots: list[AccountSlot]) -> None:
        """RSS self-halt guard (W1.9).

        Read process RSS and, if it exceeds ``rss_halt_kb``, latch all slots
        halted + write their kill-switch flags + publish a MemoryHalt bus
        event so AlertRules forwards a Telegram alert to the operator.

        This is the memory safety net behind the SHR-44/63 memory fixes: if
        buffered state still grows past budget we stop placing rather than die
        mid-position under the kernel OOM-killer.

        The halt latches all slots (kill-switch flag files written) and
        requires operator flag-removal + engine restart to resume.  Stop-loss
        enforcement loops intentionally keep running after the latch so open
        positions remain protected while the operator intervenes.

        The MemoryHalt event is published only once — on the tick that first
        crosses the ceiling — because _latch_kill_switch sets slot.halted=True
        and _heartbeat_loop already logs the halted flag on every subsequent
        tick, avoiding alert spam.
        """
        try:
            rss_kb = self._read_rss_kb()
        except Exception:
            logger.warning("could not read process RSS; skipping RSS guard")
            return
        if rss_kb <= self.rss_halt_kb:
            return
        logger.error(
            "RSS {}KB exceeds ceiling {}KB — self-halting all slots to prevent "
            "OOM kill mid-position (W1.9)",
            rss_kb, self.rss_halt_kb,
        )
        for slot in slots:
            self._latch_kill_switch(slot)
        # Publish once so AlertRules fires a Telegram alert. account_alias=""
        # keeps the event cross-slot (no per-slot prefix) — RSS is a
        # process-wide condition, not per-slot.
        await self.bus.publish(MemoryHalt(
            ts_ns=self._now_ns(),
            rss_kb=rss_kb,
            ceiling_kb=self.rss_halt_kb,
        ))

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
            # Await the cancelled futures so their CancelledError is observed
            # (no "Task was destroyed but it is pending" warnings under load).
            await asyncio.gather(dirty, stop, return_exceptions=True)
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
