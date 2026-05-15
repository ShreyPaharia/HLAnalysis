from __future__ import annotations

import asyncio
import signal
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable

import aiohttp
from loguru import logger

from ..adapters.base import VenueAdapter
from ..config import Subscription
from ..events import NormalizedEvent
from .config import DeployConfig, StrategyConfig
from .event_bus import EventBus
from .hl_client import HLClient
from .market_state import MarketState
from .reconcile import Reconciler
from .restart_drift import RestartDriftGate
from .risk import RiskGate
from .risk_events import (
    DailyLossHalt, KillSwitchActivated, NewQuestion, StaleDataHalt, StopLossTriggered,
)
from .router import Router
from .scanner import Scanner
from .state import StateDAL
from ..alerts.rules import AlertRules
from ..alerts.telegram import TelegramClient
from ..strategy.late_resolution import (
    LateResolutionConfig, LateResolutionStrategy,
)


def build_late_resolution_config(cfg: StrategyConfig) -> LateResolutionConfig:
    """Build the strategy runtime config from a loaded StrategyConfig.

    Shared by EngineRuntime (live) and replay CLI. `getattr` defaults let
    older YAMLs without the safety-gate fields keep loading.
    """
    d = cfg.defaults
    return LateResolutionConfig(
        tte_min_seconds=d.tte_min_seconds, tte_max_seconds=d.tte_max_seconds,
        price_extreme_threshold=d.price_extreme_threshold,
        distance_from_strike_usd_min=d.distance_from_strike_usd_min,
        vol_max=d.vol_max, max_position_usd=d.max_position_usd,
        # LateResolutionConfig.stop_loss_pct is a non-Optional float; the
        # strategy treats values ≥1e8 as "disabled" (matches build_v1_late_resolution
        # in strategy/late_resolution.py). Map None -> sentinel here.
        stop_loss_pct=1e9 if d.stop_loss_pct is None else d.stop_loss_pct,
        max_strike_distance_pct=cfg.global_.max_strike_distance_pct,
        min_recent_volume_usd=cfg.global_.min_recent_volume_usd,
        stale_data_halt_seconds=cfg.global_.stale_data_halt_seconds,
        price_extreme_max=getattr(d, "price_extreme_max", 1.0),
        min_safety_d=getattr(d, "min_safety_d", 0.0),
        vol_lookback_seconds=getattr(d, "vol_lookback_seconds", 1800),
        exit_safety_d=getattr(d, "exit_safety_d", 0.0),
        vol_ewma_lambda=getattr(d, "vol_ewma_lambda", 0.0),
    )


@dataclass
class EngineRuntime:
    strategy_cfg: StrategyConfig
    deploy_cfg: DeployConfig
    adapter_factory: Callable[[], VenueAdapter]
    subscriptions: list[Subscription]
    # Optional dependency injections so tests can swap real components for fakes.
    hl_client_factory: Callable[[bool], HLClient] | None = None
    telegram_factory: Callable[[aiohttp.ClientSession], TelegramClient] | None = None
    market_state: MarketState = field(default_factory=MarketState)
    bus: EventBus = field(default_factory=EventBus)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    last_reconcile_ns: int = 0
    heartbeat_interval_s: float = 30.0
    # Observability counters — reset to 0 at process start, monotonic thereafter.
    events_ingested: int = 0
    scans_completed: int = 0
    decisions_emitted: int = 0  # non-HOLD decisions handed to the router

    async def run(self) -> None:
        # 1) Build collaborators
        dal = StateDAL(Path(self.deploy_cfg.state_db_path))
        dal.run_migrations()

        hl = self._make_hl_client()
        block_path = Path(self.deploy_cfg.kill_switch_path).parent / "restart_blocked"
        gate = RestartDriftGate(dal=dal, block_path=block_path)

        async with aiohttp.ClientSession() as http:
            tg = self._make_telegram(http)
            rules = AlertRules(bus=self.bus, telegram=tg)
            risk = RiskGate(self.strategy_cfg)
            router = Router(dal=dal, gate=risk, bus=self.bus, hl=hl,
                            strategy_cfg=self.strategy_cfg)

            # Strategy runtime config from the matched defaults entry
            rcfg = build_late_resolution_config(self.strategy_cfg)
            strategy = LateResolutionStrategy(rcfg)

            # 2) Restart-drift gate
            self.last_reconcile_ns = self._now_ns()
            drift_res = gate.run(
                venue_open=hl.open_orders(),
                venue_state=hl.clearinghouse_state(),
                fills_lookup=lambda c: hl.user_fills(),
                now_ns=self.last_reconcile_ns,
            )
            for ev in drift_res.drift_events:
                await self.bus.publish(ev)
            if drift_res.blocked:
                logger.warning("RESTART BLOCKED — scanner suspended\n{}", drift_res.summary)
                await tg.send(f"*RESTART BLOCKED*\n```\n{drift_res.summary[:3500]}\n```")

            scanner = Scanner(
                strategy=strategy, cfg=self.strategy_cfg,
                market_state=self.market_state, dal=dal,
                kill_switch_path=Path(self.deploy_cfg.kill_switch_path),
                last_reconcile_ns=self.last_reconcile_ns,
            )

            # 3) Wire signal handlers
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                with suppress(NotImplementedError):
                    loop.add_signal_handler(sig, self.stop_event.set)

            # 4) Spawn tasks
            alerts_sub = self.bus.subscribe()
            tasks = [
                asyncio.create_task(self._ingest_loop(dal)),
                asyncio.create_task(self._reconcile_loop(hl, dal)),
                asyncio.create_task(self._continuous_checks_loop(dal, risk, router)),
                asyncio.create_task(rules.run(alerts_sub)),
                asyncio.create_task(self._heartbeat_loop(dal)),
            ]
            if not drift_res.blocked:
                tasks.append(asyncio.create_task(self._scan_loop(scanner, router)))

            await self.stop_event.wait()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    # ---- task bodies ----

    async def _ingest_loop(self, dal: StateDAL) -> None:
        adapter = self.adapter_factory()
        # In-process cache to avoid a DB hit on every QuestionMetaEvent;
        # the SeenQuestion table is the source of truth across restarts.
        seen_questions: set[int] = set()
        from ..events import QuestionMetaEvent
        try:
            async for ev in adapter.stream(self.subscriptions):
                if self.stop_event.is_set():
                    return
                self.market_state.apply(ev)
                self.events_ingested += 1
                if isinstance(ev, QuestionMetaEvent):
                    qidx = ev.question_idx
                    if qidx not in seen_questions:
                        seen_questions.add(qidx)
                        if not dal.has_seen_question(qidx):
                            qv = self.market_state.question(qidx)
                            if qv is not None:
                                from ..strategy.render import question_description
                                now_ns = self._now_ns()
                                await self.bus.publish(NewQuestion(
                                    ts_ns=now_ns,
                                    question_idx=qidx,
                                    klass=qv.klass,
                                    description=question_description(qv),
                                    expiry_ns=qv.expiry_ns,
                                    leg_count=len(qv.leg_symbols),
                                ))
                                dal.mark_question_seen(qidx, now_ns=now_ns)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("adapter ingest crashed")

    async def _scan_loop(self, scanner: Scanner, router: Router) -> None:
        while not self.stop_event.is_set():
            try:
                now = self._now_ns()
                scanner.last_reconcile_ns = self.last_reconcile_ns
                for sd in scanner.scan(now_ns=now):
                    self.decisions_emitted += 1
                    await router.handle(sd.decision, inputs=sd.inputs, now_ns=now)
                self.scans_completed += 1
            except Exception:
                logger.exception("scan tick crashed")
            await self._sleep_or_stop(1.0)

    async def _heartbeat_loop(self, dal: StateDAL) -> None:
        """Periodic visibility into engine health. Without this the engine is
        silent on calm markets (strategy gates filter out almost everything;
        adapter doesn't log INFO during normal flow). One line every N seconds
        is the cheapest way to confirm the loops are alive."""
        prev_events = 0
        prev_scans = 0
        while not self.stop_event.is_set():
            await self._sleep_or_stop(self.heartbeat_interval_s)
            if self.stop_event.is_set():
                return
            try:
                btc_mark = self.market_state.last_mark("BTC")
                n_questions = len(self.market_state.all_questions())
                n_positions = len(dal.all_positions())
                n_live = len(dal.live_orders())
                d_events = self.events_ingested - prev_events
                d_scans = self.scans_completed - prev_scans
                prev_events = self.events_ingested
                prev_scans = self.scans_completed
                logger.info(
                    "heartbeat events={} (+{}) scans={} (+{}) decisions={} | "
                    "btc={} questions={} positions={} live_orders={}",
                    self.events_ingested, d_events,
                    self.scans_completed, d_scans,
                    self.decisions_emitted,
                    f"${btc_mark:.2f}" if btc_mark else "none",
                    n_questions, n_positions, n_live,
                )
            except Exception:
                logger.exception("heartbeat crashed")

    async def _reconcile_loop(self, hl: HLClient, dal: StateDAL) -> None:
        interval = self.strategy_cfg.global_.reconcile_interval_seconds
        while not self.stop_event.is_set():
            await self._sleep_or_stop(float(interval))
            if self.stop_event.is_set():
                return
            try:
                now = self._now_ns()
                rec = Reconciler(dal, fills_lookup=lambda c: hl.user_fills())
                res = rec.run(
                    venue_open=hl.open_orders(),
                    venue_state=hl.clearinghouse_state(),
                    now_ns=now,
                )
                for ev in res.drift_events:
                    await self.bus.publish(ev)
                for cloid, symbol in res.orphans_to_cancel:
                    hl.cancel(cloid=cloid, symbol=symbol)
                self.last_reconcile_ns = now
            except Exception:
                logger.exception("reconcile crashed")

    async def _continuous_checks_loop(
        self, dal: StateDAL, risk: RiskGate, router: Router,
    ) -> None:
        kill_path = Path(self.deploy_cfg.kill_switch_path)
        while not self.stop_event.is_set():
            try:
                now = self._now_ns()
                # Kill switch
                if risk.kill_switch_active(kill_path):
                    await self.bus.publish(KillSwitchActivated(
                        ts_ns=now, path=str(kill_path),
                    ))
                    self.stop_event.set()
                    return

                # Stop-loss enforcer
                positions_db = dal.all_positions()
                if positions_db:
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
                    breached = risk.breached_stops(sps, books)
                    for sp in breached:
                        await self.bus.publish(StopLossTriggered(
                            ts_ns=now, question_idx=sp.question_idx,
                            symbol=sp.symbol, qty=sp.qty,
                            trigger_px=sp.stop_loss_price,
                        ))
                        # Force-exit via router using a manual EXIT decision
                        from ..strategy.types import (
                            Action, Decision, OrderIntent,
                        )
                        b = books.get(sp.symbol)
                        if b is None or b.bid_px is None:
                            continue
                        intent = OrderIntent(
                            question_idx=sp.question_idx, symbol=sp.symbol,
                            side="sell" if sp.qty > 0 else "buy",
                            size=abs(sp.qty), limit_price=b.bid_px,
                            # Pure-uuid hex cloid; HLClient maps hla-{uuid} → 32-char
                            # hex via Cloid.from_str. Earlier 'hla-stop-{q}-{ns}'
                            # produced non-hex characters (s/t/o/p/-) and silently
                            # crashed Cloid.from_str → stop-loss never sent in live.
                            cloid=f"hla-{uuid.uuid4()}",
                            time_in_force="ioc", reduce_only=True,
                        )
                        # Inputs stub for the gate (size_invalid is the only check
                        # that fires on exits; everything else short-circuits).
                        from .risk import RiskInputs
                        inp = RiskInputs(
                            question=self.market_state.question(sp.question_idx) or _stub_question(sp),
                            question_fields={}, reference_price=0.0, book=b,
                            recent_volume_usd=0.0, positions=sps,
                            live_orders_total_notional=0.0,
                            realized_pnl_today=0.0, kill_switch_active=False,
                            last_reconcile_ns=self.last_reconcile_ns, now_ns=now,
                        )
                        await router.handle(
                            Decision(action=Action.EXIT, intents=(intent,)),
                            inputs=inp, now_ns=now,
                        )

                # Daily loss
                from datetime import datetime, timezone
                midnight_ns = int(datetime.fromtimestamp(now / 1e9, tz=timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0,
                ).timestamp() * 1_000_000_000)
                pnl = dal.realized_pnl_since(midnight_ns)
                if pnl < -self.strategy_cfg.global_.daily_loss_cap_usd:
                    await self.bus.publish(DailyLossHalt(
                        ts_ns=now, realized_pnl=pnl,
                        cap=self.strategy_cfg.global_.daily_loss_cap_usd,
                    ))
                    self.stop_event.set()
                    return

                # Stale-data halt is per-trade; we also surface it as an alert here.
                # Iterate held position legs (avoids spam on unsubscribed markets).
                # Skip legs whose underlying question has settled — the venue
                # legitimately stops quoting once a market is resolved, and the
                # close happens on the SettlementEvent path (router._close_settled).
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
                    for sym in risk.stale_books(books_only_held, now_ns=now):
                        b = books_only_held[sym]
                        await self.bus.publish(StaleDataHalt(
                            ts_ns=now, symbol=sym,
                            age_seconds=(now - b.last_l2_ts_ns) / 1e9,
                        ))
            except Exception:
                logger.exception("continuous checks crashed")
            await self._sleep_or_stop(1.0)

    # ---- helpers ----

    async def _sleep_or_stop(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self.stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    def _make_hl_client(self) -> HLClient:
        if self.hl_client_factory is not None:
            return self.hl_client_factory(self.strategy_cfg.paper_mode)
        return HLClient(
            account_address=self.deploy_cfg.hl.account_address,
            api_secret_key=self.deploy_cfg.hl.api_secret_key,
            base_url=self.deploy_cfg.hl.base_url,
            paper_mode=self.strategy_cfg.paper_mode,
        )

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
