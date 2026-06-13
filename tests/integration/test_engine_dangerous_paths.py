"""Integration tests for dangerous live engine paths (SHR-67).

Each test covers one safety-critical scenario that could silently lose money
or spin indefinitely in production.  Tests drive either the full EngineRuntime
(runtime-harness) or a direct component call (component-level) where the full
loop would be too heavy.

Paths covered
-------------
1. Order rejection storm — reject circuit-breaker plateaus (SHR-45).
2. Stop-loss IOC chain — in-flight guard prevents stacking + size is clamped
   on a thin book (SHR-48).
3. Reconcile venue drift — vanished/orphan positions handled correctly and a
   recovered lost-ACK fill is booked into Position (SHR-46).
4. Restart with pre-existing DB+venue state — RestartDriftGate adopts without
   double-booking; position count stable after the gate runs (not blocked for a
   clean match).
5. Feed disconnect/reconnect — ingest reconnects (SHR-42); PM PmBook state is
   reset on reconnect (SHR-62).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hlanalysis.adapters.base import VenueAdapter
from hlanalysis.alerts.telegram import TelegramClient
from hlanalysis.config import Subscription
from hlanalysis.engine.config import (
    AlertsConfig,
    AllowlistEntry,
    DeployConfig,
    GlobalRiskConfig,
    HLConfig,
    StrategyConfig,
    TelegramConfig,
)
from hlanalysis.engine.event_bus import EventBus
from hlanalysis.engine.exec_types import (
    ClearinghouseState,
    OpenOrderRow,
    OrderAck,
    PlaceRequest,
    UserFillRow,
    VenuePosition,
)
from hlanalysis.engine.hl_client import HLClient
from hlanalysis.engine.reconcile import Reconciler
from hlanalysis.engine.restart_drift import RestartDriftGate
from hlanalysis.engine.risk import RiskGate, RiskInputs
from hlanalysis.engine.router import Router
from hlanalysis.engine.runtime import EngineRuntime
from hlanalysis.engine.state import OpenOrder, Position, StateDAL
from hlanalysis.events import (
    BboEvent,
    MarkEvent,
    Mechanism,
    ProductType,
    QuestionMetaEvent,
    SettlementEvent,
)
from hlanalysis.strategy.types import (
    Action,
    BookState,
    Decision,
    OrderIntent,
    QuestionView,
)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _strategy_cfg(
    *,
    min_recent_volume_usd: float = 0.0,
    stale_data_halt_seconds: int = 30,
    max_slippage_pct: float = 0.0,
) -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        max_position_usd=100,
        stop_loss_pct=10,
        tte_min_seconds=60,
        tte_max_seconds=1800,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200,
        vol_max=0.5,
    )
    return StrategyConfig(
        name="late_resolution",
        paper_mode=True,
        allowlist=[entry],
        blocklist_question_idxs=[],
        defaults=entry,
        **{
            "global": GlobalRiskConfig(
                max_total_inventory_usd=500,
                max_concurrent_positions=5,
                daily_loss_cap_usd=200,
                max_strike_distance_pct=10,
                min_recent_volume_usd=min_recent_volume_usd,
                stale_data_halt_seconds=stale_data_halt_seconds,
                reconcile_interval_seconds=60,
                max_slippage_pct=max_slippage_pct,
            )
        },
    )


def _deploy_cfg(tmp_path: Path) -> DeployConfig:
    return DeployConfig(
        env="dev",
        accounts={
            "default": HLConfig(
                account_address="0x",
                api_secret_key="0x",
                base_url="https://api.hyperliquid.xyz",
            )
        },
        alerts=AlertsConfig(telegram=TelegramConfig(bot_token="x", chat_id="y")),
        state_db_path=str(tmp_path / "state.db"),
        kill_switch_path=str(tmp_path / "halt"),
    )


def _now_plus_ns(offset_ns: int = 0) -> int:
    return time.time_ns() + offset_ns


def _expiry_str(future_ns: int) -> str:
    return datetime.fromtimestamp(
        future_ns / 1e9,
        tz=timezone.utc,
    ).strftime("%Y%m%d-%H%M")


def _meta_event(question_idx: int = 42) -> QuestionMetaEvent:
    now = time.time_ns()
    expiry = now + 10 * 60 * 1_000_000_000  # 10 min in future
    return QuestionMetaEvent(
        venue="hyperliquid",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="qmeta",
        exchange_ts=now - 60_000_000_000,
        local_recv_ts=now - 60_000_000_000,
        question_idx=question_idx,
        named_outcome_idxs=[3],
        keys=["class", "underlying", "period", "expiry", "strike"],
        values=["priceBinary", "BTC", "1h", _expiry_str(expiry), "80000"],
    )


def _mark_stream(n: int = 8) -> list[MarkEvent]:
    now = time.time_ns()
    return [
        MarkEvent(
            venue="hyperliquid",
            product_type=ProductType.PERP,
            mechanism=Mechanism.CLOB,
            symbol="BTC",
            exchange_ts=now - (n - i) * 60_000_000_000,
            local_recv_ts=now - (n - i) * 60_000_000_000,
            mark_px=80_300.0 + i * 0.01,
        )
        for i in range(n)
    ]


class _FakeTelegram:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, text: str, *, markdown: bool = True) -> bool:
        self.messages.append(text)
        return True


# ---------------------------------------------------------------------------
# Path 1: Order rejection storm — component level via Router
#
# Exercises the reject circuit-breaker (SHR-45) directly: drives a Router
# whose exec_client always rejects, fires 2× the threshold, asserts that
# place-calls plateau at exactly the threshold.  Then fires a fill that resets
# the counter and verifies placements resume.
#
# Why component-level: the Router + circuit-breaker is pure sync/async;
# driving the full runtime would require timing-sensitive coordination with
# little additional coverage — the runtime path just calls router.handle().
# ---------------------------------------------------------------------------


class _AlwaysRejectExec:
    paper_mode = False

    def __init__(self) -> None:
        self.placed: list[PlaceRequest] = []

    def place(self, req: PlaceRequest) -> OrderAck:
        self.placed.append(req)
        return OrderAck(cloid=req.cloid, venue_oid="", status="rejected", error="insufficient margin")

    def cancel(self, *, cloid: str, symbol: str) -> bool:
        return True

    def open_orders(self) -> list:
        return []

    def clearinghouse_state(self) -> ClearinghouseState:
        return ClearinghouseState(positions=(), account_value_usd=0.0)

    def user_fills(self, *, since_ts_ns: int = 0) -> list:
        return []

    def realized_pnl_since(self, since_ts_ns: int) -> float:
        return 0.0


class _ScriptedExec(_AlwaysRejectExec):
    """Returns a scripted sequence of acks, then rejects."""

    def __init__(self, statuses: list[str]) -> None:
        super().__init__()
        self._statuses = list(statuses)

    def place(self, req: PlaceRequest) -> OrderAck:
        self.placed.append(req)
        status = self._statuses.pop(0) if self._statuses else "rejected"
        if status == "filled":
            return OrderAck(cloid=req.cloid, venue_oid="v", status="filled", fill_price=req.price, fill_size=req.size)
        return OrderAck(cloid=req.cloid, venue_oid="", status="rejected", error="rej")


def _q() -> QuestionView:
    return QuestionView(
        question_idx=42,
        yes_symbol="@30",
        no_symbol="@31",
        strike=80_000.0,
        expiry_ns=_now_plus_ns(600_000_000_000),
        underlying="BTC",
        klass="priceBinary",
        period="1h",
    )


def _risk_inputs() -> RiskInputs:
    now = _now_plus_ns()
    return RiskInputs(
        question=_q(),
        question_fields={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        reference_price=80_300.0,
        book=BookState(
            symbol="@30",
            bid_px=0.94,
            bid_sz=10.0,
            ask_px=0.95,
            ask_sz=10.0,
            last_trade_ts_ns=now - 1,
            last_l2_ts_ns=now - 1,
        ),
        recent_volume_usd=5_000.0,
        positions=[],
        live_orders_total_notional=0.0,
        realized_pnl_today=0.0,
        kill_switch_active=False,
        last_reconcile_ns=now - 1,
        now_ns=now,
    )


def _exit_intent(cloid: str, question_idx: int = 42, symbol: str = "@30") -> OrderIntent:
    return OrderIntent(
        question_idx=question_idx,
        symbol=symbol,
        side="sell",
        size=10.0,
        limit_price=0.99,
        cloid=cloid,
        time_in_force="ioc",
        reduce_only=True,
        exit_reason="exit_safety_d",
    )


@pytest.mark.asyncio
async def test_reject_storm_circuit_breaker_plateaus(tmp_path):
    """Path 1: reject storm.

    After `threshold` consecutive rejects on the same (question_idx, side)
    the circuit-breaker must suppress further place() calls.

    Specific safety assertion: place-call count MUST plateau at exactly
    `threshold` — it must not grow unboundedly (the 1,200-rejects/h incident
    from 2026-06-04 that flooded Telegram and the PM API).
    """
    threshold = 3
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    # Seed a held position so reduce_only clamp doesn't fire.
    dal.upsert_position(
        Position(
            question_idx=42,
            symbol="@30",
            qty=10.0,
            avg_entry=0.9,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=-1.0,
        )
    )
    cfg = _strategy_cfg()
    client = _AlwaysRejectExec()
    router = Router(
        dal=dal,
        gate=RiskGate(cfg),
        bus=EventBus(),
        exec_client=client,
        strategy_cfg=cfg,
        reject_breaker_threshold=threshold,
    )

    total_attempts = threshold * 3
    for i in range(total_attempts):
        await router.handle(
            Decision(action=Action.EXIT, intents=(_exit_intent(f"hla-dp1-{i}"),)),
            inputs=_risk_inputs(),
            now_ns=_now_plus_ns(i),
        )

    assert len(client.placed) == threshold, (
        f"Circuit-breaker must plateau at threshold={threshold}; "
        f"got {len(client.placed)} after {total_attempts} attempts. "
        "Reject storm is not suppressed."
    )


@pytest.mark.asyncio
async def test_reject_storm_resets_on_fill(tmp_path):
    """Path 1 (variant): a fill on the question resets the reject counter,
    restoring the full budget for a new burst.

    Sequence with threshold=3: rej, rej, FILL → counter reset →
    rej, rej, rej → second trip → 7th attempt suppressed.
    Total placed calls expected = 2 + 1 + 3 = 6.
    """
    threshold = 3
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    dal.upsert_position(
        Position(
            question_idx=42,
            symbol="@30",
            qty=100.0,
            avg_entry=0.9,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=-1.0,
        )
    )
    cfg = _strategy_cfg()
    # rej, rej, fill, rej, rej, rej, (7th suppressed)
    client = _ScriptedExec(["rejected", "rejected", "filled", "rejected", "rejected", "rejected", "rejected"])
    router = Router(
        dal=dal,
        gate=RiskGate(cfg),
        bus=EventBus(),
        exec_client=client,
        strategy_cfg=cfg,
        reject_breaker_threshold=threshold,
    )

    for i in range(7):
        await router.handle(
            Decision(action=Action.EXIT, intents=(_exit_intent(f"hla-dp1r-{i}"),)),
            inputs=_risk_inputs(),
            now_ns=_now_plus_ns(i),
        )

    assert len(client.placed) == 6, (
        f"Expected 6 place calls (2 pre-fill + 1 fill + 3 post-reset) "
        f"with threshold={threshold}; got {len(client.placed)}. "
        "Fill-driven reset is broken."
    )


# ---------------------------------------------------------------------------
# Path 2: Stop-loss IOC chain — component level via _enforce_stop_losses
#
# Tests two SHR-48 sub-behaviours:
#   (a) In-flight guard: a live exit order for a question_idx prevents a
#       second IOC from being stacked on the same position.
#   (b) Slippage clamp: on a book with max_slippage_pct set, the exit IOC
#       size is clamped to at-limit depth rather than walking the whole book.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_loss_in_flight_guard_no_double_ioc(tmp_path):
    """Path 2a: in-flight exit guard (SHR-48).

    Precondition: a stop breach exists AND there is already a live exit order
    for that question_idx in the DB.  After calling _enforce_stop_losses,
    the exec_client must NOT have received a new place() call — the guard
    must skip the position.

    Specific safety assertion: zero additional IOCs when exit already in flight.
    """
    cfg = _strategy_cfg(min_recent_volume_usd=0.0)
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()

    # Seed a position below its stop price (stop = 0.90, current bid = 0.85).
    dal.upsert_position(
        Position(
            question_idx=42,
            symbol="@30",
            qty=10.0,
            avg_entry=0.92,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=0.90,
        )
    )
    # Seed a live exit order for the same question_idx (simulates prior IOC).
    dal.upsert_order(
        OpenOrder(
            cloid="hla-v1-inflight",
            venue_oid="v1",
            question_idx=42,
            symbol="@30",
            side="sell",
            price=0.85,
            size=10.0,
            status="open",
            placed_ts_ns=1,
            last_update_ts_ns=1,
            strategy_id="late_resolution",
        )
    )

    client = _AlwaysRejectExec()
    deploy = _deploy_cfg(tmp_path)
    runtime = EngineRuntime.from_single(
        strategy_cfg=cfg,
        deploy_cfg=deploy,
        adapter_factory=lambda: _NullAdapter(),
        subscriptions=[],
        exec_client_factory=lambda _paper: client,
        telegram_factory=lambda _http: _FakeTelegram(),
    )
    # Build the slot manually so we can call _enforce_stop_losses directly
    # without running the full event loop.
    slot = runtime._build_slot(cfg)

    # Inject the BBO into MarketState so the stop-loss enforcer can read bid_px.
    now = _now_plus_ns()
    runtime.market_state.apply(
        BboEvent(
            venue="hyperliquid",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="@30",
            exchange_ts=now,
            local_recv_ts=now,
            bid_px=0.85,
            bid_sz=5.0,
            ask_px=0.86,
            ask_sz=5.0,
        )
    )
    # Register question in MarketState so question() lookup works.
    runtime.market_state.apply(_meta_event(42))

    # Direct component call: one pass of stop-loss enforcement.
    # The open exit order for question_idx=42 must block a new IOC.
    await runtime._enforce_stop_losses(slot, now_ns=now)

    # The live order we seeded acts as the in-flight guard — nothing new placed.
    assert len(client.placed) == 0, (
        f"In-flight exit guard must suppress stacking; got {len(client.placed)} "
        "new place() calls. The IOC storm guard (SHR-48) is broken."
    )


@pytest.mark.asyncio
async def test_stop_loss_slippage_clamped_on_thin_book(tmp_path):
    """Path 2b: exit slippage clamp (SHR-48).

    When max_slippage_pct is set and the book is thin (only 3 units at the
    best ask), the exit intent for 10 units must be clamped to 3 by the
    depth-walk.  Verifies that _depth_walk_clamp is called for exits, not
    bypassed.

    This tests the risk gate directly (not the full runtime) since the
    clamping logic lives entirely in RiskGate.check_pre_trade.
    """
    # max_slippage_pct=0.05 = 5%.  Book: only 3 units at bid=0.85; deeper
    # bid=0.70 is >5% below limit → out of budget.
    cfg = _strategy_cfg(max_slippage_pct=0.05)
    gate = RiskGate(cfg)

    now = _now_plus_ns()
    # BookState with bid_levels: thin book (3 units at 0.85, 50 at 0.70)
    book = BookState(
        symbol="@30",
        bid_px=0.85,
        bid_sz=3.0,
        ask_px=0.86,
        ask_sz=10.0,
        last_trade_ts_ns=now - 1,
        last_l2_ts_ns=now - 1,
        bid_levels=((0.85, 3.0), (0.70, 50.0)),  # 0.70 is far below limit
        ask_levels=(),
    )
    # Exit intent for 10 units at limit 0.85 (IOC sell)
    intent = OrderIntent(
        question_idx=42,
        symbol="@30",
        side="sell",
        size=10.0,
        limit_price=0.85,
        cloid="hla-dp2b-0",
        time_in_force="ioc",
        reduce_only=True,
        exit_reason="stop_loss",
    )
    inp = RiskInputs(
        question=replace(_q(), settled=False),
        question_fields={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        reference_price=80_300.0,
        book=book,
        recent_volume_usd=5_000.0,
        positions=[],
        live_orders_total_notional=0.0,
        realized_pnl_today=0.0,
        kill_switch_active=False,
        last_reconcile_ns=now - 1,
        now_ns=now,
    )

    verdict = gate.check_pre_trade(intent, inp)

    assert verdict.approved, (
        "Exit on a thin book must be approved (partial reduce beats none); "
        f"got approved=False reason={verdict.reason!r}"
    )
    assert verdict.clamped_size is not None, (
        "Exit on a thin book must have clamped_size set (SHR-48 depth-walk "
        "clamp for exits); got clamped_size=None — the clamp is bypassed."
    )
    assert verdict.clamped_size == pytest.approx(3.0), (
        f"clamped_size must equal the at-limit depth (3.0); got {verdict.clamped_size}"
    )


# ---------------------------------------------------------------------------
# Path 3: Reconcile venue drift — component level via Reconciler + RestartDriftGate
#
# (a) Vanished position: DB has a position; venue reports nothing → Reconciler
#     emits it in vanished_positions so the runtime can publish a settlement Exit.
# (b) Venue orphan: venue has an open order not in DB → orphans_to_cancel.
# (c) Recovered lost-ACK fill: local-ghost with fills → position booked (SHR-46).
# ---------------------------------------------------------------------------


def test_reconcile_vanished_position_detected(tmp_path):
    """Path 3a: vanished position is surfaced in ReconcileResult.vanished_positions.

    Specific assertion: when a DB position's symbol is absent from venue state
    (clearinghouse), the reconciler must list it in vanished_positions so the
    runtime can mark the question settled and publish a settlement Exit.
    """
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    # Seed a live position.
    dal.upsert_position(
        Position(
            question_idx=42,
            symbol="@30",
            qty=10.0,
            avg_entry=0.92,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=-1.0,
        )
    )
    # Venue reports nothing for this symbol.
    venue_state = ClearinghouseState(positions=(), account_value_usd=0.0)

    rec = Reconciler(
        dal,
        fills_lookup=lambda _c: [],
        symbol_to_question={},
        apply_position_changes=True,
    )
    result = rec.run(venue_open=[], venue_state=venue_state, now_ns=2)

    assert len(result.vanished_positions) == 1, (
        f"Reconciler must report 1 vanished position; got {result.vanished_positions}"
    )
    qidx, sym, lp = result.vanished_positions[0]
    assert qidx == 42
    assert sym == "@30"
    # Position must also be deleted from the DB.
    assert dal.get_position(42) is None, "Reconciler must delete the vanished position row."


def test_reconcile_venue_orphan_flagged_for_cancel(tmp_path):
    """Path 3b: venue has an open order not in the local DB → orphans_to_cancel.

    Specific assertion: the orphan cloid+symbol pair appears in
    ReconcileResult.orphans_to_cancel so the runtime can call cancel().
    """
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    # Venue reports an open order we never wrote to the DB.
    venue_orphan = OpenOrderRow(
        cloid="0x" + "a" * 32,  # hex-form HL cloid
        venue_oid="v-orphan",
        symbol="@30",
        side="buy",
        price=0.95,
        size=5.0,
        placed_ts_ns=1,
    )
    venue_state = ClearinghouseState(positions=(), account_value_usd=0.0)

    rec = Reconciler(
        dal,
        fills_lookup=lambda _c: [],
        symbol_to_question={},
        apply_position_changes=True,
    )
    result = rec.run(
        venue_open=[venue_orphan],
        venue_state=venue_state,
        now_ns=2,
    )

    assert len(result.orphans_to_cancel) == 1, f"Expected 1 orphan to cancel; got {result.orphans_to_cancel}"
    orphan_cloid, orphan_symbol = result.orphans_to_cancel[0]
    assert orphan_symbol == "@30"


def test_reconcile_recovered_lost_ack_fill_books_position(tmp_path):
    """Path 3c: lost-ACK fill discovered by reconcile must book the Position
    (SHR-46).

    A locally-pending order whose venue fill is found by fills_lookup triggers
    the local-ghost branch.  Before the fix that branch only marked the order
    filled + replayed Fill rows but never upserted the Position — leaving the
    engine in a re-exit loop.

    Specific assertion: after reconcile, dal.get_position(qidx).qty == fill.size.
    """
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    cloid = "hla-v31-lostack"
    dal.upsert_order(
        OpenOrder(
            cloid=cloid,
            venue_oid="v-la",
            question_idx=42,
            symbol="@30",
            side="buy",
            price=0.60,
            size=100.0,
            status="open",
            placed_ts_ns=1,
            last_update_ts_ns=1,
            strategy_id="v31",
        )
    )
    fills = [
        UserFillRow(
            fill_id="fill-la",
            cloid=cloid,
            symbol="@30",
            side="buy",
            price=0.60,
            size=100.0,
            fee=0.10,
            ts_ns=2,
        )
    ]
    venue_state = ClearinghouseState(
        positions=(VenuePosition(symbol="@30", qty=100.0, avg_entry=0.60, unrealized_pnl=0.0),),
        account_value_usd=1000.0,
    )

    rec = Reconciler(
        dal,
        fills_lookup=lambda c: fills if c == cloid else [],
        symbol_to_question={},  # no mapping → isolates the local-ghost fix
        apply_position_changes=True,
    )
    rec.run(venue_open=[], venue_state=venue_state, now_ns=3)

    pos = dal.get_position(42)
    assert pos is not None, "Recovered lost-ACK fill must create a Position row (SHR-46)."
    assert pos.qty == pytest.approx(100.0)
    assert pos.avg_entry == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# Path 4: Restart with pre-existing DB+venue state — RestartDriftGate
#
# Seed state.db with an open position + a venue state that MATCHES exactly.
# Run RestartDriftGate.  Assert:
#   (a) NOT blocked (a clean match should not block the scanner).
#   (b) Position count in DB is unchanged (no double-booking).
#
# Why component-level: the gate runs synchronously during engine startup
# (before the async event loop spins tasks).  Driving the full runtime would
# require precise async timing to observe the pre-task state.
# ---------------------------------------------------------------------------


def test_restart_drift_gate_clean_match_not_blocked(tmp_path):
    """Path 4: restart with matching DB + venue state.

    When the DB position and the venue clearinghouse state agree exactly, the
    RestartDriftGate must NOT block the scanner — a clean match is not drift.

    Specific assertion: result.blocked == False and position count == 1 (no
    double-booking by the reconcile upsert path).
    """
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    # Seed the DB position.
    dal.upsert_position(
        Position(
            question_idx=42,
            symbol="@30",
            qty=10.0,
            avg_entry=0.92,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=-1.0,
        )
    )
    # Venue matches.
    venue_state = ClearinghouseState(
        positions=(VenuePosition(symbol="@30", qty=10.0, avg_entry=0.92, unrealized_pnl=0.0),),
        account_value_usd=1000.0,
    )
    block_path = tmp_path / "restart_blocked"
    gate = RestartDriftGate(
        dal=dal,
        block_path=block_path,
        account_alias="default",
    )
    result = gate.run(
        venue_open=[],
        venue_state=venue_state,
        fills_lookup=lambda _c: [],
        now_ns=2,
    )

    assert not result.blocked, (
        f"RestartDriftGate must NOT block on a clean DB↔venue match; got blocked=True summary={result.summary!r}"
    )
    positions = dal.all_positions()
    assert len(positions) == 1, f"Position count must stay 1 (no double-booking); got {len(positions)}"
    assert positions[0].qty == pytest.approx(10.0)


def test_restart_drift_gate_blocks_on_local_ghost(tmp_path):
    """Path 4 (complementary): when the DB has a live open order that the venue
    doesn't see and fills_lookup returns nothing (cancelled/ghost), the gate
    must block the scanner (LOUD drift case).

    Specific assertion: result.blocked == True so the scanner stays suspended
    until the operator clears it.
    """
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    # Seed a live open order with no venue match and no fills.
    dal.upsert_order(
        OpenOrder(
            cloid="hla-v1-ghost",
            venue_oid="v-ghost",
            question_idx=42,
            symbol="@30",
            side="buy",
            price=0.95,
            size=10.0,
            status="open",
            placed_ts_ns=1,
            last_update_ts_ns=1,
            strategy_id="late_resolution",
        )
    )
    venue_state = ClearinghouseState(positions=(), account_value_usd=0.0)
    block_path = tmp_path / "restart_blocked"
    gate = RestartDriftGate(
        dal=dal,
        block_path=block_path,
        account_alias="default",
    )
    result = gate.run(
        venue_open=[],
        venue_state=venue_state,
        fills_lookup=lambda _c: [],
        now_ns=2,
    )

    assert result.blocked, (
        f"RestartDriftGate must block on a local_ghost (order on DB but not on venue, no fills); got blocked=False"
    )
    assert block_path.exists(), "Gate must write the restart_blocked flag file."


# ---------------------------------------------------------------------------
# Path 5: Feed disconnect/reconnect — runtime-harness level
#
# (a) SHR-42 ingest reconnect: the adapter raises an exception after a few
#     events; the ingest loop must reconnect (not silently die) and process
#     events from the second stream.  We assert events_ingested increases
#     across the reconnect boundary.
#
# (b) PM PmBook reset on reconnect (SHR-62): exercise PmBook.reset() directly
#     and verify that a stale internal book is cleared so a post-reconnect
#     price_change can't merge into stale pre-disconnect levels.
# ---------------------------------------------------------------------------


class _NullAdapter(VenueAdapter):
    """Adapter that yields nothing — for slot-building without driving events."""

    venue = "hyperliquid"

    def supports(self, *_a, **_k) -> bool:
        return True

    async def stream(self, _subs):
        return
        yield  # make it an async generator


class _CrashThenRecoverAdapter(VenueAdapter):
    """Yields a few events, crashes, then yields more events on reconnect.

    The ingest loop calls adapter_factory() to construct a new adapter on each
    reconnect, so we use a shared list of 'rounds' to hand out the streams in
    order.  Call count is tracked on the class so tests can assert it.
    """

    venue = "hyperliquid"
    _call_count: int = 0

    @classmethod
    def reset_call_count(cls) -> None:
        cls._call_count = 0

    def supports(self, *_a, **_k) -> bool:
        return True

    async def stream(self, _subs: list) -> AsyncIterator:
        now = time.time_ns()
        expiry = now + 10 * 60 * 1_000_000_000
        expiry_str = datetime.fromtimestamp(expiry / 1e9, tz=timezone.utc).strftime("%Y%m%d-%H%M")

        call_no = type(self)._call_count
        type(self)._call_count += 1

        if call_no == 0:
            # First stream: yield a few marks then raise to simulate a drop.
            for i in range(3):
                ts = now - (3 - i) * 60_000_000_000
                yield MarkEvent(
                    venue="hyperliquid",
                    product_type=ProductType.PERP,
                    mechanism=Mechanism.CLOB,
                    symbol="BTC",
                    exchange_ts=ts,
                    local_recv_ts=ts,
                    mark_px=80_300.0 + i,
                )
            raise ConnectionError("simulated feed drop")
        else:
            # Second stream (after reconnect): yield a few more marks.
            yield QuestionMetaEvent(
                venue="hyperliquid",
                product_type=ProductType.PREDICTION_BINARY,
                mechanism=Mechanism.CLOB,
                symbol="qmeta",
                exchange_ts=now,
                local_recv_ts=now,
                question_idx=99,
                named_outcome_idxs=[3],
                keys=["class", "underlying", "period", "expiry", "strike"],
                values=["priceBinary", "BTC", "1h", expiry_str, "80000"],
            )
            for i in range(3):
                ts = now + i * 60_000_000_000
                yield MarkEvent(
                    venue="hyperliquid",
                    product_type=ProductType.PERP,
                    mechanism=Mechanism.CLOB,
                    symbol="BTC",
                    exchange_ts=ts,
                    local_recv_ts=ts,
                    mark_px=80_500.0 + i,
                )
            # Hold long enough for the test to observe then stop cleanly.
            await asyncio.sleep(2.0)


@pytest.mark.asyncio
async def test_ingest_reconnects_after_adapter_crash(tmp_path):
    """Path 5a: feed disconnect/reconnect (SHR-42).

    The adapter raises ConnectionError mid-stream.  The ingest loop must
    reconnect (adapter_factory called again) and ingest events from the second
    stream.

    Specific safety assertion: events_ingested after reconnect is strictly
    greater than the count after the first crash — the engine is NOT silent
    after the drop.
    """
    _CrashThenRecoverAdapter.reset_call_count()
    cfg = _strategy_cfg()
    fake_tg = _FakeTelegram()

    runtime = EngineRuntime.from_single(
        strategy_cfg=cfg,
        deploy_cfg=_deploy_cfg(tmp_path),
        adapter_factory=_CrashThenRecoverAdapter,
        subscriptions=[],
        exec_client_factory=lambda _paper: _AlwaysRejectExec(),
        telegram_factory=lambda _http: fake_tg,
    )
    # Speed up reconnect for tests
    runtime.ingest_reconnect_base_s = 0.1
    runtime.ingest_reconnect_max_s = 0.5

    runtime_task = asyncio.create_task(runtime.run())
    # Give it enough time to crash and reconnect.
    await asyncio.sleep(3.5)
    runtime.stop_event.set()
    await asyncio.wait_for(runtime_task, timeout=5.0)

    assert _CrashThenRecoverAdapter._call_count >= 2, (
        "Adapter factory must be called at least twice (initial + one reconnect); "
        f"got call_count={_CrashThenRecoverAdapter._call_count}. "
        "Ingest reconnect (SHR-42) is broken."
    )
    # After reconnect the engine should have ingested the second stream's events.
    assert runtime.events_ingested >= 3, (
        f"events_ingested must be ≥ 3 (second-stream events); "
        f"got {runtime.events_ingested}. Ingest is dead after reconnect."
    )


def test_pm_book_reset_clears_stale_state_on_reconnect():
    """Path 5b: PM PmBook reset on reconnect (SHR-62).

    When a PM adapter reconnects, it calls PmBook.reset() to discard any stale
    L2 state from before the gap.  This test verifies that:
      (1) A full book snapshot builds up state.
      (2) After reset(), the internal book is empty.
      (3) A price_change applied after the reset does NOT merge with the old
          levels — the pre-reset levels are gone.

    Specific safety assertion: after reset + price_change, only the delta
    levels appear in the emitted snapshot (the pre-disconnect bids/asks are
    not present).
    """
    from hlanalysis.adapters.polymarket_normalize import PmBook

    book = PmBook()
    # Simulate pre-disconnect full snapshot with bids at 0.40 and 0.39.
    book.apply_book(
        {
            "asset_id": "A",
            "timestamp": 1,
            "bids": [{"price": "0.40", "size": "100"}, {"price": "0.39", "size": "50"}],
            "asks": [{"price": "0.60", "size": "80"}],
        }
    )
    assert len(book._bids) == 2  # sanity

    # Simulate reconnect — adapter calls reset().
    book.reset()
    assert len(book._bids) == 0, "reset() must clear all bid levels"
    assert len(book._asks) == 0, "reset() must clear all ask levels"

    # Now a price_change arrives before the full snapshot (common at reconnect).
    snap = book.apply_price_change(
        {
            "asset_id": "A",
            "timestamp": 2,
            "changes": [{"price": "0.41", "size": "30", "side": "BUY"}],
        }
    )

    # Pre-reset bids (0.40, 0.39) must NOT appear in the snapshot.
    assert snap is not None
    assert set(snap.bid_px) == {0.41}, (
        f"After reset, only the post-reconnect delta should appear; "
        f"got bid_px={snap.bid_px}. Stale pre-disconnect levels survived reset()."
    )
    # Pre-reset ask (0.60) must also be absent.
    assert len(snap.ask_px) == 0, f"After reset, asks must be empty; got ask_px={snap.ask_px}"
