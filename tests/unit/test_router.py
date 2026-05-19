from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from hlanalysis.engine.config import (
    AllowlistEntry, GlobalRiskConfig, StrategyConfig,
)
from hlanalysis.engine.event_bus import EventBus
from hlanalysis.engine.hl_client import HLClient
from hlanalysis.engine.risk import RiskGate, RiskInputs
from hlanalysis.engine.risk_events import RiskVeto
from hlanalysis.engine.router import Router
from hlanalysis.engine.state import StateDAL
from hlanalysis.strategy.types import (
    Action, BookState, Decision, OrderIntent, QuestionView,
)


def _strategy_cfg() -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        max_position_usd=100, stop_loss_pct=10, tte_min_seconds=60,
        tte_max_seconds=1800, price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200, vol_max=0.5,
    )
    return StrategyConfig(
        name="late_resolution", paper_mode=True,
        allowlist=[entry], blocklist_question_idxs=[],
        defaults=entry,
        **{"global": GlobalRiskConfig(
            max_total_inventory_usd=500, max_concurrent_positions=5,
            daily_loss_cap_usd=200, max_strike_distance_pct=10,
            min_recent_volume_usd=1000, stale_data_halt_seconds=5,
            reconcile_interval_seconds=60,
        )},
    )


def _q() -> QuestionView:
    return QuestionView(
        question_idx=42, yes_symbol="@30", no_symbol="@31",
        strike=80_000.0, expiry_ns=10_000_000_000_000_001 + 600_000_000_000,
        underlying="BTC", klass="priceBinary", period="1h",
    )


def _decision_enter() -> Decision:
    return Decision(
        action=Action.ENTER,
        intents=(OrderIntent(
            question_idx=42, symbol="@30", side="buy", size=10.0,
            limit_price=0.95, cloid="hla-router-1", time_in_force="ioc",
        ),),
    )


def _approval_inputs() -> RiskInputs:
    return RiskInputs(
        question=_q(),
        question_fields={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        reference_price=80_300.0,
        book=BookState(symbol="@30", bid_px=0.94, bid_sz=10.0, ask_px=0.95,
                       ask_sz=10.0, last_trade_ts_ns=10_000_000_000_000_000,
                       last_l2_ts_ns=10_000_000_000_000_000),
        recent_volume_usd=5_000.0,
        positions=[],
        live_orders_total_notional=0.0,
        realized_pnl_today=0.0,
        kill_switch_active=False,
        last_reconcile_ns=10_000_000_000_000_000,
        now_ns=10_000_000_000_000_001,
    )


@pytest.mark.asyncio
async def test_approved_decision_writes_db_row_and_calls_place(tmp_path):
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    sub = bus.subscribe()
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    cfg = _strategy_cfg()
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, hl=client, strategy_cfg=cfg)
    await router.handle(_decision_enter(), inputs=_approval_inputs(), now_ns=2)
    o = dal.get_order("hla-router-1")
    assert o is not None and o.status == "filled"
    assert sub.qsize() >= 1  # an Entry event was published


@pytest.mark.asyncio
async def test_exit_reason_is_threaded_to_exit_event(tmp_path):
    """A reduce-only exit with a strategy-supplied exit_reason must surface that
    reason on the Exit bus event — not the legacy "stop_loss" catch-all. This is
    what makes Telegram alerts distinguish safety_d / edge / time_stop exits
    from a true stop loss."""
    from hlanalysis.engine.risk_events import Entry, Exit
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    sub = bus.subscribe()
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    cfg = _strategy_cfg()
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, hl=client, strategy_cfg=cfg)

    # 1) Open a position.
    await router.handle(_decision_enter(), inputs=_approval_inputs(), now_ns=2)
    open_ev = await asyncio.wait_for(sub.get(), timeout=0.5)
    assert isinstance(open_ev, Entry)

    # 2) Close it with a non-stop_loss reason.
    exit_intent = OrderIntent(
        question_idx=42, symbol="@30", side="sell", size=10.0,
        limit_price=0.95, cloid="hla-router-exit", time_in_force="ioc",
        reduce_only=True, exit_reason="exit_safety_d_5m",
    )
    exit_dec = Decision(action=Action.EXIT, intents=(exit_intent,))
    await router.handle(exit_dec, inputs=_approval_inputs(), now_ns=3)
    exit_ev = await asyncio.wait_for(sub.get(), timeout=0.5)
    assert isinstance(exit_ev, Exit)
    assert exit_ev.reason == "exit_safety_d_5m"


@pytest.mark.asyncio
async def test_reduce_only_without_exit_reason_falls_back_to_stop_loss(tmp_path):
    """Backward compat: an old code path that issues a reduce-only intent
    without setting exit_reason keeps the legacy "stop_loss" label — so any
    pre-existing tests / external observers continue to see the same wire shape."""
    from hlanalysis.engine.risk_events import Entry, Exit
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    sub = bus.subscribe()
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    cfg = _strategy_cfg()
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, hl=client, strategy_cfg=cfg)
    await router.handle(_decision_enter(), inputs=_approval_inputs(), now_ns=2)
    await asyncio.wait_for(sub.get(), timeout=0.5)  # drain entry

    legacy_exit = OrderIntent(
        question_idx=42, symbol="@30", side="sell", size=10.0,
        limit_price=0.95, cloid="hla-router-exit-legacy", time_in_force="ioc",
        reduce_only=True,
    )
    await router.handle(Decision(action=Action.EXIT, intents=(legacy_exit,)),
                        inputs=_approval_inputs(), now_ns=3)
    ev = await asyncio.wait_for(sub.get(), timeout=0.5)
    assert isinstance(ev, Exit)
    assert ev.reason == "stop_loss"


@pytest.mark.asyncio
async def test_book_fill_persists_fill_row_with_closed_pnl(tmp_path):
    """Daily-loss accounting and post-hoc audit both rely on the local fill
    table being populated. Router._book_fill must write a Fill row on every
    venue fill, with closed_pnl set on reduces (0 on opens)."""
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    bus.subscribe()  # so publishes don't block
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    cfg = _strategy_cfg()
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, hl=client, strategy_cfg=cfg)

    # Open at 0.95 size 10, then close at 0.90 reduce_only.
    await router.handle(_decision_enter(), inputs=_approval_inputs(), now_ns=2)

    exit_intent = OrderIntent(
        question_idx=42, symbol="@30", side="sell", size=10.0,
        limit_price=0.90, cloid="hla-router-close", time_in_force="ioc",
        reduce_only=True, exit_reason="exit_safety_d",
    )
    # Approval requires a non-stale book and matching question; reuse the
    # entry inputs and patch the reference_price to keep gate happy.
    await router.handle(Decision(action=Action.EXIT, intents=(exit_intent,)),
                        inputs=_approval_inputs(), now_ns=3)

    fills = dal.fills_for_cloid("hla-router-1") + dal.fills_for_cloid("hla-router-close")
    assert len(fills) == 2
    # Open leg: closed_pnl=0; close leg: -$0.50 = (0.90 - 0.95) * 10
    by_side = {f.side: f for f in fills}
    assert by_side["buy"].closed_pnl == pytest.approx(0.0)
    assert by_side["sell"].closed_pnl == pytest.approx((0.90 - 0.95) * 10)


@pytest.mark.asyncio
async def test_vetoed_decision_publishes_veto_and_does_not_call_place(tmp_path):
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    sub = bus.subscribe()
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    cfg = _strategy_cfg()
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, hl=client, strategy_cfg=cfg)
    inputs = replace(_approval_inputs(), kill_switch_active=True)
    await router.handle(_decision_enter(), inputs=inputs, now_ns=2)
    assert dal.get_order("hla-router-1") is None
    ev = await asyncio.wait_for(sub.get(), timeout=0.5)
    assert isinstance(ev, RiskVeto)
