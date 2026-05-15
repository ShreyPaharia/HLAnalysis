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
