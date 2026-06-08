"""SHR-83 — engine-hook wiring of the trade journal (router + reconcile).

These exercise the journal across a real decision → send → fill → (reject)
sequence driven through the Router, plus the late-fill path discovered by the
Reconciler. The journal must capture the lifecycle without altering any
trading behavior (a Router built with journal=None behaves exactly as before).
"""
from __future__ import annotations

import pytest

from hlanalysis.engine.config import (
    AllowlistEntry, GlobalRiskConfig, StrategyConfig,
)
from hlanalysis.engine.event_bus import EventBus
from hlanalysis.engine.exec_types import OrderAck, UserFillRow
from hlanalysis.engine.hl_client import HLClient
from hlanalysis.engine.reconcile import Reconciler
from hlanalysis.engine.risk import RiskGate, RiskInputs
from hlanalysis.engine.router import Router
from hlanalysis.engine.state import OpenOrder, StateDAL
from hlanalysis.engine.trade_journal import HaltSnapshot, TradeJournal
from hlanalysis.strategy.types import (
    Action, BookState, Decision, Diagnostic, OrderIntent, QuestionView,
)


def _cfg() -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        max_position_usd=100, stop_loss_pct=10, tte_min_seconds=60,
        tte_max_seconds=1800, price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200, vol_max=0.5,
    )
    return StrategyConfig(
        name="late_resolution", paper_mode=True,
        allowlist=[entry], blocklist_question_idxs=[], defaults=entry,
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


def _inputs() -> RiskInputs:
    return RiskInputs(
        question=_q(),
        question_fields={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        reference_price=80_300.0,
        book=BookState(symbol="@30", bid_px=0.94, bid_sz=10.0, ask_px=0.95,
                       ask_sz=10.0, last_trade_ts_ns=10_000_000_000_000_000,
                       last_l2_ts_ns=10_000_000_000_000_000,
                       bid_levels=((0.94, 10.0), (0.93, 20.0)),
                       ask_levels=((0.95, 10.0), (0.96, 20.0))),
        recent_volume_usd=5_000.0, positions=[],
        live_orders_total_notional=0.0, realized_pnl_today=0.0,
        kill_switch_active=False, last_reconcile_ns=10_000_000_000_000_000,
        now_ns=10_000_000_000_000_001,
    )


def _enter() -> Decision:
    return Decision(
        action=Action.ENTER,
        intents=(OrderIntent(
            question_idx=42, symbol="@30", side="buy", size=10.0,
            limit_price=0.95, cloid="hla-jrnl-1", time_in_force="ioc",
        ),),
        diagnostics=(Diagnostic("info", "entry", (("vol", "0.0321"),)),),
    )


class _RejectClient:
    """Minimal ExecutionClient stub whose place() always rejects."""
    def place(self, req):
        return OrderAck(cloid=req.cloid, venue_oid="", status="rejected",
                        error="insufficient margin")


def _stamped(cloid: str) -> str:
    # With the default cloid_prefix "hla-", a cloid already starting with "hla-"
    # is returned unchanged by Router._stamp_cloid (idempotent), so the journal
    # row is keyed by the original cloid.
    return cloid


@pytest.mark.asyncio
async def test_router_journals_decision_send_and_fill(tmp_path):
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    journal = TradeJournal(dal)
    cfg = _cfg()
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    router = Router(dal=dal, gate=RiskGate(cfg), bus=EventBus(),
                    exec_client=client, strategy_cfg=cfg, journal=journal)
    halt = HaltSnapshot(restart_blocked=False, daily_loss_halted=False,
                        realized_pnl_today=-7.0, daily_loss_cap_usd=200.0)
    await router.handle(_enter(), inputs=_inputs(), now_ns=500,
                        recent_returns=(0.001, -0.002, 0.0), halt=halt)

    cloid = _stamped("hla-jrnl-1")
    row = dal.get_journal_row(cloid)
    assert row is not None
    assert row.action == "enter" and row.question_idx == 42
    assert row.decision_ts_ns == 500
    assert row.reference_price == 80_300.0
    assert row.recent_volume_usd == 5_000.0
    assert row.sigma == pytest.approx(0.0321)  # from the entry diagnostic
    assert row.send_ts_ns == 500              # sent before the (paper) fill
    assert row.fill_ts_ns == 500              # paper client fills synchronously
    assert row.fill_sz == 10.0
    assert row.reject_reason is None
    # halt snapshot persisted.
    import json
    halt_d = json.loads(row.halt_json)
    assert halt_d["realized_pnl_today"] == -7.0


@pytest.mark.asyncio
async def test_router_journals_reject_reason(tmp_path):
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    journal = TradeJournal(dal)
    cfg = _cfg()
    router = Router(dal=dal, gate=RiskGate(cfg), bus=EventBus(),
                    exec_client=_RejectClient(), strategy_cfg=cfg, journal=journal)
    await router.handle(_enter(), inputs=_inputs(), now_ns=600)
    row = dal.get_journal_row(_stamped("hla-jrnl-1"))
    assert row is not None
    assert row.send_ts_ns == 600
    assert row.reject_reason == "insufficient margin"
    assert row.fill_ts_ns is None


@pytest.mark.asyncio
async def test_router_journals_risk_veto_as_reject(tmp_path):
    """A pre-trade risk veto is journaled as the reject_reason (the order was
    never sent), so decision-parity sees why the live engine declined."""
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    journal = TradeJournal(dal)
    cfg = _cfg()
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    router = Router(dal=dal, gate=RiskGate(cfg), bus=EventBus(),
                    exec_client=client, strategy_cfg=cfg, journal=journal)
    # An oversized order trips the per-position max_position_usd cap.
    big = Decision(action=Action.ENTER, intents=(OrderIntent(
        question_idx=42, symbol="@30", side="buy", size=100_000.0,
        limit_price=0.95, cloid="hla-jrnl-veto", time_in_force="ioc",
    ),))
    await router.handle(big, inputs=_inputs(), now_ns=700)
    row = dal.get_journal_row(_stamped("hla-jrnl-veto"))
    assert row is not None
    assert row.reject_reason  # some veto reason recorded
    assert row.send_ts_ns is None  # never sent


@pytest.mark.asyncio
async def test_router_without_journal_is_unchanged(tmp_path):
    """journal=None (the default) must not change behavior or raise."""
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    cfg = _cfg()
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    router = Router(dal=dal, gate=RiskGate(cfg), bus=EventBus(),
                    exec_client=client, strategy_cfg=cfg)
    await router.handle(_enter(), inputs=_inputs(), now_ns=800)
    assert dal.get_order(_stamped("hla-jrnl-1")) is not None  # order still placed


def test_reconcile_records_late_discovered_fill_in_journal(tmp_path):
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    journal = TradeJournal(dal)
    # A decision row exists (the router journaled it) but the synchronous ack
    # carried no fill — the fill is later discovered via user_fills.
    cloid = "hla-recon-1"
    journal.record_decision(cloid=cloid, question_idx=7, decision_ts_ns=1,
                            action="enter", side="buy", symbol="@30",
                            intended_size=4.0, intended_price=0.9)
    dal.upsert_order(OpenOrder(
        cloid=cloid, venue_oid=None, question_idx=7, symbol="@30", side="buy",
        price=0.9, size=4.0, status="open", placed_ts_ns=1,
        last_update_ts_ns=1, strategy_id="late_resolution",
    ))
    fill = UserFillRow(fill_id="tid-1", cloid=cloid, symbol="@30", side="buy",
                       price=0.901, size=4.0, fee=0.0, ts_ns=999, closed_pnl=0.0)
    rec = Reconciler(dal, fills_lookup=lambda c: [fill], journal=journal)
    from hlanalysis.engine.exec_types import ClearinghouseState
    rec.run(
        venue_open=[],
        venue_state=ClearinghouseState(positions=(), account_value_usd=0.0),
        now_ns=1000,
    )
    row = dal.get_journal_row(cloid)
    assert row.fill_ts_ns == 999
    assert row.fill_px == pytest.approx(0.901)
    assert row.fill_sz == 4.0
