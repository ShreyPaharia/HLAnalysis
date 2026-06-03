from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from hlanalysis.engine.config import (
    AllowlistEntry, GlobalRiskConfig, StrategyConfig,
)
from hlanalysis.engine.event_bus import EventBus
from hlanalysis.engine.exec_types import (
    ClearinghouseState, OrderAck, PlaceRequest,
)
from hlanalysis.engine.hl_client import HLClient
from hlanalysis.engine.risk import RiskGate, RiskInputs
from hlanalysis.engine.risk_events import RiskVeto
from hlanalysis.engine.router import Router
from hlanalysis.engine.state import Position, StateDAL
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
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, exec_client=client, strategy_cfg=cfg)
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
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, exec_client=client, strategy_cfg=cfg)

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
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, exec_client=client, strategy_cfg=cfg)
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
async def test_addon_fill_publishes_entry_for_telegram(tmp_path):
    """Topups and other same-direction add-ons must emit Entry so Telegram
    alerts fire — not only the initial open."""
    from hlanalysis.engine.risk_events import Entry
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    sub = bus.subscribe()
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    cfg = _strategy_cfg()
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, exec_client=client, strategy_cfg=cfg)

    await router.handle(_decision_enter(), inputs=_approval_inputs(), now_ns=2)
    assert isinstance(await asyncio.wait_for(sub.get(), timeout=0.5), Entry)

    topup = Decision(
        action=Action.ENTER,
        intents=(OrderIntent(
            question_idx=42, symbol="@30", side="buy", size=5.0,
            limit_price=0.95, cloid="hla-router-topup", time_in_force="ioc",
        ),),
    )
    await router.handle(topup, inputs=_approval_inputs(), now_ns=3)
    ev = await asyncio.wait_for(sub.get(), timeout=0.5)
    assert isinstance(ev, Entry)
    assert ev.cloid == "hla-router-topup"
    assert ev.size == 5.0
    p = dal.get_position(42)
    assert p is not None and p.qty == pytest.approx(15.0)


@pytest.mark.asyncio
async def test_partial_reduce_does_not_publish_entry(tmp_path):
    """Selling part of a long must not look like a new ENTRY in Telegram."""
    from hlanalysis.engine.risk_events import Entry, Exit
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    sub = bus.subscribe()
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    cfg = _strategy_cfg()
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, exec_client=client, strategy_cfg=cfg)

    await router.handle(_decision_enter(), inputs=_approval_inputs(), now_ns=2)
    assert isinstance(await asyncio.wait_for(sub.get(), timeout=0.5), Entry)

    partial = OrderIntent(
        question_idx=42, symbol="@30", side="sell", size=4.0,
        limit_price=0.95, cloid="hla-router-partial", time_in_force="ioc",
        reduce_only=True, exit_reason="exit_edge",
    )
    await router.handle(Decision(action=Action.EXIT, intents=(partial,)),
                        inputs=_approval_inputs(), now_ns=3)
    assert sub.qsize() == 0
    p = dal.get_position(42)
    assert p is not None and p.qty == pytest.approx(6.0)


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
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, exec_client=client, strategy_cfg=cfg)

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
async def test_partial_reduce_preserves_avg_entry_and_pnl(tmp_path):
    """Closing a long via multiple partial reduce-only sells must:
    1) keep position.avg_entry pinned at the original cost basis
    2) record per-fill closed_pnl using that original cost basis

    Regression: prior code recomputed avg as
    (qty*avg + signed*price)/new_qty on every reduce. With signed<0 on a
    sell, this *inflates* avg_entry on each subsequent partial close, and
    each subsequent fill's closed_pnl absorbs that inflation. On the prod
    v31 q=21 trade on 2026-05-27 this turned a real $94 loss into a
    reported $161 loss.
    """
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    bus.subscribe()
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    cfg = _strategy_cfg()
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, exec_client=client, strategy_cfg=cfg)

    # Open 10 @ 0.95 (cost basis = 0.95).
    await router.handle(_decision_enter(), inputs=_approval_inputs(), now_ns=2)
    p = dal.get_position(42)
    assert p is not None and p.qty == pytest.approx(10.0)
    assert p.avg_entry == pytest.approx(0.95)

    # Partial reduce 1: sell 4 @ 0.80.
    sell1 = OrderIntent(
        question_idx=42, symbol="@30", side="sell", size=4.0,
        limit_price=0.80, cloid="hla-router-sell1", time_in_force="ioc",
        reduce_only=True, exit_reason="exit_edge",
    )
    await router.handle(Decision(action=Action.EXIT, intents=(sell1,)),
                        inputs=_approval_inputs(), now_ns=3)
    p = dal.get_position(42)
    assert p is not None and p.qty == pytest.approx(6.0)
    assert p.avg_entry == pytest.approx(0.95), \
        f"avg_entry drifted after partial reduce: {p.avg_entry}"
    f1 = dal.fills_for_cloid("hla-router-sell1")[0]
    assert f1.closed_pnl == pytest.approx((0.80 - 0.95) * 4.0)

    # Partial reduce 2: sell 3 @ 0.70. Must use the ORIGINAL 0.95 basis.
    sell2 = OrderIntent(
        question_idx=42, symbol="@30", side="sell", size=3.0,
        limit_price=0.70, cloid="hla-router-sell2", time_in_force="ioc",
        reduce_only=True, exit_reason="exit_edge",
    )
    await router.handle(Decision(action=Action.EXIT, intents=(sell2,)),
                        inputs=_approval_inputs(), now_ns=4)
    p = dal.get_position(42)
    assert p is not None and p.qty == pytest.approx(3.0)
    assert p.avg_entry == pytest.approx(0.95)
    f2 = dal.fills_for_cloid("hla-router-sell2")[0]
    assert f2.closed_pnl == pytest.approx((0.70 - 0.95) * 3.0)


@pytest.mark.asyncio
async def test_addon_buy_recomputes_avg_entry(tmp_path):
    """Same-direction add-on (topup buy on an existing long) must
    qty-weight-average the prior basis with the new fill price."""
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    bus.subscribe()
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    cfg = _strategy_cfg()
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, exec_client=client, strategy_cfg=cfg)

    await router.handle(_decision_enter(), inputs=_approval_inputs(), now_ns=2)

    addon = OrderIntent(
        question_idx=42, symbol="@30", side="buy", size=5.0,
        limit_price=0.92, cloid="hla-router-addon", time_in_force="ioc",
    )
    await router.handle(Decision(action=Action.ENTER, intents=(addon,)),
                        inputs=_approval_inputs(), now_ns=3)
    p = dal.get_position(42)
    expected = (10.0 * 0.95 + 5.0 * 0.92) / 15.0
    assert p is not None and p.qty == pytest.approx(15.0)
    assert p.avg_entry == pytest.approx(expected)


def _strategy_cfg_with_slippage(slip_cap: float = 0.005) -> StrategyConfig:
    # Mirrors _strategy_cfg but enables the depth-walk gate (HL default 0).
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
            reconcile_interval_seconds=60, max_slippage_pct=slip_cap,
        )},
    )


@pytest.mark.asyncio
async def test_depth_walk_clamps_intent_size_before_place(tmp_path):
    """When at-limit liquidity is below the intended size, the gate approves
    with `clamped_size` and the router must resize the intent before placement
    rather than veto. The DB row should reflect the clamped size — PM CLOB IOC
    partial-fills at the inside ask, and the strategy's topup re-fires on the
    next tick to close the residual."""
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    cfg = _strategy_cfg_with_slippage(slip_cap=0.005)
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, exec_client=client, strategy_cfg=cfg)
    # Inside ask 0.95 has only 3 contracts. Strategy intends 10 @ 0.95.
    inputs = replace(
        _approval_inputs(),
        book=BookState(
            symbol="@30", bid_px=0.94, bid_sz=10.0, ask_px=0.95, ask_sz=3.0,
            last_trade_ts_ns=10_000_000_000_000_000,
            last_l2_ts_ns=10_000_000_000_000_000,
            ask_levels=((0.95, 3.0), (0.97, 50.0)),
        ),
    )
    await router.handle(_decision_enter(), inputs=inputs, now_ns=2)
    o = dal.get_order("hla-router-1")
    assert o is not None and o.status == "filled"
    assert o.size == 3.0  # clamped from 10 → 3 (inside-ask depth)
    assert o.price == 0.95


@pytest.mark.asyncio
async def test_vetoed_decision_publishes_veto_and_does_not_call_place(tmp_path):
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    sub = bus.subscribe()
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    cfg = _strategy_cfg()
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, exec_client=client, strategy_cfg=cfg)
    inputs = replace(_approval_inputs(), kill_switch_active=True)
    await router.handle(_decision_enter(), inputs=inputs, now_ns=2)
    assert dal.get_order("hla-router-1") is None
    ev = await asyncio.wait_for(sub.get(), timeout=0.5)
    assert isinstance(ev, RiskVeto)


def _cfg_with_cooldown(seconds: int) -> StrategyConfig:
    """Return a strategy config whose defaults set entry_cooldown_seconds."""
    base = _strategy_cfg()
    new_defaults = base.defaults.model_copy(update={"entry_cooldown_seconds": seconds})
    return base.model_copy(update={"defaults": new_defaults})


@pytest.mark.asyncio
async def test_post_exit_cooldown_blocks_immediate_reentry(tmp_path):
    """After a close, ENTER decisions on the same question_idx within the
    cooldown window must be vetoed with reason=post_exit_cooldown — not the
    risk gate's normal rejection path. Catches the v1 churn pattern observed
    on 2026-05-19 where the strategy re-bought 200 shares ~1 second after
    selling, paying the spread on every cycle."""
    from hlanalysis.engine.risk_events import Entry, Exit, RiskVeto
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    sub = bus.subscribe()
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    cfg = _cfg_with_cooldown(60)
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, exec_client=client, strategy_cfg=cfg)

    # 1) Open position.
    await router.handle(_decision_enter(), inputs=_approval_inputs(), now_ns=1_000_000_000)
    assert isinstance(await asyncio.wait_for(sub.get(), timeout=0.5), Entry)

    # 2) Close it (publishes Exit).
    exit_intent = OrderIntent(
        question_idx=42, symbol="@30", side="sell", size=10.0,
        limit_price=0.95, cloid="hla-router-close", time_in_force="ioc",
        reduce_only=True, exit_reason="exit_safety_d",
    )
    close_ts = 2_000_000_000
    await router.handle(Decision(action=Action.EXIT, intents=(exit_intent,)),
                        inputs=_approval_inputs(), now_ns=close_ts)
    assert isinstance(await asyncio.wait_for(sub.get(), timeout=0.5), Exit)

    # 3) Try to re-enter 5 seconds later — should be vetoed by cooldown.
    reentry = Decision(
        action=Action.ENTER,
        intents=(OrderIntent(
            question_idx=42, symbol="@30", side="buy", size=10.0,
            limit_price=0.95, cloid="hla-router-reentry", time_in_force="ioc",
        ),),
    )
    await router.handle(reentry, inputs=_approval_inputs(),
                        now_ns=close_ts + 5_000_000_000)
    veto = await asyncio.wait_for(sub.get(), timeout=0.5)
    assert isinstance(veto, RiskVeto)
    assert veto.reason == "post_exit_cooldown"
    # The reentry order must NOT have been persisted.
    assert dal.get_order("hla-router-reentry") is None


@pytest.mark.asyncio
async def test_post_exit_cooldown_expires_and_allows_reentry(tmp_path):
    """After the cooldown window has elapsed, ENTER decisions on the same
    question_idx must proceed through the risk gate as normal."""
    from hlanalysis.engine.risk_events import Entry, Exit
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    sub = bus.subscribe()
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    cfg = _cfg_with_cooldown(60)
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, exec_client=client, strategy_cfg=cfg)
    await router.handle(_decision_enter(), inputs=_approval_inputs(), now_ns=1_000_000_000)
    await asyncio.wait_for(sub.get(), timeout=0.5)
    exit_intent = OrderIntent(
        question_idx=42, symbol="@30", side="sell", size=10.0,
        limit_price=0.95, cloid="hla-router-close-2", time_in_force="ioc",
        reduce_only=True, exit_reason="exit_safety_d",
    )
    close_ts = 2_000_000_000
    await router.handle(Decision(action=Action.EXIT, intents=(exit_intent,)),
                        inputs=_approval_inputs(), now_ns=close_ts)
    await asyncio.wait_for(sub.get(), timeout=0.5)  # drain exit

    # 61 seconds later — cooldown expired.
    reentry = Decision(
        action=Action.ENTER,
        intents=(OrderIntent(
            question_idx=42, symbol="@30", side="buy", size=10.0,
            limit_price=0.95, cloid="hla-router-reentry-2", time_in_force="ioc",
        ),),
    )
    await router.handle(reentry, inputs=_approval_inputs(),
                        now_ns=close_ts + 61_000_000_000)
    # Should have placed and got back an Entry event (paper client fills immediately).
    ev = await asyncio.wait_for(sub.get(), timeout=0.5)
    assert isinstance(ev, Entry)


@pytest.mark.asyncio
async def test_post_exit_cooldown_disabled_when_seconds_is_zero(tmp_path):
    """Default config (entry_cooldown_seconds=0) preserves legacy behavior:
    re-entry can fire on the very next scan tick. Regression guard so the
    feature doesn't quietly become mandatory."""
    from hlanalysis.engine.risk_events import Entry
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    sub = bus.subscribe()
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    cfg = _strategy_cfg()  # default cooldown=0
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, exec_client=client, strategy_cfg=cfg)
    await router.handle(_decision_enter(), inputs=_approval_inputs(), now_ns=1_000_000_000)
    await asyncio.wait_for(sub.get(), timeout=0.5)
    exit_intent = OrderIntent(
        question_idx=42, symbol="@30", side="sell", size=10.0,
        limit_price=0.95, cloid="hla-router-close-3", time_in_force="ioc",
        reduce_only=True, exit_reason="exit_safety_d",
    )
    await router.handle(Decision(action=Action.EXIT, intents=(exit_intent,)),
                        inputs=_approval_inputs(), now_ns=2_000_000_000)
    await asyncio.wait_for(sub.get(), timeout=0.5)
    # 1 ns later — should re-enter immediately.
    reentry = Decision(
        action=Action.ENTER,
        intents=(OrderIntent(
            question_idx=42, symbol="@30", side="buy", size=10.0,
            limit_price=0.95, cloid="hla-router-reentry-3", time_in_force="ioc",
        ),),
    )
    await router.handle(reentry, inputs=_approval_inputs(), now_ns=2_000_000_001)
    ev = await asyncio.wait_for(sub.get(), timeout=0.5)
    assert isinstance(ev, Entry)


def _make_router(tmp_path, cfg=None) -> Router:
    """Build a Router pointed at tmp_path/state.db. Shared cooldown-persistence helper."""
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    bus.subscribe()
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    cfg = cfg or _strategy_cfg()
    return Router(dal=dal, gate=RiskGate(cfg), bus=bus, exec_client=client, strategy_cfg=cfg)


def test_cooldown_init_with_no_file_is_empty(tmp_path):
    """Router constructed against a clean state dir has an empty cooldown map."""
    router = _make_router(tmp_path)
    assert router._last_exit_ts == {}
    assert not (tmp_path / "exit_cooldowns.json").exists()


def test_cooldown_init_loads_valid_file(tmp_path):
    """Router populates _last_exit_ts from an existing exit_cooldowns.json."""
    # Pre-seed the cooldown file alongside where state.db will live.
    (tmp_path / "exit_cooldowns.json").write_text(json.dumps({"42": 1_500_000_000, "7": 9}))
    router = _make_router(tmp_path)
    assert router._last_exit_ts == {42: 1_500_000_000, 7: 9}


def test_cooldown_init_handles_malformed_file(tmp_path):
    """A corrupt cooldown file must not crash router init; cooldown map starts empty."""
    (tmp_path / "exit_cooldowns.json").write_text("{not valid json")
    router = _make_router(tmp_path)
    assert router._last_exit_ts == {}


@pytest.mark.asyncio
async def test_close_persists_cooldown_to_file(tmp_path):
    """After a full close via _book_fill, exit_cooldowns.json contains the
    question_idx → close_ts mapping."""
    cfg = _cfg_with_cooldown(60)
    router = _make_router(tmp_path, cfg=cfg)
    await router.handle(_decision_enter(), inputs=_approval_inputs(), now_ns=1_000_000_000)
    close_ts = 2_000_000_000
    exit_intent = OrderIntent(
        question_idx=42, symbol="@30", side="sell", size=10.0,
        limit_price=0.95, cloid="hla-router-close-persist", time_in_force="ioc",
        reduce_only=True, exit_reason="exit_safety_d",
    )
    await router.handle(Decision(action=Action.EXIT, intents=(exit_intent,)),
                        inputs=_approval_inputs(), now_ns=close_ts)
    raw = json.loads((tmp_path / "exit_cooldowns.json").read_text())
    assert raw == {"42": close_ts}


@pytest.mark.asyncio
async def test_cooldown_roundtrip_across_router_instances(tmp_path):
    """A second Router built from the same state.db path reads back the
    cooldown map persisted by the first one."""
    cfg = _cfg_with_cooldown(60)
    r1 = _make_router(tmp_path, cfg=cfg)
    await r1.handle(_decision_enter(), inputs=_approval_inputs(), now_ns=1_000_000_000)
    close_ts = 2_000_000_000
    exit_intent = OrderIntent(
        question_idx=42, symbol="@30", side="sell", size=10.0,
        limit_price=0.95, cloid="hla-router-close-rt", time_in_force="ioc",
        reduce_only=True, exit_reason="exit_safety_d",
    )
    await r1.handle(Decision(action=Action.EXIT, intents=(exit_intent,)),
                    inputs=_approval_inputs(), now_ns=close_ts)

    r2 = _make_router(tmp_path, cfg=cfg)
    assert r2._last_exit_ts == {42: close_ts}


@pytest.mark.asyncio
async def test_cooldown_veto_uses_persisted_state_after_restart(tmp_path):
    """The persisted cooldown must enforce the post_exit_cooldown veto on a
    freshly-constructed Router (the restart scenario this fix targets)."""
    from hlanalysis.engine.risk_events import RiskVeto as _RiskVeto
    cfg = _cfg_with_cooldown(60)
    close_ts = 2_000_000_000
    # Simulate the "previous process" by writing the cooldown file directly.
    (tmp_path / "exit_cooldowns.json").write_text(json.dumps({"42": close_ts}))

    # New Router (= post-restart). Use its own bus so we can subscribe fresh.
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    sub = bus.subscribe()
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, exec_client=client, strategy_cfg=cfg)

    reentry = Decision(
        action=Action.ENTER,
        intents=(OrderIntent(
            question_idx=42, symbol="@30", side="buy", size=10.0,
            limit_price=0.95, cloid="hla-router-restart-reentry", time_in_force="ioc",
        ),),
    )
    await router.handle(reentry, inputs=_approval_inputs(),
                        now_ns=close_ts + 5_000_000_000)
    veto = await asyncio.wait_for(sub.get(), timeout=0.5)
    assert isinstance(veto, _RiskVeto)
    assert veto.reason == "post_exit_cooldown"
    assert dal.get_order("hla-router-restart-reentry") is None


@pytest.mark.asyncio
async def test_close_settled_persists_settlement_pnl(tmp_path):
    """SHR-53: settling a position must persist its realized PnL (not just emit
    an Exit alert) so the daily-loss gate can see it."""
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    cfg = _strategy_cfg()
    client = HLClient(account_address="0x", api_secret_key="0x",
                      base_url="x", paper_mode=True)
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, exec_client=client,
                    strategy_cfg=cfg)
    dal.upsert_position(Position(
        question_idx=42, symbol="@30", qty=10.0, avg_entry=0.6,
        realized_pnl=0.0, last_update_ts_ns=1, stop_loss_price=0.0,
    ))
    settled_q = replace(_q(), settled=True, settled_symbol="@30")

    await router._close_settled(42, now_ns=100, question=settled_q)

    assert dal.get_position(42) is None
    # Winning leg: 10 * (1.0 - 0.6) = +4.0, now persisted.
    assert dal.settlement_pnl_since(0) == pytest.approx(4.0)


class _CountingExec:
    """ExecutionClient stand-in that auto-fills like paper mode but records
    every PlaceRequest, so a test can assert a reduce-only re-fire against a
    flat position is suppressed (SHR-47)."""

    paper_mode = False

    def __init__(self) -> None:
        self.placed: list[PlaceRequest] = []

    def place(self, req: PlaceRequest) -> OrderAck:
        self.placed.append(req)
        return OrderAck(cloid=req.cloid, venue_oid="v", status="filled",
                        fill_price=req.price, fill_size=req.size)

    def cancel(self, *, cloid: str, symbol: str) -> bool:
        return True

    def open_orders(self):
        return []

    def clearinghouse_state(self) -> ClearinghouseState:
        return ClearinghouseState(positions=(), account_value_usd=0.0)

    def user_fills(self, *, since_ts_ns: int = 0):
        return []

    def realized_pnl_since(self, since_ts_ns: int) -> float:
        return 0.0


@pytest.mark.asyncio
async def test_reduce_only_sell_suppressed_when_position_already_flat(tmp_path):
    """The stop-loss enforcer re-fires a full-size IOC every ~1s. If the prior
    fill already closed the position, a re-fired reduce-only sell must NOT be
    placed — otherwise it opens a naked short on the outcome leg (SHR-47)."""
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    client = _CountingExec()
    cfg = _strategy_cfg()
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, exec_client=client,
                    strategy_cfg=cfg)

    # 1) Open a long (qty +10).
    await router.handle(_decision_enter(), inputs=_approval_inputs(), now_ns=2)
    assert dal.get_position(42).qty == 10.0

    # 2) Reduce-only sell closes it (qty -> 0).
    close = OrderIntent(
        question_idx=42, symbol="@30", side="sell", size=10.0, limit_price=0.95,
        cloid="hla-exit-1", time_in_force="ioc", reduce_only=True,
        exit_reason="stop_loss",
    )
    await router.handle(Decision(action=Action.EXIT, intents=(close,)),
                        inputs=_approval_inputs(), now_ns=3)
    pos = dal.get_position(42)
    assert pos is None or abs(pos.qty) < 1e-9
    n_after_close = len(client.placed)

    # 3) Re-fire the same reduce-only sell against the now-flat position.
    refire = OrderIntent(
        question_idx=42, symbol="@30", side="sell", size=10.0, limit_price=0.95,
        cloid="hla-exit-2", time_in_force="ioc", reduce_only=True,
        exit_reason="stop_loss",
    )
    await router.handle(Decision(action=Action.EXIT, intents=(refire,)),
                        inputs=_approval_inputs(), now_ns=4)

    assert len(client.placed) == n_after_close, (
        "re-fired reduce-only order was placed against a flat position — "
        "oversell into a naked short"
    )
    pos = dal.get_position(42)
    assert pos is None or abs(pos.qty) < 1e-9, (
        f"reduce-only re-fire created a naked position: {pos}"
    )


class _PartialExec(_CountingExec):
    """Fills at most `fill_cap` units per order, so reduce-only exits partial-
    fill and the position drains gradually across stop-loss ticks (SHR-47)."""

    def __init__(self, fill_cap: float) -> None:
        super().__init__()
        self.fill_cap = fill_cap

    def place(self, req: PlaceRequest) -> OrderAck:
        self.placed.append(req)
        filled = min(req.size, self.fill_cap)
        return OrderAck(cloid=req.cloid, venue_oid="v", status="filled",
                        fill_price=req.price, fill_size=filled)


@pytest.mark.asyncio
async def test_reduce_only_clamps_to_remaining_qty_after_partial_fill(tmp_path):
    """A re-fired stop-loss must never request more than is currently held, even
    when the prior IOC only partially filled — otherwise the residual order
    oversells past flat into a naked short (SHR-47)."""
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    client = _PartialExec(fill_cap=4.0)
    cfg = _strategy_cfg()
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, exec_client=client,
                    strategy_cfg=cfg)

    # Open long qty +10 (entry isn't reduce-only, fills 4 of 10).
    await router.handle(_decision_enter(), inputs=_approval_inputs(), now_ns=2)
    held0 = dal.get_position(42).qty
    assert held0 == 4.0  # only 4 filled

    # Fire a full-size (10) reduce-only sell. It must be clamped to held (4),
    # and the resulting placed size must never exceed what we hold.
    stop = OrderIntent(
        question_idx=42, symbol="@30", side="sell", size=10.0, limit_price=0.95,
        cloid="hla-stop-1", time_in_force="ioc", reduce_only=True,
        exit_reason="stop_loss",
    )
    await router.handle(Decision(action=Action.EXIT, intents=(stop,)),
                        inputs=_approval_inputs(), now_ns=3)

    sell = client.placed[-1]
    assert sell.side == "sell"
    assert sell.size <= held0, (
        f"reduce-only sell size {sell.size} exceeded held {held0}"
    )
    assert sell.size == 4.0  # clamped to held, not the requested 10
    # Position never crosses zero into a short.
    pos = dal.get_position(42)
    assert pos is None or pos.qty >= 0.0, f"position went short: {pos}"
