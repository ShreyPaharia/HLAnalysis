from __future__ import annotations

import asyncio
import json
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
