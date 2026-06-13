"""#19: EventBus has no `unsubscribe`, and the halted-slot watchdog drops
settlement exits.

Part (a): EventBus.unsubscribe removes a handler so it no longer receives events.
Part (b): settlement-driven exits (reduce_only) must be permitted even when a slot
          is halted (the risk gate short-circuits exits before the halt check).

The risk-gate exit exemption is already present in risk.py (exits bypass entry
gates). This test verifies it holds even with last_reconcile_ns=0 (the #30 fix
must not accidentally block exits on halted slots).
"""
from __future__ import annotations

import asyncio

import pytest

from hlanalysis.engine.event_bus import EventBus
from hlanalysis.engine.risk import RiskGate, RiskInputs
from hlanalysis.engine.config import (
    AllowlistEntry,
    GlobalRiskConfig,
    StrategyConfig,
)
from hlanalysis.strategy.types import BookState, OrderIntent, QuestionView

# ── Part (a): unsubscribe ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unsubscribe_removes_handler():
    """#19a: After unsubscribe, the queue no longer receives published events."""
    bus = EventBus(maxsize=8)
    from hlanalysis.engine.risk_events import RiskVeto

    q1 = bus.subscribe()
    q2 = bus.subscribe()
    await bus.publish(RiskVeto(ts_ns=1, reason="before"))
    # Both should have received the first event
    assert not q1.empty()
    assert not q2.empty()
    _ = q1.get_nowait()
    _ = q2.get_nowait()

    # Unsubscribe q2
    bus.unsubscribe(q2)
    await bus.publish(RiskVeto(ts_ns=2, reason="after"))
    # q1 still receives
    assert not q1.empty()
    # q2 must NOT receive anything
    assert q2.empty()


@pytest.mark.asyncio
async def test_unsubscribe_unknown_queue_is_noop():
    """#19a: Unsubscribing a queue that was never subscribed must not raise."""
    bus = EventBus(maxsize=8)
    orphan: asyncio.Queue = asyncio.Queue()
    # Should not raise
    bus.unsubscribe(orphan)


@pytest.mark.asyncio
async def test_unsubscribe_is_idempotent():
    """#19a: Calling unsubscribe twice on the same queue must not raise."""
    bus = EventBus(maxsize=8)
    q = bus.subscribe()
    bus.unsubscribe(q)
    bus.unsubscribe(q)  # second call must be a no-op


# ── Part (b): exits allowed even when slot/inputs indicate halt-like conditions ─


NOW = 10_000_000_000_000_000


def _strategy_cfg() -> StrategyConfig:
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
                min_recent_volume_usd=1000,
                stale_data_halt_seconds=5,
                reconcile_interval_seconds=60,
            )
        },
    )


def _settlement_exit_inputs(kill_switch: bool = False) -> RiskInputs:
    """Build RiskInputs simulating a worst-case halted slot:
    - kill_switch_active may be True
    - last_reconcile_ns=0 (never reconciled — the #30 fix)
    - stale reference
    """
    q = QuestionView(
        question_idx=42,
        yes_symbol="@30",
        no_symbol="@31",
        strike=80_000.0,
        expiry_ns=NOW + 600_000_000_000,
        underlying="BTC",
        klass="priceBinary",
        period="1h",
    )
    book = BookState(
        symbol="@30",
        bid_px=0.94,
        bid_sz=10.0,
        ask_px=0.95,
        ask_sz=10.0,
        last_trade_ts_ns=NOW,
        last_l2_ts_ns=NOW,
    )
    return RiskInputs(
        question=q,
        question_fields={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        reference_price=80_300.0,
        book=book,
        recent_volume_usd=5_000.0,
        positions=[],
        live_orders_total_notional=0.0,
        realized_pnl_today=0.0,
        kill_switch_active=kill_switch,
        last_reconcile_ns=0,  # never reconciled
        now_ns=NOW,
    )


def _exit_intent() -> OrderIntent:
    return OrderIntent(
        question_idx=42,
        symbol="@30",
        side="sell",
        size=100.0,
        limit_price=0.94,
        cloid="hla-test-exit",
        time_in_force="ioc",
        reduce_only=True,
    )


def test_settlement_exit_allowed_with_never_reconciled():
    """#19b / #30: Settlement exits (reduce_only) must be allowed even when
    last_reconcile_ns=0 (engine just started, no reconcile yet)."""
    gate = RiskGate(_strategy_cfg())
    inp = _settlement_exit_inputs(kill_switch=False)
    v = gate.check_pre_trade(_exit_intent(), inp)
    assert v.approved is True, f"settlement exit must be allowed; got reason={v.reason!r}"


def test_settlement_exit_allowed_with_kill_switch_active():
    """#19b: Settlement exits must be allowed even when kill_switch is active
    (exits bypass all entry-only gates)."""
    gate = RiskGate(_strategy_cfg())
    inp = _settlement_exit_inputs(kill_switch=True)
    v = gate.check_pre_trade(_exit_intent(), inp)
    assert v.approved is True, f"settlement exit must be allowed with kill_switch; got reason={v.reason!r}"
