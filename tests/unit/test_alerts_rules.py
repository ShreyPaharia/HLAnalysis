from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from hlanalysis.alerts.rules import AlertRules
from hlanalysis.engine.event_bus import EventBus
from hlanalysis.engine.risk_events import (
    DailyLossHalt, Entry, Exit, KillSwitchActivated, OrderRejected,
    ReconcileDrift, RiskVeto,
)


class _FakeTelegram:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, text: str, *, markdown: bool = True) -> bool:
        self.messages.append(text)
        return True


@pytest.mark.asyncio
async def test_kill_switch_alerts_immediately():
    tg = _FakeTelegram()
    bus = EventBus()
    rules = AlertRules(bus=bus, telegram=tg, dedupe_window_s=60)
    sub = bus.subscribe()
    task = asyncio.create_task(rules.run(sub))
    await bus.publish(KillSwitchActivated(ts_ns=1, path="data/engine/halt"))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert any("KILL SWITCH" in m for m in tg.messages)


@pytest.mark.asyncio
async def test_risk_veto_dedupes_within_window():
    tg = _FakeTelegram()
    bus = EventBus()
    rules = AlertRules(bus=bus, telegram=tg, dedupe_window_s=60)
    sub = bus.subscribe()
    task = asyncio.create_task(rules.run(sub))
    for i in range(5):
        await bus.publish(RiskVeto(ts_ns=i, reason="max_position_usd"))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    veto_msgs = [m for m in tg.messages if "max_position_usd" in m]
    # First message goes through, the rest dedupe within the window
    assert len(veto_msgs) == 1


@pytest.mark.asyncio
async def test_entry_and_exit_format_pnl():
    tg = _FakeTelegram()
    bus = EventBus()
    rules = AlertRules(bus=bus, telegram=tg, dedupe_window_s=60)
    sub = bus.subscribe()
    task = asyncio.create_task(rules.run(sub))
    await bus.publish(Entry(
        ts_ns=1, cloid="hla-1", question_idx=42, symbol="@30",
        side="buy", size=10.0, price=0.95,
    ))
    await bus.publish(Exit(
        ts_ns=2, question_idx=42, symbol="@30", qty=10.0,
        realized_pnl=5.0, reason="settlement",
    ))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert any("ENTRY" in m and "@30" in m for m in tg.messages)
    assert any("EXIT" in m and "5.00" in m for m in tg.messages)


@pytest.mark.asyncio
async def test_order_rejected_alerts_with_error_text():
    tg = _FakeTelegram()
    bus = EventBus()
    rules = AlertRules(bus=bus, telegram=tg, dedupe_window_s=60)
    sub = bus.subscribe()
    task = asyncio.create_task(rules.run(sub))
    await bus.publish(OrderRejected(
        ts_ns=1, cloid="hla-1", question_idx=10, symbol="#551",
        side="buy", size=100.10, price=0.999,
        error="Order has invalid size.",
    ))
    # Different error string → not deduped
    await bus.publish(OrderRejected(
        ts_ns=2, cloid="hla-2", question_idx=10, symbol="#551",
        side="buy", size=100.0, price=0.999,
        error="Insufficient margin",
    ))
    # Same (symbol, error) — deduped
    await bus.publish(OrderRejected(
        ts_ns=3, cloid="hla-3", question_idx=10, symbol="#551",
        side="buy", size=100.0, price=0.999,
        error="Insufficient margin",
    ))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    rej_msgs = [m for m in tg.messages if "REJECTED" in m]
    assert len(rej_msgs) == 2
    assert any("invalid size" in m for m in rej_msgs)
    assert any("Insufficient margin" in m for m in rej_msgs)


@pytest.mark.asyncio
async def test_reconcile_drift_alerts_every_time():
    tg = _FakeTelegram()
    bus = EventBus()
    rules = AlertRules(bus=bus, telegram=tg, dedupe_window_s=60)
    sub = bus.subscribe()
    task = asyncio.create_task(rules.run(sub))
    await bus.publish(ReconcileDrift(ts_ns=1, case="local_ghost", cloid="hla-1"))
    await bus.publish(ReconcileDrift(ts_ns=2, case="venue_orphan", cloid="hla-2"))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    drift_msgs = [m for m in tg.messages if "DRIFT" in m]
    assert len(drift_msgs) == 2  # no dedupe for drift
