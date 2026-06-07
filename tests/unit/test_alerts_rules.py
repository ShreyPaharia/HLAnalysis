from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from hlanalysis.alerts.rules import AlertRules
from hlanalysis.engine.event_bus import EventBus
from hlanalysis.engine.risk_events import (
    DailyLossHalt, EngineHeartbeat, Entry, Exit, FeedDown, FeedRecovered,
    FeedStale, KillSwitchActivated, MemoryHalt,
    OrderRejected, OrderUnconfirmed, PMStrikeMismatch, ReconcileDrift,
    RedemptionTimeout, RiskVeto,
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
async def test_feed_stale_alerts_and_heartbeat_is_silent():
    """FeedStale must reach Telegram (the feed is dead); EngineHeartbeat must
    NOT (it fires every interval — alerting on it would be constant spam)."""
    tg = _FakeTelegram()
    bus = EventBus()
    rules = AlertRules(bus=bus, telegram=tg, dedupe_window_s=60)
    sub = bus.subscribe()
    task = asyncio.create_task(rules.run(sub))
    await bus.publish(EngineHeartbeat(ts_ns=1, events_ingested=10,
                                      d_events=0, n_questions=2))
    await bus.publish(FeedStale(ts_ns=2, d_events=0, interval_seconds=30.0))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert any("FEED STALE" in m for m in tg.messages)
    assert not any("heartbeat" in m.lower() for m in tg.messages)


@pytest.mark.asyncio
async def test_feed_down_and_recovered_alert():
    tg = _FakeTelegram()
    bus = EventBus()
    rules = AlertRules(bus=bus, telegram=tg, dedupe_window_s=60)
    sub = bus.subscribe()
    task = asyncio.create_task(rules.run(sub))
    await bus.publish(FeedDown(ts_ns=1, consecutive_failures=1))
    await bus.publish(FeedRecovered(ts_ns=2))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert any("FEED DOWN" in m for m in tg.messages)
    assert any("FEED RECOVERED" in m for m in tg.messages)


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
async def test_reconcile_drift_distinct_events_alert():
    # Different (case, question_idx, cloid) tuples should each get their own
    # alert — dedupe only collapses identical events.
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
    assert len(drift_msgs) == 2


@pytest.mark.asyncio
async def test_order_unconfirmed_formats_with_age_and_cloid():
    tg = _FakeTelegram()
    bus = EventBus()
    rules = AlertRules(bus=bus, telegram=tg, dedupe_window_s=60)
    sub = bus.subscribe()
    task = asyncio.create_task(rules.run(sub))
    await bus.publish(OrderUnconfirmed(
        ts_ns=1, account_alias="v31_pm", cloid="hla-v31-abc",
        symbol="0xdeadbeef", side="buy", size=10.0, limit_price=0.55,
        age_seconds=45.0, venue_oid="ven-1",
    ))
    # Same cloid → deduped within window
    await bus.publish(OrderUnconfirmed(
        ts_ns=2, account_alias="v31_pm", cloid="hla-v31-abc",
        symbol="0xdeadbeef", side="buy", size=10.0, limit_price=0.55,
        age_seconds=46.0, venue_oid="ven-1",
    ))
    # Different cloid → fires
    await bus.publish(OrderUnconfirmed(
        ts_ns=3, account_alias="v31_pm", cloid="hla-v31-xyz",
        symbol="0xdeadbeef", side="sell", size=5.0, limit_price=0.42,
        age_seconds=33.0,
    ))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    unconf_msgs = [m for m in tg.messages if "ORDER UNCONFIRMED" in m]
    assert len(unconf_msgs) == 2
    assert any("hla-v31-abc" in m and "45s" in m for m in unconf_msgs)
    assert any("hla-v31-xyz" in m and "33s" in m for m in unconf_msgs)
    assert all("[v31_pm]" in m for m in unconf_msgs)


@pytest.mark.asyncio
async def test_venue_prefix_renders_when_configured():
    """Two slots with same-shape events render with their respective venue
    tags so HL and PM alerts are visually separable on Telegram."""
    tg = _FakeTelegram()
    bus = EventBus()
    rules = AlertRules(
        bus=bus, telegram=tg, dedupe_window_s=60,
        venue_by_alias={"v31": "HL", "v31_pm": "PM"},
    )
    sub = bus.subscribe()
    task = asyncio.create_task(rules.run(sub))
    await bus.publish(RiskVeto(
        ts_ns=1, account_alias="v31", reason="max_position_usd", question_idx=7,
    ))
    await bus.publish(RiskVeto(
        ts_ns=2, account_alias="v31_pm", reason="max_position_usd", question_idx=7,
    ))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert any("[HL:v31]" in m for m in tg.messages)
    assert any("[PM:v31_pm]" in m for m in tg.messages)


@pytest.mark.asyncio
async def test_venue_prefix_omitted_when_alias_not_in_map():
    """An alias missing from venue_by_alias renders the legacy alias-only
    prefix instead of "[:v1]" — preserves single-venue deployments."""
    tg = _FakeTelegram()
    bus = EventBus()
    rules = AlertRules(
        bus=bus, telegram=tg, dedupe_window_s=60,
        venue_by_alias={"v31_pm": "PM"},
    )
    sub = bus.subscribe()
    task = asyncio.create_task(rules.run(sub))
    await bus.publish(RiskVeto(
        ts_ns=1, account_alias="v1", reason="max_position_usd", question_idx=7,
    ))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert any("[v1]" in m and "[:v1]" not in m for m in tg.messages)


@pytest.mark.asyncio
async def test_redemption_timeout_formats_with_age_hours():
    tg = _FakeTelegram()
    bus = EventBus()
    rules = AlertRules(bus=bus, telegram=tg, dedupe_window_s=60)
    sub = bus.subscribe()
    task = asyncio.create_task(rules.run(sub))
    settled_ts = 1_700_000_000_000_000_000
    # Winner with expected $100 payout
    await bus.publish(RedemptionTimeout(
        ts_ns=settled_ts + 6 * 3600 * 10**9, account_alias="v31_pm",
        question_idx=42, symbol="0xabcdef0123456789", qty=100.0,
        settled_ts_ns=settled_ts, age_seconds=6.5 * 3600.0,
        expected_payout_usd=100.0,
    ))
    # Loser: $0 expected
    await bus.publish(RedemptionTimeout(
        ts_ns=settled_ts + 7 * 3600 * 10**9, account_alias="v31_pm",
        question_idx=43, symbol="0x1234567890abcdef", qty=50.0,
        settled_ts_ns=settled_ts, age_seconds=7.0 * 3600.0,
        expected_payout_usd=0.0,
    ))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    redempt_msgs = [m for m in tg.messages if "REDEMPTION TIMEOUT" in m]
    assert len(redempt_msgs) == 2
    # Winner: $100.00 expected, 6.5h age, symbol truncated
    assert any(
        "q=42" in m and "100.00" in m and "6.5h" in m and "0xabcdef" in m
        for m in redempt_msgs
    )
    # Loser: $0.00 expected
    assert any("q=43" in m and "0.00" in m and "7.0h" in m for m in redempt_msgs)


@pytest.mark.asyncio
async def test_reconcile_drift_dedupes_repeated_identical_events():
    # Regression: every reconcile cycle (~60s) re-fires position_mismatch for
    # the same question while HL reports a slightly-different avg_entry, so
    # without dedupe Telegram fills with identical DRIFT spam.
    tg = _FakeTelegram()
    bus = EventBus()
    rules = AlertRules(bus=bus, telegram=tg, dedupe_window_s=60)
    sub = bus.subscribe()
    task = asyncio.create_task(rules.run(sub))
    for i in range(5):
        await bus.publish(ReconcileDrift(
            ts_ns=i, case="position_mismatch", question_idx=680,
            detail={"hl_qty": "15.0", "db_qty": "15.0"},
        ))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    drift_msgs = [m for m in tg.messages if "DRIFT" in m]
    assert len(drift_msgs) == 1


@pytest.mark.asyncio
async def test_memory_halt_formats_with_rss_and_ceiling():
    """MemoryHalt must reach Telegram with MB-converted RSS and ceiling values
    and a clear 'HALT' indicator so the operator knows the engine self-halted."""
    tg = _FakeTelegram()
    bus = EventBus()
    rules = AlertRules(bus=bus, telegram=tg, dedupe_window_s=60)
    sub = bus.subscribe()
    task = asyncio.create_task(rules.run(sub))
    # 900 MB RSS, 850 MB ceiling (typical 1 GB box configuration)
    await bus.publish(MemoryHalt(
        ts_ns=1, rss_kb=921_600, ceiling_kb=870_400,
    ))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert tg.messages, "MemoryHalt must produce at least one Telegram message"
    msg = tg.messages[0]
    # Must mention the halt condition clearly
    assert "MEMORY" in msg or "HALT" in msg, (
        "message must mention MEMORY or HALT so operators recognise it as a "
        "self-halt event"
    )
    # MB values: 921600 KB → 900 MB, 870400 KB → 850 MB
    assert "900" in msg, "message must include current RSS in MB"
    assert "850" in msg, "message must include ceiling in MB"


@pytest.mark.asyncio
async def test_pm_strike_mismatch_renders_to_telegram():
    tg = _FakeTelegram()
    bus = EventBus()
    rules = AlertRules(bus=bus, telegram=tg, dedupe_window_s=60)
    sub = bus.subscribe()
    task = asyncio.create_task(rules.run(sub))
    ev = PMStrikeMismatch(
        ts_ns=1, account_alias="v31_pm", question_idx=909002,
        captured_strike=73644.92, reference_mark=73501.0, divergence_bps=19.5,
    )
    rendered = rules._format(ev)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert rendered is not None
    _alias, body = rendered
    assert "strike" in body.lower()
    assert "73644.92" in body and "73501" in body
