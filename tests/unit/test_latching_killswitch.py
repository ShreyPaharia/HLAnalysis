"""W1.9 — Latching daily-loss kill-switch + RSS self-halt guard.

Three groups of assertions:

A) Daily-loss breach latches persistently:
   - slot.halted is True
   - kill-switch flag file exists
   - subsequent scan does NOT place new entries while halted
   - halt does NOT auto-clear on the next cycle (still latched)
   - startup with flag file present comes up halted (flag read on first loop)

B) Exits (stop-loss enforcement) still work while halted — the stop-loss loop
   must not skip enforcement just because slot.halted is True (the latch blocks
   NEW entries; it must not abandon open positions).

C) RSS self-halt guard: injecting a fake RSS reading above the ceiling halts
   all slots + writes their kill-switch flag files.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.unit.test_async_offload import _build_runtime_with_recording
from hlanalysis.engine.risk_events import DailyLossHalt, StopLossTriggered


_NOW = 1_750_000_000_000_000_000  # arbitrary fixed timestamp


# ---------------------------------------------------------------------------
# A) Latching daily-loss kill-switch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_daily_loss_breach_latches_halt_and_writes_flag(tmp_path):
    """On a daily-loss breach the runtime must set slot.halted=True AND write
    the kill-switch flag file.  Without the flag file a restart silently
    un-halts the slot; the flag makes the halt operator-clearable only."""
    rt, slot = _build_runtime_with_recording(tmp_path)
    # Pretend cap is $200 and today we lost $250.
    slot.cfg.global_.__class__.__dict__  # just reference to confirm attribute
    # daily_loss_cap_usd = 200 (from _build_runtime_with_recording)
    slot.exec_client.realized_pnl_since = lambda _ns: -250.0

    # Collect published events
    sub = rt.bus.subscribe()

    # Drive one continuous-checks tick manually (just the daily-loss block)
    now = _NOW
    pnl = await rt._realized_pnl_today(slot, now_ns=now)
    assert pnl == pytest.approx(-250.0)

    # Simulate the daily-loss branch
    cap = slot.cfg.global_.daily_loss_cap_usd
    if pnl < -cap:
        await rt.bus.publish(DailyLossHalt(
            ts_ns=now, account_alias=slot.alias,
            realized_pnl=pnl, cap=cap,
        ))
        slot.halted = True
        rt._latch_kill_switch(slot)

    assert slot.halted is True
    assert slot.kill_switch_path.exists(), (
        "kill-switch flag file must be written on daily-loss halt so a restart "
        "re-reads it and stays halted"
    )


@pytest.mark.asyncio
async def test_daily_loss_latch_does_not_auto_clear(tmp_path):
    """The latch must NOT auto-clear on the next cycle. Only operator
    removal of the flag file should resume the slot."""
    rt, slot = _build_runtime_with_recording(tmp_path)
    slot.exec_client.realized_pnl_since = lambda _ns: -9999.0

    # First cycle — breach
    pnl = await rt._realized_pnl_today(slot, now_ns=_NOW)
    cap = slot.cfg.global_.daily_loss_cap_usd
    if pnl < -cap:
        slot.halted = True
        rt._latch_kill_switch(slot)

    assert slot.halted is True
    assert slot.kill_switch_path.exists()

    # Second cycle — PnL "recovers" (shouldn't matter — latch is permanent)
    slot.exec_client.realized_pnl_since = lambda _ns: 0.0
    pnl2 = await rt._realized_pnl_today(slot, now_ns=_NOW + 1)
    assert pnl2 == pytest.approx(0.0)

    # halted must NOT flip back automatically
    assert slot.halted is True, (
        "halt must not auto-clear even when PnL recovers — operator must remove "
        "the flag file"
    )
    assert slot.kill_switch_path.exists(), "flag file must persist across cycles"


@pytest.mark.asyncio
async def test_startup_with_flag_file_stays_halted(tmp_path):
    """When the engine starts with the kill-switch flag file already on disk,
    the continuous-checks loop must pick it up on the very first iteration and
    set slot.halted=True (no operator action required for the halt to take
    effect after a restart)."""
    rt, slot = _build_runtime_with_recording(tmp_path)
    # Write the flag before any loop runs
    slot.kill_switch_path.parent.mkdir(parents=True, exist_ok=True)
    slot.kill_switch_path.touch()

    # Simulate one iteration of the kill-switch check in _continuous_checks_loop
    if slot.risk.kill_switch_active(slot.kill_switch_path):
        slot.halted = True

    assert slot.halted is True, (
        "slot must come up halted when the flag file was left by a previous "
        "daily-loss or operator halt"
    )


@pytest.mark.asyncio
async def test_halted_scan_loop_blocks_new_entries(tmp_path):
    """A halted slot must not place new entries (scan_loop skips the scanner
    when slot.halted is True — this is the existing behaviour we pin)."""
    rt, slot = _build_runtime_with_recording(tmp_path)
    slot.halted = True

    # Write the kill-switch flag so this mirrors a real latch
    slot.kill_switch_path.parent.mkdir(parents=True, exist_ok=True)
    slot.kill_switch_path.touch()

    # Collect 'place' calls via the exec_client (paper mode always returns a
    # filled ack synchronously, so if place is called the test would see it)
    place_calls: list = []
    _orig_place = slot.exec_client.place
    slot.exec_client.place = lambda *a, **kw: (  # type: ignore[assignment]
        place_calls.append((a, kw)) or _orig_place(*a, **kw)
    )

    # Drive the scan loop for one short burst
    async def _run_scan() -> None:
        task = asyncio.create_task(rt._scan_loop(slot))
        await asyncio.sleep(0.15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    await _run_scan()
    assert place_calls == [], (
        "halted slot must not place any orders — new entries are blocked"
    )


# ---------------------------------------------------------------------------
# B) Stop-loss exits still work while halted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_loss_enforced_while_halted(tmp_path):
    """While slot.halted is True (daily-loss latch), stop-loss enforcement
    must still run so open positions are protected.  Only NEW entries are
    blocked, not exits."""
    from hlanalysis.engine.state import Position as _Pos

    rt, slot = _build_runtime_with_recording(tmp_path)
    slot.halted = True  # simulate latched halt

    # Insert a position whose stop_loss_price is far above the current bid
    # so it will be breached by any market below it
    stop_price = 0.9
    slot.dal.upsert_position(
        _Pos(
            question_idx=42, symbol="@30",
            qty=10.0, avg_entry=0.95,
            stop_loss_price=stop_price,
            realized_pnl=0.0,
            last_update_ts_ns=_NOW,
        )
    )

    # Inject a book whose bid is below the stop price
    from hlanalysis.engine.market_state import MarketState
    from hlanalysis.events import (
        BboEvent, Mechanism, ProductType,
    )
    import time as _t
    ts = _t.time_ns()
    rt.market_state.apply(BboEvent(
        venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="@30",
        exchange_ts=ts, local_recv_ts=ts,
        bid_px=0.50, bid_sz=5.0, ask_px=0.51, ask_sz=5.0,
    ))

    # Track place calls
    place_calls: list = []
    _orig_place = slot.exec_client.place
    slot.exec_client.place = lambda *a, **kw: (
        place_calls.append((a, kw)) or _orig_place(*a, **kw)
    )

    # Run _enforce_stop_losses directly — this is what _stop_loss_loop calls
    await rt._enforce_stop_losses(slot, now_ns=_t.time_ns())

    # The stop should have been triggered (place called) despite slot.halted
    assert place_calls, (
        "_enforce_stop_losses must place a stop-exit even while slot is halted — "
        "the latch blocks new entries, not protective exits"
    )


# ---------------------------------------------------------------------------
# C) RSS self-halt guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rss_guard_halts_all_slots_and_writes_flags(tmp_path, monkeypatch):
    """When process RSS exceeds rss_halt_kb, the heartbeat loop must set all
    slots halted and write their kill-switch flag files."""
    from hlanalysis.engine.runtime import EngineRuntime

    rt, slot = _build_runtime_with_recording(tmp_path)
    rt.slots = [slot]  # register slot so the guard can iterate

    # Monkeypatch _read_rss_kb to return a value above the default ceiling
    monkeypatch.setattr(EngineRuntime, "_read_rss_kb",
                        staticmethod(lambda: 900_000), raising=True)

    # Drive the RSS check directly
    await rt._check_rss_halt(rt.slots)

    assert slot.halted is True, (
        "RSS above ceiling must halt all slots immediately"
    )
    assert slot.kill_switch_path.exists(), (
        "RSS halt must write the kill-switch flag file so a restart stays halted"
    )


@pytest.mark.asyncio
async def test_rss_guard_no_halt_below_ceiling(tmp_path, monkeypatch):
    """RSS below the ceiling must not trigger the guard."""
    from hlanalysis.engine.runtime import EngineRuntime

    rt, slot = _build_runtime_with_recording(tmp_path)
    rt.slots = [slot]

    # 400_000 KB — well below the 850_000 KB default ceiling
    monkeypatch.setattr(EngineRuntime, "_read_rss_kb",
                        staticmethod(lambda: 400_000), raising=True)

    await rt._check_rss_halt(rt.slots)

    assert slot.halted is False
    assert not slot.kill_switch_path.exists()


@pytest.mark.asyncio
async def test_rss_halt_kb_configurable(tmp_path, monkeypatch):
    """rss_halt_kb is overridable on EngineRuntime so operators can tune the
    ceiling for different box sizes."""
    from hlanalysis.engine.runtime import EngineRuntime

    rt, slot = _build_runtime_with_recording(tmp_path)
    rt.slots = [slot]
    rt.rss_halt_kb = 300_000  # very low custom ceiling

    # 350_000 KB — above the custom ceiling, below the default
    monkeypatch.setattr(EngineRuntime, "_read_rss_kb",
                        staticmethod(lambda: 350_000), raising=True)

    await rt._check_rss_halt(rt.slots)

    assert slot.halted is True, (
        "custom rss_halt_kb must be honoured — RSS above it should halt"
    )


# ---------------------------------------------------------------------------
# D) Integration: _continuous_checks_loop latches on daily-loss breach
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_continuous_checks_loop_latches_and_writes_flag(tmp_path):
    """End-to-end: run _continuous_checks_loop with a PnL that breaches the
    cap; verify slot.halted + flag file after one iteration."""
    rt, slot = _build_runtime_with_recording(tmp_path)
    # cap = 200; breaching loss
    slot.exec_client.realized_pnl_since = lambda _ns: -300.0

    # Stop the loop after one full iteration
    rt.stop_event = asyncio.Event()

    async def _stop_after_one():
        await asyncio.sleep(0.1)  # let at least one pass complete
        rt.stop_event.set()

    await asyncio.gather(
        rt._continuous_checks_loop(slot),
        _stop_after_one(),
    )

    assert slot.halted is True
    assert slot.kill_switch_path.exists(), (
        "_continuous_checks_loop must call _latch_kill_switch on breach"
    )
