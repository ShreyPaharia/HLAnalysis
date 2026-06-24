"""The daily-loss gate must read venue-truth PnL without double-counting, and
fail safe on outage.

Venue settlement semantics differ:
  * HL exposes HIP-4 / bucket settlement as a *fill* (``dir="Settlement"``,
    ``closedPnl`` populated), so ``realized_pnl_since`` ALREADY includes the
    settlement payout. Adding the separately-persisted settlement PnL on top
    double-counts it — and historically mis-signed it for multi-leg buckets
    (a winning leg booked as a total loss → spurious DAILY LOSS HALT). So for
    HL the gate uses the venue read alone.
  * PM settles via an on-chain redeem, which is NOT a CLOB fill, so PM's
    ``realized_pnl_since`` misses it. PM still needs the persisted add-on.

On a sustained venue-read outage the gate must fail safe (halt the slot)
rather than trade on an understated number.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.unit.test_async_offload import _build_runtime_with_recording

_NOW = 1_700_000_000_000_000_000


@pytest.mark.asyncio
async def test_daily_pnl_hl_uses_venue_only(tmp_path):
    # HL: settlement is already in the venue fills' closedPnl, so the gate must
    # NOT add the persisted settlement again (no double-count, no mis-sign).
    rt, slot = _build_runtime_with_recording(tmp_path)
    assert slot.is_pm is False  # default builder is an HL slot
    slot.exec_client.realized_pnl_since = lambda ns, **_kw: -10.0  # venue (incl. settle)
    slot.dal.record_settlement(question_idx=1, symbol="@30", realized_pnl=-50.0, ts_ns=_NOW)

    pnl = await rt._realized_pnl_today(slot, now_ns=_NOW)

    assert pnl == pytest.approx(-10.0)  # venue only; persisted settle ignored


@pytest.mark.asyncio
async def test_daily_pnl_pm_adds_persisted_settlement(tmp_path):
    # PM: redeem is not a fill, so the persisted settlement must be added.
    rt, slot = _build_runtime_with_recording(tmp_path)
    slot.venue = "polymarket"  # flip the single venue predicate → PM path
    slot.exec_client.realized_pnl_since = lambda ns, **_kw: -10.0  # venue fills only
    slot.dal.record_settlement(question_idx=1, symbol="@30", realized_pnl=-50.0, ts_ns=_NOW)

    pnl = await rt._realized_pnl_today(slot, now_ns=_NOW)

    assert pnl == pytest.approx(-60.0)  # venue (-10) + settlement (-50)


@pytest.mark.asyncio
async def test_venue_failure_uses_settlement_inclusive_dal_then_halts(tmp_path):
    rt, slot = _build_runtime_with_recording(tmp_path)
    rt.daily_loss_venue_fail_halt = 2

    def _boom(ns, **_kw):
        raise RuntimeError("HL down")

    slot.exec_client.realized_pnl_since = _boom
    slot.dal.record_settlement(question_idx=1, symbol="@30", realized_pnl=-7.0, ts_ns=_NOW)

    # 1st failure: fall back to the settlement-inclusive DAL value (NOT zero),
    # don't halt yet.
    p1 = await rt._realized_pnl_today(slot, now_ns=_NOW)
    assert p1 == pytest.approx(-7.0)
    assert slot.halted is False

    # 2nd consecutive failure reaches the threshold → fail-safe halt.
    await rt._realized_pnl_today(slot, now_ns=_NOW)
    assert slot.halted is True


@pytest.mark.asyncio
async def test_venue_success_resets_failure_streak(tmp_path):
    rt, slot = _build_runtime_with_recording(tmp_path)
    rt.daily_loss_venue_fail_halt = 2
    calls = {"n": 0}

    def _flaky(ns, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("blip")
        return 0.0

    slot.exec_client.realized_pnl_since = _flaky

    await rt._realized_pnl_today(slot, now_ns=_NOW)  # fail (streak=1)
    await rt._realized_pnl_today(slot, now_ns=_NOW)  # success (streak reset)
    await rt._realized_pnl_today(slot, now_ns=_NOW)  # success
    assert slot.halted is False  # a single blip must not latch a halt


@pytest.mark.asyncio
async def test_venue_recovery_auto_clears_fail_halt(tmp_path):
    # A transient HL outage (429 burst) that latched the venue-fail halt must
    # AUTO-CLEAR once the venue read works again — otherwise a few-second blip
    # permanently downs the slot until an engine restart (incident 2026-06-24:
    # all three HL slots stuck halted 5h+ after a 47s API outage).
    rt, slot = _build_runtime_with_recording(tmp_path)
    rt.daily_loss_venue_fail_halt = 2
    flag = slot.kill_switch_path.parent / "venue_pnl_halt"
    calls = {"n": 0}

    def _flaky(ns, **_kw):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("HL down")
        return 0.0

    slot.exec_client.realized_pnl_since = _flaky

    await rt._realized_pnl_today(slot, now_ns=_NOW)  # fail 1
    await rt._realized_pnl_today(slot, now_ns=_NOW)  # fail 2 → fail-safe halt
    assert slot.halted is True
    assert slot.venue_pnl_halted is True
    assert flag.exists()  # surfaced to engine-diag via a flag file

    await rt._realized_pnl_today(slot, now_ns=_NOW)  # venue readable again
    assert slot.halted is False  # latch auto-cleared
    assert slot.venue_pnl_halted is False
    assert not flag.exists()  # visibility flag removed


@pytest.mark.asyncio
async def test_recovery_does_not_clear_operator_killswitch(tmp_path):
    # Auto-clear is scoped to the venue-fail halt ONLY. An operator / daily-loss
    # kill (kill-switch flag file present, venue_pnl_halted False) must stay
    # latched even when the venue read succeeds — the operator must clear it.
    rt, slot = _build_runtime_with_recording(tmp_path)
    slot.halted = True
    slot.venue_pnl_halted = False
    slot.kill_switch_path.touch()  # persistent operator/daily-loss kill flag
    slot.exec_client.realized_pnl_since = lambda ns, **_kw: 0.0

    await rt._realized_pnl_today(slot, now_ns=_NOW)

    assert slot.halted is True  # NOT auto-cleared by a successful venue read


@pytest.mark.asyncio
async def test_continuous_checks_probe_recovers_venue_fail_halt(tmp_path):
    # The production wiring: the scan loop / cap check that drive the venue read
    # are suspended while halted, so _continuous_checks_loop itself must probe
    # the venue while venue_pnl_halted and clear the latch on recovery. Without
    # this the helper-level auto-clear would never fire live.
    rt, slot = _build_runtime_with_recording(tmp_path)
    slot.halted = True
    slot.venue_pnl_halted = True
    slot.exec_client.realized_pnl_since = lambda _ns, **_kw: 0.0  # venue readable again

    task = asyncio.create_task(rt._continuous_checks_loop(slot))
    try:
        for _ in range(40):
            if not slot.halted:
                break
            await asyncio.sleep(0.05)
    finally:
        rt.stop_event.set()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            task.cancel()

    assert slot.halted is False  # loop probe cleared the venue-fail latch
    assert slot.venue_pnl_halted is False
