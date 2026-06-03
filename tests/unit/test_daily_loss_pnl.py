"""SHR-49: the daily-loss gate must see settlement PnL and fail safe on outage.

HL `realized_pnl_since` returns closedPnl from fills only — HIP-4 settlement
payouts are not fills, so the cap was blind to the dominant PnL component of the
binary strategy. And on any HL error the code fell back to a structurally-zero
local PnL and kept trading, so a transient outage defeated the cap. The daily
PnL must now add persisted settlement PnL (SHR-53) to the venue read, and a
sustained venue-read outage must fail safe (halt the slot) rather than trade on
an understated number.
"""
from __future__ import annotations

import pytest

from tests.unit.test_async_offload import _build_runtime_with_recording

_NOW = 1_700_000_000_000_000_000


@pytest.mark.asyncio
async def test_daily_pnl_adds_settlement_to_venue_read(tmp_path):
    rt, slot = _build_runtime_with_recording(tmp_path)
    slot.exec_client.realized_pnl_since = lambda ns: -10.0  # HL fills pnl
    slot.dal.record_settlement(question_idx=1, symbol="@30",
                               realized_pnl=-50.0, ts_ns=_NOW)

    pnl = await rt._realized_pnl_today(slot, now_ns=_NOW)

    assert pnl == pytest.approx(-60.0)  # venue (-10) + settlement (-50)


@pytest.mark.asyncio
async def test_venue_failure_uses_settlement_inclusive_dal_then_halts(tmp_path):
    rt, slot = _build_runtime_with_recording(tmp_path)
    rt.daily_loss_venue_fail_halt = 2

    def _boom(ns):
        raise RuntimeError("HL down")

    slot.exec_client.realized_pnl_since = _boom
    slot.dal.record_settlement(question_idx=1, symbol="@30",
                               realized_pnl=-7.0, ts_ns=_NOW)

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

    def _flaky(ns):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("blip")
        return 0.0

    slot.exec_client.realized_pnl_since = _flaky

    await rt._realized_pnl_today(slot, now_ns=_NOW)  # fail (streak=1)
    await rt._realized_pnl_today(slot, now_ns=_NOW)  # success (streak reset)
    await rt._realized_pnl_today(slot, now_ns=_NOW)  # success
    assert slot.halted is False  # a single blip must not latch a halt
