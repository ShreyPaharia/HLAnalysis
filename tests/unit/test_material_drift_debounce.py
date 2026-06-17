"""Material qty-drift debounce + auto-clear (incident 2026-06-16 v31 #4010).

A single reconcile sample can show a transient >material qty gap during HFT
churn — a large in-flight order counted on the venue clearinghouse but not yet
in the local hot-path counter (or vice-versa). The old gate latched a permanent
block off ONE such sample, freezing v31 for 7h on a gap its own fills later
proved was clean, with no path back short of an operator restart.

The gate now:
  - DEBOUNCES: the material drift must PERSIST across `material_drift_debounce_cycles`
    consecutive reconciles before the slot is blocked.
  - AUTO-CLEARS: a slot blocked BY material_qty_drift auto-resumes once it has
    been clean for `material_drift_clear_cycles` consecutive reconciles.
  - Only material_qty_drift blocks auto-clear; a startup restart-drift block
    (material_drift_blocked=False) stays latched for the operator.
"""

from __future__ import annotations

import pytest

from tests.unit.test_async_offload import _build_runtime_with_recording


@pytest.mark.asyncio
async def test_transient_drift_does_not_block(tmp_path):
    """Material drift for fewer than the debounce count must NOT block — the
    transient HFT sample that wedged v31 must be tolerated."""
    rt, slot = _build_runtime_with_recording(tmp_path)
    rt.material_drift_debounce_cycles = 3

    assert rt._gate_material_drift(slot, True) is None  # cycle 1
    assert slot.blocked is False
    assert rt._gate_material_drift(slot, True) is None  # cycle 2
    assert slot.blocked is False, "2 cycles < debounce of 3 must not block"


@pytest.mark.asyncio
async def test_clean_cycle_resets_debounce_streak(tmp_path):
    """A clean reconcile resets the consecutive-drift streak, so drift must be
    CONSECUTIVE to count toward the debounce."""
    rt, slot = _build_runtime_with_recording(tmp_path)
    rt.material_drift_debounce_cycles = 3

    rt._gate_material_drift(slot, True)  # 1
    rt._gate_material_drift(slot, True)  # 2
    rt._gate_material_drift(slot, False)  # clean — resets streak
    rt._gate_material_drift(slot, True)  # 1 again
    assert rt._gate_material_drift(slot, True) is None  # 2 again
    assert slot.blocked is False, "non-consecutive drift must not accumulate to a block"


@pytest.mark.asyncio
async def test_persistent_drift_blocks_after_debounce(tmp_path):
    """Material drift persisting across the full debounce count blocks the slot
    (a genuinely stuck drift still halts)."""
    rt, slot = _build_runtime_with_recording(tmp_path)
    rt.material_drift_debounce_cycles = 3

    assert rt._gate_material_drift(slot, True) is None  # 1
    assert rt._gate_material_drift(slot, True) is None  # 2
    assert rt._gate_material_drift(slot, True) == "blocked"  # 3 → block
    assert slot.blocked is True
    assert slot.material_drift_blocked is True


@pytest.mark.asyncio
async def test_block_auto_clears_after_clean_streak(tmp_path):
    """A material_qty_drift block auto-resumes once clean for the clear count —
    no operator restart required (the v31 wedge fix)."""
    rt, slot = _build_runtime_with_recording(tmp_path)
    rt.material_drift_debounce_cycles = 1  # block on the first drift
    rt.material_drift_clear_cycles = 3

    assert rt._gate_material_drift(slot, True) == "blocked"
    assert slot.blocked is True

    assert rt._gate_material_drift(slot, False) is None  # clean 1
    assert slot.blocked is True
    assert rt._gate_material_drift(slot, False) is None  # clean 2
    assert slot.blocked is True
    assert rt._gate_material_drift(slot, False) == "cleared"  # clean 3 → resume
    assert slot.blocked is False
    assert slot.material_drift_blocked is False


@pytest.mark.asyncio
async def test_drift_reappearing_resets_clear_streak(tmp_path):
    """A drift reappearing during the clean streak resets the clear counter, so
    the slot stays blocked until it is TRULY clean for the full count."""
    rt, slot = _build_runtime_with_recording(tmp_path)
    rt.material_drift_debounce_cycles = 1
    rt.material_drift_clear_cycles = 3

    rt._gate_material_drift(slot, True)  # block
    rt._gate_material_drift(slot, False)  # clean 1
    rt._gate_material_drift(slot, True)  # drift again — resets clean streak, stays blocked
    assert slot.blocked is True
    rt._gate_material_drift(slot, False)  # clean 1 (restarted)
    rt._gate_material_drift(slot, False)  # clean 2
    assert slot.blocked is True
    assert rt._gate_material_drift(slot, False) == "cleared"  # clean 3 → resume
    assert slot.blocked is False


@pytest.mark.asyncio
async def test_auto_clear_does_not_touch_startup_block(tmp_path):
    """A startup restart-drift block (material_drift_blocked=False) must NOT be
    auto-cleared — only runtime material_qty_drift blocks self-heal; startup
    drift still requires operator inspection."""
    rt, slot = _build_runtime_with_recording(tmp_path)
    rt.material_drift_clear_cycles = 1
    slot.blocked = True
    slot.material_drift_blocked = False  # blocked by the startup restart-drift gate

    assert rt._gate_material_drift(slot, False) is None
    assert slot.blocked is True, "startup restart-drift block must stay latched"


@pytest.mark.asyncio
async def test_paper_slot_material_drift_never_blocks(tmp_path):
    """A paper slot reconciles its DB against the real (empty-by-design) venue,
    so a material qty gap is STRUCTURAL and permanent — it must never escalate
    to a block. Otherwise the paper burn-in wedges forever, unable to ever
    satisfy the clean-streak to auto-clear (incident 2026-06-16 v31_pm_eth_ms)."""
    rt, slot = _build_runtime_with_recording(tmp_path)
    slot.exec_client.paper_mode = True
    rt.material_drift_debounce_cycles = 1  # would block on the first drift if live

    assert rt._gate_material_drift(slot, True) is None
    assert rt._gate_material_drift(slot, True) is None
    assert rt._gate_material_drift(slot, True) is None
    assert slot.blocked is False, "paper slot must never block on material drift"


@pytest.mark.asyncio
async def test_paper_slot_existing_material_block_self_heals(tmp_path):
    """A paper slot already wedged by a material_qty_drift block (the live
    v31_pm_eth_ms state) must auto-resume on the next reconcile even while the
    structural drift is still present — paper treats venue divergence as clean."""
    rt, slot = _build_runtime_with_recording(tmp_path)
    slot.exec_client.paper_mode = True
    rt.material_drift_clear_cycles = 1
    slot.blocked = True
    slot.material_drift_blocked = True  # wedged by the gate before the paper fix

    # Drift is STILL present on the venue, but paper exempts it → resume.
    assert rt._gate_material_drift(slot, True) == "cleared"
    assert slot.blocked is False
    assert slot.material_drift_blocked is False
