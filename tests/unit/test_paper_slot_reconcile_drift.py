"""Paper slots must not publish reconcile-drift (incident 2026-06-17 v31_pm_eth_ms).

A paper slot has no real venue: its ``clearinghouse_state`` returns the paper
client's own in-memory shadow ledger (``_paper_positions``), built from its fake
fills. The reconcile loop compares the engine DB ledger against that shadow
ledger — two copies of the same fake positions — which transiently diverge on
every open/close, flipping ``venue_absent`` ↔ ``venue_orphan``. Each flip is a
fresh Telegram dedupe key, so a churning paper slot pages on every position
transition (and spams the event table at INFO every cycle), with nothing an
operator can act on. Suppress drift publication at the source for paper slots.
"""

from __future__ import annotations

import pytest

from hlanalysis.engine.risk_events import ReconcileDrift
from tests.unit.test_async_offload import _build_runtime_with_recording


def _drift(alias: str) -> ReconcileDrift:
    return ReconcileDrift(
        ts_ns=1,
        account_alias=alias,
        case="position_mismatch",
        question_idx=42,
        detail={"resolution": "venue_absent_alert_only"},
    )


@pytest.mark.asyncio
async def test_paper_slot_does_not_publish_reconcile_drift(tmp_path):
    """A paper slot reconciles its DB against its own shadow ledger — the drift
    is pure noise with no real venue behind it, so nothing must be published."""
    rt, slot = _build_runtime_with_recording(tmp_path)
    slot.exec_client.paper_mode = True
    sub = rt.bus.subscribe()

    await rt._publish_drift_events(slot, [_drift(slot.alias)])

    assert sub.empty(), "paper slot must not publish reconcile drift"


@pytest.mark.asyncio
async def test_live_slot_still_publishes_reconcile_drift(tmp_path):
    """Regression guard: a live slot still publishes every drift event (real
    venue divergence remains operator-actionable)."""
    rt, slot = _build_runtime_with_recording(tmp_path)
    slot.exec_client.paper_mode = False
    sub = rt.bus.subscribe()

    ev = _drift(slot.alias)
    await rt._publish_drift_events(slot, [ev])

    assert sub.qsize() == 1, "live slot must publish reconcile drift"
    assert sub.get_nowait() is ev
