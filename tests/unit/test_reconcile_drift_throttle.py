"""P2: alert-only reconcile-drift flood throttle.

Live PM reconcile can emit an identical alert-only ``ReconcileDrift`` every
reconcile cycle (~15s) forever when a position sits just past the qty-mismatch
tolerance and alert-only mode keeps it local (never converges). Observed:
v31_pm produced 1655 ``qty_mismatch_alert_only`` events in 65h on ONE position
(hl_qty=53.892109 vs db_qty=53.868533, 2026-07-02). This mirrors the router's
benign-veto throttle (``Router._throttle_benign_veto`` /
``_BENIGN_VETO_THROTTLE_NS``): emit on first sight, emit again immediately if
the drift detail changes (on-change), otherwise suppress until a heartbeat
window elapses — then emit once more carrying ``suppressed_since_last``.

Only ALERT-ONLY drifts (``detail["resolution"]`` ending in ``_alert_only``) are
throttled. Anything else (apply-mode adoptions, material drift, or missing
resolution) always publishes — fail-safe, never throttled.

We reuse ``_build_runtime_with_recording`` (a real, cheap ``EngineRuntime`` +
one HL slot) rather than hand-rolling fakes, so the tests exercise the real
``EngineRuntime._throttle_drift_alert`` / ``_publish_drift_events`` methods and
the real per-runtime throttle-state dataclass fields end-to-end.
"""

from __future__ import annotations

import pytest

from hlanalysis.engine.risk_events import ReconcileDrift
from hlanalysis.engine.runtime import _DRIFT_ALERT_THROTTLE_NS
from tests.unit.test_async_offload import _build_runtime_with_recording

_ALIAS = "v31_pm"
_QIDX = 4242


def _alert_only_drift(*, ts_ns: int, db_qty: str = "53.868533") -> ReconcileDrift:
    return ReconcileDrift(
        ts_ns=ts_ns,
        account_alias=_ALIAS,
        case="position_mismatch",
        question_idx=_QIDX,
        detail={
            "resolution": "qty_mismatch_alert_only",
            "hl_qty": "53.892109",
            "db_qty": db_qty,
        },
    )


def _non_alert_only_drift(*, ts_ns: int, resolution: str | None = None) -> ReconcileDrift:
    detail = {} if resolution is None else {"resolution": resolution}
    return ReconcileDrift(
        ts_ns=ts_ns,
        account_alias=_ALIAS,
        case="venue_orphan",
        question_idx=_QIDX,
        detail=detail,
    )


@pytest.mark.asyncio
async def test_first_alert_only_drift_emits(tmp_path):
    rt, _slot = _build_runtime_with_recording(tmp_path)

    emit, suppressed = rt._throttle_drift_alert(_alert_only_drift(ts_ns=1_000), now_ns=1_000)

    assert emit is True
    assert suppressed == 0


@pytest.mark.asyncio
async def test_second_identical_within_window_is_suppressed(tmp_path):
    rt, _slot = _build_runtime_with_recording(tmp_path)

    rt._throttle_drift_alert(_alert_only_drift(ts_ns=1_000), now_ns=1_000)
    emit, suppressed = rt._throttle_drift_alert(_alert_only_drift(ts_ns=1_100), now_ns=1_100)

    assert emit is False
    assert suppressed == 0
    key = (_ALIAS, _QIDX, "qty_mismatch_alert_only")
    assert rt._drift_alert_suppressed[key] == 1

    # a third identical drift within the window keeps accumulating suppressions
    emit2, _ = rt._throttle_drift_alert(_alert_only_drift(ts_ns=1_200), now_ns=1_200)
    assert emit2 is False
    assert rt._drift_alert_suppressed[key] == 2


@pytest.mark.asyncio
async def test_heartbeat_after_window_elapses_emits_with_suppressed_count(tmp_path):
    rt, _slot = _build_runtime_with_recording(tmp_path)

    rt._throttle_drift_alert(_alert_only_drift(ts_ns=1_000), now_ns=1_000)
    rt._throttle_drift_alert(_alert_only_drift(ts_ns=1_100), now_ns=1_100)
    rt._throttle_drift_alert(_alert_only_drift(ts_ns=1_200), now_ns=1_200)

    later_ns = 1_200 + _DRIFT_ALERT_THROTTLE_NS + 1
    emit, suppressed = rt._throttle_drift_alert(_alert_only_drift(ts_ns=later_ns), now_ns=later_ns)

    assert emit is True
    assert suppressed > 0
    assert suppressed == 2

    # the suppressed counter resets after the heartbeat emit
    key = (_ALIAS, _QIDX, "qty_mismatch_alert_only")
    assert rt._drift_alert_suppressed.get(key, 0) == 0


@pytest.mark.asyncio
async def test_changed_signature_within_window_emits_immediately(tmp_path):
    rt, _slot = _build_runtime_with_recording(tmp_path)

    rt._throttle_drift_alert(_alert_only_drift(ts_ns=1_000, db_qty="53.868533"), now_ns=1_000)
    emit, suppressed = rt._throttle_drift_alert(_alert_only_drift(ts_ns=1_050, db_qty="53.900000"), now_ns=1_050)

    assert emit is True
    assert suppressed == 0


@pytest.mark.asyncio
async def test_non_alert_only_drift_never_throttled(tmp_path):
    rt, _slot = _build_runtime_with_recording(tmp_path)

    # missing resolution
    ev1 = _non_alert_only_drift(ts_ns=1_000)
    emit1, _ = rt._throttle_drift_alert(ev1, now_ns=1_000)
    assert emit1 is True

    emit2, _ = rt._throttle_drift_alert(ev1, now_ns=1_001)
    assert emit2 is True, "repeated identical non-alert-only drift must still always emit"

    # apply-mode adoption resolution
    ev2 = _non_alert_only_drift(ts_ns=1_002, resolution="adopted_venue_orphan")
    emit3, _ = rt._throttle_drift_alert(ev2, now_ns=1_002)
    emit4, _ = rt._throttle_drift_alert(ev2, now_ns=1_003)
    assert emit3 is True
    assert emit4 is True


class _FakeBus:
    def __init__(self) -> None:
        self.published: list[object] = []

    async def publish(self, ev: object) -> None:
        self.published.append(ev)


@pytest.mark.asyncio
async def test_publish_drift_events_dedupes_repeat_alert_only(tmp_path):
    rt, slot = _build_runtime_with_recording(tmp_path)
    slot.exec_client.paper_mode = False
    fake_bus = _FakeBus()
    rt.bus = fake_bus

    ev1 = _alert_only_drift(ts_ns=1_000)
    ev2 = _alert_only_drift(ts_ns=1_100)

    await rt._publish_drift_events(slot, [ev1])
    await rt._publish_drift_events(slot, [ev2])

    assert len(fake_bus.published) == 1, "identical alert-only drift within window must publish once"

    # a non-alert-only drift always publishes, even back-to-back
    ev3 = _non_alert_only_drift(ts_ns=1_200, resolution="adopted_venue_orphan")
    ev4 = _non_alert_only_drift(ts_ns=1_201, resolution="adopted_venue_orphan")
    await rt._publish_drift_events(slot, [ev3])
    await rt._publish_drift_events(slot, [ev4])

    assert len(fake_bus.published) == 3
