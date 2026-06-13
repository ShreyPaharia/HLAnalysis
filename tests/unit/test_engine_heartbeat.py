"""SHR-43: dead-man's-switch / liveness.

The engine must not treat silence as health. Each heartbeat interval it
publishes an EngineHeartbeat bus event and touches a server-side heartbeat file
(so an external monitor can alert on the ABSENCE of updates — the engine can't
alert on its own death). When zero market-data events are ingested over a full
interval while subscriptions are active, it publishes a FeedStale alert (a live
feed delivers a steady mark/book stream; zero events means the feed is dead, not
a calm market).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.test_async_offload import _build_runtime_with_recording


async def _drain(sub) -> list:
    out = []
    while not sub.empty():
        out.append(sub.get_nowait())
    return out


@pytest.mark.asyncio
async def test_heartbeat_publishes_engine_heartbeat_and_no_feedstale_when_live(tmp_path):
    rt, _slot = _build_runtime_with_recording(tmp_path)
    rt.subscriptions = ["BTC"]  # active subscription
    sub = rt.bus.subscribe()

    await rt._publish_heartbeat(d_events=5, n_questions=3)

    kinds = [getattr(e, "kind", None) for e in await _drain(sub)]
    assert "engine_heartbeat" in kinds
    assert "feed_stale" not in kinds


@pytest.mark.asyncio
async def test_heartbeat_publishes_feedstale_when_no_events_ingested(tmp_path):
    rt, _slot = _build_runtime_with_recording(tmp_path)
    rt.subscriptions = ["BTC"]
    sub = rt.bus.subscribe()

    await rt._publish_heartbeat(d_events=0, n_questions=0)

    kinds = [getattr(e, "kind", None) for e in await _drain(sub)]
    assert "feed_stale" in kinds


@pytest.mark.asyncio
async def test_no_feedstale_before_subscriptions_active(tmp_path):
    rt, _slot = _build_runtime_with_recording(tmp_path)
    rt.subscriptions = []  # nothing subscribed yet → 0 events is expected
    sub = rt.bus.subscribe()

    await rt._publish_heartbeat(d_events=0, n_questions=0)

    kinds = [getattr(e, "kind", None) for e in await _drain(sub)]
    assert "feed_stale" not in kinds


@pytest.mark.asyncio
async def test_heartbeat_touchfile_is_written(tmp_path):
    rt, _slot = _build_runtime_with_recording(tmp_path)
    rt.subscriptions = ["BTC"]

    await rt._publish_heartbeat(d_events=1, n_questions=0)

    hb = Path(rt.deploy_cfg.state_db_path).parent / "engine_heartbeat"
    assert hb.exists(), "dead-man's-switch heartbeat file was not written"
    assert hb.read_text().strip() != ""
