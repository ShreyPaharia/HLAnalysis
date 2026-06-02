"""SHR-42: the ingest loop must survive a dying feed.

Previously `_ingest_loop` caught all exceptions, logged "adapter ingest
crashed", and returned — the only market-data task died with no reconnect and no
alert, while scan/stop-loss/reconcile kept running on a frozen MarketState (so a
real adverse move after the feed died never tripped a stop). The loop must now
reconnect with bounded backoff, alert FeedDown/FeedRecovered, and latch all slots
halted on a prolonged outage.
"""
from __future__ import annotations

import asyncio

import pytest

from hlanalysis.events import (
    Mechanism, MarkEvent, ProductType,
)
from tests.unit.test_async_offload import _build_runtime_with_recording


class _AlwaysRaiseAdapter:
    """stream() raises on every connection attempt (a dead/half-open WS)."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, subs):
        self.calls += 1
        raise RuntimeError("ws dead")
        yield  # unreachable — makes this an async generator


class _RaiseThenYieldAdapter:
    """First connection raises; subsequent connections deliver one event then
    end — exercises the FeedRecovered path."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, subs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("ws dead")
            yield  # unreachable
        yield MarkEvent(
            venue="hyperliquid", product_type=ProductType.PERP,
            mechanism=Mechanism.CLOB, symbol="BTC",
            exchange_ts=1, local_recv_ts=1, mark_px=80_000.0,
        )


def _fast_reconnect(rt) -> None:
    rt.subscriptions = ["BTC"]
    rt.ingest_reconnect_base_s = 0.001
    rt.ingest_reconnect_max_s = 0.001
    rt.ingest_halt_after_failures = 3


async def _drain(sub) -> list:
    out = []
    while not sub.empty():
        out.append(sub.get_nowait())
    return out


@pytest.mark.asyncio
async def test_dead_feed_reconnects_then_latches_halt(tmp_path):
    rt, slot = _build_runtime_with_recording(tmp_path)
    _fast_reconnect(rt)
    adapter = _AlwaysRaiseAdapter()
    rt.adapter_factory = lambda: adapter
    sub = rt.bus.subscribe()

    await asyncio.wait_for(rt._ingest_loop(rt.slots), timeout=2.0)

    kinds = [getattr(e, "kind", None) for e in await _drain(sub)]
    # Reconnected up to the halt threshold (didn't die on first failure).
    assert adapter.calls == 3
    assert "feed_down" in kinds
    # Prolonged outage latched the slot halted and stopped the engine.
    assert slot.halted is True
    assert rt.stop_event.is_set()


@pytest.mark.asyncio
async def test_feed_recovers_after_transient_drop(tmp_path):
    rt, slot = _build_runtime_with_recording(tmp_path)
    _fast_reconnect(rt)
    adapter = _RaiseThenYieldAdapter()
    rt.adapter_factory = lambda: adapter
    sub = rt.bus.subscribe()

    async def _stop_soon():
        await asyncio.sleep(0.05)
        rt.stop_event.set()

    await asyncio.gather(
        asyncio.wait_for(rt._ingest_loop(rt.slots), timeout=2.0),
        _stop_soon(),
    )

    kinds = [getattr(e, "kind", None) for e in await _drain(sub)]
    assert "feed_down" in kinds
    assert "feed_recovered" in kinds
    assert rt.events_ingested >= 1  # the recovery event was ingested
