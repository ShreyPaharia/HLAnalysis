from __future__ import annotations

import asyncio

import pytest

from hlanalysis.engine.event_bus import EventBus
from hlanalysis.engine.risk_events import RiskVeto


@pytest.mark.asyncio
async def test_fanout_to_two_subscribers():
    bus = EventBus(maxsize=8)
    sub1 = bus.subscribe()
    sub2 = bus.subscribe()
    await bus.publish(RiskVeto(ts_ns=1, reason="cap"))
    e1 = await asyncio.wait_for(sub1.get(), timeout=0.5)
    e2 = await asyncio.wait_for(sub2.get(), timeout=0.5)
    assert e1.reason == "cap" and e2.reason == "cap"


@pytest.mark.asyncio
async def test_slow_consumer_does_not_block_fast_one():
    bus = EventBus(maxsize=2, drop_when_full=True)
    fast = bus.subscribe(maxsize=0)  # unlimited — never considered slow
    slow = bus.subscribe()           # maxsize=2 — will fill up and drop
    # Fill slow's queue
    for i in range(5):
        await bus.publish(RiskVeto(ts_ns=i, reason=str(i)))
    # Fast must still drain everything; slow drops
    drained = []
    while not fast.empty():
        drained.append(await fast.get())
    assert len(drained) == 5
    # Slow received only what fit (2 events) + dropped the rest
    assert slow.qsize() <= 2
