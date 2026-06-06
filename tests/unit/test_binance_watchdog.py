"""SHR-60: the Binance reference feed needs a per-frame staleness watchdog.

`async for raw in ws` has no per-stream staleness detection — ping_interval/
timeout only catch a dead TCP layer. A half-open or silently-idle stream (the
geo-blocked "subscribed but silent" case) keeps the socket open and blocks
forever with no exception and no reconnect. The receive path must force-close
(→ reconnect) and emit a feed_stale health event if no frame arrives within the
watchdog window. Thanks to wait_for, the same window is also the first-data
deadline.
"""
from __future__ import annotations

import asyncio

import pytest

from hlanalysis.adapters.binance import BinanceAdapter


class _SilentWS:
    """recv() never returns a frame — models a half-open/idle stream."""

    async def recv(self):
        await asyncio.sleep(3600)


class _OneFrameThenSilentWS:
    """Delivers one (unparseable) frame, then goes silent."""

    def __init__(self) -> None:
        self._sent = False

    async def recv(self):
        if not self._sent:
            self._sent = True
            return "not-json"  # exercises the JSONDecodeError continue path
        await asyncio.sleep(3600)


async def _drain_kinds(q: asyncio.Queue) -> list[str]:
    kinds = []
    while not q.empty():
        kinds.append(q.get_nowait().kind)
    return kinds


@pytest.mark.asyncio
async def test_watchdog_emits_feed_stale_and_returns_on_silent_stream():
    adapter = BinanceAdapter()
    adapter.stale_timeout_s = 0.01
    q: asyncio.Queue = asyncio.Queue()

    await asyncio.wait_for(
        adapter._recv_until_stale(_SilentWS(), lambda m, t: [], "perp", q), timeout=1.0,
    )

    kinds = await _drain_kinds(q)
    assert any(k.startswith("feed_stale") for k in kinds), kinds


@pytest.mark.asyncio
async def test_watchdog_fires_after_a_frame_then_silence():
    adapter = BinanceAdapter()
    adapter.stale_timeout_s = 0.01
    q: asyncio.Queue = asyncio.Queue()

    await asyncio.wait_for(
        adapter._recv_until_stale(_OneFrameThenSilentWS(), lambda m, t: [], "perp", q),
        timeout=1.0,
    )

    kinds = await _drain_kinds(q)
    assert any(k.startswith("feed_stale") for k in kinds), kinds
