"""Merges multiple VenueAdapter streams into a single AsyncIterator.

Each child adapter receives only its own subs (filtered by `venue`); their
event streams are interleaved fairly via an asyncio.Queue. The engine's
ingest loop sees one merged stream and doesn't need to care that
hyperliquid and polymarket are separate WS connections under the hood.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from ..config import Subscription
from ..events import Mechanism, NormalizedEvent, ProductType
from .base import VenueAdapter


class CompositeAdapter(VenueAdapter):
    venue = "composite"

    def __init__(self, adapters: list[VenueAdapter]) -> None:
        self._adapters = adapters

    def supports(self, p: ProductType, m: Mechanism) -> bool:
        return any(a.supports(p, m) for a in self._adapters)

    async def stream(
        self, subscriptions: list[Subscription],
    ) -> AsyncIterator[NormalizedEvent]:
        queue: asyncio.Queue[NormalizedEvent] = asyncio.Queue(maxsize=10000)

        async def _drain(adapter: VenueAdapter) -> None:
            subs = [s for s in subscriptions if s.venue == adapter.venue]
            if not subs:
                return
            async for ev in adapter.stream(subs):
                await queue.put(ev)

        tasks = [asyncio.create_task(_drain(a)) for a in self._adapters]
        try:
            while True:
                yield await queue.get()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
