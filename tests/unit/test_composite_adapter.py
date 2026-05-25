"""CompositeAdapter merges multiple VenueAdapter streams into one async
iterator. Engine ingest loop sees a single stream; under the hood each
child adapter only receives subs matching its own `venue`."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from hlanalysis.adapters.base import VenueAdapter
from hlanalysis.adapters.composite import CompositeAdapter
from hlanalysis.config import Subscription
from hlanalysis.events import (
    Mechanism,
    NormalizedEvent,
    ProductType,
    TradeEvent,
)


def _trade(venue: str, symbol: str, price: float) -> TradeEvent:
    return TradeEvent(
        venue=venue,
        product_type=ProductType.PERP if venue == "hyperliquid" else ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol=symbol,
        exchange_ts=0,
        local_recv_ts=0,
        price=price,
        size=1.0,
        side="buy",
    )


class _StubAdapter(VenueAdapter):
    """Yields a fixed list of events for subs that target its `venue`. Records
    the subs it was handed so the test can assert per-venue filtering."""

    def __init__(self, venue: str, events: list[NormalizedEvent]) -> None:
        self.venue = venue
        self._events = events
        self.received_subs: list[Subscription] = []

    def supports(self, p: ProductType, m: Mechanism) -> bool:
        return True

    async def stream(
        self, subscriptions: list[Subscription],
    ) -> AsyncIterator[NormalizedEvent]:
        self.received_subs = list(subscriptions)
        for ev in self._events:
            await asyncio.sleep(0)  # cooperative yield so interleaving is exercised
            yield ev
        # Idle forever so the consumer can break on count rather than EOF.
        while True:
            await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_composite_interleaves_events_from_both_adapters():
    hl = _StubAdapter("hyperliquid", [
        _trade("hyperliquid", "BTC", 100.0),
        _trade("hyperliquid", "BTC", 101.0),
    ])
    pm = _StubAdapter("polymarket", [
        _trade("polymarket", "tok-yes", 0.5),
        _trade("polymarket", "tok-yes", 0.51),
    ])
    composite = CompositeAdapter([hl, pm])

    hl_sub = Subscription(
        venue="hyperliquid", product_type=ProductType.PERP,
        mechanism=Mechanism.CLOB, symbol="BTC", channels=("trades",),
    )
    pm_sub = Subscription(
        venue="polymarket", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="*", channels=("trades",),
    )

    received: list[NormalizedEvent] = []
    async def _drain() -> None:
        async for ev in composite.stream([hl_sub, pm_sub]):
            received.append(ev)
            if len(received) >= 4:
                return

    await asyncio.wait_for(_drain(), timeout=2.0)

    # All 4 events surfaced through the composite stream.
    assert len(received) == 4
    venues = {ev.venue for ev in received}
    assert venues == {"hyperliquid", "polymarket"}
    # Each child saw only its own venue's subs.
    assert [s.venue for s in hl.received_subs] == ["hyperliquid"]
    assert [s.venue for s in pm.received_subs] == ["polymarket"]


@pytest.mark.asyncio
async def test_composite_skips_adapter_with_no_matching_subs():
    """If a sub list contains zero subs for an adapter's venue, that adapter
    must NOT be invoked (no spurious WS connect)."""
    hl = _StubAdapter("hyperliquid", [_trade("hyperliquid", "BTC", 100.0)])
    pm = _StubAdapter("polymarket", [])
    composite = CompositeAdapter([hl, pm])

    hl_sub = Subscription(
        venue="hyperliquid", product_type=ProductType.PERP,
        mechanism=Mechanism.CLOB, symbol="BTC", channels=("trades",),
    )

    received: list[NormalizedEvent] = []
    async def _drain() -> None:
        async for ev in composite.stream([hl_sub]):
            received.append(ev)
            return

    await asyncio.wait_for(_drain(), timeout=2.0)
    assert len(received) == 1
    # PM adapter received no subs — stream() was never asked to produce.
    assert pm.received_subs == []
