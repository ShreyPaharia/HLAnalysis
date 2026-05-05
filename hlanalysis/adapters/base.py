from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..config import Subscription
from ..events import Mechanism, NormalizedEvent, ProductType


class VenueAdapter(ABC):
    """One adapter per exchange.

    The adapter is responsible for:
      - Negotiating its own websocket / REST connection(s).
      - Translating venue-specific messages into NormalizedEvent.
      - Reconnecting on failure and emitting HealthEvent on transitions.
      - Tagging events with the correct (product_type, mechanism, symbol).
    """

    venue: str  # class-level constant set by subclasses

    @abstractmethod
    def supports(self, product_type: ProductType, mechanism: Mechanism) -> bool: ...

    @abstractmethod
    async def stream(
        self, subscriptions: list[Subscription]
    ) -> AsyncIterator[NormalizedEvent]: ...
