from __future__ import annotations

import asyncio
from collections.abc import Iterable

from loguru import logger

from .risk_events import BusEvent


class EventBus:
    """In-process fanout pub/sub backed by asyncio.Queue per subscriber.

    Subscribers receive their own Queue from `subscribe()`. `publish()` enqueues
    to every subscriber. If `drop_when_full`, slow consumers' overflow is dropped
    with a warn log (live trading must not block on alerts).
    """

    def __init__(self, *, maxsize: int = 1024, drop_when_full: bool = True) -> None:
        self._subs: list[asyncio.Queue[BusEvent]] = []
        self._maxsize = maxsize
        self._drop_when_full = drop_when_full

    def subscribe(self, maxsize: int | None = None) -> asyncio.Queue[BusEvent]:
        effective_maxsize = maxsize if maxsize is not None else self._maxsize
        q: asyncio.Queue[BusEvent] = asyncio.Queue(maxsize=effective_maxsize)
        self._subs.append(q)
        return q

    async def publish(self, ev: BusEvent) -> None:
        # Surface every bus event in journalctl so post-mortems aren't limited
        # to Telegram (ephemeral, no search) and gate_decisions.jsonl
        # (gate-side only). Pydantic's model_dump_json gives a compact
        # one-line representation that survives the discriminator. INFO level
        # so routine operation is searchable without flipping the engine to
        # debug — the noisy stuff (heartbeat, topup_skip) is logged
        # separately at the appropriate level.
        try:
            logger.info("bus {} {}", ev.kind, ev.model_dump_json())
        except Exception:
            # Defensive: a logging failure must never block a bus publish.
            logger.exception("event_bus: failed to log {}", ev.kind)
        for q in self._subs:
            if self._drop_when_full and q.full():
                logger.warning(
                    "event_bus: slow consumer; dropping {} (qsize={})", ev.kind, q.qsize()
                )
                continue
            await q.put(ev)

    async def publish_many(self, evs: Iterable[BusEvent]) -> None:
        for ev in evs:
            await self.publish(ev)
