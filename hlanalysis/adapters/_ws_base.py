"""Shared websocket plumbing for every venue adapter.

The connect → subscribe → recv → reconnect-with-exponential-backoff loop used to
be copy-pasted in each of binance.py / hyperliquid.py / polymarket.py, and only
binance carried the per-frame staleness watchdog (SHR-60). A half-open or
silently-idle stream keeps the socket open and `recv()` blocks forever with no
exception and no reconnect — `ping_interval/timeout` only catch a dead TCP
layer. Centralising the loop here means ALL three venues get:

  - the per-frame staleness watchdog (force-reconnect + ``feed_stale`` health
    when no frame arrives within ``stale_timeout_s``; also the first-data
    deadline for the "subscribed but silent" geo-blocked case),
  - exponential backoff on reconnect,
  - a reconnect circuit-breaker (long cooldown after a burst of reconnects in a
    short window — originally HL-only, see the 2026-05-06 settlement blackout),
  - health events tagged with the venue's real product type.

Subclasses provide, per websocket connection, a ``subscribe(ws)`` coroutine and
a ``handle(msg, recv_ns) -> list[NormalizedEvent]`` callable, and drive
everything through :meth:`_run_ws`.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable

import websockets

log = logging.getLogger(__name__)

from .._fastjson import decode as _json_decode
from ..events import HealthEvent, Mechanism, NormalizedEvent, ProductType
from .base import VenueAdapter

# Type aliases for the per-connection hooks.
HandleFn = Callable[[dict, int], list[NormalizedEvent]]
SubscribeFn = Callable[[object], Awaitable[None]]
ConnectFn = Callable[[str], object]  # url -> async context manager yielding a ws


class BaseWsAdapter(VenueAdapter):
    """Base for websocket venue adapters; owns connect/backoff/health/watchdog."""

    venue: str  # set by subclasses

    # Per-frame staleness watchdog (SHR-60). If no frame arrives within this
    # window the stream is force-closed (→ reconnect). Binance/PM/HL all deliver
    # frames multiple times per second under normal conditions, so a frameless
    # window is unambiguously a dead/half-open stream — not a quiet market. Also
    # acts as the first-data deadline ("subscribed but silent").
    stale_timeout_s: float = 30.0

    # Decode-failure visibility. A sustained run of bad frames means the venue
    # changed its wire format. We don't log every frame (could be high-rate) but
    # emit one warning per `_decode_warn_every` failures once past the initial
    # threshold, and keep an instance counter so ops can see it in metrics.
    _decode_fail_warn_threshold: int = 5
    _decode_fail_warn_every: int = 100

    def __init__(self) -> None:
        self.decode_failures: int = 0

    # Reconnect circuit-breaker: if this many reconnects happen within the
    # window, sleep a long cooldown instead of thrashing. Tripped during the
    # 2026-05-06 06:00 UTC HIP-4 settlement, where 478 reconnects in 30 min
    # produced a 12-min blackout across the most-informative moment of the day.
    reconnect_window_s: float = 60.0
    reconnect_threshold: int = 8
    reconnect_cooldown_s: float = 300.0

    # Exponential-backoff bounds for ordinary reconnects.
    reconnect_backoff_initial_s: float = 1.0
    reconnect_backoff_max_s: float = 30.0

    # Default product type stamped onto health events (overridable per-call so a
    # spot/PM adapter doesn't mislabel its health as PERP).
    health_product_type: ProductType = ProductType.PERP

    # ---- connection ------------------------------------------------------

    def _connect(self, url: str):
        """Return an async context manager yielding a websocket.

        Override (or pass ``connect=`` to :meth:`_run_ws`) to inject a factory
        — e.g. Polymarket's tests, or a non-default ping config.
        """
        return websockets.connect(
            url, ping_interval=20, ping_timeout=20, max_size=2**24
        )

    def _health(
        self, kind: str, detail: str = "", product_type: ProductType | None = None
    ) -> HealthEvent:
        now = time.time_ns()
        return HealthEvent(
            venue=self.venue,
            product_type=product_type or self.health_product_type,
            mechanism=Mechanism.CLOB,
            symbol="*",
            exchange_ts=now,
            local_recv_ts=now,
            kind=kind,
            detail=detail,
        )

    # ---- receive loop (the watchdog) -------------------------------------

    async def _recv_until_stale(
        self,
        ws,
        handle: HandleFn,
        label: str,
        queue: asyncio.Queue[NormalizedEvent],
        *,
        product_type: ProductType | None = None,
        after_message: Callable[[object, dict], Awaitable[None]] | None = None,
    ) -> None:
        """Consume frames until the per-frame watchdog trips (SHR-60).

        Bounds every ``recv()`` with ``stale_timeout_s``: on timeout we emit a
        ``feed_stale`` health event and return, which exits the caller's
        ``async with connect(...)`` and forces a reconnect. The first ``recv()``
        is bounded too, so "subscribed but never delivered" is caught as a
        first-data deadline. ``after_message`` (HL) runs post-dispatch with the
        live ws so it can (un)subscribe in reaction to a frame.
        """
        suffix = f"/{label}" if label else ""
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=self.stale_timeout_s)
            except asyncio.TimeoutError:
                await queue.put(
                    self._health(
                        f"feed_stale{suffix}",
                        f"no frame in {self.stale_timeout_s:.0f}s",
                        product_type,
                    )
                )
                return
            recv_ns = time.time_ns()
            try:
                msg = _json_decode(raw)
            except (ValueError, TypeError):
                self.decode_failures += 1
                n = self.decode_failures
                if n == self._decode_fail_warn_threshold or (
                    n > self._decode_fail_warn_threshold
                    and (n - self._decode_fail_warn_threshold)
                    % self._decode_fail_warn_every == 0
                ):
                    log.warning(
                        "%s: %d JSON decode failures so far — venue wire format may have changed",
                        self.venue, n,
                    )
                continue
            for ev in handle(msg, recv_ns):
                await queue.put(ev)
            if after_message is not None:
                await after_message(ws, msg)

    # ---- reconnect loop --------------------------------------------------

    async def _run_ws(
        self,
        *,
        url: str,
        subscribe: SubscribeFn,
        handle: HandleFn,
        queue: asyncio.Queue[NormalizedEvent],
        label: str = "",
        connect: ConnectFn | None = None,
        product_type: ProductType | None = None,
        before_connect: Callable[[], Awaitable[None]] | None = None,
        after_message: Callable[[object, dict], Awaitable[None]] | None = None,
        circuit_breaker: bool = True,
    ) -> None:
        """Run one websocket connection forever: connect, subscribe, receive with
        the staleness watchdog, and reconnect with exponential backoff + an
        optional circuit-breaker. All events/health are pushed to ``queue``.

        ``before_connect`` runs before each (re)connect attempt (HL re-expands
        its HIP-4 wildcards there so a roll that happened while disconnected is
        picked up). Exceptions are turned into a ``reconnect`` health event; the
        loop only exits via cancellation.
        """
        connect = connect or self._connect
        suffix = f"/{label}" if label else ""
        backoff = self.reconnect_backoff_initial_s
        reconnect_times: deque[float] = deque(maxlen=self.reconnect_threshold + 1)
        while True:
            if before_connect is not None:
                await before_connect()
            try:
                async with connect(url) as ws:
                    backoff = self.reconnect_backoff_initial_s
                    await queue.put(self._health(f"connected{suffix}", "", product_type))
                    await subscribe(ws)
                    await self._recv_until_stale(
                        ws, handle, label, queue,
                        product_type=product_type, after_message=after_message,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - any ws/transport error → reconnect
                now = time.monotonic()
                await queue.put(
                    self._health(f"reconnect{suffix}", str(e)[:200], product_type)
                )
                if circuit_breaker:
                    reconnect_times.append(now)
                    if (
                        len(reconnect_times) >= self.reconnect_threshold
                        and (now - reconnect_times[0]) < self.reconnect_window_s
                    ):
                        await queue.put(
                            self._health(
                                "circuit-breaker",
                                f"{len(reconnect_times)} reconnects in "
                                f"{now - reconnect_times[0]:.0f}s",
                                product_type,
                            )
                        )
                        await asyncio.sleep(self.reconnect_cooldown_s)
                        reconnect_times.clear()
                        backoff = self.reconnect_backoff_initial_s
                        continue
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.reconnect_backoff_max_s)
