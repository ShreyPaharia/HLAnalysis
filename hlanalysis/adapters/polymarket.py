"""Polymarket CLOB live-data adapter.

Inputs: list[Subscription] (typically a single wildcard for the BTC Up/Down
daily series). Outputs: AsyncIterator[NormalizedEvent] — BookSnapshotEvent,
TradeEvent, QuestionMetaEvent, SettlementEvent.

Architecture:
  - On startup: poll Gamma /events to learn the currently-active markets;
    emit one QuestionMetaEvent per match-filtered market.
  - Subscribe to the PM CLOB WS for all matched token IDs (yes + no legs).
  - Receive loop: dispatch by `event_type` to the normalizers in
    polymarket_normalize.py and yield results to the consumer.
  - Background poller: re-fetch Gamma every 60s to pick up new daily
    markets and resolved settlements. Resolved markets emit a
    SettlementEvent; new markets emit a QuestionMetaEvent + subscribe their
    new token IDs.

Observed live-wire deviations (validated against the recorded fixture in
`tests/fixtures/pm/ws_book_frames.jsonl`):
  - PM frames are JSON; a single WS message often carries an *array* of
    payloads (e.g. one `book` snapshot per asset_id in the subscribe set on
    initial connect). `_dispatch_frame` unwraps the array case transparently.
  - PM occasionally sends literal "PONG" string keepalives; non-JSON frames
    are discarded with a debug log.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import websockets

from ..config import Subscription
from ..events import HealthEvent, Mechanism, NormalizedEvent, ProductType
from .base import VenueAdapter
from .polymarket_gamma import GammaClient
from .polymarket_normalize import (
    parse_book_message,
    parse_gamma_market_to_question_meta,
    parse_gamma_market_to_settlement,
    parse_price_change_message,
    parse_trade_message,
)

log = logging.getLogger(__name__)

_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
_GAMMA_POLL_S = 60


class PolymarketAdapter(VenueAdapter):
    venue = "polymarket"

    def __init__(
        self,
        *,
        ws_factory: Callable[[str], Any] | None = None,
        gamma_client: GammaClient | None = None,
    ) -> None:
        self._ws_factory = ws_factory or (
            lambda url: websockets.connect(url, ping_interval=30)
        )
        self._gamma = gamma_client or GammaClient()

    def supports(self, product_type: ProductType, mechanism: Mechanism) -> bool:
        return (
            product_type == ProductType.PREDICTION_BINARY
            and mechanism == Mechanism.CLOB
        )

    async def stream(
        self, subscriptions: list[Subscription]
    ) -> AsyncIterator[NormalizedEvent]:
        pm_subs = [s for s in subscriptions if s.venue == self.venue]
        if not pm_subs:
            return

        queue: asyncio.Queue[NormalizedEvent] = asyncio.Queue(maxsize=10000)
        active_tokens: set[str] = set()
        known_conditions: dict[str, dict] = {}

        async def _gamma_poll_once() -> None:
            for sub in pm_subs:
                series = (sub.match or {}).get("series_slug")
                if not series:
                    continue
                series_str = series if isinstance(series, str) else series[0]
                events = self._gamma.fetch_events(
                    series_slug=series_str, closed=False,
                )
                for mk in self._gamma.iter_binary_markets(events):
                    cond_id = str(mk["conditionId"])
                    is_new = cond_id not in known_conditions
                    known_conditions[cond_id] = mk
                    if is_new:
                        qmeta = parse_gamma_market_to_question_meta(
                            mk, series_slug=series_str,
                            local_recv_ts=time.time_ns(),
                        )
                        await queue.put(qmeta)
                        toks = json.loads(mk["clobTokenIds"])
                        active_tokens.update(str(t) for t in toks)
                    settle = parse_gamma_market_to_settlement(
                        mk, series_slug=series_str,
                        local_recv_ts=time.time_ns(),
                    )
                    if settle is not None:
                        await queue.put(settle)

        async def _gamma_loop() -> None:
            while True:
                try:
                    await _gamma_poll_once()
                except Exception:
                    log.exception("gamma poll crashed")
                await asyncio.sleep(_GAMMA_POLL_S)

        async def _ws_loop() -> None:
            await _gamma_poll_once()
            if not active_tokens:
                await queue.put(
                    self._health("no_active_markets",
                                 "Gamma returned 0 active markets")
                )
                return
            ws_ctx = self._ws_factory(_WS_URL)
            async with ws_ctx as ws:
                await ws.send(json.dumps({
                    "type": "market",
                    "assets_ids": sorted(active_tokens),
                }))
                while True:
                    raw = await ws.recv()
                    self._dispatch_frame(raw, queue)

        tasks = [
            asyncio.create_task(_ws_loop()),
            asyncio.create_task(_gamma_loop()),
        ]
        try:
            while True:
                ev = await queue.get()
                yield ev
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def _dispatch_frame(
        self, raw: str | bytes, queue: asyncio.Queue,
    ) -> None:
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                log.warning("pm ws: undecodable bytes discarded")
                return
        try:
            payloads = json.loads(raw)
        except json.JSONDecodeError:
            # PM sends literal "PONG" keepalives between bursts; not JSON.
            log.debug("pm ws: non-json frame discarded (%r)", raw[:40])
            return
        items = payloads if isinstance(payloads, list) else [payloads]
        now = time.time_ns()
        for p in items:
            t = p.get("event_type")
            try:
                if t == "book":
                    queue.put_nowait(parse_book_message(p, local_recv_ts=now))
                elif t == "price_change":
                    ev = parse_price_change_message(p, local_recv_ts=now)
                    if ev is not None:
                        queue.put_nowait(ev)
                elif t == "last_trade_price":
                    queue.put_nowait(parse_trade_message(p, local_recv_ts=now))
                # Unknown event_types (e.g. tick_size_change) ignored — not
                # needed by the strategy or recorder.
            except (KeyError, ValueError, TypeError) as e:
                log.warning("pm ws: malformed %s frame discarded: %s", t, e)

    def _health(self, kind: str, detail: str) -> HealthEvent:
        return HealthEvent(
            venue=self.venue,
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="*",
            exchange_ts=time.time_ns(),
            local_recv_ts=time.time_ns(),
            kind=kind,
            detail=detail,
        )
