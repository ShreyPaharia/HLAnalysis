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

from .._fastjson import decode as _json_decode

import websockets

from ..config import Subscription
from ..events import Mechanism, NormalizedEvent, ProductType
from ._ws_base import BaseWsAdapter
from .polymarket_gamma import GammaClient
from .polymarket_normalize import (
    PmBook,
    parse_book_message,
    parse_gamma_event_to_bucket_question_meta,
    parse_gamma_market_to_question_meta,
    parse_gamma_market_to_settlement,
    parse_price_change_message,
    parse_trade_message,
)

log = logging.getLogger(__name__)

_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
_GAMMA_POLL_S = 60


class PolymarketAdapter(BaseWsAdapter):
    venue = "polymarket"
    health_product_type = ProductType.PREDICTION_BINARY

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
        # SHR-62: per-asset stateful L2 book so price_change deltas emit MERGED
        # full snapshots rather than partial ones.  MarketState.apply treats
        # BookSnapshotEvent as a full replace, so a partial emit would corrupt
        # the book to 1-2 phantom levels between full frames.
        self._pm_books: dict[str, PmBook] = {}

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

        async def _poll_one_series(sub: Subscription, series_str: str) -> None:
            # Per-subscription underlying (BTC / ETH / NVDA / NBA / …).
            # Default to "BTC" only for legacy single-subscription configs.
            underlying = (sub.match or {}).get("underlying", "BTC")
            if not isinstance(underlying, str):
                underlying = underlying[0] if underlying else "BTC"
            # Offload ONLY the blocking 30s requests.get to a worker thread;
            # all cache mutation (known_conditions / active_tokens) and
            # queue puts below stay on the event-loop thread to avoid races.
            klass = (sub.match or {}).get("class")
            if not isinstance(klass, str):
                klass = klass[0] if klass else "priceBinary"
            events = await asyncio.to_thread(
                self._gamma.fetch_events,
                series_slug=series_str, closed=False,
            )
            if klass == "priceBucket":
                # Group the entire multi-strike event into one priceBucket
                # QuestionMetaEvent; subscribe ALL leg tokens so the WS
                # receives book/trade frames for every leg. Per-leg
                # settlement uses parse_gamma_market_to_settlement which
                # names the winning token, routed by leg-symbol membership.
                for ev in events:
                    cond_id = str(ev.get("id") or ev.get("slug") or "")
                    is_new = cond_id not in known_conditions
                    known_conditions[cond_id] = ev
                    if is_new:
                        qmeta = parse_gamma_event_to_bucket_question_meta(
                            ev, series_slug=series_str,
                            local_recv_ts=time.time_ns(),
                            underlying=underlying,
                        )
                        if qmeta is not None:
                            await queue.put(qmeta)
                            for mk in ev.get("markets") or []:
                                toks = _json_decode(mk.get("clobTokenIds") or "[]")
                                active_tokens.update(str(t) for t in toks)
                    # Per-leg settlement: each sub-market resolves
                    # independently; _mark_settled routes by leg-symbol.
                    for mk in ev.get("markets") or []:
                        settle = parse_gamma_market_to_settlement(
                            mk, series_slug=series_str,
                            local_recv_ts=time.time_ns(),
                        )
                        if settle is not None:
                            await queue.put(settle)
                return
            for mk in self._gamma.iter_binary_markets(events):
                cond_id = str(mk["conditionId"])
                is_new = cond_id not in known_conditions
                known_conditions[cond_id] = mk
                if is_new:
                    qmeta = parse_gamma_market_to_question_meta(
                        mk, series_slug=series_str,
                        local_recv_ts=time.time_ns(),
                        underlying=underlying,
                    )
                    await queue.put(qmeta)
                    toks = _json_decode(mk["clobTokenIds"])
                    active_tokens.update(str(t) for t in toks)
                settle = parse_gamma_market_to_settlement(
                    mk, series_slug=series_str,
                    local_recv_ts=time.time_ns(),
                )
                if settle is not None:
                    await queue.put(settle)

        async def _gamma_poll_once() -> None:
            for sub in pm_subs:
                series = (sub.match or {}).get("series_slug")
                if not series:
                    continue
                series_str = series if isinstance(series, str) else series[0]
                # Isolate per-subscription failures: an intermittent Gamma 500
                # (or any parse error) on ONE series must not abort the whole
                # poll cycle and starve the remaining subscriptions. Without
                # this, a `eth-up-or-down-daily` 500 silently suppressed the BTC
                # multi-strike bucket subscriptions for the rest of the cycle.
                try:
                    await _poll_one_series(sub, series_str)
                except Exception:
                    log.exception(
                        "gamma poll failed for series=%s; skipping it this "
                        "cycle so other subscriptions still poll", series_str,
                    )
                    continue

        async def _gamma_loop() -> None:
            while True:
                try:
                    await _gamma_poll_once()
                except Exception:
                    log.exception("gamma poll crashed")
                await asyncio.sleep(_GAMMA_POLL_S)

        async def _subscribe(ws) -> None:
            # SHR-62: reset per-asset books on each (re)connect so a stale
            # local book cannot survive a feed gap.  The first "book" message
            # after resubscription will re-seed each asset's book from scratch.
            self._pm_books.clear()
            await ws.send(json.dumps({
                "type": "market",
                "assets_ids": sorted(active_tokens),
            }))
            await queue.put(self._health(
                "subscribed", f"{len(active_tokens)} tokens",
            ))

        async def _ws_loop() -> None:
            await _gamma_poll_once()
            if not active_tokens:
                await queue.put(
                    self._health("no_active_markets",
                                 "Gamma returned 0 active markets")
                )
                return
            # BaseWsAdapter owns connect/backoff/circuit-breaker and the
            # per-frame staleness watchdog — PM previously used a bare recv()
            # with NO half-open-socket detection (SHR-60 was binance-only).
            await self._run_ws(
                url=_WS_URL,
                subscribe=_subscribe,
                handle=self._handle,
                queue=queue,
                connect=self._ws_factory,
            )

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

    def _handle(self, msg: Any, recv_ns: int) -> list[NormalizedEvent]:
        """Normalize one decoded PM frame into events.

        A single PM WS message often carries an *array* of payloads (e.g. one
        `book` snapshot per asset_id on initial connect); we unwrap that
        transparently. The base receive loop already JSON-decodes the frame and
        drops non-JSON keepalives ("PONG"), so we only see parsed structures.
        """
        items = msg if isinstance(msg, list) else [msg]
        out: list[NormalizedEvent] = []
        for p in items:
            t = p.get("event_type")
            try:
                if t == "book":
                    # SHR-62: apply_book resets the per-asset book and emits a
                    # full merged snapshot (identical to the old stateless path
                    # for a full "book" frame, but now the book is seeded for
                    # subsequent price_change deltas).
                    asset_id = str(p.get("asset_id", ""))
                    if asset_id not in self._pm_books:
                        self._pm_books[asset_id] = PmBook()
                    out.append(self._pm_books[asset_id].apply_book(
                        p, local_recv_ts=recv_ns
                    ))
                elif t == "price_change":
                    # SHR-62: merge delta into the full book; emit the merged
                    # snapshot so MarketState receives a complete replace rather
                    # than a 1-2 level phantom.  Falls back to the stateless
                    # path if no prior "book" frame was seen for this asset.
                    asset_id = str(p.get("asset_id", ""))
                    if asset_id in self._pm_books:
                        ev = self._pm_books[asset_id].apply_price_change(
                            p, local_recv_ts=recv_ns
                        )
                    else:
                        # No prior full snapshot for this asset — fall back to
                        # the stateless partial emit so we don't drop the delta
                        # entirely. The next "book" frame will reseed the book.
                        ev = parse_price_change_message(p, local_recv_ts=recv_ns)
                    if ev is not None:
                        out.append(ev)
                elif t == "last_trade_price":
                    out.append(parse_trade_message(p, local_recv_ts=recv_ns))
                # Unknown event_types (e.g. tick_size_change) ignored — not
                # needed by the strategy or recorder.
            except (KeyError, ValueError, TypeError) as e:
                log.warning("pm ws: malformed %s frame discarded: %s", t, e)
        return out
