from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

import requests
import websockets

from ..config import Subscription
from ..events import (
    BboEvent,
    BookSnapshotEvent,
    FundingEvent,
    HealthEvent,
    LiquidationEvent,
    MarkEvent,
    Mechanism,
    NormalizedEvent,
    ProductType,
    TradeEvent,
)
from .base import VenueAdapter

log = logging.getLogger(__name__)

SPOT_WS = "wss://stream.binance.com:9443/ws"
PERP_WS = "wss://fstream.binance.com/ws"
PERP_REST_PREMIUM_INDEX = "https://fapi.binance.com/fapi/v1/premiumIndex"
PERP_MARK_POLL_INTERVAL_S = 3.0

# Several perp WS streams (aggTrade, markPrice, forceOrder, kline_*) are silently
# unavailable from non-allowlisted IPs (e.g. US residential), even though REST works
# and `bookTicker`/`depth20`/`@trade` deliver fine. We use `@trade` for perp trades
# and poll REST for mark+funding. Liquidations are dropped (no public REST equivalent).
_STREAMS_PERP: dict[str, str] = {
    "trades": "trade",
    "book": "depth20@100ms",
    "bbo": "bookTicker",
}
_STREAMS_SPOT: dict[str, str] = {
    "trades": "trade",
    "book": "depth20@100ms",
    "bbo": "bookTicker",
}
# Channels handled out-of-band (REST poll) for perp.
_PERP_REST_CHANNELS = {"mark", "funding"}


class BinanceAdapter(VenueAdapter):
    venue = "binance"

    def supports(self, product_type: ProductType, mechanism: Mechanism) -> bool:
        return mechanism == Mechanism.CLOB and product_type in {
            ProductType.PERP,
            ProductType.SPOT,
        }

    async def stream(
        self, subscriptions: list[Subscription]
    ) -> AsyncIterator[NormalizedEvent]:
        spot_subs = [s for s in subscriptions if s.product_type == ProductType.SPOT]
        perp_subs = [s for s in subscriptions if s.product_type == ProductType.PERP]

        queue: asyncio.Queue[NormalizedEvent] = asyncio.Queue(maxsize=10000)
        tasks: list[asyncio.Task] = []
        if spot_subs:
            tasks.append(
                asyncio.create_task(
                    self._run_one(SPOT_WS, spot_subs, _STREAMS_SPOT, queue, "spot")
                )
            )
        if perp_subs:
            tasks.append(
                asyncio.create_task(
                    self._run_one(PERP_WS, perp_subs, _STREAMS_PERP, queue, "perp")
                )
            )
            # Mark/funding via REST polling because the WS streams are IP-restricted.
            rest_subs = [
                s for s in perp_subs
                if any(c in _PERP_REST_CHANNELS for c in s.channels)
            ]
            if rest_subs:
                tasks.append(asyncio.create_task(self._poll_perp_premium(rest_subs, queue)))
        try:
            while True:
                ev = await queue.get()
                yield ev
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_one(
        self,
        ws_url: str,
        subscriptions: list[Subscription],
        stream_map: dict[str, str],
        queue: asyncio.Queue[NormalizedEvent],
        label: str,
    ) -> None:
        sym_to_sub = {s.symbol.upper(): s for s in subscriptions}
        # Build set of stream names like "btcusdt@aggTrade"
        streams: list[str] = []
        for sub in subscriptions:
            sym_lc = sub.symbol.lower()
            for ch in sub.channels:
                suffix = stream_map.get(ch)
                if suffix is None:
                    log.debug("binance/%s: channel %r not supported, skipping", label, ch)
                    continue
                streams.append(f"{sym_lc}@{suffix}")
        # Dedup while preserving order (same stream listed multiple times = duplicate frames).
        streams = list(dict.fromkeys(streams))

        backoff = 1.0
        while True:
            try:
                async with websockets.connect(
                    ws_url, ping_interval=20, ping_timeout=20, max_size=2**24
                ) as ws:
                    backoff = 1.0
                    await queue.put(self._health(f"connected/{label}", ""))
                    await ws.send(
                        json.dumps({"method": "SUBSCRIBE", "params": streams, "id": 1})
                    )
                    async for raw in ws:
                        recv_ns = time.time_ns()
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        for ev in self._handle(msg, recv_ns, sym_to_sub, label):
                            await queue.put(ev)
            except Exception as e:
                log.warning("binance/%s ws error: %s; backoff=%.1fs", label, e, backoff)
                await queue.put(self._health(f"reconnect/{label}", str(e)))
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _handle(
        self,
        msg: dict,
        recv_ns: int,
        sym_to_sub: dict[str, Subscription],
        label: str,
    ) -> list[NormalizedEvent]:
        # SUBSCRIBE response: {"result":null,"id":1}
        if "result" in msg and "id" in msg:
            return [self._health(f"subscribed/{label}", json.dumps(msg))]

        e = msg.get("e")
        symbol = (msg.get("s") or "").upper()
        sub = sym_to_sub.get(symbol)
        out: list[NormalizedEvent] = []

        # Spot bookTicker has no "e" field; identify by "u" + "b" + "a".
        # Binance Spot's @bookTicker payload is {u, s, b, B, a, A} with NO timestamp
        # field — neither E nor T. We therefore cannot record an exchange-side ts here;
        # use 0 as an explicit "exchange ts unavailable" sentinel rather than copying
        # recv_ns (which would silently masquerade as a real exchange ts and produce
        # an apparent zero `local_recv_ts - exchange_ts` latency in analytics).
        if e is None and "b" in msg and "a" in msg and "s" in msg:
            sub = sym_to_sub.get(msg["s"].upper())
            if sub is None:
                return out
            out.append(
                BboEvent(
                    venue=self.venue,
                    product_type=sub.product_type,
                    mechanism=sub.mechanism,
                    symbol=msg["s"].upper(),
                    exchange_ts=0,  # not provided by Binance spot @bookTicker
                    local_recv_ts=recv_ns,
                    seq=int(msg.get("u", 0)),
                    bid_px=float(msg["b"]),
                    bid_sz=float(msg["B"]),
                    ask_px=float(msg["a"]),
                    ask_sz=float(msg["A"]),
                )
            )
            return out

        # Spot partial depth has no "e" but has lastUpdateId + bids/asks.
        # Same caveat as spot @bookTicker: <symbol>@depth<levels>[@100ms] payload is
        # {lastUpdateId, bids, asks} with no timestamp. exchange_ts=0 sentinel.
        if e is None and "lastUpdateId" in msg and "bids" in msg and "asks" in msg:
            # Spot depth doesn't carry a symbol — look it up from the active sub if there's
            # exactly one spot symbol; multi-symbol spot needs combined-stream URL form.
            if len(sym_to_sub) != 1:
                return out
            sub = next(iter(sym_to_sub.values()))
            out.append(
                BookSnapshotEvent(
                    venue=self.venue,
                    product_type=sub.product_type,
                    mechanism=sub.mechanism,
                    symbol=sub.symbol.upper(),
                    exchange_ts=0,  # not provided by Binance spot partial-depth stream
                    local_recv_ts=recv_ns,
                    seq=int(msg["lastUpdateId"]),
                    bid_px=[float(b[0]) for b in msg["bids"]],
                    bid_sz=[float(b[1]) for b in msg["bids"]],
                    ask_px=[float(a[0]) for a in msg["asks"]],
                    ask_sz=[float(a[1]) for a in msg["asks"]],
                )
            )
            return out

        if sub is None:
            return out

        common = dict(
            venue=self.venue,
            product_type=sub.product_type,
            mechanism=sub.mechanism,
            symbol=symbol,
            local_recv_ts=recv_ns,
        )

        if e == "trade":  # spot per-fill
            out.append(
                TradeEvent(
                    **common,
                    exchange_ts=int(msg["T"]) * 1_000_000,
                    price=float(msg["p"]),
                    size=float(msg["q"]),
                    side="sell" if msg.get("m") else "buy",
                    trade_id=str(msg.get("t", "")) or None,
                )
            )
        elif e == "aggTrade":  # perp aggregated
            out.append(
                TradeEvent(
                    **common,
                    exchange_ts=int(msg["T"]) * 1_000_000,
                    price=float(msg["p"]),
                    size=float(msg["q"]),
                    side="sell" if msg.get("m") else "buy",
                    trade_id=str(msg.get("a", "")) or None,
                )
            )
        elif e == "bookTicker":  # perp BBO carries event_type marker
            out.append(
                BboEvent(
                    **common,
                    exchange_ts=int(msg.get("T", msg.get("E", 0))) * 1_000_000,
                    seq=int(msg.get("u", 0)),
                    bid_px=float(msg["b"]),
                    bid_sz=float(msg["B"]),
                    ask_px=float(msg["a"]),
                    ask_sz=float(msg["A"]),
                )
            )
        elif e == "depthUpdate":  # perp partial book is sent as snapshot via this stream
            ts = int(msg.get("T", msg.get("E", 0))) * 1_000_000
            out.append(
                BookSnapshotEvent(
                    **common,
                    exchange_ts=ts,
                    seq=int(msg.get("u", 0)),
                    bid_px=[float(b[0]) for b in msg.get("b", [])],
                    bid_sz=[float(b[1]) for b in msg.get("b", [])],
                    ask_px=[float(a[0]) for a in msg.get("a", [])],
                    ask_sz=[float(a[1]) for a in msg.get("a", [])],
                )
            )
        elif e == "markPriceUpdate":  # perp mark + funding bundled
            ts = int(msg.get("E", 0)) * 1_000_000
            out.append(MarkEvent(**common, exchange_ts=ts, mark_px=float(msg["p"])))
            out.append(
                FundingEvent(
                    **common,
                    exchange_ts=ts,
                    funding_rate=float(msg.get("r", 0.0)),
                    next_funding_ts=int(msg["T"]) * 1_000_000 if "T" in msg else None,
                )
            )
        elif e == "forceOrder":
            o = msg.get("o", {})
            order_side = o.get("S", "")
            # Liquidator side = SELL means a long was force-closed, BUY means a short was.
            liq_side: str = "long" if order_side == "SELL" else "short"
            out.append(
                LiquidationEvent(
                    **common,
                    exchange_ts=int(o.get("T", msg.get("E", 0))) * 1_000_000,
                    price=float(o.get("ap") or o.get("p") or 0.0),
                    size=float(o.get("q") or 0.0),
                    side=liq_side,
                )
            )

        return out

    async def _poll_perp_premium(
        self,
        subs: list[Subscription],
        queue: asyncio.Queue[NormalizedEvent],
    ) -> None:
        """Poll /fapi/v1/premiumIndex for mark + funding; emit MarkEvent and FundingEvent."""
        while True:
            for sub in subs:
                try:
                    r = await asyncio.to_thread(
                        requests.get,
                        PERP_REST_PREMIUM_INDEX,
                        params={"symbol": sub.symbol.upper()},
                        timeout=4,
                    )
                    if r.status_code != 200:
                        continue
                    p = r.json()
                except Exception as e:
                    log.warning("binance premiumIndex poll failed: %s", e)
                    continue
                ts_ns = int(p.get("time", time.time() * 1000)) * 1_000_000
                recv_ns = time.time_ns()
                common = dict(
                    venue=self.venue,
                    product_type=sub.product_type,
                    mechanism=sub.mechanism,
                    symbol=sub.symbol.upper(),
                    exchange_ts=ts_ns,
                    local_recv_ts=recv_ns,
                )
                if "mark" in sub.channels and p.get("markPrice") is not None:
                    await queue.put(MarkEvent(**common, mark_px=float(p["markPrice"])))
                if "funding" in sub.channels and p.get("lastFundingRate") is not None:
                    next_funding = (
                        int(p["nextFundingTime"]) * 1_000_000 if p.get("nextFundingTime") else None
                    )
                    await queue.put(
                        FundingEvent(
                            **common,
                            funding_rate=float(p["lastFundingRate"]),
                            premium=float(p["estimatedSettlePrice"])
                            if p.get("estimatedSettlePrice") is not None
                            else None,
                            next_funding_ts=next_funding,
                        )
                    )
            await asyncio.sleep(PERP_MARK_POLL_INTERVAL_S)

    def _health(self, kind: str, detail: str) -> HealthEvent:
        now = time.time_ns()
        return HealthEvent(
            venue=self.venue,
            product_type=ProductType.PERP,
            mechanism=Mechanism.CLOB,
            symbol="*",
            exchange_ts=now,
            local_recv_ts=now,
            kind=kind,
            detail=detail,
        )
