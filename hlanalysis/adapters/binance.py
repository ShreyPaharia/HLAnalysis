from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

import requests

from ..config import Subscription
from ..events import (
    BboEvent,
    BookSnapshotEvent,
    FundingEvent,
    LiquidationEvent,
    MarkEvent,
    Mechanism,
    NormalizedEvent,
    ProductType,
    TradeEvent,
)
from ._ws_base import BaseWsAdapter

log = logging.getLogger(__name__)


def _label_product_type(label: str) -> ProductType:
    """Health events for the spot connection must not be mislabelled PERP."""
    return ProductType.SPOT if label == "spot" else ProductType.PERP


SPOT_WS = "wss://stream.binance.com:9443/ws"
PERP_WS = "wss://fstream.binance.com/ws"
PERP_REST_PREMIUM_INDEX = "https://fapi.binance.com/fapi/v1/premiumIndex"
PERP_MARK_POLL_INTERVAL_S = 3.0

# Exponential-backoff bounds for the REST poller generic error path (#33).
# On a persistent failure we ramp from PERP_MARK_POLL_INTERVAL_S up to this cap
# so a network outage doesn't hammer the endpoint at full rate.
_POLL_BACKOFF_INITIAL_S: float = PERP_MARK_POLL_INTERVAL_S
_POLL_BACKOFF_MAX_S: float = 60.0

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


class BinanceAdapter(BaseWsAdapter):
    venue = "binance"
    # Per-frame staleness watchdog (SHR-60) and reconnect/backoff live in
    # BaseWsAdapter. Binance BBO/trade frames arrive multiple times per second,
    # so the base's 30s frameless window is unambiguously a dead/half-open
    # stream — not a quiet market.

    # SHR-61 (#22): per-symbol last-seen seq, for monotonic-seq enforcement.
    # Key = symbol (e.g. "BTCUSDT"), value = last accepted seq number.
    # Reset on adapter construction; NOT reset on reconnect (intentional: a lower
    # seq after reconnect is exactly the stale-overwrite scenario we guard against).
    # BboEvent/BookSnapshotEvent frames with seq <= _last_seq[sym] are dropped.
    # Initialized lazily on first frame per symbol.

    # SHR-61: drop counter exposed for ops/tests.
    bbo_drops: int = 0

    def __init__(self) -> None:
        # Per-symbol last-seen sequence numbers.  Separate from decode_failures
        # (base class) — this is a per-instance mutable dict, not a class attr.
        self._last_seq: dict[str, int] = {}
        # bbo_drops is a class attr default; rebind as instance attr on first drop.
        self.bbo_drops = 0

    def supports(self, product_type: ProductType, mechanism: Mechanism) -> bool:
        return mechanism == Mechanism.CLOB and product_type in {
            ProductType.PERP,
            ProductType.SPOT,
        }

    async def stream(self, subscriptions: list[Subscription]) -> AsyncIterator[NormalizedEvent]:
        spot_subs = [s for s in subscriptions if s.product_type == ProductType.SPOT]
        perp_subs = [s for s in subscriptions if s.product_type == ProductType.PERP]

        queue: asyncio.Queue[NormalizedEvent] = asyncio.Queue(maxsize=10000)
        tasks: list[asyncio.Task] = []
        if spot_subs:
            tasks.append(asyncio.create_task(self._run_one(SPOT_WS, spot_subs, _STREAMS_SPOT, queue, "spot")))
        if perp_subs:
            tasks.append(asyncio.create_task(self._run_one(PERP_WS, perp_subs, _STREAMS_PERP, queue, "perp")))
            # Mark/funding via REST polling because the WS streams are IP-restricted.
            rest_subs = [s for s in perp_subs if any(c in _PERP_REST_CHANNELS for c in s.channels)]
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

        product_type = _label_product_type(label)

        async def _subscribe(ws) -> None:
            await ws.send(json.dumps({"method": "SUBSCRIBE", "params": streams, "id": 1}))

        def _handle(msg: dict, recv_ns: int) -> list[NormalizedEvent]:
            return self._handle(msg, recv_ns, sym_to_sub, label, product_type)

        # BaseWsAdapter owns connect/backoff/circuit-breaker and the SHR-60
        # per-frame staleness watchdog (force-reconnect + feed_stale health on a
        # silent/half-open socket, plus the first-data deadline).
        await self._run_ws(
            url=ws_url,
            subscribe=_subscribe,
            handle=_handle,
            queue=queue,
            label=label,
            product_type=product_type,
        )

    # ------------------------------------------------------------------
    # SHR-61: price-validation helpers
    # ------------------------------------------------------------------

    def _check_seq(self, symbol: str, seq: int) -> bool:
        """Return True iff `seq` is strictly greater than the last accepted seq.

        On first frame for a symbol (no prior state) always returns True.
        Updates internal state on acceptance; callers must not update state
        on rejection.
        """
        last = self._last_seq.get(symbol)
        if last is not None and seq <= last:
            return False
        self._last_seq[symbol] = seq
        return True

    def _drop_bbo(self, reason: str, symbol: str, bid: float, ask: float, seq: int) -> list:
        """Increment drop counter, emit a warning log, and return []."""
        self.bbo_drops += 1
        log.warning(
            "binance: drop invalid BBO symbol=%s bid=%.6g ask=%.6g seq=%d reason=%s",
            symbol,
            bid,
            ask,
            seq,
            reason,
        )
        return []

    def _drop_depth(self, reason: str, symbol: str, seq: int) -> list:
        """Increment drop counter, emit a warning log, and return []."""
        self.bbo_drops += 1
        log.warning(
            "binance: drop invalid depth symbol=%s seq=%d reason=%s",
            symbol,
            seq,
            reason,
        )
        return []

    def _handle(
        self,
        msg: dict,
        recv_ns: int,
        sym_to_sub: dict[str, Subscription],
        label: str,
        product_type: ProductType = ProductType.PERP,
    ) -> list[NormalizedEvent]:
        # SUBSCRIBE response: {"result":null,"id":1}
        if "result" in msg and "id" in msg:
            return [self._health(f"subscribed/{label}", json.dumps(msg), product_type)]

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
            sym = msg["s"].upper()
            seq = int(msg.get("u", 0))
            bid_px = float(msg["b"])
            ask_px = float(msg["a"])
            # SHR-61: validate prices and monotonic seq before emitting.
            if bid_px <= 0.0 or ask_px <= 0.0:
                return self._drop_bbo("non-positive-price", sym, bid_px, ask_px, seq)
            if bid_px >= ask_px:
                return self._drop_bbo("crossed", sym, bid_px, ask_px, seq)
            if not self._check_seq(sym, seq):
                return self._drop_bbo("seq-regression", sym, bid_px, ask_px, seq)
            out.append(
                BboEvent(
                    venue=self.venue,
                    product_type=sub.product_type,
                    mechanism=sub.mechanism,
                    symbol=sym,
                    exchange_ts=0,  # not provided by Binance spot @bookTicker
                    local_recv_ts=recv_ns,
                    seq=seq,
                    bid_px=bid_px,
                    bid_sz=float(msg["B"]),
                    ask_px=ask_px,
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
            sym = sub.symbol.upper()
            seq = int(msg["lastUpdateId"])
            # SHR-61: validate monotonic seq.
            if not self._check_seq(sym + ":depth", seq):
                return self._drop_depth("seq-regression", sym, seq)
            # SHR-61: validate prices — any non-positive price in the book is
            # suspicious (transient zero-level or stale snapshot); drop the whole frame.
            bid_prices = [float(b[0]) for b in msg["bids"]]
            ask_prices = [float(a[0]) for a in msg["asks"]]
            if any(p <= 0.0 for p in bid_prices) or any(p <= 0.0 for p in ask_prices):
                return self._drop_depth("non-positive-price", sym, seq)
            out.append(
                BookSnapshotEvent(
                    venue=self.venue,
                    product_type=sub.product_type,
                    mechanism=sub.mechanism,
                    symbol=sym,
                    exchange_ts=0,  # not provided by Binance spot partial-depth stream
                    local_recv_ts=recv_ns,
                    seq=seq,
                    bid_px=bid_prices,
                    bid_sz=[float(b[1]) for b in msg["bids"]],
                    ask_px=ask_prices,
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
            seq = int(msg.get("u", 0))
            bid_px = float(msg["b"])
            ask_px = float(msg["a"])
            # SHR-61: validate prices and monotonic seq.
            if bid_px <= 0.0 or ask_px <= 0.0:
                return self._drop_bbo("non-positive-price", symbol, bid_px, ask_px, seq)
            if bid_px >= ask_px:
                return self._drop_bbo("crossed", symbol, bid_px, ask_px, seq)
            if not self._check_seq(symbol, seq):
                return self._drop_bbo("seq-regression", symbol, bid_px, ask_px, seq)
            out.append(
                BboEvent(
                    **common,
                    exchange_ts=int(msg.get("T", msg.get("E", 0))) * 1_000_000,
                    seq=seq,
                    bid_px=bid_px,
                    bid_sz=float(msg["B"]),
                    ask_px=ask_px,
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
        """Poll /fapi/v1/premiumIndex for mark + funding; emit MarkEvent and FundingEvent.

        #33: The generic `except` path now applies exponential backoff (capped at
        _POLL_BACKOFF_MAX_S) so a persistent network failure does not hammer the
        endpoint at the normal poll rate. Backoff resets to base on a successful
        response.
        """
        backoff = _POLL_BACKOFF_INITIAL_S
        while True:
            error_this_round = False
            for sub in subs:
                try:
                    r = await asyncio.to_thread(
                        requests.get,
                        PERP_REST_PREMIUM_INDEX,
                        params={"symbol": sub.symbol.upper()},
                        timeout=(5, 10),
                    )
                    if r.status_code != 200:
                        log.warning(
                            "binance premiumIndex non-200: status=%d symbol=%s; backing off",
                            r.status_code,
                            sub.symbol.upper(),
                        )
                        error_this_round = True
                        await asyncio.sleep(min(backoff * 2, _POLL_BACKOFF_MAX_S))
                        continue
                    p = r.json()
                except Exception as exc:
                    log.warning("binance premiumIndex poll failed: %s; backing off %.1fs", exc, backoff)
                    error_this_round = True
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, _POLL_BACKOFF_MAX_S)
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
                    next_funding = int(p["nextFundingTime"]) * 1_000_000 if p.get("nextFundingTime") else None
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
            if not error_this_round:
                # Successful round: reset backoff and sleep the normal poll interval.
                backoff = _POLL_BACKOFF_INITIAL_S
                await asyncio.sleep(PERP_MARK_POLL_INTERVAL_S)
            # On error rounds we already slept inside the except block; skip the
            # normal poll-interval sleep so we don't add a second, shorter delay
            # that would make the overall backoff sequence non-monotone.
