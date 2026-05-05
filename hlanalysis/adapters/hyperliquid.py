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
    MarketMetaEvent,
    MarkEvent,
    Mechanism,
    NormalizedEvent,
    OracleEvent,
    ProductType,
    TradeEvent,
)
from .base import VenueAdapter

log = logging.getLogger(__name__)

# Map standard channels -> set of HL native subscription types.
# `mark`, `funding`, `oracle` all share one HL subscription; we dedup before sending.
_STD_TO_HL: dict[str, str] = {
    "trades": "trades",
    "book": "l2Book",
    "bbo": "bbo",
    "mark": "activeAssetCtx",
    "funding": "activeAssetCtx",
    "oracle": "activeAssetCtx",
}
HL_INFO_URL = "https://api.hyperliquid.xyz/info"
OUTCOME_REFRESH_INTERVAL_S = 60.0


class HyperliquidAdapter(VenueAdapter):
    venue = "hyperliquid"
    WS_URL = "wss://api.hyperliquid.xyz/ws"

    def supports(self, product_type: ProductType, mechanism: Mechanism) -> bool:
        if mechanism != Mechanism.CLOB:
            return False
        return product_type in {
            ProductType.PERP,
            ProductType.SPOT,
            ProductType.PREDICTION_BINARY,
        }

    async def stream(
        self, subscriptions: list[Subscription]
    ) -> AsyncIterator[NormalizedEvent]:
        # Separate wildcard prediction_binary subs (symbol="*") — these auto-discover
        # via outcomeMeta and dynamically subscribe to active HIP-4 markets.
        wildcard_templates: list[Subscription] = []
        sym_to_sub: dict[str, Subscription] = {}
        for s in subscriptions:
            if s.product_type == ProductType.PREDICTION_BINARY and s.symbol == "*":
                wildcard_templates.append(s)
            else:
                sym_to_sub[s.symbol] = s

        # Initial wildcard expansion before WS connect.
        for tmpl in wildcard_templates:
            for expanded in self._expand_wildcard(tmpl):
                sym_to_sub.setdefault(expanded.symbol, expanded)

        meta_seen: set[str] = set()
        # Emit initial market_meta for any prediction_binary subs in scope.
        if any(s.product_type == ProductType.PREDICTION_BINARY for s in sym_to_sub.values()):
            for ev in self._fetch_outcome_meta_events(
                [s for s in sym_to_sub.values() if s.product_type == ProductType.PREDICTION_BINARY],
                meta_seen,
            ):
                yield ev

        last_meta_refresh = time.monotonic()
        backoff = 1.0

        while True:
            try:
                async with websockets.connect(
                    self.WS_URL, ping_interval=20, ping_timeout=20, max_size=2**24
                ) as ws:
                    backoff = 1.0
                    yield self._health("connected", "ws open")
                    ws_subscribed: set[tuple[str, str]] = set()
                    for sub in sym_to_sub.values():
                        await self._subscribe_for(ws, sub, ws_subscribed)

                    async for raw in ws:
                        recv_ns = time.time_ns()
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        for ev in self._handle(msg, recv_ns, sym_to_sub):
                            yield ev

                        if (
                            wildcard_templates
                            and time.monotonic() - last_meta_refresh > OUTCOME_REFRESH_INTERVAL_S
                        ):
                            last_meta_refresh = time.monotonic()
                            async for ev in self._refresh_wildcards(
                                ws, wildcard_templates, sym_to_sub, ws_subscribed, meta_seen
                            ):
                                yield ev
            except Exception as e:
                log.warning("hyperliquid ws error: %s; backoff=%.1fs", e, backoff)
                yield self._health("reconnect", str(e))
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _subscribe_for(
        self,
        ws: websockets.WebSocketClientProtocol,
        sub: Subscription,
        ws_subscribed: set[tuple[str, str]],
    ) -> None:
        native_types: set[str] = set()
        for ch in sub.channels:
            native = _STD_TO_HL.get(ch)
            if native is None:
                log.warning("hyperliquid: unsupported channel %r", ch)
                continue
            native_types.add(native)
        for native in native_types:
            key = (sub.symbol, native)
            if key in ws_subscribed:
                continue
            ws_subscribed.add(key)
            await ws.send(
                json.dumps(
                    {
                        "method": "subscribe",
                        "subscription": {"type": native, "coin": sub.symbol},
                    }
                )
            )

    async def _refresh_wildcards(
        self,
        ws: websockets.WebSocketClientProtocol,
        templates: list[Subscription],
        sym_to_sub: dict[str, Subscription],
        ws_subscribed: set[tuple[str, str]],
        meta_seen: set[str],
    ) -> AsyncIterator[NormalizedEvent]:
        # Compute the full current set of active HIP-4 outcomes.
        active_now: dict[str, Subscription] = {}
        for tmpl in templates:
            for expanded in self._expand_wildcard(tmpl):
                active_now[expanded.symbol] = expanded

        prev_outcome_syms = {
            s for s, sub in sym_to_sub.items() if sub.product_type == ProductType.PREDICTION_BINARY
        }
        new_syms = set(active_now.keys()) - prev_outcome_syms
        rolled_syms = prev_outcome_syms - set(active_now.keys())

        for sym in sorted(new_syms):
            sym_to_sub[sym] = active_now[sym]
            yield self._health("outcome-discovered", sym)
            await self._subscribe_for(ws, active_now[sym], ws_subscribed)

        for sym in sorted(rolled_syms):
            yield self._health("outcome-rolled", sym)
            # Don't drop from sym_to_sub: any in-flight messages tagged with the old coin
            # should still be tagged correctly. The outcome stops emitting after settlement.

        # Emit market_meta only for newly discovered coins; existing ones already have it.
        if new_syms:
            new_subs = [active_now[s] for s in new_syms]
            for ev in self._fetch_outcome_meta_events(new_subs, meta_seen):
                yield ev

    def _expand_wildcard(self, template: Subscription) -> list[Subscription]:
        try:
            r = requests.post(HL_INFO_URL, json={"type": "outcomeMeta"}, timeout=5)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning("outcomeMeta fetch failed: %s", e)
            return []
        out: list[Subscription] = []
        for o in data.get("outcomes", []):
            outcome_idx = o.get("outcome")
            for side_idx, _ in enumerate(o.get("sideSpecs", [])):
                coin = f"#{10 * outcome_idx + side_idx}"
                out.append(template.model_copy(update={"symbol": coin}))
        return out

    def _handle(
        self,
        msg: dict,
        recv_ns: int,
        sym_to_sub: dict[str, Subscription],
    ) -> list[NormalizedEvent]:
        channel = msg.get("channel")
        data = msg.get("data")
        out: list[NormalizedEvent] = []

        if channel == "trades" and isinstance(data, list):
            for t in data:
                sub = sym_to_sub.get(t.get("coin"))
                if sub is None:
                    continue
                out.append(
                    TradeEvent(
                        venue=self.venue,
                        product_type=sub.product_type,
                        mechanism=sub.mechanism,
                        symbol=t["coin"],
                        exchange_ts=int(t["time"]) * 1_000_000,
                        local_recv_ts=recv_ns,
                        price=float(t["px"]),
                        size=float(t["sz"]),
                        side="buy" if t.get("side") == "B" else "sell",
                        trade_id=str(t.get("tid", "")) or None,
                    )
                )

        elif channel == "l2Book" and isinstance(data, dict):
            sub = sym_to_sub.get(data.get("coin"))
            if sub is not None:
                levels = data.get("levels") or [[], []]
                bids, asks = levels[0], levels[1]
                out.append(
                    BookSnapshotEvent(
                        venue=self.venue,
                        product_type=sub.product_type,
                        mechanism=sub.mechanism,
                        symbol=data["coin"],
                        exchange_ts=int(data.get("time", 0)) * 1_000_000,
                        local_recv_ts=recv_ns,
                        bid_px=[float(b["px"]) for b in bids],
                        bid_sz=[float(b["sz"]) for b in bids],
                        ask_px=[float(a["px"]) for a in asks],
                        ask_sz=[float(a["sz"]) for a in asks],
                    )
                )

        elif channel == "bbo" and isinstance(data, dict):
            sub = sym_to_sub.get(data.get("coin"))
            bbo = data.get("bbo") if sub is not None else None
            if sub is not None and bbo and bbo[0] and bbo[1]:
                bid, ask = bbo[0], bbo[1]
                out.append(
                    BboEvent(
                        venue=self.venue,
                        product_type=sub.product_type,
                        mechanism=sub.mechanism,
                        symbol=data["coin"],
                        exchange_ts=int(data.get("time", 0)) * 1_000_000,
                        local_recv_ts=recv_ns,
                        bid_px=float(bid["px"]),
                        bid_sz=float(bid["sz"]),
                        ask_px=float(ask["px"]),
                        ask_sz=float(ask["sz"]),
                    )
                )

        elif channel in ("activeAssetCtx", "activeSpotAssetCtx") and isinstance(data, dict):
            coin = data.get("coin")
            ctx = data.get("ctx") or {}
            sub = sym_to_sub.get(coin)
            if sub is None or not ctx:
                return out
            common = dict(
                venue=self.venue,
                product_type=sub.product_type,
                mechanism=sub.mechanism,
                symbol=coin,
                exchange_ts=recv_ns,  # ctx has no per-message ts
                local_recv_ts=recv_ns,
            )
            if "funding" in ctx:
                out.append(
                    FundingEvent(
                        **common,
                        funding_rate=float(ctx["funding"]),
                        premium=float(ctx["premium"]) if ctx.get("premium") is not None else None,
                    )
                )
            if "markPx" in ctx:
                out.append(MarkEvent(**common, mark_px=float(ctx["markPx"])))
            if "oraclePx" in ctx:
                out.append(OracleEvent(**common, oracle_px=float(ctx["oraclePx"])))

        elif channel == "subscriptionResponse":
            out.append(self._health("subscribed", json.dumps(data) if data else ""))

        return out

    def _fetch_outcome_meta_events(
        self, outcome_subs: list[Subscription], meta_seen: set[str]
    ) -> list[NormalizedEvent]:
        """Fetch HIP-4 outcome metadata; emit one MarketMetaEvent per *newly seen* coin id.

        `meta_seen` is a caller-owned set tracking coins for which we've already emitted
        metadata, so periodic refreshes don't duplicate static metadata. A coin reappearing
        after a roll (with the same encoding but new strike/expiry) IS treated as new because
        the encoding changes per roll.

        Coin identifier convention: encoding = 10 * outcome + side_index, written as `#<encoding>`.
        """
        try:
            r = requests.post(HL_INFO_URL, json={"type": "outcomeMeta"}, timeout=5)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning("outcomeMeta fetch failed: %s", e)
            return []

        sub_by_coin = {s.symbol: s for s in outcome_subs}
        recv_ns = time.time_ns()
        out: list[NormalizedEvent] = []
        for o in data.get("outcomes", []):
            outcome_idx = o.get("outcome")
            for side_idx, side in enumerate(o.get("sideSpecs", [])):
                coin = f"#{10 * outcome_idx + side_idx}"
                sub = sub_by_coin.get(coin)
                if sub is None or coin in meta_seen:
                    continue
                meta_seen.add(coin)
                desc = o.get("description") or ""
                fields = dict(
                    p.split(":", 1)
                    for p in desc.split("|")
                    if ":" in p
                )
                keys = ["outcome_idx", "side_idx", "side_name", "outcome_name"]
                values = [
                    str(outcome_idx),
                    str(side_idx),
                    str(side.get("name", "")),
                    str(o.get("name", "")),
                ]
                for k, v in fields.items():
                    keys.append(k)
                    values.append(v)
                out.append(
                    MarketMetaEvent(
                        venue=self.venue,
                        product_type=sub.product_type,
                        mechanism=sub.mechanism,
                        symbol=coin,
                        exchange_ts=recv_ns,
                        local_recv_ts=recv_ns,
                        keys=keys,
                        values=values,
                    )
                )
        return out

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
