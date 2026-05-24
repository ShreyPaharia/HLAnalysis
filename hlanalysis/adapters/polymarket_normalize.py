"""Pure PM CLOB WS payload → NormalizedEvent translators.

PM WS message format reference: docs.polymarket.com/developers/CLOB/websocket.
All `timestamp` fields are milliseconds since epoch; we convert to ns to
match the rest of the engine. Token IDs are 76-digit ERC-1155 ID strings
and are carried through verbatim as `symbol`.
"""
from __future__ import annotations

from typing import Any

from ..events import (
    BookSnapshotEvent,
    Mechanism,
    ProductType,
    TradeEvent,
)

_VENUE = "polymarket"


def _ts_ms_to_ns(ms_str: str | int) -> int:
    return int(ms_str) * 1_000_000


def parse_book_message(payload: dict[str, Any], *, local_recv_ts: int) -> BookSnapshotEvent:
    bids = payload.get("bids") or []
    asks = payload.get("asks") or []
    return BookSnapshotEvent(
        venue=_VENUE,
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol=str(payload["asset_id"]),
        exchange_ts=_ts_ms_to_ns(payload.get("timestamp", 0)),
        local_recv_ts=local_recv_ts,
        bid_px=[float(b["price"]) for b in bids],
        bid_sz=[float(b["size"]) for b in bids],
        ask_px=[float(a["price"]) for a in asks],
        ask_sz=[float(a["size"]) for a in asks],
    )


def parse_trade_message(payload: dict[str, Any], *, local_recv_ts: int) -> TradeEvent:
    side_raw = str(payload.get("side", "")).upper()
    side = "buy" if side_raw == "BUY" else ("sell" if side_raw == "SELL" else "unknown")
    return TradeEvent(
        venue=_VENUE,
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol=str(payload["asset_id"]),
        exchange_ts=_ts_ms_to_ns(payload.get("timestamp", 0)),
        local_recv_ts=local_recv_ts,
        price=float(payload["price"]),
        size=float(payload["size"]),
        side=side,
        trade_id=str(payload.get("trade_id", "")) or None,
    )


def parse_price_change_message(
    payload: dict[str, Any], *, local_recv_ts: int,
) -> BookSnapshotEvent | None:
    """PM `price_change` events carry a partial book delta (changed levels
    only). We snapshot what's present; missing sides yield empty lists.
    Returns None when no changes are present to avoid clobbering a previous
    full snapshot.
    """
    changes = payload.get("changes") or []
    bids = [c for c in changes if str(c.get("side", "")).upper() == "BUY"]
    asks = [c for c in changes if str(c.get("side", "")).upper() == "SELL"]
    if not bids and not asks:
        return None
    return BookSnapshotEvent(
        venue=_VENUE,
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol=str(payload["asset_id"]),
        exchange_ts=_ts_ms_to_ns(payload.get("timestamp", 0)),
        local_recv_ts=local_recv_ts,
        bid_px=[float(b["price"]) for b in bids],
        bid_sz=[float(b["size"]) for b in bids],
        ask_px=[float(a["price"]) for a in asks],
        ask_sz=[float(a["size"]) for a in asks],
    )
