"""Pure PM CLOB WS payload → NormalizedEvent translators.

PM WS message format reference: docs.polymarket.com/developers/CLOB/websocket.
All `timestamp` fields are milliseconds since epoch; we convert to ns to
match the rest of the engine. Token IDs are 76-digit ERC-1155 ID strings
and are carried through verbatim as `symbol`.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from ..events import (
    BookSnapshotEvent,
    Mechanism,
    ProductType,
    QuestionMetaEvent,
    SettlementEvent,
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


_BTC_UPDOWN_STRIKE_RULE = re.compile(
    r"Binance 1 minute candle for BTC/USDT\s+(\w+)\s+(\d+)\s+'(\d{2})\s+"
    r"(\d{1,2}):(\d{2})\s+in the ET timezone",
    re.IGNORECASE,
)
_MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
           "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def _parse_strike_ref_ts_ns(description: str) -> int | None:
    if not description:
        return None
    m = _BTC_UPDOWN_STRIKE_RULE.search(description)
    if not m:
        return None
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
    except Exception:
        return None
    mon, day, yr2, hh, mm = m.groups()
    mon_key = mon.capitalize()
    if mon_key not in _MONTHS:
        return None
    dt = datetime(2000 + int(yr2), _MONTHS[mon_key], int(day), int(hh), int(mm), tzinfo=et)
    return int(dt.astimezone(timezone.utc).timestamp() * 1e9)


def _question_idx_from_condition(condition_id: str) -> int:
    """Stable 31-bit hash so the index fits in a SQLite int column."""
    return hash(condition_id) & 0x7FFFFFFF


def _parse_token_ids(market: dict) -> tuple[str, str] | None:
    raw = market.get("clobTokenIds")
    if not raw:
        return None
    toks = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(toks, list) or len(toks) < 2:
        return None
    return str(toks[0]), str(toks[1])


def _parse_iso_ns(s: str) -> int:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


def parse_gamma_market_to_question_meta(
    market: dict, *, series_slug: str, local_recv_ts: int,
) -> QuestionMetaEvent:
    cond_id = str(market.get("conditionId") or market.get("id") or "")
    tokens = _parse_token_ids(market)
    if tokens is None:
        raise ValueError(f"market {cond_id}: clobTokenIds missing or malformed")
    yes_t, no_t = tokens
    end_iso = market.get("endDate") or ""
    desc = market.get("description") or ""
    strike_ref_ts_ns = _parse_strike_ref_ts_ns(desc)

    keys = ["class", "underlying", "yes_token_id", "no_token_id",
            "expiry_ns", "series_slug", "condition_id"]
    values = ["priceBinary", "BTC", yes_t, no_t,
              str(_parse_iso_ns(end_iso)) if end_iso else "0",
              series_slug, cond_id]
    if strike_ref_ts_ns is not None:
        keys.append("strike_ref_ts_ns")
        values.append(str(strike_ref_ts_ns))

    return QuestionMetaEvent(
        venue=_VENUE,
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol=yes_t,
        exchange_ts=0,
        local_recv_ts=local_recv_ts,
        question_idx=_question_idx_from_condition(cond_id),
        named_outcome_idxs=[0, 1],
        keys=keys,
        values=values,
    )


def parse_gamma_market_to_settlement(
    market: dict, *, series_slug: str, local_recv_ts: int,
) -> SettlementEvent | None:
    """Returns a SettlementEvent iff the market has resolved (one of YES/NO
    has outcome price ≥ 0.99). Open markets return None.
    """
    raw = market.get("outcomePrices")
    if not raw:
        return None
    prices = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(prices, list) or len(prices) != 2:
        return None
    yes_p, no_p = float(prices[0]), float(prices[1])
    if yes_p >= 0.99:
        settled_idx = 0
    elif no_p >= 0.99:
        settled_idx = 1
    else:
        return None
    tokens = _parse_token_ids(market)
    if tokens is None:
        return None
    return SettlementEvent(
        venue=_VENUE,
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol=tokens[settled_idx],
        exchange_ts=local_recv_ts,
        local_recv_ts=local_recv_ts,
        settled_side_idx=settled_idx,
        settle_price=1.0,
        settle_ts=local_recv_ts,
        keys=["series_slug", "condition_id"],
        values=[series_slug, str(market.get("conditionId") or "")],
    )
