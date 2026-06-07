"""Pure PM CLOB WS payload → NormalizedEvent translators.

PM WS message format reference: docs.polymarket.com/developers/CLOB/websocket.
All `timestamp` fields are milliseconds since epoch; we convert to ns to
match the rest of the engine. Token IDs are 76-digit ERC-1155 ID strings
and are carried through verbatim as `symbol`.
"""
from __future__ import annotations

import hashlib
import json
import re

from .._fastjson import decode as _json_decode
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


def _levels_best_first(
    raw: list[dict[str, Any]], *, is_bid: bool,
) -> tuple[list[float], list[float]]:
    """Return (prices, sizes) sorted best-first: bids descending (highest
    price = best bid first), asks ascending (lowest price = best ask first).

    PM CLOB `book`/`price_change` frames list levels WORST-first (bids
    ascending, asks descending). The consumer (MarketState.apply) reads index
    [0] as top-of-book and walks bid_levels/ask_levels for depth, so the raw
    order would pin top-of-book to the 0.01/0.99 extremes — mid 0.50, no
    favorite, no PM trade. Sorting here makes [0] the best level for every
    downstream reader.
    """
    pairs = [(float(lvl["price"]), float(lvl["size"])) for lvl in raw]
    pairs.sort(key=lambda ps: ps[0], reverse=is_bid)
    return [p for p, _ in pairs], [s for _, s in pairs]


def parse_book_message(payload: dict[str, Any], *, local_recv_ts: int) -> BookSnapshotEvent:
    bid_px, bid_sz = _levels_best_first(payload.get("bids") or [], is_bid=True)
    ask_px, ask_sz = _levels_best_first(payload.get("asks") or [], is_bid=False)
    return BookSnapshotEvent(
        venue=_VENUE,
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol=str(payload["asset_id"]),
        exchange_ts=_ts_ms_to_ns(payload.get("timestamp", 0)),
        local_recv_ts=local_recv_ts,
        bid_px=bid_px,
        bid_sz=bid_sz,
        ask_px=ask_px,
        ask_sz=ask_sz,
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

    NOTE: this function is a stateless helper retained for direct callers and
    tests. The live adapter uses :class:`PmBook` to emit MERGED full snapshots
    instead (SHR-62) — do not call this from the adapter's _handle method.
    """
    changes = payload.get("changes") or []
    bids = [c for c in changes if str(c.get("side", "")).upper() == "BUY"]
    asks = [c for c in changes if str(c.get("side", "")).upper() == "SELL"]
    if not bids and not asks:
        return None
    bid_px, bid_sz = _levels_best_first(bids, is_bid=True)
    ask_px, ask_sz = _levels_best_first(asks, is_bid=False)
    return BookSnapshotEvent(
        venue=_VENUE,
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol=str(payload["asset_id"]),
        exchange_ts=_ts_ms_to_ns(payload.get("timestamp", 0)),
        local_recv_ts=local_recv_ts,
        bid_px=bid_px,
        bid_sz=bid_sz,
        ask_px=ask_px,
        ask_sz=ask_sz,
    )


class PmBook:
    """Per-asset L2 book the PM adapter maintains so price_change deltas merge
    into a FULL book before emission (SHR-62). MarketState.apply treats
    BookSnapshotEvent as a full replace, so emitting only-changed levels as a
    snapshot corrupts the book to 1-2 phantom levels between full frames.

    Usage::

        book = PmBook()
        # On "book" message (full snapshot):
        ev = book.apply_book(payload, local_recv_ts=recv_ns)
        # On "price_change" message (delta):
        ev = book.apply_price_change(payload, local_recv_ts=recv_ns)
        if ev is not None:
            ...  # emit the merged full snapshot
    """

    def __init__(self) -> None:
        self._bids: dict[float, float] = {}  # price → size
        self._asks: dict[float, float] = {}

    def apply_book(
        self, payload: dict[str, Any], *, local_recv_ts: int = 0
    ) -> BookSnapshotEvent:
        """Full replace: reset internal state from a complete book snapshot."""
        self._bids = {
            float(lvl["price"]): float(lvl["size"])
            for lvl in (payload.get("bids") or [])
        }
        self._asks = {
            float(lvl["price"]): float(lvl["size"])
            for lvl in (payload.get("asks") or [])
        }
        return self._emit(payload, local_recv_ts)

    def apply_price_change(
        self, payload: dict[str, Any], *, local_recv_ts: int = 0
    ) -> BookSnapshotEvent | None:
        """Apply a delta and return the merged full-book snapshot.

        Returns None when ``changes`` is empty so the caller can skip the emit
        (same contract as the stateless ``parse_price_change_message``).
        """
        changes = payload.get("changes") or []
        if not changes:
            return None
        for c in changes:
            side = str(c.get("side", "")).upper()
            px = float(c["price"])
            sz = float(c["size"])
            book = self._bids if side == "BUY" else self._asks
            if sz <= 0:
                book.pop(px, None)
            else:
                book[px] = sz
        return self._emit(payload, local_recv_ts)

    def reset(self) -> None:
        """Clear all book state.  Call on reconnect/resubscribe so a stale
        local book cannot survive a feed gap."""
        self._bids.clear()
        self._asks.clear()

    def _emit(self, payload: dict[str, Any], local_recv_ts: int) -> BookSnapshotEvent:
        bid_px = sorted(self._bids, reverse=True)  # best (highest) first
        ask_px = sorted(self._asks)                # best (lowest) first
        return BookSnapshotEvent(
            venue=_VENUE,
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol=str(payload["asset_id"]),
            exchange_ts=_ts_ms_to_ns(payload.get("timestamp", 0)),
            local_recv_ts=local_recv_ts,
            bid_px=bid_px,
            bid_sz=[self._bids[p] for p in bid_px],
            ask_px=ask_px,
            ask_sz=[self._asks[p] for p in ask_px],
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
    """Deterministic 31-bit id so the index fits in a SQLite int column.

    Must be stable for a given condition_id across process restarts — it's the
    primary key for PM rows (pm_strike, position, seen_question). Python's
    built-in hash() is salted per-process (PYTHONHASHSEED), which would re-key
    the same market on every engine restart, so use a fixed digest instead.
    """
    digest = hashlib.sha256(condition_id.encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _parse_token_ids(market: dict) -> tuple[str, str] | None:
    raw = market.get("clobTokenIds")
    if not raw:
        return None
    toks = _json_decode(raw) if isinstance(raw, str) else raw
    if not isinstance(toks, list) or len(toks) < 2:
        return None
    return str(toks[0]), str(toks[1])


def _parse_iso_ns(s: str) -> int:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


def _iso_to_hl_expiry(s: str) -> str:
    """Render ISO `endDate` in HL's YYYYMMDD-HHMM form so MarketState picks
    up PM expiries via the same kv["expiry"] path it already reads for HL.
    """
    if not s:
        return ""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y%m%d-%H%M")


def _strike_from_group_item_title(title: str) -> float | None:
    """PM bucket-style markets put their strike in `groupItemTitle` (e.g.
    "$80,000"). Returns the numeric strike or None if unparseable. Daily
    up/down binaries leave this blank and resolve against a per-day
    reference candle — no static strike is known here."""
    if not title:
        return None
    cleaned = title.strip().lstrip("$").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_gamma_market_to_question_meta(
    market: dict, *, series_slug: str, local_recv_ts: int,
    underlying: str = "BTC",
) -> QuestionMetaEvent:
    cond_id = str(market.get("conditionId") or market.get("id") or "")
    tokens = _parse_token_ids(market)
    if tokens is None:
        raise ValueError(f"market {cond_id}: clobTokenIds missing or malformed")
    yes_t, no_t = tokens
    end_iso = market.get("endDate") or ""
    desc = market.get("description") or ""
    # Prefer the explicit `question` field, then the bucket-leg title, then
    # the first sentence of the long-form description so daily up/down
    # markets still get a useful label on Telegram.
    question_text = (
        market.get("question")
        or market.get("groupItemTitle")
        or (desc.split(".", 1)[0].strip() if desc else "")
    )
    strike_ref_ts_ns = _parse_strike_ref_ts_ns(desc)
    strike = _strike_from_group_item_title(market.get("groupItemTitle") or "")
    expiry_ns_str = str(_parse_iso_ns(end_iso)) if end_iso else "0"
    expiry_hl = _iso_to_hl_expiry(end_iso)

    # `expiry` mirrors HL's YYYYMMDD-HHMM format so MarketState's existing
    # parser picks PM expiries up without branching on venue. `expiry_ns` is
    # retained for downstream consumers that want epoch ns directly.
    # `question_name` is the human-readable label rendered as a fallback by
    # the alert formatter when no numeric strike is available (daily PM
    # up/down markets).
    keys = ["class", "underlying", "yes_token_id", "no_token_id",
            "expiry", "expiry_ns", "series_slug", "condition_id",
            "question_name"]
    values = ["priceBinary", underlying, yes_t, no_t,
              expiry_hl, expiry_ns_str,
              series_slug, cond_id, str(question_text)]
    if strike is not None:
        keys.append("targetPrice")
        values.append(f"{strike:.0f}")
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


def parse_bucket_event(ev: dict) -> dict | None:
    """Group a Gamma multi-strike event's sub-markets into a bucket manifest.

    Each sub-market is an "above $X" binary with its own conditionId + (YES,NO)
    clob token pair. Sub-markets are sorted by strike ascending so leg_tokens +
    thresholds align with the canonical (yes_o0,no_o0,yes_o1,no_o1,...) layout
    the strategy's above_ladder region map expects. Returns None for events with
    fewer than 2 parseable legs. Mirrors the backtest's bucket manifest shape.
    """
    markets = ev.get("markets") or []
    if len(markets) < 2:
        return None
    legs: list[tuple[float, str, str, str]] = []  # (strike, cond, yes, no)
    for mk in markets:
        raw = mk.get("clobTokenIds")
        if not raw:
            continue
        tok = _json_decode(raw) if isinstance(raw, str) else raw
        if not isinstance(tok, list) or len(tok) != 2:
            continue
        try:
            strike = float(str(mk.get("groupItemTitle") or "").lstrip("$").replace(",", ""))
        except ValueError:
            continue
        cond = str(mk.get("conditionId") or mk.get("id") or "")
        legs.append((strike, cond, str(tok[0]), str(tok[1])))
    if len(legs) < 2:
        return None
    legs.sort(key=lambda x: x[0])
    return {
        "thresholds": [l[0] for l in legs],
        "leg_condition_ids": [l[1] for l in legs],
        "leg_tokens": [[l[2], l[3]] for l in legs],
    }


def parse_gamma_event_to_bucket_question_meta(
    ev: dict, *, series_slug: str, local_recv_ts: int, underlying: str = "BTC",
) -> QuestionMetaEvent | None:
    """One priceBucket QuestionMetaEvent for a multi-strike event, or None.

    kv carries the full leg layout so MarketState._update_question can rebuild
    the flat leg_symbols tuple and the strategy's winning_region(above_ladder)
    can map each leg. question_idx is keyed on the event slug so it is stable
    across restarts (per-leg conditionIds route settlement by leg-symbol
    membership, so the question_idx only needs to be stable+unique)."""
    rec = parse_bucket_event(ev)
    if rec is None:
        return None
    slug = str(ev.get("slug") or ev.get("ticker") or "")
    thresholds = rec["thresholds"]
    flat_legs = [t for pair in rec["leg_tokens"] for t in pair]
    end_iso = ev.get("endDate") or ""
    keys = ["class", "underlying", "leg_token_ids", "priceThresholds",
            "bucketLayout", "expiry", "expiry_ns", "series_slug",
            "condition_id", "question_name"]
    values = [
        "priceBucket", underlying, ",".join(flat_legs),
        ",".join(f"{t:.0f}" for t in thresholds), "above_ladder",
        _iso_to_hl_expiry(end_iso),
        str(_parse_iso_ns(end_iso)) if end_iso else "0",
        series_slug, slug, slug,
    ]
    return QuestionMetaEvent(
        venue=_VENUE,
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol=flat_legs[0],
        exchange_ts=0,
        local_recv_ts=local_recv_ts,
        question_idx=_question_idx_from_condition(slug),
        named_outcome_idxs=list(range(len(thresholds))),
        keys=keys,
        values=values,
    )


def parse_gamma_market_to_settlement(
    market: dict, *, series_slug: str, local_recv_ts: int,
) -> SettlementEvent | None:
    """Returns a SettlementEvent iff the market has actually resolved: the poll
    time is at/after the market's ``endDate`` AND one of YES/NO has an outcome
    price ≥ 0.99. Open markets return None.

    The endDate gate is load-bearing: these BTC up/down dailies resolve on the
    endDate reference candle (often hours after the favourite is already priced
    at 0.99). Treating a ≥0.99 price BEFORE endDate as "settled" fired a phantom
    settlement that de-tracked a still-open live position and booked bogus PnL
    (2026-06-02). A 0.99 price with hours to go is a strong favourite, not a
    resolution. If endDate is missing/unparseable we fall back to price-only so
    we never silently stop settling.
    """
    raw = market.get("outcomePrices")
    if not raw:
        return None
    end_iso = market.get("endDate") or ""
    if end_iso:
        try:
            if local_recv_ts < _parse_iso_ns(end_iso):
                return None  # not yet at resolution — 0.99 is a favourite, not a settle
        except ValueError:
            pass  # unparseable endDate → fall back to price-only (legacy)
    prices = _json_decode(raw) if isinstance(raw, str) else raw
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
