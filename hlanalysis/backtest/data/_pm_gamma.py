"""Polymarket Gamma API + CLOB data-api helpers.

Covers:
- HTTP helpers (_http_get, _parse_iso_ns, _event_in_window)
- Series/market event fetching (_fetch_series_events, _fetch_trades_raw)
- Event parsing (_parse_binary_event, _parse_bucket_event,
  _parse_strike_ref_ts_ns and companions)

Extracted verbatim from polymarket.py — no logic changes.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Literal

import requests
from loguru import logger

from hlanalysis.adapters.polymarket_normalize import parse_bucket_event as _shared_parse_bucket_event

_GAMMA_BASE = "https://gamma-api.polymarket.com"
_CLOB_DATA_BASE = "https://data-api.polymarket.com"
_CLOB_BASE = "https://clob.polymarket.com"
_SERIES_PAGE_LIMIT = 100  # Gamma /events caps responses at 100 even when limit > 100;
# requesting 500 silently truncates and we mistake the short response for "end of data".
_TRADES_PAGE_SIZE = 500


# ---- HTTP helpers ------------------------------------------------------------


def _http_get(url: str, params: dict | None = None) -> dict | list:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _parse_iso_ns(s: str) -> int:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1e9)


def _event_in_window(ev: dict, start_iso: str, end_iso: str) -> bool:
    end = ev.get("endDate") or ""
    if len(end) < 10:
        return False
    return start_iso <= end[:10] < end_iso


def _fetch_series_events(series_slug: str) -> list[dict]:
    """Paginate Gamma `/events?series_slug=...&closed=true`."""
    out: list[dict] = []
    offset = 0
    while True:
        try:
            page = _http_get(
                f"{_GAMMA_BASE}/events",
                params={
                    "series_slug": series_slug,
                    "closed": "true",
                    "limit": _SERIES_PAGE_LIMIT,
                    "offset": offset,
                },
            )
        except Exception as e:
            logger.warning(f"PM series fetch failed at offset={offset}: {e}")
            break
        if not isinstance(page, list) or not page:
            break
        out.extend(page)
        if len(page) < _SERIES_PAGE_LIMIT:
            break
        offset += len(page)
    return out


def _fetch_trades_raw(condition_id: str) -> list[dict]:
    """Page PM data-api `/trades?market=<conditionId>`."""
    out: list[dict] = []
    offset = 0
    while True:
        try:
            page = _http_get(
                f"{_CLOB_DATA_BASE}/trades",
                params={"market": condition_id, "limit": _TRADES_PAGE_SIZE, "offset": offset},
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 400 and offset > 0:
                logger.warning(
                    f"PM trades pagination capped at offset={offset} for {condition_id}; "
                    f"returning partial result."
                )
                break
            raise
        if not page:
            break
        out.extend(page)
        if len(page) < _TRADES_PAGE_SIZE:
            break
        offset += _TRADES_PAGE_SIZE
    return out


# ---- Strike ref-ts parsing ---------------------------------------------------

_BTC_UPDOWN_STRIKE_RULE = re.compile(
    r"Binance 1 minute candle for BTC/USDT\s+(\w+)\s+(\d+)\s+'(\d{2})\s+(\d{1,2}):(\d{2})\s+in the ET timezone",
    re.IGNORECASE,
)
_MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
           "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def _parse_strike_ref_ts_ns(description: str) -> int | None:
    """Pull the strike-reference timestamp out of a PM 'BTC Up or Down' market
    description. The text says, e.g., "...Binance 1 minute candle for BTC/USDT
    Nov 27 '25 12:00 in the ET timezone...". The strike is the CLOSE of that
    candle. Returns ns since epoch (UTC) or None if the description doesn't
    match the expected pattern.
    """
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
    if mon not in _MONTHS:
        return None
    year = 2000 + int(yr2)
    dt = datetime(year, _MONTHS[mon], int(day), int(hh), int(mm), tzinfo=et)
    return int(dt.astimezone(UTC).timestamp() * 1e9)


# ---- Event parsing -----------------------------------------------------------


def _parse_binary_event(ev: dict) -> dict | None:
    """Parse a Gamma binary event into a manifest-shaped record. Returns None
    if it doesn't look like a single-market binary."""
    markets = ev.get("markets") or []
    if not markets:
        return None
    mk = markets[0]
    token_ids_raw = mk.get("clobTokenIds")
    if not token_ids_raw:
        return None
    token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else token_ids_raw
    if len(token_ids) != 2:
        return None
    cond_id = mk.get("conditionId") or mk.get("id")
    if not cond_id:
        return None
    start_iso = mk.get("startDate") or ev.get("startDate")
    end_iso = mk.get("endDate") or ev.get("endDate")
    if not (start_iso and end_iso):
        return None
    outcome: Literal["yes", "no", "unknown"] = "unknown"
    op = mk.get("outcomePrices")
    if op:
        prices = json.loads(op) if isinstance(op, str) else op
        if len(prices) == 2:
            yes_p, no_p = float(prices[0]), float(prices[1])
            if yes_p >= 0.99:
                outcome = "yes"
            elif no_p >= 0.99:
                outcome = "no"
    description = mk.get("description") or ev.get("description") or ""
    strike_ref_ts_ns = _parse_strike_ref_ts_ns(description)
    out: dict = {
        "condition_id": str(cond_id),
        "yes_token_id": str(token_ids[0]),
        "no_token_id": str(token_ids[1]),
        "start_ts_ns": _parse_iso_ns(start_iso),
        "end_ts_ns": _parse_iso_ns(end_iso),
        "resolved_outcome": outcome,
        "total_volume_usd": float(mk.get("volume") or 0.0),
        "n_trades": 0,
    }
    if strike_ref_ts_ns is not None:
        out["strike_ref_ts_ns"] = strike_ref_ts_ns
    return out


def _parse_bucket_event(ev: dict) -> dict | None:
    """Parse a Gamma multi-strike event into a bucket manifest record.

    Delegates leg-collection + strike-sorting to the shared
    `parse_bucket_event` (polymarket_normalize) so the live adapter and
    backtest always produce identical leg orderings. The backtest-only fields
    (`event_slug`, `start_ts_ns`, `end_ts_ns`, `leg_resolutions`) are
    appended here; `thresholds`/`leg_tokens`/`leg_condition_ids` come verbatim
    from the shared helper to avoid drift.
    """
    base = _shared_parse_bucket_event(ev)
    if base is None:
        return None
    slug = ev.get("slug") or ev.get("ticker")
    start_iso, end_iso = ev.get("startDate"), ev.get("endDate")
    if not (slug and start_iso and end_iso):
        return None
    # leg_resolutions stays backtest-only (settlement replay); recompute in
    # the SAME strike-ascending order the shared parser used.
    pairs = []  # (strike, yes_p, no_p)
    for mk in ev.get("markets") or []:
        raw = mk.get("clobTokenIds")
        if not raw:
            continue
        tok = json.loads(raw) if isinstance(raw, str) else raw
        if len(tok) != 2:
            continue
        try:
            # Match the shared parser's strike cleaning (lstrip "$") so the
            # re-sorted leg_resolutions stay positionally aligned with the
            # shared helper's thresholds/leg_tokens even for "$80,000" titles.
            strike = float(str(mk.get("groupItemTitle") or "").lstrip("$").replace(",", ""))
        except ValueError:
            continue
        res = "unknown"
        op = mk.get("outcomePrices")
        if op:
            pr = json.loads(op) if isinstance(op, str) else op
            if len(pr) == 2:
                if float(pr[0]) >= 0.99:
                    res = "yes"
                elif float(pr[1]) >= 0.99:
                    res = "no"
        pairs.append((strike, res))
    pairs.sort(key=lambda x: x[0])
    return {
        "event_slug": str(slug),
        "start_ts_ns": _parse_iso_ns(start_iso),
        "end_ts_ns": _parse_iso_ns(end_iso),
        "thresholds": base["thresholds"],
        "leg_tokens": base["leg_tokens"],
        "leg_condition_ids": base["leg_condition_ids"],
        "leg_resolutions": [r for _, r in pairs],
    }
