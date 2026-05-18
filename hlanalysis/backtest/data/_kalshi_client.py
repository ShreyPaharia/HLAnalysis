"""Minimal HTTP client for Kalshi's public REST surface.

Covers the three endpoints the `KalshiDataSource` needs (events, event detail,
market trades). Pure functions over `requests`; no state. Pagination helpers
return iterators; retry on transient 5xx / network errors with capped
exponential backoff.
"""
from __future__ import annotations

import time
from typing import Iterator

import requests

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

_TIMEOUT_S = 30.0
_RETRY_STATUSES = {500, 502, 503, 504}
_MAX_RETRIES = 5
_BACKOFF_S = (1, 2, 4, 8, 16)


def _get(path: str, params: dict | None = None) -> dict:
    url = f"{KALSHI_BASE}{path}"
    last: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            r = requests.get(url, params=params or {}, timeout=_TIMEOUT_S)
            if r.status_code in _RETRY_STATUSES:
                last = requests.HTTPError(f"{r.status_code} on {url}")
                time.sleep(_BACKOFF_S[attempt])
                continue
            r.raise_for_status()
            return r.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            last = e
            time.sleep(_BACKOFF_S[attempt])
    raise last  # type: ignore[misc]


def fetch_events_page(
    *,
    series_ticker: str,
    status: str = "settled",
    limit: int = 200,
    cursor: str | None = None,
) -> tuple[list[dict], str]:
    """One page of events. Returns (events, next_cursor). Empty cursor => end."""
    params: dict = {"series_ticker": series_ticker, "status": status, "limit": limit}
    if cursor:
        params["cursor"] = cursor
    body = _get("/events", params=params)
    return list(body.get("events") or []), str(body.get("cursor") or "")


def iter_events(*, series_ticker: str, status: str = "settled") -> Iterator[dict]:
    cursor: str | None = None
    while True:
        page, cursor = fetch_events_page(
            series_ticker=series_ticker, status=status, cursor=cursor,
        )
        for ev in page:
            yield ev
        if not cursor:
            return


def fetch_event_detail(event_ticker: str) -> tuple[dict, list[dict]]:
    body = _get(f"/events/{event_ticker}")
    return dict(body.get("event") or {}), list(body.get("markets") or [])


def fetch_market_trades(
    market_ticker: str,
    *,
    min_ts: int | None = None,
    limit: int = 1000,
    cursor: str | None = None,
) -> tuple[list[dict], str]:
    params: dict = {"ticker": market_ticker, "limit": limit}
    if min_ts is not None:
        params["min_ts"] = int(min_ts)
    if cursor:
        params["cursor"] = cursor
    body = _get("/markets/trades", params=params)
    return list(body.get("trades") or []), str(body.get("cursor") or "")


def iter_market_trades(
    market_ticker: str, *, min_ts: int | None = None,
) -> Iterator[dict]:
    cursor: str | None = None
    while True:
        page, cursor = fetch_market_trades(
            market_ticker, min_ts=min_ts, cursor=cursor,
        )
        for t in page:
            yield t
        if not cursor:
            return
