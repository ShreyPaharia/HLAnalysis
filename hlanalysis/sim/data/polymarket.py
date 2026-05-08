from __future__ import annotations

import json
from datetime import datetime, timezone

import requests
from loguru import logger

from .schemas import PMMarket, PMTrade

_GAMMA_BASE = "https://gamma-api.polymarket.com"


def _http_get(url: str, params: dict | None = None) -> dict | list:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _parse_iso(s: str) -> int:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


def _gamma_event_to_pmmarket(event: dict) -> PMMarket | None:
    markets = event.get("markets") or []
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
    start_iso = mk.get("startDate") or event.get("startDate")
    end_iso = mk.get("endDate") or event.get("endDate")
    if not (start_iso and end_iso):
        return None
    outcome = "unknown"
    op = mk.get("outcomePrices")
    if op:
        prices = json.loads(op) if isinstance(op, str) else op
        if len(prices) == 2:
            yes_p, no_p = float(prices[0]), float(prices[1])
            if yes_p >= 0.99:
                outcome = "yes"
            elif no_p >= 0.99:
                outcome = "no"
    return PMMarket(
        condition_id=str(cond_id),
        yes_token_id=str(token_ids[0]),
        no_token_id=str(token_ids[1]),
        start_ts_ns=_parse_iso(start_iso),
        end_ts_ns=_parse_iso(end_iso),
        resolved_outcome=outcome,  # type: ignore[arg-type]
        total_volume_usd=float(mk.get("volume") or 0.0),
        n_trades=0,  # populated later when trades are pulled
    )


def discover_btc_updown_markets(date_iso_list: list[str]) -> list[PMMarket]:
    """Discover BTC daily Up/Down markets for given ISO dates (YYYY-MM-DD).

    For each date, query Gamma's /events with the canonical slug, parse, return
    PMMarket rows. Caller filters for liquidity later.
    """
    results: list[PMMarket] = []
    for date_iso in date_iso_list:
        slug = _slug_for_date(date_iso)
        try:
            payload = _http_get(f"{_GAMMA_BASE}/events", params={"slug": slug})
        except Exception as e:
            logger.warning(f"PM gamma fetch failed for {slug}: {e}")
            continue
        events = payload if isinstance(payload, list) else [payload]
        for ev in events:
            m = _gamma_event_to_pmmarket(ev)
            if m is not None:
                results.append(m)
    return results


def _slug_for_date(date_iso: str) -> str:
    """Map 2026-05-09 → bitcoin-up-or-down-on-may-9-2026 (PM's slug convention)."""
    dt = datetime.fromisoformat(date_iso)
    month = dt.strftime("%B").lower()
    return f"bitcoin-up-or-down-on-{month}-{dt.day}-{dt.year}"


_CLOB_DATA_BASE = "https://data-api.polymarket.com"
_PAGE_SIZE = 500


def fetch_trades(condition_id: str) -> list[PMTrade]:
    """Page through CLOB /trades for a single market and return parsed trades."""
    out: list[PMTrade] = []
    offset = 0
    while True:
        page = _http_get(
            f"{_CLOB_DATA_BASE}/trades",
            params={"market": condition_id, "limit": _PAGE_SIZE, "offset": offset},
        )
        if not page:
            break
        for row in page:
            t = _parse_trade(row)
            if t is not None:
                out.append(t)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return out


def _parse_trade(row: dict) -> PMTrade | None:
    try:
        ts = float(row.get("timestamp", 0))
        return PMTrade(
            ts_ns=int(ts * 1e9),
            token_id=str(row["asset"]),
            side="buy" if str(row.get("side", "")).upper() == "BUY" else "sell",
            price=float(row["price"]),
            size=float(row["size"]),
        )
    except (KeyError, ValueError, TypeError):
        return None


_CLOB_BASE = "https://clob.polymarket.com"


def fetch_prices_history(token_id: str, interval: str = "1m") -> list[dict]:
    """Return [{ts_ns: int, price: float}, ...] sorted by ts ascending."""
    payload = _http_get(
        f"{_CLOB_BASE}/prices-history",
        params={"market": token_id, "interval": interval},
    )
    history = payload.get("history") if isinstance(payload, dict) else []
    rows = [{"ts_ns": int(float(r["t"]) * 1e9), "price": float(r["p"])} for r in history]
    rows.sort(key=lambda r: r["ts_ns"])
    return rows
