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


_BTC_UPDOWN_SERIES_SLUG = "btc-up-or-down-daily"
_SERIES_PAGE_LIMIT = 500


def _fetch_series_events(series_slug: str) -> list[dict]:
    """Paginate Gamma `/events?series_slug=...&closed=true`. Returns raw event dicts.

    Each call uses the documented `limit`/`offset` pair; loop terminates when a
    short page comes back. We only ask for `closed=true` since the backtester
    needs resolved markets — open events have no outcome to settle against.
    """
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
            logger.warning(f"PM gamma series fetch failed at offset={offset}: {e}")
            break
        if not isinstance(page, list) or not page:
            break
        out.extend(page)
        if len(page) < _SERIES_PAGE_LIMIT:
            break
        offset += _SERIES_PAGE_LIMIT
    return out


def _event_in_window(ev: dict, start_iso: str, end_iso: str) -> bool:
    """True iff event.endDate (resolution time) is in [start_iso, end_iso).

    `start_iso` and `end_iso` are date-only `YYYY-MM-DD`. Compare lexicographically
    against the prefix of an ISO timestamp — works because ISO-8601 sorts naturally.
    """
    end = ev.get("endDate") or ""
    if len(end) < 10:
        return False
    return start_iso <= end[:10] < end_iso


def discover_btc_updown_markets(start_iso: str, end_iso: str) -> list[PMMarket]:
    """Discover BTC daily Up/Down markets resolving in [start_iso, end_iso).

    Pulls the canonical PM `btc-up-or-down-daily` series (single source of truth
    for the daily binary) and filters client-side by `endDate`. Replaces the
    earlier per-date slug lookup, which silently missed all markets before
    2026-03-13 because PM changed the slug convention then.

    Caller (cmd_fetch) is responsible for spec §4.1 liquidity floors — this
    function reports counts but does not drop on volume/n_trades.

    Args:
        start_iso: inclusive lower bound, "YYYY-MM-DD".
        end_iso:   exclusive upper bound, "YYYY-MM-DD".
    """
    raw = _fetch_series_events(_BTC_UPDOWN_SERIES_SLUG)
    in_window = [ev for ev in raw if _event_in_window(ev, start_iso, end_iso)]
    parsed: list[PMMarket] = []
    dropped = 0
    for ev in in_window:
        m = _gamma_event_to_pmmarket(ev)
        if m is None:
            dropped += 1
            continue
        parsed.append(m)
    n_resolved = sum(1 for m in parsed if m.resolved_outcome != "unknown")
    logger.info(
        f"PM discovery: series_total={len(raw)} "
        f"in_window=[{start_iso},{end_iso})={len(in_window)} "
        f"parsed={len(parsed)} resolved={n_resolved} drop_parse={dropped}"
    )
    return parsed


_CLOB_DATA_BASE = "https://data-api.polymarket.com"
_PAGE_SIZE = 500


def fetch_trades(condition_id: str) -> list[PMTrade]:
    """Page through CLOB /trades for a single market and return parsed trades.

    PM data-api caps offset around ~3500 — beyond that it returns 400. Treat
    that as end-of-data and warn; for our use case (24h binary markets) we
    rarely exceed that, but very-active markets will lose the tail.
    """
    out: list[PMTrade] = []
    offset = 0
    while True:
        try:
            page = _http_get(
                f"{_CLOB_DATA_BASE}/trades",
                params={"market": condition_id, "limit": _PAGE_SIZE, "offset": offset},
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 400 and offset > 0:
                logger.warning(
                    f"PM trades pagination capped at offset={offset} for {condition_id} "
                    f"(returned 400). Returning partial result."
                )
                break
            raise
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


_FIDELITY_FOR_INTERVAL = {"1m": 10, "5m": 10, "15m": 10, "1h": 60, "4h": 240, "1d": 1440}


def fetch_prices_history(
    token_id: str,
    interval: str = "1m",
    fidelity: int | None = None,
) -> list[dict]:
    """Return [{ts_ns: int, price: float}, ...] sorted by ts ascending.

    PM CLOB requires `fidelity` for sub-hourly intervals (undocumented; empirically
    needed for 1m). Defaults are picked to satisfy the API minimum per interval;
    callers can override by passing `fidelity=N` explicitly.

    Currently used only as a sparse-trade fallback; not wired into `cmd_fetch`.
    """
    params = {"market": token_id, "interval": interval}
    fid = fidelity if fidelity is not None else _FIDELITY_FOR_INTERVAL.get(interval)
    if fid is not None:
        params["fidelity"] = fid
    payload = _http_get(f"{_CLOB_BASE}/prices-history", params=params)
    history = payload.get("history") if isinstance(payload, dict) else []
    rows = [{"ts_ns": int(float(r["t"]) * 1e9), "price": float(r["p"])} for r in history]
    rows.sort(key=lambda r: r["ts_ns"])
    return rows
