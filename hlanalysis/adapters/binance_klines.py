"""Binance 1-minute kline close lookup — used to backfill the open-strike of a
Polymarket up/down market whose open the engine did not observe live (e.g. the
market was listed before the engine started, or the engine restarted past the
open). PM "BTC Up or Down" markets resolve against the Binance 1m candle close
at a reference time; this fetches that exact value.

Perp (fapi) is tried first so the backfilled strike shares the perp series of
the engine's live bbo reference feed — the perp/spot basis then cancels in the
strategy's log(reference_price / strike). Spot (api) is the fallback (it is
PM's literal resolution source; the residual basis vs the perp live reference
is a few bps — immaterial for favourite-threshold entries).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import requests

_PERP_URL = "https://fapi.binance.com/fapi/v1/klines"
_SPOT_URL = "https://api.binance.com/api/v3/klines"


def _real_http_get(url: str, params: dict[str, Any]) -> Any:
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def _close_from_klines(data: Any, *, minute_start_ms: int) -> float | None:
    """Pull the close (index 4) of the candle whose openTime (index 0) is
    exactly ``minute_start_ms``. Returns None if the response is empty or the
    returned candle is for a different minute (a gap — never use the wrong
    candle as a strike)."""
    if not isinstance(data, list) or not data:
        return None
    k = data[0]
    try:
        if int(k[0]) != minute_start_ms:
            return None
        return float(k[4])
    except (IndexError, TypeError, ValueError):
        return None


def binance_1m_close_at(
    ts_ns: int,
    *,
    http_get: Callable[[str, dict[str, Any]], Any] = _real_http_get,
) -> float | None:
    """Return the Binance BTCUSDT 1m candle close at the minute containing
    ``ts_ns``, or None if it can't be fetched. Tries perp then spot."""
    minute_start_ms = (ts_ns // 1_000_000_000) // 60 * 60 * 1000
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "startTime": minute_start_ms,
        "limit": 1,
    }
    for url in (_PERP_URL, _SPOT_URL):
        try:
            data = http_get(url, params)
        except Exception:
            continue
        close = _close_from_klines(data, minute_start_ms=minute_start_ms)
        if close is not None:
            return close
    return None
