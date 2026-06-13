"""Binance 1-minute kline close lookup — the canonical source of a Polymarket
up/down market's open-strike. PM "BTC Up or Down" markets resolve against the
Binance *spot* BTC/USDT 1m candle close at a reference time; this fetches that
exact value.

Spot (api.binance.com) ONLY. This must be the same instrument PM settles on —
the verbatim rule on each market is "the Close price for the Binance 1 minute
candle for BTC/USDT … according to Binance BTC/USDT", and the displayed strike
matches the spot close exactly (e.g. 2026-05-31 16:00 UTC → 73644.92 spot vs
73609.30 perp). The perp/spot basis is tens of dollars — a few bps of price —
which is enough to flip the favourite near a coin-flip strike, so we never use
the perp (fapi) value. If spot can't be fetched we return None and the slot
skips the market rather than trade on a basis-biased strike.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import requests

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
    """Return the Binance *spot* BTCUSDT 1m candle close at the minute
    containing ``ts_ns``, or None if it can't be fetched. Spot only — never
    falls back to the perp (see module docstring)."""
    minute_start_ms = (ts_ns // 1_000_000_000) // 60 * 60 * 1000
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "startTime": minute_start_ms,
        "limit": 1,
    }
    try:
        data = http_get(_SPOT_URL, params)
    except Exception:
        return None
    return _close_from_klines(data, minute_start_ms=minute_start_ms)
