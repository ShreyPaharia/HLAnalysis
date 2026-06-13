"""Binance 1m kline puller (BTCUSDT). Used by the Polymarket source for reference HLC.

Moved verbatim from ``hlanalysis/sim/data/binance_klines.py``. The reference
stream is consumed via ``ReferenceEvent`` by the runner regardless of which
data source emits it.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

_BASE = "https://api.binance.com"
_LIMIT = 1000

# Bar width per Binance kline ``interval`` string, in ms. Used to advance the
# paging cursor one bar past the last row of each page. The legacy code
# hardcoded 60_000 (correct only for "1m"); "1s" needs 1_000 or it skips 59 of
# every 60 seconds. Extend this map to support more intervals as needed.
_INTERVAL_MS = {
    "1s": 1_000,
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def _interval_ms(interval: str) -> int:
    try:
        return _INTERVAL_MS[interval]
    except KeyError:
        raise ValueError(f"unsupported kline interval {interval!r}; known: {sorted(_INTERVAL_MS)}") from None


@dataclass(frozen=True, slots=True)
class Kline:
    ts_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def _http_get(url: str, params: dict) -> list:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_klines(
    start_ts_ms: int,
    end_ts_ms: int,
    symbol: str = "BTCUSDT",
    interval: str = "1m",
) -> list[Kline]:
    """Page through Binance klines and return parsed rows ascending by ts.

    ``interval`` accepts any key in ``_INTERVAL_MS`` (e.g. "1s", "1m"); the
    paging cursor advances one bar-width past each page's last open so 1s pulls
    page correctly (the legacy hardcoded 60_000 advance only suited 1m).
    """
    step_ms = _interval_ms(interval)
    out: list[Kline] = []
    cursor = start_ts_ms
    while cursor < end_ts_ms:
        page = _http_get(
            f"{_BASE}/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ts_ms,
                "limit": _LIMIT,
            },
        )
        if not page:
            break
        for row in page:
            out.append(
                Kline(
                    ts_ns=int(row[0]) * 1_000_000,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        last_open_ms = int(page[-1][0])
        cursor = last_open_ms + step_ms
        if len(page) < _LIMIT:
            break
    return out


_PERP_BASE = "https://fapi.binance.com"


def fetch_perp_klines(
    start_ts_ms: int,
    end_ts_ms: int,
    symbol: str = "BTCUSDT",
    interval: str = "1m",
) -> list[Kline]:
    """Page through Binance USDM perp klines and return parsed rows ascending by ts."""
    out: list[Kline] = []
    cursor = start_ts_ms
    while cursor < end_ts_ms:
        page = _http_get(
            f"{_PERP_BASE}/fapi/v1/klines",
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ts_ms,
                "limit": _LIMIT,
            },
        )
        if not page:
            break
        for row in page:
            out.append(
                Kline(
                    ts_ns=int(row[0]) * 1_000_000,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        cursor = int(page[-1][0]) + 60_000  # advance past last bar's open_time
        if len(page) < _LIMIT:
            break
    return out


__all__ = ["Kline", "fetch_klines", "fetch_perp_klines"]
