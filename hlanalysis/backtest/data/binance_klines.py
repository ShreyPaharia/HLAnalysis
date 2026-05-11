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
    """Page through Binance klines and return parsed rows ascending by ts."""
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
        cursor = last_open_ms + 60_000
        if len(page) < _LIMIT:
            break
    return out


__all__ = ["Kline", "fetch_klines"]
