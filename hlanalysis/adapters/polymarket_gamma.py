"""Synchronous Gamma REST client for Polymarket market discovery + resolution.

The PM CLOB WS does not push market-creation or market-resolution events;
those are observed by polling the Gamma `/events` REST endpoint. This
module is pure HTTP — no asyncio — so it can be called from a background
asyncio task via `loop.run_in_executor` without the adapter pulling in
httpx-async or aiohttp just for this path.

`http_get` is injected so tests don't hit the network.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import requests
from loguru import logger

_GAMMA_BASE = "https://gamma-api.polymarket.com"
_PAGE_LIMIT = 100  # Gamma caps responses at 100 even when limit > 100


def _real_http_get(url: str, params: dict[str, Any]) -> list[dict] | dict:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


class GammaClient:
    def __init__(
        self, *,
        http_get: Callable[[str, dict[str, Any]], Any] = _real_http_get,
        base_url: str = _GAMMA_BASE,
    ) -> None:
        self._get = http_get
        self._base = base_url

    def fetch_events(
        self, *, series_slug: str, closed: bool, max_pages: int = 50,
    ) -> list[dict]:
        out: list[dict] = []
        offset = 0
        for _ in range(max_pages):
            page = self._get(f"{self._base}/events", {
                "series_slug": series_slug,
                "closed": "true" if closed else "false",
                "limit": _PAGE_LIMIT,
                "offset": offset,
            })
            if not isinstance(page, list) or not page:
                break
            out.extend(page)
            if len(page) < _PAGE_LIMIT:
                break
            offset += len(page)
        else:
            logger.warning("gamma fetch_events hit max_pages={}", max_pages)
        return out

    @staticmethod
    def iter_binary_markets(events: list[dict]) -> Iterator[dict]:
        """Yield single-market binary entries (skips multi-market bucket events)."""
        for ev in events:
            markets = ev.get("markets") or []
            if len(markets) != 1:
                continue
            mk = markets[0]
            if not mk.get("clobTokenIds") or not mk.get("conditionId"):
                continue
            yield mk
