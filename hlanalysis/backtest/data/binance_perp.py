"""Synthesizes BBO BookSnapshot events from Binance USDM perp 1m klines.

Consumed by the runner as the hedge book stream for v5_delta_hedged.

Note: BookSnapshot uses bids/asks as tuple[tuple[float, float], ...] where each
inner tuple is (price, size). The plan's spec used bid_px/ask_px as separate
fields, but the real dataclass uses tuple-of-levels. Semantics are identical.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..core.events import BookSnapshot

_HEDGE_DEPTH = 1_000_000.0  # treat hedge venue as infinitely deep at top-of-book

# Module-level cache: path → parsed list[dict].  The JSON file is 40-60 MB for
# a year of 1m klines; re-parsing it per question (363 markets) costs ~minutes.
# Keyed by the resolved absolute path string so different Path objects pointing
# to the same file share the same cache entry.
_KLINES_CACHE: dict[str, list[dict]] = {}


def _load_klines(path: Path) -> list[dict]:
    """Return cached parsed rows for ``path``, loading once on first call."""
    key = str(path.resolve())
    if key not in _KLINES_CACHE:
        _KLINES_CACHE[key] = json.loads(path.read_text())
    return _KLINES_CACHE[key]


@dataclass(frozen=True, slots=True)
class BinancePerpKlinesSource:
    path: Path
    symbol: str = "BTC-PERP"
    half_spread_bps: float = 1.0

    def book_events(self, *, start_ts_ns: int, end_ts_ns: int) -> Iterator[BookSnapshot]:
        rows = _load_klines(self.path)
        h = self.half_spread_bps / 1e4
        for row in rows:
            ts = int(row["ts_ns"])
            if ts < start_ts_ns or ts >= end_ts_ns:
                continue
            close = float(row["close"])
            bid_px = close * (1.0 - h)
            ask_px = close * (1.0 + h)
            yield BookSnapshot(
                ts_ns=ts,
                symbol=self.symbol,
                bids=((bid_px, _HEDGE_DEPTH),),
                asks=((ask_px, _HEDGE_DEPTH),),
            )


__all__ = ["BinancePerpKlinesSource"]
