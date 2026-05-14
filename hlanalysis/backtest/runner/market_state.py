"""Runner-side market state alongside hftbacktest's depth.

hftbacktest owns per-leg book + position; this module owns reference-HLC
state the strategies read on each scan tick:

- top-of-book ``BookState`` per leg (updated from ``BookSnapshot``)
- recent reference HLC + close-to-close log returns over a lookback window

Returns + HL windows are stored in `KlineRingBuffer` (a numpy-backed ring
buffer with O(1) append and O(1) windowed slice — see
`hlanalysis/strategy/_numba/returns_buffer.py`). The per-tick rebuild that
the original `sim.market_state.SimMarketState` did was the dominant
hot-path cost at scan cadence; switching to the ring buffer is what makes
the JIT'd σ helpers actually pay off.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hlanalysis.strategy._numba.returns_buffer import KlineRingBuffer
from hlanalysis.strategy.types import BookState

from ..core.events import BookSnapshot, ReferenceEvent


@dataclass(slots=True)
class MarketState:
    """In-memory book + recent reference HLC."""

    _books: dict[str, BookState] = None  # type: ignore[assignment]
    _klines: KlineRingBuffer = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._books = {}
        self._klines = KlineRingBuffer()

    # ---- L2 / trade timestamps -------------------------------------------

    def apply_l2(self, snap: BookSnapshot) -> None:
        bid_px = snap.bids[0][0] if snap.bids else None
        bid_sz = snap.bids[0][1] if snap.bids else None
        ask_px = snap.asks[0][0] if snap.asks else None
        ask_sz = snap.asks[0][1] if snap.asks else None
        existing = self._books.get(snap.symbol)
        last_trade_ts_ns = existing.last_trade_ts_ns if existing is not None else 0
        self._books[snap.symbol] = BookState(
            symbol=snap.symbol,
            bid_px=bid_px,
            bid_sz=bid_sz,
            ask_px=ask_px,
            ask_sz=ask_sz,
            last_trade_ts_ns=last_trade_ts_ns,
            last_l2_ts_ns=snap.ts_ns,
        )

    def apply_trade_ts(self, symbol: str, ts_ns: int) -> None:
        b = self._books.get(symbol)
        if b is None:
            return
        self._books[symbol] = BookState(
            symbol=b.symbol,
            bid_px=b.bid_px,
            bid_sz=b.bid_sz,
            ask_px=b.ask_px,
            ask_sz=b.ask_sz,
            last_trade_ts_ns=ts_ns,
            last_l2_ts_ns=b.last_l2_ts_ns,
        )

    def book(self, symbol: str) -> BookState | None:
        return self._books.get(symbol)

    # ---- Reference HLC (kline-like) --------------------------------------

    def apply_reference(self, ev: ReferenceEvent) -> None:
        self._klines.append(ts_ns=ev.ts_ns, high=ev.high, low=ev.low, close=ev.close)

    def latest_btc_close(self) -> float | None:
        return self._klines.latest_close()

    def recent_returns(self, *, now_ns: int, lookback_seconds: int) -> np.ndarray:
        rets, _hl = self._klines.slice_window(
            now_ns=now_ns, lookback_seconds=lookback_seconds
        )
        return rets

    def recent_hl_bars(self, *, now_ns: int, lookback_seconds: int) -> np.ndarray:
        _rets, hl = self._klines.slice_window(
            now_ns=now_ns, lookback_seconds=lookback_seconds
        )
        return hl


__all__ = ["MarketState"]
