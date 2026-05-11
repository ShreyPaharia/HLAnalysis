"""Side-by-side state the strategy reads alongside hftbacktest's depth.

hftbacktest owns per-leg book + position. This module owns the runner-side
state the strategy needs that hftbacktest does not track:

- recent BTC reference HLC bars (from ``ReferenceEvent``s)
- close-to-close returns + per-bar (high, low) windows over a lookback

Method signatures and field names mirror ``sim.market_state.SimMarketState``
exactly so Task D can swap in numba-accelerated implementations without
touching the runner. The L2 input is now the abstract ``BookSnapshot`` from
``backtest.core.events``.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from hlanalysis.strategy.types import BookState

from ..core.events import BookSnapshot, ReferenceEvent


@dataclass(slots=True)
class MarketState:
    """In-memory book + recent reference HLC. Strategies decide sampling cadence."""

    _books: dict[str, BookState] = None  # type: ignore[assignment]
    _ref_bars: deque = None  # type: ignore[assignment]  # (ts_ns, high, low, close)

    def __post_init__(self) -> None:
        self._books = {}
        self._ref_bars = deque()

    # ---- L2 / trade timestamps -------------------------------------------

    def apply_l2(self, snap: BookSnapshot) -> None:
        """Update top-of-book state from a snapshot.

        Strategies consume the best level only via ``BookState``; deeper
        levels live in hftbacktest's depth and are not mirrored here.
        """
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
        self._ref_bars.append((ev.ts_ns, ev.high, ev.low, ev.close))

    def latest_btc_close(self) -> float | None:
        return self._ref_bars[-1][3] if self._ref_bars else None

    def recent_returns(self, *, now_ns: int, lookback_seconds: int) -> tuple[float, ...]:
        window_ns = lookback_seconds * 1_000_000_000
        cutoff = now_ns - window_ns
        prices = [(t, c) for (t, _h, _l, c) in self._ref_bars if cutoff <= t <= now_ns]
        if len(prices) < 2:
            return ()
        returns: list[float] = []
        for i in range(1, len(prices)):
            p_prev, p_now = prices[i - 1][1], prices[i][1]
            if p_prev > 0 and p_now > 0:
                returns.append(math.log(p_now / p_prev))
        return tuple(returns)

    def recent_hl_bars(
        self, *, now_ns: int, lookback_seconds: int
    ) -> tuple[tuple[float, float], ...]:
        window_ns = lookback_seconds * 1_000_000_000
        cutoff = now_ns - window_ns
        return tuple(
            (h, l)
            for (t, h, l, _c) in self._ref_bars
            if cutoff <= t <= now_ns and h > 0 and l > 0
        )


__all__ = ["MarketState"]
