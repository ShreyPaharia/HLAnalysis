# hlanalysis/sim/market_state.py
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from hlanalysis.strategy.types import BookState

from .data.binance_klines import Kline
from .synthetic_l2 import L2Snapshot


@dataclass(slots=True)
class SimMarketState:
    vol_sampling_dt_seconds: int
    _books: dict[str, BookState] = None  # type: ignore[assignment]
    _kline_closes: deque = None          # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._books = {}
        self._kline_closes = deque()  # (ts_ns, close)

    def apply_l2(self, snap: L2Snapshot) -> None:
        self._books[snap.token_id] = BookState(
            symbol=snap.token_id,
            bid_px=snap.bid_px, bid_sz=snap.bid_sz,
            ask_px=snap.ask_px, ask_sz=snap.ask_sz,
            last_trade_ts_ns=0,
            last_l2_ts_ns=snap.ts_ns,
        )

    def apply_trade_ts(self, token_id: str, ts_ns: int) -> None:
        b = self._books.get(token_id)
        if b is not None:
            self._books[token_id] = BookState(
                symbol=b.symbol, bid_px=b.bid_px, bid_sz=b.bid_sz,
                ask_px=b.ask_px, ask_sz=b.ask_sz,
                last_trade_ts_ns=ts_ns, last_l2_ts_ns=b.last_l2_ts_ns,
            )

    def book(self, token_id: str) -> BookState | None:
        return self._books.get(token_id)

    def apply_kline(self, k: Kline) -> None:
        self._kline_closes.append((k.ts_ns, k.close))

    def latest_btc_close(self) -> float | None:
        return self._kline_closes[-1][1] if self._kline_closes else None

    def recent_returns(self, *, now_ns: int, lookback_seconds: int) -> tuple[float, ...]:
        window_ns = lookback_seconds * 1_000_000_000
        cutoff = now_ns - window_ns
        prices = [(t, c) for (t, c) in self._kline_closes if t >= cutoff and t <= now_ns]
        if len(prices) < 2:
            return ()
        returns = []
        for i in range(1, len(prices)):
            p_prev, p_now = prices[i - 1][1], prices[i][1]
            if p_prev > 0 and p_now > 0:
                returns.append(math.log(p_now / p_prev))
        return tuple(returns)
