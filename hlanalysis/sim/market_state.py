# hlanalysis/sim/market_state.py
from __future__ import annotations

from dataclasses import dataclass

from hlanalysis.strategy._numba.returns_buffer import KlineRingBuffer
from hlanalysis.strategy.types import BookState

from .data.binance_klines import Kline
from .synthetic_l2 import L2Snapshot


@dataclass(slots=True)
class SimMarketState:
    """In-memory book + recent BTC log-returns. Strategies decide their own
    sampling cadence; this just collects the inputs."""
    _books: dict[str, BookState] = None  # type: ignore[assignment]
    _klines: KlineRingBuffer = None      # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._books = {}
        self._klines = KlineRingBuffer()

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
        self._klines.append(ts_ns=k.ts_ns, high=k.high, low=k.low, close=k.close)

    def latest_btc_close(self) -> float | None:
        return self._klines.latest_close()

    def recent_returns(
        self, *, now_ns: int, lookback_seconds: int
    ) -> tuple[float, ...]:
        rets, _hls = self._klines.slice_window(
            now_ns=now_ns, lookback_seconds=lookback_seconds
        )
        return rets

    def recent_hl_bars(
        self, *, now_ns: int, lookback_seconds: int
    ) -> tuple[tuple[float, float], ...]:
        """Return (high, low) tuples for klines in the lookback window. Used by
        the Parkinson range-based σ estimator."""
        _rets, hls = self._klines.slice_window(
            now_ns=now_ns, lookback_seconds=lookback_seconds
        )
        return hls
