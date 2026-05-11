"""Incremental kline buffer used by ``SimMarketState`` (and, after task E,
by ``backtest/runner/market_state.py``).

Replaces the per-tick deque scan that the 10k-tick profile showed
dominates the strategy hot path (62% recent_returns + 34% recent_hl_bars
of cumtime in the legacy implementation).

Layout: parallel float64 arrays of ``ts_ns / high / low / close`` plus a
precomputed ``log_return`` array aligned to the SAME index — the return
AT index i is computed at append time as ``log(close[i] / close[i-1])``
when both closes are positive, else NaN. ``log_return[0]`` is always
NaN.

``slice_window(now_ns, lookback_seconds)`` returns:
  - ``returns`` — tuple of floats matching the legacy ``recent_returns``:
    log returns whose BOTH endpoints lie in ``[now - window, now]`` AND
    whose closes were both positive. NaN entries (degenerate prices) are
    filtered.
  - ``hl_bars`` — tuple of ``(high, low)`` pairs for kept bars with
    ``H > 0`` and ``L > 0`` (matches legacy ``recent_hl_bars``).
"""
from __future__ import annotations

import math

import numpy as np


class KlineRingBuffer:
    """Append-only kline store with O(1) updates and O(log N) windowed lookups."""

    __slots__ = ("_ts", "_high", "_low", "_close", "_ret", "_len", "_cap")

    def __init__(self, initial_capacity: int = 4096) -> None:
        self._cap = int(initial_capacity)
        self._ts = np.empty(self._cap, dtype=np.int64)
        self._high = np.empty(self._cap, dtype=np.float64)
        self._low = np.empty(self._cap, dtype=np.float64)
        self._close = np.empty(self._cap, dtype=np.float64)
        self._ret = np.empty(self._cap, dtype=np.float64)
        self._len = 0

    def _grow(self) -> None:
        new_cap = self._cap * 2
        for name in ("_ts", "_high", "_low", "_close", "_ret"):
            old = getattr(self, name)
            new = np.empty(new_cap, dtype=old.dtype)
            new[: self._len] = old[: self._len]
            setattr(self, name, new)
        self._cap = new_cap

    def __len__(self) -> int:
        return self._len

    def append(
        self, *, ts_ns: int, high: float, low: float, close: float
    ) -> None:
        if self._len == self._cap:
            self._grow()
        i = self._len
        self._ts[i] = ts_ns
        self._high[i] = high
        self._low[i] = low
        self._close[i] = close
        if i == 0:
            self._ret[i] = math.nan
        else:
            prev_close = self._close[i - 1]
            if prev_close > 0.0 and close > 0.0:
                self._ret[i] = math.log(close / prev_close)
            else:
                self._ret[i] = math.nan
        self._len = i + 1

    def latest_close(self) -> float | None:
        if self._len == 0:
            return None
        return float(self._close[self._len - 1])

    def slice_window(
        self, *, now_ns: int, lookback_seconds: int
    ) -> tuple[tuple[float, ...], tuple[tuple[float, float], ...]]:
        if self._len == 0:
            return (), ()
        cutoff = now_ns - lookback_seconds * 1_000_000_000
        ts_view = self._ts[: self._len]
        lo_idx = int(np.searchsorted(ts_view, cutoff, side="left"))
        hi_idx = int(np.searchsorted(ts_view, now_ns, side="right"))
        if hi_idx <= lo_idx:
            return (), ()
        # Returns: ret[i] is valid only when BOTH endpoints (i-1 and i) are
        # in the window. The first kept index can't have a valid return since
        # its predecessor is outside the window, so start at lo_idx + 1.
        ret_slice = self._ret[lo_idx + 1 : hi_idx]
        if ret_slice.size:
            mask = ~np.isnan(ret_slice)
            rets = tuple(float(x) for x in ret_slice[mask])
        else:
            rets = ()
        # HL bars: include any kept bar whose H>0 and L>0.
        h_slice = self._high[lo_idx:hi_idx]
        l_slice = self._low[lo_idx:hi_idx]
        hl_mask = (h_slice > 0.0) & (l_slice > 0.0)
        hls = tuple(
            (float(h), float(l)) for h, l in zip(h_slice[hl_mask], l_slice[hl_mask])
        )
        return rets, hls
