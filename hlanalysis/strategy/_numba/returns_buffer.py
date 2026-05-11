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

``slice_window(now_ns, lookback_seconds)`` returns ``(returns, hl_bars)``
as numpy arrays:
  - ``returns`` — 1-D float64 of log returns whose BOTH endpoints lie in
    ``[now - window, now]`` and whose closes were both positive. Degenerate
    bars are filtered.
  - ``hl_bars`` — 2-D float64 of shape ``(N, 2)`` with rows ``[high, low]``
    for kept bars with ``H > 0`` and ``L > 0``.

Both arrays are zero-copy where possible; the only allocations are the
boolean masks. Returning ndarrays (instead of tuples of floats) is the
single biggest perf win identified in the 2026-05-11 profile — building
the tuple of Python floats per tick is ~4× slower than the buffer's
indexing work itself.
"""
from __future__ import annotations

import math

import numpy as np

_EMPTY_F64 = np.empty(0, dtype=np.float64)
_EMPTY_HL = np.empty((0, 2), dtype=np.float64)


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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(returns_1d, hl_2d)`` views into the kept-bar window.

        Returns are emitted only when BOTH endpoints (i-1 and i) are in
        ``[now - window, now]`` AND both closes were positive (NaN entries
        from the precomputed ret array are filtered). The HL array contains
        one row per kept bar with ``H > 0`` and ``L > 0``.
        """
        if self._len == 0:
            return _EMPTY_F64, _EMPTY_HL
        cutoff = now_ns - lookback_seconds * 1_000_000_000
        ts_view = self._ts[: self._len]
        lo_idx = int(np.searchsorted(ts_view, cutoff, side="left"))
        hi_idx = int(np.searchsorted(ts_view, now_ns, side="right"))
        if hi_idx <= lo_idx:
            return _EMPTY_F64, _EMPTY_HL
        ret_slice = self._ret[lo_idx + 1 : hi_idx]
        if ret_slice.size:
            mask = ~np.isnan(ret_slice)
            rets = np.ascontiguousarray(ret_slice[mask])
        else:
            rets = _EMPTY_F64
        h_slice = self._high[lo_idx:hi_idx]
        l_slice = self._low[lo_idx:hi_idx]
        hl_mask = (h_slice > 0.0) & (l_slice > 0.0)
        if hl_mask.all() and hl_mask.size:
            hl = np.empty((hl_mask.size, 2), dtype=np.float64)
            hl[:, 0] = h_slice
            hl[:, 1] = l_slice
        else:
            kept_h = h_slice[hl_mask]
            kept_l = l_slice[hl_mask]
            hl = np.empty((kept_h.size, 2), dtype=np.float64)
            hl[:, 0] = kept_h
            hl[:, 1] = kept_l
        return rets, hl
