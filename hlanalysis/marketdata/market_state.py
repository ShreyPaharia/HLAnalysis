"""Shared, pure event-driven MarketState core (SHR-81 / SHR-73, Spec 2 T1).

The ONE market-state implementation that both the live engine and the backtest
will route through (Wave 2: SHR-86/87). Today there are two copies —
``engine/market_state.py`` (deque-backed, multi-symbol, streaming ticks) and
``backtest/runner/market_state.py`` (KlineRingBuffer-backed, single reference
symbol, pre-bucketed bars) — and the input-skew between them is the residual
live-vs-sim gap this program is closing by *code reuse* rather than a diff
harness (volume SHR-78, windowing/cadence SHR-66/80).

This module is the unified core. It is a **pure function of the event
sequence** — deterministic, no IO, no asyncio, no process-salted hashing — so a
fixed event stream always yields identical query outputs. It normalizes at the
edges: callers map their native events (engine ``NormalizedEvent`` / backtest
``MarketEvent``) onto the minimal shared structs below, and everything
downstream is shared.

What it maintains, per symbol:
  * the L2 book (top-of-book + full levels);
  * OHLC reference bars, bucketed into ``dt``-wide windows via the canonical
    ``marketdata.ohlc`` bucketer, one independent buffer per registered
    ``(symbol, dt)`` cadence (engine fan-out semantics);
  * a 1-hour rolling trade-volume window with eviction on insert AND on read
    (the live-engine semantics SHR-78 pinned);
  * the latest reference price (``last_mark``).

Query surface (the canonical form both contexts route through; returns numpy
arrays — the backtest's measured zero-copy perf win, and the native input shape
for the σ estimators):

    recent_returns(symbol, *, now_ns, lookback_seconds, dt=None) -> np.ndarray
    recent_hl_bars(symbol, *, now_ns, lookback_seconds, dt=None) -> np.ndarray  # (N, 2)
    recent_returns_and_hl(symbol, *, now_ns, lookback_seconds, dt=None) -> (rets, hl)
    recent_volume_usd(symbols, *, now_ns) -> float
    book(symbol) -> BookState | None
    last_mark(symbol) -> float | None
    sigma(symbol, *, estimator, now_ns, lookback_seconds, dt=None, ewma_lambda=0.0) -> float

The windowing rule is exactly SHR-66's: a return is kept only when BOTH its
endpoints fall in ``[now - lookback, now]`` and both closes were positive
(``ret_slice = ret[lo+1:hi]``); HL rows are kept for bars in ``[lo:hi]`` with
``H > 0`` and ``L > 0``. ``_OhlcBuffer`` below reimplements
``KlineRingBuffer.slice_window`` to the line so the backtest path is
bit-identical, and adds in-place coalescing of the in-progress bucket so the
streaming tick path reproduces the engine's bars (``KlineRingBuffer`` is
append-only and lives behind the ``strategy._numba`` boundary; Wave 2 can merge
the in-place update upstream).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from hlanalysis.strategy._numba.vol import parkinson_sigma_window
from hlanalysis.strategy.types import BookState
from hlanalysis.strategy.vol import bipower_variation_sigma, sample_std_returns

from .ohlc import bucket_index, update_bar

# 1-hour rolling volume window — matches engine/market_state.py and
# backtest/runner/market_state.py (both hardcode this exact constant).
_VOLUME_WINDOW_NS: int = 60 * 60 * 1_000_000_000
# Default OHLC bucket for an unregistered symbol (60s) — matches the engine's
# ``mark_bucket_ns`` default.
_DEFAULT_BUCKET_NS: int = 60 * 1_000_000_000

_EMPTY_F64 = np.empty(0, dtype=np.float64)
_EMPTY_HL = np.empty((0, 2), dtype=np.float64)


# --------------------------------------------------------------------------
# Minimal shared event structs — callers normalize their native events onto
# these at the edge. Frozen so the core stays a pure function of the sequence.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BookEvent:
    """Full L2 snapshot. ``bids``/``asks`` are best-first ``(price, size)``."""

    ts_ns: int
    symbol: str
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class TradeEvent:
    """A trade print, consumed only for the rolling-volume gate."""

    ts_ns: int
    symbol: str
    price: float
    size: float


@dataclass(frozen=True, slots=True)
class ReferenceTickEvent:
    """A raw reference-price tick (e.g. perp mark, BBO mid) to be bucketed."""

    ts_ns: int
    symbol: str
    price: float


@dataclass(frozen=True, slots=True)
class ReferenceBarEvent:
    """A pre-bucketed reference OHLC bar (the backtest loader path)."""

    ts_ns: int
    symbol: str
    high: float
    low: float
    close: float


# --------------------------------------------------------------------------
# Per-(symbol, dt) OHLC bar store.
# --------------------------------------------------------------------------


class _OhlcBuffer:
    """Append-only OHLC bars with O(1) updates and SHR-66 windowed slices.

    Mirrors ``strategy._numba.returns_buffer.KlineRingBuffer`` field-for-field
    and ``slice_window`` line-for-line (so the backtest's bar-append path is
    bit-identical) and adds ``ingest_tick`` — in-place coalescing of the
    in-progress ``dt`` bucket, which the streaming tick path needs and the
    append-only ``KlineRingBuffer`` cannot do. The merge rule is the shared
    ``ohlc.update_bar``; the per-bar return is precomputed at append/update so
    ``slice_window`` is a pure slice + mask.
    """

    __slots__ = (
        "_ts",
        "_high",
        "_low",
        "_close",
        "_ret",
        "_len",
        "_cap",
        "_bucket_ns",
        "_last_bucket",
        "_slice_cache",
    )

    def __init__(self, bucket_ns: int, *, initial_capacity: int = 256) -> None:
        self._cap = int(initial_capacity)
        self._ts = np.empty(self._cap, dtype=np.int64)
        self._high = np.empty(self._cap, dtype=np.float64)
        self._low = np.empty(self._cap, dtype=np.float64)
        self._close = np.empty(self._cap, dtype=np.float64)
        self._ret = np.empty(self._cap, dtype=np.float64)
        self._len = 0
        self._bucket_ns = int(bucket_ns)
        self._last_bucket: int | None = None
        # Bit-identical slice memo (perf): consecutive scans (e.g. the 0.2s
        # event-driven cadence over a dt=5 feed, or idle-backoff scans) often
        # query a window whose content is byte-for-byte unchanged. The slice is
        # fully determined by the window endpoints plus the ONLY mutable bar —
        # the in-progress last bar (``ingest_tick`` only ever updates index
        # ``_len-1``; all earlier bars are frozen once their bucket closes). So
        # ``(lo_idx, hi_idx, last_high, last_low, last_close)`` is a complete
        # validity signature; on an exact match we return the SAME array objects
        # so callers can also memo their downstream tuple conversion by identity.
        # ``None`` = cold. Holds ``(key, rets, hl)``.
        self._slice_cache: tuple[tuple, np.ndarray, np.ndarray] | None = None

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

    def _recompute_ret(self, i: int) -> None:
        if i == 0:
            self._ret[i] = math.nan
            return
        prev_close = self._close[i - 1]
        close = self._close[i]
        if prev_close > 0.0 and close > 0.0:
            self._ret[i] = math.log(close / prev_close)
        else:
            self._ret[i] = math.nan

    def append_bar(self, ts_ns: int, high: float, low: float, close: float) -> None:
        """Append a fresh bar (the pre-bucketed / backtest path)."""
        if self._len == self._cap:
            self._grow()
        i = self._len
        self._ts[i] = ts_ns
        self._high[i] = high
        self._low[i] = low
        self._close[i] = close
        self._len = i + 1
        self._recompute_ret(i)
        # Keep the bucket cursor coherent if ticks are later mixed in.
        self._last_bucket = bucket_index(ts_ns, self._bucket_ns)

    def ingest_tick(self, ts_ns: int, price: float) -> None:
        """Fold one scalar tick into the current ``dt`` bucket (engine path).

        A tick in a new bucket appends a degenerate ``(price, price, price)``
        bar; a tick in the current bucket merges in place via ``ohlc.update_bar``
        (high=max, low=min, close=last) and refreshes the bar's timestamp to the
        latest tick — matching ``resample_ohlc`` (bar ts = last sample in
        window) and the engine's ``_ingest_reference_price``.
        """
        bucket = bucket_index(ts_ns, self._bucket_ns)
        if self._last_bucket is None or bucket != self._last_bucket or self._len == 0:
            self.append_bar(ts_ns, price, price, price)
            self._last_bucket = bucket
            return
        i = self._len - 1
        h, l, c = update_bar((self._high[i], self._low[i], self._close[i]), price, price, price)
        self._ts[i] = ts_ns
        self._high[i] = h
        self._low[i] = l
        self._close[i] = c
        self._recompute_ret(i)

    def latest_close(self) -> float | None:
        if self._len == 0:
            return None
        return float(self._close[self._len - 1])

    def slice_window(self, *, now_ns: int, lookback_seconds: int) -> tuple[np.ndarray, np.ndarray]:
        """``(returns_1d, hl_2d)`` over the SHR-66 window — see module docstring.

        Verbatim port of ``KlineRingBuffer.slice_window``; keep in lockstep.
        """
        if self._len == 0:
            return _EMPTY_F64, _EMPTY_HL
        cutoff = now_ns - lookback_seconds * 1_000_000_000
        ts_view = self._ts[: self._len]
        lo_idx = int(np.searchsorted(ts_view, cutoff, side="left"))
        hi_idx = int(np.searchsorted(ts_view, now_ns, side="right"))
        if hi_idx <= lo_idx:
            return _EMPTY_F64, _EMPTY_HL
        # Bit-identical memo: the window content is fully pinned by the endpoints
        # and the sole mutable bar (last bar, index hi_idx-1). On an exact-key
        # hit, return the SAME objects computed last time (see __init__ note).
        last = hi_idx - 1
        cache_key = (lo_idx, hi_idx, self._high[last], self._low[last], self._close[last])
        cached = self._slice_cache
        if cached is not None and cached[0] == cache_key:
            return cached[1], cached[2]
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
        self._slice_cache = (cache_key, rets, hl)
        return rets, hl


# --------------------------------------------------------------------------
# The shared MarketState.
# --------------------------------------------------------------------------


@dataclass(slots=True)
class _MutableBook:
    bid_px: float | None = None
    bid_sz: float | None = None
    ask_px: float | None = None
    ask_sz: float | None = None
    last_trade_ts_ns: int = 0
    last_l2_ts_ns: int = 0
    ask_levels: tuple[tuple[float, float], ...] = ()
    bid_levels: tuple[tuple[float, float], ...] = ()


class MarketState:
    """Shared event-driven market state. Pure (no IO, no asyncio)."""

    def __init__(
        self,
        *,
        volume_window_ns: int = _VOLUME_WINDOW_NS,
        default_bucket_ns: int = _DEFAULT_BUCKET_NS,
    ) -> None:
        self._volume_window_ns = int(volume_window_ns)
        self._default_bucket_ns = int(default_bucket_ns)
        self._books: dict[str, _MutableBook] = {}
        self._last_mark: dict[str, float] = {}
        self._last_mark_ts: dict[str, int] = {}
        # Per-symbol deque of (ts_ns, price, size) for rolling-volume accounting.
        self._trades: dict[str, deque[tuple[int, float, float]]] = {}
        # Version-cached re-sum for recent_volume_usd (FIX A).
        # _trades_version[sym] is bumped on every deque mutation (append or popleft).
        # _vol_cache[sym] stores (version_when_computed, summed_notional); if the
        # stored version equals the current version the sum is reused, else recomputed
        # with the identical sum(px*sz ...) expression so the float is bit-identical.
        self._trades_version: dict[str, int] = {}
        self._vol_cache: dict[str, tuple[int, float]] = {}
        # Per-(symbol, dt_ns) OHLC bar buffers; one feed fans into every cadence
        # registered for the symbol (engine semantics).
        self._buffers: dict[tuple[str, int], _OhlcBuffer] = {}
        # Cadences registered per symbol, in registration order. The FIRST is the
        # symbol's default (what a ``dt=None`` read resolves to).
        self._cadences: dict[str, list[int]] = {}

    # ---- cadence registration ------------------------------------------

    def set_reference_cadence(
        self,
        symbol: str,
        *,
        sampling_dt_seconds: int,
        lookback_seconds: int | None = None,  # noqa: ARG002 (sizing is dynamic)
    ) -> None:
        """Register a ``dt``-wide bucketing cadence for ``symbol``.

        A symbol may carry MULTIPLE cadences, each bucketed independently from
        the same reference feed; the first registered is the default. Re-
        registering an existing ``(symbol, dt)`` is a no-op. ``lookback_seconds``
        is accepted for signature-compatibility with the engine but ignored —
        ``_OhlcBuffer`` grows unbounded geometrically, so there is no fixed
        ring to pre-size.
        """
        if sampling_dt_seconds <= 0:
            raise ValueError(f"sampling_dt_seconds must be positive, got {sampling_dt_seconds!r}")
        ns = int(sampling_dt_seconds) * 1_000_000_000
        cadences = self._cadences.setdefault(symbol, [])
        if ns not in cadences:
            cadences.append(ns)

    def _resolve_dt_ns(self, symbol: str, dt: int | None) -> int:
        """Bucket period (ns) for a ``(symbol, dt)`` read. ``dt=None`` resolves
        to the symbol's FIRST registered cadence, else the global default."""
        if dt is not None:
            return int(dt) * 1_000_000_000
        cadences = self._cadences.get(symbol)
        return cadences[0] if cadences else self._default_bucket_ns

    def _cadences_for(self, symbol: str) -> list[int]:
        return self._cadences.get(symbol) or [self._default_bucket_ns]

    def _buffer(self, symbol: str, bucket_ns: int) -> _OhlcBuffer:
        key = (symbol, bucket_ns)
        buf = self._buffers.get(key)
        if buf is None:
            buf = _OhlcBuffer(bucket_ns)
            self._buffers[key] = buf
        return buf

    # ---- ingest --------------------------------------------------------

    def apply(
        self,
        ev: BookEvent | TradeEvent | ReferenceTickEvent | ReferenceBarEvent,
    ) -> None:
        """Dispatch one shared event onto the appropriate ingest path."""
        match ev:
            case BookEvent():
                self.apply_book(ev.symbol, ts_ns=ev.ts_ns, bids=ev.bids, asks=ev.asks)
            case TradeEvent():
                self.apply_trade(ev.symbol, ts_ns=ev.ts_ns, price=ev.price, size=ev.size)
            case ReferenceTickEvent():
                self.apply_reference_tick(ev.symbol, ts_ns=ev.ts_ns, price=ev.price)
            case ReferenceBarEvent():
                self.apply_reference_bar(
                    ev.symbol,
                    ts_ns=ev.ts_ns,
                    high=ev.high,
                    low=ev.low,
                    close=ev.close,
                )
            case _:  # pragma: no cover - defensive
                raise TypeError(f"unknown event type: {type(ev).__name__}")

    def apply_book(
        self,
        symbol: str,
        *,
        ts_ns: int,
        bids: tuple[tuple[float, float], ...],
        asks: tuple[tuple[float, float], ...],
    ) -> None:
        b = self._books.setdefault(symbol, _MutableBook())
        if bids:
            b.bid_px, b.bid_sz = bids[0]
            b.bid_levels = tuple(bids)
        if asks:
            b.ask_px, b.ask_sz = asks[0]
            b.ask_levels = tuple(asks)
        b.last_l2_ts_ns = max(b.last_l2_ts_ns, ts_ns)

    def apply_trade(self, symbol: str, *, ts_ns: int, price: float, size: float) -> None:
        dq = self._trades.setdefault(symbol, deque())
        dq.append((ts_ns, price, size))
        self._trades_version[symbol] = self._trades_version.get(symbol, 0) + 1
        self._evict(symbol, dq, now_ns=ts_ns)
        b = self._books.setdefault(symbol, _MutableBook())
        b.last_trade_ts_ns = max(b.last_trade_ts_ns, ts_ns)

    def apply_reference_tick(self, symbol: str, *, ts_ns: int, price: float) -> None:
        self._last_mark[symbol] = price
        self._last_mark_ts[symbol] = ts_ns
        for bucket_ns in self._cadences_for(symbol):
            self._buffer(symbol, bucket_ns).ingest_tick(ts_ns, price)

    def apply_reference_bar(self, symbol: str, *, ts_ns: int, high: float, low: float, close: float) -> None:
        self._last_mark[symbol] = close
        self._last_mark_ts[symbol] = ts_ns
        # A pre-bucketed bar is appended to the symbol's default cadence buffer
        # (bars carry their own dt; multi-cadence fan-out is a tick-path concern).
        bucket_ns = self._resolve_dt_ns(symbol, None)
        self._buffer(symbol, bucket_ns).append_bar(ts_ns, high, low, close)

    def _evict(self, symbol: str, dq: deque[tuple[int, float, float]], *, now_ns: int) -> None:
        cutoff = now_ns - self._volume_window_ns
        while dq and dq[0][0] < cutoff:
            dq.popleft()
            self._trades_version[symbol] = self._trades_version.get(symbol, 0) + 1

    # ---- query ---------------------------------------------------------

    def book(self, symbol: str) -> BookState | None:
        b = self._books.get(symbol)
        if b is None:
            return None
        return BookState(
            symbol=symbol,
            bid_px=b.bid_px,
            bid_sz=b.bid_sz,
            ask_px=b.ask_px,
            ask_sz=b.ask_sz,
            last_trade_ts_ns=b.last_trade_ts_ns,
            last_l2_ts_ns=b.last_l2_ts_ns,
            ask_levels=b.ask_levels,
            bid_levels=b.bid_levels,
        )

    def last_mark(self, symbol: str) -> float | None:
        return self._last_mark.get(symbol)

    def last_mark_ts(self, symbol: str) -> int | None:
        return self._last_mark_ts.get(symbol)

    def _slice(self, symbol: str, dt: int | None, now_ns: int, lookback_seconds: int) -> tuple[np.ndarray, np.ndarray]:
        buf = self._buffers.get((symbol, self._resolve_dt_ns(symbol, dt)))
        if buf is None:
            return _EMPTY_F64, _EMPTY_HL
        return buf.slice_window(now_ns=now_ns, lookback_seconds=lookback_seconds)

    def recent_returns(
        self,
        symbol: str,
        *,
        now_ns: int,
        lookback_seconds: int,
        dt: int | None = None,
    ) -> np.ndarray:
        return self._slice(symbol, dt, now_ns, lookback_seconds)[0]

    def recent_hl_bars(
        self,
        symbol: str,
        *,
        now_ns: int,
        lookback_seconds: int,
        dt: int | None = None,
    ) -> np.ndarray:
        return self._slice(symbol, dt, now_ns, lookback_seconds)[1]

    def recent_returns_and_hl(
        self,
        symbol: str,
        *,
        now_ns: int,
        lookback_seconds: int,
        dt: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._slice(symbol, dt, now_ns, lookback_seconds)

    def recent_volume_usd(self, symbols: str | Iterable[str], *, now_ns: int) -> float:
        """Total traded notional (Σ price·size) over the last hour as of
        ``now_ns``, summed across ``symbols`` (a single symbol or an iterable of
        leg symbols). Evicts stale entries per-symbol on read — the live engine
        + backtest semantics SHR-78 pinned.

        Performance (FIX A): uses a version-cached re-sum.  The deque version
        is incremented on every mutation (append in ``apply_trade``, each
        ``popleft`` in ``_evict``).  If the version has not changed since the
        last computed sum the cached value is returned without iterating the
        deque.  When the version has changed the identical
        ``sum(px*sz for _,px,sz in dq)`` expression is used so the float
        result is bit-identical to the un-cached path.
        """
        syms: Iterable[str] = (symbols,) if isinstance(symbols, str) else symbols
        total = 0.0
        for sym in syms:
            dq = self._trades.get(sym)
            if dq is None:
                continue
            self._evict(sym, dq, now_ns=now_ns)
            ver = self._trades_version.get(sym, 0)
            cached = self._vol_cache.get(sym)
            if cached is not None and cached[0] == ver:
                total += cached[1]
            else:
                s = sum(px * sz for _, px, sz in dq)
                self._vol_cache[sym] = (ver, s)
                total += s
        return total

    def sigma(
        self,
        symbol: str,
        *,
        estimator: str,
        now_ns: int,
        lookback_seconds: int,
        dt: int | None = None,
        ewma_lambda: float = 0.0,
    ) -> float:
        """Per-sample / per-bar realized σ over the SHR-66 window.

        Delegates to the shared estimators (no annualization — that is a
        strategy-layer convention the v2/v3.1 families apply differently and
        which ``vol.py`` documents must NOT be unified here):
          * ``"stdev"``     → ``sample_std_returns`` over close-to-close returns;
          * ``"bipower"``   → ``bipower_variation_sigma`` over those returns;
          * ``"parkinson"`` → ``parkinson_sigma_window`` over ``(H, L)`` bars,
            with optional EWMA aggregation via ``ewma_lambda`` (0 = mean).
        """
        rets, hl = self._slice(symbol, dt, now_ns, lookback_seconds)
        if estimator == "stdev":
            return float(sample_std_returns(rets))
        if estimator == "bipower":
            return float(bipower_variation_sigma(rets))
        if estimator == "parkinson":
            highs = np.ascontiguousarray(hl[:, 0]) if hl.size else _EMPTY_F64
            lows = np.ascontiguousarray(hl[:, 1]) if hl.size else _EMPTY_F64
            return float(parkinson_sigma_window(highs, lows, float(ewma_lambda)))
        raise ValueError(f"unknown vol estimator: {estimator!r}")


__all__ = [
    "BookEvent",
    "TradeEvent",
    "ReferenceTickEvent",
    "ReferenceBarEvent",
    "MarketState",
]
