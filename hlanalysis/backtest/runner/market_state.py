"""Runner-side market state — a THIN ADAPTER over the shared core.

SHR-86 collapses the backtest's bespoke ``MarketState`` onto the one shared
implementation in ``hlanalysis/marketdata/market_state.py`` (SHR-81), so the
sim and the live engine read σ / returns / HL bars / volume through the exact
same code — closing the train/serve skew by *reuse*, not a diff harness.

This module keeps the runner's narrow, single-reference public surface so
``hftbt_runner.py`` needs no change:

  * ``apply_l2`` / ``apply_trade`` / ``apply_reference`` / ``apply_reference_tick``
    map the runner's ``core.events`` structs onto the shared core's ingest methods;
  * ``recent_returns`` / ``recent_hl_bars`` / ``recent_returns_and_hl`` /
    ``recent_volume_usd`` / ``book`` / ``latest_btc_close`` delegate to it.

The runner is *single-reference*: its query methods take no ``symbol`` (there is
one reference feed per question), so every reference bar routes through one
fixed key (``_REFERENCE_KEY``) in the shared multi-symbol core, exactly as the
old ``KlineRingBuffer``-backed implementation kept a single bar buffer. Books
and trades are still keyed by their real (leg) symbol.

Bit-identical: the shared core's ``_OhlcBuffer`` is a verbatim port of
``KlineRingBuffer`` (append + SHR-66 ``slice_window``), and the rolling-volume
window matches the old hardcoded 1-hour constant, so v31/v1 HL backtest outputs
reproduce byte-for-byte (validated by the parity + speedup suites and the
SHR-86 before/after fingerprint).
"""

from __future__ import annotations

import numpy as np

from hlanalysis.marketdata.decision_kernel import build_decision_inputs
from hlanalysis.marketdata.market_state import MarketState as _CoreMarketState
from hlanalysis.strategy.types import BookState

from ..core.events import BookSnapshot, ReferenceEvent, TradeEvent

# 1-hour rolling window — matches engine/market_state.py default and the old
# runner constant.
_VOLUME_WINDOW_NS: int = 60 * 60 * 1_000_000_000

# Single-reference key. The runner feeds one reference symbol per question and
# its query methods omit the symbol, so all reference bars share one buffer in
# the shared core (mirrors the old single ``KlineRingBuffer``).
_REFERENCE_KEY: str = "__reference__"


class MarketState:
    """Thin single-reference adapter over the shared event-driven core."""

    __slots__ = ("_core",)

    def __init__(self) -> None:
        self._core = _CoreMarketState(volume_window_ns=_VOLUME_WINDOW_NS)

    # ---- L2 / trade ingest ----------------------------------------------

    def apply_l2(self, snap: BookSnapshot) -> None:
        self._core.apply_book(snap.symbol, ts_ns=snap.ts_ns, bids=snap.bids, asks=snap.asks)

    def apply_trade(self, ev: TradeEvent) -> None:
        """Record a traded (price, size) pair for rolling-volume accounting.

        Delegates to the shared core, which evicts entries older than the
        1-hour window on insert AND on read.
        """
        self._core.apply_trade(ev.symbol, ts_ns=ev.ts_ns, price=ev.price, size=ev.size)

    def recent_volume_usd(self, leg_symbols: tuple[str, ...] | list[str], *, now_ns: int) -> float:
        """Total traded notional (Σ price·size) across ``leg_symbols`` within
        the last hour, as of ``now_ns``. Window-bounded per symbol (stale
        entries evicted on read), so recycled coins never leak old volume."""
        return self._core.recent_volume_usd(leg_symbols, now_ns=now_ns)

    def book(self, symbol: str) -> BookState | None:
        return self._core.book(symbol)

    # ---- cadence configuration ------------------------------------------

    def set_reference_cadence(self, dt_seconds: int) -> None:
        """Configure the OHLC bucket width for the reference feed.

        Must be called BEFORE the first ``apply_reference_tick`` when using the
        raw-tick path (SHR-96). The ``dt_seconds`` must match
        ``vol_sampling_dt_seconds`` so the buffer produces dt-spaced bar
        close-to-close returns — the same spacing the strategy assumes when
        annualizing σ.

        Without this call the buffer uses the 60s default, which inflates σ by
        ``sqrt(60 / dt_seconds)`` when ``dt_seconds < 60``.

        Re-registering the same ``dt`` is a no-op (shared core semantics).
        """
        self._core.set_reference_cadence(_REFERENCE_KEY, sampling_dt_seconds=int(dt_seconds))

    # ---- reference HLC (kline-like) -------------------------------------

    def apply_reference(self, ev: ReferenceEvent) -> None:
        self._core.apply_reference_bar(_REFERENCE_KEY, ts_ns=ev.ts_ns, high=ev.high, low=ev.low, close=ev.close)

    def apply_reference_tick(self, ev: ReferenceEvent) -> None:
        """Feed a raw reference tick (H=L=C=mid) into the shared core's tick path.

        Unlike ``apply_reference`` (which appends a pre-bucketed bar), this calls
        ``apply_reference_tick`` on the shared core, which:
          - sets ``last_mark`` to the tick price IMMEDIATELY (live-parity), and
          - folds the tick into the current ``dt`` OHLC bucket in-place so σ is
            computed from correctly-bucketed returns.

        Used by the runner when ``HLHip4DataSource.reference_ticks == "raw"``
        (SHR-93): the loader yields one ``ReferenceEvent`` per recorded tick and
        the runner calls this method instead of ``apply_reference``, matching the
        live engine path that also routes raw ticks through ``apply_reference_tick``.
        """
        self._core.apply_reference_tick(_REFERENCE_KEY, ts_ns=ev.ts_ns, price=ev.close)

    def latest_btc_close(self) -> float | None:
        return self._core.last_mark(_REFERENCE_KEY)

    def recent_returns(self, *, now_ns: int, lookback_seconds: int) -> np.ndarray:
        rets, _ = build_decision_inputs(
            self._core,
            ref_symbol=_REFERENCE_KEY,
            now_ns=now_ns,
            lookback_seconds=lookback_seconds,
        )
        return rets

    def recent_hl_bars(self, *, now_ns: int, lookback_seconds: int) -> np.ndarray:
        _, hl = build_decision_inputs(
            self._core,
            ref_symbol=_REFERENCE_KEY,
            now_ns=now_ns,
            lookback_seconds=lookback_seconds,
        )
        return hl

    def recent_returns_and_hl(self, *, now_ns: int, lookback_seconds: int) -> tuple[np.ndarray, np.ndarray]:
        return build_decision_inputs(
            self._core,
            ref_symbol=_REFERENCE_KEY,
            now_ns=now_ns,
            lookback_seconds=lookback_seconds,
        )


__all__ = ["MarketState"]
