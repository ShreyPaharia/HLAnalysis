"""Shared decision-kernel input assembly (R1).

Single function that reads the strategy's σ / drift inputs — ``recent_returns``
and ``recent_hl_bars`` — from the shared ``MarketState`` core.  Before R1 both
the live engine (``engine/scanner.py``) and the backtest runner
(``backtest/runner/hftbt_runner.py``) hand-assembled these reads independently,
meaning a future cadence-wiring change had to be applied in two places.

``build_decision_inputs`` collapses the read into one canonical call so both
paths are guaranteed to query the same symbol, at the same cadence (``dt``), over
the same window (``lookback_seconds``), at the same logical time (``now_ns``).

Usage
-----
Engine (``scanner.py``, per-question cadence branch)::

    from hlanalysis.marketdata.decision_kernel import build_decision_inputs
    rets, hl = build_decision_inputs(
        self.ms._core,
        ref_symbol=self.ref_symbol,
        now_ns=now_ns,
        lookback_seconds=lookback_s,
        dt=dt_s,           # None → symbol's default registered cadence
    )
    recent_returns = tuple(rets.tolist())
    recent_hl_bars = tuple((float(h), float(lo)) for h, lo in hl)

Backtest (``hftbt_runner.py``)::

    from hlanalysis.marketdata.decision_kernel import build_decision_inputs
    recent_returns, recent_hl = build_decision_inputs(
        state._core,
        ref_symbol=_REFERENCE_KEY,
        now_ns=now_ns,
        lookback_seconds=cfg.vol_lookback_seconds,
    )

DISCREPANCIES SURFACED BY R1
-----------------------------
R1 audited the two pre-existing hand-assembly paths and found ONE structural
difference:

1. **Return types**: the engine path converted numpy arrays to ``tuple``
   (``tuple(arr.tolist())``) before passing them to ``strategy.evaluate()``,
   while the backtest passed numpy arrays directly.  The ``Strategy.evaluate()``
   contract declares ``recent_returns: tuple[float, ...]`` and
   ``recent_hl_bars: tuple[tuple[float, float], ...]`` — the backtest was
   violating the typed contract.  In practice this was silent because the
   strategy implementations call ``np.asarray(window)`` internally, so duck-
   typing papered over the mismatch.  The engine conversion is the CORRECT
   behavior; callers of this function should follow the engine pattern (convert
   to tuple before ``evaluate()``) rather than the backtest pattern.

The σ-input VALUES (the numpy arrays themselves) are byte-identical between the
two paths when both are configured with the same ``lookback_seconds`` and ``dt``
— confirmed by the R3 parity gate (sigma median_rel = 0.0 at all 110 comparable
points on the fixture corpus, 2026-06-12).

Everything else (books assembly, reference_price, recent_volume_usd) is
DATA-SOURCE-SPECIFIC and cannot be shared: the engine reads from live L2 and the
backtest reads from hftbacktest depth arrays.  Those assembly steps stay in their
respective callers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hlanalysis.marketdata.market_state import MarketState


def build_decision_inputs(
    core: MarketState,
    *,
    ref_symbol: str,
    now_ns: int,
    lookback_seconds: int,
    dt: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read σ/drift inputs from the shared MarketState core.

    Parameters
    ----------
    core:
        The shared ``marketdata.MarketState`` instance (NOT the engine or
        backtest adapter wrappers).  Both adapters expose a ``._core``
        attribute pointing to it.
    ref_symbol:
        Reference symbol to query (e.g. ``"BTC"`` for the engine, or the
        runner-internal ``_REFERENCE_KEY = "__reference__"`` for the backtest).
    now_ns:
        Current logical clock (nanoseconds since epoch).  The SHR-66 window
        is ``[now_ns - lookback_seconds * 1e9, now_ns]``.
    lookback_seconds:
        Σ/drift window width in seconds.  Passed through to the core's
        ``slice_window`` as-is; do NOT pre-multiply by the bar width.
    dt:
        OHLC bucket cadence in seconds, or ``None`` to resolve to the
        symbol's first registered cadence (the default cadence).  Must match
        ``vol_sampling_dt_seconds`` so the buffer produces correctly-spaced
        bars (otherwise σ is inflated by ``sqrt(dt_registered / dt_requested)``
        — the SHR-96 interaction bug).

    Returns
    -------
    recent_returns : np.ndarray, shape (N,), dtype float64
        Close-to-close log returns over the SHR-66 window.  ``N`` may be 0
        if fewer than 2 bars have been ingested.
    recent_hl_bars : np.ndarray, shape (M, 2), dtype float64
        Per-bucket ``[[high, low], ...]`` rows over the same window.  ``M``
        may differ from ``N`` by ±1 because returns are edge-to-edge while HL
        bars are per-bucket.

    Notes
    -----
    The returned arrays are numpy views / slices; callers that need immutable
    tuples (``Strategy.evaluate`` contract) must convert::

        rets_t = tuple(recent_returns.tolist())
        hl_t   = tuple((float(h), float(lo)) for h, lo in recent_hl_bars)

    The backtest historically passed numpy arrays directly to ``evaluate()``
    (duck-typed silence).  New code should follow the contract and convert.
    """
    return core.recent_returns_and_hl(
        ref_symbol,
        now_ns=now_ns,
        lookback_seconds=lookback_seconds,
        dt=dt,
    )


__all__ = ["build_decision_inputs"]
