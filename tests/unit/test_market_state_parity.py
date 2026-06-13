"""Engine vs backtest MarketState parity.

``engine/market_state.py`` (event-driven, the LIVE path) and
``backtest/runner/market_state.py`` (vectorized, the SIM path) are two different
``MarketState`` classes the strategy must see IDENTICALLY — same recent_returns,
same recent_hl_bars from the same reference feed. They reach those bars by
different routes:

  * LIVE: raw reference ticks → engine internal OHLC bucketing (``apply``) →
    ``recent_returns(symbol, n, dt)`` / ``recent_hl_bars(symbol, n, dt)``.
  * SIM:  raw reference ticks → loader resample (``resample_ohlc``) →
    ``SimMarketState.apply_reference`` (KlineRingBuffer) →
    ``recent_returns(now_ns, lookback_seconds)`` / ``recent_hl_bars(...)``.

Both routes now bucket through the SAME canonical ``marketdata.ohlc`` code, so
the strategy-facing reads must agree. This test feeds one synthetic tick stream
through both and asserts equality — the guard against the σ-sampling train/serve
skew that recurred historically.
"""

from __future__ import annotations

import numpy as np

from hlanalysis.backtest.core.events import ReferenceEvent
from hlanalysis.backtest.data._fastpath_core import _resample_reference_rows
from hlanalysis.backtest.runner.market_state import MarketState as SimMarketState
from hlanalysis.engine.market_state import MarketState as EngineMarketState
from hlanalysis.events import MarkEvent

_S = 1_000_000_000  # 1 second in ns
_DT_SECONDS = 5
_DT_NS = _DT_SECONDS * _S


def _mark(symbol: str, px: float, ts_ns: int) -> MarkEvent:
    return MarkEvent(
        venue="hyperliquid",
        product_type="perp",
        mechanism="clob",
        symbol=symbol,
        exchange_ts=ts_ns,
        local_recv_ts=ts_ns,
        mark_px=px,
    )


# Dense, irregular tick stream spanning many dt=5s buckets. Scalar ticks
# (high=low=close=price), exactly like a real bbo-mid / mark reference feed.
_TICKS: list[tuple[int, float]] = [
    (0 * _S, 100.0),
    (1 * _S, 100.6),
    (3 * _S, 99.4),
    (4 * _S, 101.2),
    (5 * _S, 101.0),
    (8 * _S, 103.5),
    (9 * _S, 98.7),
    (10 * _S, 100.2),
    (12 * _S, 102.9),
    (14 * _S, 97.1),
    (20 * _S, 101.4),
    (23 * _S, 100.0),
    (26 * _S, 104.0),
    (31 * _S, 99.9),
]


def _build_engine() -> EngineMarketState:
    ms = EngineMarketState()
    # Register the dt=5s cadence with a lookback wide enough to retain every bar.
    ms.set_reference_cadence("BTC", sampling_dt_seconds=_DT_SECONDS, lookback_seconds=10_000)
    for ts, px in _TICKS:
        ms.apply(_mark("BTC", px, ts))
    return ms


def _build_sim() -> SimMarketState:
    ms = SimMarketState()
    raw = [ReferenceEvent(ts, "BTC", px, px, px) for ts, px in _TICKS]
    bars = _resample_reference_rows(raw, resample_ns=_DT_NS)
    for bar in bars:
        ms.apply_reference(bar)
    return ms


def test_recent_returns_identical() -> None:
    engine = _build_engine()
    sim = _build_sim()

    # Wide enough n / lookback to capture every bar through both read APIs.
    now_ns = _TICKS[-1][0] + 1
    engine_rets = engine.recent_returns("BTC", n=1000, dt=_DT_SECONDS)
    sim_rets = sim.recent_returns(now_ns=now_ns, lookback_seconds=10_000)

    assert len(engine_rets) > 0
    np.testing.assert_allclose(np.asarray(engine_rets), sim_rets, rtol=0, atol=0)


def test_recent_hl_bars_identical() -> None:
    engine = _build_engine()
    sim = _build_sim()

    now_ns = _TICKS[-1][0] + 1
    engine_hl = engine.recent_hl_bars("BTC", n=1000, dt=_DT_SECONDS)
    sim_hl = sim.recent_hl_bars(now_ns=now_ns, lookback_seconds=10_000)

    engine_arr = np.asarray(engine_hl, dtype=np.float64)
    assert engine_arr.shape == sim_hl.shape
    np.testing.assert_allclose(engine_arr, sim_hl, rtol=0, atol=0)


def test_close_sequence_matches_resampled_bars() -> None:
    """Sanity anchor: the engine's per-bucket closes equal the resampled bars'
    closes one-for-one (both go through the shared canonical bucketer)."""
    engine = _build_engine()
    raw = [ReferenceEvent(ts, "BTC", px, px, px) for ts, px in _TICKS]
    bars = _resample_reference_rows(raw, resample_ns=_DT_NS)
    # Engine returns are close-to-close; reconstruct expected returns from bars.
    closes = [b.close for b in bars]
    expected = [np.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    engine_rets = engine.recent_returns("BTC", n=1000, dt=_DT_SECONDS)
    np.testing.assert_allclose(np.asarray(engine_rets), np.asarray(expected), rtol=0, atol=1e-12)
