"""10k-tick replay benchmark: JIT'd strategy + ring-buffer state vs the
pre-JIT pure-Python baseline. Asserts the spec's ≥5× speedup.

The pre-JIT baseline is built in-test by subclassing LateResolutionStrategy
to override the σ helpers with the original list-based implementations and
by holding a deque-backed MarketState mirror of the legacy logic. This
makes the comparison self-contained — no need to wire up an "old" tree.
"""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from hlanalysis.sim.data.binance_klines import Kline
from hlanalysis.sim.market_state import SimMarketState
from hlanalysis.strategy._numba.vol import (
    ewma_std,
    parkinson_sigma_window,
    sample_std_returns,
)
from hlanalysis.strategy.late_resolution import (
    LateResolutionConfig,
    LateResolutionStrategy,
)
from hlanalysis.strategy.types import BookState, QuestionView

N_TICKS = 10_000
NS_PER_MIN = 60_000_000_000


def _build_klines(n: int) -> list[Kline]:
    out: list[Kline] = []
    price = 80_000.0
    for i in range(n):
        price *= math.exp(0.0001 * math.sin(i / 50.0))
        out.append(Kline(ts_ns=i * NS_PER_MIN, open=price,
                         high=price * 1.001, low=price * 0.999,
                         close=price, volume=1000.0))
    return out


def _cfg() -> LateResolutionConfig:
    return LateResolutionConfig(
        tte_min_seconds=60, tte_max_seconds=86_400,
        price_extreme_threshold=0.90, distance_from_strike_usd_min=0.0,
        vol_max=2.0, max_position_usd=100.0, stop_loss_pct=10.0,
        max_strike_distance_pct=50.0, min_recent_volume_usd=0.0,
        stale_data_halt_seconds=60_000, price_extreme_max=0.995,
        min_safety_d=1.0, vol_lookback_seconds=1800,
        exit_safety_d=0.5, exit_safety_d_5m=0.3,
        exit_vol_lookback_5m_seconds=300, vol_ewma_lambda=0.94,
        vol_estimator="parkinson", drift_aware_d=True,
    )


def _q(expiry_ns: int) -> QuestionView:
    return QuestionView(
        question_idx=1, yes_symbol="@30", no_symbol="@31",
        strike=80_000.0, expiry_ns=expiry_ns,
        underlying="BTC", klass="priceBinary", period="1d",
        leg_symbols=("@30", "@31"),
    )


_PARK_K = 1.0 / (4.0 * math.log(2.0))


class _LegacyMarketState:
    """Pre-JIT baseline: deque + per-tick list-comp rebuild (matches the
    old SimMarketState.recent_returns logic byte-for-byte)."""

    def __init__(self) -> None:
        self._k: deque[tuple[int, float, float, float]] = deque()

    def apply_kline(self, k: Kline) -> None:
        self._k.append((k.ts_ns, k.high, k.low, k.close))

    def recent_returns(self, *, now_ns: int, lookback_seconds: int) -> tuple[float, ...]:
        cutoff = now_ns - lookback_seconds * 1_000_000_000
        prices = [(t, c) for (t, _h, _l, c) in self._k if cutoff <= t <= now_ns]
        if len(prices) < 2:
            return ()
        out = []
        for i in range(1, len(prices)):
            p_prev, p_now = prices[i - 1][1], prices[i][1]
            if p_prev > 0 and p_now > 0:
                out.append(math.log(p_now / p_prev))
        return tuple(out)

    def recent_hl_bars(self, *, now_ns: int, lookback_seconds: int):
        cutoff = now_ns - lookback_seconds * 1_000_000_000
        return tuple(
            (h, l) for (t, h, l, _c) in self._k
            if cutoff <= t <= now_ns and h > 0 and l > 0
        )


class _LegacySigmaStrategy(LateResolutionStrategy):
    """Reverts σ to pure Python by overriding the JIT'd methods.

    safety_d_for_region is left on the JIT path because the profiler showed
    it at <0.01% — the meaningful difference here is the σ inner loop and
    the state rebuild, which together account for ~96% of cumtime.
    """

    @staticmethod
    def _ewma_std(returns, lam):
        var = float(returns[0]) ** 2
        for r in returns[1:]:
            var = lam * var + (1.0 - lam) * float(r) ** 2
        return math.sqrt(var)

    def _sigma_stdev(self, returns_window):
        if self.cfg.vol_ewma_lambda > 0.0:
            return self._ewma_std(returns_window, self.cfg.vol_ewma_lambda)
        return float(np.std(returns_window, ddof=1))

    def _sigma_parkinson(self, hl_window):
        pb = []
        for h, l in hl_window:
            if h > 0 and l > 0 and h >= l:
                pb.append(_PARK_K * math.log(h / l) ** 2)
        if not pb:
            return 0.0
        if self.cfg.vol_ewma_lambda > 0.0:
            var = pb[0]
            for v in pb[1:]:
                var = self.cfg.vol_ewma_lambda * var + (1.0 - self.cfg.vol_ewma_lambda) * v
            return math.sqrt(max(var, 0.0))
        return math.sqrt(sum(pb) / len(pb))


def _replay(strat_cls, state) -> float:
    cfg = _cfg()
    strat = strat_cls(cfg)
    klines = _build_klines(N_TICKS)
    q = _q(expiry_ns=(N_TICKS + 60) * NS_PER_MIN)
    yes = BookState(symbol="@30", bid_px=0.93, bid_sz=100, ask_px=0.94, ask_sz=100,
                    last_trade_ts_ns=0, last_l2_ts_ns=0)
    no_ = BookState(symbol="@31", bid_px=0.05, bid_sz=100, ask_px=0.06, ask_sz=100,
                    last_trade_ts_ns=0, last_l2_ts_ns=0)

    t0 = time.perf_counter()
    for k in klines:
        state.apply_kline(k)
        now = k.ts_ns
        books = {
            "@30": BookState(symbol="@30", bid_px=yes.bid_px, bid_sz=yes.bid_sz,
                            ask_px=yes.ask_px, ask_sz=yes.ask_sz,
                            last_trade_ts_ns=now, last_l2_ts_ns=now),
            "@31": BookState(symbol="@31", bid_px=no_.bid_px, bid_sz=no_.bid_sz,
                            ask_px=no_.ask_px, ask_sz=no_.ask_sz,
                            last_trade_ts_ns=now, last_l2_ts_ns=now),
        }
        rets = state.recent_returns(now_ns=now, lookback_seconds=86_400)
        hls = state.recent_hl_bars(now_ns=now, lookback_seconds=86_400)
        strat.evaluate(question=q, books=books, reference_price=k.close,
                       recent_returns=rets, recent_volume_usd=1000.0,
                       position=None, now_ns=now, recent_hl_bars=hls)
    return time.perf_counter() - t0


def _warmup_jits() -> None:
    arr = np.array([0.001, -0.001, 0.002], dtype=np.float64)
    ewma_std(arr, 0.94)
    sample_std_returns(arr)
    parkinson_sigma_window(
        np.array([1.001, 1.002], dtype=np.float64),
        np.array([0.999, 0.998], dtype=np.float64),
        0.94,
    )


def test_jit_path_at_least_5x_faster_than_pure_python(capsys):
    _warmup_jits()
    # Run baseline (legacy state + legacy σ) and JIT (ring buffer + JIT σ).
    t_legacy = _replay(_LegacySigmaStrategy, _LegacyMarketState())
    t_jit = _replay(LateResolutionStrategy, SimMarketState())
    speedup = t_legacy / t_jit
    with capsys.disabled():
        print(f"\n10k-tick replay  legacy={t_legacy:.3f}s  "
              f"jit={t_jit:.3f}s  speedup={speedup:.2f}×")
    assert speedup >= 5.0, f"expected ≥5× speedup, got {speedup:.2f}×"


def test_jit_calls_are_cached_after_first_compile():
    """Warm dispatch: 1000 calls of an already-compiled JIT function must
    complete in well under a second. If cache=True were silently disabled
    (or AOT compile broke), each call would re-enter the dispatcher's slow
    path and this would take many seconds.
    """
    _warmup_jits()
    arr = np.array([0.001, -0.001, 0.002, 0.003, -0.002], dtype=np.float64)
    t0 = time.perf_counter()
    for _ in range(1000):
        ewma_std(arr, 0.94)
    t_warm = time.perf_counter() - t0
    assert t_warm < 0.5, (
        f"warm 1000 calls of ewma_std took {t_warm:.3f}s — JIT cache likely missed"
    )
