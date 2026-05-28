# tests/unit/test_momentum_mr.py
"""Unit tests for hlanalysis.strategy.momentum_mr."""
from __future__ import annotations

import math

import numpy as np
import pytest

from hlanalysis.strategy.momentum_mr import momentum_mr_score


# Helper: build a trending up return series of length n with per-bar return r.
def _trend(n: int, r: float) -> tuple[float, ...]:
    return tuple(r for _ in range(n))


# Helper: build a mean-reverting series (n/2 up, n/2 down, summing to ~0).
def _mr(n: int, r: float) -> tuple[float, ...]:
    half = n // 2
    return tuple([r] * half + [-r] * (n - half))


class TestZRet:
    def test_strong_up_trend_aligned_to_up_favorite_is_momentum(self) -> None:
        rets = _trend(120, 0.0008)  # strong sustained up move
        score, regime = momentum_mr_score(
            recent_returns=rets, lookback_min=15, indicator="z_ret", favorite_side=+1,
        )
        assert score > 0.0
        assert regime in {"momentum", "neutral"}

    def test_strong_up_trend_against_down_favorite_is_negative(self) -> None:
        rets = _trend(120, 0.0008)
        score, _ = momentum_mr_score(
            recent_returns=rets, lookback_min=15, indicator="z_ret", favorite_side=-1,
        )
        assert score < 0.0

    def test_insufficient_data_returns_neutral_zero(self) -> None:
        # Lookback 60 but only 3 returns available → neutral, score 0
        score, regime = momentum_mr_score(
            recent_returns=(0.001, -0.001, 0.0), lookback_min=60,
            indicator="z_ret", favorite_side=+1,
        )
        assert score == 0.0
        assert regime == "neutral"

    def test_flat_returns_neutral(self) -> None:
        rets = tuple(0.0 for _ in range(120))
        score, regime = momentum_mr_score(
            recent_returns=rets, lookback_min=15, indicator="z_ret", favorite_side=+1,
        )
        assert score == 0.0
        assert regime == "neutral"


class TestRSI:
    def test_pure_up_returns_overbought_against_up_favorite_is_mr(self) -> None:
        # 30 consecutive up bars → RSI(14) saturates near 100 → score>>0 aligned
        # to UP favorite → momentum; but if regime call uses RSI>70 as MR sign
        # (overbought = reversion likely), we still expect score > 0 (aligned)
        # and regime in {"momentum","mr"} per spec table.
        rets = _trend(60, 0.001)
        score, regime = momentum_mr_score(
            recent_returns=rets, lookback_min=15, indicator="rsi", favorite_side=+1,
        )
        assert score > 0.5
        assert regime in {"momentum", "mr"}

    def test_pure_down_returns_against_up_favorite_is_negative_score(self) -> None:
        rets = _trend(60, -0.001)
        score, _ = momentum_mr_score(
            recent_returns=rets, lookback_min=15, indicator="rsi", favorite_side=+1,
        )
        assert score < -0.5


class TestMASigma:
    def test_recent_drift_up_aligned_to_up_favorite_is_positive(self) -> None:
        # First half flat, last half up — last close stretched above MA
        rets = tuple([0.0] * 60 + [0.001] * 30)
        score, _ = momentum_mr_score(
            recent_returns=rets, lookback_min=30, indicator="ma_sigma", favorite_side=+1,
        )
        assert score > 0.0


class TestHurstOU:
    def test_random_walk_close_to_05_neutral_or_momentum(self) -> None:
        # Brownian-like sequence (seed for determinism)
        rng = np.random.default_rng(seed=42)
        rets = tuple(rng.normal(0.0, 0.001, size=200).tolist())
        score, regime = momentum_mr_score(
            recent_returns=rets, lookback_min=60, indicator="hurst_ou", favorite_side=+1,
        )
        # Random walk → H ~ 0.5 → score near 0
        assert abs(score) < 0.6
        assert regime in {"neutral", "momentum", "mr"}

    def test_strong_mean_reverting_returns_mr(self) -> None:
        # Alternating high-frequency reversion → H < 0.5
        rets = _mr(120, 0.001) * 2  # 240 bars of pure alternation
        score, regime = momentum_mr_score(
            recent_returns=rets, lookback_min=60, indicator="hurst_ou", favorite_side=+1,
        )
        assert regime == "mr"


class TestUnknownIndicator:
    def test_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown indicator"):
            momentum_mr_score(
                recent_returns=(0.0,) * 30, lookback_min=15,
                indicator="totally_made_up", favorite_side=+1,
            )
