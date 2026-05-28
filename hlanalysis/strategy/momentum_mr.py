# hlanalysis/strategy/momentum_mr.py
"""Momentum / mean-reversion indicators on BTC 1m log-return series.

The `recent_returns` tuple passed to `ThetaHarvesterStrategy.evaluate` is a
chronological tuple of 60-second log-returns. `lookback_min` selects the last
N elements (one per minute under the default vol_sampling_dt_seconds=60).

Score convention: positive ⇒ underlying trend ALIGNED with `favorite_side`
(continuation likely), negative ⇒ against the favorite (reversion likely).
Score is clipped to [-1.0, +1.0].

Regime:
  - "momentum": trend aligned, |score| not extreme
  - "mr":       mean-reversion signal (overextended) — caller may gate against
  - "neutral":  insufficient signal / inadequate data
"""
from __future__ import annotations

import math

import numpy as np


# Reference window for z-score variance — fixed at 240 minutes (4h) regardless
# of indicator lookback. Long enough to be stable, short enough to track
# regime shifts within a 24h PM market.
_Z_RET_REF_WINDOW_MIN = 240
_Z_RET_MR_THRESHOLD = 2.5
_Z_RET_MOMENTUM_THRESHOLD = 1.0
_MA_SIGMA_MR_THRESHOLD = 2.0
_MA_SIGMA_MOMENTUM_THRESHOLD = 0.5
_HURST_MR_BAND = 0.45
_HURST_MOMENTUM_BAND = 0.55


def momentum_mr_score(
    *,
    recent_returns: tuple[float, ...],
    lookback_min: int,
    indicator: str,
    favorite_side: int,
) -> tuple[float, str]:
    """Return (score in [-1, +1], regime).

    Parameters
    ----------
    recent_returns: chronological tuple of 60s log-returns.
    lookback_min:   number of trailing 1-minute bars to read.
    indicator:      one of "z_ret", "rsi", "ma_sigma", "hurst_ou".
    favorite_side:  +1 if current favorite is UP (yes), -1 if DOWN (no).

    Score is always signed relative to favorite_side: positive ⇒ underlying
    moving WITH the favorite; negative ⇒ AGAINST.
    """
    if favorite_side not in (-1, +1):
        raise ValueError(f"favorite_side must be ±1, got {favorite_side}")
    if lookback_min <= 0:
        raise ValueError(f"lookback_min must be positive, got {lookback_min}")

    if indicator == "z_ret":
        return _z_ret(recent_returns, lookback_min, favorite_side)
    if indicator == "rsi":
        return _rsi(recent_returns, lookback_min, favorite_side)
    if indicator == "ma_sigma":
        return _ma_sigma(recent_returns, lookback_min, favorite_side)
    if indicator == "hurst_ou":
        return _hurst_ou(recent_returns, lookback_min, favorite_side)
    raise ValueError(f"Unknown indicator: {indicator!r}")


def _z_ret(
    recent_returns: tuple[float, ...], lookback_min: int, favorite_side: int,
) -> tuple[float, str]:
    """z-score of the cumulative return over lookback_min vs a 4h reference σ."""
    if len(recent_returns) < max(lookback_min, 30):
        return 0.0, "neutral"
    arr = np.asarray(recent_returns, dtype=np.float64)
    ref_window = arr[-_Z_RET_REF_WINDOW_MIN:] if len(arr) >= _Z_RET_REF_WINDOW_MIN else arr
    sigma_per_bar = float(np.std(ref_window, ddof=1)) if len(ref_window) > 1 else 0.0
    if sigma_per_bar <= 0.0:
        return 0.0, "neutral"
    window = arr[-lookback_min:]
    cum_ret = float(np.sum(window))
    z = cum_ret / (sigma_per_bar * math.sqrt(lookback_min))
    signed_z = z * favorite_side
    score = float(np.clip(signed_z / _Z_RET_MR_THRESHOLD, -1.0, 1.0))
    abs_z = abs(z)
    regime = "neutral"
    if abs_z >= _Z_RET_MR_THRESHOLD:
        # Check for sign-flip in last 3 bars → MR signal
        last3 = window[-3:] if len(window) >= 3 else window
        if len(last3) >= 2 and float(np.sign(last3[-1])) != float(np.sign(cum_ret)):
            regime = "mr"
        else:
            regime = "momentum" if signed_z > 0 else "mr"
    elif abs_z >= _Z_RET_MOMENTUM_THRESHOLD:
        regime = "momentum" if signed_z > 0 else "mr"
    return score, regime


def _rsi(
    recent_returns: tuple[float, ...], lookback_min: int, favorite_side: int,
) -> tuple[float, str]:
    """Wilder RSI(14) computed over the last `lookback_min + 14` bars (need 14
    to seed the smoothing). Returns score = (RSI − 50) / 50 signed to favorite.
    """
    period = 14
    needed = lookback_min + period
    if len(recent_returns) < needed:
        return 0.0, "neutral"
    arr = np.asarray(recent_returns[-needed:], dtype=np.float64)
    # Convert log-returns to "price changes" — for RSI the sign is what matters,
    # so use the returns directly as up/down deltas.
    gains = np.where(arr > 0, arr, 0.0)
    losses = np.where(arr < 0, -arr, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, len(arr)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0.0:
        rsi = 100.0 if avg_gain > 0 else 50.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
    score_raw = (rsi - 50.0) / 50.0
    signed = score_raw * favorite_side
    score = float(np.clip(signed, -1.0, 1.0))
    regime = "neutral"
    if rsi > 70.0 or rsi < 30.0:
        # Overbought/oversold: if aligned with favorite, momentum extreme; if
        # against favorite, MR signal (overstretched the wrong way).
        regime = "momentum" if signed > 0 else "mr"
    return score, regime


def _ma_sigma(
    recent_returns: tuple[float, ...], lookback_min: int, favorite_side: int,
) -> tuple[float, str]:
    """Distance of last cum-return from rolling-mean cum-return in σ units.

    Build a synthetic log-price path from returns (start at 0), take MA and σ
    over the last `lookback_min` bars, return (last − MA) / σ.
    """
    if len(recent_returns) < lookback_min:
        return 0.0, "neutral"
    arr = np.asarray(recent_returns[-lookback_min:], dtype=np.float64)
    log_price = np.cumsum(arr)  # relative path starting at 0
    ma = float(np.mean(log_price))
    sd = float(np.std(log_price, ddof=1)) if len(log_price) > 1 else 0.0
    if sd <= 0.0:
        return 0.0, "neutral"
    d = (float(log_price[-1]) - ma) / sd
    signed = d * favorite_side
    score = float(np.clip(signed / _MA_SIGMA_MR_THRESHOLD, -1.0, 1.0))
    regime = "neutral"
    if abs(d) >= _MA_SIGMA_MR_THRESHOLD:
        regime = "mr" if signed < 0 else "momentum"
    elif abs(d) >= _MA_SIGMA_MOMENTUM_THRESHOLD and signed > 0:
        regime = "momentum"
    return score, regime


def _hurst_ou(
    recent_returns: tuple[float, ...], lookback_min: int, favorite_side: int,
) -> tuple[float, str]:
    """Estimate a pseudo-Hurst exponent via the lag-k autocorrelation of
    the return series (full available history, lag = lookback_min).

    H ∈ (0,1). H < 0.5 ⇒ mean-reverting; H > 0.5 ⇒ trending. The mapping
    is: H = clip(0.5 + 0.4 * acf(lookback_min), 0.1, 0.9). A block-reverting
    series (e.g. 60 bars up then 60 bars down) produces acf(60) ≈ −1 → H → 0.1
    (MR). A random walk produces acf ≈ 0 → H ≈ 0.5 (neutral).

    Score: `direction * 2*(H − 0.5)`, where direction = sign(cum_ret *
    favorite_side) over the lookback window.
    """
    # Need enough data for both the lookback window and a reliable lag-k ACF
    if len(recent_returns) < max(lookback_min, 60):
        return 0.0, "neutral"
    arr_full = np.asarray(recent_returns, dtype=np.float64)
    arr_lookback = arr_full[-lookback_min:]
    n = len(arr_full)
    lag = lookback_min
    # Require at least lookback + lookback//2 observations for ACF at lag k
    if n < lag + max(lag // 2, 8):
        return 0.0, "neutral"
    # Autocorrelation at lag=lookback_min over the full available series
    acf = float(np.corrcoef(arr_full[:-lag], arr_full[lag:])[0, 1])
    if not np.isfinite(acf):
        return 0.0, "neutral"
    # Map ACF → pseudo-Hurst: acf=−1 → H=0.1, acf=0 → H=0.5, acf=+1 → H=0.9
    H = float(np.clip(0.5 + 0.4 * acf, 0.1, 0.9))
    # Direction of the price path over lookback window × favorite_side
    cum = float(np.sum(arr_lookback))
    direction = 1.0 if cum * favorite_side > 0 else -1.0
    score = float(np.clip(direction * 2.0 * (H - 0.5), -1.0, 1.0))
    regime = "neutral"
    if H < _HURST_MR_BAND:
        regime = "mr"
    elif H > _HURST_MOMENTUM_BAND:
        regime = "momentum"
    return score, regime
