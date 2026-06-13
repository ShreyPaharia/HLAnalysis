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

from .vol import bipower_variation_sigma

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
_SDR_MR_THRESHOLD = _Z_RET_MR_THRESHOLD
_SDR_MOMENTUM_THRESHOLD = _Z_RET_MOMENTUM_THRESHOLD
_OU_Z_MR_THRESHOLD = 2.0


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
    indicator:      one of "z_ret", "rsi", "ma_sigma", "hurst_ou", "sdr", "ou_z".
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
    if indicator == "sdr":
        return _sdr(recent_returns, lookback_min, favorite_side)
    if indicator == "ou_z":
        return _ou_z(recent_returns, lookback_min, favorite_side)
    raise ValueError(f"Unknown indicator: {indicator!r}")


def _z_ret(
    recent_returns: tuple[float, ...],
    lookback_min: int,
    favorite_side: int,
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
    recent_returns: tuple[float, ...],
    lookback_min: int,
    favorite_side: int,
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
    recent_returns: tuple[float, ...],
    lookback_min: int,
    favorite_side: int,
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
    recent_returns: tuple[float, ...],
    lookback_min: int,
    favorite_side: int,
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


def _sdr(
    recent_returns: tuple[float, ...],
    lookback_min: int,
    favorite_side: int,
) -> tuple[float, str]:
    """Signed Drift Ratio — same as z_ret but uses √BPV as the denominator
    instead of rolling sample-std, making it robust to jump contamination.

    BPV (bipower variation) per bar: (π/2) · Σ|r_i||r_{i-1}| / (n-1).
    Denominator: √BPV · √lookback_min (matching z_ret's σ·√n structure).

    Falls back to z_ret-style sample-std when BPV ≤ 0 (pure-trend / flat).
    """
    if len(recent_returns) < max(lookback_min, 30):
        return 0.0, "neutral"
    arr = np.asarray(recent_returns, dtype=np.float64)
    window = arr[-lookback_min:]
    n = len(window)
    cum_ret = float(np.sum(window))

    # Bipower variation σ over the lookback window (jump-robust; shared kernel).
    sigma_bpv = bipower_variation_sigma(window)
    if sigma_bpv <= 0.0:
        return 0.0, "neutral"

    z = cum_ret / (sigma_bpv * math.sqrt(n))
    signed_z = z * favorite_side
    score = float(np.clip(signed_z / _SDR_MR_THRESHOLD, -1.0, 1.0))
    abs_z = abs(z)
    regime = "neutral"
    if abs_z >= _SDR_MR_THRESHOLD:
        last3 = window[-3:] if len(window) >= 3 else window
        if len(last3) >= 2 and float(np.sign(last3[-1])) != float(np.sign(cum_ret)):
            regime = "mr"
        else:
            regime = "momentum" if signed_z > 0 else "mr"
    elif abs_z >= _SDR_MOMENTUM_THRESHOLD:
        regime = "momentum" if signed_z > 0 else "mr"
    return score, regime


def _ou_z(
    recent_returns: tuple[float, ...],
    lookback_min: int,
    favorite_side: int,
) -> tuple[float, str]:
    """OU mean-reversion z-score.

    Fits AR(1) P_t = c + φ·P_{t-1} + ε_t to the log-price path (cumsum of
    returns). Scores the current price vs the equilibrium mean μ_eq = c/(1−φ)
    in units of the equilibrium std σ_eq = std(ε) / √(1−φ²).

    φ ∈ (0, 1) → mean-reverting; regime "mr" when |z_eq| ≥ 2.
    φ ≥ 0.99   → near-unit-root / trending; score 0, regime "momentum".
    φ ≤ 0      → explosive or anti-persistent; score 0, regime "neutral".
    """
    needed = lookback_min + 1  # need LB+1 bars for LB log-price diffs
    min_reliable = 30
    if len(recent_returns) < max(needed, min_reliable):
        return 0.0, "neutral"

    arr = np.asarray(recent_returns[-(needed):], dtype=np.float64)
    log_price = np.cumsum(arr)  # length = needed

    # OLS: P[1:] = c + φ·P[:-1]
    # numpy.polyfit(x, y, 1) → [slope, intercept] i.e. y = slope·x + intercept
    try:
        phi, c = np.polyfit(log_price[:-1], log_price[1:], 1)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0, "neutral"
    phi = float(phi)
    c = float(c)
    if not (np.isfinite(phi) and np.isfinite(c)):
        return 0.0, "neutral"

    if phi >= 0.99:
        # Near-unit-root or trending — not mean-reverting
        return 0.0, "momentum"
    if phi <= 0.0:
        # Anti-persistent or explosive
        return 0.0, "neutral"

    mu_eq = c / (1.0 - phi)
    residuals = log_price[1:] - (c + phi * log_price[:-1])
    sigma_e = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

    one_minus_phi2 = 1.0 - phi**2
    if one_minus_phi2 <= 0.0 or sigma_e <= 0.0:
        return 0.0, "neutral"

    sigma_eq = sigma_e / math.sqrt(one_minus_phi2)
    if sigma_eq <= 0.0:
        return 0.0, "neutral"

    p_last = float(log_price[-1])
    z_eq = (p_last - mu_eq) / sigma_eq
    # Divide by 2 to map |z_eq|=2 (MR threshold) onto |score|=1
    signed = z_eq * favorite_side / 2.0
    score = float(np.clip(signed, -1.0, 1.0))

    regime = "neutral"
    if abs(z_eq) >= _OU_Z_MR_THRESHOLD:
        regime = "mr"
    return score, regime
