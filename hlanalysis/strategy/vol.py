"""Shared realized-volatility estimators and annualization.

This module is the single home for:

* ``ANNUAL_SECONDS`` — the seconds-per-year constant used to annualize per-bar
  σ and per-bar drift (previously redefined verbatim in ``model_edge.py``,
  ``theta_harvester.py`` and inlined as ``365.25 * 86400.0`` in
  ``delta_hedged.py``).
* The two non-recursive per-sample estimators ``sample_std_returns`` and
  ``bipower_variation_sigma`` — plain numpy one-liners. They used to live as
  ``numba.njit`` kernels in ``_numba/vol.py`` with NaN/short-window boundary
  plumbing; numpy does the same work without the JIT overhead. The recursive
  estimators (``ewma_std``, ``parkinson_sigma_window``) stay JIT'd in
  ``_numba/vol.py`` because their scalar recurrences don't vectorize cleanly.
* ``annualized_sigma`` — the annualize-and-clip pipeline shared by the v2
  (model_edge) and v3.1 (theta/nba) families.

NOTE ON DIVERGENCE: v1 (late_resolution) deliberately does NOT annualize — it
works in per-bar σ units and additionally supports Parkinson/EWMA estimators.
It composes ``sample_std_returns`` directly and is intentionally *not* routed
through ``annualized_sigma``. Do not unify the two conventions; the tuned
params do not transfer between them.
"""

from __future__ import annotations

import math

import numpy as np

ANNUAL_SECONDS = 365.25 * 86400.0


def sample_std_returns(returns: np.ndarray) -> float:
    """Sample standard deviation (ddof=1) of ``returns``.

    Returns 0.0 for fewer than two samples (matching the prior numba kernel,
    whose callers always guaranteed n >= 2 before invoking it).
    """
    if returns.shape[0] < 2:
        return 0.0
    return float(np.std(returns, ddof=1))


def bipower_variation_sigma(returns: np.ndarray) -> float:
    """Jump-robust per-sample σ via Barndorff-Nielsen & Shephard bipower variation.

    σ²_BV = (π/2) · (1/(n−1)) · Σ_{i=0..n−2} |r_i|·|r_{i+1}|

    A single large |r_k| only contributes to two consecutive products, so wicks
    do not inflate σ_BV the way they inflate sample-stdev (which squares the
    wick). In the no-jump limit BV converges to the same per-sample variance as
    ``sample_std_returns`` (assuming zero-mean returns). Returns 0.0 for fewer
    than two samples or when the variation is non-positive.
    """
    n = returns.shape[0]
    if n < 2:
        return 0.0
    abs_r = np.abs(returns)
    var = (math.pi / 2.0) * float(np.sum(abs_r[1:] * abs_r[:-1])) / (n - 1)
    if var <= 0.0:
        return 0.0
    return math.sqrt(var)


def annualized_sigma(
    returns: np.ndarray,
    *,
    dt_seconds: float,
    estimator: str,
    clip_min: float,
    clip_max: float,
) -> float:
    """Annualized, clipped realized σ for the v2/v3.1 strategy family.

    ``estimator`` selects the per-sample estimator:
      * ``"sample_std"`` — close-to-close sample stdev (ddof=1).
      * ``"bipower"``    — jump-robust bipower variation σ.

    The per-sample σ is annualized by ``sqrt(ANNUAL_SECONDS / dt_seconds)`` and
    clipped to ``[clip_min, clip_max]``. Callers decide what a non-positive
    result means (e.g. theta maps σ <= 0 to ``None``).
    """
    if estimator == "bipower":
        raw = bipower_variation_sigma(returns)
    elif estimator == "sample_std":
        raw = sample_std_returns(returns)
    else:
        raise ValueError(f"Unknown vol_estimator: {estimator!r}")
    ann = math.sqrt(ANNUAL_SECONDS / float(dt_seconds))
    return max(clip_min, min(clip_max, raw * ann))
