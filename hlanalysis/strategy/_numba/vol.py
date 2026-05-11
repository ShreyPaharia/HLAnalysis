"""JIT'd σ estimators used by LateResolutionStrategy and ModelEdgeStrategy.

Reference Python implementations live in
``hlanalysis/strategy/late_resolution.py::_ewma_std`` and
``_parkinson_per_bar_var`` / ``_sigma_parkinson``. The functions here must
match those references to 1e-12 relative (asserted in
``tests/unit/strategy/test_numba_vol.py``).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

_PARK_K = 1.0 / (4.0 * math.log(2.0))


@njit(cache=True, fastmath=False)
def ewma_std(returns: np.ndarray, lam: float) -> float:
    """σ from recursive EWMA: σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t, seeded σ²_0 = r²_0.

    Caller must ensure ``returns`` has at least one element.
    """
    n = returns.shape[0]
    var = returns[0] * returns[0]
    for i in range(1, n):
        r = returns[i]
        var = lam * var + (1.0 - lam) * r * r
    return math.sqrt(var)


@njit(cache=True, fastmath=False)
def sample_std_returns(returns: np.ndarray) -> float:
    """Sample stdev (ddof=1). Matches ``np.std(arr, ddof=1)`` at 1e-12.

    Caller must ensure ``returns`` has at least two elements.
    """
    n = returns.shape[0]
    s = 0.0
    for i in range(n):
        s += returns[i]
    mean = s / n
    ss = 0.0
    for i in range(n):
        d = returns[i] - mean
        ss += d * d
    return math.sqrt(ss / (n - 1))


@njit(cache=True, fastmath=False)
def parkinson_sigma_window(
    highs: np.ndarray, lows: np.ndarray, lam: float
) -> float:
    """Window-level Parkinson σ.

    Per-bar variance: σ²_i = (ln(H_i/L_i))² / (4 ln 2) for bars with H > 0,
    L > 0, H >= L. Skips degenerate bars (matches the legacy filter).

    Aggregation: if ``lam > 0``, recursive EWMA across kept per-bar variances
    seeded with the first kept σ²; else arithmetic mean. Returns 0.0 when no
    bars are valid.
    """
    n = highs.shape[0]
    if lam > 0.0:
        var = -1.0
        for i in range(n):
            h = highs[i]
            l = lows[i]
            if h > 0.0 and l > 0.0 and h >= l:
                ln_hl = math.log(h / l)
                pb = _PARK_K * ln_hl * ln_hl
                if var < 0.0:
                    var = pb
                else:
                    var = lam * var + (1.0 - lam) * pb
        if var < 0.0:
            return 0.0
        if var < 0.0:
            var = 0.0
        return math.sqrt(var)
    total = 0.0
    count = 0
    for i in range(n):
        h = highs[i]
        l = lows[i]
        if h > 0.0 and l > 0.0 and h >= l:
            ln_hl = math.log(h / l)
            total += _PARK_K * ln_hl * ln_hl
            count += 1
    if count == 0:
        return 0.0
    return math.sqrt(total / count)
