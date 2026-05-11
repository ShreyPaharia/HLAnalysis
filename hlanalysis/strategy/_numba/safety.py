"""JIT'd safety_d_for_region core.

Encodes None bounds via has_lo/has_hi flags and returns NaN for the "no
gate" cases. The strategy-side wrapper maps NaN → None to preserve the
public ``Optional[float]`` contract.

Reference: ``hlanalysis.strategy.late_resolution._safety_d_for_region``.
"""
from __future__ import annotations

import math

from numba import njit


@njit(cache=True, fastmath=False)
def safety_d_for_region_core(
    ref_price: float,
    has_lo: bool,
    lo: float,
    has_hi: bool,
    hi: float,
    sigma_window: float,
    mu: float,
    tte_min: float,
    drift_aware: bool,
) -> float:
    """Signed safety distance from ``ref_price`` to the nearer adverse boundary,
    in σ-window units. Returns NaN when no gate can be computed."""
    if sigma_window <= 0.0:
        return math.nan
    if has_lo and has_hi:
        d_lo = math.log(ref_price / lo)
        d_hi = math.log(hi / ref_price)
        d_unscaled = d_lo if d_lo < d_hi else d_hi
        return d_unscaled / sigma_window
    if has_lo and not has_hi:
        d_unscaled = math.log(ref_price / lo)
        if drift_aware:
            d_unscaled += mu * tte_min
        return d_unscaled / sigma_window
    if (not has_lo) and has_hi:
        d_unscaled = math.log(hi / ref_price)
        if drift_aware:
            d_unscaled -= mu * tte_min
        return d_unscaled / sigma_window
    return math.nan
