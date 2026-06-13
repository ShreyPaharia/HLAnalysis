"""Pure-math helpers and constants for the theta_harvester strategy family.

This module contains only stateless module-level functions and constants —
nothing that touches ``self`` or strategy configuration. All symbols are
re-exported from ``theta_harvester.py`` so any internal or external code that
imports from that module continues to work unchanged.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm  # type: ignore[import-untyped]

from .vol import ANNUAL_SECONDS

_ANNUAL_SECONDS = ANNUAL_SECONDS


_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _phi(d: float) -> float:
    """Standard-normal density at ``d``: φ(d) = exp(−d²/2) / √(2π)."""
    return _INV_SQRT_2PI * math.exp(-0.5 * d * d)


def _p_leg_win_prob_and_phi(
    *,
    reference_price: float,
    lo: float | None,
    hi: float | None,
    sigma: float,
    mu_eff: float,
    tau_yr: float,
) -> tuple[float, float] | None:
    """P(lo < S_T ≤ hi) under GBM (Itô-corrected) AND φ(d) at the nearest
    leg boundary in d-space.

    The φ(d) value is the per-share path-stdev of the held leg's fair value:
    std[Δp_t | hold for τ] ≈ φ(d) / S (S=1 for binary contracts), independent
    of τ. It is used as a path-variance / gamma risk premium term in the
    effective edge: effective_edge = edge − gamma_lambda · φ(d). At d=0 (S=K)
    φ peaks at 0.399; at |d|=2 it drops to 0.054.

    Returns None when the leg has no contiguous winning region (NO of a middle
    bucket — caller skips).
    """
    if lo is None and hi is None:
        return None
    sigma_sqrt_tau = sigma * math.sqrt(tau_yr)
    drift = (mu_eff - 0.5 * sigma * sigma) * tau_yr

    # Zero-guard: when σ=0 or τ=0, sigma_sqrt_tau=0 and _d() would divide by
    # zero. In the degenerate limit the GBM has no variance → outcome is
    # determined by whether reference_price is already inside the winning region.
    # phi(±∞) = 0 so the gamma proxy collapses to zero as well.
    if sigma_sqrt_tau <= 0.0:
        p_above_lo = 1.0 if (lo is None or reference_price > lo) else 0.0
        p_above_hi = 1.0 if (hi is not None and reference_price > hi) else 0.0
        p_win = max(0.0, p_above_lo - p_above_hi)
        return (p_win, 0.0)

    def _d(k: float) -> float:
        return (math.log(reference_price / k) + drift) / sigma_sqrt_tau

    d_lo = _d(lo) if lo is not None else None
    d_hi = _d(hi) if hi is not None else None

    p_above_lo = 1.0 if d_lo is None else float(norm.cdf(d_lo))
    p_above_hi = 0.0 if d_hi is None else float(norm.cdf(d_hi))
    p_win = max(0.0, p_above_lo - p_above_hi)

    # Gamma proxy at the closer-to-strike boundary — that's where path-variance
    # is highest. Conservative for middle buckets.
    if d_lo is not None and d_hi is not None:
        phi_d = max(_phi(d_lo), _phi(d_hi))
    elif d_lo is not None:
        phi_d = _phi(d_lo)
    else:
        phi_d = _phi(d_hi)  # type: ignore[arg-type]

    return (p_win, phi_d)


def _safety_d_for_region(
    *,
    reference_price: float,
    lo: float | None,
    hi: float | None,
    sigma: float,
    mu_eff: float,
    tau_yr: float,
) -> float | None:
    """Signed σ-normalized distance from ``reference_price`` to the NEAREST
    adverse boundary of the leg's winning region. Positive when BTC is safely
    inside the winning region; negative once already on the losing side.

    Uses the same Itô-corrected d-machinery as ``_p_leg_win_prob_and_phi``:
    ``d(k) = (ln(S/k) + (μ_eff − ½σ²)·τ) / (σ√τ)`` with drift baked in.
    Standalone helper (no dependency on v1's numba kernel) so v3.1 can diverge.
    Returns ``None`` when neither bound is known (e.g. NO leg of a middle bucket)
    or when σ·√τ is non-positive.
    """
    if lo is None and hi is None:
        return None
    if sigma <= 0.0 or tau_yr <= 0.0:
        return None
    sigma_sqrt_tau = sigma * math.sqrt(tau_yr)
    drift = (mu_eff - 0.5 * sigma * sigma) * tau_yr

    def _d(k: float) -> float:
        return (math.log(reference_price / k) + drift) / sigma_sqrt_tau

    if lo is not None and hi is not None:
        return min(_d(lo), -_d(hi))
    if lo is not None:
        return _d(lo)
    return -_d(hi)  # type: ignore[arg-type]


def _p_leg_win_prob(
    *,
    reference_price: float,
    lo: float | None,
    hi: float | None,
    sigma: float,
    mu_eff: float,
    tau_yr: float,
) -> float | None:
    """Back-compat thin wrapper around ``_p_leg_win_prob_and_phi`` for callers
    that only need the probability (tests, binary diagnostic edge_yes/edge_no).
    """
    res = _p_leg_win_prob_and_phi(
        reference_price=reference_price,
        lo=lo,
        hi=hi,
        sigma=sigma,
        mu_eff=mu_eff,
        tau_yr=tau_yr,
    )
    if res is None:
        return None
    return res[0]


def _jr_trust_weight(recent_returns: tuple[float, ...], lookback_min: int) -> float:
    """Jump fraction JR = max(0, (RV − BPV)/RV) over the last lookback_min returns.
    Returns the trust scalar `(1 − JR)` ∈ [0, 1]. 1.0 = pure continuous, 0.0 = pure jump."""
    if len(recent_returns) < max(lookback_min, 3):
        return 1.0
    arr = np.asarray(recent_returns[-lookback_min:], dtype=np.float64)
    rv = float(np.sum(arr * arr))
    if rv <= 0.0:
        return 1.0
    # Bipower variation: π/2 · Σ |r_i|·|r_{i-1}| (over consecutive pairs)
    abs_r = np.abs(arr)
    bpv = (np.pi / 2.0) * float(np.sum(abs_r[1:] * abs_r[:-1]))
    jr = max(0.0, min(1.0, (rv - bpv) / rv))
    return 1.0 - jr
