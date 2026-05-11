"""Numba-accelerated hot-path helpers for the strategy package.

All public callables are `numba.njit(cache=True)` and return scalars or
contiguous float64 arrays. Wrappers in the strategy modules handle Optional
results and tuple→ndarray conversion at the boundary.
"""
from __future__ import annotations
