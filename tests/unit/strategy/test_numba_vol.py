from __future__ import annotations

import math

import numpy as np
import pytest

from hlanalysis.strategy._numba.vol import (
    ewma_std,
    parkinson_sigma_window,
    sample_std_returns,
)


def _ref_ewma_std(returns, lam: float) -> float:
    var = float(returns[0]) ** 2
    for r in returns[1:]:
        rf = float(r)
        var = lam * var + (1.0 - lam) * rf * rf
    return math.sqrt(var)


_PARK_K = 1.0 / (4.0 * math.log(2.0))


def _ref_parkinson_per_bar(hl):
    out = []
    for h, l in hl:
        if h > 0 and l > 0 and h >= l:
            out.append(_PARK_K * math.log(h / l) ** 2)
    return out


def _ref_parkinson_window(hl, lam: float) -> float:
    pb = _ref_parkinson_per_bar(hl)
    if not pb:
        return 0.0
    if lam > 0.0:
        var = pb[0]
        for v in pb[1:]:
            var = lam * var + (1.0 - lam) * v
        return math.sqrt(max(var, 0.0))
    return math.sqrt(sum(pb) / len(pb))


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.94, 0.99])
@pytest.mark.parametrize("n", [1, 2, 30, 1440])
def test_ewma_std_matches_reference(lam: float, n: int) -> None:
    rng = np.random.default_rng(seed=n * 100 + int(lam * 100))
    rets = rng.normal(0, 0.001, n).astype(np.float64)
    expected = _ref_ewma_std(rets.tolist(), lam)
    got = ewma_std(rets, lam)
    if expected == 0.0:
        assert got == 0.0
    else:
        assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=0.0)


def test_ewma_std_single_element() -> None:
    arr = np.array([0.01], dtype=np.float64)
    assert math.isclose(ewma_std(arr, 0.94), 0.01, rel_tol=1e-12, abs_tol=0.0)


def test_ewma_std_zero_lambda_collapses_to_last_squared_return() -> None:
    arr = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    # λ=0 → σ²_t = r²_t → σ = |r_last|
    got = ewma_std(arr, 0.0)
    assert math.isclose(got, 0.03, rel_tol=1e-12)


@pytest.mark.parametrize("n", [2, 30, 1440])
def test_sample_std_returns_matches_numpy(n: int) -> None:
    rng = np.random.default_rng(seed=n)
    arr = rng.normal(0, 0.001, n).astype(np.float64)
    expected = float(np.std(arr, ddof=1))
    got = sample_std_returns(arr)
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=0.0)


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.94])
@pytest.mark.parametrize("n", [1, 5, 1440])
def test_parkinson_window_matches_reference(lam: float, n: int) -> None:
    rng = np.random.default_rng(seed=n + int(lam * 1000))
    closes = 80_000.0 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
    h = closes * (1 + np.abs(rng.normal(0, 0.0005, n)))
    l = closes * (1 - np.abs(rng.normal(0, 0.0005, n)))
    hl = list(zip(h.tolist(), l.tolist()))
    expected = _ref_parkinson_window(hl, lam)
    got = parkinson_sigma_window(
        np.asarray(h, dtype=np.float64),
        np.asarray(l, dtype=np.float64),
        lam,
    )
    if expected == 0.0:
        assert got == 0.0
    else:
        assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=0.0)


def test_parkinson_window_returns_zero_when_all_degenerate() -> None:
    h = np.array([100.0, 100.0], dtype=np.float64)
    l = np.array([100.0, 100.0], dtype=np.float64)
    assert parkinson_sigma_window(h, l, 0.0) == 0.0
    assert parkinson_sigma_window(h, l, 0.94) == 0.0


def test_parkinson_window_skips_invalid_bars() -> None:
    # Mix valid + invalid bars; result must match reference filter (h>0,l>0,h>=l).
    h = np.array([1.001, 0.0, 1.002, 1.003], dtype=np.float64)
    l = np.array([0.999, 1.0, 0.0, 1.001], dtype=np.float64)
    hl = list(zip(h.tolist(), l.tolist()))
    expected = _ref_parkinson_window(hl, 0.0)
    got = parkinson_sigma_window(h, l, 0.0)
    if expected == 0.0:
        assert got == 0.0
    else:
        assert math.isclose(got, expected, rel_tol=1e-12)
