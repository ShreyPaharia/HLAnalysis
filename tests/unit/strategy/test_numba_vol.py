from __future__ import annotations

import math

import numpy as np
import pytest

from hlanalysis.strategy._numba.vol import (
    ewma_std,
    parkinson_sigma_window,
)
from hlanalysis.strategy.vol import (
    bipower_variation_sigma,
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


def _ref_bipower(returns) -> float:
    n = len(returns)
    if n < 2:
        return 0.0
    s = 0.0
    for i in range(n - 1):
        s += abs(returns[i]) * abs(returns[i + 1])
    var = (math.pi / 2.0) * s / (n - 1)
    return math.sqrt(max(var, 0.0))


@pytest.mark.parametrize("n", [2, 30, 1440])
def test_bipower_matches_reference(n: int) -> None:
    rng = np.random.default_rng(seed=4242 + n)
    arr = rng.normal(0, 0.001, n).astype(np.float64)
    expected = _ref_bipower(arr.tolist())
    got = bipower_variation_sigma(arr)
    if expected == 0.0:
        assert got == 0.0
    else:
        assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=0.0)


def test_bipower_handles_short_window() -> None:
    # 0 or 1 return: not enough adjacent products to form even one term.
    assert bipower_variation_sigma(np.array([], dtype=np.float64)) == 0.0
    assert bipower_variation_sigma(np.array([0.01], dtype=np.float64)) == 0.0


def test_bipower_zero_returns_yields_zero_sigma() -> None:
    assert bipower_variation_sigma(np.zeros(10, dtype=np.float64)) == 0.0


def test_bipower_is_jump_robust_vs_sample_std() -> None:
    # Calm series with one giant wick. sample_std (which squares the wick)
    # should explode; bipower variation (which multiplies adjacent |r|) should
    # stay close to the calm baseline because both neighbors of the wick are
    # tiny, so the contaminated products are |wick|·|small| — not |wick|².
    rng = np.random.default_rng(seed=2026)
    calm = rng.normal(0, 1e-4, 200).astype(np.float64)
    spiked = calm.copy()
    spiked[100] = 0.05  # one ~500σ wick in middle of window

    bv_calm = bipower_variation_sigma(calm)
    bv_spiked = bipower_variation_sigma(spiked)
    rv_calm = sample_std_returns(calm)
    rv_spiked = sample_std_returns(spiked)

    # Sample-std blows up by ~30×; bipower stays within ~3× of calm σ.
    # (Loose bounds — point is the relative immunity, not exact ratios.)
    assert rv_spiked / rv_calm > 20.0
    assert bv_spiked / bv_calm < 5.0


def test_bipower_normal_returns_consistent_with_sample_std() -> None:
    # In a no-jump iid-normal regime BV and RV should be close to each other
    # (asymptotically equal). With 5000 samples expect within ~10%.
    rng = np.random.default_rng(seed=7)
    arr = rng.normal(0, 0.001, 5000).astype(np.float64)
    bv = bipower_variation_sigma(arr)
    rv = sample_std_returns(arr)
    assert math.isclose(bv, rv, rel_tol=0.10)


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
