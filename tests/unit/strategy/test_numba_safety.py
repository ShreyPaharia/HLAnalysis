from __future__ import annotations

import math

import pytest

from hlanalysis.strategy._numba.safety import safety_d_for_region_core
from hlanalysis.strategy.late_resolution import _safety_d_for_region as ref_fn


def _call_core(*, ref_price, lo, hi, sigma_window, mu, tte_min, drift_aware):
    has_lo = lo is not None
    has_hi = hi is not None
    return safety_d_for_region_core(
        float(ref_price),
        has_lo,
        float(lo) if has_lo else 0.0,
        has_hi,
        float(hi) if has_hi else 0.0,
        float(sigma_window),
        float(mu),
        float(tte_min),
        bool(drift_aware),
    )


@pytest.mark.parametrize("drift_aware", [False, True])
@pytest.mark.parametrize("mu", [0.0, 0.001, -0.001])
@pytest.mark.parametrize("tte_min", [0.0, 10.0])
def test_safety_d_lower_bounded_matches_reference(drift_aware, mu, tte_min):
    expected = ref_fn(
        ref_price=80_300.0, lo=80_000.0, hi=None,
        sigma_window=1.0, mu=mu, tte_min=tte_min, drift_aware=drift_aware,
    )
    got = _call_core(
        ref_price=80_300.0, lo=80_000.0, hi=None,
        sigma_window=1.0, mu=mu, tte_min=tte_min, drift_aware=drift_aware,
    )
    assert expected is not None
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=0.0)


@pytest.mark.parametrize("drift_aware", [False, True])
@pytest.mark.parametrize("mu", [0.0, 0.001, -0.001])
@pytest.mark.parametrize("tte_min", [0.0, 10.0])
def test_safety_d_upper_bounded_matches_reference(drift_aware, mu, tte_min):
    expected = ref_fn(
        ref_price=79_700.0, lo=None, hi=80_000.0,
        sigma_window=1.0, mu=mu, tte_min=tte_min, drift_aware=drift_aware,
    )
    got = _call_core(
        ref_price=79_700.0, lo=None, hi=80_000.0,
        sigma_window=1.0, mu=mu, tte_min=tte_min, drift_aware=drift_aware,
    )
    assert expected is not None
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=0.0)


@pytest.mark.parametrize("drift_aware", [False, True])
def test_safety_d_middle_bucket_drops_drift(drift_aware):
    expected = ref_fn(
        ref_price=79_500.0, lo=77_991.0, hi=81_174.0,
        sigma_window=1.0, mu=0.01, tte_min=100.0, drift_aware=drift_aware,
    )
    got = _call_core(
        ref_price=79_500.0, lo=77_991.0, hi=81_174.0,
        sigma_window=1.0, mu=0.01, tte_min=100.0, drift_aware=drift_aware,
    )
    assert expected is not None
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=0.0)


def test_safety_d_middle_bucket_adverse_negative():
    got = _call_core(
        ref_price=76_000.0, lo=77_991.0, hi=81_174.0,
        sigma_window=1.0, mu=0.0, tte_min=0.0, drift_aware=False,
    )
    assert got < 0.0


def test_safety_d_sigma_nonpositive_returns_nan():
    got = _call_core(
        ref_price=80_300.0, lo=80_000.0, hi=None,
        sigma_window=0.0, mu=0.0, tte_min=0.0, drift_aware=False,
    )
    assert math.isnan(got)


def test_safety_d_no_bounds_returns_nan():
    got = _call_core(
        ref_price=80_300.0, lo=None, hi=None,
        sigma_window=1.0, mu=0.0, tte_min=0.0, drift_aware=False,
    )
    assert math.isnan(got)


def test_safety_d_adverse_lower_bound_is_negative():
    got = _call_core(
        ref_price=79_700.0, lo=80_000.0, hi=None,
        sigma_window=1.0, mu=0.0, tte_min=0.0, drift_aware=False,
    )
    assert got < 0.0
