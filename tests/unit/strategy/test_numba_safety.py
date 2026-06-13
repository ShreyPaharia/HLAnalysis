"""Tests for ``late_resolution._safety_d_for_region``.

Formerly this exercised the ``_numba.safety.safety_d_for_region_core`` kernel
against the Python wrapper. The kernel was collapsed into the wrapper (plain
scalar arithmetic, ``None`` instead of a NaN sentinel), so these tests now
assert the function directly against analytic expectations.
"""

from __future__ import annotations

import math

import pytest

from hlanalysis.strategy.late_resolution import _safety_d_for_region as ref_fn


@pytest.mark.parametrize("drift_aware", [False, True])
@pytest.mark.parametrize("mu", [0.0, 0.001, -0.001])
@pytest.mark.parametrize("tte_min", [0.0, 10.0])
def test_safety_d_lower_bounded(drift_aware, mu, tte_min):
    got = ref_fn(
        ref_price=80_300.0,
        lo=80_000.0,
        hi=None,
        sigma_window=1.0,
        mu=mu,
        tte_min=tte_min,
        drift_aware=drift_aware,
    )
    expected = math.log(80_300.0 / 80_000.0)
    if drift_aware:
        expected += mu * tte_min
    assert got is not None
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=0.0)


@pytest.mark.parametrize("drift_aware", [False, True])
@pytest.mark.parametrize("mu", [0.0, 0.001, -0.001])
@pytest.mark.parametrize("tte_min", [0.0, 10.0])
def test_safety_d_upper_bounded(drift_aware, mu, tte_min):
    got = ref_fn(
        ref_price=79_700.0,
        lo=None,
        hi=80_000.0,
        sigma_window=1.0,
        mu=mu,
        tte_min=tte_min,
        drift_aware=drift_aware,
    )
    expected = math.log(80_000.0 / 79_700.0)
    if drift_aware:
        expected -= mu * tte_min
    assert got is not None
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=0.0)


@pytest.mark.parametrize("drift_aware", [False, True])
def test_safety_d_middle_bucket_drops_drift(drift_aware):
    # Middle (two-sided) bucket: nearest adverse boundary, drift NOT applied
    # regardless of drift_aware.
    got = ref_fn(
        ref_price=79_500.0,
        lo=77_991.0,
        hi=81_174.0,
        sigma_window=1.0,
        mu=0.01,
        tte_min=100.0,
        drift_aware=drift_aware,
    )
    expected = min(
        math.log(79_500.0 / 77_991.0),
        math.log(81_174.0 / 79_500.0),
    )
    assert got is not None
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=0.0)


def test_safety_d_middle_bucket_adverse_negative():
    got = ref_fn(
        ref_price=76_000.0,
        lo=77_991.0,
        hi=81_174.0,
        sigma_window=1.0,
        mu=0.0,
        tte_min=0.0,
        drift_aware=False,
    )
    assert got is not None and got < 0.0


def test_safety_d_sigma_nonpositive_returns_none():
    got = ref_fn(
        ref_price=80_300.0,
        lo=80_000.0,
        hi=None,
        sigma_window=0.0,
        mu=0.0,
        tte_min=0.0,
        drift_aware=False,
    )
    assert got is None


def test_safety_d_no_bounds_returns_none():
    got = ref_fn(
        ref_price=80_300.0,
        lo=None,
        hi=None,
        sigma_window=1.0,
        mu=0.0,
        tte_min=0.0,
        drift_aware=False,
    )
    assert got is None


def test_safety_d_adverse_lower_bound_is_negative():
    got = ref_fn(
        ref_price=79_700.0,
        lo=80_000.0,
        hi=None,
        sigma_window=1.0,
        mu=0.0,
        tte_min=0.0,
        drift_aware=False,
    )
    assert got is not None and got < 0.0
