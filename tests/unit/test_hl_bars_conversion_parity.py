"""FIX B parity tests: recent_hl_bars tuple conversion is bit-identical.

Guards the claim that::

    tuple((h, lo) for h, lo in arr.tolist())

produces exactly the same Python floats as::

    tuple((float(h), float(lo)) for h, lo in arr)

for any numpy float64 array of shape (N, 2), including edge cases like N=0.

``np.float64(x).tolist()`` (C-level unbox) and ``float(np.float64(x))``
(Python-level cast) both call the same C function and return exactly equal
Python floats, so bit-identical results are guaranteed.  This test pins that
guarantee.
"""

from __future__ import annotations

import numpy as np
import pytest


def _old_conversion(arr: np.ndarray) -> tuple[tuple[float, float], ...]:
    """The current (pre-fix) per-element path from hftbt_runner.py:686."""
    return tuple((float(h), float(lo)) for h, lo in arr)


def _new_conversion(arr: np.ndarray) -> tuple[tuple[float, float], ...]:
    """The optimised bulk-tolist path (FIX B)."""
    return tuple((h, lo) for h, lo in arr.tolist())


# ---------------------------------------------------------------------------
# Parameterised test cases
# ---------------------------------------------------------------------------


def _make_array(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(90_000.0, 110_000.0, size=(n, 2)).astype(np.float64)


@pytest.mark.parametrize(
    "seed,n",
    [
        (0, 0),  # empty — edge case
        (1, 1),  # single row
        (2, 5),  # small
        (3, 50),  # typical lookback (e.g. 50 × 5s bars)
        (4, 200),  # larger
        (5, 1000),  # stress
    ],
)
def test_conversion_bit_identical(seed: int, n: int):
    arr = _make_array(seed, n)
    old = _old_conversion(arr)
    new = _new_conversion(arr)
    assert old == new, (
        f"seed={seed} n={n}: old and new conversions differ\n"
        f"  first diff at: {next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), 'lengths differ')}"
    )


def test_empty_array_shape():
    """N=0 must produce an empty tuple, not error."""
    arr = np.empty((0, 2), dtype=np.float64)
    assert _new_conversion(arr) == ()
    assert _old_conversion(arr) == ()


def test_known_values():
    """Spot-check a handful of exact float64 boundary values."""
    vals = [
        [0.0, 0.0],
        [1.0, 1.0],
        [1e-300, 1e300],
        [float("inf"), float("-inf")],
        [np.finfo(np.float64).max, np.finfo(np.float64).tiny],
    ]
    arr = np.array(vals, dtype=np.float64)
    assert _old_conversion(arr) == _new_conversion(arr)


def test_result_elements_are_plain_floats():
    """Values from the new path must be Python floats, not numpy scalars."""
    arr = _make_array(42, 3)
    result = _new_conversion(arr)
    for h, lo in result:
        assert type(h) is float, f"Expected float, got {type(h)}"
        assert type(lo) is float, f"Expected float, got {type(lo)}"
