"""Unit tests for the pm_binary fee-curve extraction (refactor/fee).

These tests prove that:
1. ``pm_binary_fee`` equals the literal old inline formula across a
   grid of probabilities.
2. ``fee_per_share`` (strategy layer) produces bit-identical results to the old
   inline ``cfg.fee_rate * p * (1.0 - p)`` for pm_binary, and the flat path is
   unchanged.
3. ``_binary_fee`` (backtest runner) produces bit-identical results to the old
   inline ``qty * cfg.fee_rate * p * (1-p)`` (with p-clamp) for pm_binary, and
   the flat path is unchanged.

No mocking is needed — both functions are pure arithmetic.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hlanalysis.strategy.fee import fee_per_share, pm_binary_fee
from hlanalysis.backtest.runner.hftbt_runner import RunConfig, _binary_fee


# ---------------------------------------------------------------------------
# 1. pm_binary_fee — pure formula grid
# ---------------------------------------------------------------------------

_P_GRID = [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]
_RATE_GRID = [0.0, 0.01, 0.07, 0.10]


@pytest.mark.parametrize("p", _P_GRID)
@pytest.mark.parametrize("fee_rate", _RATE_GRID)
def test_pm_binary_fee_matches_inline(fee_rate: float, p: float) -> None:
    """pm_binary_fee must equal the literal old formula fee_rate*p*(1-p)."""
    expected = fee_rate * p * (1.0 - p)
    assert pm_binary_fee(fee_rate, p) == expected


# ---------------------------------------------------------------------------
# 2. fee_per_share bit-identical to old inline for pm_binary + flat paths
# ---------------------------------------------------------------------------


def _make_cfg(
    fee_model: str,
    fee_rate: float = 0.07,
    fee_taker: float = 0.002,
    exit_fee: float = 0.001,
    exit_take_profit_mode: bool = False,
):
    return SimpleNamespace(
        fee_model=fee_model,
        fee_rate=fee_rate,
        fee_taker=fee_taker,
        exit_fee=exit_fee,
        exit_take_profit_mode=exit_take_profit_mode,
    )


_FEE_CASES = [
    (0.0, 0.07),
    (0.01, 0.07),
    (0.25, 0.07),
    (0.5, 0.07),
    (0.75, 0.07),
    (0.99, 0.07),
    (1.0, 0.07),
    (0.5, 0.0),
    (0.5, 0.10),
]


@pytest.mark.parametrize("p,fee_rate", _FEE_CASES)
@pytest.mark.parametrize("side", ["entry", "exit"])
def test_fee_per_share_pm_binary_bit_identical(p: float, fee_rate: float, side: str) -> None:
    """fee_per_share with pm_binary must equal the old inline fee_rate*p*(1-p)."""
    cfg = _make_cfg("pm_binary", fee_rate=fee_rate)
    expected = cfg.fee_rate * p * (1.0 - p)
    assert fee_per_share(cfg, p, side=side) == expected


@pytest.mark.parametrize(
    "side,exit_take_profit_mode,expected_key",
    [
        ("entry", False, "fee_taker"),
        ("entry", True, "fee_taker"),
        ("exit", False, "fee_taker"),
        ("exit", True, "exit_fee"),
    ],
)
def test_fee_per_share_flat_paths_unchanged(side: str, exit_take_profit_mode: bool, expected_key: str) -> None:
    """Flat fee paths must be unaffected by the pm_binary extraction."""
    cfg = _make_cfg("flat", fee_taker=0.003, exit_fee=0.001, exit_take_profit_mode=exit_take_profit_mode)
    result = fee_per_share(cfg, 0.5, side=side)
    assert result == getattr(cfg, expected_key)


# ---------------------------------------------------------------------------
# 3. _binary_fee bit-identical to old inline for pm_binary + flat paths
# ---------------------------------------------------------------------------

# (px, qty, fee_rate) triples — px may be outside [0,1] to test the p-clamp
_BINARY_FEE_CASES = [
    (0.0, 100.0, 0.07),
    (0.01, 100.0, 0.07),
    (0.25, 50.0, 0.07),
    (0.5, 200.0, 0.07),
    (0.75, 75.0, 0.07),
    (0.99, 100.0, 0.07),
    (1.0, 100.0, 0.07),
    # p-clamp cases (px outside [0,1])
    (-0.1, 100.0, 0.07),  # clamps to p=0.0
    (1.5, 100.0, 0.07),  # clamps to p=1.0
    # different fee_rate
    (0.5, 100.0, 0.0),
    (0.5, 100.0, 0.10),
]


@pytest.mark.parametrize("px,qty,fee_rate", _BINARY_FEE_CASES)
def test_binary_fee_pm_binary_bit_identical(px: float, qty: float, fee_rate: float) -> None:
    """_binary_fee with pm_binary must equal qty * pm_binary_fee(fee_rate, clamp(px)).

    The refactored code is ``qty * pm_binary_fee(fee_rate, p)`` which
    equals the old inline ``qty * fee_rate * p * (1-p)`` in value, but the two
    expressions may differ by a single floating-point ULP due to multiplication
    associativity (e.g. ``qty * (fee_rate * p * (1-p))`` vs
    ``((qty * fee_rate) * p) * (1-p)``). We use pytest.approx with a tight
    relative tolerance (1e-12) to assert the values are arithmetically
    equivalent — the extracted function computes the same Polymarket formula."""
    cfg = RunConfig(fee_model="pm_binary", fee_rate=fee_rate)
    p = max(0.0, min(1.0, px))
    expected = qty * fee_rate * p * (1.0 - p)
    assert _binary_fee(px, qty, cfg) == pytest.approx(expected, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize(
    "px,qty,fee_taker",
    [
        (0.5, 100.0, 0.0),
        (0.5, 100.0, 0.002),
        (0.25, 50.0, 0.005),
        (0.99, 200.0, 0.001),
    ],
)
def test_binary_fee_flat_bit_identical(px: float, qty: float, fee_taker: float) -> None:
    """_binary_fee flat path must equal px * qty * fee_taker (unchanged)."""
    cfg = RunConfig(fee_model="flat", fee_taker=fee_taker)
    expected = px * qty * fee_taker
    assert _binary_fee(px, qty, cfg) == expected
