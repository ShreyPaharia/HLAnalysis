# tests/unit/backtest/test_hedge_settlement.py
"""SHR-55: open hedge residuals must be marked-to-market at end-of-data, and
the hedge cost basis must be qty-weighted across fills.

Pure-logic tests for the two helpers the runner uses:
  - `_hedge_avg_entry`: qty-weighted average on same-direction adds, basis
    preserved on reductions, reset on a sign flip.
  - `_hedge_mtm_fill`: a closing Fill that marks an open residual to the last
    hedge mid (so realized PnL captures the hedge's full round-trip, not just
    the opening leg).
"""

from __future__ import annotations

import pytest

from hlanalysis.backtest.runner.hftbt_runner import _hedge_avg_entry, _hedge_mtm_fill


def test_avg_entry_open_from_flat_uses_fill_price() -> None:
    new_qty, new_avg = _hedge_avg_entry(0.0, 0.0, "buy", 0.01, 80_000.0)
    assert new_qty == pytest.approx(0.01)
    assert new_avg == pytest.approx(80_000.0)


def test_avg_entry_same_direction_is_qty_weighted() -> None:
    # Hold 0.01 @ 80_000, buy 0.03 @ 84_000 → 0.04 @ weighted 83_000.
    new_qty, new_avg = _hedge_avg_entry(0.01, 80_000.0, "buy", 0.03, 84_000.0)
    assert new_qty == pytest.approx(0.04)
    assert new_avg == pytest.approx((0.01 * 80_000.0 + 0.03 * 84_000.0) / 0.04)


def test_avg_entry_reduce_keeps_basis() -> None:
    # Long 0.04 @ 83_000, sell 0.01 → basis unchanged.
    new_qty, new_avg = _hedge_avg_entry(0.04, 83_000.0, "sell", 0.01, 90_000.0)
    assert new_qty == pytest.approx(0.03)
    assert new_avg == pytest.approx(83_000.0)


def test_avg_entry_flip_through_zero_resets_basis() -> None:
    # Long 0.01 @ 80_000, sell 0.03 → net short 0.02, basis = the new fill price.
    new_qty, new_avg = _hedge_avg_entry(0.01, 80_000.0, "sell", 0.03, 85_000.0)
    assert new_qty == pytest.approx(-0.02)
    assert new_avg == pytest.approx(85_000.0)


def test_mtm_fill_closes_long_with_sell_at_mark() -> None:
    f = _hedge_mtm_fill("BTC-PERP", qty=0.01, mark_px=81_000.0)
    assert f.is_hedge is True
    assert f.side == "sell"  # closes a long
    assert f.size == pytest.approx(0.01)
    assert f.price == pytest.approx(81_000.0)


def test_mtm_fill_closes_short_with_buy_at_mark() -> None:
    f = _hedge_mtm_fill("BTC-PERP", qty=-0.02, mark_px=79_000.0)
    assert f.side == "buy"  # closes a short
    assert f.size == pytest.approx(0.02)
    assert f.price == pytest.approx(79_000.0)
