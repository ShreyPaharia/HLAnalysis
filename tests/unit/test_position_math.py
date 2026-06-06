"""Canonical signed-position fill math — single source of truth shared by the
live router (``router._book_fill``) and the backtest runner (``hftbt_runner``).

Both used to reimplement: signed qty, weighted-average entry on add-ons,
realized-PnL on partial reduce, close detection, stop-price stamping. Drift
between the copies caused the position-accounting fidelity bugs in the project
history. These tests pin the canonical semantics that both callers now share.
"""
from __future__ import annotations

import math

from hlanalysis.marketdata.position_math import (
    STOP_DISABLED_SENTINEL,
    PositionState,
    apply_fill,
    stop_price,
)


def test_sentinel_is_negative_one() -> None:
    assert STOP_DISABLED_SENTINEL == -1.0


def test_stop_price_disabled_when_pct_none() -> None:
    assert stop_price(0.92, None) == STOP_DISABLED_SENTINEL


def test_stop_price_pct() -> None:
    # 10% stop on a 0.90 entry → 0.81
    assert stop_price(0.90, 10.0) == 0.81
    # floored at 0
    assert stop_price(0.50, 300.0) == 0.0


def test_open_long_from_flat() -> None:
    new, realized = apply_fill(None, "buy", 100.0, 0.90)
    assert new == PositionState(qty=100.0, avg_entry=0.90, realized_pnl=0.0)
    assert realized == 0.0


def test_open_short_from_flat() -> None:
    new, realized = apply_fill(None, "sell", 50.0, 0.40)
    assert new == PositionState(qty=-50.0, avg_entry=0.40, realized_pnl=0.0)
    assert realized == 0.0


def test_addon_weighted_average_entry() -> None:
    pos = PositionState(qty=100.0, avg_entry=0.90, realized_pnl=0.0)
    new, realized = apply_fill(pos, "buy", 100.0, 0.80)
    # weighted avg = (100*0.90 + 100*0.80) / 200 = 0.85
    assert new is not None
    assert new.qty == 200.0
    assert math.isclose(new.avg_entry, 0.85)
    assert new.realized_pnl == 0.0  # add-ons realize nothing
    assert realized == 0.0


def test_partial_reduce_realizes_on_closed_lot_basis_unchanged() -> None:
    pos = PositionState(qty=100.0, avg_entry=0.89, realized_pnl=0.0)
    new, realized = apply_fill(pos, "sell", 40.0, 0.72)
    # realized on the 40 closed: (0.72 - 0.89) * 40 = -6.8; basis stays 0.89
    assert new is not None
    assert new.qty == 60.0
    assert new.avg_entry == 0.89
    assert math.isclose(realized, (0.72 - 0.89) * 40.0)
    assert math.isclose(new.realized_pnl, (0.72 - 0.89) * 40.0)


def test_full_close_long_returns_none_with_realized() -> None:
    pos = PositionState(qty=100.0, avg_entry=0.89, realized_pnl=-2.0)
    new, realized = apply_fill(pos, "sell", 100.0, 0.95)
    assert new is None
    assert math.isclose(realized, (0.95 - 0.89) * 100.0)


def test_full_close_short_returns_none_with_realized() -> None:
    pos = PositionState(qty=-100.0, avg_entry=0.40, realized_pnl=0.0)
    new, realized = apply_fill(pos, "buy", 100.0, 0.30)
    assert new is None
    # short realized = (avg - price) * closed = (0.40 - 0.30) * 100
    assert math.isclose(realized, (0.40 - 0.30) * 100.0)


def test_reduce_partial_short() -> None:
    pos = PositionState(qty=-100.0, avg_entry=0.40, realized_pnl=1.0)
    new, realized = apply_fill(pos, "buy", 30.0, 0.35)
    assert new is not None
    assert new.qty == -70.0
    assert new.avg_entry == 0.40
    assert math.isclose(realized, (0.40 - 0.35) * 30.0)
    assert math.isclose(new.realized_pnl, 1.0 + (0.40 - 0.35) * 30.0)


def test_close_atol_parameter_treats_small_residual_as_closed() -> None:
    """The backtest runner closes when the residual drops below one lot, not at
    1e-9. close_atol parameterizes that without forking the math."""
    pos = PositionState(qty=10.0, avg_entry=0.90, realized_pnl=0.0)
    # sell 9.999 leaves 0.001 residual; with lot-size atol that's a close
    new, _ = apply_fill(pos, "sell", 9.999, 0.95, close_atol=0.01)
    assert new is None
    # with the default 1e-9 atol it stays open as a tiny residual
    new2, _ = apply_fill(pos, "sell", 9.999, 0.95)
    assert new2 is not None
    assert math.isclose(new2.qty, 0.001)
