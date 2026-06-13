"""Full fill→PnL pipeline tests for the shared accounting core (SHR-82).

The basic ``apply_fill`` semantics (signed qty, weighted-average entry, realized
on partial reduce, close detection, ``closed_qty`` accumulation) are pinned by
``tests/unit/test_position_math.py``. This module pins the rest of the *single*
fill→PnL pipeline that the engine and backtest will both route through (Wave-2
SHR-88):

  * settlement payout from a venue-supplied resolved outcome (NEVER a YES
    re-derivation — the bug that booked winning bucket legs as total losses),
  * open-position mark-to-market,
  * the optional ``venue_closed_pnl`` override (live HL trusts venue truth; sim
    computes) — which *replaces* the computed realized so the SHR-72 settlement
    double-count is impossible by construction (one code path, one number).
"""

from __future__ import annotations

import math

from hlanalysis.marketdata.position_math import (
    PositionState,
    apply_fill,
    open_mtm,
    settle,
    settlement_payoff_price,
)


# --- settlement payoff price ------------------------------------------------


def test_settlement_payoff_price_winning_leg_is_one() -> None:
    assert settlement_payoff_price(position_side_idx=2, settled_side_idx=2) == 1.0


def test_settlement_payoff_price_losing_leg_is_zero() -> None:
    assert settlement_payoff_price(position_side_idx=0, settled_side_idx=2) == 0.0


# --- settlement payout (full close at the resolved outcome) -----------------


def test_settle_long_winning_leg_books_gain() -> None:
    pos = PositionState(qty=100.0, avg_entry=0.80, realized_pnl=0.0)
    new, realized = settle(pos, position_side_idx=1, settled_side_idx=1)
    assert new is None  # settlement fully closes
    # won → payoff 1.0; realized = (1.0 - 0.80) * 100
    assert math.isclose(realized, (1.0 - 0.80) * 100.0)


def test_settle_long_losing_leg_books_loss() -> None:
    pos = PositionState(qty=100.0, avg_entry=0.80, realized_pnl=0.0)
    new, realized = settle(pos, position_side_idx=0, settled_side_idx=1)
    assert new is None
    # lost → payoff 0.0; realized = (0.0 - 0.80) * 100 = -80
    assert math.isclose(realized, -0.80 * 100.0)


def test_settle_multi_leg_bucket_winner_supplied_not_rederived() -> None:
    """Regression for the YES-re-derivation bug: a winning bucket leg must book a
    GAIN, not a total loss. Two legs of one bucket settle from the SAME supplied
    ``settled_side_idx`` — the winning leg gains, the losing leg loses."""
    won_leg = PositionState(qty=50.0, avg_entry=0.30, realized_pnl=0.0)
    lost_leg = PositionState(qty=50.0, avg_entry=0.30, realized_pnl=0.0)
    settled_side_idx = 3  # venue truth: leg 3 won

    _, won_realized = settle(won_leg, position_side_idx=3, settled_side_idx=settled_side_idx)
    _, lost_realized = settle(lost_leg, position_side_idx=1, settled_side_idx=settled_side_idx)

    assert won_realized > 0.0  # winning leg is a GAIN, never a total loss
    assert math.isclose(won_realized, (1.0 - 0.30) * 50.0)
    assert math.isclose(lost_realized, -0.30 * 50.0)


def test_settle_short_winning_leg_books_loss() -> None:
    # short the leg that wins → pays out 1.0 → loss of (avg - 1.0) * |qty|
    pos = PositionState(qty=-100.0, avg_entry=0.40, realized_pnl=0.0)
    new, realized = settle(pos, position_side_idx=2, settled_side_idx=2)
    assert new is None
    assert math.isclose(realized, (0.40 - 1.0) * 100.0)


def test_settle_accumulates_prior_realized_into_returned_pnl() -> None:
    """settle goes through the same close path as apply_fill, so the per-event
    realized it returns is the settlement lot only; pairing it with prior
    realized is the caller's job (matching apply_fill's contract)."""
    pos = PositionState(qty=100.0, avg_entry=0.80, realized_pnl=-5.0)
    _, realized = settle(pos, position_side_idx=1, settled_side_idx=1)
    # returned realized is this event only (not + the prior -5.0)
    assert math.isclose(realized, (1.0 - 0.80) * 100.0)


# --- open-position mark-to-market -------------------------------------------


def test_open_mtm_long() -> None:
    pos = PositionState(qty=100.0, avg_entry=0.80, realized_pnl=0.0)
    assert math.isclose(open_mtm(pos, 0.92), (0.92 - 0.80) * 100.0)


def test_open_mtm_short() -> None:
    pos = PositionState(qty=-100.0, avg_entry=0.40, realized_pnl=0.0)
    # short gains when mark falls below entry
    assert math.isclose(open_mtm(pos, 0.30), (0.30 - 0.40) * -100.0)
    assert open_mtm(pos, 0.30) > 0.0


def test_open_mtm_flat_position_is_zero() -> None:
    assert open_mtm(None, 0.50) == 0.0


# --- venue_closed_pnl override (live HL truth) ------------------------------


def test_apply_fill_venue_override_replaces_computed_realized() -> None:
    """When the venue reports closedPnl for a reduce, the override REPLACES the
    computed realized (it does not add to it) — so settlement booked once on a
    venue fill can never be double-counted by also computing it."""
    pos = PositionState(qty=100.0, avg_entry=0.89, realized_pnl=0.0)
    # computed realized would be (0.72-0.89)*40 = -6.8; venue says -7.5
    new, realized = apply_fill(pos, "sell", 40.0, 0.72, venue_closed_pnl=-7.5)
    assert realized == -7.5
    assert new is not None
    assert math.isclose(new.realized_pnl, -7.5)  # not -6.8, not -14.3


def test_apply_fill_venue_override_equals_compute_parity() -> None:
    """When the venue value equals what we'd compute, override and compute paths
    are bit-identical — the parity guarantee the live/sim unification relies on."""
    pos = PositionState(qty=100.0, avg_entry=0.89, realized_pnl=0.0)
    computed_new, computed_realized = apply_fill(pos, "sell", 40.0, 0.72)
    override_new, override_realized = apply_fill(
        pos,
        "sell",
        40.0,
        0.72,
        venue_closed_pnl=computed_realized,
    )
    assert override_realized == computed_realized
    assert override_new == computed_new


def test_settle_venue_override_replaces_computed() -> None:
    """Settlement with a venue closedPnl uses venue truth, not the 1.0/0.0
    payoff computation — the live HL path where the venue books the settlement
    fill's closedPnl directly."""
    pos = PositionState(qty=100.0, avg_entry=0.80, realized_pnl=0.0)
    _, realized = settle(
        pos,
        position_side_idx=1,
        settled_side_idx=1,
        venue_closed_pnl=18.75,
    )
    assert realized == 18.75  # venue truth, not the computed (1.0-0.80)*100=20
