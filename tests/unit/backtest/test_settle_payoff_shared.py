"""SHR-88: the backtest settlement payoff MUST route through the shared
``position_math.settlement_payoff_price`` — never a hardcoded YES re-derivation.

Bit-identical regression: every (resolved outcome, held leg) combination must
produce exactly the payoff the runner booked before the routing change, and that
payoff must equal ``settlement_payoff_price(position_side_idx, settled_side_idx)``
where the winner is supplied by the venue/recorded resolved outcome.
"""
import pytest

from hlanalysis.backtest.core.data_source import QuestionDescriptor
from hlanalysis.backtest.runner.hftbt_runner import Position, _settle_px_for_outcome
from hlanalysis.marketdata.position_math import settlement_payoff_price


def _q(legs: tuple[str, ...]) -> QuestionDescriptor:
    return QuestionDescriptor(
        question_id="q1", question_idx=1, start_ts_ns=0, end_ts_ns=10,
        leg_symbols=legs, klass="priceBinary", underlying="BTC",
    )


def _pos(symbol: str) -> Position:
    return Position(
        question_idx=1, symbol=symbol, qty=10.0, avg_entry=0.4,
        stop_loss_price=-1.0, last_update_ts_ns=0,
    )


# (outcome, held_leg_idx) -> expected payoff, exactly as the pre-SHR-88 runner
# booked it. leg[0] is the YES leg, leg[1] the NO leg.
@pytest.mark.parametrize(
    "outcome, held_idx, expected",
    [
        ("yes", 0, 1.0),   # hold YES, YES won  -> win
        ("yes", 1, 0.0),   # hold NO,  YES won  -> loss
        ("no", 1, 1.0),    # hold NO,  NO won   -> win
        ("no", 0, 0.0),    # hold YES, NO won   -> loss
        ("unknown", 0, 0.0),  # unresolved -> worthless
        ("unknown", 1, 0.0),
    ],
)
def test_settle_payoff_bit_identical(outcome: str, held_idx: int, expected: float) -> None:
    legs = ("yA", "nA")
    px = _settle_px_for_outcome(_pos(legs[held_idx]), _q(legs), outcome)
    assert px == expected


def test_settle_payoff_matches_shared_settlement_payoff_price() -> None:
    """The runner's binary payoff is exactly the shared function over the
    venue-supplied winner index (settled_side_idx), with YES=0 / NO=1."""
    legs = ("yA", "nA")
    for outcome, settled_idx in [("yes", 0), ("no", 1)]:
        for held_idx in (0, 1):
            px = _settle_px_for_outcome(_pos(legs[held_idx]), _q(legs), outcome)
            assert px == settlement_payoff_price(held_idx, settled_idx)


def test_settle_payoff_held_leg_not_in_question_is_worthless() -> None:
    """A position whose symbol isn't one of the question's legs can't win."""
    assert _settle_px_for_outcome(_pos("other"), _q(("yA", "nA")), "yes") == 0.0
