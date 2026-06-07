import math
from hlanalysis.strategy.types import QuestionView
from hlanalysis.strategy.render import settlement_pnl_usd


def _qv(**kw):
    base = dict(
        question_idx=1, yes_symbol="", no_symbol="", strike=float("nan"),
        expiry_ns=0, underlying="BTC", klass="priceBucket", period="1d",
        settled=True, leg_symbols=("y80", "n80", "y90", "n90"),
    )
    base.update(kw)
    return QuestionView(**base)


def test_held_winning_leg_booked_as_win_even_when_other_leg_event_arrived():
    # We hold y80 (won). Settlement events arrived for y80 AND n90 (both winners
    # of their pairs). The held leg must book a win regardless of arrival order.
    qv = _qv(settled_symbols=("n90", "y80"))
    pnl = settlement_pnl_usd(qv, "y80", qty=10.0, avg_entry=0.9)
    assert pnl == 10.0 * (1.0 - 0.9)  # +1.0


def test_held_losing_leg_booked_as_loss():
    qv = _qv(settled_symbols=("n80", "y90"))  # y80 lost (n80 won its pair)
    pnl = settlement_pnl_usd(qv, "y80", qty=10.0, avg_entry=0.9)
    assert pnl == 10.0 * (0.0 - 0.9)  # -9.0


def test_binary_settled_symbol_still_works():
    qv = QuestionView(
        question_idx=1, yes_symbol="yA", no_symbol="nA", strike=100.0,
        expiry_ns=0, underlying="BTC", klass="priceBinary", period="1d",
        settled=True, settled_symbol="yA", leg_symbols=("yA", "nA"),
    )
    assert math.isclose(settlement_pnl_usd(qv, "yA", qty=5.0, avg_entry=0.8), 5.0 * 0.2)


def test_binary_via_settled_symbols_path():
    # After the _mark_settled change, live binaries ALSO populate settled_symbols
    # (single winner). Confirm the membership branch books a binary correctly.
    qv = QuestionView(
        question_idx=1, yes_symbol="yA", no_symbol="nA", strike=100.0,
        expiry_ns=0, underlying="BTC", klass="priceBinary", period="1d",
        settled=True, settled_symbol="yA", settled_symbols=("yA",),
        leg_symbols=("yA", "nA"),
    )
    assert math.isclose(settlement_pnl_usd(qv, "yA", qty=5.0, avg_entry=0.8), 5.0 * 0.2)
    assert math.isclose(settlement_pnl_usd(qv, "nA", qty=5.0, avg_entry=0.2), 5.0 * -0.2)
