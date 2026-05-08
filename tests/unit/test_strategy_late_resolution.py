from __future__ import annotations

import dataclasses

import pytest

from hlanalysis.strategy.types import Action, Decision, Diagnostic, OrderIntent


def test_decision_is_frozen():
    d = Decision(action=Action.HOLD, intents=(), diagnostics=(Diagnostic("info", "noop"),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.action = Action.ENTER  # type: ignore[misc]


def test_order_intent_signed_size_is_required():
    intent = OrderIntent(
        question_idx=42,
        symbol="@30",
        side="buy",
        size=10.0,
        limit_price=0.95,
        cloid="hla-test",
        time_in_force="ioc",
    )
    assert intent.size > 0
    assert intent.side in ("buy", "sell")


from hlanalysis.strategy.base import Strategy


def test_strategy_abc_cannot_be_instantiated_directly():
    import pytest as _p
    with _p.raises(TypeError):
        Strategy()  # type: ignore[abstract]


import math

from hlanalysis.strategy.late_resolution import (
    LateResolutionConfig,
    LateResolutionStrategy,
)
from hlanalysis.strategy.types import Action, BookState, QuestionView


def _cfg(**overrides) -> LateResolutionConfig:
    base = dict(
        tte_min_seconds=60,
        tte_max_seconds=1800,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200.0,
        vol_max=0.5,
        max_position_usd=100.0,
        stop_loss_pct=10.0,
        max_strike_distance_pct=10.0,
        min_recent_volume_usd=1000.0,
        stale_data_halt_seconds=5,
    )
    base.update(overrides)
    return LateResolutionConfig(**base)


def _ref_book(symbol: str, ask: float, bid: float, *, ts_ns: int = 1_000) -> BookState:
    return BookState(
        symbol=symbol,
        bid_px=bid,
        bid_sz=100.0,
        ask_px=ask,
        ask_sz=100.0,
        last_trade_ts_ns=ts_ns,
        last_l2_ts_ns=ts_ns,
    )


def _q(strike: float = 80_000.0, expiry_ns: int = 0) -> QuestionView:
    return QuestionView(
        question_idx=42,
        yes_symbol="@30",
        no_symbol="@31",
        strike=strike,
        expiry_ns=expiry_ns,
        underlying="BTC",
        klass="priceBinary",
        period="1h",
    )


# --- Entry: happy path (BTC > strike by margin, YES under 1.0, in window, low vol) ---


def test_entry_yes_when_btc_above_strike_and_yes_book_extreme():
    now = 10_000_000_000_000  # ns
    expiry = now + 600 * 1_000_000_000  # 10 min TTE
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.06, bid=0.04, ts_ns=now - 100),
    }
    s = LateResolutionStrategy(_cfg())
    d = s.evaluate(
        question=q,
        books=books,
        reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60),  # ~very low vol
        recent_volume_usd=5_000.0,
        position=None,
        now_ns=now,
    )
    assert d.action is Action.ENTER
    assert len(d.intents) == 1
    intent = d.intents[0]
    assert intent.symbol == "@30"
    assert intent.side == "buy"
    assert intent.time_in_force == "ioc"
    assert math.isclose(intent.limit_price, 0.96, rel_tol=1e-9)


def test_entry_no_when_btc_below_strike_and_no_book_extreme():
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.06, bid=0.04, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.96, bid=0.95, ts_ns=now - 100),
    }
    s = LateResolutionStrategy(_cfg())
    d = s.evaluate(
        question=q,
        books=books,
        reference_price=79_700.0,
        recent_returns=tuple([0.0001] * 60),
        recent_volume_usd=5_000.0,
        position=None,
        now_ns=now,
    )
    assert d.action is Action.ENTER
    assert d.intents[0].symbol == "@31"


# --- Hold: each gate, one at a time ---


def test_hold_when_tte_too_long():
    now = 10_000_000_000_000
    expiry = now + 3600 * 1_000_000_000  # 1h TTE > 30 min cap
    q = _q(expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.06, bid=0.04, ts_ns=now - 100),
    }
    s = LateResolutionStrategy(_cfg())
    d = s.evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.HOLD


def test_hold_when_tte_too_short():
    now = 10_000_000_000_000
    expiry = now + 30 * 1_000_000_000  # 30s TTE < 60s floor
    q = _q(expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.06, bid=0.04, ts_ns=now - 100),
    }
    d = LateResolutionStrategy(_cfg()).evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.HOLD


def test_hold_when_winning_leg_not_extreme():
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q()
    books = {
        # YES book is winning side (BTC > strike), but ask=0.80 < 0.95 threshold
        "@30": _ref_book("@30", ask=0.80, bid=0.78, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.22, bid=0.20, ts_ns=now - 100),
    }
    d = LateResolutionStrategy(_cfg()).evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.HOLD


def test_hold_when_distance_below_min():
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0)
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.04, bid=0.03, ts_ns=now - 100),
    }
    d = LateResolutionStrategy(_cfg()).evaluate(
        question=q, books=books,
        reference_price=80_050.0,  # only $50 above strike < $200 min
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.HOLD


def test_hold_when_realized_vol_above_cap():
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q()
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.04, bid=0.03, ts_ns=now - 100),
    }
    high_vol_returns = tuple([0.05, -0.05] * 30)  # huge swings
    d = LateResolutionStrategy(_cfg()).evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=high_vol_returns, recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.HOLD


def test_hold_when_book_stale():
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q()
    stale_ts = now - 10 * 1_000_000_000  # 10s old > 5s halt
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=stale_ts),
        "@31": _ref_book("@31", ask=0.04, bid=0.03, ts_ns=stale_ts),
    }
    d = LateResolutionStrategy(_cfg()).evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.HOLD


def test_hold_when_position_already_held():
    from hlanalysis.strategy.types import Position
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q()
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.04, bid=0.03, ts_ns=now - 100),
    }
    pos = Position(
        question_idx=q.question_idx, symbol="@30", qty=10.0, avg_entry=0.95,
        stop_loss_price=0.855, last_update_ts_ns=now - 1_000_000,
    )
    d = LateResolutionStrategy(_cfg()).evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=pos, now_ns=now,
    )
    # No re-entry; exit logic runs separately (Task 6)
    assert d.action is not Action.ENTER


def test_exit_signal_when_question_is_settled():
    from hlanalysis.strategy.types import Position
    now = 10_000_000_000_000
    q = QuestionView(
        question_idx=42, yes_symbol="@30", no_symbol="@31",
        strike=80_000.0, expiry_ns=now - 1_000_000,
        underlying="BTC", klass="priceBinary", period="1h",
        settled=True, settled_side="yes",
    )
    books = {
        "@30": _ref_book("@30", ask=1.0, bid=1.0, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.0, bid=0.0, ts_ns=now - 100),
    }
    pos = Position(
        question_idx=q.question_idx, symbol="@30", qty=10.0, avg_entry=0.95,
        stop_loss_price=0.855, last_update_ts_ns=now - 1_000_000,
    )
    d = LateResolutionStrategy(_cfg()).evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=pos, now_ns=now,
    )
    assert d.action is Action.EXIT
    # Settlement-driven exit: zero intents — engine marks the position closed
    # at the venue's settlement value, no order needed.
    assert d.intents == ()


def test_exit_intent_when_price_below_stop_loss():
    from hlanalysis.strategy.types import Position
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.84, bid=0.83, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.16, bid=0.15, ts_ns=now - 100),
    }
    pos = Position(
        question_idx=q.question_idx, symbol="@30", qty=10.0, avg_entry=0.95,
        stop_loss_price=0.855, last_update_ts_ns=now - 1_000_000,
    )
    d = LateResolutionStrategy(_cfg()).evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=pos, now_ns=now,
    )
    assert d.action is Action.EXIT
    assert len(d.intents) == 1
    intent = d.intents[0]
    assert intent.symbol == "@30"
    assert intent.side == "sell"
    assert intent.reduce_only is True
    assert intent.time_in_force == "ioc"
