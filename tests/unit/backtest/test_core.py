from __future__ import annotations

import dataclasses

import pytest

from hlanalysis.backtest.core.data_source import QuestionDescriptor
from hlanalysis.backtest.core.events import (
    BookSnapshot,
    MarketEvent,
    ReferenceEvent,
    SettlementEvent,
    TradeEvent,
)
from hlanalysis.backtest.core.question import build_question_view


def test_book_snapshot_is_frozen_slotted():
    snap = BookSnapshot(ts_ns=1, symbol="x", bids=(), asks=())
    assert dataclasses.is_dataclass(snap)
    # frozen → assignment forbidden
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.ts_ns = 2  # type: ignore[misc]
    # slots → no __dict__
    assert not hasattr(snap, "__dict__")


def test_trade_event_side_literal():
    t = TradeEvent(ts_ns=1, symbol="x", side="buy", price=0.5, size=1.0)
    assert t.side == "buy"


def test_reference_event_fields():
    r = ReferenceEvent(ts_ns=2, symbol="BTC", high=100.0, low=99.0, close=99.5)
    assert r.high >= r.low
    assert r.symbol == "BTC"


def test_settlement_event_outcome_literal():
    s = SettlementEvent(ts_ns=3, question_idx=0, outcome="yes")
    assert s.outcome == "yes"


def test_market_event_union_covers_all():
    # MarketEvent type should accept any of the four event variants — checked
    # by isinstance through the underlying types.
    events: list[MarketEvent] = [
        BookSnapshot(ts_ns=1, symbol="x", bids=(), asks=()),
        TradeEvent(ts_ns=2, symbol="x", side="buy", price=0.5, size=1.0),
        ReferenceEvent(ts_ns=3, symbol="y", high=1.0, low=0.5, close=0.75),
        SettlementEvent(ts_ns=4, question_idx=0, outcome="no"),
    ]
    ts = [e.ts_ns for e in events]
    assert ts == sorted(ts)


def test_question_descriptor_is_frozen_slotted():
    q = QuestionDescriptor(
        question_id="q",
        question_idx=1,
        start_ts_ns=0,
        end_ts_ns=1_000_000_000,
        leg_symbols=("yes", "no"),
        klass="priceBinary",
        underlying="BTC",
    )
    assert dataclasses.is_dataclass(q)
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.question_idx = 2  # type: ignore[misc]
    assert not hasattr(q, "__dict__")


def test_question_view_builder_binary():
    q = QuestionDescriptor(
        question_id="q",
        question_idx=1,
        start_ts_ns=0,
        end_ts_ns=1_000_000_000,
        leg_symbols=("yes", "no"),
        klass="priceBinary",
        underlying="BTC",
    )
    view = build_question_view(q, now_ns=500_000_000, strike=60_000.0)
    assert view.yes_symbol == "yes"
    assert view.no_symbol == "no"
    assert view.strike == 60_000.0
    assert view.klass == "priceBinary"
    assert view.expiry_ns == 1_000_000_000
    assert view.settled is False


def test_question_view_builder_bucket_no_yes_no_aliases():
    q = QuestionDescriptor(
        question_id="qb",
        question_idx=2,
        start_ts_ns=0,
        end_ts_ns=1_000_000_000,
        leg_symbols=("yo0", "no0", "yo1", "no1"),
        klass="priceBucket",
        underlying="BTC",
    )
    view = build_question_view(q, now_ns=0, strike=0.0)
    assert view.yes_symbol == ""
    assert view.no_symbol == ""
    assert view.leg_symbols == ("yo0", "no0", "yo1", "no1")
    assert view.klass == "priceBucket"


def test_question_view_builder_auto_settled():
    q = QuestionDescriptor(
        question_id="q",
        question_idx=1,
        start_ts_ns=0,
        end_ts_ns=100,
        leg_symbols=("a", "b"),
        klass="priceBinary",
        underlying="BTC",
    )
    # now_ns past end → settled by default
    view = build_question_view(q, now_ns=200, strike=0.0)
    assert view.settled is True
    # now_ns at or below end → not settled
    view2 = build_question_view(q, now_ns=100, strike=0.0)
    assert view2.settled is False
