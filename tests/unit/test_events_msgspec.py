"""Part C: NormalizedEvent is a msgspec.Struct tagged union, not pydantic.

The recorder serializes every event to a plain dict via `to_record(ev)`. That
dict must be byte-for-byte equivalent to the pre-migration pydantic
`model_dump(mode="python")` output — SAME columns, SAME values, SAME order —
so existing parquet partitions stay schema-compatible. These expected dicts
are pinned from the pydantic output captured before the migration.
"""

from __future__ import annotations

import msgspec

from hlanalysis.events import (
    BboEvent,
    BookSnapshotEvent,
    FundingEvent,
    HealthEvent,
    MarkEvent,
    Mechanism,
    NormalizedEvent,
    ProductType,
    QuestionMetaEvent,
    SettlementEvent,
    TradeEvent,
    to_record,
)


def test_events_are_msgspec_structs_not_pydantic():
    assert issubclass(TradeEvent, msgspec.Struct)
    # no pydantic round-trip API on the hot-path event
    assert not hasattr(TradeEvent, "model_dump")


def test_trade_record_matches_legacy_pydantic_dump():
    ev = TradeEvent(
        venue="binance",
        product_type=ProductType.PERP,
        mechanism=Mechanism.CLOB,
        symbol="BTCUSDT",
        exchange_ts=1,
        local_recv_ts=2,
        price=1.5,
        size=2.0,
        side="buy",
        trade_id="t1",
    )
    assert to_record(ev) == {
        "venue": "binance",
        "product_type": "perp",
        "mechanism": "clob",
        "symbol": "BTCUSDT",
        "exchange_ts": 1,
        "local_recv_ts": 2,
        "seq": None,
        "event_type": "trade",
        "price": 1.5,
        "size": 2.0,
        "side": "buy",
        "trade_id": "t1",
        "block_ts": None,
        "buyer": None,
        "seller": None,
        "block_hash": None,
    }
    # exact column order is part of the parquet-schema equivalence contract
    assert list(to_record(ev).keys()) == [
        "venue",
        "product_type",
        "mechanism",
        "symbol",
        "exchange_ts",
        "local_recv_ts",
        "seq",
        "event_type",
        "price",
        "size",
        "side",
        "trade_id",
        "block_ts",
        "buyer",
        "seller",
        "block_hash",
    ]


def test_bbo_record_matches_legacy():
    ev = BboEvent(
        venue="binance",
        product_type=ProductType.SPOT,
        mechanism=Mechanism.CLOB,
        symbol="X",
        exchange_ts=0,
        local_recv_ts=2,
        seq=7,
        bid_px=1.0,
        bid_sz=2.0,
        ask_px=3.0,
        ask_sz=4.0,
    )
    assert to_record(ev) == {
        "venue": "binance",
        "product_type": "spot",
        "mechanism": "clob",
        "symbol": "X",
        "exchange_ts": 0,
        "local_recv_ts": 2,
        "seq": 7,
        "event_type": "bbo",
        "bid_px": 1.0,
        "bid_sz": 2.0,
        "ask_px": 3.0,
        "ask_sz": 4.0,
    }


def test_question_meta_record_matches_legacy_including_list_defaults():
    ev = QuestionMetaEvent(
        venue="hyperliquid",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="Q1",
        exchange_ts=0,
        local_recv_ts=1,
        question_idx=5,
        named_outcome_idxs=[0, 1],
    )
    assert to_record(ev) == {
        "venue": "hyperliquid",
        "product_type": "prediction_binary",
        "mechanism": "clob",
        "symbol": "Q1",
        "exchange_ts": 0,
        "local_recv_ts": 1,
        "seq": None,
        "event_type": "question_meta",
        "question_idx": 5,
        "named_outcome_idxs": [0, 1],
        "fallback_outcome_idx": None,
        "settled_named_outcome_idxs": [],
        "keys": [],
        "values": [],
    }


def test_settlement_record_matches_legacy():
    ev = SettlementEvent(
        venue="polymarket",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="tok",
        exchange_ts=9,
        local_recv_ts=9,
        settled_side_idx=0,
        settle_price=1.0,
        settle_ts=9,
        keys=["a"],
        values=["b"],
    )
    assert to_record(ev) == {
        "venue": "polymarket",
        "product_type": "prediction_binary",
        "mechanism": "clob",
        "symbol": "tok",
        "exchange_ts": 9,
        "local_recv_ts": 9,
        "seq": None,
        "event_type": "settlement",
        "settled_side_idx": 0,
        "settle_price": 1.0,
        "settle_ts": 9,
        "keys": ["a"],
        "values": ["b"],
    }


def test_health_and_funding_and_mark_records():
    h = HealthEvent(
        venue="binance",
        product_type=ProductType.PERP,
        mechanism=Mechanism.CLOB,
        symbol="*",
        exchange_ts=1,
        local_recv_ts=1,
        kind="connected",
        detail="",
    )
    assert to_record(h) == {
        "venue": "binance",
        "product_type": "perp",
        "mechanism": "clob",
        "symbol": "*",
        "exchange_ts": 1,
        "local_recv_ts": 1,
        "seq": None,
        "event_type": "health",
        "kind": "connected",
        "detail": "",
    }
    f = FundingEvent(
        venue="binance",
        product_type=ProductType.PERP,
        mechanism=Mechanism.CLOB,
        symbol="BTCUSDT",
        exchange_ts=1,
        local_recv_ts=1,
        funding_rate=0.01,
    )
    assert to_record(f)["event_type"] == "funding"
    assert to_record(f)["funding_rate"] == 0.01
    assert to_record(f)["premium"] is None
    m = MarkEvent(
        venue="binance",
        product_type=ProductType.PERP,
        mechanism=Mechanism.CLOB,
        symbol="BTCUSDT",
        exchange_ts=1,
        local_recv_ts=1,
        mark_px=50000.0,
    )
    assert to_record(m) == {
        "venue": "binance",
        "product_type": "perp",
        "mechanism": "clob",
        "symbol": "BTCUSDT",
        "exchange_ts": 1,
        "local_recv_ts": 1,
        "seq": None,
        "event_type": "mark",
        "mark_px": 50000.0,
    }


def test_book_snapshot_record():
    ev = BookSnapshotEvent(
        venue="hyperliquid",
        product_type=ProductType.PERP,
        mechanism=Mechanism.CLOB,
        symbol="BTC",
        exchange_ts=1,
        local_recv_ts=1,
        bid_px=[1.0],
        bid_sz=[2.0],
        ask_px=[3.0],
        ask_sz=[4.0],
    )
    rec = to_record(ev)
    assert rec["event_type"] == "book_snapshot"
    assert rec["bid_px"] == [1.0] and rec["ask_sz"] == [4.0]


def test_frozen_and_replace_semantics():
    ev = TradeEvent(
        venue="binance",
        product_type=ProductType.PERP,
        mechanism=Mechanism.CLOB,
        symbol="BTCUSDT",
        exchange_ts=1,
        local_recv_ts=2,
        price=1.5,
        size=2.0,
        side="buy",
    )
    import msgspec.structs as ms

    ev2 = ms.replace(ev, symbol="OTHER")
    assert ev2.symbol == "OTHER"
    assert ev.symbol == "BTCUSDT"  # original untouched (frozen)
    assert ev2 is not ev


def test_normalized_event_is_union_of_structs():
    # Type alias still importable and usable as an annotation target.
    assert TradeEvent in NormalizedEvent.__args__
    assert HealthEvent in NormalizedEvent.__args__
