from hlanalysis.adapters.polymarket_normalize import (
    parse_book_message,
    parse_price_change_message,
    parse_trade_message,
)
from hlanalysis.events import BookSnapshotEvent, TradeEvent


def test_parse_book_message_yields_book_snapshot_event():
    payload = {
        "event_type": "book",
        "asset_id": "71321045679252212594626385532706912750332728571942532289631379312455583992563",
        "market": "0xabc",
        "timestamp": "1716545000123",
        "hash": "h",
        "bids": [{"price": "0.92", "size": "150"}, {"price": "0.91", "size": "80"}],
        "asks": [{"price": "0.93", "size": "60"}, {"price": "0.94", "size": "120"}],
    }
    ev = parse_book_message(payload, local_recv_ts=1716545000200_000_000)
    assert isinstance(ev, BookSnapshotEvent)
    assert ev.venue == "polymarket"
    assert ev.symbol == payload["asset_id"]
    assert ev.bid_px == [0.92, 0.91]
    assert ev.bid_sz == [150.0, 80.0]
    assert ev.ask_px == [0.93, 0.94]
    assert ev.ask_sz == [60.0, 120.0]
    assert ev.exchange_ts == 1716545000123 * 1_000_000  # ms → ns
    assert ev.local_recv_ts == 1716545000200_000_000


def test_parse_trade_message_yields_trade_event():
    payload = {
        "event_type": "last_trade_price",
        "asset_id": "71321045679252212594626385532706912750332728571942532289631379312455583992563",
        "market": "0xabc",
        "price": "0.927",
        "size": "100",
        "side": "BUY",
        "timestamp": "1716545001000",
        "trade_id": "tid-9",
    }
    ev = parse_trade_message(payload, local_recv_ts=1716545001100_000_000)
    assert isinstance(ev, TradeEvent)
    assert ev.symbol == payload["asset_id"]
    assert ev.price == 0.927
    assert ev.size == 100.0
    assert ev.side == "buy"
    assert ev.trade_id == "tid-9"
    assert ev.exchange_ts == 1716545001000 * 1_000_000


def test_parse_trade_message_sell_side_lowercased():
    payload = {
        "event_type": "last_trade_price",
        "asset_id": "tok-x",
        "price": "0.5",
        "size": "10",
        "side": "SELL",
        "timestamp": "1716545001000",
    }
    ev = parse_trade_message(payload, local_recv_ts=0)
    assert ev.side == "sell"
    assert ev.trade_id is None


def test_parse_price_change_message_splits_bids_and_asks():
    payload = {
        "event_type": "price_change",
        "asset_id": "tok-y",
        "timestamp": "1716545002000",
        "changes": [
            {"price": "0.90", "size": "50", "side": "BUY"},
            {"price": "0.88", "size": "25", "side": "BUY"},
            {"price": "0.95", "size": "70", "side": "SELL"},
        ],
    }
    ev = parse_price_change_message(payload, local_recv_ts=1716545002100_000_000)
    assert ev is not None
    assert isinstance(ev, BookSnapshotEvent)
    assert ev.symbol == "tok-y"
    assert ev.bid_px == [0.90, 0.88]
    assert ev.bid_sz == [50.0, 25.0]
    assert ev.ask_px == [0.95]
    assert ev.ask_sz == [70.0]
    assert ev.exchange_ts == 1716545002000 * 1_000_000


def test_parse_price_change_message_returns_none_on_empty_changes():
    payload = {
        "event_type": "price_change",
        "asset_id": "tok-z",
        "timestamp": "1716545002000",
        "changes": [],
    }
    assert parse_price_change_message(payload, local_recv_ts=0) is None
