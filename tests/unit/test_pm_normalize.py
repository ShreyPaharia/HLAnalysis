import os
import subprocess
import sys

from hlanalysis.adapters.polymarket_normalize import (
    _question_idx_from_condition,
    parse_book_message,
    parse_gamma_market_to_question_meta,
    parse_gamma_market_to_settlement,
    parse_price_change_message,
    parse_trade_message,
)
from hlanalysis.events import (
    BookSnapshotEvent,
    QuestionMetaEvent,
    SettlementEvent,
    TradeEvent,
)


def _qid_in_subprocess(condition_id: str, *, hashseed: str) -> int:
    code = (
        "from hlanalysis.adapters.polymarket_normalize import "
        "_question_idx_from_condition as f; print(f(%r))" % condition_id
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONHASHSEED": hashseed},
    )
    return int(out.stdout.strip())


def test_question_idx_is_stable_across_process_restarts():
    # question_idx is the SQLite primary key for PM rows (pm_strike, position,
    # seen_question). It MUST be deterministic for a given condition_id, not
    # dependent on the per-process hash seed — otherwise the same market gets a
    # new idx every engine restart, orphaning its persisted strike/position.
    cid = "0x1c908f9bae95c801df44ce29f284800cd1330d81d1c17dd250fcb5e1f6cc9dd1"
    a = _qid_in_subprocess(cid, hashseed="1")
    b = _qid_in_subprocess(cid, hashseed="2")
    assert a == b, f"question_idx not stable across hash seeds: {a} != {b}"
    # And matches an in-process call, in range for a positive 31-bit int.
    v = _question_idx_from_condition(cid)
    assert v == a
    assert 0 <= v <= 0x7FFFFFFF


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


def test_parse_book_message_orders_levels_best_first():
    # PM CLOB `book` frames list levels WORST-first: bids ascending
    # (0.01 → 0.49) and asks descending (0.99 → 0.51). The consumer
    # (MarketState.apply) reads index [0] as top-of-book, so the normalizer
    # MUST re-order to best-first (highest bid first, lowest ask first) or the
    # engine reads the penny/99c extremes — mid pinned at 0.50, never a
    # favorite, no PM trade ever. See gate_decisions.jsonl bid_px=0.01/ask=0.99.
    payload = {
        "event_type": "book",
        "asset_id": "tok-z",
        "timestamp": "1716545000123",
        "bids": [
            {"price": "0.01", "size": "2536.99"},
            {"price": "0.30", "size": "10"},
            {"price": "0.49", "size": "114.6"},
        ],
        "asks": [
            {"price": "0.99", "size": "2536.99"},
            {"price": "0.70", "size": "10"},
            {"price": "0.51", "size": "66.22"},
        ],
    }
    ev = parse_book_message(payload, local_recv_ts=0)
    # Best bid = highest price first; best ask = lowest price first.
    assert ev.bid_px == [0.49, 0.30, 0.01]
    assert ev.bid_sz == [114.6, 10.0, 2536.99]
    assert ev.ask_px == [0.51, 0.70, 0.99]
    assert ev.ask_sz == [66.22, 10.0, 2536.99]


def test_parse_price_change_message_orders_levels_best_first():
    payload = {
        "event_type": "price_change",
        "asset_id": "tok-w",
        "timestamp": "1716545002000",
        "changes": [
            {"price": "0.01", "size": "500", "side": "BUY"},
            {"price": "0.49", "size": "25", "side": "BUY"},
            {"price": "0.99", "size": "70", "side": "SELL"},
            {"price": "0.51", "size": "40", "side": "SELL"},
        ],
    }
    ev = parse_price_change_message(payload, local_recv_ts=0)
    assert ev is not None
    assert ev.bid_px == [0.49, 0.01]
    assert ev.bid_sz == [25.0, 500.0]
    assert ev.ask_px == [0.51, 0.99]
    assert ev.ask_sz == [40.0, 70.0]


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


_SAMPLE_GAMMA_MARKET = {
    "conditionId": "0xcond123",
    "clobTokenIds": '["71321...992563", "71321...111111"]',
    "startDate": "2026-05-24T00:00:00Z",
    "endDate": "2026-05-25T00:00:00Z",
    "description": (
        "Will BTC go up or down? Resolves based on the Binance 1 "
        "minute candle for BTC/USDT May 24 '26 20:00 in the ET timezone..."
    ),
    "outcomePrices": '["0.92","0.08"]',
    "groupItemTitle": "",
}


def test_parse_gamma_market_to_question_meta_binary():
    ev = parse_gamma_market_to_question_meta(
        _SAMPLE_GAMMA_MARKET,
        series_slug="btc-up-or-down-daily",
        local_recv_ts=1716545000000_000_000,
    )
    assert isinstance(ev, QuestionMetaEvent)
    assert ev.venue == "polymarket"
    assert ev.symbol == "71321...992563"
    assert ev.named_outcome_idxs == [0, 1]
    keys = dict(zip(ev.keys, ev.values))
    assert keys["class"] == "priceBinary"
    assert keys["underlying"] == "BTC"
    assert keys["yes_token_id"] == "71321...992563"
    assert keys["no_token_id"] == "71321...111111"
    assert keys["series_slug"] == "btc-up-or-down-daily"
    assert keys["condition_id"] == "0xcond123"
    assert "strike_ref_ts_ns" in keys
    assert int(keys["expiry_ns"]) > 0
    # The HL-shaped `expiry` (YYYYMMDD-HHMM) is what MarketState reads — PM
    # daily 00:00Z endDate must mirror it so PM expiries land on alerts.
    assert keys["expiry"] == "20260525-0000"
    # No static strike on daily up/down — `targetPrice` should be absent
    # rather than emitted as an empty string (which would render as "$").
    assert "targetPrice" not in keys
    # `question_name` carries the human-readable label used by the alert
    # renderer's fallback path when no numeric strike is available.
    assert keys["question_name"].startswith("Will BTC go up or down")


def test_parse_gamma_market_emits_target_price_when_group_item_title_numeric():
    """Polymarket bucket-style markets put their strike in `groupItemTitle`
    (e.g. "$80,000"). The normalizer should lift it to `targetPrice` so
    the existing render path picks it up."""
    bucket_leg = dict(
        _SAMPLE_GAMMA_MARKET,
        groupItemTitle="$80,000",
        question="BTC above $80,000 on May 25?",
    )
    ev = parse_gamma_market_to_question_meta(
        bucket_leg,
        series_slug="btc-strike-buckets",
        local_recv_ts=1716545000000_000_000,
    )
    keys = dict(zip(ev.keys, ev.values))
    assert keys["targetPrice"] == "80000"
    assert keys["question_name"] == "BTC above $80,000 on May 25?"


# endDate of _SAMPLE_GAMMA_MARKET ("2026-05-25T00:00:00Z") == 1779667200e9 ns.
_AFTER_ENDDATE_NS = 1779667260_000_000_000   # endDate + 60s (resolved, polled just after)
_BEFORE_ENDDATE_NS = 1779494400_000_000_000  # endDate − 2d (still open)


def test_parse_gamma_market_to_settlement_when_resolved_yes():
    resolved = dict(_SAMPLE_GAMMA_MARKET, outcomePrices='["1.0","0.0"]')
    ev = parse_gamma_market_to_settlement(
        resolved,
        series_slug="btc-up-or-down-daily",
        local_recv_ts=_AFTER_ENDDATE_NS,
    )
    assert ev is not None
    assert isinstance(ev, SettlementEvent)
    assert ev.settled_side_idx == 0  # YES won
    assert ev.symbol == "71321...992563"
    assert ev.settle_price == 1.0


def test_parse_gamma_market_to_settlement_when_resolved_no():
    resolved = dict(_SAMPLE_GAMMA_MARKET, outcomePrices='["0.0","1.0"]')
    ev = parse_gamma_market_to_settlement(
        resolved,
        series_slug="btc-up-or-down-daily",
        local_recv_ts=_AFTER_ENDDATE_NS,
    )
    assert ev is not None
    assert ev.settled_side_idx == 1
    assert ev.symbol == "71321...111111"


def test_parse_gamma_market_to_settlement_returns_none_when_open():
    ev = parse_gamma_market_to_settlement(
        _SAMPLE_GAMMA_MARKET,
        series_slug="btc-up-or-down-daily",
        local_recv_ts=0,
    )
    assert ev is None


def test_parse_gamma_market_to_settlement_none_before_enddate_even_at_extreme_price():
    # Regression: a BTC up/down daily priced at 0.99/1.0 BEFORE its endDate is
    # just a strong favorite — the market resolves on the endDate candle (often
    # hours later). Treating the extreme price as "settled" de-tracked a still-
    # open live position and booked phantom PnL. Must return None until endDate.
    almost = dict(_SAMPLE_GAMMA_MARKET, outcomePrices='["0.0","1.0"]')
    ev = parse_gamma_market_to_settlement(
        almost,
        series_slug="btc-up-or-down-daily",
        local_recv_ts=_BEFORE_ENDDATE_NS,
    )
    assert ev is None


def test_parse_gamma_market_to_settlement_falls_back_to_price_when_enddate_missing():
    # No endDate → can't time-gate; preserve the legacy price>=0.99 behaviour
    # rather than never settling.
    resolved = dict(_SAMPLE_GAMMA_MARKET, outcomePrices='["1.0","0.0"]')
    del resolved["endDate"]
    ev = parse_gamma_market_to_settlement(
        resolved, series_slug="btc-up-or-down-daily", local_recv_ts=0,
    )
    assert ev is not None
    assert ev.settled_side_idx == 0
