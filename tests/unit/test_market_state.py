# tests/unit/test_market_state.py
from __future__ import annotations

import math
from collections import deque

from hlanalysis.engine.market_state import MarketState
from hlanalysis.events import (
    BboEvent, BookSnapshotEvent, MarkEvent, Mechanism, ProductType,
    QuestionMetaEvent, SettlementEvent, TradeEvent,
)


def _bbo(symbol: str, bid: float, ask: float, ts: int = 1) -> BboEvent:
    return BboEvent(
        venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol=symbol,
        exchange_ts=ts, local_recv_ts=ts,
        bid_px=bid, bid_sz=10.0, ask_px=ask, ask_sz=10.0,
    )


def test_bbo_updates_book_state():
    ms = MarketState()
    ms.apply(_bbo("#30", 0.94, 0.95, ts=1_000_000_000))
    bs = ms.book("#30")
    assert bs is not None
    assert bs.bid_px == 0.94 and bs.ask_px == 0.95
    assert bs.last_l2_ts_ns == 1_000_000_000


def test_question_registry_built_from_question_meta():
    ms = MarketState()
    qm = QuestionMetaEvent(
        venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="qmeta",
        exchange_ts=1, local_recv_ts=1,
        question_idx=42, named_outcome_idxs=[3],
        keys=["class", "underlying", "period", "expiry", "strike"],
        values=["priceBinary", "BTC", "1h", "20260508-1200", "80000"],
    )
    ms.apply(qm)
    q = ms.question(42)
    assert q is not None
    assert q.underlying == "BTC"
    assert q.klass == "priceBinary"
    assert q.strike == 80_000.0
    assert q.yes_symbol == "#30"
    assert q.no_symbol == "#31"


def test_settlement_marks_question_settled():
    ms = MarketState()
    qm = QuestionMetaEvent(
        venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="qmeta",
        exchange_ts=1, local_recv_ts=1,
        question_idx=42, named_outcome_idxs=[3],
        keys=["class", "underlying", "period", "expiry", "strike"],
        values=["priceBinary", "BTC", "1h", "20260508-1200", "80000"],
    )
    ms.apply(qm)
    s = SettlementEvent(
        venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="#30",
        exchange_ts=2, local_recv_ts=2,
        settled_side_idx=30, settle_price=1.0, settle_ts=2,
    )
    ms.apply(s)
    q = ms.question(42)
    assert q is not None
    assert q.settled is True


def test_recent_returns_for_btc_perp_uses_marks():
    ms = MarketState()
    # 2026-05-21: marks are now bucketed to 1m windows. Space timestamps
    # 60s apart so each MarkEvent lands in its own bucket.
    one_minute_ns = 60 * 1_000_000_000
    for i, px in enumerate([100.0, 100.1, 100.2, 100.05]):
        ms.apply(MarkEvent(
            venue="hyperliquid", product_type=ProductType.PERP, mechanism=Mechanism.CLOB,
            symbol="BTC", exchange_ts=(i + 1) * one_minute_ns,
            local_recv_ts=(i + 1) * one_minute_ns, mark_px=px,
        ))
    rets = ms.recent_returns("BTC", n=3)
    assert len(rets) == 3
    assert all(math.isfinite(r) for r in rets)
    assert ms.last_mark("BTC") == 100.05


def test_marks_bucketed_to_1m_within_bucket_last_wins():
    """High-frequency markPx ticks within a 1m bucket should collapse to
    one entry per bucket (last-tick close). Otherwise the strategy's σ
    formula sees sub-second returns annualized as if they were 60s.
    """
    ms = MarketState()
    one_minute_ns = 60 * 1_000_000_000
    # Bucket 1 (t = 1m + 0..3s): 4 ticks collapse to 1 entry, last wins.
    for i, px in enumerate([100.0, 100.1, 100.2, 100.5]):
        ms.apply(MarkEvent(
            venue="hyperliquid", product_type=ProductType.PERP, mechanism=Mechanism.CLOB,
            symbol="BTC", exchange_ts=one_minute_ns + i, local_recv_ts=one_minute_ns + i,
            mark_px=px,
        ))
    # Bucket 2 (t = 2m): 1 tick.
    ms.apply(MarkEvent(
        venue="hyperliquid", product_type=ProductType.PERP, mechanism=Mechanism.CLOB,
        symbol="BTC", exchange_ts=2 * one_minute_ns, local_recv_ts=2 * one_minute_ns,
        mark_px=101.0,
    ))
    rets = ms.recent_returns("BTC", n=10)
    # Only 2 entries in the buffer → 1 return (between the two buckets).
    assert len(rets) == 1
    # Returned ln(101.0 / 100.5) — last tick of bucket 1 (100.5) is the canonical close.
    assert math.isclose(rets[0], math.log(101.0 / 100.5), rel_tol=1e-9)
    # last_mark still tracks the absolute latest tick, unbucketed.
    assert ms.last_mark("BTC") == 101.0


def test_recent_volume_usd_sums_recent_trades():
    ms = MarketState(volume_window_ns=10_000_000_000)  # 10s window
    now = 100_000_000_000
    for i, sz in enumerate([1.0, 2.0, 3.0]):
        ms.apply(TradeEvent(
            venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol="#30",
            exchange_ts=now + i, local_recv_ts=now + i,
            price=0.95, size=sz, side="buy",
        ))
    # All inside the window
    assert math.isclose(ms.recent_volume_usd("#30", now=now + 5), (1 + 2 + 3) * 0.95)
    # Outside window → 0
    assert ms.recent_volume_usd("#30", now=now + 10**11) == 0.0
