# tests/unit/test_market_state.py
from __future__ import annotations

import math
from collections import deque

import pytest

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


def test_mark_question_settled_by_idx():
    # Used by the reconciler when it detects a local position vanished from
    # the venue — we need to mark the question settled BEFORE the polled
    # outcomeMeta SettlementEvent arrives so the continuous-checks loop
    # skips the now-silent leg's stale-data check. Idempotent: returns True
    # only on the first transition.
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
    assert ms.question(42).settled is False
    assert ms.mark_question_settled(42) is True
    assert ms.question(42).settled is True
    # Idempotent — second call is a no-op.
    assert ms.mark_question_settled(42) is False
    # Unknown question_idx — no-op (no exception).
    assert ms.mark_question_settled(9999) is False


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


def _mark(symbol: str, px: float, ts: int) -> MarkEvent:
    return MarkEvent(
        venue="hyperliquid", product_type=ProductType.PERP,
        mechanism=Mechanism.CLOB, symbol=symbol,
        exchange_ts=ts, local_recv_ts=ts, mark_px=px,
    )


def test_default_mark_bucket_is_60s():
    """Cadence port: with no registration the bucket period stays 60s, so
    existing deployments are bit-for-bit unchanged."""
    ms = MarketState()
    assert ms.mark_bucket_ns_for("BTC") == 60 * 1_000_000_000
    assert ms.mark_bucket_ns_for("BTCUSDT") == 60 * 1_000_000_000


def test_set_reference_cadence_buckets_per_symbol():
    """Registering BTC at 5s buckets its marks every 5s; an unregistered
    symbol (BTCUSDT) still buckets at the 60s default — HL/PM independence."""
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5)
    assert ms.mark_bucket_ns_for("BTC") == 5 * 1_000_000_000
    assert ms.mark_bucket_ns_for("BTCUSDT") == 60 * 1_000_000_000

    s = 1_000_000_000  # 1s in ns
    # Four BTC ticks 5s apart → four buckets → three returns.
    for i, px in enumerate([100.0, 100.1, 100.2, 100.3]):
        ms.apply(_mark("BTC", px, ts=(i + 1) * 5 * s))
    assert len(ms.recent_returns("BTC", n=10)) == 3

    # Within one 5s bucket, sub-period ticks collapse (last-wins).
    ms2 = MarketState()
    ms2.set_reference_cadence("BTC", sampling_dt_seconds=5)
    base = 100 * s
    for i, px in enumerate([200.0, 200.1, 200.2]):  # t = 100s + 0,1,2s → same bucket
        ms2.apply(_mark("BTC", px, ts=base + i * s))
    ms2.apply(_mark("BTC", 201.0, ts=base + 5 * s))  # next bucket
    rets = ms2.recent_returns("BTC", n=10)
    assert len(rets) == 1
    assert math.isclose(rets[0], math.log(201.0 / 200.2), rel_tol=1e-9)


def test_set_reference_cadence_conflict_raises():
    """The shared mark history for a symbol can only be bucketed one way, so a
    second registration at a different cadence must fail fast (prevents silent
    skew when two slots read the same reference_symbol)."""
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5)
    # Same value is idempotent.
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5)
    with pytest.raises(ValueError, match="conflicting mark-bucket cadence"):
        ms.set_reference_cadence("BTC", sampling_dt_seconds=60)


def test_set_reference_cadence_sizes_history_for_sub_minute():
    """At dt=5s / 3600s lookback the default 256-entry deque is too short
    (~720 bars needed); registration grows it so the σ window isn't truncated
    relative to the backtest's auto-growing buffer."""
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5, lookback_seconds=3600)
    s = 1_000_000_000
    for i in range(800):
        ms.apply(_mark("BTC", 100.0 + 0.001 * i, ts=(i + 1) * 5 * s))
    # Would be capped at 255 returns under the legacy 256-deep deque.
    assert len(ms.recent_returns("BTC", n=10_000)) > 256


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
