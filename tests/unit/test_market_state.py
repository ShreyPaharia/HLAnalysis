# tests/unit/test_market_state.py
from __future__ import annotations

import math
from collections import deque

import pytest

from hlanalysis.engine.market_state import MarketState
from hlanalysis.events import (
    BboEvent,
    BookSnapshotEvent,
    MarkEvent,
    Mechanism,
    ProductType,
    QuestionMetaEvent,
    SettlementEvent,
    TradeEvent,
)


def _bbo(symbol: str, bid: float, ask: float, ts: int = 1) -> BboEvent:
    return BboEvent(
        venue="hyperliquid",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol=symbol,
        exchange_ts=ts,
        local_recv_ts=ts,
        bid_px=bid,
        bid_sz=10.0,
        ask_px=ask,
        ask_sz=10.0,
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
        venue="hyperliquid",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="qmeta",
        exchange_ts=1,
        local_recv_ts=1,
        question_idx=42,
        named_outcome_idxs=[3],
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


def test_hl_question_records_venue():
    # QuestionView must carry the originating venue so strategy slots can be
    # scoped to one venue (HL slots must not match PM questions and vice versa).
    ms = MarketState()
    ms.apply(
        QuestionMetaEvent(
            venue="hyperliquid",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="qmeta",
            exchange_ts=1,
            local_recv_ts=1,
            question_idx=42,
            named_outcome_idxs=[3],
            keys=["class", "underlying", "period", "expiry", "strike"],
            values=["priceBinary", "BTC", "1h", "20260508-1200", "80000"],
        )
    )
    assert ms.question(42).venue == "hyperliquid"


def test_pm_question_uses_clob_token_ids_as_leg_symbols():
    # Polymarket questions carry the real ERC-1155 CLOB token ids in
    # yes_token_id / no_token_id (the same ids the PM book/trade WS frames are
    # keyed by). The engine must use those as the leg symbols so (a) books
    # match the question's legs and (b) live orders submit a valid token id.
    # The HL "#{10*o+s}" coin convention must NOT be applied to PM questions.
    yes_t = "71321045679252212594626385532706912750332728571942532289631379312455583992563"
    no_t = "52114319501245915516055106046884209969926127482827954674443846427813813222426"
    ms = MarketState()
    ms.apply(
        QuestionMetaEvent(
            venue="polymarket",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol=yes_t,
            exchange_ts=1,
            local_recv_ts=1,
            question_idx=1000126,
            named_outcome_idxs=[0, 1],
            keys=["class", "underlying", "yes_token_id", "no_token_id", "expiry", "series_slug"],
            values=["priceBinary", "BTC", yes_t, no_t, "20260601-1200", "btc-up-or-down-daily"],
        )
    )
    q = ms.question(1000126)
    assert q is not None
    assert q.venue == "polymarket"
    assert q.yes_symbol == yes_t
    assert q.no_symbol == no_t
    assert q.leg_symbols == (yes_t, no_t)


def _pm_updown_meta(qidx: int, strike_ref_ts_ns: int) -> QuestionMetaEvent:
    return QuestionMetaEvent(
        venue="polymarket",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="YES_TOKEN",
        exchange_ts=1,
        local_recv_ts=1,
        question_idx=qidx,
        named_outcome_idxs=[0, 1],
        # No "strike"/"targetPrice" — up/down markets resolve vs a reference
        # candle, carried as strike_ref_ts_ns.
        keys=["class", "underlying", "yes_token_id", "no_token_id", "series_slug", "strike_ref_ts_ns"],
        values=["priceBinary", "BTC", "YES_TOKEN", "NO_TOKEN", "btc-up-or-down-daily", str(strike_ref_ts_ns)],
    )


def _mark(symbol: str, px: float, ts: int) -> MarkEvent:
    return MarkEvent(
        venue="binance",
        product_type=ProductType.PERP,
        mechanism=Mechanism.CLOB,
        symbol=symbol,
        exchange_ts=ts,
        local_recv_ts=ts,
        mark_px=px,
    )


def test_last_mark_ts_tracks_latest_reference_tick():
    """SHR-60: MarketState must expose the timestamp of the latest reference
    tick so the risk gate can detect a stale reference feed."""
    ms = MarketState()
    assert ms.last_mark_ts("BTC") is None
    ms.apply(_mark("BTC", 100.0, ts=1_000))
    ms.apply(_mark("BTC", 101.0, ts=2_000))
    assert ms.last_mark_ts("BTC") == 2_000


def test_pm_updown_question_has_nan_strike_until_captured():
    # PM up/down markets carry no numeric strike; QuestionView.strike is NaN
    # so the strategy can't price it until the open is captured.
    ms = MarketState()
    ms.apply(_pm_updown_meta(1000126, strike_ref_ts_ns=5_000_000_000))
    assert math.isnan(ms.question(1000126).strike)


def test_set_question_strike_stamps_only_when_unset():
    # Used to reload a persisted open-strike after a restart: stamp when the
    # strike is still NaN, but never clobber a strike already known.
    ref_ts = 1_700_000_000_000_000_000
    ms = MarketState()
    ms.apply(_pm_updown_meta(1000126, strike_ref_ts_ns=ref_ts))
    assert ms.set_question_strike(1000126, 73_500.0) is True
    assert ms.question(1000126).strike == 73_500.0
    # No-op when a real strike is already set (returns False, keeps value).
    assert ms.set_question_strike(1000126, 99_999.0) is False
    assert ms.question(1000126).strike == 73_500.0
    # Unknown question_idx — no-op.
    assert ms.set_question_strike(424242, 1.0) is False


def test_mark_question_settled_by_idx():
    # Used by the reconciler when it detects a local position vanished from
    # the venue — we need to mark the question settled BEFORE the polled
    # outcomeMeta SettlementEvent arrives so the continuous-checks loop
    # skips the now-silent leg's stale-data check. Idempotent: returns True
    # only on the first transition.
    ms = MarketState()
    qm = QuestionMetaEvent(
        venue="hyperliquid",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="qmeta",
        exchange_ts=1,
        local_recv_ts=1,
        question_idx=42,
        named_outcome_idxs=[3],
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
        venue="hyperliquid",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="qmeta",
        exchange_ts=1,
        local_recv_ts=1,
        question_idx=42,
        named_outcome_idxs=[3],
        keys=["class", "underlying", "period", "expiry", "strike"],
        values=["priceBinary", "BTC", "1h", "20260508-1200", "80000"],
    )
    ms.apply(qm)
    s = SettlementEvent(
        venue="hyperliquid",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="#30",
        exchange_ts=2,
        local_recv_ts=2,
        settled_side_idx=30,
        settle_price=1.0,
        settle_ts=2,
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
        ms.apply(
            MarkEvent(
                venue="hyperliquid",
                product_type=ProductType.PERP,
                mechanism=Mechanism.CLOB,
                symbol="BTC",
                exchange_ts=(i + 1) * one_minute_ns,
                local_recv_ts=(i + 1) * one_minute_ns,
                mark_px=px,
            )
        )
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
        ms.apply(
            MarkEvent(
                venue="hyperliquid",
                product_type=ProductType.PERP,
                mechanism=Mechanism.CLOB,
                symbol="BTC",
                exchange_ts=one_minute_ns + i,
                local_recv_ts=one_minute_ns + i,
                mark_px=px,
            )
        )
    # Bucket 2 (t = 2m): 1 tick.
    ms.apply(
        MarkEvent(
            venue="hyperliquid",
            product_type=ProductType.PERP,
            mechanism=Mechanism.CLOB,
            symbol="BTC",
            exchange_ts=2 * one_minute_ns,
            local_recv_ts=2 * one_minute_ns,
            mark_px=101.0,
        )
    )
    rets = ms.recent_returns("BTC", n=10)
    # Only 2 entries in the buffer → 1 return (between the two buckets).
    assert len(rets) == 1
    # Returned ln(101.0 / 100.5) — last tick of bucket 1 (100.5) is the canonical close.
    assert math.isclose(rets[0], math.log(101.0 / 100.5), rel_tol=1e-9)
    # last_mark still tracks the absolute latest tick, unbucketed.
    assert ms.last_mark("BTC") == 101.0


def _mark(symbol: str, px: float, ts: int) -> MarkEvent:
    return MarkEvent(
        venue="hyperliquid",
        product_type=ProductType.PERP,
        mechanism=Mechanism.CLOB,
        symbol=symbol,
        exchange_ts=ts,
        local_recv_ts=ts,
        mark_px=px,
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


def test_set_reference_cadence_accumulates_multiple_cadences() -> None:
    """A symbol may carry multiple cadences — one tick stream fans into each.
    (The old single-cadence conflict-guard was removed by the (symbol, dt)
    refactor; v31 buckets dt=2 and v31 binary dt=5 share the BTC feed.)"""
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5)
    ms.set_reference_cadence("BTC", sampling_dt_seconds=2)  # no raise
    # Both cadences are actually registered (not just resolvable by explicit dt,
    # which holds unconditionally); first registered is the dt-less default.
    assert ms._cadences_by_symbol["BTC"] == [5 * 1_000_000_000, 2 * 1_000_000_000]
    assert ms.mark_bucket_ns_for("BTC") == 5 * 1_000_000_000


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
        ms.apply(
            TradeEvent(
                venue="hyperliquid",
                product_type=ProductType.PREDICTION_BINARY,
                mechanism=Mechanism.CLOB,
                symbol="#30",
                exchange_ts=now + i,
                local_recv_ts=now + i,
                price=0.95,
                size=sz,
                side="buy",
            )
        )
    # All inside the window
    assert math.isclose(ms.recent_volume_usd("#30", now=now + 5), (1 + 2 + 3) * 0.95)
    # Outside window → 0
    assert ms.recent_volume_usd("#30", now=now + 10**11) == 0.0


# ---- per-bucket OHLC + recent_hl_bars (Part A: dormant-Parkinson fix) -------


def test_recent_hl_bars_empty_when_no_marks():
    ms = MarketState()
    assert ms.recent_hl_bars("BTC", n=10) == ()


def test_recent_hl_bars_mark_fed_tracks_bucket_extremes():
    """Within one mark bucket, the bar's (high, low) is max/min of the marks
    and the close is the last tick. recent_hl_bars returns (high, low)."""
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5)
    base = 100 * 1_000_000_000
    # One 5s bucket: ticks swing 100 → 102 → 98 → 101 (close=101, H=102, L=98).
    for i, px in enumerate([100.0, 102.0, 98.0, 101.0]):
        ms.apply(_mark("BTC", px, ts=base + i))
    hl = ms.recent_hl_bars("BTC", n=10)
    assert hl == ((102.0, 98.0),)
    # close (last tick) drives recent_returns, not the extremes.
    assert ms.last_mark("BTC") == 101.0


def test_recent_hl_bars_one_bar_per_bucket():
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5)
    s = 1_000_000_000
    # Three distinct 5s buckets, each with its own H/L range.
    for b, (lo, hi, close) in enumerate([(99.0, 101.0, 100.0), (100.0, 103.0, 102.0), (101.0, 104.0, 103.0)]):
        base = (b + 1) * 5 * s
        ms.apply(_mark("BTC", close, ts=base))  # open
        ms.apply(_mark("BTC", hi, ts=base + 1))  # high
        ms.apply(_mark("BTC", lo, ts=base + 2))  # low
        ms.apply(_mark("BTC", close, ts=base + 3))  # close
    hl = ms.recent_hl_bars("BTC", n=10)
    assert hl == ((101.0, 99.0), (103.0, 100.0), (104.0, 101.0))
    # n caps to the last n bars.
    assert ms.recent_hl_bars("BTC", n=2) == ((103.0, 100.0), (104.0, 101.0))


def test_recent_returns_still_close_to_close_after_ohlc():
    """OHLC tracking must not change the close-to-close return series — the
    legacy stdev path stays bit-identical."""
    ms = MarketState()
    one_minute_ns = 60 * 1_000_000_000
    for i, px in enumerate([100.0, 100.1, 100.2, 100.05]):
        ms.apply(_mark("BTC", px, ts=(i + 1) * one_minute_ns))
    rets = ms.recent_returns("BTC", n=3)
    assert len(rets) == 3
    assert math.isclose(rets[0], math.log(100.1 / 100.0), rel_tol=1e-12)
    assert math.isclose(rets[2], math.log(100.05 / 100.2), rel_tol=1e-12)


# ---- per-symbol σ source: mark | bbo (Part B) ------------------------------


def test_default_reference_source_is_mark_bbo_does_not_feed_ohlc():
    """Without registration a symbol is mark-sourced: a BboEvent updates the
    top-of-book but does NOT feed the σ/OHLC machinery or last_mark."""
    ms = MarketState()
    ms.apply(_bbo("BTCUSDT", 100.0, 102.0, ts=1_000_000_000))
    assert ms.recent_hl_bars("BTCUSDT", n=10) == ()
    assert ms.last_mark("BTCUSDT") is None
    # book still tracked
    assert ms.book("BTCUSDT") is not None


def test_set_reference_source_bbo_feeds_ohlc_from_mid():
    """source=bbo: BboEvents feed the per-bucket OHLC machinery using
    mid=(bid+ask)/2; last_mark returns the latest mid."""
    ms = MarketState()
    ms.set_reference_cadence("BTCUSDT", sampling_dt_seconds=5)
    ms.set_reference_source("BTCUSDT", "bbo")
    base = 100 * 1_000_000_000
    # mids: 100, 102, 98, 101 within one 5s bucket.
    for i, (bid, ask) in enumerate([(99.0, 101.0), (101.0, 103.0), (97.0, 99.0), (100.0, 102.0)]):
        ms.apply(_bbo("BTCUSDT", bid, ask, ts=base + i))
    assert ms.recent_hl_bars("BTCUSDT", n=10) == ((102.0, 98.0),)
    assert ms.last_mark("BTCUSDT") == 101.0  # last mid


def test_bbo_sourced_symbol_ignores_mark_events():
    """When a symbol is bbo-sourced, MarkEvents for it must NOT touch the
    reference price or OHLC (the σ source is the dense BBO mid)."""
    ms = MarketState()
    ms.set_reference_cadence("BTCUSDT", sampling_dt_seconds=5)
    ms.set_reference_source("BTCUSDT", "bbo")
    base = 100 * 1_000_000_000
    ms.apply(_bbo("BTCUSDT", 99.0, 101.0, ts=base))  # mid 100
    ms.apply(_mark("BTCUSDT", 500.0, ts=base + 1))  # must be ignored
    assert ms.last_mark("BTCUSDT") == 100.0
    assert ms.recent_hl_bars("BTCUSDT", n=10) == ((100.0, 100.0),)


def test_set_reference_source_conflict_raises():
    """Two slots sharing a reference symbol must agree on the σ source — the
    one shared OHLC history can only be fed one way."""
    ms = MarketState()
    ms.set_reference_source("BTCUSDT", "bbo")
    ms.set_reference_source("BTCUSDT", "bbo")  # idempotent
    with pytest.raises(ValueError, match="conflicting reference source"):
        ms.set_reference_source("BTCUSDT", "mark")


def test_set_reference_source_rejects_unknown():
    ms = MarketState()
    with pytest.raises(ValueError, match="reference source"):
        ms.set_reference_source("BTCUSDT", "oracle")


def test_reference_source_for_reports_default_and_override():
    ms = MarketState()
    assert ms.reference_source_for("BTC") == "mark"
    ms.set_reference_source("BTCUSDT", "bbo")
    assert ms.reference_source_for("BTCUSDT") == "bbo"
    assert ms.reference_source_for("BTC") == "mark"


def test_bbo_mid_ohlc_matches_load_binance_bbo_reference(tmp_path):
    """Parity: BTCUSDT bbo-mid OHLC fed live must match the backtest
    `_load_binance_bbo_reference` bar-for-bar on the same tick sequence."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from hlanalysis.backtest.data.polymarket import PolymarketDataSource

    s = 1_000_000_000
    start_ns = 100 * s
    end_ns = 140 * s
    # Build a tick sequence spanning several 5s buckets with intra-bucket range.
    ticks: list[tuple[int, float, float]] = []
    ts = start_ns
    i = 0
    while ts < end_ns:
        bid = 80_000.0 + (i % 7) * 3.0
        ask = bid + 2.0
        ticks.append((ts, bid, ask))
        ts += 1_300_000_000  # 1.3s spacing → multiple ticks per 5s bucket
        i += 1

    # Write the recorded binance perp BBO partition the loader reads.
    from datetime import datetime, timezone

    date_str = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).date().isoformat()
    part = (
        tmp_path
        / "venue=binance"
        / "product_type=perp"
        / "mechanism=clob"
        / "event=bbo"
        / "symbol=BTCUSDT"
        / f"date={date_str}"
    )
    part.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "exchange_ts": [t[0] for t in ticks],
                "bid_px": [t[1] for t in ticks],
                "ask_px": [t[2] for t in ticks],
            }
        ),
        part / "ticks.parquet",
    )

    ds = PolymarketDataSource(
        cache_root=tmp_path / "sim",
        reference_symbol="BTC",
        reference_source="binance_bbo",
        reference_resample_seconds=5,
        binance_data_root=tmp_path,
    )
    ref_events = ds._load_binance_bbo_reference(start_ns, end_ns)
    assert ref_events, "fixture produced no reference bars"
    expected_hl = tuple((e.high, e.low) for e in ref_events)
    expected_closes = [e.close for e in ref_events]

    # Feed the SAME ticks through the live MarketState as a bbo-sourced symbol.
    ms = MarketState()
    ms.set_reference_cadence("BTCUSDT", sampling_dt_seconds=5)
    ms.set_reference_source("BTCUSDT", "bbo")
    for t, bid, ask in ticks:
        ms.apply(_bbo("BTCUSDT", bid, ask, ts=t))

    assert ms.recent_hl_bars("BTCUSDT", n=10_000) == expected_hl
    # Returns are derived from bucket closes — compare to backtest closes.
    live_rets = ms.recent_returns("BTCUSDT", n=10_000)
    expected_rets = tuple(math.log(expected_closes[i] / expected_closes[i - 1]) for i in range(1, len(expected_closes)))
    assert len(live_rets) == len(expected_rets)
    for a, b in zip(live_rets, expected_rets, strict=True):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
