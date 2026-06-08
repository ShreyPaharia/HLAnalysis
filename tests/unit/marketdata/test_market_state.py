"""Unit tests for the shared MarketState core (SHR-81).

The core must (a) reproduce the backtest MarketState's query outputs
bit-for-bit on a fixed event sequence and (b) match the live engine's
volume-window + OHLC-bucketing semantics, so a later ticket (T6/T7) can swap
it into both contexts. These tests pin both halves: direct-bar parity against
``KlineRingBuffer.slice_window`` (the SHR-66 windowing rule) and tick-bucketing
parity against the live ``engine.market_state.MarketState``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from hlanalysis.marketdata.market_state import (
    BookEvent,
    MarketState,
    ReferenceBarEvent,
    ReferenceTickEvent,
    TradeEvent,
)

S = 1_000_000_000  # one second in ns
HOUR_NS = 3600 * S


# --------------------------------------------------------------------------
# book()
# --------------------------------------------------------------------------

def test_book_none_for_unknown_symbol() -> None:
    ms = MarketState()
    assert ms.book("BTC") is None


def test_book_exposes_top_of_book_and_full_levels() -> None:
    ms = MarketState()
    ms.apply_book(
        "BTC",
        ts_ns=10 * S,
        bids=((100.0, 1.0), (99.0, 2.0)),
        asks=((101.0, 3.0), (102.0, 4.0)),
    )
    b = ms.book("BTC")
    assert b is not None
    assert (b.bid_px, b.bid_sz) == (100.0, 1.0)
    assert (b.ask_px, b.ask_sz) == (101.0, 3.0)
    assert b.bid_levels == ((100.0, 1.0), (99.0, 2.0))
    assert b.ask_levels == ((101.0, 3.0), (102.0, 4.0))
    assert b.last_l2_ts_ns == 10 * S
    assert b.last_trade_ts_ns == 0


def test_trade_updates_last_trade_ts_on_book() -> None:
    ms = MarketState()
    ms.apply_book("BTC", ts_ns=10 * S, bids=((100.0, 1.0),), asks=((101.0, 1.0),))
    ms.apply_trade("BTC", ts_ns=12 * S, price=100.5, size=2.0)
    b = ms.book("BTC")
    assert b is not None
    assert b.last_trade_ts_ns == 12 * S
    # The book top is unchanged by a trade.
    assert b.bid_px == 100.0


# --------------------------------------------------------------------------
# recent_returns / recent_hl_bars / recent_returns_and_hl
# (bit-identical to KlineRingBuffer.slice_window — the SHR-66 rule)
# --------------------------------------------------------------------------

def _reference_bars() -> list[tuple[int, float, float, float]]:
    # (ts_ns, high, low, close) — irregular spacing, a couple of wicks.
    return [
        (1 * S, 100.0, 99.0, 100.0),
        (61 * S, 101.0, 99.5, 100.5),
        (121 * S, 102.0, 100.0, 101.0),
        (181 * S, 101.5, 100.5, 100.8),
        (241 * S, 103.0, 100.7, 102.2),
    ]


def _kline_buffer(bars):
    from hlanalysis.strategy._numba.returns_buffer import KlineRingBuffer

    buf = KlineRingBuffer()
    for ts, h, l, c in bars:
        buf.append(ts_ns=ts, high=h, low=l, close=c)
    return buf


def test_recent_returns_matches_klineringbuffer() -> None:
    bars = _reference_bars()
    buf = _kline_buffer(bars)
    ms = MarketState()
    for ts, h, l, c in bars:
        ms.apply_reference_bar("BTC", ts_ns=ts, high=h, low=l, close=c)

    now = 241 * S
    lookback = 600  # covers all five bars
    want, _ = buf.slice_window(now_ns=now, lookback_seconds=lookback)
    got = ms.recent_returns("BTC", now_ns=now, lookback_seconds=lookback)
    np.testing.assert_array_equal(got, want)


def test_recent_returns_window_drops_old_bars() -> None:
    bars = _reference_bars()
    buf = _kline_buffer(bars)
    ms = MarketState()
    for ts, h, l, c in bars:
        ms.apply_reference_bar("BTC", ts_ns=ts, high=h, low=l, close=c)

    now = 241 * S
    lookback = 120  # only the most recent few bars survive the cutoff
    want, _ = buf.slice_window(now_ns=now, lookback_seconds=lookback)
    got = ms.recent_returns("BTC", now_ns=now, lookback_seconds=lookback)
    np.testing.assert_array_equal(got, want)
    # And it must actually be a strict subset (the window is doing something).
    assert got.size < len(bars) - 1


def test_recent_hl_bars_matches_klineringbuffer() -> None:
    bars = _reference_bars()
    buf = _kline_buffer(bars)
    ms = MarketState()
    for ts, h, l, c in bars:
        ms.apply_reference_bar("BTC", ts_ns=ts, high=h, low=l, close=c)

    now = 241 * S
    lookback = 600
    _, want = buf.slice_window(now_ns=now, lookback_seconds=lookback)
    got = ms.recent_hl_bars("BTC", now_ns=now, lookback_seconds=lookback)
    assert got.shape == want.shape
    np.testing.assert_array_equal(got, want)


def test_recent_returns_and_hl_matches_separate_calls() -> None:
    bars = _reference_bars()
    ms = MarketState()
    for ts, h, l, c in bars:
        ms.apply_reference_bar("BTC", ts_ns=ts, high=h, low=l, close=c)

    now = 241 * S
    lookback = 600
    rets, hl = ms.recent_returns_and_hl("BTC", now_ns=now, lookback_seconds=lookback)
    np.testing.assert_array_equal(
        rets, ms.recent_returns("BTC", now_ns=now, lookback_seconds=lookback)
    )
    np.testing.assert_array_equal(
        hl, ms.recent_hl_bars("BTC", now_ns=now, lookback_seconds=lookback)
    )


def test_recent_returns_empty_for_unknown_symbol() -> None:
    ms = MarketState()
    got = ms.recent_returns("BTC", now_ns=10 * S, lookback_seconds=600)
    assert isinstance(got, np.ndarray)
    assert got.size == 0


# --------------------------------------------------------------------------
# last_mark
# --------------------------------------------------------------------------

def test_last_mark_tracks_latest_bar_close() -> None:
    ms = MarketState()
    assert ms.last_mark("BTC") is None
    ms.apply_reference_bar("BTC", ts_ns=1 * S, high=100.0, low=99.0, close=100.0)
    ms.apply_reference_bar("BTC", ts_ns=61 * S, high=101.0, low=99.5, close=100.5)
    assert ms.last_mark("BTC") == 100.5


def test_last_mark_tracks_latest_tick() -> None:
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=60)
    ms.apply_reference_tick("BTC", ts_ns=1 * S, price=100.0)
    ms.apply_reference_tick("BTC", ts_ns=2 * S, price=100.7)
    assert ms.last_mark("BTC") == 100.7


# --------------------------------------------------------------------------
# recent_volume_usd — 1h window, eviction on insert AND on read
# --------------------------------------------------------------------------

def test_recent_volume_usd_sums_price_times_size() -> None:
    ms = MarketState()
    ms.apply_trade("YES", ts_ns=10 * S, price=0.5, size=100.0)
    ms.apply_trade("YES", ts_ns=20 * S, price=0.6, size=50.0)
    assert ms.recent_volume_usd("YES", now_ns=30 * S) == pytest.approx(0.5 * 100 + 0.6 * 50)


def test_recent_volume_usd_evicts_on_read() -> None:
    ms = MarketState()
    ms.apply_trade("YES", ts_ns=10 * S, price=0.5, size=100.0)
    # Read more than an hour after the only trade: it has aged out of the window.
    assert ms.recent_volume_usd("YES", now_ns=10 * S + HOUR_NS + 1) == 0.0


def test_recent_volume_usd_evicts_on_insert() -> None:
    ms = MarketState()
    ms.apply_trade("YES", ts_ns=10 * S, price=0.5, size=100.0)
    # A trade an hour+ later evicts the first on insert; reading exactly at the
    # new trade's ts (so it is in-window) must see only the new one.
    t2 = 10 * S + HOUR_NS + 5 * S
    ms.apply_trade("YES", ts_ns=t2, price=0.7, size=10.0)
    assert ms.recent_volume_usd("YES", now_ns=t2) == pytest.approx(0.7 * 10.0)


def test_recent_volume_usd_sums_across_legs() -> None:
    ms = MarketState()
    ms.apply_trade("YES", ts_ns=10 * S, price=0.5, size=100.0)
    ms.apply_trade("NO", ts_ns=11 * S, price=0.5, size=40.0)
    got = ms.recent_volume_usd(("YES", "NO"), now_ns=20 * S)
    assert got == pytest.approx(0.5 * 100 + 0.5 * 40)


def test_recent_volume_usd_unknown_symbol_is_zero() -> None:
    ms = MarketState()
    assert ms.recent_volume_usd("NOPE", now_ns=10 * S) == 0.0
    assert ms.recent_volume_usd(("A", "B"), now_ns=10 * S) == 0.0


# --------------------------------------------------------------------------
# σ estimators — parkinson / stdev / bipower
# --------------------------------------------------------------------------

def test_sigma_stdev_matches_estimator() -> None:
    from hlanalysis.strategy.vol import sample_std_returns

    bars = _reference_bars()
    ms = MarketState()
    for ts, h, l, c in bars:
        ms.apply_reference_bar("BTC", ts_ns=ts, high=h, low=l, close=c)
    now, lookback = 241 * S, 600
    rets = ms.recent_returns("BTC", now_ns=now, lookback_seconds=lookback)
    assert ms.sigma("BTC", estimator="stdev", now_ns=now, lookback_seconds=lookback) == (
        pytest.approx(sample_std_returns(rets))
    )


def test_sigma_bipower_matches_estimator() -> None:
    from hlanalysis.strategy.vol import bipower_variation_sigma

    bars = _reference_bars()
    ms = MarketState()
    for ts, h, l, c in bars:
        ms.apply_reference_bar("BTC", ts_ns=ts, high=h, low=l, close=c)
    now, lookback = 241 * S, 600
    rets = ms.recent_returns("BTC", now_ns=now, lookback_seconds=lookback)
    assert ms.sigma("BTC", estimator="bipower", now_ns=now, lookback_seconds=lookback) == (
        pytest.approx(bipower_variation_sigma(rets))
    )


def test_sigma_parkinson_matches_estimator() -> None:
    from hlanalysis.strategy._numba.vol import parkinson_sigma_window

    bars = _reference_bars()
    ms = MarketState()
    for ts, h, l, c in bars:
        ms.apply_reference_bar("BTC", ts_ns=ts, high=h, low=l, close=c)
    now, lookback = 241 * S, 600
    hl = ms.recent_hl_bars("BTC", now_ns=now, lookback_seconds=lookback)
    want = parkinson_sigma_window(
        np.ascontiguousarray(hl[:, 0]), np.ascontiguousarray(hl[:, 1]), 0.0
    )
    got = ms.sigma(
        "BTC", estimator="parkinson", now_ns=now, lookback_seconds=lookback,
    )
    assert got == pytest.approx(want)


def test_sigma_unknown_estimator_raises() -> None:
    ms = MarketState()
    ms.apply_reference_bar("BTC", ts_ns=1 * S, high=100.0, low=99.0, close=100.0)
    with pytest.raises(ValueError):
        ms.sigma("BTC", estimator="nope", now_ns=2 * S, lookback_seconds=600)


# --------------------------------------------------------------------------
# tick-bucketing — coalescing within a dt bucket (engine semantics, ohlc.py)
# --------------------------------------------------------------------------

def test_tick_bucketing_coalesces_within_bucket() -> None:
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=60)
    # Three sub-minute ticks all in bucket 0, then one in bucket 1.
    ms.apply_reference_tick("BTC", ts_ns=1 * S, price=100.0)
    ms.apply_reference_tick("BTC", ts_ns=2 * S, price=103.0)  # high
    ms.apply_reference_tick("BTC", ts_ns=3 * S, price=98.0)   # low, close
    ms.apply_reference_tick("BTC", ts_ns=61 * S, price=99.0)
    now = 61 * S
    hl = ms.recent_hl_bars("BTC", now_ns=now, lookback_seconds=600)
    # Two buckets → two bars. First bar aggregates the three ticks.
    assert hl.shape == (2, 2)
    np.testing.assert_array_equal(hl[0], np.array([103.0, 98.0]))
    rets = ms.recent_returns("BTC", now_ns=now, lookback_seconds=600)
    # One return between the two bucket closes (98.0 -> 99.0).
    assert rets.shape == (1,)
    assert rets[0] == pytest.approx(math.log(99.0 / 98.0))


# --------------------------------------------------------------------------
# determinism — pure function of the event sequence
# --------------------------------------------------------------------------

def test_determinism_same_events_same_outputs() -> None:
    bars = _reference_bars()
    now, lookback = 241 * S, 600

    def build_and_query():
        ms = MarketState()
        ms.apply_trade("YES", ts_ns=5 * S, price=0.5, size=100.0)
        for ts, h, l, c in bars:
            ms.apply_reference_bar("BTC", ts_ns=ts, high=h, low=l, close=c)
        return (
            ms.recent_returns("BTC", now_ns=now, lookback_seconds=lookback),
            ms.recent_hl_bars("BTC", now_ns=now, lookback_seconds=lookback),
            ms.recent_volume_usd("YES", now_ns=now),
            ms.last_mark("BTC"),
        )

    r1, h1, v1, m1 = build_and_query()
    r2, h2, v2, m2 = build_and_query()
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(h1, h2)
    assert v1 == v2
    assert m1 == m2


# --------------------------------------------------------------------------
# apply() dispatch over the shared event structs
# --------------------------------------------------------------------------

def test_apply_dispatches_all_event_types() -> None:
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=60)
    ms.apply(BookEvent(ts_ns=1 * S, symbol="YES", bids=((0.4, 10.0),), asks=((0.6, 10.0),)))
    ms.apply(TradeEvent(ts_ns=2 * S, symbol="YES", price=0.5, size=100.0))
    ms.apply(ReferenceTickEvent(ts_ns=1 * S, symbol="BTC", price=100.0))
    ms.apply(ReferenceBarEvent(ts_ns=61 * S, symbol="BTC", high=101.0, low=99.0, close=100.5))
    assert ms.book("YES") is not None
    assert ms.recent_volume_usd("YES", now_ns=3 * S) == pytest.approx(50.0)
    assert ms.last_mark("BTC") == 100.5


# --------------------------------------------------------------------------
# parity with the live engine MarketState (tick-bucketing path)
# --------------------------------------------------------------------------

def test_parity_with_engine_marketstate_tick_path() -> None:
    from hlanalysis.engine.market_state import MarketState as EngineMS
    from hlanalysis.events import MarkEvent, Mechanism, ProductType

    ticks = [
        (1 * S, 100.0), (2 * S, 100.4), (3 * S, 99.8),
        (61 * S, 100.9), (75 * S, 101.3),
        (121 * S, 100.2), (181 * S, 102.0), (245 * S, 101.1),
    ]
    eng = EngineMS()
    core = MarketState()
    core.set_reference_cadence("BTC", sampling_dt_seconds=60)
    for ts, px in ticks:
        eng.apply(MarkEvent(
            venue="hl", product_type=ProductType.PERP, mechanism=Mechanism.CLOB,
            symbol="BTC", exchange_ts=ts, local_recv_ts=ts, mark_px=px,
        ))
        core.apply_reference_tick("BTC", ts_ns=ts, price=px)

    now = 300 * S
    lookback = 600
    eng_rets = np.array(
        eng.recent_returns("BTC", 999, now_ns=now, lookback_seconds=lookback),
        dtype=np.float64,
    )
    core_rets = core.recent_returns("BTC", now_ns=now, lookback_seconds=lookback)
    np.testing.assert_array_equal(core_rets, eng_rets)

    eng_hl = np.array(
        eng.recent_hl_bars("BTC", 999, now_ns=now, lookback_seconds=lookback),
        dtype=np.float64,
    ).reshape(-1, 2)
    core_hl = core.recent_hl_bars("BTC", now_ns=now, lookback_seconds=lookback)
    np.testing.assert_array_equal(core_hl, eng_hl)

    assert core.last_mark("BTC") == eng.last_mark("BTC")


def test_parity_with_engine_volume_window() -> None:
    from hlanalysis.engine.market_state import MarketState as EngineMS
    from hlanalysis.events import Mechanism, ProductType
    from hlanalysis.events import TradeEvent as EngTrade

    trades = [
        (10 * S, 0.5, 100.0),
        (10 * S + HOUR_NS - S, 0.6, 40.0),  # in window
        (10 * S + HOUR_NS + 5 * S, 0.7, 20.0),  # evicts the first
    ]
    eng = EngineMS()
    core = MarketState()
    for ts, px, sz in trades:
        eng.apply(EngTrade(
            venue="pm", product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol="YES",
            exchange_ts=ts, local_recv_ts=ts, price=px, size=sz, side="buy",
        ))
        core.apply_trade("YES", ts_ns=ts, price=px, size=sz)

    now = trades[-1][0]
    assert core.recent_volume_usd("YES", now_ns=now) == pytest.approx(
        eng.recent_volume_usd("YES", now=now)
    )


# --------------------------------------------------------------------------
# parity with the backtest runner MarketState (pre-bucketed bar path)
# --------------------------------------------------------------------------

def test_parity_with_backtest_marketstate_bar_path() -> None:
    from hlanalysis.backtest.core.events import BookSnapshot, ReferenceEvent
    from hlanalysis.backtest.core.events import TradeEvent as BtTrade
    from hlanalysis.backtest.runner.market_state import MarketState as BtMS

    bars = _reference_bars()
    bt = BtMS()
    core = MarketState()
    for ts, h, l, c in bars:
        bt.apply_reference(ReferenceEvent(ts_ns=ts, symbol="BTC", high=h, low=l, close=c))
        core.apply_reference_bar("BTC", ts_ns=ts, high=h, low=l, close=c)

    # Trades for the volume gate.
    bt.apply_trade(BtTrade(ts_ns=5 * S, symbol="YES", side="buy", price=0.5, size=100.0))
    core.apply_trade("YES", ts_ns=5 * S, price=0.5, size=100.0)

    # Book.
    snap = BookSnapshot(ts_ns=6 * S, symbol="YES", bids=((0.4, 10.0),), asks=((0.6, 12.0),))
    bt.apply_l2(snap)
    core.apply_book("YES", ts_ns=6 * S, bids=((0.4, 10.0),), asks=((0.6, 12.0),))

    now, lookback = 241 * S, 600
    np.testing.assert_array_equal(
        core.recent_returns("BTC", now_ns=now, lookback_seconds=lookback),
        bt.recent_returns(now_ns=now, lookback_seconds=lookback),
    )
    np.testing.assert_array_equal(
        core.recent_hl_bars("BTC", now_ns=now, lookback_seconds=lookback),
        bt.recent_hl_bars(now_ns=now, lookback_seconds=lookback),
    )
    c_rets, c_hl = core.recent_returns_and_hl("BTC", now_ns=now, lookback_seconds=lookback)
    b_rets, b_hl = bt.recent_returns_and_hl(now_ns=now, lookback_seconds=lookback)
    np.testing.assert_array_equal(c_rets, b_rets)
    np.testing.assert_array_equal(c_hl, b_hl)

    assert core.recent_volume_usd(("YES",), now_ns=now) == pytest.approx(
        bt.recent_volume_usd(("YES",), now_ns=now)
    )
    assert core.last_mark("BTC") == bt.latest_btc_close()

    cb, bb = core.book("YES"), bt.book("YES")
    assert (cb.bid_px, cb.ask_px, cb.bid_sz, cb.ask_sz) == (
        bb.bid_px, bb.ask_px, bb.bid_sz, bb.ask_sz
    )
    assert cb.last_l2_ts_ns == bb.last_l2_ts_ns
