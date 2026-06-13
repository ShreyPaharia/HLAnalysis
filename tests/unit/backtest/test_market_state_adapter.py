"""SHR-86 — runner ``MarketState`` thin-adapter guards.

The runner ``MarketState`` is a single-reference adapter over the shared
``marketdata.market_state`` core. These tests pin the two cross-cutting
guards the spec folds into this ticket:

  * Coin-recycling: a leg symbol that gets reused must read **window-bounded**
    volume — a trade older than the 1-hour window must never leak into a later
    read, even for the same symbol (HL recycles ``#NN`` coins across days).
  * Determinism: the state is a pure function of the event sequence — two
    instances fed the identical stream yield byte-identical query outputs (no
    process-salted hashing on the HL path).
"""

from __future__ import annotations

import numpy as np

from hlanalysis.backtest.core.events import BookSnapshot, ReferenceEvent, TradeEvent
from hlanalysis.backtest.runner.market_state import MarketState

_S = 1_000_000_000
_HOUR_NS = 3600 * _S


def _trade(symbol: str, ts_ns: int, price: float, size: float) -> TradeEvent:
    return TradeEvent(ts_ns=ts_ns, symbol=symbol, side="buy", price=price, size=size)


# ---- coin-recycling: window-bounded volume reads -------------------------


def test_recycled_coin_volume_is_window_bounded() -> None:
    """A trade older than the 1h window must not leak into a later read on the
    SAME (recycled) leg symbol."""
    ms = MarketState()
    # Old trade for #30 (a coin that will be "recycled" the next day).
    ms.apply_trade(_trade("#30", ts_ns=1 * _S, price=0.50, size=100.0))
    # ~25h later the same coin symbol is reused with a fresh trade.
    recycled_ts = 25 * _HOUR_NS
    ms.apply_trade(_trade("#30", ts_ns=recycled_ts, price=0.40, size=10.0))

    vol = ms.recent_volume_usd(("#30",), now_ns=recycled_ts)
    # Only the recycled trade counts (0.40 * 10); the day-old one is evicted.
    assert vol == 0.40 * 10.0


def test_volume_zero_once_window_passes_with_no_new_trades() -> None:
    """With no fresh trade, a read past the window boundary returns 0 — the
    stale entry is evicted on read, not carried forward."""
    ms = MarketState()
    ms.apply_trade(_trade("#30", ts_ns=1 * _S, price=0.50, size=100.0))
    assert ms.recent_volume_usd(("#30",), now_ns=1 * _S) == 0.50 * 100.0
    # Two hours later, nothing new → window empty → 0.
    assert ms.recent_volume_usd(("#30",), now_ns=2 * _HOUR_NS) == 0.0


# ---- determinism: pure function of the event sequence --------------------


def _feed(ms: MarketState) -> None:
    for i in range(200):
        ts = (i + 1) * 5 * _S
        close = 80_000.0 + (i % 13) - 0.5 * (i % 7)
        ms.apply_reference(ReferenceEvent(ts, "BTC", close + 5.0, close - 4.0, close, close))
        ms.apply_trade(_trade(("#30", "#31")[i % 2], ts, 0.40 + 0.001 * (i % 9), 1.0 + (i % 5)))
        if i % 4 == 0:
            ms.apply_l2(
                BookSnapshot(
                    ts_ns=ts,
                    symbol="#30",
                    bids=((0.40, 10.0),),
                    asks=((0.41, 12.0),),
                )
            )


def test_identical_event_stream_yields_identical_outputs() -> None:
    a = MarketState()
    b = MarketState()
    _feed(a)
    _feed(b)

    now_ns = 200 * 5 * _S
    for lb in (300, 3600):
        ra, ha = a.recent_returns_and_hl(now_ns=now_ns, lookback_seconds=lb)
        rb, hb = b.recent_returns_and_hl(now_ns=now_ns, lookback_seconds=lb)
        assert np.array_equal(ra, rb)
        assert np.array_equal(ha, hb)
        assert a.recent_volume_usd(("#30", "#31"), now_ns=now_ns) == b.recent_volume_usd(("#30", "#31"), now_ns=now_ns)
    assert a.latest_btc_close() == b.latest_btc_close()


def test_recent_returns_and_hl_matches_separate_calls() -> None:
    """The fused accessor must return exactly what the two single accessors do
    (the runner calls the fused one; tests/strategies call the singles)."""
    ms = MarketState()
    _feed(ms)
    now_ns = 200 * 5 * _S
    rets, hl = ms.recent_returns_and_hl(now_ns=now_ns, lookback_seconds=3600)
    assert np.array_equal(rets, ms.recent_returns(now_ns=now_ns, lookback_seconds=3600))
    assert np.array_equal(hl, ms.recent_hl_bars(now_ns=now_ns, lookback_seconds=3600))
