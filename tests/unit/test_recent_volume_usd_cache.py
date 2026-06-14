"""FIX A parity/correctness tests: version-cached recent_volume_usd.

The optimised implementation must be bit-identical (exact float ==) to the
naive re-sum path on every query. We drive a shared MarketState with a
seeded random stream of apply_trade calls interleaved with queries and
compare cached vs. naive at each query point.
"""

from __future__ import annotations

import random
from collections import deque

import pytest

from hlanalysis.marketdata.market_state import MarketState

# One hour rolling window in nanoseconds (matches the constant in market_state.py).
_ONE_HOUR_NS: int = 60 * 60 * 1_000_000_000


# ---------------------------------------------------------------------------
# Naive reference: replicate the pre-optimisation logic on a fresh instance
# so we don't depend on the old implementation still existing.
# ---------------------------------------------------------------------------


def _naive_volume(
    trades: dict[str, deque[tuple[int, float, float]]],
    symbols: list[str],
    now_ns: int,
    volume_window_ns: int,
) -> float:
    """Re-sum without any cache — the old O(window) path."""
    cutoff = now_ns - volume_window_ns
    total = 0.0
    for sym in symbols:
        dq = trades.get(sym)
        if dq is None:
            continue
        # evict in a copy so we don't mutate the reference deque
        active = [(ts, px, sz) for ts, px, sz in dq if ts >= cutoff]
        total += sum(px * sz for _, px, sz in active)
    return total


def _copy_trades(ms: MarketState) -> dict[str, deque[tuple[int, float, float]]]:
    """Snapshot the internal _trades deques (shallow copy — tuples are immutable)."""
    return {sym: deque(dq) for sym, dq in ms._trades.items()}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_symbol_basic():
    """Single symbol: cached result == naive re-sum at each query."""
    rng = random.Random(42)
    ms = MarketState()
    sym = "BTC"

    now_ns = 0
    for i in range(30):
        now_ns += rng.randint(1, 10) * 1_000_000_000
        px = rng.uniform(50_000, 60_000)
        sz = rng.uniform(0.01, 2.0)
        ms.apply_trade(sym, ts_ns=now_ns, price=px, size=sz)

        # query
        query_ns = now_ns + rng.randint(0, 5) * 1_000_000_000
        cached = ms.recent_volume_usd(sym, now_ns=query_ns)
        naive = _naive_volume(_copy_trades(ms), [sym], query_ns, ms._volume_window_ns)
        assert cached == naive, f"Step {i}: cached={cached} naive={naive} differ"


def test_multiple_symbols_in_one_query():
    """Multi-symbol query: sum of per-symbol naive values must equal cached."""
    rng = random.Random(99)
    ms = MarketState()
    symbols = ["BTC", "ETH", "SOL"]

    now_ns = 1_000_000_000
    for i in range(40):
        sym = rng.choice(symbols)
        now_ns += rng.randint(1, 5) * 1_000_000_000
        ms.apply_trade(sym, ts_ns=now_ns, price=rng.uniform(100, 70_000), size=rng.uniform(0.01, 5.0))

        query_ns = now_ns + rng.randint(0, 2) * 1_000_000_000
        cached = ms.recent_volume_usd(symbols, now_ns=query_ns)
        naive = _naive_volume(_copy_trades(ms), symbols, query_ns, ms._volume_window_ns)
        assert cached == naive, f"Step {i}: cached={cached} naive={naive} differ"


def test_eviction_drops_old_trades():
    """Trades older than the volume window must not count."""
    ms = MarketState()
    sym = "BTC"

    # Insert one trade at t=0.
    ms.apply_trade(sym, ts_ns=0, price=100.0, size=1.0)

    # Query at t < window — trade is still live.
    assert ms.recent_volume_usd(sym, now_ns=_ONE_HOUR_NS - 1) == pytest.approx(100.0)

    # Query at t = window + 1ns — trade falls outside the window.
    result = ms.recent_volume_usd(sym, now_ns=_ONE_HOUR_NS + 1)
    assert result == 0.0, f"Expected 0.0 after eviction, got {result}"


def test_cache_invalidation_after_new_trade():
    """After a new trade the cached value updates (cache is invalidated)."""
    ms = MarketState()
    sym = "ETH"

    ms.apply_trade(sym, ts_ns=1_000_000_000, price=2_000.0, size=1.0)
    v1 = ms.recent_volume_usd(sym, now_ns=2_000_000_000)
    assert v1 == pytest.approx(2_000.0)

    # New trade — should increase the total.
    ms.apply_trade(sym, ts_ns=3_000_000_000, price=3_000.0, size=2.0)
    v2 = ms.recent_volume_usd(sym, now_ns=4_000_000_000)
    assert v2 == pytest.approx(2_000.0 + 6_000.0)


def test_no_trades_returns_zero():
    """Querying a symbol with no trades must return 0.0."""
    ms = MarketState()
    assert ms.recent_volume_usd("UNKNOWN", now_ns=1_000_000_000) == 0.0
    assert ms.recent_volume_usd(["A", "B", "C"], now_ns=1_000_000_000) == 0.0


def test_eviction_in_mixed_stream():
    """Eviction interleaved with cache reads: result always equals naive.

    Uses a small volume_window so evictions happen frequently.
    """
    small_window_ns = 10 * 1_000_000_000  # 10 seconds
    ms = MarketState(volume_window_ns=small_window_ns)
    sym = "SOL"

    rng = random.Random(7)
    now_ns = 0
    for i in range(50):
        now_ns += rng.randint(1, 5) * 1_000_000_000
        ms.apply_trade(sym, ts_ns=now_ns, price=rng.uniform(10, 100), size=rng.uniform(0.1, 10.0))

        query_ns = now_ns + rng.randint(0, 3) * 1_000_000_000
        cached = ms.recent_volume_usd(sym, now_ns=query_ns)
        naive = _naive_volume(_copy_trades(ms), [sym], query_ns, small_window_ns)
        assert cached == naive, f"Step {i}: cached={cached} naive={naive} differ"


def test_cache_invalidated_by_eviction_on_read():
    """Eviction triggered inside recent_volume_usd must bump the version and
    recompute so the next query on the same now_ns also reflects the eviction."""
    small_window_ns = 5 * 1_000_000_000  # 5 seconds
    ms = MarketState(volume_window_ns=small_window_ns)
    sym = "XRP"

    # Insert a trade at t=0, then query at t=window+1 to force eviction.
    ms.apply_trade(sym, ts_ns=0, price=1.0, size=100.0)
    ms.apply_trade(sym, ts_ns=2 * 1_000_000_000, price=2.0, size=50.0)

    # Before eviction window: both trades live.
    v_before = ms.recent_volume_usd(sym, now_ns=4 * 1_000_000_000)
    assert v_before == pytest.approx(1.0 * 100.0 + 2.0 * 50.0)

    # After eviction window: first trade falls out (ts=0 < cutoff=1ns with window=5s at now=6s).
    v_after = ms.recent_volume_usd(sym, now_ns=6 * 1_000_000_000)
    assert v_after == pytest.approx(2.0 * 50.0)


def test_multi_symbol_partial_presence():
    """Some queried symbols have trades, others don't — totals correct."""
    ms = MarketState()
    ms.apply_trade("A", ts_ns=1_000_000_000, price=10.0, size=5.0)
    # "B" has no trades.

    result = ms.recent_volume_usd(["A", "B"], now_ns=2_000_000_000)
    assert result == pytest.approx(50.0)
