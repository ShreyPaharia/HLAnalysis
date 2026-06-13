"""Test for finding #55: reference-bar bucketing boundary guard.

Bug: resample_ohlc in ohlc.py has no guard for the case where the buffer
history is empty and the first tick falls exactly on a bucket boundary.
The live _OhlcBuffer.ingest_tick guards against a degenerate state
(self._len == 0 with _last_bucket set), but the batch resample_ohlc
generator has no equivalent.

Additionally: when resample_ohlc is resumed from a prior state (e.g., via
an optional `_last_bucket` parameter), a first tick in the same bucket as
the prior last bar must merge rather than open a new bar.

Fix: add the boundary guard to resample_ohlc via an optional `initial_last_bucket`
parameter that lets callers seed the prior bucket state, preventing a
duplicate/degenerate bar at the boundary.

Engine-path parity: the fix must NOT change the live engine's existing
bucketing output (the _OhlcBuffer.ingest_tick path is unchanged; only
resample_ohlc gets the optional parameter which defaults to None, making
it a no-op for all existing callers).
"""
from __future__ import annotations

import pytest

from hlanalysis.marketdata.ohlc import bucket_index, resample_ohlc


# ---------------------------------------------------------------------------
# Unit: resample_ohlc boundary guard
# ---------------------------------------------------------------------------


def test_resample_ohlc_first_tick_on_boundary_opens_correct_bucket():
    """The first tick at exactly ts=bucket_ns (bucket 1) must open bucket 1.

    This is the basic case — no double-counting, just verifying the existing
    behavior is preserved by the fix.
    """
    bucket_ns = 5 * 1_000_000_000  # 5s
    # First tick at ts=5s (bucket 1 boundary).
    ticks = [
        (5 * 1_000_000_000, 100.0, 100.0, 100.0),  # bucket=1
        (7 * 1_000_000_000, 102.0, 102.0, 102.0),  # bucket=1
        (10 * 1_000_000_000, 98.0, 98.0, 98.0),    # bucket=2
    ]
    bars = list(resample_ohlc(ticks, bucket_ns=bucket_ns))
    # Bucket 1 (ts=5..9): max(100,102)=102, min=100, close=102 at ts=7s
    # Bucket 2 (ts=10): 98 at ts=10s
    assert len(bars) == 2, f"expected 2 bars, got {len(bars)}: {bars}"
    assert bars[0][0] == 7 * 1_000_000_000, f"bucket 1 last_ts should be 7s; got {bars[0][0]}"
    assert bars[0][2] == pytest.approx(100.0), f"bucket 1 low should be 100; got {bars[0][2]}"
    assert bars[0][1] == pytest.approx(102.0), f"bucket 1 high should be 102; got {bars[0][1]}"


def test_resample_ohlc_no_degenerate_bar_after_empty_gap():
    """After a gap (empty buckets), the next tick opens a fresh bar correctly.

    Buckets 1,2,3 are empty; tick at bucket 4 must open exactly ONE bar for
    bucket 4, not a degenerate empty bar followed by the real bar.
    """
    bucket_ns = 5 * 1_000_000_000
    ticks = [
        (2 * 1_000_000_000, 100.0, 100.0, 100.0),   # bucket 0
        (22 * 1_000_000_000, 105.0, 105.0, 105.0),  # bucket 4 (gap 1..3)
    ]
    bars = list(resample_ohlc(ticks, bucket_ns=bucket_ns))
    assert len(bars) == 2, f"expected exactly 2 bars (bucket 0 + bucket 4); got {len(bars)}: {bars}"
    assert bars[0][3] == pytest.approx(100.0), f"bucket 0 close should be 100"
    assert bars[1][3] == pytest.approx(105.0), f"bucket 4 close should be 105"


def test_resample_ohlc_initial_last_bucket_merges_same_bucket():
    """With initial_last_bucket set, a first tick in the SAME bucket as prior
    history merges into the in-progress bar rather than opening a new one.

    Scenario: prior call ended with a partial bar in bucket 2 (some ticks
    already processed). The new call starts with a tick also in bucket 2.
    Without initial_last_bucket, a new bar for bucket 2 would be opened
    (double-counting). With it, the tick merges into the continuing bucket 2 bar.
    """
    bucket_ns = 5 * 1_000_000_000

    # Fresh call with initial_last_bucket=None (normal use): first tick opens a new bar.
    ticks = [(10 * 1_000_000_000, 100.0, 100.0, 100.0)]  # bucket=2
    bars_no_seed = list(resample_ohlc(ticks, bucket_ns=bucket_ns))
    assert len(bars_no_seed) == 1

    # With initial_last_bucket = 2 (prior session ended in bucket 2):
    # first tick at bucket 2 must NOT yield a new bar immediately —
    # it merges into the ongoing bucket 2 bar.
    bars_seeded = list(resample_ohlc(ticks, bucket_ns=bucket_ns, initial_last_bucket=2))
    # The tick merges (no NEW bar emitted for bucket 2 yet since no transition)
    # but since it's the last tick in the iterable, the final bar IS emitted.
    assert len(bars_seeded) == 1, (
        f"seeded call: bucket-2 tick must yield exactly 1 bar (not a duplicate); "
        f"got {len(bars_seeded)}: {bars_seeded}"
    )
    # The bar should have the same ts as the tick.
    assert bars_seeded[0][0] == 10 * 1_000_000_000

    # BUT: with initial_last_bucket = DIFFERENT bucket (1), first tick opens bucket 2.
    # This simulates "previous session ended in bucket 1, new tick in bucket 2".
    bars_diff_seed = list(resample_ohlc(ticks, bucket_ns=bucket_ns, initial_last_bucket=1))
    # Bucket 1 was the prior session (no prior bar to emit since h/l/c not set).
    # Tick in bucket 2 → new bar for bucket 2.
    assert len(bars_diff_seed) == 1


def test_resample_ohlc_initial_last_bucket_prevents_duplicate_on_transition():
    """Multi-tick: a tick that transitions buckets while initial_last_bucket is set
    correctly yields the prior bucket bar and opens the new one.

    Scenario: initial_last_bucket=1 (partial prior session), first tick in
    bucket 2. With initial_last_bucket, the transition from 1→2 should open
    bucket 2 (yielding whatever partial bar was in bucket 1 if any).
    In this case since there's no prior partial H/L/C data (only bucket idx),
    the generator just opens bucket 2 fresh when it sees a transition.
    """
    bucket_ns = 5 * 1_000_000_000
    ticks = [
        (10 * 1_000_000_000, 100.0, 100.0, 100.0),  # bucket 2
        (12 * 1_000_000_000, 103.0, 103.0, 103.0),  # bucket 2
        (16 * 1_000_000_000, 99.0, 99.0, 99.0),     # bucket 3
    ]

    # Without seed: 2 bars (bucket 2 + bucket 3)
    bars_no_seed = list(resample_ohlc(ticks, bucket_ns=bucket_ns))
    assert len(bars_no_seed) == 2, f"no seed: expected 2 bars; got {bars_no_seed}"

    # With initial_last_bucket=1 (prior ended in bucket 1):
    # first tick in bucket 2 triggers transition 1→2 (yields prior state if any)
    # then accumulates bucket 2 ticks, then yields bucket 2 + opens 3.
    bars_seeded = list(resample_ohlc(ticks, bucket_ns=bucket_ns, initial_last_bucket=1))
    # Bucket 2 (ticks at 10s, 12s): high=103, low=100, close=103, ts=12s
    # Bucket 3 (tick at 16s): 99
    assert len(bars_seeded) == 2, f"seeded: expected 2 bars; got {bars_seeded}"
    assert bars_seeded[0][3] == pytest.approx(103.0), "bucket 2 close should be 103"
    assert bars_seeded[1][3] == pytest.approx(99.0), "bucket 3 close should be 99"


# ---------------------------------------------------------------------------
# Parity: engine path (_OhlcBuffer.ingest_tick) output is unchanged
# ---------------------------------------------------------------------------


def test_engine_ohlc_buffer_ingest_tick_output_unchanged():
    """The fix (adding initial_last_bucket param) must NOT change the live
    engine path (_OhlcBuffer.ingest_tick) output.

    We run the engine path on a fixed tick sequence and verify the bars
    produced match the expected output from before the fix. The fix ONLY
    adds a parameter to resample_ohlc (default None = no-op); it does NOT
    touch _OhlcBuffer at all.
    """
    from hlanalysis.marketdata.market_state import _OhlcBuffer

    bucket_ns = 5 * 1_000_000_000
    buf = _OhlcBuffer(bucket_ns)

    ticks = [
        (2 * 1_000_000_000, 100.0),  # bucket 0
        (3 * 1_000_000_000, 102.0),  # bucket 0
        (7 * 1_000_000_000, 98.0),   # bucket 1
        (10 * 1_000_000_000, 105.0), # bucket 2
        (14 * 1_000_000_000, 103.0), # bucket 2
    ]
    for ts, price in ticks:
        buf.ingest_tick(ts, price)

    assert buf._len == 3, f"expected 3 buckets (0,1,2); got {buf._len}"
    # Bucket 0: high=102, low=100, close=102, ts=3s
    assert buf._ts[0] == 3 * 1_000_000_000
    assert buf._high[0] == pytest.approx(102.0)
    assert buf._low[0] == pytest.approx(100.0)
    assert buf._close[0] == pytest.approx(102.0)
    # Bucket 1: scalar ts=7s, price=98
    assert buf._ts[1] == 7 * 1_000_000_000
    assert buf._high[1] == pytest.approx(98.0)
    # Bucket 2: high=105, low=103, close=103, ts=14s
    assert buf._ts[2] == 14 * 1_000_000_000
    assert buf._high[2] == pytest.approx(105.0)
    assert buf._low[2] == pytest.approx(103.0)
    assert buf._close[2] == pytest.approx(103.0)


def test_resample_ohlc_default_no_seed_bit_identical_to_prior():
    """Without initial_last_bucket (default None), resample_ohlc produces
    exactly the same output as before the fix — backward compatible.
    """
    bucket_ns = 5 * 1_000_000_000
    ticks = [
        (0 * 1_000_000_000, 100.0, 100.0, 100.0),
        (1 * 1_000_000_000, 102.0, 102.0, 102.0),
        (2 * 1_000_000_000, 99.0, 99.0, 99.0),
        (5 * 1_000_000_000, 101.0, 101.0, 101.0),
        (7 * 1_000_000_000, 103.0, 103.0, 103.0),
        (12 * 1_000_000_000, 98.0, 98.0, 98.0),
    ]
    expected = [
        (2 * 1_000_000_000, 102.0, 99.0, 99.0),
        (7 * 1_000_000_000, 103.0, 101.0, 103.0),
        (12 * 1_000_000_000, 98.0, 98.0, 98.0),
    ]
    bars = list(resample_ohlc(ticks, bucket_ns=bucket_ns))
    assert bars == expected, f"resample_ohlc output changed with fix (backward-compat broken)"
