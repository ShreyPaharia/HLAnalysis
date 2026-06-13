"""SHR-66 — sim/live MarketState windowing parity.

The bug: live ``recent_returns`` windows by COUNT (``[-(n+1):]``), backtest by
TIME (``slice_window(now_ns, lookback_seconds)``). After a feed gap the count
window reaches further back in wall-clock and includes bars the time window
excludes — different σ/p_model/safety_d than backtest.

This test constructs an explicit feed gap, calls ``recent_returns`` with the
TIME arguments ``(now_ns, lookback_seconds)`` on the live engine MarketState,
and independently computes the expected returns by slicing the same bars by
wall-clock time (the backtest rule). Both must agree after the fix.

The test ALSO demonstrates the pre-fix COUNT bug: calling the count path with
``n`` large enough to span the gap returns MORE bars than the time path —
confirming they diverge after a gap.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hlanalysis.engine.market_state import MarketState
from hlanalysis.events import MarkEvent, Mechanism, ProductType


_S = 1_000_000_000  # 1 second in ns


def _mark(symbol: str, px: float, ts_ns: int) -> MarkEvent:
    return MarkEvent(
        venue="hyperliquid",
        product_type=ProductType.PERP,
        mechanism=Mechanism.CLOB,
        symbol=symbol,
        exchange_ts=ts_ns,
        local_recv_ts=ts_ns,
        mark_px=px,
    )


def _build_ms_with_gap(
    *,
    dt_seconds: int = 5,
    lookback_seconds: int = 60,
    gap_seconds: int = 120,
) -> tuple[MarketState, list[tuple[int, float]], int]:
    """Feed a known tick stream with a gap.

    Timeline:
      - 10 bars before the gap  (t=0s … t=(10*dt)s)
      - gap of ``gap_seconds`` (no ticks)
      - 4 bars after the gap   (t=(10*dt + gap)s … +3*dt)

    Returns (ms, all_bars_as_(ts_ns, close), now_ns) where now_ns is the
    timestamp of the last bar.
    """
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=dt_seconds, lookback_seconds=1000)

    bars: list[tuple[int, float]] = []
    px = 100.0

    # Pre-gap bars
    for i in range(10):
        ts_ns = i * dt_seconds * _S
        px += 0.1
        ms.apply(_mark("BTC", px, ts_ns))
        bars.append((ts_ns, px))

    # Post-gap bars (gap_seconds after the last pre-gap bar)
    gap_start_ns = 10 * dt_seconds * _S + gap_seconds * _S
    for j in range(4):
        ts_ns = gap_start_ns + j * dt_seconds * _S
        px += 0.1
        ms.apply(_mark("BTC", px, ts_ns))
        bars.append((ts_ns, px))

    now_ns = bars[-1][0]
    return ms, bars, now_ns


# ---------------------------------------------------------------------------
# Core parity test: time-bounded live == time-bounded backtest rule
# ---------------------------------------------------------------------------


def test_time_bounded_recent_returns_excludes_pre_gap_bars():
    """After a feed gap, the time-bounded path must drop bars older than the
    lookback window — matching the backtest's ``slice_window`` rule."""
    dt_seconds = 5
    lookback_seconds = 60  # 12 bars of dt=5 needed to cover 60s
    gap_seconds = 120  # gap is 2× the lookback — pre-gap bars must be excluded

    ms, all_bars, now_ns = _build_ms_with_gap(
        dt_seconds=dt_seconds,
        lookback_seconds=lookback_seconds,
        gap_seconds=gap_seconds,
    )
    cutoff_ns = now_ns - lookback_seconds * _S

    # Independent expected result: backtest TIME rule — both bar endpoints in
    # [now - lookback, now], positive closes.
    in_window = [(ts, c) for ts, c in all_bars if ts >= cutoff_ns]
    expected_rets: list[float] = []
    for k in range(1, len(in_window)):
        ts_prev, c_prev = in_window[k - 1]
        ts_curr, c_curr = in_window[k]
        # Both endpoints must be in the window (i.e. ts_prev >= cutoff_ns too).
        # slice_window uses lo_idx+1..hi_idx so the pair (lo_idx, lo_idx+1) is the
        # first kept return — both endpoints in [cutoff, now].
        if ts_prev >= cutoff_ns and c_prev > 0 and c_curr > 0:
            expected_rets.append(math.log(c_curr / c_prev))

    # TIME-bounded live call
    live_rets = ms.recent_returns(
        "BTC",
        n=1000,
        dt=dt_seconds,
        now_ns=now_ns,
        lookback_seconds=lookback_seconds,
    )

    assert len(live_rets) > 0, "no returns returned — gap too wide or lookback too small"
    np.testing.assert_allclose(
        np.asarray(live_rets, dtype=np.float64),
        np.asarray(expected_rets, dtype=np.float64),
        rtol=0,
        atol=0,
        err_msg="live time-bounded returns do not match backtest TIME rule",
    )


def test_time_bounded_excludes_more_bars_than_count_path_after_gap():
    """Demonstrate the bug: the COUNT path (n large) includes pre-gap bars that
    the TIME path drops — they MUST diverge after a gap (pre-fix) and converge
    (post-fix applies time rule to both time-bounded and the count path below).

    This test asserts only the TIME-bounded path returns fewer bars than a
    COUNT path large enough to include pre-gap bars, confirming the gap
    produces different bar sets.
    """
    dt_seconds = 5
    lookback_seconds = 30  # 6 bars
    gap_seconds = 200  # far beyond lookback

    ms, all_bars, now_ns = _build_ms_with_gap(
        dt_seconds=dt_seconds,
        lookback_seconds=lookback_seconds,
        gap_seconds=gap_seconds,
    )

    # TIME-bounded path — should see only post-gap bars
    time_rets = ms.recent_returns(
        "BTC",
        n=1000,
        dt=dt_seconds,
        now_ns=now_ns,
        lookback_seconds=lookback_seconds,
    )
    # COUNT path with a very large n — spans the gap
    count_rets = ms.recent_returns("BTC", n=1000, dt=dt_seconds)

    # There are only 4 post-gap bars → at most 3 post-gap returns.
    # The pre-gap bars add 9 more — count path must return more.
    assert len(count_rets) > len(time_rets), (
        f"expected count path ({len(count_rets)}) > time path ({len(time_rets)}) after a gap wider than the lookback"
    )
    # The time path returns only the in-window bars.
    assert len(time_rets) <= 3, f"time path returned {len(time_rets)} returns but only 4 post-gap bars exist"


# ---------------------------------------------------------------------------
# Parity: time-bounded live == KlineRingBuffer slice_window (SIM path)
# ---------------------------------------------------------------------------


def test_time_bounded_matches_kline_ring_buffer():
    """The live time-bounded path must match the backtest KlineRingBuffer
    slice_window on the SAME bar stream (including the gap)."""
    from hlanalysis.backtest.core.events import ReferenceEvent
    from hlanalysis.backtest.data._fastpath_core import _resample_reference_rows
    from hlanalysis.backtest.runner.market_state import MarketState as SimMS

    dt_seconds = 5
    lookback_seconds = 60
    gap_seconds = 120

    ms_live, all_bars, now_ns = _build_ms_with_gap(
        dt_seconds=dt_seconds,
        lookback_seconds=lookback_seconds,
        gap_seconds=gap_seconds,
    )

    # Build the SIM MarketState from the same bar stream
    ms_sim = SimMS()
    raw = [ReferenceEvent(ts, "BTC", px, px, px) for ts, px in all_bars]
    bars = _resample_reference_rows(raw, resample_ns=dt_seconds * _S)
    for bar in bars:
        ms_sim.apply_reference(bar)

    live_rets = ms_live.recent_returns(
        "BTC",
        n=1000,
        dt=dt_seconds,
        now_ns=now_ns,
        lookback_seconds=lookback_seconds,
    )
    sim_rets = ms_sim.recent_returns(now_ns=now_ns, lookback_seconds=lookback_seconds)

    assert len(live_rets) > 0, "no returns from live path"
    assert len(sim_rets) > 0, "no returns from sim path"
    np.testing.assert_allclose(
        np.asarray(live_rets, dtype=np.float64),
        sim_rets,
        rtol=0,
        atol=0,
        err_msg="live time-bounded path does not match sim KlineRingBuffer slice_window",
    )


# ---------------------------------------------------------------------------
# recent_hl_bars time-bounded parity
# ---------------------------------------------------------------------------


def test_time_bounded_hl_bars_excludes_pre_gap():
    """The time-bounded recent_hl_bars call must also exclude pre-gap bars."""
    dt_seconds = 5
    lookback_seconds = 60
    gap_seconds = 120

    ms, all_bars, now_ns = _build_ms_with_gap(
        dt_seconds=dt_seconds,
        lookback_seconds=lookback_seconds,
        gap_seconds=gap_seconds,
    )
    cutoff_ns = now_ns - lookback_seconds * _S

    # Only 4 post-gap bars → at most 4 HL bars in window
    hl_bars = ms.recent_hl_bars(
        "BTC",
        n=1000,
        dt=dt_seconds,
        now_ns=now_ns,
        lookback_seconds=lookback_seconds,
    )
    in_window_count = sum(1 for ts, _ in all_bars if ts >= cutoff_ns)
    assert len(hl_bars) == in_window_count, f"expected {in_window_count} hl bars in window, got {len(hl_bars)}"


# ---------------------------------------------------------------------------
# No-gap case: time path == count path (both produce all bars)
# ---------------------------------------------------------------------------


def test_time_bounded_same_as_count_when_no_gap():
    """Without a gap, time-bounded and count paths must produce identical
    returns (the regression guard — the fix must not break normal operation)."""
    dt_seconds = 5
    lookback_seconds = 100  # wide enough for all bars
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=dt_seconds, lookback_seconds=1000)
    closes = [100.0 + 0.1 * i for i in range(10)]
    for i, px in enumerate(closes):
        ms.apply(_mark("BTC", px, ts_ns=i * dt_seconds * _S))
    now_ns = (len(closes) - 1) * dt_seconds * _S

    time_rets = ms.recent_returns(
        "BTC",
        n=1000,
        dt=dt_seconds,
        now_ns=now_ns,
        lookback_seconds=lookback_seconds,
    )
    count_rets = ms.recent_returns("BTC", n=1000, dt=dt_seconds)

    assert len(time_rets) == len(count_rets)
    np.testing.assert_allclose(
        np.asarray(time_rets, dtype=np.float64),
        np.asarray(count_rets, dtype=np.float64),
        rtol=0,
        atol=0,
    )


# ---------------------------------------------------------------------------
# Scanner integration: scan() must use time-bounded path
# ---------------------------------------------------------------------------


def test_scanner_exposes_default_lookback_secs(tmp_path):
    """The Scanner must expose _default_lookback_secs so scan() can pass
    lookback_seconds to recent_returns / recent_hl_bars — SHR-66."""
    from hlanalysis.engine.config import AllowlistEntry, GlobalRiskConfig, StrategyConfig
    from hlanalysis.engine.market_state import MarketState as EngineMS
    from hlanalysis.engine.scanner import Scanner
    from hlanalysis.engine.state import StateDAL

    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        max_position_usd=100,
        stop_loss_pct=10,
        tte_min_seconds=60,
        tte_max_seconds=1800,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200,
        vol_max=0.5,
        vol_lookback_seconds=60,
    )
    cfg = StrategyConfig(
        name="late_resolution",
        paper_mode=True,
        allowlist=[entry],
        blocklist_question_idxs=[],
        defaults=entry,
        **{
            "global": GlobalRiskConfig(
                max_total_inventory_usd=500,
                max_concurrent_positions=5,
                daily_loss_cap_usd=200,
                max_strike_distance_pct=10,
                min_recent_volume_usd=0,
                stale_data_halt_seconds=5,
                reconcile_interval_seconds=60,
            )
        },
    )

    ms = EngineMS()
    dal = StateDAL(tmp_path / "state.db")
    scanner = Scanner(
        strategy=None,  # strategy not invoked — testing only scanner construction
        cfg=cfg,
        market_state=ms,
        dal=dal,
        kill_switch_path=tmp_path / "kill",
        last_reconcile_ns=0,
        reference_symbol="BTC",
    )

    # Verify _default_lookback_secs is exposed for time-bounded calls
    assert hasattr(scanner, "_default_lookback_secs"), (
        "Scanner must expose _default_lookback_secs for the time-bounded call in scan() — SHR-66"
    )
    # And that it has a positive integer value matching the configured lookback
    assert scanner._default_lookback_secs == 60
