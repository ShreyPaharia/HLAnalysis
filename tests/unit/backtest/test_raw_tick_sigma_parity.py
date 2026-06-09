"""SHR-96 regression: raw-tick σ parity with bars path, independent of scan cadence.

Root cause (SHR-96): when ``reference_ticks="raw"``, the runner's ``MarketState``
uses ``_OhlcBuffer`` with ``bucket_ns = _default_bucket_ns = 60s`` regardless of
``vol_sampling_dt_seconds``.  If ``vol_sampling_dt_seconds < 60`` (e.g. dt=5s),
the buffer buckets ticks at 60s instead of 5s, producing ~12× fewer returns per
lookback window, each spanning ~12× longer time.  When the strategy annualises
these using ``vol_sampling_dt_seconds=5`` it inflates σ by √(60/5) ≈ 3.46×.

In event-scan mode (0.2 s cadence) σ reaches 2.5–4.1 → p_model collapses →
``edge_buffer`` never clears → 0 trades.

Fix (SHR-96): the runner calls
``state.set_reference_cadence(reference_resample_seconds)`` after creating
``MarketState`` and before ingesting raw ticks.  This wires the correct
``bucket_ns`` into ``_OhlcBuffer`` so returns are always dt-spaced, independent
of scan cadence.

Acceptance tests
────────────────
1. ``test_raw_tick_returns_match_bars_returns`` — the raw-tick path (with
   ``set_reference_cadence`` called) produces the SAME return series as the
   pre-bucketed bars path on a fixture with sub-dt tick spacing.

2. ``test_raw_tick_sigma_close_to_bars_sigma`` — σ from the two paths agrees
   within a tight tolerance (1e-10 absolute; they are derived from identical
   return series so should be bit-identical).

3. ``test_raw_tick_scan_cadence_independent`` — feeding the SAME tick stream
   through raw mode and querying at "fixed" (60 s cadence) vs "event"
   (0.2 s cadence, querying after every tick) gives IDENTICAL return series
   and σ.  This is the event-mode regression guard.

4. ``test_raw_tick_dt_mismatch_inflates_sigma`` — NEGATIVE control: calling
   raw-tick mode WITHOUT ``set_reference_cadence`` (so the buffer defaults to
   60 s) and querying at a mis-matched dt produces inflated σ relative to the
   bars path, confirming the bug existed before the fix.

5. ``test_runner_market_state_set_reference_cadence_api`` — the
   ``MarketState.set_reference_cadence(dt_seconds)`` method exists and
   registers the cadence on the underlying shared core.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from hlanalysis.backtest.core.events import ReferenceEvent
from hlanalysis.backtest.data._fastpath_core import _resample_reference_rows
from hlanalysis.backtest.runner.market_state import MarketState

# ---------------------------------------------------------------------------
# Shared fixture — sub-dt ticks spanning many dt=5s buckets.
#
# Ticks arrive at ~1s intervals across a 120s window.  At dt=5s this means
# 5 ticks per bucket on average — the dense-tick case where the bug manifests
# most strongly (each tick would become its own 60s bar under the default).
# ---------------------------------------------------------------------------

_S = 1_000_000_000  # 1 ns in ns
_DT_SECONDS = 5
_DT_NS = _DT_SECONDS * _S
_LOOKBACK_SECONDS = 300  # wide enough to capture all bars in the fixture

# 120 ticks at ~1s spacing, with price variation to generate non-trivial returns.
_BASE_PRICE = 95_000.0

_TICKS: list[tuple[int, float]] = [
    (i * _S, _BASE_PRICE * (1.0 + 0.001 * math.sin(i * 0.7) + 0.0003 * (i % 7 - 3)))
    for i in range(120)
]
_LAST_TS = _TICKS[-1][0]
_NOW_NS = _LAST_TS + 1  # query point just past the last tick


def _make_raw_events() -> list[ReferenceEvent]:
    return [ReferenceEvent(ts, "BTC", px, px, px) for ts, px in _TICKS]


def _make_bar_events(dt_seconds: int = _DT_SECONDS) -> list[ReferenceEvent]:
    raw = _make_raw_events()
    return _resample_reference_rows(raw, resample_ns=dt_seconds * _S)


def _build_bars_state(dt_seconds: int = _DT_SECONDS) -> MarketState:
    ms = MarketState()
    for ev in _make_bar_events(dt_seconds):
        ms.apply_reference(ev)
    return ms


def _build_raw_state(dt_seconds: int = _DT_SECONDS) -> MarketState:
    """Raw-tick state WITH correct cadence configured (the fix)."""
    ms = MarketState()
    ms.set_reference_cadence(dt_seconds)  # SHR-96 fix
    for ev in _make_raw_events():
        ms.apply_reference_tick(ev)
    return ms


def _build_raw_state_no_cadence() -> MarketState:
    """Raw-tick state WITHOUT cadence (reproduces the pre-fix bug)."""
    ms = MarketState()
    # Deliberately omit set_reference_cadence → defaults to 60s
    for ev in _make_raw_events():
        ms.apply_reference_tick(ev)
    return ms


# ---------------------------------------------------------------------------
# 1. Return series must be identical (bars vs raw + set_reference_cadence)
# ---------------------------------------------------------------------------


def test_raw_tick_returns_match_bars_returns() -> None:
    """Raw-tick path (with set_reference_cadence) must produce the same return
    series as pre-bucketed bars on the same tick fixture."""
    bars_ms = _build_bars_state()
    raw_ms = _build_raw_state()

    rets_bars = bars_ms.recent_returns(now_ns=_NOW_NS, lookback_seconds=_LOOKBACK_SECONDS)
    rets_raw = raw_ms.recent_returns(now_ns=_NOW_NS, lookback_seconds=_LOOKBACK_SECONDS)

    assert len(rets_bars) > 0, "bars path produced no returns — fixture too small?"
    assert len(rets_raw) > 0, "raw path produced no returns — cadence not configured?"
    assert len(rets_raw) == len(rets_bars), (
        f"Return series lengths differ: raw={len(rets_raw)}, bars={len(rets_bars)}. "
        f"set_reference_cadence({_DT_SECONDS}) may not have been applied."
    )
    np.testing.assert_allclose(rets_raw, rets_bars, rtol=0, atol=1e-10,
                               err_msg="Raw-tick returns differ from bars returns")


# ---------------------------------------------------------------------------
# 2. σ must be bit-identical (derived from the same return series)
# ---------------------------------------------------------------------------


def test_raw_tick_sigma_close_to_bars_sigma() -> None:
    """σ from raw-tick mode (cadence configured) must match bars-mode σ exactly,
    since both derive from the same dt-spaced close-to-close return series."""
    bars_ms = _build_bars_state()
    raw_ms = _build_raw_state()

    rets_bars = bars_ms.recent_returns(now_ns=_NOW_NS, lookback_seconds=_LOOKBACK_SECONDS)
    rets_raw = raw_ms.recent_returns(now_ns=_NOW_NS, lookback_seconds=_LOOKBACK_SECONDS)

    sigma_bars = float(np.std(rets_bars, ddof=1)) if len(rets_bars) > 1 else 0.0
    sigma_raw = float(np.std(rets_raw, ddof=1)) if len(rets_raw) > 1 else 0.0

    assert sigma_bars > 1e-8, "bars σ is essentially zero — fixture degenerate?"
    assert abs(sigma_raw - sigma_bars) < 1e-10, (
        f"σ mismatch: raw={sigma_raw:.8f}, bars={sigma_bars:.8f}. "
        f"Diff={abs(sigma_raw - sigma_bars):.2e}. "
        f"set_reference_cadence may not have been applied."
    )


# ---------------------------------------------------------------------------
# 3. Scan-cadence independence: fixed vs event mode must yield the same σ
#
# We simulate scan cadence by querying at different time points. "Fixed" =
# query only at the end (one query per 60s window, coarse).  "Event" = query
# after every single tick (one query per ~1s tick).  Both must converge on the
# same return series because the dt bucket structure is determined by ingest
# order (ticks are coalesced in-place), not by query frequency.
# ---------------------------------------------------------------------------


def test_raw_tick_scan_cadence_independent() -> None:
    """Querying at coarse (60s) vs fine (per-tick ~1s) cadence must produce the
    SAME return series and σ — the fix is that returns are always dt-spaced."""
    # "Fixed" scan cadence: query only at the very end.
    ms_fixed = MarketState()
    ms_fixed.set_reference_cadence(_DT_SECONDS)

    # "Event" scan cadence: query after EVERY tick (simulates 0.2s event mode).
    ms_event = MarketState()
    ms_event.set_reference_cadence(_DT_SECONDS)

    raw_evs = _make_raw_events()

    # Both states receive the same ticks. The "event" state also triggers a
    # read after each tick (matching the scan loop querying at each event).
    _intermediate_rets: list[np.ndarray] = []
    for ev in raw_evs:
        ms_fixed.apply_reference_tick(ev)
        ms_event.apply_reference_tick(ev)
        # Simulate event-mode: read right after each tick lands.
        r = ms_event.recent_returns(now_ns=ev.ts_ns + 1, lookback_seconds=_LOOKBACK_SECONDS)
        _intermediate_rets.append(r)

    # Final query at the same now_ns for both.
    rets_fixed = ms_fixed.recent_returns(now_ns=_NOW_NS, lookback_seconds=_LOOKBACK_SECONDS)
    rets_event = ms_event.recent_returns(now_ns=_NOW_NS, lookback_seconds=_LOOKBACK_SECONDS)

    assert len(rets_fixed) > 0, "fixed-cadence produced no returns"
    assert len(rets_event) == len(rets_fixed), (
        f"Return series lengths differ: fixed={len(rets_fixed)}, event={len(rets_event)}"
    )
    np.testing.assert_allclose(rets_event, rets_fixed, rtol=0, atol=1e-10,
                               err_msg="Event-cadence returns differ from fixed-cadence")

    # σ must agree.
    sigma_fixed = float(np.std(rets_fixed, ddof=1)) if len(rets_fixed) > 1 else 0.0
    sigma_event = float(np.std(rets_event, ddof=1)) if len(rets_event) > 1 else 0.0
    assert abs(sigma_event - sigma_fixed) < 1e-10, (
        f"σ differs across scan cadences: fixed={sigma_fixed:.8f}, event={sigma_event:.8f}"
    )

    # The intermediate "event" reads must NOT be inflated by sub-dt tick spacing.
    # Pre-fix, event-scan would produce σ ~3.46× higher than bars (dt=5s, default
    # 60s bucket → sqrt(60/5) inflation).  Post-fix, intermediate σ should be within
    # a reasonable range of the final bars σ (we use 50% tolerance because the
    # intermediate read is at a slightly earlier now_ns than the bars reference, so
    # some window-boundary variation is expected — but NOT 3×).
    last_intermediate = _intermediate_rets[-2]  # one before the final tick
    if len(last_intermediate) > 1:
        sigma_intermediate = float(np.std(last_intermediate, ddof=1))
        bars_ms_ref = _build_bars_state()
        rets_bars_ref = bars_ms_ref.recent_returns(
            now_ns=_NOW_NS, lookback_seconds=_LOOKBACK_SECONDS
        )
        sigma_bars_ref = float(np.std(rets_bars_ref, ddof=1))
        # Inflation guard: intermediate σ must be < 2× bars σ (pre-fix was ~3.46×).
        ratio = sigma_intermediate / max(sigma_bars_ref, 1e-8)
        assert ratio < 2.0, (
            f"Intermediate event-scan σ={sigma_intermediate:.6f} is {ratio:.2f}× bars "
            f"σ={sigma_bars_ref:.6f} — dt-mismatch inflation detected (pre-fix was ~3.46×)."
        )


# ---------------------------------------------------------------------------
# 4. NEGATIVE control: without set_reference_cadence, σ is inflated
#    (confirms the bug existed and the fix addresses it)
# ---------------------------------------------------------------------------


def test_raw_tick_dt_mismatch_inflates_sigma() -> None:
    """Without set_reference_cadence, the raw-tick buffer defaults to 60s.
    For dt=5s ticks (√(60/5)=3.46×), σ should be measurably higher than
    bars-path σ.  This is the pre-fix behaviour we are guarding against."""
    bars_ms = _build_bars_state()
    buggy_ms = _build_raw_state_no_cadence()  # no set_reference_cadence

    rets_bars = bars_ms.recent_returns(now_ns=_NOW_NS, lookback_seconds=_LOOKBACK_SECONDS)
    # The buggy state buckets at 60s; our 120s fixture produces only ~2 bars
    # (buckets 0..60s and 60..120s), vs ~24 bars at dt=5s.  The buggy state
    # may return a different number of returns.
    rets_buggy = buggy_ms.recent_returns(now_ns=_NOW_NS, lookback_seconds=_LOOKBACK_SECONDS)

    sigma_bars = float(np.std(rets_bars, ddof=1)) if len(rets_bars) > 1 else 0.0
    sigma_buggy = float(np.std(rets_buggy, ddof=1)) if len(rets_buggy) > 1 else 0.0

    # The buggy path produces far fewer returns (buckets at 60s, not 5s).
    # Either it produces 0 returns (no bucket boundary crossed in lookback) or
    # a drastically different σ. In either case it must NOT match the bars σ.
    if sigma_bars > 1e-8 and sigma_buggy > 1e-8:
        rel_diff = abs(sigma_buggy - sigma_bars) / sigma_bars
        assert rel_diff > 0.1, (
            f"Expected pre-fix raw σ to diverge from bars σ, but got "
            f"raw={sigma_buggy:.6f}, bars={sigma_bars:.6f}, rel_diff={rel_diff:.1%}. "
            f"The dt-mismatch inflation may not be detectable on this fixture."
        )
    else:
        # One of them is 0/near-0: the bucket sizes differ so fundamentally
        # that one has no returns — still counts as a mismatch.
        assert len(rets_buggy) != len(rets_bars), (
            f"Buggy and bars return series have the same length ({len(rets_bars)}), "
            f"meaning the default 60s bucket matches the bars cadence — fixture needs "
            f"to use a shorter dt to expose the bug."
        )


# ---------------------------------------------------------------------------
# 5. API presence test: set_reference_cadence exists and registers the cadence
# ---------------------------------------------------------------------------


def test_runner_market_state_set_reference_cadence_api() -> None:
    """MarketState.set_reference_cadence(dt_seconds) must exist and configure
    the underlying _OhlcBuffer with the correct bucket_ns."""
    ms = MarketState()
    # Before any registration the buffer does not exist yet (lazy init).
    # After one tick (with default 60s bucket), a 60s buffer is created.
    ms.set_reference_cadence(5)

    # Feed a tick so the buffer is materialised.
    ms.apply_reference_tick(ReferenceEvent(
        ts_ns=1 * _S, symbol="BTC", high=100.0, low=100.0, close=100.0,
    ))
    # A second tick in a new dt=5s bucket must produce a return.
    ms.apply_reference_tick(ReferenceEvent(
        ts_ns=6 * _S, symbol="BTC", high=101.0, low=101.0, close=101.0,
    ))
    rets = ms.recent_returns(now_ns=7 * _S, lookback_seconds=60)
    assert len(rets) == 1, (
        f"Expected exactly 1 return (from bucket 0→1 crossing) but got {len(rets)}. "
        f"set_reference_cadence(5) may not have set bucket_ns=5s."
    )
    # Re-registration must be a no-op (no assertion — just must not raise).
    ms.set_reference_cadence(5)


# ---------------------------------------------------------------------------
# 6. Bars path is unaffected by set_reference_cadence (backward-compat guard)
# ---------------------------------------------------------------------------


def test_bars_path_unaffected_by_cadence_call() -> None:
    """Calling set_reference_cadence before apply_reference (bars path) must
    produce the same result as without the call — the fix must not break
    existing callers that use the bars path."""
    # Without cadence registration.
    ms_no_reg = MarketState()
    for ev in _make_bar_events():
        ms_no_reg.apply_reference(ev)

    # With cadence registration (the same dt as the bars).
    ms_reg = MarketState()
    ms_reg.set_reference_cadence(_DT_SECONDS)
    for ev in _make_bar_events():
        ms_reg.apply_reference(ev)

    rets_no = ms_no_reg.recent_returns(now_ns=_NOW_NS, lookback_seconds=_LOOKBACK_SECONDS)
    rets_reg = ms_reg.recent_returns(now_ns=_NOW_NS, lookback_seconds=_LOOKBACK_SECONDS)

    assert len(rets_no) == len(rets_reg), (
        f"Bars path returned different lengths with/without cadence registration: "
        f"no_reg={len(rets_no)}, reg={len(rets_reg)}"
    )
    np.testing.assert_allclose(rets_no, rets_reg, rtol=0, atol=1e-10,
                               err_msg="Bars path changed behaviour with cadence registration")
