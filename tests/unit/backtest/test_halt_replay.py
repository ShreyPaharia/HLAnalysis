"""SHR-85 — sim state/halt replay + daily-loss-cap + inventory caps.

Unit tests for the pure, injection-testable core in
``hlanalysis.backtest.halt_replay``: halt-window suppression, the daily-loss-cap
slot-halt, the inventory caps, and the loader that turns engine halt/event-log
records into halt windows. None of these tests touch hftbacktest or the runner —
the suppression logic is exercised with fully injected inputs.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hlanalysis.backtest.halt_replay import (
    EntryGateInputs,
    HaltWindow,
    SimRiskCaps,
    daily_window_start_ns,
    entry_veto,
    in_halt_window,
    load_halt_windows,
)


def _ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1e9)


# ---------------------------------------------------------------------------
# HaltWindow / in_halt_window
# ---------------------------------------------------------------------------


def test_halt_window_contains_is_half_open():
    """[start, end): start is inside, end is the first instant outside."""
    w = HaltWindow(start_ns=100, end_ns=200, reason="stale_data_halt")
    assert w.contains(100)
    assert w.contains(150)
    assert not w.contains(200)
    assert not w.contains(99)


def test_in_halt_window_returns_matching_window_or_none():
    windows = [
        HaltWindow(10, 20, "feed_stale"),
        HaltWindow(50, 80, "daily_loss_halt"),
    ]
    assert in_halt_window(windows, 5) is None
    assert in_halt_window(windows, 15) is windows[0]
    assert in_halt_window(windows, 20) is None  # gap between windows
    assert in_halt_window(windows, 60) is windows[1]
    assert in_halt_window(windows, 80) is None


# ---------------------------------------------------------------------------
# daily_window_start_ns — mirrors Scanner._daily_window_start_ns
# ---------------------------------------------------------------------------


def test_daily_window_start_at_midnight():
    now = _ns(datetime(2026, 6, 8, 14, 30, tzinfo=timezone.utc))
    ws = daily_window_start_ns(now, hour=0)
    assert ws == _ns(datetime(2026, 6, 8, 0, 0, tzinfo=timezone.utc))


def test_daily_window_start_rolls_back_when_before_boundary():
    """At 03:00 UTC with a 06:00 window start, the window began yesterday 06:00."""
    now = _ns(datetime(2026, 6, 8, 3, 0, tzinfo=timezone.utc))
    ws = daily_window_start_ns(now, hour=6)
    assert ws == _ns(datetime(2026, 6, 7, 6, 0, tzinfo=timezone.utc))


def test_daily_window_start_at_or_after_boundary_is_today():
    now = _ns(datetime(2026, 6, 8, 9, 0, tzinfo=timezone.utc))
    ws = daily_window_start_ns(now, hour=6)
    assert ws == _ns(datetime(2026, 6, 8, 6, 0, tzinfo=timezone.utc))


# ---------------------------------------------------------------------------
# entry_veto — halt window
# ---------------------------------------------------------------------------


def _inp(**kw) -> EntryGateInputs:
    base = dict(
        now_ns=0,
        intent_notional=0.0,
        held_inventory_usd=0.0,
        n_held_positions=0,
        is_topup=False,
        realized_pnl_window=0.0,
    )
    base.update(kw)
    return EntryGateInputs(**base)


def test_entry_veto_blocks_inside_halt_window():
    caps = SimRiskCaps()
    windows = [HaltWindow(100, 200, "stale_data_halt")]
    assert entry_veto(caps, windows, _inp(now_ns=150)) == "halt_window:stale_data_halt"
    assert entry_veto(caps, windows, _inp(now_ns=250)) is None


# ---------------------------------------------------------------------------
# entry_veto — daily loss cap
# ---------------------------------------------------------------------------


def test_entry_veto_daily_loss_cap_blocks_when_window_loss_below_cap():
    caps = SimRiskCaps(daily_loss_cap_usd=100.0)
    # realized loss of -150 in the window is past the -100 cap → block
    assert entry_veto(caps, [], _inp(realized_pnl_window=-150.0)) == "daily_loss_cap"
    # -50 is within budget → allowed
    assert entry_veto(caps, [], _inp(realized_pnl_window=-50.0)) is None


def test_entry_veto_daily_loss_cap_disabled_when_none():
    caps = SimRiskCaps(daily_loss_cap_usd=None)
    assert entry_veto(caps, [], _inp(realized_pnl_window=-9999.0)) is None


# ---------------------------------------------------------------------------
# entry_veto — inventory caps
# ---------------------------------------------------------------------------


def test_entry_veto_total_inventory_blocks_over_cap():
    caps = SimRiskCaps(max_total_inventory_usd=300.0)
    # 250 held + 100 new = 350 > 300 → block
    assert (
        entry_veto(caps, [], _inp(held_inventory_usd=250.0, intent_notional=100.0))
        == "max_total_inventory"
    )
    # 250 held + 40 new = 290 ≤ 300 → allowed
    assert (
        entry_veto(caps, [], _inp(held_inventory_usd=250.0, intent_notional=40.0))
        is None
    )


def test_entry_veto_concurrent_positions_blocks_n_plus_1():
    caps = SimRiskCaps(max_concurrent_positions=3)
    # Holding 3 already, a NEW position (not a top-up) is the 4th → blocked
    assert (
        entry_veto(caps, [], _inp(n_held_positions=3, is_topup=False))
        == "max_concurrent_positions"
    )
    # Holding 2 → a 3rd new position is allowed
    assert entry_veto(caps, [], _inp(n_held_positions=2, is_topup=False)) is None


def test_entry_veto_concurrent_cap_allows_topup_to_existing():
    """A top-up to an already-held position is not a new slot — the live gate
    allows it even at the concurrent cap."""
    caps = SimRiskCaps(max_concurrent_positions=2)
    assert entry_veto(caps, [], _inp(n_held_positions=2, is_topup=True)) is None


def test_entry_veto_no_caps_configured_always_allows():
    assert entry_veto(SimRiskCaps(), [], _inp(now_ns=123)) is None


# ---------------------------------------------------------------------------
# load_halt_windows — engine event-log → halt windows
# ---------------------------------------------------------------------------


def test_load_pairs_feed_stale_with_feed_recovered():
    events = [
        {"ts_ns": 1_000, "kind": "stale_data_halt"},
        {"ts_ns": 5_000, "kind": "feed_recovered"},
    ]
    windows = load_halt_windows(events, fallback_duration_ns=10_000)
    assert len(windows) == 1
    assert windows[0].start_ns == 1_000
    assert windows[0].end_ns == 5_000
    assert windows[0].reason == "stale_data_halt"


def test_load_uses_fallback_duration_for_unpaired_halt():
    """A halt with no matching clear (reject-breaker / dust-block / OOM) gets a
    fixed fallback duration so the sim still sits out a plausible window."""
    events = [{"ts_ns": 1_000, "kind": "memory_halt"}]
    windows = load_halt_windows(events, fallback_duration_ns=30_000)
    assert len(windows) == 1
    assert windows[0].start_ns == 1_000
    assert windows[0].end_ns == 31_000
    assert windows[0].reason == "memory_halt"


def test_load_daily_loss_halt_clears_at_next_window_boundary():
    """A daily_loss_halt suppresses entries to the end of its daily window."""
    halt_ts = _ns(datetime(2026, 6, 8, 14, 0, tzinfo=timezone.utc))
    events = [{"ts_ns": halt_ts, "kind": "daily_loss_halt"}]
    windows = load_halt_windows(
        events, fallback_duration_ns=1, daily_window_start_hour_utc=6
    )
    assert len(windows) == 1
    assert windows[0].start_ns == halt_ts
    # window started today 06:00 → next boundary is tomorrow 06:00
    assert windows[0].end_ns == _ns(datetime(2026, 6, 9, 6, 0, tzinfo=timezone.utc))
    assert windows[0].reason == "daily_loss_halt"


def test_load_ignores_non_halt_events():
    events = [
        {"ts_ns": 1, "kind": "entry"},
        {"ts_ns": 2, "kind": "engine_heartbeat"},
        {"ts_ns": 3, "kind": "exit"},
    ]
    assert load_halt_windows(events, fallback_duration_ns=10) == []


def test_load_handles_multiple_windows_in_order():
    events = [
        {"ts_ns": 100, "kind": "feed_stale"},
        {"ts_ns": 200, "kind": "feed_recovered"},
        {"ts_ns": 500, "kind": "memory_halt"},
    ]
    windows = load_halt_windows(events, fallback_duration_ns=50)
    assert [(w.start_ns, w.end_ns) for w in windows] == [(100, 200), (500, 550)]
