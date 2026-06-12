"""Unit tests for ``hlanalysis.risk.caps`` — shared entry-cap predicates.

Tests cover:
1. Each predicate across boundary inputs (at/above/below cap, None disables,
   top-up exemption, daily-window boundary just-before/just-after HH:00).
2. Equivalence: the shared predicates produce the SAME decisions that the
   engine's inline logic and the sim's ``entry_veto`` produced before the
   refactor, verified across a grid of inputs.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from hlanalysis.risk.caps import (
    concurrent_cap_exceeded,
    daily_loss_exceeded,
    daily_window_start_ns,
    inventory_cap_exceeded,
)


# ---------------------------------------------------------------------------
# inventory_cap_exceeded
# ---------------------------------------------------------------------------

class TestInventoryCapExceeded:
    def test_none_cap_always_false(self):
        assert inventory_cap_exceeded(900.0, 200.0, None) is False

    def test_below_cap_allowed(self):
        # 100 + 50 = 150 < 200 → False
        assert inventory_cap_exceeded(100.0, 50.0, 200.0) is False

    def test_exactly_at_cap_allowed(self):
        # 100 + 100 = 200 == 200 → False (strict >)
        assert inventory_cap_exceeded(100.0, 100.0, 200.0) is False

    def test_one_cent_above_cap_blocked(self):
        # 100 + 100.01 = 200.01 > 200 → True
        assert inventory_cap_exceeded(100.0, 100.01, 200.0) is True

    def test_well_above_cap_blocked(self):
        assert inventory_cap_exceeded(300.0, 100.0, 200.0) is True

    def test_zero_held_zero_intent_zero_cap_at_limit(self):
        # 0 + 0 = 0 == 0 → False (strict >)
        assert inventory_cap_exceeded(0.0, 0.0, 0.0) is False

    def test_tiny_intent_above_zero_cap_blocked(self):
        assert inventory_cap_exceeded(0.0, 0.001, 0.0) is True

    def test_matches_engine_inline_logic(self):
        """Engine: new_total > max_total_inventory_usd where new_total =
        live_orders_total_notional + notional + sum(|qty|*avg_entry).
        Sim: held_inventory_usd + intent_notional > max_total_inventory_usd.
        Both collapse to: held_plus_orders + intent > cap.
        """
        for held, intent, cap, expected in [
            (0.0,   0.0,   500.0, False),
            (499.0, 0.0,   500.0, False),
            (499.0, 1.0,   500.0, False),   # exactly at cap
            (499.0, 1.01,  500.0, True),    # 1 cent over
            (400.0, 200.0, 500.0, True),    # well over
        ]:
            result = inventory_cap_exceeded(held, intent, cap)
            assert result is expected, (
                f"inventory_cap_exceeded({held}, {intent}, {cap}) "
                f"expected {expected}, got {result}"
            )


# ---------------------------------------------------------------------------
# concurrent_cap_exceeded
# ---------------------------------------------------------------------------

class TestConcurrentCapExceeded:
    def test_none_cap_always_false(self):
        assert concurrent_cap_exceeded(100, False, None) is False

    def test_topup_always_allowed_even_at_cap(self):
        # is_topup=True must never block regardless of n_held or cap
        assert concurrent_cap_exceeded(5, True, 5) is False
        assert concurrent_cap_exceeded(100, True, 1) is False

    def test_below_cap_allowed(self):
        # 3 held, cap=5 → not yet at limit
        assert concurrent_cap_exceeded(3, False, 5) is False

    def test_exactly_at_cap_blocked_for_new_position(self):
        # 5 held, cap=5 → a NEW position is the 6th → blocked (>=)
        assert concurrent_cap_exceeded(5, False, 5) is True

    def test_one_below_cap_allowed(self):
        assert concurrent_cap_exceeded(4, False, 5) is False

    def test_well_above_cap_blocked(self):
        assert concurrent_cap_exceeded(10, False, 5) is True

    def test_zero_cap_zero_held_blocked(self):
        # 0 >= 0 → True for a new position
        assert concurrent_cap_exceeded(0, False, 0) is True

    def test_matches_engine_inline_logic(self):
        """Engine: len(positions) >= max_concurrent_positions AND
        not any(p.question_idx == intent.question_idx for p in positions).
        The second condition is the top-up exemption.
        """
        for n_held, is_topup, cap, expected in [
            (0,  False, 5, False),
            (4,  False, 5, False),
            (5,  False, 5, True),    # at cap → new slot blocked
            (6,  False, 5, True),
            (5,  True,  5, False),   # at cap but it's a top-up → allowed
            (10, True,  5, False),   # well over cap, top-up → still allowed
        ]:
            result = concurrent_cap_exceeded(n_held, is_topup, cap)
            assert result is expected, (
                f"concurrent_cap_exceeded({n_held}, {is_topup}, {cap}) "
                f"expected {expected}, got {result}"
            )


# ---------------------------------------------------------------------------
# daily_loss_exceeded
# ---------------------------------------------------------------------------

class TestDailyLossExceeded:
    def test_none_cap_always_false(self):
        assert daily_loss_exceeded(-9999.0, None) is False

    def test_zero_pnl_not_exceeded(self):
        assert daily_loss_exceeded(0.0, 200.0) is False

    def test_positive_pnl_not_exceeded(self):
        assert daily_loss_exceeded(100.0, 200.0) is False

    def test_small_loss_within_cap_not_exceeded(self):
        # -50 < -200? No → False
        assert daily_loss_exceeded(-50.0, 200.0) is False

    def test_loss_exactly_at_cap_not_exceeded(self):
        # -200 < -200? No (strict <) → False
        assert daily_loss_exceeded(-200.0, 200.0) is False

    def test_loss_one_cent_past_cap_exceeded(self):
        # -200.01 < -200 → True
        assert daily_loss_exceeded(-200.01, 200.0) is True

    def test_large_loss_exceeded(self):
        assert daily_loss_exceeded(-500.0, 200.0) is True

    def test_zero_cap_positive_pnl_not_exceeded(self):
        # 1.0 < 0? No → False
        assert daily_loss_exceeded(1.0, 0.0) is False

    def test_zero_cap_tiny_loss_exceeded(self):
        # -0.001 < 0 → True
        assert daily_loss_exceeded(-0.001, 0.0) is True

    def test_matches_engine_inline_logic(self):
        """Engine: realized_pnl_today < -daily_loss_cap_usd.
        Sim:       realized_pnl_window < -daily_loss_cap_usd.
        """
        for pnl, cap, expected in [
            (0.0,    200.0, False),
            (-199.0, 200.0, False),
            (-200.0, 200.0, False),  # exactly at cap — strict <
            (-200.1, 200.0, True),
            (-999.0, 200.0, True),
            (50.0,   200.0, False),  # profit → never exceeded
        ]:
            result = daily_loss_exceeded(pnl, cap)
            assert result is expected, (
                f"daily_loss_exceeded({pnl}, {cap}) "
                f"expected {expected}, got {result}"
            )


# ---------------------------------------------------------------------------
# daily_window_start_ns
# ---------------------------------------------------------------------------

def _ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1e9)


class TestDailyWindowStartNs:
    def test_at_exact_boundary_returns_same_time(self):
        """At exactly HH:00:00 the boundary is the current moment."""
        ts = _ns(datetime(2026, 6, 8, 0, 0, 0, tzinfo=timezone.utc))
        assert daily_window_start_ns(ts, hour=0) == ts

    def test_just_after_boundary_returns_today_boundary(self):
        """One nanosecond past boundary → today's HH:00."""
        boundary = _ns(datetime(2026, 6, 8, 0, 0, 0, tzinfo=timezone.utc))
        assert daily_window_start_ns(boundary + 1, hour=0) == boundary

    def test_just_before_boundary_rolls_to_yesterday(self):
        """One *second* before HH:00 → yesterday's HH:00.

        Note: 1-ns-before loses precision when divided by 1e9 (float), so
        the smallest reliable sub-boundary step is 1 second.
        """
        ts = _ns(datetime(2026, 6, 8, 0, 0, 0, tzinfo=timezone.utc)) - 1_000_000_000
        expected = _ns(datetime(2026, 6, 7, 0, 0, 0, tzinfo=timezone.utc))
        assert daily_window_start_ns(ts, hour=0) == expected

    def test_midday_with_zero_hour(self):
        now = _ns(datetime(2026, 6, 8, 14, 30, tzinfo=timezone.utc))
        expected = _ns(datetime(2026, 6, 8, 0, 0, tzinfo=timezone.utc))
        assert daily_window_start_ns(now, hour=0) == expected

    def test_rolls_back_when_before_boundary_hour(self):
        """At 03:00 with a 06:00 window, the window started yesterday 06:00."""
        now = _ns(datetime(2026, 6, 8, 3, 0, tzinfo=timezone.utc))
        expected = _ns(datetime(2026, 6, 7, 6, 0, tzinfo=timezone.utc))
        assert daily_window_start_ns(now, hour=6) == expected

    def test_at_or_after_boundary_hour_is_today(self):
        now = _ns(datetime(2026, 6, 8, 9, 0, tzinfo=timezone.utc))
        expected = _ns(datetime(2026, 6, 8, 6, 0, tzinfo=timezone.utc))
        assert daily_window_start_ns(now, hour=6) == expected

    def test_exactly_at_six_boundary(self):
        ts = _ns(datetime(2026, 6, 8, 6, 0, 0, tzinfo=timezone.utc))
        assert daily_window_start_ns(ts, hour=6) == ts

    def test_matches_halt_replay_impl(self):
        """The shared impl must be byte-identical to the old halt_replay copy."""
        from hlanalysis.backtest.halt_replay import daily_window_start_ns as sim_impl
        for hour in [0, 6, 12, 18]:
            for dt_str in [
                "2026-06-08T00:00:00", "2026-06-08T05:59:59",
                "2026-06-08T06:00:00", "2026-06-08T14:30:00",
                "2026-06-08T23:59:59",
            ]:
                ts = _ns(datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc))
                assert daily_window_start_ns(ts, hour=hour) == sim_impl(ts, hour=hour), (
                    f"mismatch at hour={hour} ts={dt_str}"
                )

    def test_matches_scanner_impl(self):
        """The shared impl must also match Scanner._daily_window_start_ns."""
        from hlanalysis.engine.scanner import Scanner
        for hour in [0, 6]:
            for dt_str in ["2026-06-08T03:00:00", "2026-06-08T09:00:00"]:
                ts = _ns(datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc))
                assert daily_window_start_ns(ts, hour=hour) == Scanner._daily_window_start_ns(ts, hour=hour)


# ---------------------------------------------------------------------------
# Cross-caller equivalence grid
# ---------------------------------------------------------------------------
# Drive both the shared predicates and the old inline/sim logic simultaneously
# to confirm bit-identical verdicts across a grid of representative inputs.


class TestEquivalenceWithEngineCaps:
    """Confirm shared predicates match what engine.risk.RiskGate would decide
    inline — i.e. reproducing the three cap conditions from check_pre_trade
    directly and asserting they match."""

    @pytest.mark.parametrize("held_orders,intent_ntl,cap,expected", [
        (0.0,   90.0,  500.0, False),   # well under
        (410.0, 90.0,  500.0, False),   # 500 == cap, at limit
        (410.0, 90.01, 500.0, True),    # 500.01 > cap
        (600.0, 1.0,   500.0, True),    # already over
    ])
    def test_inventory_matches_engine_condition(self, held_orders, intent_ntl, cap, expected):
        # Engine inline: new_total > cap
        engine_result = (held_orders + intent_ntl) > cap
        shared_result = inventory_cap_exceeded(held_orders, intent_ntl, cap)
        assert shared_result is engine_result is expected

    @pytest.mark.parametrize("n_held,is_topup,cap,expected", [
        (0,  False, 5,  False),
        (4,  False, 5,  False),
        (5,  False, 5,  True),
        (5,  True,  5,  False),
        (10, False, 5,  True),
    ])
    def test_concurrent_matches_engine_condition(self, n_held, is_topup, cap, expected):
        # Engine inline: len(positions) >= cap AND NOT is_topup
        engine_result = (n_held >= cap) and (not is_topup)
        shared_result = concurrent_cap_exceeded(n_held, is_topup, cap)
        assert shared_result is engine_result is expected

    @pytest.mark.parametrize("pnl,cap,expected", [
        (0.0,    200.0, False),
        (-200.0, 200.0, False),
        (-200.1, 200.0, True),
        (-500.0, 200.0, True),
    ])
    def test_daily_loss_matches_engine_condition(self, pnl, cap, expected):
        # Engine inline: realized_pnl_today < -daily_loss_cap_usd
        engine_result = pnl < -cap
        shared_result = daily_loss_exceeded(pnl, cap)
        assert shared_result is engine_result is expected


class TestEquivalenceWithSimEntryVeto:
    """Confirm shared predicates agree with halt_replay.entry_veto's inline
    logic for the three cap checks (excluding halt windows, which are not
    part of the shared predicates)."""

    @pytest.mark.parametrize("held_inv,intent,cap,expected_veto", [
        (250.0, 40.0,  300.0, None),                    # 290 <= 300
        (250.0, 50.01, 300.0, "max_total_inventory"),   # 300.01 > 300
        (0.0,   0.0,   None,  None),                    # disabled
    ])
    def test_inventory_cap_matches_sim_veto(self, held_inv, intent, cap, expected_veto):
        from hlanalysis.backtest.halt_replay import EntryGateInputs, SimRiskCaps, entry_veto
        caps = SimRiskCaps(max_total_inventory_usd=cap)
        inp = EntryGateInputs(
            now_ns=0, intent_notional=intent, held_inventory_usd=held_inv,
            n_held_positions=0, is_topup=False, realized_pnl_window=0.0,
        )
        actual_veto = entry_veto(caps, [], inp)
        assert actual_veto == expected_veto
        # Also check shared predicate agrees
        shared = inventory_cap_exceeded(held_inv, intent, cap)
        assert shared == (expected_veto == "max_total_inventory")

    @pytest.mark.parametrize("n_held,is_topup,cap,expected_veto", [
        (2,  False, 3,  None),
        (3,  False, 3,  "max_concurrent_positions"),
        (3,  True,  3,  None),   # top-up exempt
        (0,  False, None, None), # disabled
    ])
    def test_concurrent_cap_matches_sim_veto(self, n_held, is_topup, cap, expected_veto):
        from hlanalysis.backtest.halt_replay import EntryGateInputs, SimRiskCaps, entry_veto
        caps = SimRiskCaps(max_concurrent_positions=cap)
        inp = EntryGateInputs(
            now_ns=0, intent_notional=0.0, held_inventory_usd=0.0,
            n_held_positions=n_held, is_topup=is_topup, realized_pnl_window=0.0,
        )
        actual_veto = entry_veto(caps, [], inp)
        assert actual_veto == expected_veto
        shared = concurrent_cap_exceeded(n_held, is_topup, cap)
        assert shared == (expected_veto == "max_concurrent_positions")

    @pytest.mark.parametrize("pnl,cap,expected_veto", [
        (-50.0,  100.0, None),
        (-100.0, 100.0, None),          # at cap, strict < → not blocked
        (-100.1, 100.0, "daily_loss_cap"),
        (0.0,    None,  None),           # disabled
    ])
    def test_daily_loss_matches_sim_veto(self, pnl, cap, expected_veto):
        from hlanalysis.backtest.halt_replay import EntryGateInputs, SimRiskCaps, entry_veto
        caps = SimRiskCaps(daily_loss_cap_usd=cap)
        inp = EntryGateInputs(
            now_ns=0, intent_notional=0.0, held_inventory_usd=0.0,
            n_held_positions=0, is_topup=False, realized_pnl_window=pnl,
        )
        actual_veto = entry_veto(caps, [], inp)
        assert actual_veto == expected_veto
        shared = daily_loss_exceeded(pnl, cap)
        assert shared == (expected_veto == "daily_loss_cap")
