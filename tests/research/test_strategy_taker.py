"""Light tests for the FLB taker strategy card.

Tests that run without real data (structural / API checks) are always active.
Data-dependent integration tests are skipped when the recorded data is absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Data availability detection
# ---------------------------------------------------------------------------

_WORKTREE_ROOT = Path(__file__).parent.parent.parent

if os.environ.get("HLBT_HL_DATA_ROOT"):
    _DATA_ROOT = Path(os.environ["HLBT_HL_DATA_ROOT"]).resolve()
else:
    _DATA_ROOT = (_WORKTREE_ROOT / ".." / ".." / "data").resolve()

_HL_MARKER = _DATA_ROOT / "venue=hyperliquid"
_DATA_AVAILABLE = _HL_MARKER.exists()


# ---------------------------------------------------------------------------
# Unit tests — no data required
# ---------------------------------------------------------------------------


class TestCleanFlbParams:
    """Verify CLEAN FLB param builder has no σ-distance gate (primary research object)."""

    def test_clean_flb_params_defaults(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import _clean_flb_params

        p = _clean_flb_params()
        assert p["price_extreme_threshold"] == pytest.approx(0.80)
        assert p["price_extreme_max"] == pytest.approx(0.95)
        assert p["tte_min_seconds"] == 3600
        assert p["tte_max_seconds"] == 6 * 3600

    def test_clean_flb_no_safety_d_gate(self) -> None:
        """Clean FLB must have min_safety_d=0 — that is the research object."""
        from hlanalysis.research.cards.strategy_taker_flb import _clean_flb_params

        p = _clean_flb_params()
        assert p["min_safety_d"] == pytest.approx(0.0), "clean FLB must have min_safety_d=0.0 to measure gross FLB edge"

    def test_clean_flb_other_gates_preserved(self) -> None:
        """Non-σ-distance safety gates must remain active in clean FLB."""
        from hlanalysis.research.cards.strategy_taker_flb import _clean_flb_params

        p = _clean_flb_params()
        # vol_max still caps extreme vol
        assert p["vol_max"] > 0, "vol_max gate should be active"
        # spoof filter
        assert p["min_bid_notional_usd"] >= 10.0, "spoof filter should be active"
        # exit_safety_d still active (mid-hold protection)
        assert p["exit_safety_d"] >= 0.5, "exit_safety_d should be active"

    def test_clean_flb_params_custom(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import _clean_flb_params

        p = _clean_flb_params(price_lo=0.82, price_hi=0.92, tte_min_h=1.5, tte_max_h=4.0)
        assert p["price_extreme_threshold"] == pytest.approx(0.82)
        assert p["price_extreme_max"] == pytest.approx(0.92)
        assert p["tte_min_seconds"] == int(1.5 * 3600)
        assert p["tte_max_seconds"] == int(4.0 * 3600)

    def test_clean_flb_max_position_usd_param(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import _clean_flb_params

        p = _clean_flb_params(max_position_usd=250.0)
        assert p["max_position_usd"] == pytest.approx(250.0)


class TestGatedFlbParams:
    """Verify GATED FLB param builder preserves min_safety_d=3.0 (live comparison)."""

    def test_gated_flb_params_defaults(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import _flb_params

        p = _flb_params()
        assert p["price_extreme_threshold"] == pytest.approx(0.80)
        assert p["price_extreme_max"] == pytest.approx(0.95)
        assert p["tte_min_seconds"] == 3600
        assert p["tte_max_seconds"] == 6 * 3600

    def test_gated_flb_has_safety_d_gate(self) -> None:
        """Gated FLB must keep min_safety_d=3.0 for live comparison."""
        from hlanalysis.research.cards.strategy_taker_flb import _flb_params

        p = _flb_params()
        assert p["min_safety_d"] >= 2.0, "gated FLB should have min_safety_d ≥ 2σ (live gate)"

    def test_gated_flb_safety_gates_preserved(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import _flb_params

        p = _flb_params()
        assert p["exit_safety_d"] >= 0.5, "exit_safety_d should be active"
        assert p["vol_max"] > 0, "vol_max gate should be active"
        assert p["min_bid_notional_usd"] >= 10.0, "spoof filter should be active"

    def test_flb_base_params_is_gated(self) -> None:
        """FLB_BASE_PARAMS alias is the gated variant (backward compat)."""
        from hlanalysis.research.cards.strategy_taker_flb import FLB_BASE_PARAMS

        assert FLB_BASE_PARAMS["min_safety_d"] >= 2.0, "FLB_BASE_PARAMS should be gated variant"


class TestParamDifferences:
    """Clean and gated FLB differ only in min_safety_d."""

    def test_only_min_safety_d_differs(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import (
            _clean_flb_params,
            _flb_params,
        )

        clean = _clean_flb_params()
        gated = _flb_params()
        diffs = {k for k in clean if clean.get(k) != gated.get(k)}
        assert diffs == {"min_safety_d"}, (
            f"Clean and gated FLB should differ only in min_safety_d, but differ in: {diffs}"
        )


class TestVolScaledParams:
    """Vol-regime sizing helper tests."""

    def test_flat_when_sigma_zero(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import (
            _clean_flb_params,
            _vol_scaled_params,
        )

        base = _clean_flb_params(max_position_usd=100.0)
        p = _vol_scaled_params(base, sigma=0.0, sigma_median=0.5, max_position_usd=100.0)
        assert p["max_position_usd"] == pytest.approx(100.0)

    def test_scales_up_high_vol(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import (
            _clean_flb_params,
            _vol_scaled_params,
        )

        base = _clean_flb_params(max_position_usd=100.0)
        # sigma 2× median → scale = 2.0 (at cap)
        p = _vol_scaled_params(base, sigma=1.0, sigma_median=0.5, max_position_usd=100.0)
        assert p["max_position_usd"] == pytest.approx(200.0)

    def test_scales_down_low_vol(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import (
            _clean_flb_params,
            _vol_scaled_params,
        )

        base = _clean_flb_params(max_position_usd=100.0)
        # sigma 0.1× median → scale capped at 0.5 (1/scale_cap=2)
        p = _vol_scaled_params(base, sigma=0.05, sigma_median=0.5, max_position_usd=100.0)
        assert p["max_position_usd"] == pytest.approx(50.0)

    def test_cap_prevents_extreme_scaling(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import (
            _clean_flb_params,
            _vol_scaled_params,
        )

        base = _clean_flb_params(max_position_usd=100.0)
        # sigma 100× median → should be capped at 2×
        p = _vol_scaled_params(base, sigma=50.0, sigma_median=0.5, max_position_usd=100.0, scale_cap=2.0)
        assert p["max_position_usd"] == pytest.approx(200.0)


class TestCapacityTable:
    """Capacity model structural tests."""

    def test_returns_three_rows(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import _capacity_table

        fake_stats = {
            "total_pnl_usd": 50.0,
            "n_markets": 7,
            "sharpe": 2.0,
            "hit_rate": 0.7,
            "max_drawdown_usd": 10.0,
            "n_trades": 7,
        }
        rows = _capacity_table(fake_stats, desk_sizes=(1_000.0, 5_000.0, 25_000.0))
        assert len(rows) == 3
        assert rows[0]["desk_usd"] == 1_000.0
        assert rows[-1]["desk_usd"] == 25_000.0

    def test_saturation_flag(self) -> None:
        """$25k desk with 1 market/day should saturate (Card A hi depth $679 < $25k)."""
        from hlanalysis.research.cards.strategy_taker_flb import _capacity_table

        fake_stats = {
            "total_pnl_usd": 50.0,
            "n_markets": 1,
            "sharpe": 2.0,
            "hit_rate": 0.7,
            "max_drawdown_usd": 10.0,
            "n_trades": 7,
        }
        rows = _capacity_table(
            fake_stats,
            desk_sizes=(25_000.0,),
            depth_per_level_hi=679.0,
            levels=1,
        )
        # clip = $25k / 1 market = $25k >> $679 available → saturates
        assert rows[0]["saturates"] is True

    def test_no_saturation_small_desk(self) -> None:
        """$500 desk with 7 markets should not saturate at Card A hi depth."""
        from hlanalysis.research.cards.strategy_taker_flb import _capacity_table

        fake_stats = {
            "total_pnl_usd": 50.0,
            "n_markets": 7,
            "sharpe": 2.0,
            "hit_rate": 0.7,
            "max_drawdown_usd": 10.0,
            "n_trades": 7,
        }
        rows = _capacity_table(
            fake_stats,
            desk_sizes=(500.0,),
            depth_per_level_hi=679.0,
            levels=1,
        )
        # clip = $500 / 7 ≈ $71 << $679 → no saturation at hi depth
        assert rows[0]["fill_frac_hi"] == pytest.approx(1.0)

    def test_fill_frac_bounded(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import _capacity_table

        fake_stats = {
            "total_pnl_usd": 10.0,
            "n_markets": 5,
            "sharpe": 1.5,
            "hit_rate": 0.6,
            "max_drawdown_usd": 5.0,
            "n_trades": 5,
        }
        for desk in (1_000.0, 5_000.0, 25_000.0):
            rows = _capacity_table(fake_stats, desk_sizes=(desk,))
            assert 0 <= rows[0]["fill_frac_lo"] <= 1.0
            assert 0 <= rows[0]["fill_frac_hi"] <= 1.0

    def test_card_a_depths_used_as_defaults(self) -> None:
        """Capacity model defaults should use Card A actual depth measurements."""
        from hlanalysis.research.cards.strategy_taker_flb import (
            CARD_A_TOB_USD,
            CARD_A_WITHIN_100BPS_USD,
            _capacity_table,
        )

        fake_stats = {
            "total_pnl_usd": 50.0,
            "n_markets": 7,
            "sharpe": 2.0,
            "hit_rate": 0.7,
            "max_drawdown_usd": 10.0,
            "n_trades": 7,
        }
        rows = _capacity_table(fake_stats, desk_sizes=(1_000.0,))
        # $1k / 7 markets = $143/clip < $107 TOB → fill_frac_lo could be < 1
        # Just check the function runs with Card A defaults
        assert len(rows) == 1
        assert pytest.approx(107.0) == CARD_A_TOB_USD
        assert pytest.approx(679.0) == CARD_A_WITHIN_100BPS_USD


class TestSummariseHelper:
    """Test _summarise wraps summarise_run correctly."""

    def test_empty(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import _summarise

        s = _summarise([], 0)
        assert s["n_markets"] == 0
        assert s["total_pnl_usd"] == 0.0
        assert s["sharpe"] == 0.0

    def test_positive_pnl(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import _summarise

        s = _summarise([10.0, 5.0, 8.0], 3)
        assert s["total_pnl_usd"] == pytest.approx(23.0)
        assert s["n_trades"] == 3
        assert s["sharpe"] > 0
        assert s["hit_rate"] == pytest.approx(1.0)
        assert s["max_drawdown_usd"] == pytest.approx(0.0)

    def test_mixed_pnl(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import _summarise

        s = _summarise([10.0, -5.0, 8.0], 3)
        assert s["total_pnl_usd"] == pytest.approx(13.0)
        assert s["hit_rate"] == pytest.approx(2 / 3)

    def test_max_drawdown(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import _summarise

        # Sequence: +10, -8, +5 → peak 10, then 2 → DD = 8
        s = _summarise([10.0, -8.0, 5.0], 3)
        assert s["max_drawdown_usd"] == pytest.approx(8.0)


class TestUnderpoweredNote:
    """Test statistical power helper."""

    def test_underpowered_below_threshold(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import _underpowered_note

        note = _underpowered_note(5)
        assert "UNDERPOWERED" in note
        assert "5" in note

    def test_powered_above_threshold(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import _underpowered_note

        note = _underpowered_note(20)
        assert "UNDERPOWERED" not in note
        assert "20" in note


class TestFLBBaseParamsStrategy:
    """FLB params are valid v1_late_resolution params dicts."""

    def test_can_build_clean_flb_strategy(self) -> None:
        """Clean FLB params should be accepted by the strategy registry builder."""
        import hlanalysis.strategy  # noqa: F401  — ensure strategies registered
        from hlanalysis.backtest.runner.parallel import build_strategy_for_run
        from hlanalysis.research.cards.strategy_taker_flb import _clean_flb_params

        p = _clean_flb_params()
        strategy = build_strategy_for_run("v1_late_resolution", p)
        assert strategy is not None

    def test_can_build_gated_flb_strategy(self) -> None:
        """Gated FLB params should be accepted by the strategy registry builder."""
        import hlanalysis.strategy  # noqa: F401
        from hlanalysis.backtest.runner.parallel import build_strategy_for_run
        from hlanalysis.research.cards.strategy_taker_flb import _flb_params

        p = _flb_params()
        strategy = build_strategy_for_run("v1_late_resolution", p)
        assert strategy is not None

    def test_clean_flb_strategy_type(self) -> None:
        import hlanalysis.strategy  # noqa: F401
        from hlanalysis.backtest.runner.parallel import build_strategy_for_run
        from hlanalysis.research.cards.strategy_taker_flb import _clean_flb_params
        from hlanalysis.strategy.late_resolution import LateResolutionStrategy

        p = _clean_flb_params()
        strategy = build_strategy_for_run("v1_late_resolution", p)
        assert isinstance(strategy, LateResolutionStrategy)


# ---------------------------------------------------------------------------
# Integration tests — require recorded data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="Recorded data not available at ../../data")
class TestFlbCardIntegration:
    """End-to-end integration tests against real recorded data."""

    def test_discover_questions_is(self) -> None:
        """IS range should yield ≥ 5 binary questions."""
        from hlanalysis.research.cards.strategy_taker_flb import (
            IS_END,
            IS_START,
            _discover_questions,
        )

        qs = _discover_questions(str(_DATA_ROOT), IS_START, IS_END)
        assert len(qs) >= 5, f"Expected ≥5 IS questions, got {len(qs)}"

    def test_discover_questions_oos(self) -> None:
        """OOS range should yield ≥ 1 binary question."""
        from hlanalysis.research.cards.strategy_taker_flb import (
            OOS_END,
            OOS_START,
            _discover_questions,
        )

        qs = _discover_questions(str(_DATA_ROOT), OOS_START, OOS_END)
        assert len(qs) >= 1, f"Expected ≥1 OOS question, got {len(qs)}"

    def test_run_clean_flb_oos_smoke(self) -> None:
        """Clean FLB OOS run on ≤3 markets should complete without error."""
        from hlanalysis.research.cards.strategy_taker_flb import (
            OOS_END,
            OOS_START,
            _clean_flb_params,
            _discover_questions,
            _run_strategy,
            _summarise,
        )

        questions = _discover_questions(str(_DATA_ROOT), OOS_START, OOS_END)[:3]
        if not questions:
            pytest.skip("No OOS questions available")

        params = _clean_flb_params()
        pnl_list, n_trades, outcomes = _run_strategy(
            strategy_id="v1_late_resolution",
            params=params,
            questions=questions,
            data_root=str(_DATA_ROOT),
            n_workers=1,
        )
        assert isinstance(pnl_list, list)
        assert len(pnl_list) == len(questions)
        assert isinstance(n_trades, int)
        assert n_trades >= 0

        stats = _summarise(pnl_list, n_trades)
        assert isinstance(stats["total_pnl_usd"], float)
        assert isinstance(stats["sharpe"], float)

    def test_clean_flb_more_trades_than_gated(self) -> None:
        """Clean FLB (no safety_d gate) should yield ≥ as many trades as gated FLB."""
        from hlanalysis.research.cards.strategy_taker_flb import (
            OOS_END,
            OOS_START,
            _clean_flb_params,
            _discover_questions,
            _flb_params,
            _run_strategy,
        )

        questions = _discover_questions(str(_DATA_ROOT), OOS_START, OOS_END)
        if not questions:
            pytest.skip("No OOS questions available")

        clean_params = _clean_flb_params()
        gated_params = _flb_params()

        _, n_trades_clean, _ = _run_strategy(
            strategy_id="v1_late_resolution",
            params=clean_params,
            questions=questions,
            data_root=str(_DATA_ROOT),
            n_workers=1,
        )
        _, n_trades_gated, _ = _run_strategy(
            strategy_id="v1_late_resolution",
            params=gated_params,
            questions=questions,
            data_root=str(_DATA_ROOT),
            n_workers=1,
        )
        assert n_trades_clean >= n_trades_gated, (
            f"Clean FLB should have ≥ trades as gated FLB: clean={n_trades_clean}, gated={n_trades_gated}"
        )

    def test_build_card_no_sweep_smoke(self) -> None:
        """build_card(run_sweep=False) should complete and return valid findings."""
        from hlanalysis.research.cards.strategy_taker_flb import build_card

        html, findings = build_card(data_root=str(_DATA_ROOT), run_sweep=False)

        # HTML is a non-empty string
        assert isinstance(html, str)
        assert len(html) > 500

        # Findings has required keys
        assert "kpis" in findings
        assert "clean_flb" in findings
        assert "gated_flb" in findings
        assert "gate_tradeoff" in findings
        assert "split_half" in findings
        assert "capacity_table" in findings
        assert "overall_pass" in findings
        assert isinstance(findings["overall_pass"], bool)

        # Per-KPI PASS/FAIL are booleans
        for kpi_key, kpi_val in findings["kpis"].items():
            assert "pass" in kpi_val, f"KPI {kpi_key} missing 'pass' key"
            assert isinstance(kpi_val["pass"], bool), f"KPI {kpi_key}.pass should be bool"

        # No NULLs in headline fields
        assert findings["clean_flb"]["oos"]["total_pnl_usd"] is not None
        assert findings["clean_flb"]["oos"]["sharpe"] is not None
        assert findings["clean_flb"]["oos"]["n_trades"] is not None

        # Output file written
        out_path = Path(__file__).parent.parent.parent / "docs" / "research" / "_cards" / "strategy_taker.html"
        assert out_path.exists(), f"HTML output not written to {out_path}"

    def test_findings_json_serializable(self) -> None:
        """Findings dict must be JSON-serializable."""
        from hlanalysis.research.cards.strategy_taker_flb import build_card

        _, findings = build_card(data_root=str(_DATA_ROOT), run_sweep=False)
        json_str = json.dumps(findings)
        assert len(json_str) > 100

    def test_n_oos_markets_ge_1(self) -> None:
        """OOS should have ≥1 market (basic sanity on data coverage)."""
        from hlanalysis.research.cards.strategy_taker_flb import build_card

        _, findings = build_card(data_root=str(_DATA_ROOT), run_sweep=False)
        assert findings["n_oos_markets"] >= 1, (
            f"Expected ≥1 OOS market, got {findings['n_oos_markets']}. Check data coverage for 2026-06-04..06-10."
        )

    def test_gate_tradeoff_non_negative_cost(self) -> None:
        """Safety gate can only reduce or keep same PnL vs clean FLB (never adds)."""
        from hlanalysis.research.cards.strategy_taker_flb import build_card

        _, findings = build_card(data_root=str(_DATA_ROOT), run_sweep=False)
        gt = findings["gate_tradeoff"]
        # Gate cost should be ≥ 0 (gate never adds PnL vs clean)
        # Allow small floating point noise
        assert gt["oos_pnl_cost"] >= -0.01, (
            f"Gate cost should be ≥ 0 (gate can only reduce/equal PnL), got {gt['oos_pnl_cost']:.4f}"
        )
        assert gt["oos_trades_filtered"] >= 0, "Filtered trades should be ≥ 0"

    def test_clean_flb_n_trades_reported(self) -> None:
        """Clean FLB OOS n_trades should be explicitly reported in findings."""
        from hlanalysis.research.cards.strategy_taker_flb import build_card

        _, findings = build_card(data_root=str(_DATA_ROOT), run_sweep=False)
        n_trades = findings["clean_flb"]["oos"]["n_trades"]
        assert isinstance(n_trades, int)
        assert n_trades >= 0
        # Log whether powered
        powered = n_trades >= 15
        if not powered:
            pytest.warns(None)  # no assertion, just log for visibility
