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


class TestFlbParams:
    """Verify FLB param builder produces safe/consistent dicts."""

    def test_flb_params_defaults(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import _flb_params

        p = _flb_params()
        assert p["price_extreme_threshold"] == pytest.approx(0.80)
        assert p["price_extreme_max"] == pytest.approx(0.95)
        assert p["tte_min_seconds"] == 3600
        assert p["tte_max_seconds"] == 6 * 3600

    def test_flb_params_custom(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import _flb_params

        p = _flb_params(price_lo=0.82, price_hi=0.92, tte_min_h=1.5, tte_max_h=4.0)
        assert p["price_extreme_threshold"] == pytest.approx(0.82)
        assert p["price_extreme_max"] == pytest.approx(0.92)
        assert p["tte_min_seconds"] == int(1.5 * 3600)
        assert p["tte_max_seconds"] == int(4.0 * 3600)

    def test_safety_gates_preserved(self) -> None:
        """Safety gates must not be disabled in FLB params."""
        from hlanalysis.research.cards.strategy_taker_flb import _flb_params

        p = _flb_params()
        # min_safety_d gate preserved
        assert p["min_safety_d"] >= 2.0, "min_safety_d gate should be ≥2σ"
        # exit_safety_d preserved
        assert p["exit_safety_d"] >= 0.5, "exit_safety_d should be active"
        # vol_max not removed
        assert p["vol_max"] > 0, "vol_max gate should be active"
        # min_bid_notional spoof filter
        assert p["min_bid_notional_usd"] >= 10.0, "spoof filter should be active"

    def test_max_position_usd_param(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import _flb_params

        p = _flb_params(max_position_usd=250.0)
        assert p["max_position_usd"] == pytest.approx(250.0)


class TestVolScaledParams:
    """Vol-regime sizing helper tests."""

    def test_flat_when_sigma_zero(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import (
            _flb_params,
            _vol_scaled_params,
        )

        base = _flb_params(max_position_usd=100.0)
        p = _vol_scaled_params(base, sigma=0.0, sigma_median=0.5, max_position_usd=100.0)
        assert p["max_position_usd"] == pytest.approx(100.0)

    def test_scales_up_high_vol(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import (
            _flb_params,
            _vol_scaled_params,
        )

        base = _flb_params(max_position_usd=100.0)
        # sigma 2× median → scale = 2.0 (at cap)
        p = _vol_scaled_params(base, sigma=1.0, sigma_median=0.5, max_position_usd=100.0)
        assert p["max_position_usd"] == pytest.approx(200.0)

    def test_scales_down_low_vol(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import (
            _flb_params,
            _vol_scaled_params,
        )

        base = _flb_params(max_position_usd=100.0)
        # sigma 0.1× median → scale capped at 0.5 (1/scale_cap=2)
        p = _vol_scaled_params(base, sigma=0.05, sigma_median=0.5, max_position_usd=100.0)
        assert p["max_position_usd"] == pytest.approx(50.0)

    def test_cap_prevents_extreme_scaling(self) -> None:
        from hlanalysis.research.cards.strategy_taker_flb import (
            _flb_params,
            _vol_scaled_params,
        )

        base = _flb_params(max_position_usd=100.0)
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
        """$25k desk with 1 market/day should saturate."""
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
            depth_per_level_hi=200.0,
            levels=3,
        )
        # clip = $25k / 1 market = $25k >> $600 available → saturates
        assert rows[0]["saturates"] is True

    def test_no_saturation_small_desk(self) -> None:
        """$500 desk with 7 markets should not saturate at best depth."""
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
            depth_per_level_hi=200.0,
            levels=3,
        )
        # clip = $500 / 7 = $71 << $600 → no saturation at hi depth
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


class TestFLBBaseParams:
    """FLB_BASE_PARAMS is a valid v1_late_resolution params dict."""

    def test_can_build_strategy(self) -> None:
        """FLB params should be accepted by the strategy registry builder."""
        import hlanalysis.strategy  # noqa: F401  — ensure strategies registered
        from hlanalysis.backtest.runner.parallel import build_strategy_for_run
        from hlanalysis.research.cards.strategy_taker_flb import _flb_params

        p = _flb_params()
        # Should not raise
        strategy = build_strategy_for_run("v1_late_resolution", p)
        assert strategy is not None

    def test_strategy_type(self) -> None:
        import hlanalysis.strategy  # noqa: F401
        from hlanalysis.backtest.runner.parallel import build_strategy_for_run
        from hlanalysis.research.cards.strategy_taker_flb import _flb_params
        from hlanalysis.strategy.late_resolution import LateResolutionStrategy

        p = _flb_params()
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

    def test_run_flb_oos_smoke(self) -> None:
        """FLB OOS run on ≤3 markets should complete without error."""
        from hlanalysis.research.cards.strategy_taker_flb import (
            OOS_END,
            OOS_START,
            _discover_questions,
            _flb_params,
            _run_strategy,
            _summarise,
        )

        questions = _discover_questions(str(_DATA_ROOT), OOS_START, OOS_END)[:3]
        if not questions:
            pytest.skip("No OOS questions available")

        params = _flb_params()
        pnl_list, n_trades, outcomes = _run_strategy(
            strategy_id="v1_late_resolution",
            params=params,
            questions=questions,
            data_root=str(_DATA_ROOT),
            n_workers=1,
        )
        # Should complete; result types are correct
        assert isinstance(pnl_list, list)
        assert len(pnl_list) == len(questions)
        assert isinstance(n_trades, int)
        assert n_trades >= 0

        stats = _summarise(pnl_list, n_trades)
        assert isinstance(stats["total_pnl_usd"], float)
        assert isinstance(stats["sharpe"], float)

    def test_build_card_no_sweep_smoke(self) -> None:
        """build_card(run_sweep=False) should complete and return valid findings."""
        from hlanalysis.research.cards.strategy_taker_flb import build_card

        html, findings = build_card(data_root=str(_DATA_ROOT), run_sweep=False)

        # HTML is a non-empty string
        assert isinstance(html, str)
        assert len(html) > 500

        # Findings has required keys
        assert "kpis" in findings
        assert "flb_oos" in findings
        assert "split_half" in findings
        assert "capacity_table" in findings
        assert "overall_pass" in findings

        # Per-KPI PASS/FAIL are booleans
        for kpi_key, kpi_val in findings["kpis"].items():
            assert "pass" in kpi_val, f"KPI {kpi_key} missing 'pass' key"
            assert isinstance(kpi_val["pass"], bool), f"KPI {kpi_key}.pass should be bool"

        # Output file written
        out_path = Path(__file__).parent.parent.parent / "docs" / "research" / "_cards" / "strategy_taker.html"
        assert out_path.exists(), f"HTML output not written to {out_path}"

    def test_findings_json_serializable(self) -> None:
        """Findings dict must be JSON-serializable."""
        from hlanalysis.research.cards.strategy_taker_flb import build_card

        _, findings = build_card(data_root=str(_DATA_ROOT), run_sweep=False)
        # Should not raise
        json_str = json.dumps(findings)
        assert len(json_str) > 100

    def test_n_oos_markets_ge_1(self) -> None:
        """OOS should have ≥1 market (basic sanity on data coverage)."""
        from hlanalysis.research.cards.strategy_taker_flb import build_card

        _, findings = build_card(data_root=str(_DATA_ROOT), run_sweep=False)
        assert findings["n_oos_markets"] >= 1, (
            f"Expected ≥1 OOS market, got {findings['n_oos_markets']}. Check data coverage for 2026-06-04..06-10."
        )
