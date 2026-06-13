"""Tests for Card D: Mispricing Surface.

Light tests:
- Schema / type validation on build_card return value
- Pure-function sanity on helper logic
- Data-dependent tests skip when ../../data is absent
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pandas as pd
import pytest

# Path to shared data root (relative to repo root / two levels above worktree)
_DATA_ROOT = Path(__file__).parents[3] / ".." / ".." / "data"
_DATA_AVAILABLE = _DATA_ROOT.exists()

_data_skip = pytest.mark.skipif(not _DATA_AVAILABLE, reason="../../data not present; skipping data-dependent test")


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


def _import_card():
    from hlanalysis.research.cards.card_d_mispricing import build_card

    return build_card


def _import_internals():
    from hlanalysis.research.cards.card_d_mispricing import (
        HL_FEE_PER_LEG,
        HL_FEE_ROUNDTRIP,
        _analyse_bucket_coherence,
        _analyse_overround,
        _compute_model_edge,
        _compute_taker_edge,
    )

    return (
        HL_FEE_PER_LEG,
        HL_FEE_ROUNDTRIP,
        _analyse_overround,
        _analyse_bucket_coherence,
        _compute_model_edge,
        _compute_taker_edge,
    )


# ---------------------------------------------------------------------------
# Tests: pure logic (no data)
# ---------------------------------------------------------------------------


class TestFeeConstants:
    def test_fee_per_leg_positive(self) -> None:
        HL_FEE_PER_LEG, HL_FEE_ROUNDTRIP, *_ = _import_internals()
        assert HL_FEE_PER_LEG > 0.0
        assert pytest.approx(HL_FEE_ROUNDTRIP) == 2 * HL_FEE_PER_LEG

    def test_fee_per_leg_reasonable(self) -> None:
        """Fee should be in range 0.001..0.01 for a sensible taker cost."""
        HL_FEE_PER_LEG, *_ = _import_internals()
        assert 0.001 <= HL_FEE_PER_LEG <= 0.01


class TestAnalyseOverround:
    def _make_df(self, yes_asks, no_asks, yes_bids=None, no_bids=None) -> pd.DataFrame:
        n = len(yes_asks)
        if yes_bids is None:
            yes_bids = [a - 0.01 for a in yes_asks]
        if no_bids is None:
            no_bids = [a - 0.01 for a in no_asks]
        return pd.DataFrame(
            {
                "yes_sym": ["#10"] * n,
                "no_sym": ["#11"] * n,
                "expiry_str": ["20260601-0600"] * n,
                "target_price": [80000.0] * n,
                "local_recv_ts": list(range(n)),
                "yes_ask": yes_asks,
                "yes_bid": yes_bids,
                "yes_ask_sz": [100.0] * n,
                "yes_bid_sz": [100.0] * n,
                "no_ask": no_asks,
                "no_bid": no_bids,
                "no_ask_sz": [100.0] * n,
                "no_bid_sz": [100.0] * n,
                "buy_both_cost": [y + n_ for y, n_ in zip(yes_asks, no_asks)],
                "sell_both_proceeds": [yb + nb for yb, nb in zip(yes_bids, no_bids)],
                "utc_hour": [12] * n,
                "tte_s": [3600.0] * n,
                "tte_h": [1.0] * n,
            }
        )

    def test_empty_df_returns_empty_dict(self) -> None:
        _, _, _analyse_overround, *_ = _import_internals()
        result = _analyse_overround(pd.DataFrame())
        assert result == {}

    def test_known_overround(self) -> None:
        """yes_ask=0.55 + no_ask=0.52 → buy_both=1.07 → overround=0.07."""
        _, _, _analyse_overround, *_ = _import_internals()
        df = self._make_df([0.55], [0.52])
        stats = _analyse_overround(df)
        assert abs(stats["overround_p50"] - 0.07) < 1e-6
        assert stats["n_expiries"] == 1
        assert stats["n_ticks"] == 1

    def test_buy_arb_detection(self) -> None:
        """When buy_both_cost < 1 - 2*fee, buy_arb_net_of_fee > 0."""
        HL_FEE_PER_LEG, _, _analyse_overround, *_ = _import_internals()
        # Very cheap: cost = 0.90 → well below 1 - 2*fee
        df = self._make_df([0.45], [0.45])
        stats = _analyse_overround(df)
        assert stats["frac_buy_arb_net_of_fee"] == 1.0

    def test_no_buy_arb_when_cost_high(self) -> None:
        HL_FEE_PER_LEG, _, _analyse_overround, *_ = _import_internals()
        # Normal: cost = 1.07 → no arb
        df = self._make_df([0.55], [0.52])
        stats = _analyse_overround(df)
        assert stats["frac_buy_arb_net_of_fee"] == 0.0

    def test_sell_arb_detection(self) -> None:
        """When sell proceeds > 1 + 2*fee, sell_arb_net_of_fee > 0."""
        HL_FEE_PER_LEG, _, _analyse_overround, *_ = _import_internals()
        # Sell both at proceeds = 1.10 → well above 1 + 2*fee
        df = self._make_df([0.50], [0.50], yes_bids=[0.57], no_bids=[0.53])
        df["sell_both_proceeds"] = df["yes_bid"] + df["no_bid"]
        stats = _analyse_overround(df)
        assert stats["frac_sell_arb_net_of_fee"] == 1.0


class TestAnalyseBucketCoherence:
    def _make_coh_df(self, sum_mids, sum_asks) -> pd.DataFrame:
        n = len(sum_mids)
        return pd.DataFrame(
            {
                "q_symbol": ["Q0"] * n,
                "expiry_str": ["20260601-0600"] * n,
                "ts_60s": list(range(n)),
                "mid0": [sm / 3 for sm in sum_mids],
                "ask0": [sa / 3 for sa in sum_asks],
                "mid1": [sm / 3 for sm in sum_mids],
                "ask1": [sa / 3 for sa in sum_asks],
                "mid2": [sm / 3 for sm in sum_mids],
                "ask2": [sa / 3 for sa in sum_asks],
                "sum_mids": sum_mids,
                "sum_asks": sum_asks,
            }
        )

    def test_empty_df_returns_empty_dict(self) -> None:
        *_, _analyse_bucket_coherence, _, _ = _import_internals()
        assert _analyse_bucket_coherence(pd.DataFrame()) == {}

    def test_coherent_market(self) -> None:
        """sum_mids = 1.0 → p50 = 1.0."""
        *_, _analyse_bucket_coherence, _, _ = _import_internals()
        df = self._make_coh_df([1.0] * 10, [1.05] * 10)
        stats = _analyse_bucket_coherence(df)
        assert abs(stats["sum_mids_p50"] - 1.0) < 1e-9

    def test_sub_arb_detection(self) -> None:
        HL_FEE_PER_LEG, _, _, _analyse_bucket_coherence, _, _ = _import_internals()
        # sum_asks = 0.95 → well below 1 - 3*fee
        df = self._make_coh_df([0.9] * 5, [0.95] * 5)
        stats = _analyse_bucket_coherence(df)
        assert stats["frac_sub_arb_gross"] == 1.0


class TestComputeModelEdge:
    def test_empty_inputs_return_model_df(self) -> None:
        *_, _compute_model_edge, _ = _import_internals()
        result = _compute_model_edge(pd.DataFrame(), pd.DataFrame())
        assert isinstance(result, pd.DataFrame)

    def test_sigma_join_and_gbm_prob(self) -> None:
        *_, _compute_model_edge, _ = _import_internals()
        # Simple 1-row case
        model_df = pd.DataFrame(
            {
                "symbol": ["#10"],
                "expiry_str": ["20260601-0600"],
                "target_price": [80000.0],
                "ts_ns": [1780000000000000000],
                "yes_mid": [0.55],
                "perp_mid": [82000.0],
                "tte_s": [7200.0],
            }
        )
        sigma_df = pd.DataFrame(
            {
                "ts_ns": [1779999000000000000],
                "sigma_annualised": [0.60],
            }
        )
        result = _compute_model_edge(model_df, sigma_df)
        assert "sigma" in result.columns
        assert "p_model" in result.columns
        assert "model_edge" in result.columns
        # sigma should be joined
        assert result.iloc[0]["sigma"] == pytest.approx(0.60)
        # p_model should be valid float in [0, 1]
        p_model = result.iloc[0]["p_model"]
        assert math.isfinite(p_model)
        assert 0.0 <= p_model <= 1.0
        # model_edge = yes_mid - p_model
        assert result.iloc[0]["model_edge"] == pytest.approx(0.55 - p_model, abs=1e-9)

    def test_model_edge_sign_above_strike(self) -> None:
        """Spot well above strike → p_model high → market mid (0.55) likely < p_model → negative edge."""
        *_, _compute_model_edge, _ = _import_internals()
        model_df = pd.DataFrame(
            {
                "symbol": ["#10"],
                "expiry_str": ["20260601-0600"],
                "target_price": [50000.0],
                "ts_ns": [1780000000000000000],
                "yes_mid": [0.55],
                "perp_mid": [82000.0],
                "tte_s": [7200.0],
            }
        )
        sigma_df = pd.DataFrame(
            {
                "ts_ns": [1779999000000000000],
                "sigma_annualised": [0.60],
            }
        )
        result = _compute_model_edge(model_df, sigma_df)
        # p_model for spot=82k vs strike=50k at sigma=0.6 should be ~1 → edge strongly negative
        assert result.iloc[0]["p_model"] > 0.90
        assert result.iloc[0]["model_edge"] < 0.0


class TestComputeTakerEdge:
    def test_empty_inputs_returns_dict(self) -> None:
        *_, _compute_taker_edge = _import_internals()
        result = _compute_taker_edge(pd.DataFrame(), pd.DataFrame())
        assert isinstance(result, dict)

    def test_buy_arb_detected(self) -> None:
        HL_FEE_PER_LEG, _, _analyse_overround, _, _, _compute_taker_edge = _import_internals()
        n = 10
        df = pd.DataFrame(
            {
                "yes_sym": ["#10"] * n,
                "no_sym": ["#11"] * n,
                "expiry_str": ["20260601-0600"] * n,
                "local_recv_ts": list(range(n)),
                "yes_ask": [0.40] * n,
                "yes_bid": [0.39] * n,
                "yes_ask_sz": [100.0] * n,
                "yes_bid_sz": [100.0] * n,
                "no_ask": [0.40] * n,
                "no_bid": [0.39] * n,
                "no_ask_sz": [100.0] * n,
                "no_bid_sz": [100.0] * n,
                "buy_both_cost": [0.80] * n,
                "sell_both_proceeds": [0.78] * n,
                "utc_hour": [12] * n,
            }
        )
        result = _compute_taker_edge(df, pd.DataFrame())
        assert result["buy_arb_freq"] == 1.0
        assert result["buy_arb_edge_median"] > 0


# ---------------------------------------------------------------------------
# Tests: build_card schema (skip without data)
# ---------------------------------------------------------------------------


class TestBuildCardSchema:
    @_data_skip
    def test_returns_tuple(self) -> None:
        build_card = _import_card()
        con = duckdb.connect()
        result = build_card(con, str(_DATA_ROOT))
        assert isinstance(result, tuple)
        assert len(result) == 2

    @_data_skip
    def test_html_is_string(self) -> None:
        build_card = _import_card()
        con = duckdb.connect()
        card_html, _ = build_card(con, str(_DATA_ROOT))
        assert isinstance(card_html, str)
        assert "<!DOCTYPE html>" in card_html

    @_data_skip
    def test_findings_schema(self) -> None:
        build_card = _import_card()
        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))

        # Required top-level keys
        required = {"title", "headline", "metrics", "split_half", "verdict", "fee_assumption"}
        assert required.issubset(set(findings.keys())), f"Missing keys: {required - set(findings.keys())}"

        # metrics is a list of dicts with required fields
        assert isinstance(findings["metrics"], list)
        assert len(findings["metrics"]) > 0
        for m in findings["metrics"]:
            assert "name" in m
            assert "value" in m
            assert "n" in m
            assert "date_span" in m

        # split_half has overround and model_edge sub-dicts
        assert "overround" in findings["split_half"]
        assert "model_edge" in findings["split_half"]

        # verdict is a non-empty string
        assert isinstance(findings["verdict"], str)
        assert len(findings["verdict"]) > 0

    @_data_skip
    def test_findings_coverage(self) -> None:
        """≥30 expiries in the full sample (KPI gate)."""
        build_card = _import_card()
        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))

        # Find overround_p50 metric
        overround_metric = next((m for m in findings["metrics"] if m["name"] == "overround_p50"), None)
        assert overround_metric is not None

        # n field contains tick count; expiry count is in the overround stats
        # At minimum the headline should mention n_expiries
        headline = findings["headline"]
        assert "n_binary_expiries=" in headline
        n_expiries = int(headline.split("n_binary_expiries=")[1].split(";")[0].split(",")[0].strip())
        assert n_expiries >= 30, f"Expected ≥30 expiries, got {n_expiries}"

    @_data_skip
    def test_split_half_both_halves_present(self) -> None:
        build_card = _import_card()
        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))

        split_or = findings["split_half"]["overround"]
        assert "first_half" in split_or
        assert "second_half" in split_or

        # Both halves should have data
        for half in ["first_half", "second_half"]:
            d = split_or[half]
            assert d.get("n", 0) > 0, f"{half} has no data"

    @_data_skip
    def test_fee_assumption_documented(self) -> None:
        build_card = _import_card()
        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))

        fa = findings["fee_assumption"]
        assert "model" in fa
        assert "per_leg" in fa
        assert "round_trip" in fa
        assert fa["per_leg"] > 0
        assert fa["round_trip"] == pytest.approx(2 * fa["per_leg"])
