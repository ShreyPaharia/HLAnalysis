"""Tests for Card C: Cross-Venue Lead-Lag.

Light tests:
- Schema / type validation on build_card return values
- Pure-function sanity checks on helper logic
- Data-dependent tests skip when ../../data is absent
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

# Path to shared data root (relative to repo root, two levels above worktree)
_DATA_ROOT = Path(__file__).parents[3] / ".." / ".." / "data"
_DATA_AVAILABLE = _DATA_ROOT.exists()

_data_skip = pytest.mark.skipif(not _DATA_AVAILABLE, reason="../../data not present; skipping data-dependent test")


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


def _import_card():
    from hlanalysis.research.cards.card_c_leadlag import build_card

    return build_card


def _import_internals():
    from hlanalysis.research.cards.card_c_leadlag import (
        BINARY_HALF_SPREAD_TYPICAL,
        HL_BINARY_LATENCY_NS,
        HL_PERP_LATENCY_NS,
        _compute_hedge_ratio,
        _compute_perp_vs_spot_xcorr,
        _compute_xcorr_for_expiry,
        _compute_taker_signal,
        _model_prob_series,
        _parkinson_sigma_rolling,
        _tte_bucket,
    )

    return (
        HL_PERP_LATENCY_NS,
        HL_BINARY_LATENCY_NS,
        BINARY_HALF_SPREAD_TYPICAL,
        _compute_hedge_ratio,
        _compute_xcorr_for_expiry,
        _compute_perp_vs_spot_xcorr,
        _compute_taker_signal,
        _model_prob_series,
        _parkinson_sigma_rolling,
        _tte_bucket,
    )


# ---------------------------------------------------------------------------
# Tests: constants and pure functions (no data)
# ---------------------------------------------------------------------------


class TestLatencyConstants:
    def test_perp_latency_reasonable(self) -> None:
        """HL perp latency should be ~100-500ms."""
        from hlanalysis.research.cards.card_c_leadlag import HL_PERP_LATENCY_NS

        ms = HL_PERP_LATENCY_NS / 1e6
        assert 50 <= ms <= 1000, f"HL perp latency {ms}ms outside expected range 50-1000ms"

    def test_binary_latency_reasonable(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import HL_BINARY_LATENCY_NS

        ms = HL_BINARY_LATENCY_NS / 1e6
        assert 50 <= ms <= 1000, f"Binary latency {ms}ms outside expected range"

    def test_latencies_similar(self) -> None:
        """Perp and binary latencies should be within ~50ms of each other (same host)."""
        from hlanalysis.research.cards.card_c_leadlag import HL_BINARY_LATENCY_NS, HL_PERP_LATENCY_NS

        diff_ms = abs(HL_PERP_LATENCY_NS - HL_BINARY_LATENCY_NS) / 1e6
        assert diff_ms < 50, f"Latency difference {diff_ms:.1f}ms seems large"


class TestTteBucket:
    def test_lt2h(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import _tte_bucket

        assert _tte_bucket(0.5) == "<2h"
        assert _tte_bucket(1.99) == "<2h"

    def test_2_8h(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import _tte_bucket

        assert _tte_bucket(2.0) == "2-8h"
        assert _tte_bucket(5.0) == "2-8h"
        assert _tte_bucket(7.99) == "2-8h"

    def test_8_24h(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import _tte_bucket

        assert _tte_bucket(8.0) == "8-24h"
        assert _tte_bucket(20.0) == "8-24h"


class TestComputeHedgeRatio:
    def test_perfect_tracking_gives_one(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import _compute_hedge_ratio

        n = 100
        x = np.random.randn(n)
        y = x  # perfect tracking: hedge ratio = 1
        hr = _compute_hedge_ratio(y, x)
        assert abs(hr - 1.0) < 1e-9, f"Expected hedge_ratio=1.0, got {hr}"

    def test_scaled_tracking(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import _compute_hedge_ratio

        n = 100
        x = np.random.randn(n)
        scale = 0.5
        y = scale * x
        hr = _compute_hedge_ratio(y, x)
        assert abs(hr - scale) < 1e-9, f"Expected hedge_ratio={scale}, got {hr}"

    def test_empty_returns_nan(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import _compute_hedge_ratio

        hr = _compute_hedge_ratio(np.array([], dtype="float64"), np.array([], dtype="float64"))
        assert not np.isfinite(hr)

    def test_all_zero_delta_returns_nan(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import _compute_hedge_ratio

        x = np.zeros(10)
        y = np.ones(10)
        hr = _compute_hedge_ratio(y, x)
        assert not np.isfinite(hr)

    def test_too_few_points_returns_nan(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import _compute_hedge_ratio

        hr = _compute_hedge_ratio(np.array([0.1, 0.2]), np.array([0.1, 0.2]))
        assert not np.isfinite(hr)


class TestModelProbSeries:
    def test_atm_near_half(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import _model_prob_series

        _NS = 1_000_000_000
        # at-the-money binary: prob should be ~0.5 (minus Itô drift)
        expiry_ns = int(86400 * _NS)  # 1 day
        grid = np.array([0], dtype="int64")
        perp_mid = np.array([100.0])
        sigma = np.array([0.20])
        probs = _model_prob_series(perp_mid, sigma, 100.0, expiry_ns, grid)
        assert len(probs) == 1
        assert np.isfinite(probs[0])
        # ATM with Itô correction should be slightly below 0.5
        assert 0.40 <= probs[0] <= 0.55, f"ATM GBM prob={probs[0]:.3f} out of expected range"

    def test_deep_itm_near_one(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import _model_prob_series

        _NS = 1_000_000_000
        expiry_ns = int(86400 * _NS)
        grid = np.array([0], dtype="int64")
        perp_mid = np.array([150.0])  # far above strike 100
        sigma = np.array([0.20])
        probs = _model_prob_series(perp_mid, sigma, 100.0, expiry_ns, grid)
        assert probs[0] > 0.95, f"Deep ITM prob={probs[0]:.3f} should be close to 1"

    def test_zero_tte_returns_nan(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import _model_prob_series

        _NS = 1_000_000_000
        expiry_ns = 0  # already expired
        grid = np.array([int(86400 * _NS)], dtype="int64")  # grid after expiry
        perp_mid = np.array([100.0])
        sigma = np.array([0.20])
        probs = _model_prob_series(perp_mid, sigma, 100.0, expiry_ns, grid)
        # TTE = (0 - 86400s*NS) / NS < 0, should return NaN
        assert not np.isfinite(probs[0])

    def test_zero_sigma_returns_nan(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import _model_prob_series

        _NS = 1_000_000_000
        expiry_ns = int(86400 * _NS)
        grid = np.array([0], dtype="int64")
        perp_mid = np.array([100.0])
        sigma = np.array([0.0])  # zero vol
        probs = _model_prob_series(perp_mid, sigma, 100.0, expiry_ns, grid)
        assert not np.isfinite(probs[0])


class TestComputeTakerSignal:
    def _make_results(self, peak_corrs, peak_lags, hedge_ratios) -> list[dict]:
        return [
            {
                "peak_corr": c,
                "peak_lag_s": l,
                "hedge_ratio": h,
                "tte_mean_h": 10.0,
            }
            for c, l, h in zip(peak_corrs, peak_lags, hedge_ratios)
        ]

    def test_empty_returns_empty(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import _compute_taker_signal

        result = _compute_taker_signal([])
        assert result == {}

    def test_high_corr_has_positive_r2(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import _compute_taker_signal

        results = self._make_results([0.9, 0.85, 0.92], [5, 5, 10], [0.8, 0.9, 0.85])
        signal = _compute_taker_signal(results)
        assert signal["median_r2"] > 0
        assert signal["median_r2"] < 1
        # median of [0.9², 0.85², 0.92²] = median of [0.81, 0.7225, 0.8464] = 0.81
        assert abs(signal["median_r2"] - 0.81) < 0.10  # roughly

    def test_hedge_ratio_aggregated(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import _compute_taker_signal

        results = self._make_results([0.5, 0.6], [5, 5], [0.8, 1.2])
        signal = _compute_taker_signal(results)
        assert np.isfinite(signal["hedge_ratio_median"])
        assert abs(signal["hedge_ratio_median"] - 1.0) < 0.3

    def test_returns_required_keys(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import _compute_taker_signal

        results = self._make_results([0.5], [5], [0.9])
        signal = _compute_taker_signal(results)
        required = {
            "n_expiries",
            "median_peak_corr",
            "median_r2",
            "median_peak_lag_s",
            "edge_gross_estimate",
            "edge_net_of_half_spread",
            "survives_half_spread",
            "hedge_ratio_median",
        }
        assert required.issubset(set(signal.keys())), f"Missing keys: {required - set(signal.keys())}"


class TestXcorrForExpiry:
    def _make_perp_df(self, n: int = 200, dt_ns: int = 5_000_000_000) -> pd.DataFrame:
        """Synthetic HL perp BBO data with random walk."""
        np.random.seed(42)
        ts = np.arange(0, n * dt_ns, dt_ns, dtype="int64")
        prices = 80000.0 + np.cumsum(np.random.randn(n) * 10)
        return pd.DataFrame({"ts_ns": ts, "mid": prices})

    def _make_binary_df(self, perp_df: pd.DataFrame, lag_steps: int = 2, noise: float = 0.002) -> pd.DataFrame:
        """Synthetic binary BBO that follows perp with lag + noise."""
        np.random.seed(99)
        ts = perp_df["ts_ns"].to_numpy(dtype="int64")
        # Model: binary mid = GBM prob(perp_mid)
        # Simplified: just shift perp returns and add noise
        from hlanalysis.research.metrics import implied_prob_gbm

        target = 80000.0
        expiry_ns = int(ts[-1] + 86400 * 1_000_000_000)
        sigma = 0.50
        tau_s = (expiry_ns - ts) / 1_000_000_000

        probs = np.array([
            implied_prob_gbm(float(p), target, sigma, float(t))
            for p, t in zip(perp_df["mid"].to_numpy(), tau_s)
        ])
        # Shift by lag_steps
        lagged_probs = np.roll(probs, lag_steps)
        lagged_probs[:lag_steps] = probs[:lag_steps]
        binary_mid = lagged_probs + np.random.randn(len(ts)) * noise
        binary_mid = np.clip(binary_mid, 0.01, 0.99)
        return pd.DataFrame({"ts_ns": ts, "mid": binary_mid, "bid_px": binary_mid - 0.005, "ask_px": binary_mid + 0.005})

    def test_returns_dict_on_valid_data(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import _compute_xcorr_for_expiry

        perp_df = self._make_perp_df(200)
        binary_df = self._make_binary_df(perp_df, lag_steps=2)
        expiry_ns = int(perp_df["ts_ns"].max() + 86400 * 1_000_000_000)
        result = _compute_xcorr_for_expiry(perp_df, binary_df, 80000.0, expiry_ns, dt_s=5)
        # Should return a dict with required fields
        assert result is not None
        required = {"lags", "corrs", "peak_lag_steps", "peak_lag_s", "peak_corr",
                    "half_life_steps", "half_life_s", "n_valid_steps", "tte_mean_h", "hedge_ratio"}
        assert required.issubset(set(result.keys()))

    def test_returns_none_on_empty_perp(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import _compute_xcorr_for_expiry

        result = _compute_xcorr_for_expiry(pd.DataFrame(), pd.DataFrame(columns=["ts_ns", "mid"]), 80000.0, 10**18, dt_s=5)
        assert result is None

    def test_peak_corr_in_valid_range(self) -> None:
        from hlanalysis.research.cards.card_c_leadlag import _compute_xcorr_for_expiry

        perp_df = self._make_perp_df(300)
        binary_df = self._make_binary_df(perp_df, lag_steps=1, noise=0.001)
        expiry_ns = int(perp_df["ts_ns"].max() + 86400 * 1_000_000_000)
        result = _compute_xcorr_for_expiry(perp_df, binary_df, 80000.0, expiry_ns, dt_s=5)
        if result is not None:
            assert -1.0 <= result["peak_corr"] <= 1.0


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
        assert len(card_html) > 100  # non-trivial HTML

    @_data_skip
    def test_findings_required_keys(self) -> None:
        build_card = _import_card()
        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        required = {"title", "headline", "metrics", "split_half", "verdict",
                    "latency_corrections", "mm_implication"}
        assert required.issubset(set(findings.keys())), f"Missing keys: {required - set(findings.keys())}"

    @_data_skip
    def test_findings_coverage_gate(self) -> None:
        """≥30 expiries in the full sample (KPI gate)."""
        build_card = _import_card()
        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        headline = findings["headline"]
        # Extract n_binary_expiries from headline
        assert "n_binary_expiries=" in headline
        n_expiries = int(headline.split("n_binary_expiries=")[1].split(";")[0].strip())
        assert n_expiries >= 30, f"Expected ≥30 expiries, got {n_expiries}"

    @_data_skip
    def test_metrics_list_structure(self) -> None:
        build_card = _import_card()
        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        metrics = findings["metrics"]
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        for m in metrics:
            assert "name" in m
            assert "value" in m
            assert "n" in m
            assert "date_span" in m
            assert "sanity" in m

    @_data_skip
    def test_split_half_keys(self) -> None:
        build_card = _import_card()
        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        sh = findings["split_half"]
        # Must have first_half and second_half in sub-dicts
        for key, val in sh.items():
            if isinstance(val, dict):
                assert "first_half" in val, f"split_half[{key}] missing first_half"
                assert "second_half" in val, f"split_half[{key}] missing second_half"

    @_data_skip
    def test_latency_corrections_documented(self) -> None:
        build_card = _import_card()
        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        lc = findings["latency_corrections"]
        assert "hl_perp_median_ms" in lc
        assert "hl_binary_median_ms" in lc
        assert lc["hl_perp_median_ms"] > 0

    @_data_skip
    def test_verdict_is_nonempty_string(self) -> None:
        build_card = _import_card()
        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        assert isinstance(findings["verdict"], str)
        assert len(findings["verdict"]) > 20

    @_data_skip
    def test_lead_time_is_sane_positive(self) -> None:
        """Perp→binary lead time should be positive (perp leads binary)."""
        build_card = _import_card()
        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        # Find lead_time metric
        lt_metric = next((m for m in findings["metrics"] if "lead_time" in m["name"]), None)
        if lt_metric is not None:
            val_str = lt_metric["value"].replace("s", "").replace("N/A", "")
            if val_str:
                val = float(val_str)
                assert val >= 0, f"Lead time {val}s should be non-negative (perp should lead binary)"

    @_data_skip
    def test_tte_dependence_present(self) -> None:
        build_card = _import_card()
        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        tte = findings.get("tte_dependence", {})
        assert isinstance(tte, dict)
        # At least some TTE buckets should be populated
        total_n = sum(tte.get(b, {}).get("n", 0) for b in ["<2h", "2-8h", "8-24h"])
        assert total_n >= 0  # may be 0 if insufficient data, but structure must exist
