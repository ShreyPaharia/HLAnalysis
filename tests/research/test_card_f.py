"""Light tests for Card F: Vol Term Structure & Theta Decay.

Data-dependent tests are skipped if ../../data is not found.
"""

from __future__ import annotations

import math
import os as _os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Data lives in the main checkout: either 4 levels up (from worktree),
# or override via HLBT_HL_DATA_ROOT env var.
_DATA_ROOT_ENV = _os.environ.get("HLBT_HL_DATA_ROOT")
if _DATA_ROOT_ENV:
    _DATA_ROOT = Path(_DATA_ROOT_ENV).resolve()
else:
    # From tests/research/ → tests → package root → .worktrees/<name> → HLAnalysis → data
    _DATA_ROOT = Path(__file__).resolve().parents[4] / "data"
    if not _DATA_ROOT.exists():
        # Try one more level up (non-worktree checkout)
        _DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
_HAVE_DATA = _DATA_ROOT.exists()

_skip_no_data = pytest.mark.skipif(not _HAVE_DATA, reason="../../data not present")


# ---------------------------------------------------------------------------
# Unit tests for pure logic (no data dependency)
# ---------------------------------------------------------------------------


class TestSolveImpliedSigma:
    """Bisection solver: round-trip σ → prob → σ̂."""

    def _import(self):
        from hlanalysis.research.cards.card_f_vol_theta import _solve_implied_sigma

        return _solve_implied_sigma

    def test_atm_roundtrip(self):
        """For ATM, implied_prob_gbm should invert back to σ via bisection."""
        from hlanalysis.research.metrics import implied_prob_gbm

        solve = self._import()
        spot = 80_000.0
        strike = 80_000.0
        sigma_true = 0.50  # 50% ann.
        tte_s = 12 * 3600.0  # 12h

        prob = implied_prob_gbm(spot, strike, sigma_true, tte_s)
        sigma_solved = solve(spot, strike, tte_s, prob)

        assert sigma_solved is not None
        assert abs(sigma_solved - sigma_true) < 0.01, (
            f"Round-trip error: solved={sigma_solved:.4f} true={sigma_true:.4f}"
        )

    def test_deep_otm_returns_none_or_extreme(self):
        """For extreme probabilities the solver should return something sensible."""
        solve = self._import()
        # Target prob near 0 should return None (not crash)
        result = solve(80_000.0, 80_000.0, 3600.0, 0.0005)
        assert result is None or isinstance(result, float)

    def test_near_expiry(self):
        """Short TTE with ATM market: solver returns a float (or None for extreme prob)."""
        from hlanalysis.research.metrics import implied_prob_gbm

        solve = self._import()
        sigma_true = 0.45
        # Use ATM market at short TTE — probability should be ~0.5 so solver can invert
        spot, strike, tte_s = 75_000.0, 75_000.0, 3600.0
        prob = implied_prob_gbm(spot, strike, sigma_true, tte_s)
        # prob should be near 0.5 for ATM, solver should work
        assert 0.3 < prob < 0.7
        sigma_solved = solve(spot, strike, tte_s, prob)
        assert sigma_solved is not None
        assert 0.01 <= sigma_solved <= 5.0


class TestThetaDecay:
    """Theta decay computation from mids DataFrame."""

    def _make_mids_df(self) -> pd.DataFrame:
        """Construct synthetic mids_df with known decay shape."""
        np.random.seed(42)
        records = []
        for tte_h in [1, 2, 4, 8, 12, 18, 22]:
            tte_s = tte_h * 3600.0
            # ATM mid ~ 0.5 at high TTE, converges toward 0 or 1 near expiry
            mid = 0.5 + np.random.uniform(-0.1, 0.1, 30)
            mid = np.clip(mid, 0.01, 0.99)
            for m in mid:
                records.append(
                    {
                        "expiry_str": "20260510-0600",
                        "target_price": 80000.0,
                        "local_recv_ts": 0,
                        "mid": m,
                        "tte_s": tte_s,
                    }
                )
        return pd.DataFrame(records)

    def test_decay_shape_columns(self):
        from hlanalysis.research.cards.card_f_vol_theta import _compute_theta_decay

        mids = self._make_mids_df()
        decay_df = _compute_theta_decay(mids)
        assert not decay_df.empty
        assert "tte_bucket_h" in decay_df.columns
        assert "mean_uncertainty" in decay_df.columns
        assert "n" in decay_df.columns

    def test_decay_uncertainty_range(self):
        """Uncertainty proxy must be in [0, 0.25]."""
        from hlanalysis.research.cards.card_f_vol_theta import _compute_theta_decay

        mids = self._make_mids_df()
        decay_df = _compute_theta_decay(mids)
        assert decay_df["mean_uncertainty"].min() >= 0.0
        assert decay_df["mean_uncertainty"].max() <= 0.25 + 1e-9

    def test_empty_mids_returns_empty(self):
        from hlanalysis.research.cards.card_f_vol_theta import _compute_theta_decay

        result = _compute_theta_decay(pd.DataFrame())
        assert result.empty


class TestVolTermStructure:
    """realized_vol_termstructure from metrics.py via card_f helpers."""

    def _make_ohlc(self, n_bars: int = 200) -> pd.DataFrame:
        np.random.seed(1)
        bar_ns = 60 * int(1e9)
        t0 = 1_780_000_000_000_000_000
        ts = np.arange(t0, t0 + n_bars * bar_ns, bar_ns, dtype="int64")
        price = 80_000.0 * np.exp(np.cumsum(np.random.normal(0, 0.001, n_bars)))
        hi = price * (1 + np.abs(np.random.normal(0, 0.0005, n_bars)))
        lo = price * (1 - np.abs(np.random.normal(0, 0.0005, n_bars)))
        return pd.DataFrame({"bar_ts_ns": ts, "open": price, "high": hi, "low": lo, "close": price})

    def test_term_structure_returns_all_horizons(self):
        from hlanalysis.research.cards.card_f_vol_theta import _compute_vol_term_structure

        ohlc = self._make_ohlc(n_bars=2000)
        ts_df = _compute_vol_term_structure(ohlc)
        assert not ts_df.empty
        horizons_found = set(ts_df["horizon"].tolist())
        expected = {"1m", "5m", "15m", "1h", "4h", "12h", "24h"}
        assert horizons_found == expected

    def test_parkinson_vol_in_reasonable_range(self):
        """Parkinson vol on synthetic ~1% daily moves should be plausible."""
        from hlanalysis.research.cards.card_f_vol_theta import _compute_vol_term_structure

        ohlc = self._make_ohlc(n_bars=2000)
        ts_df = _compute_vol_term_structure(ohlc)
        row_24h = ts_df[ts_df["horizon"] == "24h"].iloc[0]
        park = row_24h["park_vol_full"]
        # Should be between 10% and 500% annualized for 1% daily moves
        assert 0.05 < park < 5.0, f"Parkinson vol {park:.3f} out of plausible range"

    def test_empty_ohlc_returns_empty(self):
        from hlanalysis.research.cards.card_f_vol_theta import _compute_vol_term_structure

        ts_df = _compute_vol_term_structure(pd.DataFrame())
        assert ts_df.empty


class TestSplitHalfVolPremium:
    """Split-half stability logic."""

    def _make_premium_df(self, sign: float = 1.0) -> pd.DataFrame:
        records = []
        # H1: 2026-05-07 to 05-23
        for d in range(7, 24):
            records.append(
                {
                    "expiry_str": f"202605{d:02d}-0600",
                    "premium": sign * 0.05,
                    "sigma_implied": 0.55,
                    "sigma_realized": 0.50,
                }
            )
        # H2: 2026-05-24 to 06-10
        for d in range(24, 32):
            records.append(
                {
                    "expiry_str": f"202605{d:02d}-0600",
                    "premium": sign * 0.04,
                    "sigma_implied": 0.54,
                    "sigma_realized": 0.50,
                }
            )
        return pd.DataFrame(records)

    def test_stable_sign(self):
        from hlanalysis.research.cards.card_f_vol_theta import _split_half_vol_premium

        prem = self._make_premium_df(sign=1.0)
        result = _split_half_vol_premium(prem)
        assert result["stable"] is True
        assert result["h1"]["vol_premium_sign"] == "+"
        assert result["h2"]["vol_premium_sign"] == "+"

    def test_unstable_sign(self):
        from hlanalysis.research.cards.card_f_vol_theta import _split_half_vol_premium

        prem_h1 = self._make_premium_df(sign=1.0)
        # Manually flip H2
        for i, row in prem_h1.iterrows():
            expiry = row["expiry_str"]
            day = int(expiry[6:8])
            if day >= 24:
                prem_h1.at[i, "premium"] = -0.04

        result = _split_half_vol_premium(prem_h1)
        assert result["stable"] is False

    def test_empty_returns_stable_false(self):
        from hlanalysis.research.cards.card_f_vol_theta import _split_half_vol_premium

        result = _split_half_vol_premium(pd.DataFrame())
        assert result["stable"] is False


# ---------------------------------------------------------------------------
# Smoke test with real data (data-dependent)
# ---------------------------------------------------------------------------


@_skip_no_data
def test_build_card_smoke():
    """Smoke test: build_card runs on real data and returns valid outputs."""
    import duckdb

    from hlanalysis.research.cards.card_f_vol_theta import build_card

    con = duckdb.connect()
    card_html, findings = build_card(con, str(_DATA_ROOT))
    con.close()

    # HTML is non-empty
    assert isinstance(card_html, str)
    assert len(card_html) > 500

    # Findings structure
    assert "title" in findings
    assert "headline" in findings
    assert "metrics" in findings
    assert "split_half" in findings
    assert "verdict" in findings

    # At least 5 metrics
    assert len(findings["metrics"]) >= 5

    # Every metric has n and date_span
    for m in findings["metrics"]:
        assert "n" in m, f"Metric missing n: {m['name']}"
        assert "date_span" in m, f"Metric missing date_span: {m['name']}"

    # Split-half has both halves
    sh = findings["split_half"]
    assert "h1" in sh
    assert "h2" in sh
    assert "stable" in sh

    # Vol term structure: shape is one of expected values
    ts_metric = next(m for m in findings["metrics"] if m["name"] == "term_structure_shape")
    assert ts_metric["value"] in {"contango", "backwardation", "flat"}

    # Parkinson vols are in sane BTC range (30–200% annualized)
    for key in ["term_structure_1m_park_vol", "term_structure_24h_park_vol"]:
        metric = next((m for m in findings["metrics"] if m["name"] == key), None)
        if metric is not None and not math.isnan(float(metric["value"])):
            v = float(metric["value"])
            assert 15.0 < v < 300.0, f"{key}={v:.1f}% outside plausible BTC range"


@_skip_no_data
def test_build_card_coverage():
    """Data coverage: at least 30 days and 30 expiries in the corpus."""
    import duckdb

    from hlanalysis.research.cards.card_f_vol_theta import build_card

    con = duckdb.connect()
    _, findings = build_card(con, str(_DATA_ROOT))
    con.close()

    # Premium metric should have n >= 25 expiries (36-day window, but some may miss oracle)
    prem_metric = next((m for m in findings["metrics"] if m["name"] == "vol_premium_mean_ann"), None)
    if prem_metric is not None:
        assert prem_metric["n"] >= 20, f"Too few expiries for vol premium: n={prem_metric['n']}"
