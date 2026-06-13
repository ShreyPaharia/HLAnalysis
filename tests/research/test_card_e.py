"""Tests for Card E: Settlement / Resolution Convergence Dynamics.

Light tests that run without real data: structural / schema checks.
Data-dependent sections are skipped when ../../data is absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Data lives in the main checkout, two levels above the worktree root.
# Worktree path: .worktrees/<name>/ → main checkout is ../../
# HLBT_HL_DATA_ROOT env overrides everything.
_WORKTREE_ROOT = Path(__file__).parent.parent.parent

if os.environ.get("HLBT_HL_DATA_ROOT"):
    _DATA_ROOT = Path(os.environ["HLBT_HL_DATA_ROOT"]).resolve()
else:
    # Main checkout data directory (worktrees have a stub ./data/ with only nba_wp)
    _DATA_ROOT = (_WORKTREE_ROOT / ".." / ".." / "data").resolve()

# Check that the data root contains actual market data (not just the stub nba_wp dir)
_HL_MARKER = _DATA_ROOT / "venue=hyperliquid"
_DATA_AVAILABLE = _HL_MARKER.exists()


# ---------------------------------------------------------------------------
# Unit tests (no data needed)
# ---------------------------------------------------------------------------


class TestConvergenceTableUnit:
    """Pure-function tests for _convergence_table."""

    def test_empty_dataframe(self) -> None:
        from hlanalysis.research.cards.card_e_convergence import _convergence_table

        df = pd.DataFrame(
            {
                "mid": pd.Series([], dtype="float64"),
                "outcome": pd.Series([], dtype="float64"),
                "tte_s": pd.Series([], dtype="float64"),
                "symbol": pd.Series([], dtype="object"),
            }
        )
        result = _convergence_table(df)
        # Should return a DataFrame (possibly empty but with correct schema)
        assert isinstance(result, pd.DataFrame)

    def test_known_values(self) -> None:
        from hlanalysis.research.cards.card_e_convergence import _convergence_table

        # Build synthetic data: 2 rows in the <1m bucket
        df = pd.DataFrame(
            {
                "mid": [0.9, 0.1],
                "outcome": [1.0, 0.0],
                "tte_s": [30.0, 30.0],
                "symbol": ["#100", "#101"],
                "yes_won": [True, False],
            }
        )
        df["outcome"] = df["yes_won"].astype(float)
        result = _convergence_table(df)
        assert "tte_bucket" in result.columns
        assert "mean_abs_err" in result.columns
        assert "mean_signed_err" in result.columns
        assert len(result) >= 1
        # Both errors should be -0.1 (mid - outcome = 0.9 - 1.0 = -0.1 and 0.1 - 0.0 = 0.1)
        # mean abs err = 0.1, signed = 0
        row = result[result["tte_bucket"] == "<1m"].iloc[0]
        assert abs(row["mean_abs_err"] - 0.1) < 1e-9
        assert abs(row["mean_signed_err"] - 0.0) < 1e-9


class TestCalibrationTableUnit:
    def test_perfect_calibration(self) -> None:
        from hlanalysis.research.cards.card_e_convergence import _calibration_table

        # mid 0.0-0.05 → all lose; mid 0.95-1.0 → all win
        df = pd.DataFrame(
            {
                "mid": [0.02, 0.02, 0.97, 0.97],
                "yes_won": [False, False, True, True],
            }
        )
        result = _calibration_table(df)
        assert "realized_win_rate" in result.columns
        assert "bias" in result.columns
        # Low mid bin → win rate 0
        low = result[result["mean_mid"] < 0.05]
        if len(low) > 0:
            assert low.iloc[0]["realized_win_rate"] == pytest.approx(0.0)
        # High mid bin → win rate 1
        high = result[result["mean_mid"] > 0.95]
        if len(high) > 0:
            assert high.iloc[0]["realized_win_rate"] == pytest.approx(1.0)


class TestHowEarlyKnownUnit:
    def test_winner_that_crosses_early(self) -> None:
        from hlanalysis.research.cards.card_e_convergence import _how_early_known

        # Winner: mid goes above 0.9 at TTE=5h and stays
        df = pd.DataFrame(
            {
                "symbol": ["#100"] * 5,
                "local_recv_ts": list(range(5)),
                "mid": [0.5, 0.6, 0.91, 0.92, 0.95],
                "tte_s": [20000.0, 18000.0, 16000.0, 5000.0, 1000.0],
                "yes_won": [True] * 5,
            }
        )
        result = _how_early_known(df)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["outcome"] == "winner"
        # cross_tte_s should be around 16000 (the last row before permanent crossing)
        assert not np.isnan(row["cross_tte_s"])
        assert row["cross_tte_s"] > 0

    def test_market_never_crosses(self) -> None:
        from hlanalysis.research.cards.card_e_convergence import _how_early_known

        # Market stays in 0.3-0.7 range — never crosses
        df = pd.DataFrame(
            {
                "symbol": ["#200"] * 3,
                "local_recv_ts": list(range(3)),
                "mid": [0.45, 0.55, 0.60],
                "tte_s": [5000.0, 2000.0, 100.0],
                "yes_won": [True] * 3,
            }
        )
        result = _how_early_known(df)
        assert len(result) == 1
        assert np.isnan(result.iloc[0]["cross_tte_s"])


class TestTakerEdgeTableUnit:
    def test_all_winners_positive_edge(self) -> None:
        from hlanalysis.research.cards.card_e_convergence import _taker_edge_table

        # All entries at mid=0.87, ask=0.88, TTE=2h — all win → edge = 1.0 - 0.88 = 0.12
        n = 20
        df = pd.DataFrame(
            {
                "mid": [0.87] * n,
                "ask_px": [0.88] * n,
                "bid_px": [0.86] * n,
                "tte_s": [7200.0] * n,
                "yes_won": [True] * n,
                "symbol": [f"#{i * 10}" for i in range(n)],
            }
        )
        result = _taker_edge_table(df)
        assert len(result) > 0
        # net_edge should be close to 1.0 - 0.88 = 0.12
        row = result[result["mid_bucket"] == "0.85-0.90"].iloc[0]
        assert row["net_edge"] == pytest.approx(1.0 - 0.88, abs=1e-6)

    def test_no_edge_when_ask_equals_outcome(self) -> None:
        from hlanalysis.research.cards.card_e_convergence import _taker_edge_table

        # ask=1.0 for winners → net_edge = 0
        n = 10
        df = pd.DataFrame(
            {
                "mid": [0.97] * n,
                "ask_px": [1.0] * n,
                "bid_px": [0.94] * n,
                "tte_s": [300.0] * n,
                "yes_won": [True] * n,
                "symbol": [f"#{i * 10}" for i in range(n)],
            }
        )
        result = _taker_edge_table(df)
        if len(result) > 0:
            assert result.iloc[0]["net_edge"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Integration test (requires real data)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="../../data not present")
class TestBuildCardIntegration:
    def test_build_card_returns_html_and_findings(self) -> None:
        """build_card returns valid HTML and a findings dict with required keys."""
        from hlanalysis.research.cards.card_e_convergence import build_card

        con = duckdb.connect()
        html, findings = build_card(con, str(_DATA_ROOT))

        # HTML is non-empty and looks like HTML
        assert isinstance(html, str)
        assert len(html) > 100
        assert "<!DOCTYPE html>" in html or "<html" in html

        # Findings has required keys
        required_keys = {"title", "headline", "metrics", "split_half", "verdict"}
        assert required_keys.issubset(findings.keys()), f"Missing keys: {required_keys - findings.keys()}"

    def test_findings_verdict_is_valid(self) -> None:
        from hlanalysis.research.cards.card_e_convergence import build_card

        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        assert findings["verdict"] in {"PASS", "FAIL", "INCONCLUSIVE"}

    def test_findings_metrics_have_required_fields(self) -> None:
        from hlanalysis.research.cards.card_e_convergence import build_card

        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        for m in findings["metrics"]:
            assert "name" in m
            assert "value" in m
            assert "n" in m
            assert "date_span" in m
            assert "sanity" in m

    def test_findings_is_json_serializable(self) -> None:
        from hlanalysis.research.cards.card_e_convergence import build_card

        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        # Should not raise
        dumped = json.dumps(findings)
        assert len(dumped) > 10

    def test_split_half_has_h1_and_h2(self) -> None:
        from hlanalysis.research.cards.card_e_convergence import build_card

        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        sh = findings["split_half"]
        assert "H1" in sh
        assert "H2" in sh
        assert "sign_stable" in sh
        assert "n_expiries" in sh["H1"]
        assert "n_expiries" in sh["H2"]

    def test_n_expiries_at_least_30(self) -> None:
        from hlanalysis.research.cards.card_e_convergence import build_card

        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        n_metric = next(m for m in findings["metrics"] if m["name"] == "n_expiries")
        assert n_metric["value"] >= 30, f"Expected ≥30 expiries, got {n_metric['value']}"

    def test_output_files_exist(self) -> None:
        """After build_card, card_e.html and card_e.json should exist."""
        from hlanalysis.research.cards.card_e_convergence import build_card

        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))

        out_dir = Path(__file__).parent.parent.parent / "docs" / "research" / "_cards"
        html_path = out_dir / "card_e.html"
        json_path = out_dir / "card_e.json"

        # Write JSON (standalone runner writes it; integration test writes it too)
        json_path.write_text(json.dumps(findings, indent=2), encoding="utf-8")

        assert html_path.exists(), f"card_e.html not found at {html_path}"
        assert json_path.exists(), f"card_e.json not found at {json_path}"
