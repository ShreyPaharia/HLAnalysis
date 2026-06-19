"""Tests for the markdown report renderer."""

from __future__ import annotations

import pandas as pd

from hlanalysis.research.reconcile.reconcile import (
    DecisionDivergence,
    DecisionResult,
    FillEpisode,
    FillsResult,
    PnLResult,
    PreconditionResult,
    ReconcileResult,
)
from hlanalysis.research.reconcile.report import render_markdown

_T0 = 1_718_000_000_000_000_000
_T1 = _T0 + 23 * 3600 * 1_000_000_000


def _pass_result() -> ReconcileResult:
    """Build a minimal PASS ReconcileResult for rendering tests."""
    layer0 = PreconditionResult(
        config_hash_match="PASS",
        question_identity_match="PASS",
        window_match="PASS",
        overall="PASS",
    )
    layer1 = DecisionResult(
        match_rate=1.0,
        first_divergence=None,
        diff_table=pd.DataFrame(
            [
                {
                    "bucket_ns": _T0,
                    "live_sigma": 0.01,
                    "sim_sigma": 0.01,
                    "rel_diff_sigma": 0.0,
                    "live_action": "hold",
                    "sim_action": "hold",
                    "action_match": True,
                }
            ]
        ),
        classification="match",
        n_live_buckets=5,
        n_sim_buckets=5,
        n_aligned=5,
    )
    layer2 = FillsResult(
        live_episodes=[FillEpisode(side="BUY", start_ns=_T0, end_ns=_T0 + 10, total_size=10.0, vwap=0.80, n_fills=1)],
        sim_episodes=[FillEpisode(side="BUY", start_ns=_T0, end_ns=_T0 + 10, total_size=10.0, vwap=0.80, n_fills=1)],
        episode_table=pd.DataFrame(
            [
                {
                    "live_side": "BUY",
                    "live_start_ns": _T0,
                    "live_size": 10.0,
                    "sim_size": 10.0,
                    "size_diff": 0.0,
                    "live_vwap": 0.80,
                    "sim_vwap": 0.80,
                    "vwap_diff": 0.0,
                    "latency_ns": 0,
                    "matched": True,
                }
            ]
        ),
        book_parity_pct=1.0,
        gap_classification="match",
        n_live_fills=1,
        n_sim_fills=1,
    )
    layer3 = PnLResult(
        live_realized=0.50,
        sim_realized=0.50,
        pnl_diff=0.0,
        settlement_winner_match="PASS",
        waterfall={
            "entry_vwap_diff": 0.0,
            "exit_vwap_diff": 0.0,
            "size_diff": 0.0,
            "fee_diff": 0.0,
            "residual": 0.0,
        },
        pnl_match="PASS",
    )
    return ReconcileResult(
        question_idx=4010,
        expiry_ns=_T1,
        layer0=layer0,
        layer1=layer1,
        layer2=layer2,
        layer3=layer3,
        verdict="PASS",
        fail_reasons=[],
    )


def _fail_result() -> ReconcileResult:
    """Build a minimal FAIL ReconcileResult."""
    layer0 = PreconditionResult(
        config_hash_match="FAIL:live=abc sim=def",
        question_identity_match="PASS",
        window_match="PASS",
        overall="FAIL",
    )
    layer1 = DecisionResult(
        match_rate=0.60,
        first_divergence=DecisionDivergence(
            ts_ns=_T0,
            field="sigma",
            live_val=0.01,
            sim_val=0.02,
            rel_diff=1.0,
        ),
        diff_table=pd.DataFrame(),
        classification="sigma_diff",
        n_live_buckets=10,
        n_sim_buckets=10,
        n_aligned=10,
    )
    layer2 = FillsResult(
        live_episodes=[],
        sim_episodes=[],
        episode_table=pd.DataFrame(),
        book_parity_pct=None,
        gap_classification="match",
        n_live_fills=0,
        n_sim_fills=0,
    )
    layer3 = PnLResult(
        live_realized=10.0,
        sim_realized=0.0,
        pnl_diff=10.0,
        settlement_winner_match="SKIP:no_settlement",
        waterfall={
            "entry_vwap_diff": 0.0,
            "exit_vwap_diff": 0.0,
            "size_diff": 0.0,
            "fee_diff": 0.0,
            "residual": 10.0,
        },
        pnl_match="FAIL:+10.00",
    )
    return ReconcileResult(
        question_idx=4010,
        expiry_ns=_T1,
        layer0=layer0,
        layer1=layer1,
        layer2=layer2,
        layer3=layer3,
        verdict="FAIL",
        fail_reasons=[
            "config_hash: FAIL:live=abc sim=def",
            "decision_match_rate=60.00% < 95%",
            "pnl_diff: FAIL:+10.00",
        ],
    )


class TestRenderMarkdownPass:
    def test_render_markdown_pass_contains_pass(self) -> None:
        """PASS result renders markdown with 'PASS' in output."""
        result = _pass_result()
        md = render_markdown(result)
        assert "PASS" in md

    def test_render_markdown_pass_no_fail_reasons(self) -> None:
        """PASS result has no 'Failure reasons' section."""
        result = _pass_result()
        md = render_markdown(result)
        assert "Failure reasons" not in md

    def test_render_markdown_pass_header(self) -> None:
        """Header includes question index."""
        result = _pass_result()
        md = render_markdown(result)
        assert "question #4010" in md

    def test_render_markdown_pass_has_expiry(self) -> None:
        """Expiry timestamp formatted and present."""
        result = _pass_result()
        md = render_markdown(result)
        assert "expiry:" in md
        assert "UTC" in md


class TestRenderMarkdownFail:
    def test_render_markdown_fail_contains_fail(self) -> None:
        """FAIL result renders markdown with 'FAIL' in output."""
        result = _fail_result()
        md = render_markdown(result)
        assert "FAIL" in md

    def test_render_markdown_fail_includes_reasons(self) -> None:
        """FAIL result shows all fail reasons."""
        result = _fail_result()
        md = render_markdown(result)
        assert "config_hash" in md
        assert "decision_match_rate" in md
        assert "pnl_diff" in md

    def test_render_markdown_fail_verdict_bold(self) -> None:
        """Verdict line uses markdown bold."""
        result = _fail_result()
        md = render_markdown(result)
        assert "**Verdict: FAIL**" in md


class TestRenderAllSections:
    def test_render_contains_all_sections(self) -> None:
        """All 4 layer sections are present in output."""
        result = _pass_result()
        md = render_markdown(result)
        assert "## Layer 0: Preconditions" in md
        assert "## Layer 1: Decisions" in md
        assert "## Layer 2: Fills" in md
        assert "## Layer 3: PnL" in md

    def test_render_layer0_table(self) -> None:
        """Layer 0 renders a table with check names."""
        result = _pass_result()
        md = render_markdown(result)
        assert "Config hash" in md
        assert "Question identity" in md
        assert "Window overlap" in md

    def test_render_layer1_match_rate(self) -> None:
        """Layer 1 shows match rate as percentage."""
        result = _pass_result()
        md = render_markdown(result)
        assert "100.0%" in md

    def test_render_layer2_book_parity(self) -> None:
        """Layer 2 shows book parity percentage."""
        result = _pass_result()
        md = render_markdown(result)
        assert "Book parity" in md
        assert "100.0%" in md

    def test_render_layer3_waterfall(self) -> None:
        """Layer 3 renders waterfall attribution table."""
        result = _pass_result()
        md = render_markdown(result)
        assert "Waterfall attribution" in md
        assert "entry_vwap_diff" in md
        assert "residual" in md

    def test_render_layer3_pnl_values(self) -> None:
        """Layer 3 shows live and sim PnL values."""
        result = _pass_result()
        md = render_markdown(result)
        assert "Realized PnL" in md
        assert "0.5" in md  # both live and sim = 0.50

    def test_render_first_divergence_shown(self) -> None:
        """First divergence is included in layer 1 section when present."""
        result = _fail_result()
        md = render_markdown(result)
        assert "First divergence" in md
        assert "sigma" in md
