"""SHR-150: cross-question fidelity dashboard.

Reconciliation is per-question; fidelity is an aggregate property. ``fidelity.py``
aggregates many reconciled questions into decision-agreement rate, slippage
(live − sim VWAP), per-pair latency, hit-rate delta and PnL-diff distributions,
broken out per class (binary/bucket) and track (HL/PM), and renders an HTML card.
"""

from __future__ import annotations

import pandas as pd

from hlanalysis.research.reconcile.fidelity import (
    QuestionLabel,
    aggregate_fidelity,
    render_fidelity_html,
)
from hlanalysis.research.reconcile.reconcile import (
    DecisionResult,
    FillsResult,
    PnLResult,
    PreconditionResult,
    ReconcileResult,
)

_1S_NS = 1_000_000_000


def _episode(side: str, live_vwap: float, sim_vwap: float, latency_s: float) -> dict:
    return {
        "match_status": "matched",
        "live_side": side,
        "live_vwap": live_vwap,
        "sim_vwap": sim_vwap,
        "latency_ns": latency_s * _1S_NS,
    }


def _result(
    qidx: int,
    match_rate: float,
    pnl_diff: float,
    live_realized: float,
    sim_realized: float,
    episodes: list[dict],
    verdict: str = "PASS",
) -> ReconcileResult:
    l0 = PreconditionResult("PASS", "PASS", "PASS", "PASS")
    l1 = DecisionResult(match_rate, None, pd.DataFrame(), "match", 100, 100, 100, 0, 0)
    l2 = FillsResult([], [], pd.DataFrame(episodes), 1.0, "match", 0, 0)
    l3 = PnLResult(live_realized, sim_realized, pnl_diff, "PASS", {}, "PASS")
    return ReconcileResult(qidx, 0, l0, l1, l2, l3, verdict, [])


def _fixture() -> tuple[list[ReconcileResult], list[QuestionLabel]]:
    q1 = _result(
        1,
        match_rate=1.0,
        pnl_diff=0.5,
        live_realized=10.0,
        sim_realized=9.5,
        episodes=[
            _episode("BUY", 0.90, 0.89, 2.0),  # live paid more → adverse
            _episode("SELL", 0.95, 0.95, 1.0),  # flat
        ],
    )
    q2 = _result(
        2,
        match_rate=0.8,
        pnl_diff=-2.0,
        live_realized=-1.0,
        sim_realized=1.0,
        episodes=[_episode("BUY", 0.80, 0.82, 3.0)],  # live paid less → favorable
        verdict="FAIL",
    )
    q3 = _result(
        3,
        match_rate=0.9,
        pnl_diff=1.0,
        live_realized=5.0,
        sim_realized=4.0,
        episodes=[_episode("SELL", 0.50, 0.48, 0.0)],  # live sold higher → favorable
    )
    labels = [
        QuestionLabel(klass="binary", track="HL"),
        QuestionLabel(klass="binary", track="HL"),
        QuestionLabel(klass="bucket", track="HL"),
    ]
    return [q1, q2, q3], labels


class TestAggregate:
    def test_overall_counts_and_rates(self) -> None:
        results, labels = _fixture()
        rep = aggregate_fidelity(results, labels)
        g = rep.overall
        assert g.n_questions == 3
        assert abs(g.decision_agreement_rate - 0.9) < 1e-9
        assert abs(g.verdict_pass_rate - (2 / 3)) < 1e-9

    def test_slippage_distribution(self) -> None:
        results, labels = _fixture()
        g = aggregate_fidelity(results, labels).overall
        # slippage = live - sim vwap over 4 matched episodes: 0.01, 0.0, -0.02, 0.02
        assert g.slippage.n == 4
        assert abs(g.slippage.mean - 0.0025) < 1e-9

    def test_adverse_slippage_fraction(self) -> None:
        results, labels = _fixture()
        g = aggregate_fidelity(results, labels).overall
        # adverse: q1 BUY (live>sim) only → 1 of 4
        assert abs(g.adverse_slippage_frac - 0.25) < 1e-9

    def test_latency_distribution(self) -> None:
        results, labels = _fixture()
        g = aggregate_fidelity(results, labels).overall
        # latencies (s): 2, 1, 3, 0
        assert g.latency_seconds.n == 4
        assert abs(g.latency_seconds.mean - 1.5) < 1e-9

    def test_pnl_diff_distribution(self) -> None:
        results, labels = _fixture()
        g = aggregate_fidelity(results, labels).overall
        assert g.pnl_diff.n == 3
        assert abs(g.pnl_diff.mean - (-0.5 / 3)) < 1e-9

    def test_hit_rate_delta(self) -> None:
        results, labels = _fixture()
        g = aggregate_fidelity(results, labels).overall
        # live wins: q1,q3 → 2/3 ; sim wins: all → 3/3
        assert abs(g.hit_rate_live - 2 / 3) < 1e-9
        assert abs(g.hit_rate_sim - 1.0) < 1e-9
        assert abs(g.hit_rate_delta - (2 / 3 - 1.0)) < 1e-9

    def test_per_class_breakout(self) -> None:
        results, labels = _fixture()
        rep = aggregate_fidelity(results, labels)
        assert set(rep.by_class) == {"binary", "bucket"}
        assert rep.by_class["binary"].n_questions == 2
        assert rep.by_class["bucket"].n_questions == 1

    def test_per_track_breakout(self) -> None:
        results, labels = _fixture()
        rep = aggregate_fidelity(results, labels)
        assert set(rep.by_track) == {"HL"}
        assert rep.by_track["HL"].n_questions == 3

    def test_no_labels_defaults_to_unknown(self) -> None:
        results, _ = _fixture()
        rep = aggregate_fidelity(results)
        assert rep.overall.n_questions == 3
        assert "unknown" in rep.by_class


class TestRenderHtml:
    def test_html_written(self, tmp_path) -> None:
        results, labels = _fixture()
        rep = aggregate_fidelity(results, labels)
        out = tmp_path / "fidelity.html"
        render_fidelity_html(rep, out)
        assert out.exists()
        html = out.read_text()
        assert "<html" in html.lower()
        assert "Slippage" in html
        assert "binary" in html
