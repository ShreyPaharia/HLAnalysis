"""Tests for Layer 1: decision trace alignment and comparison."""

from __future__ import annotations

import pandas as pd

from hlanalysis.research.reconcile.reconcile import reconcile_decisions

_T0 = 1_718_000_000_000_000_000  # base ns
_60S_NS = 60 * 1_000_000_000


def _make_trace(
    n: int = 10,
    start_ns: int = _T0,
    sigma: float = 0.01,
    action: str = "hold",
    step_ns: int = _60S_NS,
) -> pd.DataFrame:
    """Build a uniform decision trace."""
    rows = [
        {
            "ts_ns": start_ns + i * step_ns,
            "question_idx": 4010,
            "klass": "binary",
            "sigma": sigma,
            "p_model": 0.75,
            "edge": 0.05,
            "action": action,
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


class TestDecisionMatch:
    def test_decision_match_perfect(self) -> None:
        """Identical traces -> match_rate=1.0, no divergence, classification='match'."""
        live = _make_trace(n=10, sigma=0.01, action="hold")
        sim = _make_trace(n=10, sigma=0.01, action="hold")
        result = reconcile_decisions(live, sim, bucket_seconds=60)
        assert result.match_rate == 1.0
        assert result.first_divergence is None
        assert result.classification == "match"
        assert result.n_aligned == 10

    def test_decision_match_returns_diff_table(self) -> None:
        """Diff table has one row per aligned bucket."""
        live = _make_trace(n=5)
        sim = _make_trace(n=5)
        result = reconcile_decisions(live, sim, bucket_seconds=60)
        assert len(result.diff_table) == 5


class TestDecisionDivergence:
    def test_decision_first_divergence_sigma(self) -> None:
        """Inject a sigma difference at a known bucket -> first_divergence.field=='sigma'."""
        live = _make_trace(n=5, sigma=0.01, action="hold")
        sim_rows = _make_trace(n=5, sigma=0.01, action="hold")

        # Inject sigma difference at bucket 2 (third row)
        sim_rows_list = sim_rows.to_dict("records")
        sim_rows_list[2]["sigma"] = 0.02  # +100% relative diff, well above 5% threshold
        sim = pd.DataFrame(sim_rows_list)

        result = reconcile_decisions(live, sim, bucket_seconds=60, sigma_rel_tol=0.05)
        assert result.first_divergence is not None
        assert result.first_divergence.field == "sigma"
        assert result.first_divergence.rel_diff is not None
        assert result.first_divergence.rel_diff > 0.5

    def test_decision_gate_diff(self) -> None:
        """Sigma matches but action differs -> classification=='gate_diff'."""
        live = _make_trace(n=10, sigma=0.01, action="enter")
        sim = _make_trace(n=10, sigma=0.01, action="hold")  # same sigma, diff action
        result = reconcile_decisions(live, sim, bucket_seconds=60)
        # match_rate = 0 (all actions differ)
        assert result.match_rate == 0.0
        assert result.classification == "gate_diff"
        assert result.first_divergence is not None
        assert result.first_divergence.field == "action"

    def test_decision_action_divergence_captured(self) -> None:
        """First action divergence field is 'action' and ts is before later buckets."""
        # Use minute-aligned timestamps to avoid floor rounding surprises
        _bucket_ns = 60 * 1_000_000_000
        # Start at a clean minute boundary
        base_ts = (_T0 // _bucket_ns) * _bucket_ns
        live_rows = []
        sim_rows = []
        for i in range(6):
            ts = base_ts + i * _bucket_ns
            live_rows.append(
                {
                    "ts_ns": ts,
                    "question_idx": 4010,
                    "klass": "binary",
                    "sigma": 0.01,
                    "p_model": 0.75,
                    "edge": 0.05,
                    "action": "hold" if i < 3 else "enter",
                }
            )
            sim_rows.append(
                {
                    "ts_ns": ts,
                    "question_idx": 4010,
                    "klass": "binary",
                    "sigma": 0.01,
                    "p_model": 0.75,
                    "edge": 0.05,
                    "action": "hold",  # sim always hold
                }
            )
        live = pd.DataFrame(live_rows)
        sim = pd.DataFrame(sim_rows)
        result = reconcile_decisions(live, sim, bucket_seconds=60)
        assert result.first_divergence is not None
        assert result.first_divergence.field == "action"
        # The divergence must be at or after bucket index 3
        expected_ts = base_ts + 3 * _bucket_ns
        assert result.first_divergence.ts_ns == expected_ts


class TestDecisionCadence:
    def test_decision_cadence_live_has_buckets_sim_empty(self) -> None:
        """Live has buckets but sim has none -> classification=='cadence'."""
        live = _make_trace(n=10, action="hold")
        sim = pd.DataFrame()  # completely empty
        result = reconcile_decisions(live, sim, bucket_seconds=60)
        assert result.classification == "cadence"
        assert result.n_aligned == 0
        assert result.match_rate == 0.0

    def test_decision_cadence_large_gap(self) -> None:
        """Live has many buckets, sim has very few -> cadence gap > 10%."""
        live = _make_trace(n=20, action="hold")
        # Sim covers only buckets 0 and 1 out of 20 — large gap
        sim = _make_trace(n=2, action="hold")
        result = reconcile_decisions(live, sim, bucket_seconds=60)
        # n_aligned will be 2, live has 20 -> gap = 18/20 = 90% > 10%
        assert result.classification in ("cadence", "match")
        # With only 2 aligned out of 20 live buckets, classification should be cadence
        assert result.n_live_buckets == 20
        assert result.n_sim_buckets == 2

    def test_decision_cadence_sim_far_ahead(self) -> None:
        """Sim starts far in the future (no bucket overlap) -> cadence."""
        live = _make_trace(n=5, start_ns=_T0)
        sim = _make_trace(n=5, start_ns=_T0 + 1000 * _60S_NS)  # 1000 minutes later
        result = reconcile_decisions(live, sim, bucket_seconds=60)
        assert result.n_aligned == 0
        assert result.classification == "cadence"


class TestDecisionStats:
    def test_decision_n_buckets_counted(self) -> None:
        """n_live_buckets and n_sim_buckets reflect unique minute buckets."""
        live = _make_trace(n=8)
        sim = _make_trace(n=6)
        result = reconcile_decisions(live, sim, bucket_seconds=60)
        assert result.n_live_buckets == 8
        assert result.n_sim_buckets == 6
