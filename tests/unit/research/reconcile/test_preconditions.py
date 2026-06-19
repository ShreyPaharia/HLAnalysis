"""Tests for Layer 0: precondition checks."""

from __future__ import annotations

import pandas as pd

from hlanalysis.research.reconcile.reconcile import check_preconditions

# Shared reference timestamps (ns)
_T0 = 1_718_000_000_000_000_000  # some base ns
_T1 = _T0 + 3600 * 1_000_000_000  # +1h
_T2 = _T0 + 23 * 3600 * 1_000_000_000  # +23h
_T3 = _T0 + 48 * 3600 * 1_000_000_000  # +48h (no overlap)


def _make_trace(
    question_idx: int = 4010,
    klass: str = "binary",
    start_ns: int = _T0,
    end_ns: int = _T2,
    config_hash: str = "abc123",
    n: int = 5,
) -> pd.DataFrame:
    """Build a minimal decision trace DataFrame."""
    step = (end_ns - start_ns) // max(n - 1, 1)
    rows = [
        {
            "ts_ns": start_ns + i * step,
            "question_idx": question_idx,
            "klass": klass,
            "config_hash": config_hash,
            "action": "hold",
            "sigma": 0.01,
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


class TestPreconditionPass:
    def test_precondition_pass(self) -> None:
        """Matching config_hash, same question_idx, overlapping windows -> PASS."""
        live = _make_trace(config_hash="abc123", start_ns=_T0, end_ns=_T2)
        sim = _make_trace(config_hash="abc123", start_ns=_T0, end_ns=_T2)
        result = check_preconditions(
            live_trace=live,
            sim_trace=sim,
            live_config_hash="abc123",
            sim_config_hash="abc123",
        )
        assert result.config_hash_match == "PASS"
        assert result.question_identity_match == "PASS"
        assert result.window_match == "PASS"
        assert result.overall == "PASS"


class TestPreconditionConfigHash:
    def test_precondition_fail_config_hash(self) -> None:
        """Different config hashes -> config_hash_match FAIL, overall FAIL."""
        live = _make_trace(config_hash="abc123")
        sim = _make_trace(config_hash="def456")
        result = check_preconditions(
            live_trace=live,
            sim_trace=sim,
            live_config_hash="abc123",
            sim_config_hash="def456",
        )
        assert result.config_hash_match.startswith("FAIL")
        assert "abc123" in result.config_hash_match
        assert "def456" in result.config_hash_match
        assert result.overall == "FAIL"

    def test_precondition_skip_no_hash(self) -> None:
        """Both config hashes None -> SKIP (not FAIL)."""
        live = _make_trace()
        sim = _make_trace()
        # Remove config_hash from both traces
        live = live.drop(columns=["config_hash"])
        sim = sim.drop(columns=["config_hash"])
        result = check_preconditions(
            live_trace=live,
            sim_trace=sim,
            live_config_hash=None,
            sim_config_hash=None,
        )
        assert result.config_hash_match == "SKIP:no_live_hash"
        # SKIP should not cause overall FAIL (unless other checks fail)
        assert result.overall == "PASS"

    def test_precondition_skip_live_hash_none(self) -> None:
        """Live hash None, sim hash present -> SKIP (not FAIL)."""
        live = _make_trace()
        sim = _make_trace(config_hash="def456")
        result = check_preconditions(
            live_trace=live,
            sim_trace=sim,
            live_config_hash=None,
            sim_config_hash="def456",
        )
        assert result.config_hash_match == "SKIP:no_live_hash"


class TestPreconditionWindowMatch:
    def test_precondition_fail_no_overlap(self) -> None:
        """Windows don't overlap -> window_match FAIL."""
        live = _make_trace(start_ns=_T0, end_ns=_T1)  # window [T0, T1]
        sim = _make_trace(start_ns=_T3, end_ns=_T3 + _T1 - _T0)  # window [T3, ...] far future
        result = check_preconditions(
            live_trace=live,
            sim_trace=sim,
            live_config_hash="abc123",
            sim_config_hash="abc123",
        )
        assert result.window_match.startswith("FAIL")
        assert result.overall == "FAIL"

    def test_precondition_pass_partial_overlap(self) -> None:
        """Windows with > 50% overlap -> PASS."""
        # Live: T0 to T2 (23h window)
        # Sim: T1 to T2 + some (starts 1h in, so overlaps 22h out of 23h)
        live = _make_trace(start_ns=_T0, end_ns=_T2)
        sim = _make_trace(start_ns=_T1, end_ns=_T2)
        result = check_preconditions(
            live_trace=live,
            sim_trace=sim,
            live_config_hash="abc123",
            sim_config_hash="abc123",
        )
        assert result.window_match == "PASS"


class TestPreconditionIdentity:
    def test_precondition_fail_different_question_idx(self) -> None:
        """Different question_idx -> question_identity_match FAIL."""
        live = _make_trace(question_idx=4010)
        sim = _make_trace(question_idx=9999)
        result = check_preconditions(
            live_trace=live,
            sim_trace=sim,
            live_config_hash="abc123",
            sim_config_hash="abc123",
        )
        assert result.question_identity_match.startswith("FAIL")
        assert result.overall == "FAIL"

    def test_precondition_fail_different_klass(self) -> None:
        """Different klass -> question_identity_match FAIL."""
        live = _make_trace(klass="binary")
        sim = _make_trace(klass="bucket")
        result = check_preconditions(
            live_trace=live,
            sim_trace=sim,
            live_config_hash="abc123",
            sim_config_hash="abc123",
        )
        assert result.question_identity_match.startswith("FAIL")
