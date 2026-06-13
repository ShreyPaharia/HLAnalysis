"""Tests for deterministic question_idx in PM backtest data sources (SHR-58).

question_idx must:
1. Return the same value across repeated calls with the same input.
2. Not depend on PYTHONHASHSEED (i.e. must NOT use Python's built-in hash()).
3. Match the live adapter's _question_idx_from_condition derivation exactly,
   which uses SHA-256 and takes the first 4 bytes big-endian & 0x7FFFFFFF.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# Reference: replicate the live adapter's scheme so the test is self-contained.
# ---------------------------------------------------------------------------


def _live_adapter_qidx(condition_id: str) -> int:
    """Mirrors hlanalysis/adapters/polymarket_normalize.py::_question_idx_from_condition."""
    digest = hashlib.sha256(condition_id.encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Import the functions under test.
# ---------------------------------------------------------------------------

from hlanalysis.backtest.data.polymarket import _question_idx as pm_question_idx  # noqa: E402
from hlanalysis.backtest.data.pm_nba import _question_idx as nba_question_idx  # noqa: E402


# ---------------------------------------------------------------------------
# Sample ids used in assertions.
# ---------------------------------------------------------------------------

SAMPLE_IDS = [
    "0x1234abcd",
    "0xdeadbeef00112233445566778899aabbccddeeff",
    "abc123-condition-id",
    "some-random-question-id",
    "0x0000000000000000000000000000000000000000000000000000000000000000",
]


# ---------------------------------------------------------------------------
# Tests: polymarket.py _question_idx
# ---------------------------------------------------------------------------


class TestPolymarketQuestionIdx:
    def test_matches_live_adapter_scheme(self):
        """backtest _question_idx must return the same value as the live adapter for each sample id."""
        for qid in SAMPLE_IDS:
            assert pm_question_idx(qid) == _live_adapter_qidx(qid), (
                f"polymarket.py _question_idx({qid!r}) diverges from live adapter"
            )

    def test_deterministic_across_calls(self):
        """Same id always returns same integer (basic idempotency)."""
        for qid in SAMPLE_IDS:
            assert pm_question_idx(qid) == pm_question_idx(qid)

    def test_result_is_non_negative_31bit(self):
        """Result must be a non-negative 31-bit integer (fits SQLite int column)."""
        for qid in SAMPLE_IDS:
            v = pm_question_idx(qid)
            assert 0 <= v <= 0x7FFFFFFF, f"Out of 31-bit range: {v}"

    def test_no_pythonhashseed_dependence(self):
        """Run in a subprocess with a different PYTHONHASHSEED and confirm identical result."""
        qid = SAMPLE_IDS[1]
        expected = pm_question_idx(qid)

        script = (
            "import sys; "
            "from hlanalysis.backtest.data.polymarket import _question_idx; "
            f"v = _question_idx({qid!r}); "
            "sys.exit(0 if v == "
            f"{expected}"
            " else 1)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={"PYTHONHASHSEED": "12345", "PATH": "/usr/bin:/bin"},
            capture_output=True,
        )
        assert result.returncode == 0, (
            f"polymarket _question_idx differs across PYTHONHASHSEED values.\nstderr: {result.stderr.decode()}"
        )


# ---------------------------------------------------------------------------
# Tests: pm_nba.py _question_idx
# ---------------------------------------------------------------------------


class TestNBAQuestionIdx:
    def test_matches_live_adapter_scheme(self):
        """pm_nba.py _question_idx must return the same value as the live adapter for each sample id."""
        for qid in SAMPLE_IDS:
            assert nba_question_idx(qid) == _live_adapter_qidx(qid), (
                f"pm_nba.py _question_idx({qid!r}) diverges from live adapter"
            )

    def test_deterministic_across_calls(self):
        """Same id always returns same integer (basic idempotency)."""
        for qid in SAMPLE_IDS:
            assert nba_question_idx(qid) == nba_question_idx(qid)

    def test_result_is_non_negative_31bit(self):
        """Result must be a non-negative 31-bit integer."""
        for qid in SAMPLE_IDS:
            v = nba_question_idx(qid)
            assert 0 <= v <= 0x7FFFFFFF, f"Out of 31-bit range: {v}"

    def test_no_pythonhashseed_dependence(self):
        """Run in a subprocess with a different PYTHONHASHSEED and confirm identical result."""
        qid = SAMPLE_IDS[1]
        expected = nba_question_idx(qid)

        script = (
            "import sys; "
            "from hlanalysis.backtest.data.pm_nba import _question_idx; "
            f"v = _question_idx({qid!r}); "
            "sys.exit(0 if v == "
            f"{expected}"
            " else 1)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={"PYTHONHASHSEED": "12345", "PATH": "/usr/bin:/bin"},
            capture_output=True,
        )
        assert result.returncode == 0, (
            f"pm_nba _question_idx differs across PYTHONHASHSEED values.\nstderr: {result.stderr.decode()}"
        )


# ---------------------------------------------------------------------------
# Cross-module consistency
# ---------------------------------------------------------------------------


class TestCrossModuleConsistency:
    def test_pm_and_nba_agree(self):
        """Both modules must produce identical values for the same id."""
        for qid in SAMPLE_IDS:
            assert pm_question_idx(qid) == nba_question_idx(qid), (
                f"polymarket and pm_nba _question_idx disagree for {qid!r}"
            )
