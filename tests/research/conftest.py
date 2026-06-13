"""Pytest config for the research toolkit tests.

The card / strategy modules carry data-dependent smoke tests that re-run a
full-corpus analysis against the recorded HL data (``../../data``). Those are
**reproducibility smokes**, not fast correctness gates — a single one can take
minutes. They already ``skipif`` when the data is absent (so CI, which has no
data, skips them), but in a worktree where ``../../data`` resolves to the main
checkout they would otherwise run and make ``pytest -q tests/research`` hang for
many minutes.

So they are **opt-in**: skipped by default, even when the data is present, unless
``RESEARCH_DATA_TESTS=1`` is set. The fast unit / pure-function tests (resolver
label logic, metrics with synthetic inputs, report rendering) always run — those
are the real regression protection.

    # default: fast, data smokes skipped
    uv run pytest -q tests/research

    # opt in to the full data-dependent smokes (slow, needs ../../data)
    RESEARCH_DATA_TESTS=1 HLBT_HL_DATA_ROOT=../../data uv run pytest -q tests/research
"""

from __future__ import annotations

import os

import pytest

# Substrings that identify a data-dependent skipif reason across the card/strategy
# test modules (e.g. "../../data not present", "Recorded data not available …").
_DATA_REASON_HINTS = ("../../data", "recorded data", "data not present", "data not available")


def _is_data_dependent(item: pytest.Item) -> bool:
    for marker in item.iter_markers(name="skipif"):
        reason = str(marker.kwargs.get("reason", "")).lower()
        if any(hint in reason for hint in _DATA_REASON_HINTS):
            return True
    return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("RESEARCH_DATA_TESTS") == "1":
        return  # opted in — let the modules' own skipif handle data presence
    skip = pytest.mark.skip(reason="data-dependent research smoke; set RESEARCH_DATA_TESTS=1 to run")
    for item in items:
        if _is_data_dependent(item):
            item.add_marker(skip)
