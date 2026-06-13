"""`tune` must tell its spawn workers how many of them share the box so each
self-limits its in-proc bundle memo to total_budget / n_workers (SHR-71).

Without this, every worker process keeps its own module-global memo bounded only
per-process, so the AGGREGATE RAM is N × per-process → ~TB OOM at --workers 12.
"""

from __future__ import annotations

import pytest

from hlanalysis.backtest.tuning import _set_inproc_memo_worker_env


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HLBT_INPROC_BUNDLE_MEMO_WORKERS", raising=False)


def test_sets_worker_count_env(monkeypatch):
    import os

    _set_inproc_memo_worker_env(12)
    assert os.environ["HLBT_INPROC_BUNDLE_MEMO_WORKERS"] == "12"


def test_clamps_to_at_least_one(monkeypatch):
    import os

    _set_inproc_memo_worker_env(0)
    assert os.environ["HLBT_INPROC_BUNDLE_MEMO_WORKERS"] == "1"
