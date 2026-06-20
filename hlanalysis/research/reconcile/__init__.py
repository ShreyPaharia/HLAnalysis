"""Sim-vs-live reconciliation package for HLAnalysis.

Public API
----------
Use :func:`run_reconcile` to run all 4 layers end-to-end, then
:func:`render_markdown` to produce a human-readable report.

For live data retrieval, use :mod:`hlanalysis.research.reconcile.pull_live`.
For book parity checks, use :mod:`hlanalysis.research.reconcile.book`.
"""

from __future__ import annotations

from hlanalysis.research.reconcile.book import (
    book_has_price,
    book_parity_pct,
    recorded_book_at,
)
from hlanalysis.research.reconcile.reconcile import (
    DecisionResult,
    FillsResult,
    PnLResult,
    PreconditionResult,
    ReconcileResult,
    ReferenceGap,
    attributable_gaps,
    check_preconditions,
    check_reference_coverage,
    reconcile_decisions,
    reconcile_fills,
    reconcile_pnl,
    run_reconcile,
    verdict,
)
from hlanalysis.research.reconcile.report import render_markdown

__all__ = [
    # reconcile
    "ReconcileResult",
    "PreconditionResult",
    "DecisionResult",
    "FillsResult",
    "PnLResult",
    "ReferenceGap",
    "attributable_gaps",
    "check_preconditions",
    "check_reference_coverage",
    "reconcile_decisions",
    "reconcile_fills",
    "reconcile_pnl",
    "verdict",
    "run_reconcile",
    # book
    "recorded_book_at",
    "book_has_price",
    "book_parity_pct",
    # report
    "render_markdown",
]
