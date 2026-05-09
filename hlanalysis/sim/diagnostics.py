# hlanalysis/sim/diagnostics.py
"""Per-decision diagnostic rows persisted to parquet.

One row is written for every call to strategy.evaluate(). Model-specific
fields (p_model, sigma, tau_yr, ln_sk, edge_yes, edge_no) are parsed out of
the v2 ``Diagnostic("info", "edge", fields=(...))`` tuple; they are left as
None for v1 and any strategy whose diagnostics don't carry those fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from hlanalysis.strategy.types import Decision


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DIAGNOSTICS_SCHEMA = pa.schema([
    pa.field("ts_ns",        pa.int64()),
    pa.field("condition_id", pa.string()),
    pa.field("question_idx", pa.int64()),
    pa.field("action",       pa.string()),
    pa.field("reason",       pa.string()),
    pa.field("p_model",      pa.float64()),
    pa.field("edge_yes",     pa.float64()),
    pa.field("edge_no",      pa.float64()),
    pa.field("sigma",        pa.float64()),
    pa.field("tau_yr",       pa.float64()),
    pa.field("ln_sk",        pa.float64()),
    pa.field("ref_price",    pa.float64()),
    pa.field("yes_bid",      pa.float64()),
    pa.field("yes_ask",      pa.float64()),
    pa.field("no_bid",       pa.float64()),
    pa.field("no_ask",       pa.float64()),
])

# Fields emitted by ModelEdgeStrategy's "edge" diagnostic
_EDGE_FLOAT_FIELDS = frozenset(
    {"p_model", "edge_yes", "edge_no", "sigma", "tau_yr", "ln_sk"}
)


# ---------------------------------------------------------------------------
# Row dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DiagnosticRow:
    ts_ns: int
    condition_id: str
    question_idx: int
    action: str                    # "hold" | "enter" | "exit"
    reason: str                    # first diagnostic message
    p_model: Optional[float]
    edge_yes: Optional[float]
    edge_no: Optional[float]
    sigma: Optional[float]
    tau_yr: Optional[float]
    ln_sk: Optional[float]
    ref_price: Optional[float]
    yes_bid: Optional[float]
    yes_ask: Optional[float]
    no_bid: Optional[float]
    no_ask: Optional[float]


# ---------------------------------------------------------------------------
# Parser: extract model fields from a Decision's diagnostics
# ---------------------------------------------------------------------------

def _parse_edge_fields(decision: Decision) -> dict[str, Optional[float]]:
    """Return a dict of {field_name: float|None} parsed from the first
    "edge" Diagnostic in *decision*. Returns all-None if not found.
    """
    result: dict[str, Optional[float]] = {k: None for k in _EDGE_FLOAT_FIELDS}
    for diag in decision.diagnostics:
        if diag.message == "edge":
            for key, val_str in diag.fields:
                if key in _EDGE_FLOAT_FIELDS:
                    try:
                        result[key] = float(val_str)
                    except (ValueError, TypeError):
                        result[key] = None
            break
    return result


# ---------------------------------------------------------------------------
# Builder: create a DiagnosticRow from a strategy.evaluate() result
# ---------------------------------------------------------------------------

def build_row(
    *,
    ts_ns: int,
    condition_id: str,
    question_idx: int,
    decision: Decision,
    ref_price: Optional[float],
    yes_bid: Optional[float],
    yes_ask: Optional[float],
    no_bid: Optional[float],
    no_ask: Optional[float],
) -> DiagnosticRow:
    """Build one ``DiagnosticRow`` from the output of a single ``evaluate()`` call."""
    action = decision.action.value  # "hold" | "enter" | "exit"
    reason = decision.diagnostics[0].message if decision.diagnostics else ""
    edge = _parse_edge_fields(decision)
    return DiagnosticRow(
        ts_ns=ts_ns,
        condition_id=condition_id,
        question_idx=question_idx,
        action=action,
        reason=reason,
        p_model=edge["p_model"],
        edge_yes=edge["edge_yes"],
        edge_no=edge["edge_no"],
        sigma=edge["sigma"],
        tau_yr=edge["tau_yr"],
        ln_sk=edge["ln_sk"],
        ref_price=ref_price,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_diagnostics(rows: list[DiagnosticRow], path: Path) -> None:
    """Write *rows* to *path* as a parquet file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Write an empty table with the correct schema so readers don't error.
        table = pa.table(
            {f.name: pa.array([], type=f.type) for f in DIAGNOSTICS_SCHEMA},
            schema=DIAGNOSTICS_SCHEMA,
        )
        pq.write_table(table, path)
        return

    arrays = {
        "ts_ns":        pa.array([r.ts_ns for r in rows],        type=pa.int64()),
        "condition_id": pa.array([r.condition_id for r in rows],  type=pa.string()),
        "question_idx": pa.array([r.question_idx for r in rows],  type=pa.int64()),
        "action":       pa.array([r.action for r in rows],        type=pa.string()),
        "reason":       pa.array([r.reason for r in rows],        type=pa.string()),
        "p_model":      pa.array([r.p_model for r in rows],       type=pa.float64()),
        "edge_yes":     pa.array([r.edge_yes for r in rows],      type=pa.float64()),
        "edge_no":      pa.array([r.edge_no for r in rows],       type=pa.float64()),
        "sigma":        pa.array([r.sigma for r in rows],         type=pa.float64()),
        "tau_yr":       pa.array([r.tau_yr for r in rows],        type=pa.float64()),
        "ln_sk":        pa.array([r.ln_sk for r in rows],         type=pa.float64()),
        "ref_price":    pa.array([r.ref_price for r in rows],     type=pa.float64()),
        "yes_bid":      pa.array([r.yes_bid for r in rows],       type=pa.float64()),
        "yes_ask":      pa.array([r.yes_ask for r in rows],       type=pa.float64()),
        "no_bid":       pa.array([r.no_bid for r in rows],        type=pa.float64()),
        "no_ask":       pa.array([r.no_ask for r in rows],        type=pa.float64()),
    }
    table = pa.table(arrays, schema=DIAGNOSTICS_SCHEMA)
    pq.write_table(table, path)


__all__ = [
    "DIAGNOSTICS_SCHEMA",
    "DiagnosticRow",
    "build_row",
    "write_diagnostics",
]
