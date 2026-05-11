"""Run-level result types: fills, diagnostics rows, summary stats, parquet writers.

Consolidates what used to live across `sim/{fills,metrics,diagnostics}.py`.
The dataclasses are stable wire types — `DiagnosticRow` / `FillRow` schemas
in particular are consumed by reporting and downstream plotting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from hlanalysis.strategy.types import Decision


# ---------------------------------------------------------------------------
# Fill (per simulated order match)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Fill:
    cloid: str
    symbol: str
    side: str
    price: float
    size: float
    fee: float
    partial: bool


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RunResult:
    fills: list[Fill] = field(default_factory=list)
    n_decisions: int = 0
    realized_pnl_usd: float | None = None


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RunSummary:
    n_markets: int
    n_trades: int
    total_pnl_usd: float
    sharpe: float
    hit_rate: float
    max_drawdown_usd: float


def _annualized_sharpe(per_obs_pnl: list[float], *, periods_per_year: float) -> float:
    n = len(per_obs_pnl)
    if n < 2:
        return 0.0
    mu = sum(per_obs_pnl) / n
    var = sum((r - mu) ** 2 for r in per_obs_pnl) / (n - 1)
    if var <= 0:
        return 0.0
    return (mu / math.sqrt(var)) * math.sqrt(periods_per_year)


def _hit_rate(per_obs_pnl: list[float]) -> float:
    if not per_obs_pnl:
        return 0.0
    return sum(1 for r in per_obs_pnl if r > 0) / len(per_obs_pnl)


def summarise_run(per_market_pnl: list[float], n_trades: int) -> RunSummary:
    sharpe = _annualized_sharpe(per_market_pnl, periods_per_year=365.0)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in per_market_pnl:
        cumulative += r
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)
    return RunSummary(
        n_markets=len(per_market_pnl),
        n_trades=n_trades,
        total_pnl_usd=sum(per_market_pnl),
        sharpe=sharpe,
        hit_rate=_hit_rate(per_market_pnl),
        max_drawdown_usd=max_dd,
    )


# ---------------------------------------------------------------------------
# Diagnostic + fill rows (parquet schemas)
# ---------------------------------------------------------------------------

DIAGNOSTICS_SCHEMA = pa.schema(
    [
        pa.field("ts_ns", pa.int64()),
        pa.field("question_id", pa.string()),
        pa.field("question_idx", pa.int64()),
        pa.field("action", pa.string()),
        pa.field("reason", pa.string()),
        pa.field("p_model", pa.float64()),
        pa.field("edge_yes", pa.float64()),
        pa.field("edge_no", pa.float64()),
        pa.field("sigma", pa.float64()),
        pa.field("tau_yr", pa.float64()),
        pa.field("ln_sk", pa.float64()),
        pa.field("ref_price", pa.float64()),
        pa.field("yes_bid", pa.float64()),
        pa.field("yes_ask", pa.float64()),
        pa.field("no_bid", pa.float64()),
        pa.field("no_ask", pa.float64()),
    ]
)


FILLS_SCHEMA = pa.schema(
    [
        pa.field("cloid", pa.string()),
        pa.field("ts_ns", pa.int64()),
        pa.field("side", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.float64()),
        pa.field("fee", pa.float64()),
        pa.field("question_id", pa.string()),
        pa.field("question_idx", pa.int64()),
        pa.field("symbol", pa.string()),
        pa.field("entry_p_model", pa.float64()),
        pa.field("entry_edge_chosen_side", pa.float64()),
        pa.field("entry_sigma", pa.float64()),
        pa.field("entry_tau_yr", pa.float64()),
        pa.field("realized_pnl_at_settle", pa.float64()),
        pa.field("resolved_outcome", pa.string()),
    ]
)


_EDGE_FLOAT_FIELDS = frozenset(
    {"p_model", "edge_yes", "edge_no", "sigma", "tau_yr", "ln_sk"}
)


@dataclass(slots=True)
class DiagnosticRow:
    ts_ns: int
    question_id: str
    question_idx: int
    action: str
    reason: str
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


@dataclass(slots=True)
class FillRow:
    cloid: str
    ts_ns: int
    side: str
    price: float
    size: float
    fee: float
    question_id: str
    question_idx: int
    symbol: str
    entry_p_model: Optional[float]
    entry_edge_chosen_side: Optional[float]
    entry_sigma: Optional[float]
    entry_tau_yr: Optional[float]
    realized_pnl_at_settle: float
    resolved_outcome: Optional[str] = None


def _parse_edge_fields(decision: Decision) -> dict[str, Optional[float]]:
    """Extract edge fields from a Decision's diagnostics (v2 model-edge format)."""
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


def build_diagnostic_row(
    *,
    ts_ns: int,
    question_id: str,
    question_idx: int,
    decision: Decision,
    ref_price: Optional[float],
    yes_bid: Optional[float],
    yes_ask: Optional[float],
    no_bid: Optional[float],
    no_ask: Optional[float],
) -> DiagnosticRow:
    edge = _parse_edge_fields(decision)
    reason = decision.diagnostics[0].message if decision.diagnostics else ""
    return DiagnosticRow(
        ts_ns=ts_ns,
        question_id=question_id,
        question_idx=question_idx,
        action=decision.action.value,
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


def write_diagnostics(rows: list[DiagnosticRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        table = pa.table(
            {f.name: pa.array([], type=f.type) for f in DIAGNOSTICS_SCHEMA},
            schema=DIAGNOSTICS_SCHEMA,
        )
        pq.write_table(table, path)
        return

    arrays = {
        "ts_ns": pa.array([r.ts_ns for r in rows], type=pa.int64()),
        "question_id": pa.array([r.question_id for r in rows], type=pa.string()),
        "question_idx": pa.array([r.question_idx for r in rows], type=pa.int64()),
        "action": pa.array([r.action for r in rows], type=pa.string()),
        "reason": pa.array([r.reason for r in rows], type=pa.string()),
        "p_model": pa.array([r.p_model for r in rows], type=pa.float64()),
        "edge_yes": pa.array([r.edge_yes for r in rows], type=pa.float64()),
        "edge_no": pa.array([r.edge_no for r in rows], type=pa.float64()),
        "sigma": pa.array([r.sigma for r in rows], type=pa.float64()),
        "tau_yr": pa.array([r.tau_yr for r in rows], type=pa.float64()),
        "ln_sk": pa.array([r.ln_sk for r in rows], type=pa.float64()),
        "ref_price": pa.array([r.ref_price for r in rows], type=pa.float64()),
        "yes_bid": pa.array([r.yes_bid for r in rows], type=pa.float64()),
        "yes_ask": pa.array([r.yes_ask for r in rows], type=pa.float64()),
        "no_bid": pa.array([r.no_bid for r in rows], type=pa.float64()),
        "no_ask": pa.array([r.no_ask for r in rows], type=pa.float64()),
    }
    pq.write_table(pa.table(arrays, schema=DIAGNOSTICS_SCHEMA), path)


def write_fills(rows: list[FillRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        table = pa.table(
            {f.name: pa.array([], type=f.type) for f in FILLS_SCHEMA},
            schema=FILLS_SCHEMA,
        )
        pq.write_table(table, path)
        return
    arrays = {
        "cloid": pa.array([r.cloid for r in rows], type=pa.string()),
        "ts_ns": pa.array([r.ts_ns for r in rows], type=pa.int64()),
        "side": pa.array([r.side for r in rows], type=pa.string()),
        "price": pa.array([r.price for r in rows], type=pa.float64()),
        "size": pa.array([r.size for r in rows], type=pa.float64()),
        "fee": pa.array([r.fee for r in rows], type=pa.float64()),
        "question_id": pa.array([r.question_id for r in rows], type=pa.string()),
        "question_idx": pa.array([r.question_idx for r in rows], type=pa.int64()),
        "symbol": pa.array([r.symbol for r in rows], type=pa.string()),
        "entry_p_model": pa.array(
            [r.entry_p_model for r in rows], type=pa.float64()
        ),
        "entry_edge_chosen_side": pa.array(
            [r.entry_edge_chosen_side for r in rows], type=pa.float64()
        ),
        "entry_sigma": pa.array([r.entry_sigma for r in rows], type=pa.float64()),
        "entry_tau_yr": pa.array([r.entry_tau_yr for r in rows], type=pa.float64()),
        "realized_pnl_at_settle": pa.array(
            [r.realized_pnl_at_settle for r in rows], type=pa.float64()
        ),
        "resolved_outcome": pa.array(
            [r.resolved_outcome for r in rows], type=pa.string()
        ),
    }
    pq.write_table(pa.table(arrays, schema=FILLS_SCHEMA), path)


__all__ = [
    "Fill",
    "RunResult",
    "RunSummary",
    "summarise_run",
    "DiagnosticRow",
    "FillRow",
    "DIAGNOSTICS_SCHEMA",
    "FILLS_SCHEMA",
    "build_diagnostic_row",
    "write_diagnostics",
    "write_fills",
]
