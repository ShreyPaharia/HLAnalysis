"""Markdown reports for single runs and tuning sweeps.

Source-agnostic — keyed on ``question_id`` rather than PM-specific
``condition_id``. The PM-only calibration / vol-realized plots from the
previous ``hlanalysis/sim/report.py`` are left out of this v2 cut; they will
be re-added once the PM source lands (Task C) and the plotting layer (Task
A's plots/ directory) is fleshed out.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq

from .core.data_source import QuestionDescriptor
from .runner.result import RunSummary


def _ts_ns_to_date_utc(ts_ns: int) -> str:
    dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _ts_ns_to_utc(ts_ns: int) -> str:
    dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _config_hash(config_summary: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config_summary, sort_keys=True).encode()
    ).hexdigest()[:12]


def _load_fills_for_question(fills_dir: Path, question_id: str) -> dict[str, list]:
    path = fills_dir / f"{question_id}.parquet"
    if not path.exists():
        return {}
    table = pq.read_table(path)
    return table.to_pydict()


def _compute_market_row(
    q: QuestionDescriptor,
    realized_pnl_usd: float,
    fills_dir: Optional[Path],
    outcome: str,
) -> dict:
    if fills_dir is None:
        return {
            "question_id": q.question_id,
            "outcome": outcome,
            "n_trades": 0,
            "first_entry_ts": None,
            "last_exit_ts": None,
            "realized_pnl_usd": realized_pnl_usd,
        }
    raw = _load_fills_for_question(fills_dir, q.question_id)
    if not raw or not raw.get("cloid"):
        return {
            "question_id": q.question_id,
            "outcome": outcome,
            "n_trades": 0,
            "first_entry_ts": None,
            "last_exit_ts": None,
            "realized_pnl_usd": realized_pnl_usd,
        }
    cloids = raw["cloid"]
    ts_ns_list = raw["ts_ns"]
    trade_indices = [i for i, c in enumerate(cloids) if c != "settle"]
    first_entry_ts: Optional[int] = None
    last_exit_ts: Optional[int] = None
    if trade_indices:
        trade_ts = [ts_ns_list[i] for i in trade_indices]
        first_entry_ts = min(trade_ts)
        last_exit_ts = max(trade_ts)
    return {
        "question_id": q.question_id,
        "outcome": outcome,
        "n_trades": len(trade_indices),
        "first_entry_ts": first_entry_ts,
        "last_exit_ts": last_exit_ts,
        "realized_pnl_usd": realized_pnl_usd,
    }


def _format_per_question_table(rows: list[dict]) -> str:
    header = (
        "| question_id | outcome | n_trades | first_entry_ts | last_exit_ts | realized_pnl_usd |"
    )
    sep = (
        "| :---------- | :------ | -------: | :------------- | :----------- | ---------------: |"
    )
    lines = [header, sep]
    for r in rows:
        first_ts = (
            _ts_ns_to_utc(r["first_entry_ts"]) if r["first_entry_ts"] is not None else ""
        )
        last_ts = (
            _ts_ns_to_utc(r["last_exit_ts"]) if r["last_exit_ts"] is not None else ""
        )
        lines.append(
            f"| {r['question_id']} "
            f"| {r['outcome']} "
            f"| {r['n_trades']} "
            f"| {first_ts} "
            f"| {last_ts} "
            f"| {r['realized_pnl_usd']:,.2f} |"
        )
    return "## Per-question\n\n" + "\n".join(lines) + "\n"


def write_single_run_report(
    *,
    out_dir: Path,
    strategy_name: str,
    config_summary: dict[str, Any],
    summary: RunSummary,
    descriptors: list[QuestionDescriptor],
    per_question_pnl: list[float],
    outcomes: list[str],
    fills_dir: Optional[Path] = None,
    fee_taker: float = 0.0,
    slippage_bps: float = 0.0,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_hash = _config_hash(config_summary)
    if descriptors:
        min_ts = min(d.start_ts_ns for d in descriptors)
        max_ts = max(d.end_ts_ns for d in descriptors)
        data_range = f"{_ts_ns_to_date_utc(min_ts)} UTC → {_ts_ns_to_date_utc(max_ts)} UTC"
    else:
        data_range = "n/a"

    rows = [
        _compute_market_row(d, pnl, fills_dir, outcome)
        for d, pnl, outcome in zip(descriptors, per_question_pnl, outcomes)
    ]
    per_q_section = _format_per_question_table(rows)
    md = out_dir / "report.md"
    md.write_text(
        f"# {strategy_name} run\n\n"
        "## Run context\n\n"
        f"- **strategy:** {strategy_name}\n"
        f"- **data range:** {data_range}\n"
        f"- **questions:** {len(descriptors)}\n"
        f"- **fee_taker:** {fee_taker}\n"
        f"- **slippage_bps:** {slippage_bps}\n"
        f"- **config SHA-256:** `{cfg_hash}`\n\n"
        "## Summary\n\n"
        f"- questions: {summary.n_markets}\n"
        f"- trades: {summary.n_trades}\n"
        f"- total PnL: ${summary.total_pnl_usd:,.2f}\n"
        f"- Sharpe (annualized 365): {summary.sharpe:.3f}\n"
        f"- hit rate: {summary.hit_rate:.2%}\n"
        f"- max drawdown: ${summary.max_drawdown_usd:,.2f}\n\n"
        + per_q_section
    )
    return md


def write_tuning_report(
    *,
    out_dir: Path,
    strategy_name: str,
    rows: list[dict[str, Any]],
    top_k: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(
        rows, key=lambda r: r["summary"]["sharpe"], reverse=True
    )[:top_k]
    md = out_dir / "report.md"
    lines = [f"# {strategy_name} — tuning top-{top_k} by Sharpe\n"]
    for i, r in enumerate(rows_sorted, 1):
        s = r["summary"]
        lines.append(
            f"## #{i} Sharpe={s['sharpe']:.3f}\n\n"
            f"params: `{r['params']}`\n\n"
            f"- markets: {s['n_markets']}\n"
            f"- trades: {s['n_trades']}\n"
            f"- total PnL: ${s['total_pnl_usd']:,.2f}\n"
            f"- hit rate: {s['hit_rate']:.2%}\n"
            f"- max drawdown: ${s['max_drawdown_usd']:,.2f}\n"
        )
    md.write_text("\n".join(lines))
    return md


__all__ = ["write_single_run_report", "write_tuning_report"]
