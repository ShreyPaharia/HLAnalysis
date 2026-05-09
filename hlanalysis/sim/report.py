from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

import pyarrow.parquet as pq
import plotly.graph_objects as go

from .metrics import RunSummary
from .plots.calibration import plot_calibration


# ---------------------------------------------------------------------------
# Protocol for market objects (duck-typed; accepts PMMarket or test fakes)
# ---------------------------------------------------------------------------

class _MarketLike(Protocol):
    condition_id: str
    resolved_outcome: str
    start_ts_ns: int
    end_ts_ns: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ts_ns_to_utc(ts_ns: int) -> str:
    """Convert nanosecond timestamp to 'YYYY-MM-DD HH:MM:SS UTC' string."""
    dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _ts_ns_to_date_utc(ts_ns: int) -> str:
    """Convert nanosecond timestamp to 'YYYY-MM-DD' string (for compact date range)."""
    dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _config_hash(config_summary: dict[str, Any]) -> str:
    """Return 12-char prefix of SHA-256 over canonically sorted JSON."""
    return hashlib.sha256(
        json.dumps(config_summary, sort_keys=True).encode()
    ).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Per-market data extraction
# ---------------------------------------------------------------------------

def _load_fills_for_market(fills_dir: Path, condition_id: str) -> dict[str, list]:
    """Load fills for a market from parquet; returns column-oriented dict (pyarrow to_pydict), or empty dict if absent."""
    path = fills_dir / f"{condition_id}.parquet"
    if not path.exists():
        return {}
    table = pq.read_table(path)
    return table.to_pydict()


def _compute_market_row(
    market: _MarketLike,
    realized_pnl_usd: float,
    fills_dir: Optional[Path],
) -> dict:
    """
    Build a dict of per-market stats for the markdown table.

    calibration_residual sign: realized_pnl_per_dollar - predicted_edge.
    Positive = strategy under-priced the edge; negative = over-priced.
    """
    if fills_dir is None:
        return {
            "condition_id": market.condition_id,
            "outcome": market.resolved_outcome,
            "n_trades": 0,
            "first_entry_ts": None,
            "last_exit_ts": None,
            "realized_pnl_usd": realized_pnl_usd,
            "calibration_residual": None,
        }

    raw = _load_fills_for_market(fills_dir, market.condition_id)
    if not raw or not raw.get("cloid"):
        return {
            "condition_id": market.condition_id,
            "outcome": market.resolved_outcome,
            "n_trades": 0,
            "first_entry_ts": None,
            "last_exit_ts": None,
            "realized_pnl_usd": realized_pnl_usd,
            "calibration_residual": None,
        }

    cloids = raw["cloid"]
    ts_ns_list = raw["ts_ns"]
    prices = raw["price"]
    sizes = raw["size"]
    entry_edges = raw["entry_edge_chosen_side"]
    realized_pnls = raw["realized_pnl_at_settle"]

    # Real trades exclude the "settle" synthetic
    trade_indices = [i for i, c in enumerate(cloids) if c != "settle"]
    n_trades = len(trade_indices)

    first_entry_ts: Optional[int] = None
    last_exit_ts: Optional[int] = None
    if trade_indices:
        trade_ts = [ts_ns_list[i] for i in trade_indices]
        first_entry_ts = min(trade_ts)
        last_exit_ts = max(trade_ts)

    # calibration_residual: per ENTER fill compute realized_pnl_per_dollar - entry_edge_chosen_side
    # then average across entries. null if no entry has entry_edge_chosen_side set (v1).
    # v1/v2 guarantee at most one ENTER per market (HOLD-when-position rule); EXIT rows have entry_edge_chosen_side=None and are skipped here.
    enter_residuals: list[float] = []
    for i in trade_indices:
        edge = entry_edges[i]
        if edge is None:
            continue
        price = prices[i]
        size = sizes[i]
        pnl_at_settle = realized_pnls[i]
        if price is None or size is None or price * size == 0:
            continue
        realized_per_dollar = pnl_at_settle / (price * size)
        enter_residuals.append(realized_per_dollar - edge)

    calibration_residual: Optional[float] = None
    if enter_residuals:
        calibration_residual = sum(enter_residuals) / len(enter_residuals)

    return {
        "condition_id": market.condition_id,
        "outcome": market.resolved_outcome,
        "n_trades": n_trades,
        "first_entry_ts": first_entry_ts,
        "last_exit_ts": last_exit_ts,
        "realized_pnl_usd": realized_pnl_usd,
        "calibration_residual": calibration_residual,
    }


# ---------------------------------------------------------------------------
# Section formatters
# ---------------------------------------------------------------------------

def _format_run_context(
    *,
    strategy_name: str,
    config_summary: dict[str, Any],
    markets: Optional[list],
    fee_taker: float,
    slippage_bps: float,
    half_spread: float,
) -> str:
    cfg_hash = _config_hash(config_summary)

    if markets:
        min_ts = min(m.start_ts_ns for m in markets)
        max_ts = max(m.end_ts_ns for m in markets)
        data_range = f"{_ts_ns_to_date_utc(min_ts)} UTC → {_ts_ns_to_date_utc(max_ts)} UTC"
        n_markets_line = f"- **markets:** {len(markets)}\n"
    else:
        data_range = "n/a"
        n_markets_line = ""

    return (
        "## Run context\n\n"
        f"- **strategy:** {strategy_name}\n"
        f"- **data range:** {data_range}\n"
        f"{n_markets_line}"
        f"- **fee_taker:** {fee_taker}\n"
        f"- **slippage_bps:** {slippage_bps}\n"
        f"- **half_spread:** {half_spread}\n"
        f"- **config SHA-256:** `{cfg_hash}`\n"
    )


def _format_per_market_table(rows: list[dict]) -> str:
    """Render per-market stats as a GitHub-flavoured markdown table."""
    header = "| condition_id | outcome | n_trades | first_entry_ts | last_exit_ts | realized_pnl_usd | calibration_residual |"
    sep    = "| :----------- | :------ | -------: | :------------- | :----------- | ---------------: | -------------------: |"

    lines = [header, sep]
    for r in rows:
        first_ts = _ts_ns_to_utc(r["first_entry_ts"]) if r["first_entry_ts"] is not None else ""
        last_ts  = _ts_ns_to_utc(r["last_exit_ts"]) if r["last_exit_ts"] is not None else ""
        calib    = f"{r['calibration_residual']:.4f}" if r["calibration_residual"] is not None else "—"
        lines.append(
            f"| {r['condition_id']} "
            f"| {r['outcome']} "
            f"| {r['n_trades']} "
            f"| {first_ts} "
            f"| {last_ts} "
            f"| {r['realized_pnl_usd']:,.2f} "
            f"| {calib} |"
        )

    return "## Per-market\n\n" + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_single_run_report(
    *,
    out_dir: Path,
    strategy_name: str,
    config_summary: dict[str, Any],
    per_market_pnl: list[float],
    summary: RunSummary,
    # New optional context fields (C5+C14)
    markets: list = (),  # tuple-as-default avoids mutable-default gotcha
    fills_dir: Optional[Path] = None,
    fee_taker: float = 0.0,
    slippage_bps: float = 0.0,
    half_spread: float = 0.0,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Equity curve HTML (unchanged behaviour)
    cumulative = []
    running = 0.0
    for r in per_market_pnl:
        running += r
        cumulative.append(running)
    fig = go.Figure(go.Scatter(y=cumulative, mode="lines", name="cumulative PnL"))
    fig.update_layout(title=f"{strategy_name} — equity curve", xaxis_title="market #", yaxis_title="PnL $")
    fig.write_html(str(out_dir / "equity_curve.html"))

    # Run context section
    run_ctx = _format_run_context(
        strategy_name=strategy_name,
        config_summary=config_summary,
        markets=markets,
        fee_taker=fee_taker,
        slippage_bps=slippage_bps,
        half_spread=half_spread,
    )

    # Per-market table
    market_rows = [
        _compute_market_row(m, pnl, fills_dir)
        for m, pnl in zip(markets, per_market_pnl)
    ]
    per_market_section = _format_per_market_table(market_rows)

    # Calibration plot (v2 only: skip when fills_dir is None or no v2 fills)
    calibration_section = ""
    if fills_dir is not None:
        calib_path = plot_calibration(fills_dir, out_dir / "calibration.html")
        if calib_path is not None:
            calibration_section = "## Calibration\n\nSee `calibration.html`.\n\n"

    # Assemble markdown
    md = out_dir / "report.md"
    md.write_text(
        f"# {strategy_name} run\n\n"
        f"**Config:** {config_summary}\n\n"
        + run_ctx + "\n"
        + "## Summary\n\n"
        f"- markets: {summary.n_markets}\n"
        f"- trades: {summary.n_trades}\n"
        f"- total PnL: ${summary.total_pnl_usd:,.2f}\n"
        f"- Sharpe (annualized 365): {summary.sharpe:.3f}\n"
        f"- hit rate: {summary.hit_rate:.2%}\n"
        f"- max drawdown: ${summary.max_drawdown_usd:,.2f}\n\n"
        "## Equity curve\n\nSee `equity_curve.html`.\n\n"
        + calibration_section
        + per_market_section
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
    rows_sorted = sorted(rows, key=lambda r: r["summary"]["sharpe"], reverse=True)[:top_k]
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
