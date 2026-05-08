from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from .metrics import RunSummary


def write_single_run_report(
    *,
    out_dir: Path,
    strategy_name: str,
    config_summary: dict[str, Any],
    per_market_pnl: list[float],
    summary: RunSummary,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    cumulative = []
    running = 0.0
    for r in per_market_pnl:
        running += r
        cumulative.append(running)
    fig = go.Figure(go.Scatter(y=cumulative, mode="lines", name="cumulative PnL"))
    fig.update_layout(title=f"{strategy_name} — equity curve", xaxis_title="market #", yaxis_title="PnL $")
    fig.write_html(str(out_dir / "equity_curve.html"))

    md = out_dir / "report.md"
    md.write_text(
        f"# {strategy_name} run\n\n"
        f"**Config:** {config_summary}\n\n"
        "## Summary\n\n"
        f"- markets: {summary.n_markets}\n"
        f"- trades: {summary.n_trades}\n"
        f"- total PnL: ${summary.total_pnl_usd:,.2f}\n"
        f"- Sharpe (annualized 365): {summary.sharpe:.3f}\n"
        f"- hit rate: {summary.hit_rate:.2%}\n"
        f"- max drawdown: ${summary.max_drawdown_usd:,.2f}\n\n"
        "## Equity curve\n\nSee `equity_curve.html`.\n"
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
