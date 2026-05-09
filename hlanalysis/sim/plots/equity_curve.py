"""hlanalysis/sim/plots/equity_curve.py

Cumulative PnL equity curve plot.

Public API
----------
plot_equity_curve(per_market_pnl, strategy_name, out_path) -> Path
    Accumulate per-market PnL into a cumulative series, write an interactive
    Plotly HTML to *out_path*, and return *out_path*.
"""
from __future__ import annotations

from pathlib import Path


def plot_equity_curve(
    per_market_pnl: list[float],
    strategy_name: str,
    out_path: Path,
) -> Path:
    """Build and write the cumulative-PnL equity curve.

    Parameters
    ----------
    per_market_pnl:
        Ordered list of per-market realized PnL values.
    strategy_name:
        Used as the figure title prefix.
    out_path:
        Destination HTML path.  Parent directories are created automatically.

    Returns
    -------
    *out_path* (always; never raises on empty input).
    """
    import plotly.graph_objects as go  # noqa: PLC0415

    cumulative: list[float] = []
    running = 0.0
    for r in per_market_pnl:
        running += r
        cumulative.append(running)

    fig = go.Figure(go.Scatter(y=cumulative, mode="lines", name="cumulative PnL"))
    fig.update_layout(
        title=f"{strategy_name} — equity curve",
        xaxis_title="market #",
        yaxis_title="PnL $",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    return out_path
