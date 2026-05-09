"""hlanalysis/sim/plots/calibration.py

v2 calibration plot: predicted edge vs realised PnL per dollar.

Public API
----------
plot_calibration(fills_dir, out_path) -> Path | None
    Load every fills.parquet in *fills_dir*, build a scatter of
    predicted_edge vs realized_pnl_per_dollar, overlay the y=x reference
    line and a binned-mean calibration curve, write HTML to *out_path*,
    and return *out_path*. Returns None (writes no file) when there are no
    ENTER rows with a non-null entry_edge_chosen_side (e.g. v1 run).

_binned_calibration_curve(predicted_edges, realized_per_dollar, n_bins)
    Pure helper exposed for unit testing.  Returns a list of (x, y) tuples
    where x is the mean predicted_edge and y is the mean realized_pnl_per_dollar
    within each quantile bin.  Returns [] when len < 4.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Pure computation helper (testable without plotly)
# ---------------------------------------------------------------------------

def _binned_calibration_curve(
    predicted_edges: list[float],
    realized_per_dollar: list[float],
    n_bins: int,
) -> list[tuple[float, float]]:
    """Compute binned-mean calibration curve.

    Parameters
    ----------
    predicted_edges:
        x-axis values (entry_edge_chosen_side per ENTER fill).
    realized_per_dollar:
        y-axis values (realized_pnl_at_settle / (price * size) per ENTER fill).
    n_bins:
        Desired number of quantile bins.  The actual number used is
        ``min(n_bins, len // 2)``; if len < 4 the function returns [].

    Returns
    -------
    List of (mean_x, mean_y) tuples, one per bin, sorted by mean_x.
    """
    n = len(predicted_edges)
    if n < 4:
        return []

    actual_bins = min(n_bins, n // 2)

    # Sort both arrays together by predicted_edge
    paired = sorted(zip(predicted_edges, realized_per_dollar), key=lambda t: t[0])

    # Partition into quantile bins (roughly equal size)
    result: list[tuple[float, float]] = []
    bin_size_f = n / actual_bins
    for b in range(actual_bins):
        lo = int(round(b * bin_size_f))
        hi = int(round((b + 1) * bin_size_f))
        if lo >= hi:
            continue
        chunk = paired[lo:hi]
        mean_x = sum(x for x, _ in chunk) / len(chunk)
        mean_y = sum(y for _, y in chunk) / len(chunk)
        result.append((mean_x, mean_y))

    return result


# ---------------------------------------------------------------------------
# Plot builder
# ---------------------------------------------------------------------------

def plot_calibration(fills_dir: Path, out_path: Path) -> Optional[Path]:
    """Build and write the v2 calibration scatter plot.

    Loads all ``*.parquet`` files in *fills_dir*, extracts ENTER rows
    (cloid != "settle", entry_edge_chosen_side is not None, price*size != 0),
    and produces an interactive Plotly HTML.

    Returns *out_path* on success, or ``None`` if there are no eligible ENTER
    rows (v1 run, or empty fills).  Never raises on missing / empty parquet.
    """
    if not fills_dir.exists():
        return None

    predicted_edges: list[float] = []
    realized_per_dollar: list[float] = []

    for parquet_path in sorted(fills_dir.glob("*.parquet")):
        try:
            table = pq.read_table(parquet_path)
        except Exception:
            continue
        if table.num_rows == 0:
            continue

        raw = table.to_pydict()
        cloids = raw.get("cloid", [])
        edges = raw.get("entry_edge_chosen_side", [])
        prices = raw.get("price", [])
        sizes = raw.get("size", [])
        pnls = raw.get("realized_pnl_at_settle", [])

        for i, cloid in enumerate(cloids):
            if cloid == "settle":
                continue
            edge = edges[i] if i < len(edges) else None
            if edge is None:
                continue
            price = prices[i] if i < len(prices) else None
            size = sizes[i] if i < len(sizes) else None
            if price is None or size is None or price * size == 0:
                continue
            pnl = pnls[i] if i < len(pnls) else None
            if pnl is None:
                continue
            predicted_edges.append(float(edge))
            realized_per_dollar.append(float(pnl) / (float(price) * float(size)))

    if not predicted_edges:
        return None

    # Import plotly here to keep it out of other modules
    import plotly.graph_objects as go  # noqa: PLC0415

    # Scatter: all ENTER fills
    scatter = go.Scatter(
        x=predicted_edges,
        y=realized_per_dollar,
        mode="markers",
        name="fill",
        marker=dict(opacity=0.35, size=6, color="#5b7fad"),
        hovertemplate="edge=%{x:.4f}<br>pnl/$=%{y:.4f}<extra></extra>",
    )

    # y=x reference line
    all_vals = predicted_edges + realized_per_dollar
    ref_lo = min(all_vals)
    ref_hi = max(all_vals)
    ref_line = go.Scatter(
        x=[ref_lo, ref_hi],
        y=[ref_lo, ref_hi],
        mode="lines",
        name="y = x",
        line=dict(color="black", dash="dash", width=1),
        hoverinfo="skip",
    )

    traces: list[go.BaseTraceType] = [scatter, ref_line]

    # Binned calibration curve (10 quantile bins)
    bins = _binned_calibration_curve(predicted_edges, realized_per_dollar, n_bins=10)
    if bins:
        bin_x = [x for x, _ in bins]
        bin_y = [y for _, y in bins]
        calib_line = go.Scatter(
            x=bin_x,
            y=bin_y,
            mode="lines+markers",
            name="binned mean",
            line=dict(color="#e35d3a", width=2),
            marker=dict(size=8, color="#e35d3a"),
            hovertemplate="edge=%{x:.4f}<br>mean pnl/$=%{y:.4f}<extra></extra>",
        )
        traces.append(calib_line)

    fig = go.Figure(traces)
    fig.update_layout(
        title="v2 Calibration: predicted edge vs realised PnL per dollar",
        xaxis_title="predicted edge (entry_edge_chosen_side)",
        yaxis_title="realised PnL per dollar",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    return out_path
