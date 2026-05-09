"""hlanalysis/sim/plots/per_market_trace.py

Per-market price trace plot.

Public API
----------
plot_market_trace(condition_id, run_dir, out_path) -> Path | None
    Load per-tick diagnostics and fill data for a single market, produce an
    interactive Plotly HTML showing YES/NO mid-price time series with entry/exit
    fill markers and a settlement annotation, and return *out_path*.

    Returns None (writes no file) when the diagnostics parquet is missing.

Colour conventions
------------------
- YES mid price line:  #2196f3 (blue)
- NO mid price line:   #ff9800 (orange)
- BUY fill marker:     green  (#4caf50)
- SELL fill marker:    red    (#f44336)
- Settlement line/annotation:  #9c27b0 (purple) — labelled "settled YES"
  or "settled NO" based on the ``resolved_outcome`` field of the settle row
  in fills.parquet ("yes" → "settled YES", "no" → "settled NO").
  Falls back to plain "settled" when the field is absent or "unknown".

Note: stop-loss horizontal line is intentionally deferred. The stop level is
not stored in fills.parquet, and reading the strategy config from disk would
add excessive plumbing for marginal trace value.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq

from hlanalysis.sim.plots._common import save_fig


def _load_parquet_as_dict(path: Path) -> Optional[dict[str, list]]:
    """Load a parquet file into a column-oriented dict. Returns None on error."""
    if not path.exists():
        return None
    try:
        table = pq.read_table(path)
    except Exception:
        return None
    if table.num_rows == 0:
        return {}
    return table.to_pydict()


def _ns_to_dt_str(ts_ns: int) -> str:
    """Convert a nanosecond UNIX timestamp to an ISO-8601 string (UTC, microsecond precision).

    Returns a string that Plotly can parse as a datetime for the x-axis.
    """
    import datetime
    ts_us = ts_ns // 1_000
    dt = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(microseconds=ts_us)
    return dt.isoformat()


def plot_market_trace(
    condition_id: str,
    run_dir: Path,
    out_path: Path,
) -> Optional[Path]:
    """Build and write the per-market price trace plot.

    Loads ``<run_dir>/diagnostics/<condition_id>.parquet`` for YES/NO mid-price
    time series and ``<run_dir>/fills/<condition_id>.parquet`` for fill markers.

    Parameters
    ----------
    condition_id:
        The market identifier; used as file names and as the plot title.
    run_dir:
        Root directory of the simulation run.  Expected subdirectories:
        ``diagnostics/`` and ``fills/``.
    out_path:
        Destination HTML path.  Parent directories are created automatically.

    Returns
    -------
    *out_path* on success, or ``None`` if the diagnostics parquet is missing
    (the caller should treat this as "market not found in run").

    Raises
    ------
    Never raises; all I/O errors are silently handled by returning None or
    skipping bad rows.
    """
    diag_path = run_dir / "diagnostics" / f"{condition_id}.parquet"
    diag = _load_parquet_as_dict(diag_path)
    if diag is None:
        # diagnostics parquet missing — cannot produce trace
        return None

    # ------------------------------------------------------------------
    # Build YES/NO mid-price time series from diagnostics
    # ------------------------------------------------------------------
    ts_ns_col = diag.get("ts_ns", [])
    yes_bid_col = diag.get("yes_bid", [])
    yes_ask_col = diag.get("yes_ask", [])
    no_bid_col = diag.get("no_bid", [])
    no_ask_col = diag.get("no_ask", [])

    yes_times: list[str] = []
    yes_mids: list[float] = []
    no_times: list[str] = []
    no_mids: list[float] = []

    for i, ts in enumerate(ts_ns_col):
        yb = yes_bid_col[i] if i < len(yes_bid_col) else None
        ya = yes_ask_col[i] if i < len(yes_ask_col) else None
        nb = no_bid_col[i] if i < len(no_bid_col) else None
        na = no_ask_col[i] if i < len(no_ask_col) else None

        dt_str = _ns_to_dt_str(int(ts))

        if yb is not None and ya is not None:
            yes_times.append(dt_str)
            yes_mids.append((float(yb) + float(ya)) / 2.0)

        if nb is not None and na is not None:
            no_times.append(dt_str)
            no_mids.append((float(nb) + float(na)) / 2.0)

    # ------------------------------------------------------------------
    # Load fills (optional — market may have zero fills)
    # ------------------------------------------------------------------
    fills_path = run_dir / "fills" / f"{condition_id}.parquet"
    fills = _load_parquet_as_dict(fills_path) or {}

    fill_ts_col = fills.get("ts_ns", [])
    fill_cloid_col = fills.get("cloid", [])
    fill_side_col = fills.get("side", [])
    fill_price_col = fills.get("price", [])
    fill_resolved_outcome_col = fills.get("resolved_outcome", [])

    # Separate entry/exit fills from the settlement synthetic
    buy_times: list[str] = []
    buy_prices: list[float] = []
    sell_times: list[str] = []
    sell_prices: list[float] = []
    settle_time: Optional[str] = None
    settle_outcome: Optional[str] = None

    for i, cloid in enumerate(fill_cloid_col):
        ts = fill_ts_col[i] if i < len(fill_ts_col) else None
        side = fill_side_col[i] if i < len(fill_side_col) else None
        price = fill_price_col[i] if i < len(fill_price_col) else None

        if ts is None:
            continue

        dt_str = _ns_to_dt_str(int(ts))

        if cloid == "settle":
            settle_time = dt_str
            # Read the explicit resolved_outcome stored on the settle row.
            # Mapping: "yes" → "YES", "no" → "NO", anything else → no side label.
            # Falls back gracefully when the column is absent (old parquet schema)
            # or the value is None/"unknown".
            ro = fill_resolved_outcome_col[i] if i < len(fill_resolved_outcome_col) else None
            if ro == "yes":
                settle_outcome = "YES"
            elif ro == "no":
                settle_outcome = "NO"
            else:
                # "unknown", None, or missing column — don't infer from price
                settle_outcome = None
        else:
            # Regular fill: colour by side (buy=green, sell=red)
            if side == "buy":
                buy_times.append(dt_str)
                buy_prices.append(float(price) if price is not None else 0.5)
            else:
                sell_times.append(dt_str)
                sell_prices.append(float(price) if price is not None else 0.5)

    # ------------------------------------------------------------------
    # Assemble Plotly figure
    # ------------------------------------------------------------------
    import plotly.graph_objects as go  # noqa: PLC0415

    traces: list = []

    # YES mid line
    if yes_times:
        traces.append(go.Scatter(
            x=yes_times,
            y=yes_mids,
            mode="lines",
            name="YES mid",
            line=dict(color="#2196f3", width=2),
            hovertemplate="YES mid=%{y:.4f}<br>%{x}<extra></extra>",
        ))

    # NO mid line
    if no_times:
        traces.append(go.Scatter(
            x=no_times,
            y=no_mids,
            mode="lines",
            name="NO mid",
            line=dict(color="#ff9800", width=2),
            hovertemplate="NO mid=%{y:.4f}<br>%{x}<extra></extra>",
        ))

    # BUY fill markers (green)
    if buy_times:
        traces.append(go.Scatter(
            x=buy_times,
            y=buy_prices,
            mode="markers",
            name="buy fill",
            marker=dict(color="#4caf50", size=10, symbol="triangle-up"),
            hovertemplate="buy fill @ %{y:.4f}<br>%{x}<extra></extra>",
        ))

    # SELL fill markers (red)
    if sell_times:
        traces.append(go.Scatter(
            x=sell_times,
            y=sell_prices,
            mode="markers",
            name="sell fill",
            marker=dict(color="#f44336", size=10, symbol="triangle-down"),
            hovertemplate="sell fill @ %{y:.4f}<br>%{x}<extra></extra>",
        ))

    fig = go.Figure(traces)

    # Settlement vertical line + annotation
    shapes: list[dict] = []
    annotations: list[dict] = []
    if settle_time is not None:
        shapes.append(dict(
            type="line",
            x0=settle_time, x1=settle_time,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="#9c27b0", width=2, dash="dot"),
        ))
        label = f"settled {settle_outcome}" if settle_outcome else "settled"
        annotations.append(dict(
            x=settle_time,
            y=1.0,
            yref="paper",
            text=label,
            showarrow=False,
            font=dict(color="#9c27b0", size=12),
            xanchor="left",
            yanchor="top",
        ))

    fig.update_layout(
        title=f"Market trace — {condition_id}",
        xaxis_title="time (UTC)",
        yaxis_title="price (0–1)",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        shapes=shapes,
        annotations=annotations,
    )

    return save_fig(fig, out_path)
