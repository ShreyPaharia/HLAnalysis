"""Per-market price trace plot.

`plot_market_trace(question_id, run_dir, out_path)` loads per-tick diagnostics
and fill data for one question and writes an interactive Plotly HTML showing
YES/NO mid-price series with entry/exit fill markers and a settlement
annotation. Returns `out_path` on success or `None` when the diagnostics
parquet is missing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq

from ._common import save_fig


def _load_parquet_as_dict(path: Path) -> Optional[dict[str, list]]:
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
    import datetime
    ts_us = ts_ns // 1_000
    dt = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(microseconds=ts_us)
    return dt.isoformat()


def plot_market_trace(
    question_id: str,
    run_dir: Path,
    out_path: Path,
) -> Optional[Path]:
    diag_path = run_dir / "diagnostics" / f"{question_id}.parquet"
    diag = _load_parquet_as_dict(diag_path)
    if diag is None:
        return None

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

    fills_path = run_dir / "fills" / f"{question_id}.parquet"
    fills = _load_parquet_as_dict(fills_path) or {}

    fill_ts_col = fills.get("ts_ns", [])
    fill_cloid_col = fills.get("cloid", [])
    fill_side_col = fills.get("side", [])
    fill_price_col = fills.get("price", [])
    fill_resolved_outcome_col = fills.get("resolved_outcome", [])

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
            ro = fill_resolved_outcome_col[i] if i < len(fill_resolved_outcome_col) else None
            if ro == "yes":
                settle_outcome = "YES"
            elif ro == "no":
                settle_outcome = "NO"
            else:
                settle_outcome = None
        else:
            if side == "buy":
                buy_times.append(dt_str)
                buy_prices.append(float(price) if price is not None else 0.5)
            else:
                sell_times.append(dt_str)
                sell_prices.append(float(price) if price is not None else 0.5)

    import plotly.graph_objects as go

    traces: list = []
    if yes_times:
        traces.append(go.Scatter(
            x=yes_times, y=yes_mids, mode="lines", name="YES mid",
            line=dict(color="#2196f3", width=2),
            hovertemplate="YES mid=%{y:.4f}<br>%{x}<extra></extra>",
        ))
    if no_times:
        traces.append(go.Scatter(
            x=no_times, y=no_mids, mode="lines", name="NO mid",
            line=dict(color="#ff9800", width=2),
            hovertemplate="NO mid=%{y:.4f}<br>%{x}<extra></extra>",
        ))
    if buy_times:
        traces.append(go.Scatter(
            x=buy_times, y=buy_prices, mode="markers", name="buy fill",
            marker=dict(color="#4caf50", size=10, symbol="triangle-up"),
            hovertemplate="buy fill @ %{y:.4f}<br>%{x}<extra></extra>",
        ))
    if sell_times:
        traces.append(go.Scatter(
            x=sell_times, y=sell_prices, mode="markers", name="sell fill",
            marker=dict(color="#f44336", size=10, symbol="triangle-down"),
            hovertemplate="sell fill @ %{y:.4f}<br>%{x}<extra></extra>",
        ))

    fig = go.Figure(traces)

    shapes: list[dict] = []
    annotations: list[dict] = []
    if settle_time is not None:
        shapes.append(dict(
            type="line",
            x0=settle_time, x1=settle_time, y0=0, y1=1, yref="paper",
            line=dict(color="#9c27b0", width=2, dash="dot"),
        ))
        label = f"settled {settle_outcome}" if settle_outcome else "settled"
        annotations.append(dict(
            x=settle_time, y=1.0, yref="paper", text=label,
            showarrow=False, font=dict(color="#9c27b0", size=12),
            xanchor="left", yanchor="top",
        ))

    fig.update_layout(
        title=f"Market trace — {question_id}",
        xaxis_title="time (UTC)",
        yaxis_title="price (0–1)",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        shapes=shapes,
        annotations=annotations,
    )

    return save_fig(fig, out_path)
