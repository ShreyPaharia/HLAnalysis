"""Debug visualization for a single-market backtest run.

Reads diagnostics.parquet + fills.parquet from a run-dir and produces a 2-panel
Plotly HTML:

- Top panel: BTC reference price + strike line, with shaded "trading window"
  (last `tte_max_seconds` before expiry).
- Bottom panel: YES/NO leg asks with the entry threshold line, fill markers,
  and the resolved settlement annotation.

Run:
    uv run --extra analysis python scripts/debug_plot.py \
        --run-dir data/debug-run/pm \
        --question-id <cond_id> \
        --tte-max-seconds 7200 \
        --price-extreme-threshold 0.90 \
        --out data/debug-run/pm/debug.html
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pyarrow.parquet as pq
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def _ts_to_dt(ts_ns):
    if ts_ns is None:
        return None
    return dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
        microseconds=int(ts_ns) // 1_000
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--question-id", required=True)
    p.add_argument("--tte-max-seconds", type=int, default=7200)
    p.add_argument("--price-extreme-threshold", type=float, default=0.90)
    p.add_argument("--price-extreme-max", type=float, default=0.995)
    p.add_argument("--strike", type=float, default=None,
                   help="Override strike (else inferred from question_view)")
    p.add_argument("--title", default="")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    diag = pq.read_table(run_dir / "diagnostics" / f"{args.question_id}.parquet").to_pandas()
    fills_path = run_dir / "fills" / f"{args.question_id}.parquet"
    fills = pq.read_table(fills_path).to_pandas() if fills_path.exists() else None

    diag["dt"] = diag["ts_ns"].apply(_ts_to_dt)

    expiry_ns = int(diag["ts_ns"].max())
    tte_start_ns = expiry_ns - args.tte_max_seconds * 1_000_000_000
    tte_start_dt = _ts_to_dt(tte_start_ns)

    # Infer strike: most-likely BTC value at strike_ref_ts; we use the first
    # diag row's ref_price as an approximation if the user didn't pass one.
    strike = args.strike
    if strike is None and len(diag):
        # PM strike = day_open from kline (passed via question_view), recoverable
        # only by re-resolving; the diagnostics rows don't carry it. Use
        # ref_price[0] as visual fallback so the plot still draws a line.
        strike = float(diag["ref_price"].iloc[0]) if "ref_price" in diag.columns else 0.0

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.45, 0.55],
        vertical_spacing=0.05,
        subplot_titles=("BTC reference price vs strike",
                        "Binary leg asks (entry gate: ask ≥ threshold)"),
    )

    # --- Top: BTC ref + strike + tte window shading -----------------------
    fig.add_trace(go.Scatter(
        x=diag["dt"], y=diag["ref_price"], name="BTC ref price",
        mode="lines", line=dict(color="#3b82f6", width=1.5),
    ), row=1, col=1)
    if strike:
        fig.add_hline(
            y=strike, line=dict(color="#ef4444", dash="dash", width=2),
            annotation_text=f"strike={strike:,.0f}",
            annotation_position="top right", row=1, col=1,
        )
    # Shade the trading-eligible window
    fig.add_vrect(
        x0=tte_start_dt, x1=_ts_to_dt(expiry_ns),
        fillcolor="#22c55e", opacity=0.10, line_width=0,
        annotation_text=f"trade-eligible window ({args.tte_max_seconds//60}m before expiry)",
        annotation_position="top left", row=1, col=1,
    )

    # --- Bottom: leg asks + threshold + fills -----------------------------
    fig.add_trace(go.Scatter(
        x=diag["dt"], y=diag["yes_ask"], name="YES ask",
        mode="lines", line=dict(color="#16a34a", width=1.2),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=diag["dt"], y=diag["no_ask"], name="NO ask",
        mode="lines", line=dict(color="#dc2626", width=1.2),
    ), row=2, col=1)
    fig.add_hline(
        y=args.price_extreme_threshold,
        line=dict(color="#f59e0b", dash="dot", width=2),
        annotation_text=f"entry threshold {args.price_extreme_threshold}",
        annotation_position="bottom right", row=2, col=1,
    )
    fig.add_hline(
        y=args.price_extreme_max,
        line=dict(color="#f59e0b", dash="dot", width=1),
        annotation_text=f"price cap {args.price_extreme_max}",
        annotation_position="top right", row=2, col=1,
    )
    fig.add_vrect(
        x0=tte_start_dt, x1=_ts_to_dt(expiry_ns),
        fillcolor="#22c55e", opacity=0.10, line_width=0,
        row=2, col=1,
    )

    # Fill markers (BUY → triangle-up green, SELL → triangle-down red)
    if fills is not None and len(fills):
        fills = fills.copy()
        fills["dt"] = fills["ts_ns"].apply(_ts_to_dt)
        for side, sym, color in (
            ("buy", "triangle-up", "#16a34a"),
            ("sell", "triangle-down", "#dc2626"),
        ):
            sub = fills[fills["side"] == side]
            if not len(sub):
                continue
            fig.add_trace(go.Scatter(
                x=sub["dt"], y=sub["price"],
                mode="markers",
                marker=dict(symbol=sym, size=14, color=color,
                            line=dict(color="black", width=1)),
                name=f"fill {side}",
                text=[
                    f"{r['cloid']}<br>sym={r['symbol']}<br>size={r['size']:.2f}"
                    f"<br>price={r['price']:.4f}<br>fee={r['fee']:.4f}"
                    for _, r in sub.iterrows()
                ],
                hoverinfo="text+x+y",
            ), row=2, col=1)

    fig.update_yaxes(title_text="BTC USD", row=1, col=1)
    fig.update_yaxes(title_text="leg ask", range=[0, 1.05], row=2, col=1)
    fig.update_xaxes(title_text="UTC time", row=2, col=1)

    title = args.title or f"Debug trace: {args.question_id}"
    fig.update_layout(
        title=title,
        height=800,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.08),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
