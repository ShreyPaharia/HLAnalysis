"""Quick-and-dirty BTC + PM mid-price visualizer for one market.

Usage: uv run python scripts/plot_market_btc_pm.py <run_dir> <condition_id> [out.html]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: plot_market_btc_pm.py <run_dir> <condition_id> [out.html]", file=sys.stderr)
        sys.exit(2)
    run_dir = Path(sys.argv[1])
    cid = sys.argv[2]
    out = Path(sys.argv[3]) if len(sys.argv) >= 4 else run_dir / f"trace_{cid[:10]}.html"

    diag = pd.read_parquet(run_dir / "diagnostics" / f"{cid}.parquet")
    diag["ts"] = pd.to_datetime(diag["ts_ns"].astype("int64"), unit="ns", utc=True)
    fills = pd.read_parquet(run_dir / "fills" / f"{cid}.parquet")
    fills["ts"] = pd.to_datetime(fills["ts_ns"].astype("int64"), unit="ns", utc=True)

    manifest = json.load(open("data/sim/manifest.json"))
    m = manifest[cid]["market"]
    start_ts = int(m["start_ts_ns"])
    end_ts = int(m["end_ts_ns"])

    klines = pd.DataFrame(json.load(open("data/sim/btc_klines/2025-05-01_to_2026-05-09.json")))
    klines["ts_ns"] = klines["ts_ns"].astype("int64")
    kl = klines[(klines["ts_ns"] >= start_ts) & (klines["ts_ns"] <= end_ts + int(60 * 1e9))].copy()
    kl["ts"] = pd.to_datetime(kl["ts_ns"], unit="ns", utc=True)
    strike = float(klines[klines["ts_ns"] <= start_ts].tail(1).iloc[0]["open"])

    yes_mid = (diag["yes_bid"] + diag["yes_ask"]) / 2
    no_mid = (diag["no_bid"] + diag["no_ask"]) / 2

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=diag["ts"], y=yes_mid, name="YES mid", line=dict(color="#2196f3")),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=diag["ts"], y=no_mid, name="NO mid", line=dict(color="#ff9800")),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=kl["ts"], y=kl["close"], name="BTC close",
                             line=dict(color="#9e9e9e", width=1.5)),
                  secondary_y=True)
    fig.add_hline(y=strike, line_dash="dash", line_color="#9c27b0",
                  annotation_text=f"strike ${strike:,.0f}", annotation_position="top right",
                  secondary_y=True)

    entries = fills[fills["cloid"] != "settle"]
    settle_row = fills[fills["cloid"] == "settle"]
    settle_pnl = float(settle_row["realized_pnl_at_settle"].iloc[0]) if len(settle_row) else None
    outcome = str(settle_row["resolved_outcome"].iloc[0]) if len(settle_row) else "?"

    buys = entries[entries["side"] == "buy"]
    sells = entries[entries["side"] == "sell"]
    if len(buys):
        fig.add_trace(go.Scatter(x=buys["ts"], y=buys["price"], mode="markers", name="BUY",
                                  marker=dict(color="#4caf50", size=11, symbol="triangle-up")),
                      secondary_y=False)
    if len(sells):
        fig.add_trace(go.Scatter(x=sells["ts"], y=sells["price"], mode="markers", name="SELL (exit)",
                                  marker=dict(color="#f44336", size=11, symbol="triangle-down")),
                      secondary_y=False)
    if len(settle_row):
        settle_ts_iso = settle_row["ts"].iloc[0].isoformat()
        fig.add_shape(type="line", x0=settle_ts_iso, x1=settle_ts_iso, y0=0, y1=1,
                      yref="paper", line=dict(dash="dot", color="#9c27b0"))
        fig.add_annotation(x=settle_ts_iso, y=1.02, yref="paper", showarrow=False,
                           text=f"settle {outcome.upper()} pnl=${settle_pnl:+.2f}",
                           font=dict(color="#9c27b0"))

    fig.update_layout(
        title=f"{cid[:10]}…  strike=${strike:,.0f}  outcome={outcome}  pnl=${settle_pnl:+.2f}",
        xaxis_title="time (UTC)",
        hovermode="x unified",
        height=600,
    )
    fig.update_yaxes(title_text="PM mid (0..1)", range=[0, 1.05], secondary_y=False)
    fig.update_yaxes(title_text="BTC USD", secondary_y=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
