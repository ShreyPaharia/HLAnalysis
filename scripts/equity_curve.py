"""Equity-curve + per-trade scatter plot for a hl-bt run output.

Reads fills.parquet from a run-dir, sums per-fill PnL (sell=+notional, buy=-notional),
sorts by ts_ns, and plots a Plotly 2-panel HTML:

- Top: cumulative equity curve (USD)
- Bottom: per-trade PnL (entry-to-settle) scatter colored by outcome,
  with hit-rate / total-PnL annotations.

Usage:
    uv run --extra analysis python scripts/equity_curve.py \
        --run-dir data/debug-run/pm-full-year --out data/debug-run/pm-full-year/equity.html
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pyarrow.parquet as pq
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def _ts_to_dt(ts_ns):
    return dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
        microseconds=int(ts_ns) // 1_000
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    fills_path = Path(args.run_dir) / "fills.parquet"
    df = pq.read_table(fills_path).to_pandas()
    if df.empty:
        print(f"no fills in {fills_path}")
        return 1
    df = df.sort_values("ts_ns").reset_index(drop=True)
    df["dt"] = df["ts_ns"].apply(_ts_to_dt)
    # Signed cash flow per fill
    df["cashflow"] = df.apply(
        lambda r: (r["price"] * r["size"] - r["fee"]) if r["side"] == "sell"
        else -(r["price"] * r["size"] + r["fee"]),
        axis=1,
    )
    df["equity"] = df["cashflow"].cumsum()

    # Per-trade roundtrip PnL: pair each buy with its matching sell on the
    # same question_id+symbol (entry → settle/exit).
    trades: list[dict] = []
    open_pos: dict[tuple[str, str], dict] = {}
    for _, r in df.iterrows():
        key = (r["question_id"], r["symbol"])
        if r["side"] == "buy":
            open_pos[key] = {"entry_ts": r["dt"], "entry_px": r["price"],
                             "entry_fee": r["fee"], "size": r["size"],
                             "outcome": r.get("resolved_outcome")}
        else:
            o = open_pos.pop(key, None)
            if o is None:
                continue
            entry_cost = o["entry_px"] * o["size"] + o["entry_fee"]
            exit_proceeds = r["price"] * r["size"] - r["fee"]
            pnl = exit_proceeds - entry_cost
            trades.append({
                "ts": r["dt"], "qid": r["question_id"], "sym": r["symbol"],
                "entry_px": o["entry_px"], "exit_px": r["price"], "size": r["size"],
                "pnl": pnl, "outcome": r.get("resolved_outcome") or o["outcome"],
            })

    if trades:
        win = sum(1 for t in trades if t["pnl"] > 0)
        hit = win / len(trades)
        total = sum(t["pnl"] for t in trades)
    else:
        hit = 0.0
        total = float(df["equity"].iloc[-1])

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45],
        vertical_spacing=0.06,
        subplot_titles=(
            f"Cumulative equity — total ${total:,.2f}, n_trades={len(trades)}, hit_rate={hit:.1%}",
            "Per-trade PnL (entry→settle)",
        ),
    )
    fig.add_trace(go.Scatter(
        x=df["dt"], y=df["equity"], mode="lines+markers",
        line=dict(color="#3b82f6", width=1.5),
        marker=dict(size=3, color="#3b82f6"),
        name="equity",
        hovertemplate="%{x}<br>$%{y:.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="#94a3b8", dash="dot"), row=1, col=1)

    if trades:
        ts = [t["ts"] for t in trades]
        pnls = [t["pnl"] for t in trades]
        colors = ["#16a34a" if p > 0 else "#dc2626" for p in pnls]
        text = [
            f"qid={t['qid'][:10]}<br>sym={t['sym'][:20]}<br>"
            f"in={t['entry_px']:.4f} out={t['exit_px']:.4f} sz={t['size']:.1f}<br>"
            f"outcome={t['outcome']}"
            for t in trades
        ]
        fig.add_trace(go.Scatter(
            x=ts, y=pnls, mode="markers",
            marker=dict(size=6, color=colors,
                        line=dict(color="black", width=0.3)),
            name="trade PnL", text=text, hoverinfo="text+x+y",
        ), row=2, col=1)
        fig.add_hline(y=0, line=dict(color="#94a3b8", dash="dot"), row=2, col=1)

    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="USD per trade", row=2, col=1)
    fig.update_xaxes(title_text="UTC time", row=2, col=1)
    title = args.title or f"Equity curve: {args.run_dir}"
    fig.update_layout(title=title, height=750, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.08), showlegend=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"wrote {out}  (n_trades={len(trades)}, total=${total:,.2f}, hit={hit:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
