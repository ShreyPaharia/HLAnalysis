"""Plot a single bucket-market trade: BTC, all strike thresholds, held leg
top-of-book, entry/exit markers. Use to explain why a bucket trade lost
money — the held leg's "winning region" (lo, hi) gets highlighted so you can
see whether BTC settled outside it.

Usage:
    uv run --extra analysis python scripts/plot_bucket_trace.py \
        <run_dir> <event_slug> [out.html]

Where <run_dir> contains diagnostics/<event_slug>.parquet and
fills/<event_slug>.parquet from a previous `run_one_question` invocation.
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
        print("usage: plot_bucket_trace.py <run_dir> <event_slug> [out.html]", file=sys.stderr)
        sys.exit(2)
    run_dir = Path(sys.argv[1])
    slug = sys.argv[2]
    out = Path(sys.argv[3]) if len(sys.argv) >= 4 else run_dir / f"trace_{slug}.html"

    diag = pd.read_parquet(run_dir / "diagnostics" / f"{slug}.parquet")
    diag["ts"] = pd.to_datetime(diag["ts_ns"].astype("int64"), unit="ns", utc=True)
    fills = pd.read_parquet(run_dir / "fills" / f"{slug}.parquet")
    fills["ts"] = pd.to_datetime(fills["ts_ns"].astype("int64"), unit="ns", utc=True)

    manifest = json.load(open("data/sim/manifest.json"))
    entry = manifest[slug]
    bucket = entry["bucket"]
    thresholds = bucket["thresholds"]            # [68000, 68400, ..., 72000]
    leg_tokens = bucket["leg_tokens"]            # [[yes_o0, no_o0], [yes_o1, no_o1], ...]
    start_ts = int(bucket["start_ts_ns"])
    end_ts = int(bucket["end_ts_ns"])

    # Identify held leg from fills.
    held_sym = fills.iloc[0]["symbol"] if len(fills) else None
    leg_idx = -1
    if held_sym is not None:
        flat = [tok for pair in leg_tokens for tok in pair]
        leg_idx = flat.index(held_sym) if held_sym in flat else -1
    outcome_pos = leg_idx // 2 if leg_idx >= 0 else None

    # YES region for the held outcome.
    if outcome_pos is None:
        bucket_lo = bucket_hi = None
    elif outcome_pos == 0:
        bucket_lo, bucket_hi = None, thresholds[0]
    elif outcome_pos == len(thresholds):
        bucket_lo, bucket_hi = thresholds[-1], None
    else:
        bucket_lo, bucket_hi = thresholds[outcome_pos - 1], thresholds[outcome_pos]

    # BTC klines
    kl_path = Path("data/sim/btc_klines/2025-05-01_to_2026-05-15.json")
    klines = pd.DataFrame(json.load(open(kl_path)))
    klines["ts_ns"] = klines["ts_ns"].astype("int64")
    pad = int(15 * 60 * 1e9)
    kl = klines[(klines["ts_ns"] >= start_ts - pad) & (klines["ts_ns"] <= end_ts + pad)].copy()
    kl["ts"] = pd.to_datetime(kl["ts_ns"], unit="ns", utc=True)

    # Held leg top-of-book over time. Diagnostics columns we have depend on
    # what the runner persists; for buckets the runner emits per-leg book
    # snapshots only for legs the strategy touched. We look up the held leg's
    # bid/ask via the canonical columns yes_bid/yes_ask/no_bid/no_ask only when
    # the leg corresponds to YES of outcome 0 (binary-shaped); otherwise we
    # fall back to "decisions" diagnostics that include the chosen ask.
    held_bid = held_ask = None
    if "yes_bid" in diag.columns and outcome_pos == 0:
        held_bid = diag["yes_bid"]; held_ask = diag["yes_ask"]
    elif "no_bid" in diag.columns and outcome_pos is not None:
        # No good per-leg data for middle buckets; we'll only get the chosen
        # ask from the entry diagnostic emitted on ENTER decisions. Build a
        # sparse line.
        pass

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # BTC line
    fig.add_trace(
        go.Scatter(x=kl["ts"], y=kl["close"], name="BTC close",
                   line=dict(color="#212121", width=1.5)),
        secondary_y=True,
    )

    # All strike thresholds as light horizontal lines.
    for thr in thresholds:
        fig.add_hline(
            y=thr, line_dash="dot", line_color="#bdbdbd", line_width=1,
            annotation_text=f"${thr:,.0f}", annotation_position="right",
            annotation_font_size=9, secondary_y=True,
        )

    # Highlight the held leg's winning region (filled band).
    if bucket_lo is not None and bucket_hi is not None:
        x0 = pd.to_datetime(start_ts - pad, unit="ns", utc=True)
        x1 = pd.to_datetime(end_ts + pad, unit="ns", utc=True)
        fig.add_shape(
            type="rect", xref="x", yref="y2",
            x0=x0, x1=x1, y0=bucket_lo, y1=bucket_hi,
            fillcolor="#4caf50", opacity=0.15, line_width=0,
        )
        fig.add_annotation(
            x=x0, y=(bucket_lo + bucket_hi) / 2, yref="y2",
            text=f"held YES region<br>${bucket_lo:,.0f}–${bucket_hi:,.0f}",
            showarrow=False, font=dict(color="#1b5e20"), xanchor="left",
        )

    # Plot held leg book if available.
    if held_bid is not None and held_ask is not None:
        fig.add_trace(go.Scatter(x=diag["ts"], y=held_ask, name="held leg ASK",
                                  line=dict(color="#f44336", dash="dot")), secondary_y=False)
        fig.add_trace(go.Scatter(x=diag["ts"], y=held_bid, name="held leg BID",
                                  line=dict(color="#4caf50", dash="dot")), secondary_y=False)

    # Entry / settle markers.
    buys = fills[(fills["side"] == "buy") & (fills["cloid"] != "settle")]
    sells = fills[(fills["side"] == "sell") & (fills["cloid"] != "settle")]
    settles = fills[fills["cloid"] == "settle"]
    if len(buys):
        fig.add_trace(go.Scatter(
            x=buys["ts"], y=buys["price"], mode="markers", name="BUY entry",
            marker=dict(color="#1b5e20", size=12, symbol="triangle-up"),
            text=[f"buy @ {p:.3f}  size={s:.2f}" for p, s in zip(buys["price"], buys["size"])],
        ), secondary_y=False)
    if len(sells):
        fig.add_trace(go.Scatter(
            x=sells["ts"], y=sells["price"], mode="markers", name="SELL exit",
            marker=dict(color="#b71c1c", size=12, symbol="triangle-down"),
        ), secondary_y=False)
    if len(settles):
        s = settles.iloc[0]
        ts_iso = s["ts"].isoformat()
        outcome = str(s.get("resolved_outcome", "?"))
        pnl = float(s.get("realized_pnl_at_settle", 0.0))
        fig.add_shape(
            type="line", x0=ts_iso, x1=ts_iso, y0=0, y1=1,
            yref="paper", line=dict(dash="dot", color="#9c27b0"),
        )
        fig.add_annotation(
            x=ts_iso, y=1.02, yref="paper", showarrow=False,
            text=f"settle outcome={outcome}  pnl=${pnl:+.2f}",
            font=dict(color="#9c27b0"),
        )

    settle_pnl = float(settles.iloc[0]["realized_pnl_at_settle"]) if len(settles) else 0.0
    fig.update_layout(
        title=(
            f"{slug}  | held YES of outcome {outcome_pos} = BTC ∈ "
            f"(${bucket_lo or '−∞'}, ${bucket_hi or '+∞'})  | pnl=${settle_pnl:+.2f}"
        ),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=70, r=70, t=80, b=80),
    )
    fig.update_yaxes(title_text="leg price [0–1]", secondary_y=False)
    fig.update_yaxes(title_text="BTC $", secondary_y=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
