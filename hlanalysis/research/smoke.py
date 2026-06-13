"""Smoke test: build a 2-day research panel and render an HTML report.

Usage::

    HLBT_HL_DATA_ROOT=../../data python -m hlanalysis.research.smoke

Reads the 2026-06-08..2026-06-09 window, picks a couple of binary legs +
one bucket leg that settled in that window, builds the panel, computes
3-4 metrics, and writes docs/research/_smoke_report.html.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from hlanalysis.analysis.helpers import duck
from hlanalysis.research.dataset import build_panel
from hlanalysis.research.metrics import (
    implied_prob_gbm,
    settlement_convergence_curve,
    spread_bps,
    theta_decay_curve,
)
from hlanalysis.research.outcome_markets import load_market_reference, load_settlements
from hlanalysis.research.report import Report

_SMOKE_START = "2026-06-08"
_SMOKE_END = "2026-06-09"
_DT_SECONDS = 60  # 1-minute grid

# Output location relative to this file (worktree root)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_PATH = _REPO_ROOT / "docs" / "research" / "_smoke_report.html"


def main() -> None:
    data_root = os.environ.get("HLBT_HL_DATA_ROOT", "../../data")
    data_root = str(Path(data_root).resolve())

    print(f"[smoke] data_root = {data_root}")
    print(f"[smoke] window   = {_SMOKE_START} .. {_SMOKE_END}")

    con = duck()

    # --- Load reference ---
    print("[smoke] loading market_reference ...")
    ref = load_market_reference(con, data_root)
    if ref.empty:
        print("[smoke] ERROR: no market_reference rows — check data path", file=sys.stderr)
        sys.exit(1)
    print(f"[smoke]   {len(ref)} leg symbols in reference")

    settlements = load_settlements(con, data_root)
    print(f"[smoke]   {len(settlements)} settlement records")

    # --- Pick symbols to analyse ---
    # Binary legs: look for legs with expiry 2026-06-08 or 2026-06-09
    binary_legs = ref[
        (ref["market_class"] == "priceBinary")
        & (ref["expiry"].notna())
        & (
            ref["expiry"].apply(
                lambda e: getattr(e, "strftime", lambda _: "")("%Y-%m-%d") in (_SMOKE_START, _SMOKE_END)
            )
        )
    ]

    # Bucket legs: look for legs with expiry 2026-06-08 or 2026-06-09
    bucket_legs = ref[
        (ref["market_class"] == "priceBucket")
        & (ref["expiry"].notna())
        & (
            ref["expiry"].apply(
                lambda e: getattr(e, "strftime", lambda _: "")("%Y-%m-%d") in (_SMOKE_START, _SMOKE_END)
            )
        )
    ]

    # Pick at most 2 binary legs (one Yes, one No from the same outcome if possible)
    chosen_binary = []
    if not binary_legs.empty:
        # Get unique outcome_idx values sorted
        unique_outcomes = sorted(binary_legs["outcome_idx"].unique())
        for oidx in unique_outcomes[:1]:  # just one outcome pair
            pair = binary_legs[binary_legs["outcome_idx"] == oidx]
            chosen_binary = pair["symbol"].tolist()[:2]
            break

    # Pick at most 1 bucket leg that has a settlement record
    chosen_bucket = []
    settled_syms = set(settlements["symbol"].tolist()) if not settlements.empty else set()
    if not bucket_legs.empty:
        bucket_settled = bucket_legs[bucket_legs["symbol"].isin(settled_syms)]
        if not bucket_settled.empty:
            chosen_bucket = [bucket_settled.iloc[0]["symbol"]]
        else:
            chosen_bucket = [bucket_legs.iloc[0]["symbol"]]

    symbols = chosen_binary + chosen_bucket
    if not symbols:
        # Fallback: pick first few symbols from ref
        symbols = ref["symbol"].tolist()[:3]
        print(f"[smoke] WARNING: no symbols matched expiry filter; using fallback {symbols}")
    else:
        print(f"[smoke] chosen symbols: {symbols}")

    # --- Build panel ---
    print(f"[smoke] building panel (dt={_DT_SECONDS}s) ...")
    panel = build_panel(
        symbols=symbols,
        start=_SMOKE_START,
        end=_SMOKE_END,
        dt_seconds=_DT_SECONDS,
        data_root=data_root,
        cache_dir=None,  # no cache for smoke
        fresh=True,
    )
    print(f"[smoke] panel shape: {panel.shape}")

    if panel.empty:
        print("[smoke] ERROR: panel is empty", file=sys.stderr)
        sys.exit(1)

    # --- Build report ---
    rpt = Report(title="HL Outcome Market — Smoke Report")

    # Card 1: Overview
    n_rows = len(panel)
    n_syms = len(symbols)
    pct_perp = f"{100 * panel['perp_mid'].notna().mean():.1f}%"
    overview_html = f"""
    <table>
      <tr><th>Parameter</th><th>Value</th></tr>
      <tr><td>Window</td><td>{_SMOKE_START} to {_SMOKE_END}</td></tr>
      <tr><td>Grid spacing</td><td>{_DT_SECONDS}s</td></tr>
      <tr><td>Panel rows</td><td>{n_rows:,}</td></tr>
      <tr><td>Leg symbols</td><td>{n_syms} ({", ".join(symbols)})</td></tr>
      <tr><td>Perp mid coverage</td><td>{pct_perp}</td></tr>
    </table>
    """
    rpt.add_card("Panel Overview", overview_html)

    # Card 2: Spread bps per symbol (over time)
    fig2, axes2 = plt.subplots(len(symbols), 1, figsize=(11, 3 * len(symbols)), squeeze=False)
    plt.rcParams.update(
        {
            "axes.facecolor": "#161b22",
            "figure.facecolor": "#0d1117",
            "text.color": "#e6edf3",
            "axes.labelcolor": "#e6edf3",
            "xtick.color": "#8b949e",
            "ytick.color": "#8b949e",
            "grid.color": "#30363d",
        }
    )

    spread_stats_rows = []
    for i, sym in enumerate(symbols):
        bid_col = f"{sym}_bid"
        ask_col = f"{sym}_ask"
        ax = axes2[i][0]
        if bid_col in panel.columns and ask_col in panel.columns:
            valid = panel[bid_col].notna() & panel[ask_col].notna()
            if valid.sum() > 0:
                bids = panel.loc[valid, bid_col].to_numpy()
                asks = panel.loc[valid, ask_col].to_numpy()
                ts = panel.loc[valid, "timestamp"]
                spd = spread_bps(bids, asks)
                ax.plot(ts, spd, linewidth=0.8, color="#58a6ff", label=sym)
                ax.set_ylabel("spread (bps)")
                ax.set_title(f"{sym} spread bps", color="#e6edf3")
                ax.set_ylim(bottom=0)
                spread_stats_rows.append(
                    {
                        "symbol": sym,
                        "mean_bps": f"{np.nanmean(spd):.1f}",
                        "median_bps": f"{np.nanmedian(spd):.1f}",
                        "p95_bps": f"{np.nanpercentile(spd, 95):.1f}",
                        "n": str(int(valid.sum())),
                    }
                )
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="#8b949e")
        else:
            ax.text(0.5, 0.5, "no bid/ask columns", transform=ax.transAxes, ha="center", color="#8b949e")

    fig2.tight_layout()
    spread_table = _df_to_html_table(spread_stats_rows, ["symbol", "mean_bps", "median_bps", "p95_bps", "n"])
    rpt.add_card("Quoted Spread (bps)", spread_table, fig=fig2)

    # Card 3: Theta decay curve for the first symbol
    decay_html = ""
    fig3 = None
    if symbols:
        sym = symbols[0]
        decay_df = theta_decay_curve(panel, sym)
        if not decay_df.empty:
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.set_facecolor("#161b22")
            fig3.patch.set_facecolor("#0d1117")
            ax3.plot(decay_df["tte_bucket_h"], decay_df["mean_mid"], marker="o", color="#58a6ff", label="mean mid")
            ax3.fill_between(
                decay_df["tte_bucket_h"],
                decay_df["mean_mid"] - decay_df["std_mid"].fillna(0),
                decay_df["mean_mid"] + decay_df["std_mid"].fillna(0),
                alpha=0.2,
                color="#58a6ff",
            )
            ax3.set_xlabel("TTE (hours)", color="#e6edf3")
            ax3.set_ylabel("mid price", color="#e6edf3")
            ax3.set_title(f"Theta Decay: {sym}", color="#e6edf3")
            ax3.tick_params(colors="#8b949e")
            ax3.grid(color="#30363d", alpha=0.5)
            fig3.tight_layout()
            decay_html = f"<p>Symbol: <code>{sym}</code> — {len(decay_df)} TTE buckets</p>"
        else:
            decay_html = f"<p>No theta decay data for {sym}</p>"
    rpt.add_card("Theta Decay Curve", decay_html, fig=fig3)

    # Card 4: Yes/No overround for binary pairs
    overround_rows = []
    for sym in chosen_binary:
        meta_row = ref[ref["symbol"] == sym]
        if meta_row.empty:
            continue
        meta = meta_row.iloc[0]
        if meta.get("side_name") != "Yes":
            continue
        # Find No partner
        no_sym = sym[:-1] + "1"
        yes_ask_col = f"{sym}_ask"
        no_ask_col = f"{no_sym}_ask"
        if yes_ask_col in panel.columns and no_ask_col in panel.columns:
            valid = panel[yes_ask_col].notna() & panel[no_ask_col].notna()
            if valid.sum() > 0:
                ya = panel.loc[valid, yes_ask_col].to_numpy()
                na_ = panel.loc[valid, no_ask_col].to_numpy()
                ovr = ya + na_ - 1.0
                overround_rows.append(
                    {
                        "yes_sym": sym,
                        "no_sym": no_sym,
                        "mean_overround": f"{np.nanmean(ovr):.4f}",
                        "median_overround": f"{np.nanmedian(ovr):.4f}",
                        "n": str(int(valid.sum())),
                    }
                )

    # Also compute implied_prob_gbm for first binary symbol if we have target_price
    gbm_notes = []
    for sym in chosen_binary:
        meta_row = ref[ref["symbol"] == sym]
        if meta_row.empty:
            continue
        meta = meta_row.iloc[0]
        tp = meta.get("target_price")
        if tp and tp > 0:
            # Pick mid snapshot near end of day
            mid_col = f"{sym}_mid"
            if mid_col in panel.columns and "perp_mid" in panel.columns:
                last_valid = panel[panel["perp_mid"].notna() & panel[f"{sym}_tte_s"].notna()]
                if not last_valid.empty:
                    row_ = last_valid.iloc[-1]
                    spot = float(row_["perp_mid"])
                    tte_s = float(row_[f"{sym}_tte_s"])
                    sigma = (
                        float(panel["perp_realized_vol"].dropna().median())
                        if panel["perp_realized_vol"].notna().any()
                        else 0.01
                    )
                    prob = implied_prob_gbm(spot=spot, strike=tp, sigma=sigma, tau_s=tte_s / (365.25 * 86400))
                    gbm_notes.append(
                        f"{sym}: spot={spot:.0f}, K={tp:.0f}, σ={sigma:.3f}, τ={tte_s / 3600:.1f}h → P(S>K)={prob:.4f}"
                    )

    ovr_html = _df_to_html_table(overround_rows, ["yes_sym", "no_sym", "mean_overround", "median_overround", "n"])
    if not overround_rows:
        ovr_html = "<p>No binary Yes/No pairs available in this window.</p>"
    ovr_notes = " | ".join(gbm_notes) if gbm_notes else None
    rpt.add_card("Yes+No Overround & GBM Implied Prob", ovr_html, notes=ovr_notes)

    # Card 5: Settlement convergence
    conv_html = "<p>No convergence data available.</p>"
    fig5 = None
    if symbols:
        sym = symbols[0]
        conv_df = settlement_convergence_curve(panel, sym)
        if not conv_df.empty and len(conv_df) > 1:
            fig5, ax5 = plt.subplots(figsize=(10, 4))
            ax5.set_facecolor("#161b22")
            fig5.patch.set_facecolor("#0d1117")
            for label_val in conv_df["settlement_label"].unique():
                sub = conv_df[conv_df["settlement_label"] == label_val]
                color = "#58a6ff" if "yes" in str(label_val) else "#f78166"
                ax5.plot(sub["tte_bucket_h"], sub["mean_mid"], marker="o", label=str(label_val), color=color)
            ax5.set_xlabel("TTE (hours)", color="#e6edf3")
            ax5.set_ylabel("mean mid", color="#e6edf3")
            ax5.set_title(f"Settlement Convergence: {sym}", color="#e6edf3")
            ax5.tick_params(colors="#8b949e")
            ax5.grid(color="#30363d", alpha=0.5)
            ax5.legend(facecolor="#161b22", labelcolor="#e6edf3")
            fig5.tight_layout()
            conv_html = f"<p>Symbol: <code>{sym}</code></p>"
        else:
            conv_html = f"<p>Insufficient data for settlement convergence on {sym}.</p>"
    rpt.add_card("Settlement Convergence Curve", conv_html, fig=fig5)

    # --- Render ---
    print(f"[smoke] rendering report -> {_OUTPUT_PATH}")
    rpt.render(_OUTPUT_PATH)
    print(f"[smoke] done: {_OUTPUT_PATH}")

    plt.close("all")


def _df_to_html_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "<p>(no data)</p>"
    header = "".join(f"<th>{c}</th>" for c in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{row.get(c, '')}</td>" for c in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


if __name__ == "__main__":
    main()
