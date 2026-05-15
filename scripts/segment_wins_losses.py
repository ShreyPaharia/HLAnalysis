"""Segment a v1_late_resolution run's trades into wins vs losses by entry
features, by merging fills.parquet with diagnostics.parquet on ts_ns.

For each round-trip (buy + matching settle on the same question+symbol):
  - entry features: ask of bought leg, ask of other leg, ref_price (BTC),
    distance from BTC to manifest strike, time-to-expiry, recent vol.
  - outcome: resolved_outcome, leg bought (yes/no idx), pnl, win/loss.

Then prints summary stats segmented by win/loss + decile breakdowns of the
key features so you can see what differentiates them.

Usage:
    uv run --extra analysis python scripts/segment_wins_losses.py \
        --run-dir data/debug-run/pm-full-year \
        --pm-cache data/sim \
        --out data/debug-run/pm-full-year/segmentation.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--pm-cache", required=True, help="PM cache root (for manifest -> strike resolution)")
    p.add_argument("--out", default=None, help="optional HTML output with histograms")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    fills = pq.read_table(run_dir / "fills.parquet").to_pandas()
    diag = pq.read_table(run_dir / "diagnostics.parquet").to_pandas()
    manifest = json.loads((Path(args.pm_cache) / "manifest.json").read_text())

    # ---- Per-question metadata: strike + leg layout + outcome ----
    qmeta: dict[str, dict] = {}
    for cid, entry in manifest.items():
        mk = entry.get("market") or {}
        if entry.get("kind") != "binary":
            continue
        qmeta[cid] = {
            "yes_token_id": str(mk.get("yes_token_id", "")),
            "no_token_id": str(mk.get("no_token_id", "")),
            "resolved_outcome": mk.get("resolved_outcome", "unknown"),
            "start_ts_ns": int(mk.get("start_ts_ns", 0)),
            "end_ts_ns": int(mk.get("end_ts_ns", 0)),
            "strike_ref_ts_ns": int(mk.get("strike_ref_ts_ns")) if mk.get("strike_ref_ts_ns") else None,
        }

    # ---- Resolve strike per question via the cached PolymarketDataSource ----
    # Mirror its logic: bisect-right on cached klines at strike_ref_ts_ns
    # (fallback end_ts - 24h).
    from hlanalysis.backtest.data.polymarket import PolymarketDataSource
    ds = PolymarketDataSource(cache_root=Path(args.pm_cache))
    klines = ds._load_all_klines()
    klines_ts = [int(k["ts_ns"]) for k in klines]
    import bisect
    def strike_for(qid: str) -> float:
        m = qmeta.get(qid)
        if not m:
            return 0.0
        s_ts = m["strike_ref_ts_ns"] or (m["end_ts_ns"] - 24 * 3600 * 1_000_000_000)
        i = bisect.bisect_right(klines_ts, s_ts) - 1
        return float(klines[i]["close"]) if i >= 0 else 0.0

    # ---- Pair entry buys with their matching settles ----
    fills_sorted = fills.sort_values("ts_ns").reset_index(drop=True)
    open_pos: dict[tuple[str, str], dict] = {}
    trades: list[dict] = []
    for _, r in fills_sorted.iterrows():
        key = (r["question_id"], r["symbol"])
        if r["side"] == "buy":
            open_pos[key] = {
                "ts_ns": int(r["ts_ns"]),
                "entry_px": float(r["price"]),
                "size": float(r["size"]),
                "fee": float(r["fee"]),
            }
        else:
            o = open_pos.pop(key, None)
            if o is None:
                continue
            entry_cost = o["entry_px"] * o["size"] + o["fee"]
            exit_proc = float(r["price"]) * float(r["size"]) - float(r["fee"])
            trades.append({
                "qid": r["question_id"], "sym": str(r["symbol"]),
                "entry_ts_ns": o["ts_ns"], "exit_ts_ns": int(r["ts_ns"]),
                "entry_px": o["entry_px"], "exit_px": float(r["price"]),
                "size": o["size"],
                "pnl": exit_proc - entry_cost,
                "resolved_outcome": r.get("resolved_outcome"),
                "is_settle": r["cloid"] == "settle",
            })
    tdf = pd.DataFrame(trades)
    if tdf.empty:
        print("no paired round-trips")
        return 1

    # ---- Merge with diagnostics at the entry tick to recover features ----
    diag_idx = diag.set_index(["question_id", "ts_ns"])
    feats: list[dict] = []
    for _, t in tdf.iterrows():
        m = qmeta.get(t["qid"], {})
        leg_kind = "yes" if t["sym"] == m.get("yes_token_id") else (
            "no" if t["sym"] == m.get("no_token_id") else "?")
        try:
            d = diag_idx.loc[(t["qid"], t["entry_ts_ns"])]
        except KeyError:
            d = None
        strike = strike_for(t["qid"])
        ref_price = float(d["ref_price"]) if d is not None else np.nan
        chosen_ask = (float(d["yes_ask"]) if d is not None and leg_kind == "yes"
                      else (float(d["no_ask"]) if d is not None and leg_kind == "no"
                            else np.nan))
        other_ask = (float(d["no_ask"]) if d is not None and leg_kind == "yes"
                     else (float(d["yes_ask"]) if d is not None and leg_kind == "no"
                           else np.nan))
        tte_at_entry_s = (m["end_ts_ns"] - t["entry_ts_ns"]) / 1e9 if m else np.nan
        feats.append({
            **t,
            "leg_kind": leg_kind,
            "leg_matched_outcome": leg_kind == m.get("resolved_outcome"),
            "strike": strike,
            "ref_price": ref_price,
            "ref_minus_strike_pct": (ref_price - strike) / strike * 100 if strike else np.nan,
            "chosen_ask": chosen_ask,
            "other_ask": other_ask,
            "leg_lead_pct": (chosen_ask - other_ask) * 100 if not np.isnan(chosen_ask) else np.nan,
            "tte_at_entry_s": tte_at_entry_s,
        })
    fdf = pd.DataFrame(feats)
    fdf["win"] = fdf["pnl"] > 0

    # ---- Headline ----
    print(f"\n=== Round-trips: {len(fdf)} ===")
    print(f"  total PnL: ${fdf['pnl'].sum():,.2f}")
    print(f"  wins:   {fdf['win'].sum():4d}  total ${fdf.loc[fdf['win'], 'pnl'].sum():,.2f}  mean ${fdf.loc[fdf['win'], 'pnl'].mean():.2f}")
    print(f"  losses: {(~fdf['win']).sum():4d}  total ${fdf.loc[~fdf['win'], 'pnl'].sum():,.2f}  mean ${fdf.loc[~fdf['win'], 'pnl'].mean():.2f}")
    print(f"  hit rate: {fdf['win'].mean():.2%}")
    print(f"  worst losses: max single = ${fdf['pnl'].min():.2f}")

    # ---- Settlement vs mid-hold-exit decomposition ----
    print("\n=== Trade type (settle = held to expiry; non-settle = mid-hold exit_safety_d fired) ===")
    for is_s, label in [(True, "held to settle "), (False, "exited mid-hold")]:
        sub = fdf[fdf["is_settle"] == is_s]
        if not len(sub):
            continue
        w = sub["win"].sum()
        print(f"  {label}: {len(sub):4d} trades  | hit {w/len(sub):6.1%} ({w}/{len(sub)})  "
              f"| pnl ${sub['pnl'].sum():9.2f}  mean ${sub['pnl'].mean():6.2f}  worst ${sub['pnl'].min():6.2f}  best ${sub['pnl'].max():6.2f}")

    # ---- Settled trades: did the held leg win? ----
    print("\n=== Settled trades — did the held leg match the resolved outcome? ===")
    settled = fdf[fdf["is_settle"]]
    flip = ~settled["leg_matched_outcome"]
    matched = settled["leg_matched_outcome"]
    print(f"  held leg matched outcome: {matched.sum():4d}  pnl ${settled.loc[matched, 'pnl'].sum():,.2f}")
    print(f"  held leg flipped (lost):  {flip.sum():4d}  pnl ${settled.loc[flip, 'pnl'].sum():,.2f}")

    # ---- Feature deciles: win-rate and avg PnL per decile ----
    for col, label in [
        ("chosen_ask", "Entry ask (chosen leg)"),
        ("leg_lead_pct", "Lead vs other leg (pct pts)"),
        ("ref_minus_strike_pct", "BTC vs strike at entry (%)"),
        ("tte_at_entry_s", "TTE at entry (s)"),
    ]:
        sub = fdf[fdf[col].notna()].copy()
        if len(sub) < 10:
            continue
        sub["bucket"] = pd.qcut(sub[col], q=5, duplicates="drop")
        print(f"\n=== {label} — quintile segmentation ===")
        g = sub.groupby("bucket", observed=True).agg(
            n=("pnl", "size"), win_rate=("win", "mean"),
            mean_pnl=("pnl", "mean"), sum_pnl=("pnl", "sum"),
        ).reset_index()
        print(g.to_string(index=False))

    # ---- Worst 10 losses table ----
    print("\n=== Worst 10 single trades ===")
    cols = ["qid", "sym", "leg_kind", "resolved_outcome", "entry_px", "exit_px", "chosen_ask", "leg_lead_pct", "ref_minus_strike_pct", "tte_at_entry_s", "pnl"]
    worst = fdf.sort_values("pnl").head(10)[cols].copy()
    worst["qid"] = worst["qid"].str[:10] + "..."
    worst["sym"] = worst["sym"].str[:6] + "..."
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print(worst.to_string(index=False))

    # ---- Optional HTML with histograms ----
    if args.out:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            "Entry ask (chosen leg)", "Lead vs other leg (pct pts)",
            "BTC vs strike at entry (%)", "TTE at entry (s)"))
        for (r, c, col) in [(1, 1, "chosen_ask"), (1, 2, "leg_lead_pct"),
                            (2, 1, "ref_minus_strike_pct"), (2, 2, "tte_at_entry_s")]:
            wins = fdf.loc[fdf["win"], col].dropna()
            losses = fdf.loc[~fdf["win"], col].dropna()
            fig.add_trace(go.Histogram(x=wins, name="win", marker_color="#16a34a",
                                       opacity=0.6, nbinsx=30, legendgroup="win",
                                       showlegend=(r == 1 and c == 1)), row=r, col=c)
            fig.add_trace(go.Histogram(x=losses, name="loss", marker_color="#dc2626",
                                       opacity=0.6, nbinsx=30, legendgroup="loss",
                                       showlegend=(r == 1 and c == 1)), row=r, col=c)
        fig.update_layout(title="Win/Loss segmentation by entry features", barmode="overlay",
                          height=700, hovermode="x unified")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out), include_plotlyjs="cdn")
        print(f"\nhistogram → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
