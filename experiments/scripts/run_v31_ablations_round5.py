#!/usr/bin/env python3
"""Round 5: converge on optimal fav × d × tp_mode under fully-minimal config."""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
from run_v31_ablations import write_grid, run_tune, summarize, TUNE_OUT

# Anchor minimal — kill all dead-weight knobs, eb=0.03, half_spread retained.
ANCHOR = {
    "edge_max": [None],
    "min_distance_pct": [None],
    "min_bid_notional_usd": [0.0],
    "vol_clip_min": [0.0],
    "topup_enabled": [False],
    "edge_buffer": [0.03],
    "half_spread_assumption": [0.005],
}

def make(fav, d, tp_mode, eb=0.03, exit_fee=0.0):
    return {**ANCHOR,
            "favorite_threshold": [fav],
            "exit_safety_d": [d],
            "exit_take_profit_mode": [tp_mode],
            "exit_fee": [exit_fee],
            "edge_buffer": [eb]}


ROUND5 = []
# fav × d sweep at tp_mode=False (the winning variant from round 4)
for fav in (0.86, 0.87, 0.88, 0.89, 0.90, 0.91):
    for d in (0.9, 1.0, 1.1):
        ROUND5.append((f"r5_f{int(fav*100)}_d{int(d*10)}_tpF", make(fav, d, False)))

# Test tp_mode=True at the most promising fav/d cells
for fav in (0.87, 0.88, 0.89, 0.90):
    for d in (0.9, 1.0):
        ROUND5.append((f"r5_f{int(fav*100)}_d{int(d*10)}_tpT", make(fav, d, True, exit_fee=0.0007)))


def main():
    results = {}
    for label, override in ROUND5:
        grid_path = write_grid(label, override)
        run_dir = run_tune(label, grid_path)
        results[label] = summarize(run_dir / "results.jsonl")

    base = summarize(TUNE_OUT / "v3-1-ablation-baseline-2026-05-23" / "results.jsonl")
    # Sort by PnL descending
    rows = []
    for label, _ in ROUND5:
        r = results[label]
        rows.append((label, r))
    rows.sort(key=lambda x: x[1]["PnL"], reverse=True)

    print(f"{'config':>22} {'PnL':>9} {'vs base':>8} {'Sharpe':>7} {'maxDD':>8} {'trades':>7} {'hit':>6}")
    print(f"{'BASELINE':>22} ${base['PnL']:>7.0f} {'':>8} {base['Sharpe']:>7.2f} ${base['maxDD']:>6.0f} {base['trades']:>7} {base['hit']:>5.1%}")
    for label, r in rows[:15]:
        dp = r["PnL"] - base["PnL"]
        # Mark Pareto-dominant on baseline
        m = "*" if (r["PnL"] >= base["PnL"] and r["Sharpe"] >= base["Sharpe"] and r["maxDD"] <= base["maxDD"]) else " "
        print(f"{m}{label:>21} ${r['PnL']:>7.0f} ${dp:>+6.0f} {r['Sharpe']:>7.2f} ${r['maxDD']:>6.0f} {r['trades']:>7} {r['hit']:>5.1%}")


if __name__ == "__main__":
    main()
