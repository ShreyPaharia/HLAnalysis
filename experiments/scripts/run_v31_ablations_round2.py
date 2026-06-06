#!/usr/bin/env python3
"""Round 2: extend edge_buffer sweep, confirm d-sweep around 0.75,
and try the minimal-knob candidate."""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
from run_v31_ablations import write_grid, run_tune, summarize, ABLATIONS, PROD, TUNE_OUT
import json

ROUND2 = [
    # Sharpen edge_buffer
    ("edge_buffer_025",           {"edge_buffer": [0.025]}),
    ("edge_buffer_04",            {"edge_buffer": [0.04]}),
    ("edge_buffer_05",            {"edge_buffer": [0.05]}),
    ("edge_buffer_06",            {"edge_buffer": [0.06]}),
    ("edge_buffer_08",            {"edge_buffer": [0.08]}),
    # The minimal candidate: kill all dead-weight knobs + ship d=0.75 (Pareto winner) + edge_buf=0.03
    ("minimal_v0",                {
        "edge_max": [None],
        "min_distance_pct": [None],
        "min_bid_notional_usd": [0.0],
        "vol_clip_min": [0.0],
        "topup_enabled": [False],
        "edge_buffer": [0.03],
        "exit_safety_d": [0.75],
    }),
    # Same but keep d=1.0 (matching prod's risk preference)
    ("minimal_d10",               {
        "edge_max": [None],
        "min_distance_pct": [None],
        "min_bid_notional_usd": [0.0],
        "vol_clip_min": [0.0],
        "topup_enabled": [False],
        "edge_buffer": [0.03],
        "exit_safety_d": [1.0],
    }),
    # Kill half_spread_assumption too (set to 0). Effectively the same as
    # raising edge_buffer by 0.005. The "right" formulation is edge_buffer
    # absorbing it (cleaner math).
    ("minimal_no_half_spread",    {
        "edge_max": [None],
        "min_distance_pct": [None],
        "min_bid_notional_usd": [0.0],
        "vol_clip_min": [0.0],
        "topup_enabled": [False],
        "half_spread_assumption": [0.0],
        "edge_buffer": [0.035],   # 0.03 + 0.005 to preserve total edge bar
        "exit_safety_d": [0.75],
    }),
]


def main():
    results = {}
    for label, override in ROUND2:
        grid_path = write_grid(label, override)
        run_dir = run_tune(label, grid_path)
        results[label] = summarize(run_dir / "results.jsonl")
    # Pull baseline from round 1 cache
    baseline_run = TUNE_OUT / "v3-1-ablation-baseline-2026-05-23"
    base = summarize(baseline_run / "results.jsonl")
    print()
    print(f"{'ablation':>26} {'PnL':>9} {'ΔPnL':>9} {'Sharpe':>7} {'ΔSh':>6} {'maxDD':>8} {'ΔDD':>7} {'trades':>7} {'hit':>6}")
    print(f"{'baseline':>26} ${base['PnL']:>7.0f} {'':>9} {base['Sharpe']:>7.2f} {'':>6} ${base['maxDD']:>6.0f} {'':>7} {base['trades']:>7} {base['hit']:>5.1%}")
    for label, _ in ROUND2:
        r = results[label]
        dp = r["PnL"] - base["PnL"]
        ds = r["Sharpe"] - base["Sharpe"]
        dd = r["maxDD"] - base["maxDD"]
        marker = " "
        if dp >= 0 and ds >= 0 and dd <= 0:
            marker = "*"
        print(f"{marker}{label:>25} ${r['PnL']:>7.0f} ${dp:>+7.0f} {r['Sharpe']:>7.2f} {ds:>+6.2f} ${r['maxDD']:>6.0f} ${dd:>+6.0f} {r['trades']:>7} {r['hit']:>5.1%}")


if __name__ == "__main__":
    main()
