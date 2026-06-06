#!/usr/bin/env python3
"""Round 3: fine-tune around the minimal candidate."""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
from run_v31_ablations import write_grid, run_tune, summarize, TUNE_OUT

# Minimal baseline from round 2 — kill all dead knobs, edge_buffer=0.03, d=0.75
MINIMAL = {
    "edge_max": [None],
    "min_distance_pct": [None],
    "min_bid_notional_usd": [0.0],
    "vol_clip_min": [0.0],
    "topup_enabled": [False],
    "edge_buffer": [0.03],
    "exit_safety_d": [0.75],
}

def withk(**overrides) -> dict:
    out = {**MINIMAL}
    for k, v in overrides.items():
        out[k] = v
    return out


ROUND3 = [
    # Verify exit_take_profit_mode still earns its keep on top of minimal
    ("min_legacy_exit",           withk(exit_take_profit_mode=[False])),
    # exit_fee sensitivity under minimal
    ("min_exit_fee_0",            withk(exit_fee=[0.0])),
    ("min_exit_fee_001",          withk(exit_fee=[0.001])),
    ("min_exit_fee_002",          withk(exit_fee=[0.002])),
    # d-sweep finer
    ("min_d_05",                  withk(exit_safety_d=[0.5])),
    ("min_d_06",                  withk(exit_safety_d=[0.6])),
    ("min_d_09",                  withk(exit_safety_d=[0.9])),
    ("min_d_11",                  withk(exit_safety_d=[1.1])),
    # edge_buffer + d_75 interaction
    ("min_eb_025_d075",           withk(edge_buffer=[0.025])),
    ("min_eb_025_d10",            withk(edge_buffer=[0.025], exit_safety_d=[1.0])),
    # vol_clip_max sanity — is the high clip binding?
    ("min_vol_clip_max_2",        withk(vol_clip_max=[2.0])),
    ("min_vol_clip_max_10",       withk(vol_clip_max=[10.0])),
    # favorite_threshold sanity: 0.85 vs 0.9
    ("min_fav_thresh_09",         withk(favorite_threshold=[0.9])),
    ("min_fav_thresh_08",         withk(favorite_threshold=[0.8])),
    # vol_lookback sanity
    ("min_vol_lookback_1800",     withk(vol_lookback_seconds=[1800])),
    ("min_vol_lookback_7200",     withk(vol_lookback_seconds=[7200])),
    # max_position scaling (no real impact expected)
    # tte sweep — is 24h still optimal under minimal?
    ("min_tte_12h",               withk(tte_max_seconds=[43200])),
    ("min_tte_48h",               withk(tte_max_seconds=[172800])),
]


def main():
    results = {}
    for label, override in ROUND3:
        grid_path = write_grid(label, override)
        run_dir = run_tune(label, grid_path)
        results[label] = summarize(run_dir / "results.jsonl")
    # Reference: minimal_v0 from round 2
    ref = summarize(TUNE_OUT / "v3-1-ablation-minimal_v0-2026-05-23" / "results.jsonl")
    base = summarize(TUNE_OUT / "v3-1-ablation-baseline-2026-05-23" / "results.jsonl")
    print()
    print(f"{'config':>26} {'PnL':>9} {'vs base':>8} {'vs min':>8} {'Sharpe':>7} {'maxDD':>8} {'trades':>7} {'hit':>6}")
    print(f"{'BASELINE':>26} ${base['PnL']:>7.0f} {'':>8} {'':>8} {base['Sharpe']:>7.2f} ${base['maxDD']:>6.0f} {base['trades']:>7} {base['hit']:>5.1%}")
    print(f"{'MINIMAL (ref)':>26} ${ref['PnL']:>7.0f} ${ref['PnL']-base['PnL']:>+6.0f} {'':>8} {ref['Sharpe']:>7.2f} ${ref['maxDD']:>6.0f} {ref['trades']:>7} {ref['hit']:>5.1%}")
    for label, _ in ROUND3:
        r = results[label]
        dp_base = r["PnL"] - base["PnL"]
        dp_min = r["PnL"] - ref["PnL"]
        marker = " "
        # better than minimal on all 3 dims?
        if r["PnL"] >= ref["PnL"] and r["Sharpe"] >= ref["Sharpe"] and r["maxDD"] <= ref["maxDD"]:
            marker = "*"
        print(f"{marker}{label:>25} ${r['PnL']:>7.0f} ${dp_base:>+6.0f} ${dp_min:>+6.0f} {r['Sharpe']:>7.2f} ${r['maxDD']:>6.0f} {r['trades']:>7} {r['hit']:>5.1%}")


if __name__ == "__main__":
    main()
