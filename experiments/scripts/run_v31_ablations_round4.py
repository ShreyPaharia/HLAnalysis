#!/usr/bin/env python3
"""Round 4: converge on final minimal config.

Key knobs to optimize jointly: favorite_threshold, exit_safety_d, edge_buffer.
Other ablations to confirm: kill exit_take_profit_mode + exit_fee entirely?
"""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
from run_v31_ablations import write_grid, run_tune, summarize, TUNE_OUT

# New minimal: also kill exit_take_profit_mode and exit_fee (zero contribution)
MINIMAL2 = {
    "edge_max": [None],
    "min_distance_pct": [None],
    "min_bid_notional_usd": [0.0],
    "vol_clip_min": [0.0],
    "topup_enabled": [False],
    "edge_buffer": [0.03],
    "exit_safety_d": [0.9],
    "favorite_threshold": [0.9],
    "exit_take_profit_mode": [False],
    "exit_fee": [0.0],
    "half_spread_assumption": [0.005],   # absorbed into edge_buffer; keep for now
}

def withk(**overrides) -> dict:
    out = {**MINIMAL2}
    for k, v in overrides.items():
        out[k] = v
    return out


ROUND4 = [
    ("minimal_v1", {}),
    # favorite_threshold sweep at the new optimum
    ("min_fav_092",               withk(favorite_threshold=[0.92])),
    ("min_fav_095",               withk(favorite_threshold=[0.95])),
    ("min_fav_088",               withk(favorite_threshold=[0.88])),
    # exit_safety_d joint sweep
    ("min_d_08",                  withk(exit_safety_d=[0.8])),
    ("min_d_10",                  withk(exit_safety_d=[1.0])),
    ("min_d_12",                  withk(exit_safety_d=[1.2])),
    # edge_buffer joint sweep
    ("min_eb_025",                withk(edge_buffer=[0.025])),
    ("min_eb_035",                withk(edge_buffer=[0.035])),
    ("min_eb_04",                 withk(edge_buffer=[0.04])),
    # Confirm exit_take_profit_mode is dead weight at higher fav
    ("min_tp_mode_on",            withk(exit_take_profit_mode=[True], exit_fee=[0.0007])),
    # Confirm half_spread can also be killed (just absorbed)
    ("min_no_half_spread",        withk(half_spread_assumption=[0.0], edge_buffer=[0.035])),
    # Stress test: what about flat fees for sanity? (different model, just curious)
    # Skipped — already known.

    # Big interaction test: fav=0.92 + d=1.0
    ("min_fav092_d10",            withk(favorite_threshold=[0.92], exit_safety_d=[1.0])),
    ("min_fav092_d08",            withk(favorite_threshold=[0.92], exit_safety_d=[0.8])),
    # Combined "everything tighter"
    ("min_aggro",                 withk(favorite_threshold=[0.92], exit_safety_d=[1.0],
                                         edge_buffer=[0.035])),
]


def main():
    results = {}
    for label, override in ROUND4:
        grid_path = write_grid(label, override)
        run_dir = run_tune(label, grid_path)
        results[label] = summarize(run_dir / "results.jsonl")

    base = summarize(TUNE_OUT / "v3-1-ablation-baseline-2026-05-23" / "results.jsonl")
    ref = results["minimal_v1"]
    print()
    print(f"{'config':>22} {'PnL':>9} {'vs base':>8} {'vs min':>8} {'Sharpe':>7} {'maxDD':>8} {'trades':>7} {'hit':>6}")
    print(f"{'BASELINE':>22} ${base['PnL']:>7.0f} {'':>8} {'':>8} {base['Sharpe']:>7.2f} ${base['maxDD']:>6.0f} {base['trades']:>7} {base['hit']:>5.1%}")
    print(f"{'MIN_V1 (new ref)':>22} ${ref['PnL']:>7.0f} ${ref['PnL']-base['PnL']:>+6.0f} {'':>8} {ref['Sharpe']:>7.2f} ${ref['maxDD']:>6.0f} {ref['trades']:>7} {ref['hit']:>5.1%}")
    for label, _ in ROUND4:
        if label == "minimal_v1": continue
        r = results[label]
        dp_base = r["PnL"] - base["PnL"]
        dp_min = r["PnL"] - ref["PnL"]
        marker = " "
        if r["PnL"] >= ref["PnL"] and r["Sharpe"] >= ref["Sharpe"] and r["maxDD"] <= ref["maxDD"]:
            marker = "*"
        print(f"{marker}{label:>21} ${r['PnL']:>7.0f} ${dp_base:>+6.0f} ${dp_min:>+6.0f} {r['Sharpe']:>7.2f} ${r['maxDD']:>6.0f} {r['trades']:>7} {r['hit']:>5.1%}")


if __name__ == "__main__":
    main()
