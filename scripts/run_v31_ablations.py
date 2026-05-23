#!/usr/bin/env python3
"""V3.1 ablation runner.

Drives `hl-bt tune` once per ablation cell, using a single-cell grid yaml
generated from a prod baseline + one knob override. Walk-forward gives 5 OOS
splits per ablation. PM with realistic fee curve (pm_binary, rate=0.07).

Output: a flat table comparing each ablation to baseline on PnL / avg-Sharpe /
maxDD / trade count.
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis")
WORKTREE = REPO_ROOT / ".worktrees/do-an-analysis-of-the-final-strategies-and-see-if--53OYTm"
DATA_ROOT = REPO_ROOT / "data/sim"
TUNE_OUT = DATA_ROOT / "tuning"

PROD = {
    "vol_lookback_seconds": [3600],
    "vol_sampling_dt_seconds": [60],
    "vol_clip_min": [0.05],
    "vol_clip_max": [5.0],
    "edge_buffer": [0.02],
    "fee_taker": [0.0],
    "half_spread_assumption": [0.005],
    "drift_lookback_seconds": [3600],
    "drift_blend": [0.0],
    "favorite_threshold": [0.85],
    "tte_min_seconds": [0],
    "tte_max_seconds": [86400],          # PM 24h optimal
    "edge_max": [0.20],
    "min_distance_pct": [0.002],
    "min_bid_notional_usd": [10.0],
    "max_position_usd": [200.0],
    "stop_loss_pct": [None],
    "exit_edge_threshold": [0.0],
    "take_profit_price": [None],
    "time_stop_seconds": [0],
    "gamma_lambda": [0.0],
    "exit_take_profit_mode": [True],
    "exit_fee": [0.0007],
    "exit_safety_d": [1.0],
}

# Each ablation: (label, override_dict, optional topup flag in run yaml)
# Use list-valued overrides (grid syntax expects lists)
ABLATIONS: list[tuple[str, dict, dict]] = [
    ("baseline",                  {},                                       {}),
    # Entry-side
    ("no_edge_max",               {"edge_max": [None]},                     {}),
    ("no_half_spread",            {"half_spread_assumption": [0.0]},        {}),
    ("no_min_distance_pct",       {"min_distance_pct": [None]},             {}),
    ("no_min_bid_notional",       {"min_bid_notional_usd": [0.0]},          {}),
    ("edge_buffer_01",            {"edge_buffer": [0.01]},                  {}),
    ("edge_buffer_03",            {"edge_buffer": [0.03]},                  {}),
    ("no_vol_clip_min",           {"vol_clip_min": [0.0]},                  {}),
    # Exit-side
    ("legacy_exit_mode",          {"exit_take_profit_mode": [False],
                                    "exit_edge_threshold": [0.0]},          {}),
    ("exit_fee_0",                {"exit_fee": [0.0]},                      {}),
    ("exit_fee_0014",             {"exit_fee": [0.0014]},                   {}),
    ("d_0",                       {"exit_safety_d": [0.0]},                 {}),
    ("d_05",                      {"exit_safety_d": [0.5]},                 {}),
    ("d_075",                     {"exit_safety_d": [0.75]},                {}),
    # Topup (cfg-level, but grid still drives it through ThetaHarvesterConfig fields)
    ("no_topup",                  {"topup_enabled": [False]},               {}),
    # Sanity: ensure favorite_threshold is doing work
    ("fav_thresh_07",             {"favorite_threshold": [0.7]},            {}),
]


def write_grid(label: str, override: dict) -> Path:
    grid = {**PROD, **override}
    cfg = {
        "grids": {"v3_theta_harvester": grid},
        "run": {"train_markets": 60, "test_markets": 60, "step_markets": 60, "max_workers": 4},
    }
    out_path = WORKTREE / "config" / f"tuning.v3-1-ablation-{label}.yaml"
    import yaml  # available in venv
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def run_tune(label: str, grid_path: Path) -> Path:
    run_id = f"v3-1-ablation-{label}-2026-05-23"
    out_run = TUNE_OUT / run_id
    if (out_run / "results.jsonl").exists():
        print(f"[skip] {label} already has results")
        return out_run
    env = {"HLBT_PM_CACHE_ROOT": str(DATA_ROOT)}
    cmd = [
        "uv", "run", "hl-bt", "tune",
        "--strategy", "v3_theta_harvester",
        "--data-source", "polymarket",
        "--grid", str(grid_path),
        "--run-id", run_id,
        "--out-dir", str(TUNE_OUT),
        "--start", "2025-05-08", "--end", "2026-05-08",
        "--fee-model", "pm_binary", "--fee-rate", "0.07",
        "--kind", "binary",
        "--workers", "4",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=WORKTREE, capture_output=True, text=True,
                           env={**__import__("os").environ, **env})
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"[FAIL {label}] rc={proc.returncode}")
        print(proc.stderr[-2000:])
        raise SystemExit(1)
    print(f"[ok] {label} in {dt:.1f}s")
    return out_run


def summarize(results_path: Path) -> dict:
    rows = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    ss = [r["summary"] for r in rows]
    if not ss:
        return {"PnL": 0, "Sharpe": 0, "maxDD": 0, "trades": 0, "hit": 0, "splits": 0,
                "per_split_pnl": []}
    pnl = sum(s["total_pnl_usd"] for s in ss)
    sharpe = sum(s["sharpe"] for s in ss) / len(ss)
    mdd = max(s["max_drawdown_usd"] for s in ss)
    trades = sum(s["n_trades"] for s in ss)
    hit = sum(s["hit_rate"] * s["n_trades"] for s in ss) / max(trades, 1)
    return {"PnL": pnl, "Sharpe": sharpe, "maxDD": mdd, "trades": trades,
            "hit": hit, "splits": len(ss),
            "per_split_pnl": [s["total_pnl_usd"] for s in ss]}


def main() -> int:
    results = {}
    for label, override, _ in ABLATIONS:
        grid_path = write_grid(label, override)
        run_dir = run_tune(label, grid_path)
        results[label] = summarize(run_dir / "results.jsonl")
    base = results["baseline"]
    print()
    print(f"{'ablation':>22} {'PnL':>9} {'ΔPnL':>9} {'Sharpe':>7} {'ΔSh':>6} {'maxDD':>8} {'ΔDD':>7} {'trades':>7} {'hit':>6}")
    for label, _, _ in ABLATIONS:
        r = results[label]
        dp = r["PnL"] - base["PnL"]
        ds = r["Sharpe"] - base["Sharpe"]
        dd = r["maxDD"] - base["maxDD"]
        marker = " "
        if label != "baseline":
            # Strict Pareto: ablation drops only if all three are at least as good as baseline
            #   (i.e., higher/equal PnL AND Sharpe AND lower/equal DD)
            if dp >= 0 and ds >= 0 and dd <= 0:
                marker = "*"  # ablation Pareto-dominates baseline → knob is dead weight
        print(f"{marker}{label:>21} ${r['PnL']:>7.0f} ${dp:>+7.0f} {r['Sharpe']:>7.2f} {ds:>+6.2f} ${r['maxDD']:>6.0f} ${dd:>+6.0f} {r['trades']:>7} {r['hit']:>5.1%}")
    out_json = TUNE_OUT / "v3-1-ablation-summary-2026-05-23.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nfull results → {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
