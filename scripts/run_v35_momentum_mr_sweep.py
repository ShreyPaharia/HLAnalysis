#!/usr/bin/env python3
"""v3.5 momentum/MR sweep on PM BTC corpus.

Mirrors scripts/run_v31_ablations.py — each cell writes a grid yaml, drives
`hl-bt tune` walk-forward, and prints a comparison vs the v3.1 final
baseline (config/tuning.v3-1-final-pm.yaml).

Cells:
  - 4 indicators × 4 lookbacks × 3 tau_gate values = 48 gate cells
  - 4 indicators × 4 lookbacks × 3 alpha_tilt values = 48 tilt cells
  Total: 96 cells.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "sim"
TUNE_OUT = DATA_ROOT / "tuning"

INDICATORS = ["z_ret", "rsi", "ma_sigma", "hurst_ou"]
LOOKBACKS = [5, 15, 30, 60]
TAU_GATES = [0.5, 1.0, 1.5]
ALPHA_TILTS = [0.25, 0.5, 1.0]

# Anchor (v3.1 final). The sweep only overrides momentum_mr_* keys.
ANCHOR = {
    "vol_lookback_seconds": [3600],
    "vol_sampling_dt_seconds": [60],
    "vol_clip_min": [0.0],
    "vol_clip_max": [5.0],
    "fee_taker": [0.0],
    "half_spread_assumption": [0.005],
    "drift_lookback_seconds": [3600],
    "drift_blend": [0.0],
    "edge_max": [None],
    "min_distance_pct": [None],
    "topup_enabled": [False],
    "min_bid_notional_usd": [10.0],
    "favorite_threshold": [0.90],
    "edge_buffer": [0.03],
    "exit_safety_d": [1.0],
    "exit_take_profit_mode": [True],
    "exit_fee": [0.0007],
    "tte_min_seconds": [0],
    "tte_max_seconds": [86400],
    "max_position_usd": [200.0],
    "stop_loss_pct": [None],
    "exit_edge_threshold": [0.0],
    "take_profit_price": [None],
    "time_stop_seconds": [0],
    "gamma_lambda": [0.0],
}


def make_cell(*, indicator: str, lookback_min: int, mode: str,
              tau_gate: float | None = None, alpha_tilt: float | None = None) -> dict:
    cell = {
        **ANCHOR,
        "momentum_mr_enabled": [True],
        "momentum_mr_indicator": [indicator],
        "momentum_mr_lookback_min": [lookback_min],
        "momentum_mr_mode": [mode],
    }
    if tau_gate is not None:
        cell["momentum_mr_tau_gate"] = [tau_gate]
    if alpha_tilt is not None:
        cell["momentum_mr_alpha_tilt"] = [alpha_tilt]
    return cell


def write_grid(label: str, grid: dict) -> Path:
    import yaml
    cfg = {
        "grids": {"v3_5_momentum_mr": grid},
        "run": {
            "train_markets": 60, "test_markets": 60,
            "step_markets": 60, "max_workers": 4,
        },
    }
    out_path = REPO_ROOT / "config" / f"tuning.v3-5-{label}.yaml"
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def run_tune(label: str, grid_path: Path, workers: int = 4) -> Path:
    run_id = f"v3-5-{label}-2026-05-28"
    out_run = TUNE_OUT / run_id
    if (out_run / "results.jsonl").exists():
        print(f"[skip] {label}")
        return out_run
    cmd = [
        "uv", "run", "hl-bt", "tune",
        "--strategy", "v3_5_momentum_mr",
        "--data-source", "polymarket",
        "--grid", str(grid_path),
        "--run-id", run_id,
        "--out-dir", str(TUNE_OUT),
        "--start", "2025-05-08", "--end", "2026-05-08",
        "--fee-model", "pm_binary", "--fee-rate", "0.07",
        "--kind", "binary",
        "--workers", str(workers),
    ]
    env = {**os.environ, "HLBT_PM_CACHE_ROOT": str(DATA_ROOT)}
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
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
        return {"PnL": 0, "Sharpe": 0, "maxDD": 0, "trades": 0, "hit": 0,
                "per_split_pnl": []}
    pnl = sum(s["total_pnl_usd"] for s in ss)
    sharpe = sum(s["sharpe"] for s in ss) / len(ss)
    mdd = max(s["max_drawdown_usd"] for s in ss)
    trades = sum(s["n_trades"] for s in ss)
    hit = sum(s["hit_rate"] * s["n_trades"] for s in ss) / max(trades, 1)
    return {"PnL": pnl, "Sharpe": sharpe, "maxDD": mdd, "trades": trades,
            "hit": hit,
            "per_split_pnl": [s["total_pnl_usd"] for s in ss],
            "worst_split_sharpe": min(s["sharpe"] for s in ss),
            "splits": len(ss)}


def run_baseline(workers: int = 4) -> Path:
    """Run v3.1 final on the same date window so v3.5 cells can be Δ-compared."""
    run_id = "v3-1-final-pm-walkforward-2026-05-28"
    out_run = TUNE_OUT / run_id
    if (out_run / "results.jsonl").exists():
        print(f"[skip] baseline already exists at {out_run}")
        return out_run
    cmd = [
        "uv", "run", "hl-bt", "tune",
        "--strategy", "v3_theta_harvester",
        "--data-source", "polymarket",
        "--grid", "config/tuning.v3-1-final-pm.yaml",
        "--run-id", run_id,
        "--out-dir", str(TUNE_OUT),
        "--start", "2025-05-08", "--end", "2026-05-08",
        "--fee-model", "pm_binary", "--fee-rate", "0.07",
        "--kind", "binary",
        "--workers", str(workers),
    ]
    env = {**os.environ, "HLBT_PM_CACHE_ROOT": str(DATA_ROOT)}
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"[FAIL baseline] rc={proc.returncode}")
        print(proc.stderr[-2000:])
        raise SystemExit(1)
    print(f"[ok] baseline in {dt:.1f}s")
    return out_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indicator",
        choices=INDICATORS + ["all"],
        default="all",
        help="Restrict sweep to a single indicator (default: all 4).",
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Only run the v3.1 baseline, then exit. Use to pre-compute Δ-reference.",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip generating the v3.1 baseline (assume it exists).",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Workers for each `hl-bt tune` invocation.",
    )
    args = parser.parse_args()

    if not args.skip_baseline:
        run_baseline(workers=args.workers)
    if args.baseline_only:
        return 0

    indicators = INDICATORS if args.indicator == "all" else [args.indicator]
    results: dict[str, dict] = {}
    cells: list[tuple[str, dict]] = []

    for ind in indicators:
        for lb in LOOKBACKS:
            for tau in TAU_GATES:
                label = f"gate-{ind}-lb{lb}-tau{int(tau*10):02d}"
                cells.append((label, make_cell(
                    indicator=ind, lookback_min=lb, mode="gate", tau_gate=tau,
                )))
            for alpha in ALPHA_TILTS:
                label = f"tilt-{ind}-lb{lb}-a{int(alpha*100):03d}"
                cells.append((label, make_cell(
                    indicator=ind, lookback_min=lb, mode="tilt", alpha_tilt=alpha,
                )))

    print(f"running {len(cells)} cells for indicator(s): {indicators}")
    for label, grid in cells:
        gp = write_grid(label, grid)
        rd = run_tune(label, gp, workers=args.workers)
        results[label] = summarize(rd / "results.jsonl")

    # Baseline: v3.1 final on identical splits (regenerated above if needed)
    baseline_run = TUNE_OUT / "v3-1-final-pm-walkforward-2026-05-28"
    if (baseline_run / "results.jsonl").exists():
        base = summarize(baseline_run / "results.jsonl")
    else:
        print("[warn] v3.1 baseline run not found; "
              "absolute PnL only, no Δ-vs-baseline column")
        base = {"PnL": 0, "Sharpe": 0, "maxDD": 0, "trades": 0,
                "hit": 0, "worst_split_sharpe": 0}

    rows = [(label, results[label]) for label, _ in cells]
    rows.sort(key=lambda x: x[1]["PnL"], reverse=True)
    print()
    print(f"{'label':>45} {'PnL':>9} {'ΔPnL':>9} {'Sharpe':>7} {'worst':>7} "
          f"{'maxDD':>8} {'trades':>7} {'hit':>6}")
    print(f"{'BASELINE (v3.1 final)':>45} ${base['PnL']:>7.0f} {'':>9} "
          f"{base['Sharpe']:>7.2f} {base.get('worst_split_sharpe', 0):>7.2f} "
          f"${base['maxDD']:>6.0f} {base['trades']:>7} {base['hit']:>5.1%}")
    for label, r in rows[:30]:
        dp = r["PnL"] - base["PnL"]
        ship = (r["PnL"] >= base["PnL"]
                and r["worst_split_sharpe"] > base.get("worst_split_sharpe", 0)
                and r["maxDD"] <= base["maxDD"]
                and r["trades"] >= 0.6 * base["trades"])
        marker = "*" if ship else " "
        print(f"{marker}{label:>44} ${r['PnL']:>7.0f} ${dp:>+7.0f} "
              f"{r['Sharpe']:>7.2f} {r['worst_split_sharpe']:>7.2f} "
              f"${r['maxDD']:>6.0f} {r['trades']:>7} {r['hit']:>5.1%}")

    out_json = TUNE_OUT / "v3-5-momentum-mr-summary-2026-05-28.json"
    out_json.write_text(json.dumps({"baseline": base, "cells": results}, indent=2))
    print(f"\nfull results → {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
