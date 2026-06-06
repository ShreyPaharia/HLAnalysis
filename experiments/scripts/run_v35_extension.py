#!/usr/bin/env python3
"""v3.5 extension sweeps — densify around the winners from the first round.

Usage:
  uv run python scripts/run_v35_extension.py z_ret_corner
  uv run python scripts/run_v35_extension.py ma_sigma_neighbor

Probes:
  z_ret_corner     — lb ∈ {1,2,3,5,7,10} × α ∈ {1.0,1.5,2.0,2.5} tilt only (24)
  ma_sigma_neighbor — lb ∈ {20,25,30,35,40} × α ∈ {0.75,1.0,1.25} tilt only (15)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "sim"
TUNE_OUT = DATA_ROOT / "tuning"

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

GRIDS = {
    "z_ret_corner": {
        "indicator": "z_ret",
        "cells": [
            (lb, alpha)
            for lb in (1, 2, 3, 5, 7, 10)
            for alpha in (1.0, 1.5, 2.0, 2.5)
        ],
    },
    "ma_sigma_neighbor": {
        "indicator": "ma_sigma",
        "cells": [
            (lb, alpha)
            for lb in (20, 25, 30, 35, 40)
            for alpha in (0.75, 1.0, 1.25)
        ],
    },
}


def cell_grid(indicator: str, lookback_min: int, alpha_tilt: float) -> dict:
    return {
        **ANCHOR,
        "momentum_mr_enabled": [True],
        "momentum_mr_indicator": [indicator],
        "momentum_mr_lookback_min": [lookback_min],
        "momentum_mr_mode": ["tilt"],
        "momentum_mr_alpha_tilt": [alpha_tilt],
    }


def write_grid(label: str, grid: dict) -> Path:
    import yaml
    cfg = {
        "grids": {"v3_5_momentum_mr": grid},
        "run": {
            "train_markets": 60, "test_markets": 60,
            "step_markets": 60, "max_workers": 4,
        },
    }
    out_path = REPO_ROOT / "config" / f"tuning.v3-5-ext-{label}.yaml"
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def run_cell(label: str, grid_path: Path) -> Path:
    run_id = f"v3-5-ext-{label}-2026-05-28"
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
        "--workers", "4",
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


def summarize(p: Path) -> dict:
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    ss = [r["summary"] for r in rows]
    if not ss:
        return {}
    pnl = sum(s["total_pnl_usd"] for s in ss)
    sharpe = sum(s["sharpe"] for s in ss) / len(ss)
    mdd = max(s["max_drawdown_usd"] for s in ss)
    trades = sum(s["n_trades"] for s in ss)
    hit = sum(s["hit_rate"] * s["n_trades"] for s in ss) / max(trades, 1)
    worst = min(s["sharpe"] for s in ss)
    return {"PnL": pnl, "Sharpe": sharpe, "worst": worst, "maxDD": mdd,
            "trades": trades, "hit": hit}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in GRIDS:
        print(f"Usage: {sys.argv[0]} {{{'|'.join(GRIDS)}}}")
        return 1
    name = sys.argv[1]
    spec = GRIDS[name]
    indicator = spec["indicator"]

    baseline = summarize(TUNE_OUT / "v3-1-final-pm-walkforward-2026-05-28" / "results.jsonl")
    print(f"baseline ({indicator}): PnL ${baseline['PnL']:.0f}, Sharpe {baseline['Sharpe']:.2f}, "
          f"WFS {baseline['worst']:.3f}, maxDD ${baseline['maxDD']:.0f}, trades {baseline['trades']}")
    print()

    results = []
    for lb, alpha in spec["cells"]:
        label = f"tilt-{indicator}-lb{lb}-a{int(alpha*100):03d}"
        grid = cell_grid(indicator, lb, alpha)
        gp = write_grid(label, grid)
        run_dir = run_cell(label, gp)
        s = summarize(run_dir / "results.jsonl")
        if not s:
            continue
        ships = (s["PnL"] >= baseline["PnL"]
                 and s["worst"] > baseline["worst"]
                 and s["maxDD"] <= baseline["maxDD"]
                 and s["trades"] >= 0.6 * baseline["trades"])
        results.append((label, lb, alpha, s, ships))

    print()
    results.sort(key=lambda r: r[3]["PnL"], reverse=True)
    print(f"{'cell':<32} {'PnL':>7} {'ΔPnL':>6} {'Sharpe':>7} {'WFS':>6} {'maxDD':>7} {'trades':>6} {'hit':>6}  ship")
    for label, lb, alpha, s, ships in results:
        dp = s["PnL"] - baseline["PnL"]
        marker = "*" if ships else " "
        print(f"{marker}{label:<31} ${s['PnL']:>6.0f} ${dp:>+5.0f} {s['Sharpe']:>7.2f} "
              f"{s['worst']:>6.3f} ${s['maxDD']:>6.0f} {s['trades']:>6} {s['hit']:.1%}  {marker:>4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
