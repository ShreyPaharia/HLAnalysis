#!/usr/bin/env python3
"""v3.6 sweep: OU-Z + SDR core grid + JR-trust-weight stack on PM.

Cells:
  - OU-Z tilt: lb ∈ {5, 15, 30, 60} × α ∈ {0.25, 0.5, 1.0} = 12
  - SDR  tilt: lb ∈ {5, 15, 30, 60} × α ∈ {0.25, 0.5, 1.0} = 12
  - JR-stack: top-4 cells (v3.5 + v3.6) with momentum_mr_jr_trust_weight=True
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


def make_cell(indicator: str, lb: int, alpha: float, jr: bool = False) -> dict:
    return {
        **ANCHOR,
        "momentum_mr_enabled": [True],
        "momentum_mr_indicator": [indicator],
        "momentum_mr_lookback_min": [lb],
        "momentum_mr_mode": ["tilt"],
        "momentum_mr_alpha_tilt": [alpha],
        "momentum_mr_jr_trust_weight": [jr],
    }


def write_grid(label: str, grid: dict) -> Path:
    import yaml
    cfg = {
        "grids": {"v3_5_momentum_mr": grid},
        "run": {"train_markets": 60, "test_markets": 60,
                "step_markets": 60, "max_workers": 4},
    }
    out_path = REPO_ROOT / "config" / f"tuning.v3-6-{label}.yaml"
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def run_cell(label: str, grid_path: Path) -> Path:
    run_id = f"v3-6-{label}-2026-05-28"
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
    base = summarize(TUNE_OUT / "v3-1-final-pm-walkforward-2026-05-28" / "results.jsonl")
    print(f"baseline: PnL ${base['PnL']:.0f}, WFS {base['worst']:.3f}, "
          f"maxDD ${base['maxDD']:.0f}, trades {base['trades']}, hit {base['hit']:.2%}")
    print()

    cells: list[tuple[str, str, int, float, bool]] = []

    # Core OU-Z + SDR grid (no JR)
    for ind in ("ou_z", "sdr"):
        for lb in (5, 15, 30, 60):
            for alpha in (0.25, 0.5, 1.0):
                label = f"tilt-{ind}-lb{lb}-a{int(alpha*100):03d}"
                cells.append((label, ind, lb, alpha, False))

    # JR stack — apply jr_trust_weight=True to the best v3.5/v3.6 cells
    # v3.5 winners: ma_sigma-lb30-a100, z_ret-lb3-a100 (corner densify), z_ret-lb5-a100
    # v3.6 winners: TBD — add JR variants of best ou_z and sdr lb/alpha after core
    # For now, hard-stack JR onto the known v3.5 PM winners + leading v3.6 candidates
    jr_targets = [
        ("ma_sigma", 30, 1.0),
        ("z_ret", 3, 1.0),
        ("z_ret", 5, 1.0),
        ("ou_z", 30, 1.0),
        ("sdr", 3, 1.0),
        ("sdr", 5, 1.0),
    ]
    for ind, lb, alpha in jr_targets:
        label = f"jr-tilt-{ind}-lb{lb}-a{int(alpha*100):03d}"
        cells.append((label, ind, lb, alpha, True))

    results = []
    for label, ind, lb, alpha, jr in cells:
        grid = make_cell(ind, lb, alpha, jr)
        gp = write_grid(label, grid)
        rd = run_cell(label, gp)
        s = summarize(rd / "results.jsonl")
        if not s:
            continue
        ships = (s["PnL"] >= base["PnL"]
                 and s["worst"] > base["worst"]
                 and s["maxDD"] <= base["maxDD"]
                 and s["trades"] >= 0.6 * base["trades"])
        results.append((label, ind, lb, alpha, jr, s, ships))

    print()
    results.sort(key=lambda r: r[5]["PnL"], reverse=True)
    print(f"{'cell':<38} {'PnL':>7} {'ΔPnL':>6} {'Sharpe':>7} {'WFS':>6} {'maxDD':>7} {'trades':>6} {'hit':>6} ship")
    print(f"{'BASELINE (v3.1 final)':<38} ${base['PnL']:>6.0f} {'':>6} {base['Sharpe']:>7.2f} {base['worst']:>6.3f} ${base['maxDD']:>6.0f} {base['trades']:>6} {base['hit']:.1%}")
    print("-" * 110)
    for label, ind, lb, alpha, jr, s, ships in results:
        dp = s["PnL"] - base["PnL"]
        marker = "*" if ships else " "
        print(f"{marker}{label:<37} ${s['PnL']:>6.0f} ${dp:>+5.0f} {s['Sharpe']:>7.2f} {s['worst']:>6.3f} ${s['maxDD']:>6.0f} {s['trades']:>6} {s['hit']:.1%}")

    out_json = TUNE_OUT / "v3-6-sweep-summary-2026-05-28.json"
    out_json.write_text(json.dumps({
        "baseline": base,
        "cells": [
            {"label": r[0], "indicator": r[1], "lookback_min": r[2],
             "alpha_tilt": r[3], "jr_trust": r[4], "metrics": r[5], "ships": r[6]}
            for r in results
        ],
    }, indent=2))
    print(f"\nfull results → {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
