#!/usr/bin/env python3
"""Validate the PM-derived minimal v3.1 config on HL HIP-4 corpus.

Single-run comparison: prod baseline vs each candidate over the full HL
binary+bucket corpus, kind by kind. Uses `hl-bt run` (no walk-forward —
corpus is too small).
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis")
WORKTREE = REPO_ROOT / ".worktrees/do-an-analysis-of-the-final-strategies-and-see-if--53OYTm"
DATA_ROOT = REPO_ROOT / "data"
OUT_ROOT = REPO_ROOT / "data/sim/runs/v3-1-hl-validate-2026-05-23"

PROD = {
    "vol_lookback_seconds": 3600,
    "vol_sampling_dt_seconds": 60,
    "vol_clip_min": 0.05,
    "vol_clip_max": 5.0,
    "edge_buffer": 0.02,
    "fee_taker": 0.0,
    "half_spread_assumption": 0.005,
    "drift_lookback_seconds": 3600,
    "drift_blend": 0.0,
    "favorite_threshold": 0.85,
    "tte_min_seconds": 0,
    "tte_max_seconds": 43200,           # HL prod = 12h
    "edge_max": 0.20,
    "min_distance_pct": None,           # already null on HL
    "min_bid_notional_usd": 0.0,        # already 0 on HL
    "max_position_usd": 200.0,
    "stop_loss_pct": None,
    "exit_edge_threshold": 0.0,
    "take_profit_price": None,
    "time_stop_seconds": 0,
    "gamma_lambda": 0.0,
    "exit_take_profit_mode": True,
    "exit_fee": 0.0007,
    "exit_safety_d": 1.0,
    # topup defaults True in builder
}

# Final-PM minimal — apply to HL with d=1.0 (HL already uses d=1.0)
MIN_PM_APPLIED_TO_HL = {
    **PROD,
    "favorite_threshold": 0.9,
    "edge_buffer": 0.03,
    "edge_max": None,        # kill
    "vol_clip_min": 0.0,     # kill
    # min_distance_pct is already None on HL; topup stays True (HL needs it for partial fills)
}

# Same as above but also kill topup (sanity — would it hurt HL?)
MIN_PM_NO_TOPUP = {
    **MIN_PM_APPLIED_TO_HL,
    "topup_enabled": False,
}

# Conservative: only kill the dead-on-PM stuff that's safe on HL (edge_max,
# vol_clip_min), but keep fav=0.85 in case PM-tuned fav=0.9 over-restricts HL
CONSERVATIVE = {
    **PROD,
    "edge_max": None,
    "vol_clip_min": 0.0,
    "edge_buffer": 0.03,     # the round-1 winner
}

CONFIGS = {
    "prod": PROD,
    "min_pm_applied": MIN_PM_APPLIED_TO_HL,
    "min_pm_no_topup": MIN_PM_NO_TOPUP,
    "conservative": CONSERVATIVE,
}


def run_config(name: str, params: dict, kind: str) -> dict:
    cfg_path = OUT_ROOT / f"cfg_{name}_{kind}.json"
    out_dir = OUT_ROOT / f"{name}_{kind}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(params, indent=2))

    report_path = out_dir / "report.md"
    if report_path.exists() and report_path.stat().st_size > 100:
        return parse_report(report_path)

    cmd = [
        "uv", "run", "hl-bt", "run",
        "--strategy", "v3_theta_harvester",
        "--data-source", "hl_hip4",
        "--config", str(cfg_path),
        "--out-dir", str(out_dir),
        "--start", "2026-05-01", "--end", "2026-05-25",
        "--kind", kind,
    ]
    env = {**os.environ, "HLBT_HL_DATA_ROOT": str(DATA_ROOT)}
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=WORKTREE, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"[FAIL {name}/{kind}] rc={proc.returncode}")
        print(proc.stderr[-2000:])
        return {}
    r = parse_report(report_path) if report_path.exists() else {}
    print(f"[ok] {name}/{kind} in {dt:.1f}s : PnL=${r.get('total_pnl_usd',0):.2f} trades={r.get('n_trades',0)}")
    return r


def parse_report(path: Path) -> dict:
    import re
    text = path.read_text()
    def find(pat, cast=float, default=0):
        m = re.search(pat, text)
        if not m: return default
        return cast(m.group(1).replace(',', ''))
    return {
        'n_markets': find(r'questions:\s+(\d+)', int),
        'n_trades': find(r'trades:\s+(\d+)', int),
        'total_pnl_usd': find(r'total PnL:\s+\$([-\d.,]+)'),
        'sharpe': find(r'Sharpe[^:]*:\s+([-\d.]+)'),
        'hit_rate': find(r'hit rate:\s+([\d.]+)%') / 100.0,
        'max_drawdown_usd': find(r'max drawdown:\s+\$([-\d.,]+)'),
    }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    for kind in ("binary", "bucket"):
        for name, params in CONFIGS.items():
            results[(name, kind)] = run_config(name, params, kind)

    print()
    print(f"{'config':>20} {'kind':>7} {'PnL':>10} {'trades':>8} {'hit':>7} {'maxDD':>10}")
    for (name, kind), r in results.items():
        pnl = r.get('total_pnl_usd', 0)
        n = r.get('n_trades', 0)
        h = r.get('hit_rate', 0)
        dd = r.get('max_drawdown_usd', 0)
        print(f"{name:>20} {kind:>7} ${pnl:>8.2f} {n:>8} {h:>6.1%} ${dd:>8.2f}")


if __name__ == "__main__":
    main()
