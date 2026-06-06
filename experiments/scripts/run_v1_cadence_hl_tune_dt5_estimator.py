#!/usr/bin/env python3
"""v1 dt=5 HL tune — round 2: the cadence-SENSITIVE frozen params.

Round 1 (run_v1_cadence_hl_tune_dt5.py) swept the safety gates and capped at
$129.76 vs the $212 dt=60 baseline, but held three params constant that are the
most likely to interact with sub-minute sampling:
  - vol_estimator        (5s close-to-close is bounce-contaminated; Parkinson
                          range-based may behave very differently)
  - exit_safety_d_5m     (a FAST sigma exit is designed for sub-minute flips;
                          was OFF in round 1)
  - vol_ewma_lambda      (0.85 = ~35s half-life at dt=5, maybe too jumpy)

This sweeps those × min_safety_d ∈ {1.0 prod, 3.0 round-1 winner}, holding
exit_safety_d=1.0 and vol_lookback_seconds=3600 (round-1 showed esd=2.0 churns
and vlb is inert). dt=5 throughout. Goal: is the dt=5 "no" airtight, or can a
cadence-appropriate estimator/fast-exit beat $212?
"""
from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data")
OUT_ROOT = WORKTREE / "data" / "sim" / "runs" / "v1-cadence-hl-tune-dt5-est-2026-05-30"

BASE = {
    "tte_min_seconds": 0,
    "tte_max_seconds": 7200,
    "price_extreme_threshold": 0.85,
    "price_extreme_max": 0.99,
    "distance_from_strike_usd_min": 0,
    "vol_max": 100,
    "stop_loss_pct": None,
    "max_position_usd": 300,
    "vol_lookback_seconds": 3600,
    "exit_safety_d": 1.0,
    "exit_vol_lookback_5m_seconds": 300,
    "size_cap_near_strike_pct": 1.0,
    "size_cap_max_dist_pct": 1.5,
    "size_cap_min_ask": 0.88,
    "use_bid_for_entry_gate": True,
    "min_bid_notional_usd": 25.0,
    "vol_sampling_dt_seconds": 5,
}

MIN_SAFETY_D = [1.0, 3.0]
VOL_ESTIMATOR = ["stdev", "parkinson"]
EXIT_5M = [0.0, 1.0]
EWMA_LAMBDA = [0.0, 0.85, 0.97]


def _parse_report(path: Path) -> dict:
    text = path.read_text()

    def find(pat: str, cast=float, default=0):
        m = re.search(pat, text)
        return cast(m.group(1).replace(",", "")) if m else default

    return {
        "n_trades": find(r"trades:\s+(\d+)", int),
        "total_pnl_usd": find(r"total PnL:\s+\$([-\d.,]+)"),
        "sharpe": find(r"Sharpe[^:]*:\s+([-\d.]+)"),
        "hit_rate": find(r"hit rate:\s+([\d.]+)%") / 100.0,
        "max_drawdown_usd": find(r"max drawdown:\s+\$([-\d.,]+)"),
    }


def run_cell(msd: float, est: str, e5m: float, lam: float) -> dict:
    label = f"msd{msd}_{est}_e5m{e5m}_lam{lam}"
    params = {**BASE, "min_safety_d": msd, "vol_estimator": est,
              "exit_safety_d_5m": e5m, "vol_ewma_lambda": lam}
    out_dir = OUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(params, indent=2))

    report = out_dir / "report.md"
    if report.exists() and report.stat().st_size > 100:
        print(f"[skip] {label} (cached)")
        return {"label": label, **_parse_report(report)}

    cmd = [
        "uv", "run", "hl-bt", "run",
        "--strategy", "v1_late_resolution",
        "--data-source", "hl_hip4",
        "--config", str(out_dir / "config.json"),
        "--out-dir", str(out_dir),
        "--start", "2026-05-06", "--end", "2026-05-28",
        "--kind", "both", "--fee-taker", "0.00035", "--slippage-bps", "5.0",
    ]
    env = {**os.environ, "HLBT_HL_DATA_ROOT": str(DATA_ROOT)}
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=WORKTREE, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(f"[FAIL {label}] rc={proc.returncode}\n{proc.stderr[-2000:]}")
        raise SystemExit(1)
    r = _parse_report(report)
    print(f"[ok] {label} ({time.time()-t0:.0f}s) : PnL=${r['total_pnl_usd']:.2f} "
          f"trades={r['n_trades']} hit={r['hit_rate']:.1%} Sharpe={r['sharpe']:.2f} "
          f"maxDD=${r['max_drawdown_usd']:.2f}")
    return {"label": label, **r}


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for msd, est, e5m, lam in itertools.product(MIN_SAFETY_D, VOL_ESTIMATOR, EXIT_5M, EWMA_LAMBDA):
        results.append({"min_safety_d": msd, "vol_estimator": est,
                        "exit_safety_d_5m": e5m, "vol_ewma_lambda": lam,
                        **run_cell(msd, est, e5m, lam)})

    results_sorted = sorted(results, key=lambda r: r["total_pnl_usd"], reverse=True)
    print(f"\n{'label':>34} {'PnL':>9} {'trd':>4} {'hit':>6} {'Sharpe':>7} {'maxDD':>9}")
    for r in results_sorted:
        print(f"{r['label']:>34} ${r['total_pnl_usd']:>7.2f} {r['n_trades']:>4} "
              f"{r['hit_rate']:>5.1%} {r['sharpe']:>7.2f} ${r['max_drawdown_usd']:>7.2f}")
    print("\nbaseline dt=60: $212.01 / 62 / 57.9% / Sharpe 6.42 / DD $75.46")
    print("round-1 dt=5 best: $129.76 (msd3.0/esd1.0) / 66 / 47.4% / 4.72 / DD $38.18")

    (OUT_ROOT / "summary.json").write_text(json.dumps(results_sorted, indent=2))
    print(f"\nfull results → {OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
