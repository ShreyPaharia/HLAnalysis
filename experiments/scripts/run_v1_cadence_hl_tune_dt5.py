#!/usr/bin/env python3
"""v1 (late_resolution) dt=5 HL param tune — can re-tuned params recover the
gap vs dt=60?

Runs on the CADENCE-AWARE strategy code (vol_sampling_dt_seconds threaded into
the sigma/window math). Baseline dt=60 = $212.01 / Sharpe 6.42 / DD $75. The
same prod params at dt=5 give $87.62 / Sharpe 2.68 with MORE trades (84) and
LOWER hit (47.4%) — the denser feed trips the entry gate on noisier signals.

This sweeps the three levers that govern entry selectivity + the mid-hold cut
at dt=5, holding everything else at the v1 HL prod config:
  min_safety_d        ∈ {1.0, 1.5, 2.0, 3.0}   (entry gate tightness)
  exit_safety_d       ∈ {1.0, 2.0}             (mid-hold cut tightness)
  vol_lookback_seconds∈ {3600, 7200}           (sigma window length)
= 16 cells. Goal: find any dt=5 cell that matches/beats dt=60 ($212) at
comparable drawdown. If none does, v1 stays at 60s (keep-and-decouple).
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
OUT_ROOT = WORKTREE / "data" / "sim" / "runs" / "v1-cadence-hl-tune-dt5-2026-05-30"

BASE = {
    "tte_min_seconds": 0,
    "tte_max_seconds": 7200,
    "price_extreme_threshold": 0.85,
    "price_extreme_max": 0.99,
    "distance_from_strike_usd_min": 0,
    "vol_max": 100,
    "stop_loss_pct": None,
    "max_position_usd": 300,
    "min_safety_d": 1.0,
    "vol_lookback_seconds": 3600,
    "exit_safety_d": 1.0,
    "vol_ewma_lambda": 0.85,
    "size_cap_near_strike_pct": 1.0,
    "size_cap_max_dist_pct": 1.5,
    "size_cap_min_ask": 0.88,
    "use_bid_for_entry_gate": True,
    "min_bid_notional_usd": 25.0,
    "vol_sampling_dt_seconds": 5,
}

MIN_SAFETY_D = [1.0, 1.5, 2.0, 3.0]
EXIT_SAFETY_D = [1.0, 2.0]
VOL_LOOKBACK = [3600, 7200]


def _parse_report(path: Path) -> dict:
    text = path.read_text()

    def find(pat: str, cast=float, default=0):
        m = re.search(pat, text)
        return cast(m.group(1).replace(",", "")) if m else default

    return {
        "n_markets": find(r"questions:\s+(\d+)", int),
        "n_trades": find(r"trades:\s+(\d+)", int),
        "total_pnl_usd": find(r"total PnL:\s+\$([-\d.,]+)"),
        "sharpe": find(r"Sharpe[^:]*:\s+([-\d.]+)"),
        "hit_rate": find(r"hit rate:\s+([\d.]+)%") / 100.0,
        "max_drawdown_usd": find(r"max drawdown:\s+\$([-\d.,]+)"),
    }


def run_cell(msd: float, esd: float, vlb: int) -> dict:
    label = f"msd{msd}_esd{esd}_vlb{vlb}"
    params = {**BASE, "min_safety_d": msd, "exit_safety_d": esd, "vol_lookback_seconds": vlb}
    out_dir = OUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps(params, indent=2))

    report = out_dir / "report.md"
    if report.exists() and report.stat().st_size > 100:
        print(f"[skip] {label} (cached)")
        return {"label": label, **_parse_report(report)}

    cmd = [
        "uv", "run", "hl-bt", "run",
        "--strategy", "v1_late_resolution",
        "--data-source", "hl_hip4",
        "--config", str(cfg_path),
        "--out-dir", str(out_dir),
        "--start", "2026-05-06", "--end", "2026-05-28",
        "--kind", "both",
        "--fee-taker", "0.00035",
        "--slippage-bps", "5.0",
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
    for msd, esd, vlb in itertools.product(MIN_SAFETY_D, EXIT_SAFETY_D, VOL_LOOKBACK):
        results.append({"min_safety_d": msd, "exit_safety_d": esd,
                        "vol_lookback_seconds": vlb, **run_cell(msd, esd, vlb)})

    results_sorted = sorted(results, key=lambda r: r["total_pnl_usd"], reverse=True)
    print(f"\n{'label':>22} {'PnL':>9} {'trades':>7} {'hit':>7} {'Sharpe':>7} {'maxDD':>9}")
    for r in results_sorted:
        print(f"{r['label']:>22} ${r['total_pnl_usd']:>7.2f} {r['n_trades']:>7} "
              f"{r['hit_rate']:>6.1%} {r['sharpe']:>7.2f} ${r['max_drawdown_usd']:>7.2f}")
    print("\nBaseline dt=60 (prod params): $212.01 / 62 / 57.9% / Sharpe 6.42 / DD $75.46")

    (OUT_ROOT / "summary.json").write_text(json.dumps(results_sorted, indent=2))
    print(f"\nfull results → {OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
