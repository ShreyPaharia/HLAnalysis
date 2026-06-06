#!/usr/bin/env python3
"""dt=60 CONTROL for the round-2 dt=5 estimator sweep.

Round 2 found dt=5 cells that beat the dt=60 prod baseline ($212) once paired
with Parkinson σ and/or λ=0 — top cell msd3.0/parkinson/λ=0.97 = $269 / DD $33.
That could mean (a) cadence helps, or (b) the estimator/λ change helps at ANY
cadence. This runs the SAME winning configs at dt=60 to disambiguate:

  - if dt=60 also jumps to ~$270 → estimator effect, not cadence → keep dt=60.
  - if dt=60 stays ~$212 while dt=5 hits $269 → cadence adds value → revisit
    the lockstep-vs-decouple recommendation.

Mirrors the round-2 BASE exactly; only vol_sampling_dt_seconds = 60.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data")
OUT_ROOT = WORKTREE / "data" / "sim" / "runs" / "v1-cadence-hl-control-dt60-2026-05-30"

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
    "exit_safety_d_5m": 0.0,
    "exit_vol_lookback_5m_seconds": 300,
    "size_cap_near_strike_pct": 1.0,
    "size_cap_max_dist_pct": 1.5,
    "size_cap_min_ask": 0.88,
    "use_bid_for_entry_gate": True,
    "min_bid_notional_usd": 25.0,
    "vol_sampling_dt_seconds": 60,
}

# (min_safety_d, vol_estimator, vol_ewma_lambda) — the round-2 winners + the
# prod anchor (msd1.0/stdev/0.85 = the known $212 baseline) for sanity.
CELLS = [
    (3.0, "parkinson", 0.97),
    (1.0, "parkinson", 0.97),
    (3.0, "parkinson", 0.85),
    (1.0, "parkinson", 0.85),
    (1.0, "stdev", 0.0),
    (3.0, "stdev", 0.97),
    (1.0, "stdev", 0.85),  # == prod baseline, must reproduce $212.01
]


def _parse(path: Path) -> dict:
    t = path.read_text()

    def f(pat, cast=float, d=0):
        m = re.search(pat, t)
        return cast(m.group(1).replace(",", "")) if m else d

    return {"n_trades": f(r"trades:\s+(\d+)", int),
            "total_pnl_usd": f(r"total PnL:\s+\$([-\d.,]+)"),
            "sharpe": f(r"Sharpe[^:]*:\s+([-\d.]+)"),
            "hit_rate": f(r"hit rate:\s+([\d.]+)%") / 100.0,
            "max_drawdown_usd": f(r"max drawdown:\s+\$([-\d.,]+)")}


def run_cell(msd, est, lam):
    label = f"msd{msd}_{est}_lam{lam}"
    params = {**BASE, "min_safety_d": msd, "vol_estimator": est, "vol_ewma_lambda": lam}
    out = OUT_ROOT / label
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(params, indent=2))
    rep = out / "report.md"
    if rep.exists() and rep.stat().st_size > 100:
        print(f"[skip] {label}")
        return {"label": label, **_parse(rep)}
    cmd = ["uv", "run", "hl-bt", "run", "--strategy", "v1_late_resolution",
           "--data-source", "hl_hip4", "--config", str(out / "config.json"),
           "--out-dir", str(out), "--start", "2026-05-06", "--end", "2026-05-28",
           "--kind", "both", "--fee-taker", "0.00035", "--slippage-bps", "5.0"]
    env = {**os.environ, "HLBT_HL_DATA_ROOT": str(DATA_ROOT)}
    t0 = time.time()
    p = subprocess.run(cmd, cwd=WORKTREE, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        print(f"[FAIL {label}]\n{p.stderr[-2000:]}"); raise SystemExit(1)
    r = _parse(rep)
    print(f"[ok] {label} ({time.time()-t0:.0f}s) : PnL=${r['total_pnl_usd']:.2f} "
          f"trades={r['n_trades']} hit={r['hit_rate']:.1%} Sharpe={r['sharpe']:.2f} "
          f"maxDD=${r['max_drawdown_usd']:.2f}")
    return {"label": label, **r}


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    res = [{"min_safety_d": m, "vol_estimator": e, "vol_ewma_lambda": l, **run_cell(m, e, l)}
           for (m, e, l) in CELLS]
    res.sort(key=lambda r: r["total_pnl_usd"], reverse=True)
    print(f"\n{'label':>26} {'PnL':>9} {'trd':>4} {'hit':>6} {'Sharpe':>7} {'maxDD':>9}")
    for r in res:
        print(f"{r['label']:>26} ${r['total_pnl_usd']:>7.2f} {r['n_trades']:>4} "
              f"{r['hit_rate']:>5.1%} {r['sharpe']:>7.2f} ${r['max_drawdown_usd']:>7.2f}")
    print("\ndt=5 winner (round-2): msd3.0/parkinson/0.97 = $269.52 / 60.5% / Sharpe ? / DD $33.05")
    (OUT_ROOT / "summary.json").write_text(json.dumps(res, indent=2))
    print(f"\n→ {OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
