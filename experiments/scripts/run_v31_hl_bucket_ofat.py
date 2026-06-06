#!/usr/bin/env python3
"""OFAT (one-factor-at-a-time) sensitivity screen for v31/theta on HL HIP-4
priceBucket markets.

We now have ~27 bucket questions (2026-05-06 → 2026-06-04). Buckets currently
inherit binary-derived theta params; this screen sweeps each live knob one step
lower / one step higher around current prod, holding everything else at prod,
on the bucket-only corpus. The point is to find which params are *sensitive* on
buckets (where the optimum diverges from binary) before committing to a deeper
sweep — NOT to fit on n=27.

Mirrors scripts/run_v31_hl_validate.py invocation:
    uv run hl-bt run --strategy v3_theta_harvester --data-source hl_hip4 \
        --config <json> --kind bucket --start ... --end ...
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis")
WORKTREE = REPO_ROOT / ".worktrees/now-we-have-enough-data-for-hl-binary-and-buckets--0eWFmE"
DATA_ROOT = REPO_ROOT / "data"
OUT_ROOT = REPO_ROOT / "data/sim/runs/v31-hl-bucket-ofat-2026-06-04"
START, END = "2026-05-06", "2026-06-04"

# Current v31/theta prod (theta: block + priceBucket allowlist) — config.strategy.yaml
PROD = {
    "vol_lookback_seconds": 3600,
    "vol_sampling_dt_seconds": 5,
    "vol_clip_min": 0.0,
    "vol_clip_max": 5.0,
    "edge_buffer": 0.02,
    "fee_taker": 0.0,
    "half_spread_assumption": 0.005,
    "drift_lookback_seconds": 3600,
    "drift_blend": 0.0,
    "favorite_threshold": 0.85,
    "tte_min_seconds": 0,
    "tte_max_seconds": 43200,           # 12h
    "edge_max": None,
    "min_distance_pct": None,
    "min_bid_notional_usd": 0.0,
    "max_position_usd": 500.0,
    "stop_loss_pct": None,
    "exit_edge_threshold": 0.0,
    "time_stop_seconds": 0,
    "exit_take_profit_mode": True,
    "exit_fee": 0.0007,
    "exit_safety_d": 1.0,
}

# OFAT axes: param -> [lower, higher] (prod is the baseline run)
SWEEP = {
    "favorite_threshold":      [0.80, 0.90],
    "edge_buffer":             [0.01, 0.03],
    "exit_safety_d":           [0.5, 1.5],
    "tte_max_seconds":         [21600, 86400],     # 6h / 24h
    "vol_lookback_seconds":    [1800, 7200],
    "vol_sampling_dt_seconds": [1, 30],
    "vol_clip_max":            [3.0, 10.0],
    "half_spread_assumption":  [0.0025, 0.0075],
    "drift_blend":             [0.1],              # one-sided (>=0)
}


def run_config(name: str, params: dict) -> dict:
    cfg_path = OUT_ROOT / f"cfg_{name}.json"
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
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
        "--start", START, "--end", END,
        "--kind", "bucket",
    ]
    env = {**os.environ, "HLBT_HL_DATA_ROOT": str(DATA_ROOT)}
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=WORKTREE, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"[FAIL {name}] rc={proc.returncode}\n{proc.stderr[-1500:]}")
        return {}
    r = parse_report(report_path) if report_path.exists() else {}
    print(f"[ok] {name:38} {dt:5.1f}s  PnL=${r.get('total_pnl_usd',0):8.2f}  "
          f"DD=${r.get('max_drawdown_usd',0):7.2f}  n={r.get('n_trades',0)}")
    return r


def parse_report(path: Path) -> dict:
    text = path.read_text()
    def find(pat, cast=float, default=0):
        m = re.search(pat, text)
        return cast(m.group(1).replace(',', '')) if m else default
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
    rows = []  # (axis, label, result)

    base = run_config("baseline_prod", PROD)
    rows.append(("baseline", "prod", base))

    for axis, vals in SWEEP.items():
        for v in vals:
            params = {**PROD, axis: v}
            label = f"{axis}={v}"
            r = run_config(f"{axis}__{v}", params)
            rows.append((axis, label, r))

    b_pnl = base.get('total_pnl_usd', 0.0)
    print("\n" + "=" * 92)
    print(f"{'config':>34} {'PnL':>9} {'dPnL':>8} {'Sharpe':>7} {'maxDD':>8} {'trades':>7} {'hit':>6}")
    print("-" * 92)
    for axis, label, r in rows:
        pnl = r.get('total_pnl_usd', 0.0)
        d = pnl - b_pnl
        print(f"{label:>34} ${pnl:>7.2f} {d:>+8.2f} {r.get('sharpe',0):>7.2f} "
              f"${r.get('max_drawdown_usd',0):>6.2f} {r.get('n_trades',0):>7} {r.get('hit_rate',0):>5.1%}")
    print("=" * 92)
    (OUT_ROOT / "summary.json").write_text(json.dumps(
        [{"axis": a, "label": l, **r} for a, l, r in rows], indent=2))
    print(f"\nbaseline prod PnL = ${b_pnl:.2f}  (n_questions={base.get('n_markets',0)})")
    print(f"results: {OUT_ROOT}")


if __name__ == "__main__":
    main()
