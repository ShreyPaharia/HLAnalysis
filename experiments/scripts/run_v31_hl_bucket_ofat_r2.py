#!/usr/bin/env python3
"""Round-2 deeper OFAT for v31/theta on HL HIP-4 — BOTH kinds + time-split.

Round 1 (scripts/run_v31_hl_bucket_ofat.py) screened buckets and found the
binary-inherited prod values are mistuned for buckets in a consistent direction
(looser favorite, shorter TTE, faster sampling, lower exit_safety_d). Round 2:

  * Deepen only the live axes (drop inert vol_clip_max / half_spread / drift_blend).
  * Run on BINARY too — most of these knobs live in the shared theta: block, so
    we must see binary-vs-bucket divergence before committing per-class plumbing.
  * Add an early/late time split (n is ~26/kind) — a config we trust must beat
    its OWN-kind baseline in BOTH halves, not just full-sample.

Parallelized (ThreadPoolExecutor over subprocess `hl-bt run`).
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis")
WORKTREE = REPO_ROOT / ".worktrees/now-we-have-enough-data-for-hl-binary-and-buckets--0eWFmE"
DATA_ROOT = REPO_ROOT / "data"
OUT_ROOT = REPO_ROOT / "data/sim/runs/v31-hl-bucket-ofat-r2-2026-06-04"
WORKERS = 6

# (start, end) per window. Corpus 2026-05-06 → 2026-06-04; split at 05-20.
WINDOWS = {
    "full":  ("2026-05-06", "2026-06-04"),
    "early": ("2026-05-06", "2026-05-20"),
    "late":  ("2026-05-20", "2026-06-04"),
}

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
    "tte_max_seconds": 43200,
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

# Deepened axes (values equal to prod are skipped — baseline covers them).
SWEEP = {
    "favorite_threshold":      [0.70, 0.75, 0.80, 0.85],
    "tte_max_seconds":         [10800, 14400, 21600, 28800],   # 3h/4h/6h/8h
    "exit_safety_d":           [0.0, 0.25, 0.5, 0.75],
    "vol_lookback_seconds":    [900, 1800, 2700, 3600],
    "edge_buffer":             [0.0, 0.005, 0.01, 0.015],
    "vol_sampling_dt_seconds": [1, 2, 3, 5],                    # lockstep caveat (feed-level)
}


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


def run_one(job: dict) -> dict:
    kind, window, label, params = job["kind"], job["window"], job["label"], job["params"]
    name = f"{kind}__{window}__{label}"
    cfg_path = OUT_ROOT / f"cfg_{name}.json"
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(params, indent=2))
    report_path = out_dir / "report.md"
    if not (report_path.exists() and report_path.stat().st_size > 100):
        start, end = WINDOWS[window]
        cmd = [
            "uv", "run", "hl-bt", "run",
            "--strategy", "v3_theta_harvester",
            "--data-source", "hl_hip4",
            "--config", str(cfg_path),
            "--out-dir", str(out_dir),
            "--start", start, "--end", end,
            "--kind", kind,
        ]
        env = {**os.environ, "HLBT_HL_DATA_ROOT": str(DATA_ROOT)}
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=WORKTREE, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            print(f"[FAIL {name}] rc={proc.returncode} {proc.stderr.strip()[-300:]}")
            return {"kind": kind, "window": window, "label": label}
        print(f"[ok {name:46}] {time.time()-t0:5.0f}s")
    r = parse_report(report_path) if report_path.exists() else {}
    return {"kind": kind, "window": window, "label": label, **r}


def build_jobs() -> list[dict]:
    jobs = []
    for kind in ("bucket", "binary"):
        for window in WINDOWS:
            jobs.append({"kind": kind, "window": window, "label": "baseline", "params": dict(PROD)})
            for axis, vals in SWEEP.items():
                for v in vals:
                    if v == PROD[axis]:
                        continue
                    jobs.append({"kind": kind, "window": window,
                                 "label": f"{axis}={v}", "params": {**PROD, axis: v}})
    return jobs


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    print(f"{len(jobs)} runs, {WORKERS} workers")
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for f in as_completed(futs):
            results.append(f.result())

    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2))

    # index: (kind, window, label) -> result
    idx = {(r["kind"], r["window"], r["label"]): r for r in results}

    def pnl(kind, window, label):
        return idx.get((kind, window, label), {}).get("total_pnl_usd", float("nan"))

    for kind in ("bucket", "binary"):
        base_full = idx.get((kind, "full", "baseline"), {})
        base_e = pnl(kind, "early", "baseline")
        base_l = pnl(kind, "late", "baseline")
        print("\n" + "=" * 104)
        print(f"  {kind.upper()}  baseline: full=${base_full.get('total_pnl_usd',0):.2f} "
              f"Sharpe={base_full.get('sharpe',0):.2f} DD=${base_full.get('max_drawdown_usd',0):.2f} "
              f"| early=${base_e:.2f} late=${base_l:.2f}")
        print(f"  {'config':>32} {'full':>9} {'dFull':>8} {'Sharpe':>7} {'maxDD':>8} "
              f"{'early':>8} {'late':>8} {'ROBUST':>7}")
        print("-" * 104)
        labels = ["baseline"] + [f"{a}={v}" for a, vs in SWEEP.items() for v in vs if v != PROD[a]]
        for label in labels:
            rf = idx.get((kind, "full", label), {})
            pf = rf.get("total_pnl_usd", float("nan"))
            pe, pl = pnl(kind, "early", label), pnl(kind, "late", label)
            robust = (pe > base_e and pl > base_l) if label != "baseline" else False
            star = "  <<<" if robust else ""
            print(f"  {label:>32} ${pf:>7.2f} {pf-base_full.get('total_pnl_usd',0):>+8.2f} "
                  f"{rf.get('sharpe',0):>7.2f} ${rf.get('max_drawdown_usd',0):>6.2f} "
                  f"${pe:>6.2f} ${pl:>6.2f} {str(robust):>7}{star}")
        print("=" * 104)
    print(f"\nresults: {OUT_ROOT}")


if __name__ == "__main__":
    main()
