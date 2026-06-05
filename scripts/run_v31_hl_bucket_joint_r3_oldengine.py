#!/usr/bin/env python3
"""Round-3 joint tune for v31/theta HL HIP-4 BUCKET markets — OLD ENGINE.

Runs against the pre-regression engine (main @ 3647c31) in a side worktree,
because the merged sim-speed refactor (459871d) regressed HL backtests to 0
trades (sigma inflated / edge always negative; root-caused to _fastpath_core,
not the cache). Old engine reproduces round-1/2 numbers exactly, so the tune
answer is valid; re-validate on the fixed engine once that lands.

Design (same as run_v31_hl_bucket_joint_r3.py):
  GRID  : favorite_threshold x tte_max x vol_lookback  (3x3x3 = 27 cells)
  FIXED : exit_safety_d=0.0, vol_sampling_dt=1, edge_buffer=0.005
  Pick  : best WORST-HALF PnL across early/late time split (overfit-robust).

Old engine has no --workers flag, so parallelism is the round-2 pattern:
6 outer worker threads, each a serial `hl-bt run`.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

REPO_ROOT = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis")
WORKTREE = REPO_ROOT / ".worktrees/oldengine-3647c31"      # <-- old, correct engine
DATA_ROOT = REPO_ROOT / "data"
OUT_ROOT = REPO_ROOT / "data/sim/runs/v31-hl-bucket-joint-r3-oldengine-2026-06-05"
OUTER_WORKERS = 6

WINDOWS = {
    "full":  ("2026-05-06", "2026-06-04"),
    "early": ("2026-05-06", "2026-05-20"),
    "late":  ("2026-05-20", "2026-06-04"),
}

PROD = {
    "vol_lookback_seconds": 3600, "vol_sampling_dt_seconds": 5, "vol_clip_min": 0.0,
    "vol_clip_max": 5.0, "edge_buffer": 0.02, "fee_taker": 0.0,
    "half_spread_assumption": 0.005, "drift_lookback_seconds": 3600, "drift_blend": 0.0,
    "favorite_threshold": 0.85, "tte_min_seconds": 0, "tte_max_seconds": 43200,
    "edge_max": None, "min_distance_pct": None, "min_bid_notional_usd": 0.0,
    "max_position_usd": 500.0, "stop_loss_pct": None, "exit_edge_threshold": 0.0,
    "time_stop_seconds": 0, "exit_take_profit_mode": True, "exit_fee": 0.0007,
    "exit_safety_d": 1.0,
}
FIXED = {"exit_safety_d": 0.0, "vol_sampling_dt_seconds": 1, "edge_buffer": 0.005}

FAV = [0.75, 0.80, 0.85]
TTE = [21600, 28800, 36000]          # 6h / 8h / 10h
VLB = [1800, 2700, 3600]
CENTER = (0.80, 28800, 2700)


def make_configs() -> dict[str, dict]:
    cfgs: dict[str, dict] = {}
    for fav, tte, vlb in product(FAV, TTE, VLB):
        label = f"fav{fav}_tte{tte//3600}h_vlb{vlb}"
        cfgs[label] = {**PROD, **FIXED, "favorite_threshold": fav,
                       "tte_max_seconds": tte, "vol_lookback_seconds": vlb}
    cfgs["prod_ref"] = dict(PROD)
    cfgs["fixedknobs_only"] = {**PROD, **FIXED}
    cf, ct, cv = CENTER
    center_base = {**PROD, **FIXED, "favorite_threshold": cf,
                   "tte_max_seconds": ct, "vol_lookback_seconds": cv}
    cfgs["center_dt2"] = {**center_base, "vol_sampling_dt_seconds": 2}
    cfgs["center_dt5"] = {**center_base, "vol_sampling_dt_seconds": 5}
    return cfgs


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
    label, window, params = job["label"], job["window"], job["params"]
    name = f"{label}__{window}"
    cfg_path = OUT_ROOT / f"cfg_{name}.json"
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(params, indent=2))
    rpt = out_dir / "report.md"
    if not (rpt.exists() and rpt.stat().st_size > 100):
        start, end = WINDOWS[window]
        cmd = ["uv", "run", "hl-bt", "run", "--strategy", "v3_theta_harvester",
               "--data-source", "hl_hip4", "--config", str(cfg_path),
               "--out-dir", str(out_dir), "--start", start, "--end", end,
               "--kind", "bucket"]                      # NO --workers on old engine
        env = {**os.environ, "HLBT_HL_DATA_ROOT": str(DATA_ROOT)}
        t0 = time.time()
        p = subprocess.run(cmd, cwd=WORKTREE, capture_output=True, text=True, env=env)
        if p.returncode != 0:
            print(f"[FAIL {name}] {p.stderr.strip()[-300:]}")
            return {"label": label, "window": window}
        print(f"[ok {name:42}] {time.time()-t0:5.0f}s")
    return {"label": label, "window": window, **(parse_report(rpt) if rpt.exists() else {})}


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cfgs = make_configs()
    jobs = [{"label": l, "window": w, "params": p}
            for l, p in cfgs.items() for w in WINDOWS]
    print(f"OLD ENGINE ({WORKTREE.name}) — {len(cfgs)} configs x {len(WINDOWS)} = {len(jobs)} runs, outer={OUTER_WORKERS}")
    results = []
    with ThreadPoolExecutor(max_workers=OUTER_WORKERS) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for f in as_completed(futs):
            results.append(f.result())
    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2))

    idx = {(r["label"], r["window"]): r for r in results}
    def pnl(l, w): return idx.get((l, w), {}).get("total_pnl_usd", float("nan"))

    rows = []
    for label in cfgs:
        rf = idx.get((label, "full"), {})
        pe, pl = pnl(label, "early"), pnl(label, "late")
        rows.append({
            "label": label, "full": rf.get("total_pnl_usd", float("nan")),
            "sharpe": rf.get("sharpe", 0), "dd": rf.get("max_drawdown_usd", 0),
            "trades": rf.get("n_trades", 0), "hit": rf.get("hit_rate", 0),
            "early": pe, "late": pl, "worst_half": min(pe, pl),
        })
    refs = ("prod_ref", "fixedknobs_only", "center_dt2", "center_dt5")
    grid_rows = [r for r in rows if r["label"] not in refs]
    grid_rows.sort(key=lambda r: r["worst_half"], reverse=True)

    pr = idx.get(("prod_ref", "full"), {})
    print("\n" + "=" * 100)
    print(f"PROD_REF: full=${pr.get('total_pnl_usd',0):.2f} Sharpe={pr.get('sharpe',0):.2f} "
          f"DD=${pr.get('max_drawdown_usd',0):.2f} | early=${pnl('prod_ref','early'):.2f} "
          f"late=${pnl('prod_ref','late'):.2f}   (sanity: full should be ~$412)")
    print("-" * 100)
    print(f"{'cell (by worst-half)':>26} {'full':>8} {'Sharpe':>7} {'maxDD':>8} "
          f"{'early':>8} {'late':>8} {'worst':>8} {'trades':>6} {'hit':>5}")
    for r in grid_rows:
        print(f"{r['label']:>26} ${r['full']:>6.2f} {r['sharpe']:>7.2f} ${r['dd']:>6.2f} "
              f"${r['early']:>6.2f} ${r['late']:>6.2f} ${r['worst_half']:>6.2f} "
              f"{r['trades']:>6} {r['hit']:>4.0%}")
    print("-" * 100)
    for label in ("fixedknobs_only", "center_dt2", "center_dt5"):
        r = next(x for x in rows if x["label"] == label)
        print(f"{label:>26} ${r['full']:>6.2f} {r['sharpe']:>7.2f} ${r['dd']:>6.2f} "
              f"${r['early']:>6.2f} ${r['late']:>6.2f} ${r['worst_half']:>6.2f} "
              f"{r['trades']:>6} {r['hit']:>4.0%}")
    print("=" * 100)
    best = grid_rows[0]
    print(f"\nROBUST PICK (best worst-half): {best['label']}  worst-half=${best['worst_half']:.2f}  "
          f"full=${best['full']:.2f}  Sharpe={best['sharpe']:.2f}  DD=${best['dd']:.2f}")
    print(f"results: {OUT_ROOT}")


if __name__ == "__main__":
    main()
