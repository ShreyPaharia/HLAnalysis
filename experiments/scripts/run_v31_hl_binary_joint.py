#!/usr/bin/env python3
"""Independent walk-forward tune for v31/theta HL HIP-4 BINARY (priceBINARY) markets.

Companion to the bucket tune (run_v31_hl_bucket_*.py). Round-2 OFAT proved binary
and buckets want OPPOSITE values on the load-bearing theta axes, so each market
class gets its own config. This script nails the BINARY optimum, mirroring the
bucket-tune harness exactly (ThreadPool over `hl-bt run`, parse report.md,
walk-forward early/late split, pick by best WORST-HALF PnL on n=25).

Engine: current main @ 74fe66f (sigma-resample regression fixed; event-array
cache now default-ON with _bundle_config_sig safeguards — config_sig keys on
resample dt + feed/book source, coverage-tested, bit-exact round-trip — so it is
safe for this dt-varying tune; NO --rebuild flags). Runs from this worktree,
which reproduces the prod binary baseline to the cent on 74fe66f
($830.24 / 70 trades / Sharpe 12.345 / hit 80.00% / DD $124.57).

Phases (all on full + early + late windows):
  0. BASELINE  : prod config, --kind binary => assert exactly $830.24 / 70 trades
                 / Sharpe 12.345 / hit 80.00% / maxDD $124.57. Abort if mismatch.
  1. OFAT      : re-confirm the round-2 binary free-win directions on current main
                 and extend their boundaries (vol_lookback<900, favorite>0.85,
                 tte>12h) + an exit_safety_d=0.0 contrast (binary != bucket).
  2. JOINT GRID: favorite_threshold x edge_buffer x vol_lookback (3x3x3 = 27).
                 FIXED at the round-2 binary marginal optima:
                   tte_max=43200 (12h; binary prefers LONG, all shorter hurt),
                   vol_sampling_dt=2 (shared feed-level lockstep w/ v1+bucket),
                   exit_safety_d=1.0 (binary wants prod; <1.0 collapses early half).
                 Pick = cell with best min(early, late). DD reported prominently.
                 Plus a dt=5 cross-check of the winning cell to quantify the cost
                 of the dt=2 lockstep on binary (dt=2 helps full but hurts early).

DO NOT change live config / DO NOT deploy. Analysis + recommendation only.
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
WORKTREE = Path(__file__).resolve().parents[1]          # this worktree == current main
DATA_ROOT = REPO_ROOT / "data"                          # shared HL corpus
OUT_ROOT = REPO_ROOT / "data/sim/runs/v31-hl-binary-joint-2026-06-05-v2"
OUTER_WORKERS, RUN_WORKERS = 2, 6

WINDOWS = {
    "full":  ("2026-05-06", "2026-06-04"),
    "early": ("2026-05-06", "2026-05-20"),
    "late":  ("2026-05-20", "2026-06-04"),
}

# v31/theta production config (identical to the bucket-tune PROD block).
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
# expected prod-baseline binary numbers (must reproduce to the cent on current main)
BASELINE_EXPECT = {"n_markets": 25, "n_trades": 70, "total_pnl_usd": 830.24,
                   "sharpe": 12.345, "hit_rate": 0.80, "max_drawdown_usd": 124.57}

# ---- Phase 1: OFAT confirm + boundary extension (each = {**PROD, axis: val}) ----
OFAT = {
    "edge_buffer=0.0":            {"edge_buffer": 0.0},          # round-2 robust +$51
    "vol_lookback_seconds=900":   {"vol_lookback_seconds": 900},  # round-2 robust +$33
    "vol_lookback_seconds=600":   {"vol_lookback_seconds": 600},  # extend below 900 edge
    "vol_sampling_dt_seconds=2":  {"vol_sampling_dt_seconds": 2},  # +$42 full, hurts early
    "favorite_threshold=0.90":    {"favorite_threshold": 0.90},   # extend above 0.85
    "tte_max_seconds=50400":      {"tte_max_seconds": 50400},      # 14h, extend above 12h
    "tte_max_seconds=57600":      {"tte_max_seconds": 57600},      # 16h
    "exit_safety_d=0.0":          {"exit_safety_d": 0.0},          # bucket optimum -> contrast
}

# ---- Phase 2: joint grid (binary divergent axes) ----
FAV = [0.80, 0.85, 0.90]
EB = [0.0, 0.005, 0.01]
VLB = [600, 900, 1800]
# fixed at round-2 binary marginal optima (the binary<->bucket divergence is here)
FIXED = {"tte_max_seconds": 43200, "vol_sampling_dt_seconds": 2, "exit_safety_d": 1.0}
CENTER = (0.85, 0.0, 900)        # OFAT-implied best; dt cross-check runs on this cell


def make_configs() -> dict[str, dict]:
    cfgs: dict[str, dict] = {"prod_ref": dict(PROD)}
    for label, override in OFAT.items():
        cfgs[f"ofat::{label}"] = {**PROD, **override}
    for fav, eb, vlb in product(FAV, EB, VLB):
        label = f"fav{fav}_eb{eb}_vlb{vlb}"
        cfgs[label] = {**PROD, **FIXED, "favorite_threshold": fav,
                       "edge_buffer": eb, "vol_lookback_seconds": vlb}
    # references for the grid block
    cfgs["fixedknobs_only"] = {**PROD, **FIXED}      # prod gates + the 3 fixed knobs
    cf, ce, cv = CENTER
    center_base = {**PROD, **FIXED, "favorite_threshold": cf,
                   "edge_buffer": ce, "vol_lookback_seconds": cv}
    cfgs["center_dt5"] = {**center_base, "vol_sampling_dt_seconds": 5}  # lockstep-cost probe
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
    name = f"{label}__{window}".replace("/", "_").replace("::", "__")
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
               "--kind", "binary", "--workers", str(RUN_WORKERS)]
        env = {**os.environ, "HLBT_HL_DATA_ROOT": str(DATA_ROOT)}
        t0 = time.time()
        p = subprocess.run(cmd, cwd=WORKTREE, capture_output=True, text=True, env=env)
        if p.returncode != 0:
            print(f"[FAIL {name}] {p.stderr.strip()[-300:]}")
            return {"label": label, "window": window}
        print(f"[ok {name:44}] {time.time()-t0:5.0f}s")
    return {"label": label, "window": window, **(parse_report(rpt) if rpt.exists() else {})}


def check_baseline(idx: dict) -> bool:
    r = idx.get(("prod_ref", "full"), {})
    print("\n=== PHASE 0: prod baseline reproduction (binary, full corpus) ===")
    ok = True
    for k, exp in BASELINE_EXPECT.items():
        got = r.get(k, float("nan"))
        match = abs(got - exp) <= (0.01 if isinstance(exp, float) else 0)
        ok = ok and match
        print(f"  {k:18} got={got:>10}  expect={exp:>10}  {'OK' if match else '*** MISMATCH ***'}")
    print(f"  => baseline {'REPRODUCED' if ok else 'MISMATCH — ABORT'}")
    return ok


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cfgs = make_configs()
    jobs = [{"label": l, "window": w, "params": p}
            for l, p in cfgs.items() for w in WINDOWS]
    print(f"{len(cfgs)} configs x {len(WINDOWS)} windows = {len(jobs)} runs "
          f"(outer={OUTER_WORKERS}, run_workers={RUN_WORKERS})")
    results = []
    with ThreadPoolExecutor(max_workers=OUTER_WORKERS) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for f in as_completed(futs):
            results.append(f.result())
    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2))

    idx = {(r["label"], r["window"]): r for r in results}
    def pnl(l, w): return idx.get((l, w), {}).get("total_pnl_usd", float("nan"))

    baseline_ok = check_baseline(idx)
    pr = idx.get(("prod_ref", "full"), {})
    pr_e, pr_l = pnl("prod_ref", "early"), pnl("prod_ref", "late")

    def fmt_row(r):
        return (f"{r['label']:>30} ${r['full']:>7.2f} {r['sharpe']:>7.2f} ${r['dd']:>7.2f} "
                f"${r['early']:>7.2f} ${r['late']:>7.2f} ${r['worst_half']:>7.2f} "
                f"{r['trades']:>6} {r['hit']:>4.0%}")

    def mkrow(label):
        rf = idx.get((label, "full"), {})
        pe, pl = pnl(label, "early"), pnl(label, "late")
        return {"label": label, "full": rf.get("total_pnl_usd", float("nan")),
                "sharpe": rf.get("sharpe", 0), "dd": rf.get("max_drawdown_usd", 0),
                "trades": rf.get("n_trades", 0), "hit": rf.get("hit_rate", 0),
                "early": pe, "late": pl, "worst_half": min(pe, pl)}

    hdr = (f"{'config':>30} {'full':>8} {'Sharpe':>7} {'maxDD':>8} "
           f"{'early':>8} {'late':>8} {'worst':>8} {'trades':>6} {'hit':>5}")

    # ---- Phase 1: OFAT ----
    print("\n" + "=" * 108)
    print(f"PHASE 1 — OFAT (vs prod: full=${pr.get('total_pnl_usd',0):.2f} "
          f"early=${pr_e:.2f} late=${pr_l:.2f} worst=${min(pr_e,pr_l):.2f})")
    print("-" * 108)
    print(hdr)
    print(fmt_row({**mkrow("prod_ref"), "label": "prod_ref"}))
    ofat_rows = [mkrow(f"ofat::{l}") for l in OFAT]
    for r in ofat_rows:
        r["label"] = r["label"].replace("ofat::", "")
        print(fmt_row(r))

    # ---- Phase 2: joint grid ----
    grid_labels = [f"fav{fav}_eb{eb}_vlb{vlb}" for fav, eb, vlb in product(FAV, EB, VLB)]
    grid_rows = sorted((mkrow(l) for l in grid_labels),
                       key=lambda r: r["worst_half"], reverse=True)
    print("\n" + "=" * 108)
    print("PHASE 2 — JOINT GRID  fav x edge_buffer x vol_lookback "
          "(FIXED tte=12h, dt=2, exit_safety_d=1.0)  sorted by WORST-HALF")
    print("-" * 108)
    print(hdr)
    for r in grid_rows:
        print(fmt_row(r))
    print("-" * 108)
    for label in ("fixedknobs_only", "center_dt5", "prod_ref"):
        print(fmt_row({**mkrow(label), "label": label}))
    print("=" * 108)

    best = grid_rows[0]
    print(f"\nBASELINE {'OK' if baseline_ok else 'MISMATCH!!'}")
    print(f"ROBUST PICK (best worst-half): {best['label']}")
    print(f"  full=${best['full']:.2f}  early=${best['early']:.2f}  late=${best['late']:.2f}  "
          f"worst-half=${best['worst_half']:.2f}  Sharpe={best['sharpe']:.2f}  maxDD=${best['dd']:.2f}  "
          f"trades={best['trades']}  hit={best['hit']:.0%}")
    print(f"  vs prod: full ${best['full']-pr.get('total_pnl_usd',0):+.2f}  "
          f"worst-half ${best['worst_half']-min(pr_e,pr_l):+.2f}")
    print(f"results: {OUT_ROOT}")


if __name__ == "__main__":
    main()
