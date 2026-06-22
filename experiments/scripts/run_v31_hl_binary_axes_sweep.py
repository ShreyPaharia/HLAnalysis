#!/usr/bin/env python3
"""Walk-forward sweep of the v31/theta HL BINARY theta axes, anchored on the
CURRENT live config (msd2.5/esd1.5, deployed 2026-06-22).

Companion to run_v31_hl_safetyd_sweep.py — that one swept ONLY the safety_d
hysteresis band and produced the live msd2.5/esd1.5. This one sweeps the OTHER
theta axes (favorite_threshold, edge_buffer, vol_lookback_seconds,
vol_sampling_dt_seconds, tte_max_seconds) to check whether raising the band moved
the optimum on any of them. The safety band is frozen at the new live value.

Harness mirrors run_v31_hl_safetyd_sweep.py exactly (ThreadPool over `hl-bt run`,
parse report.md, walk-forward early/late split, rank by best WORST-HALF PnL).
Anchored on CURRENT live config => the prod_ref cell IS the comparison anchor
(validated 2026-06-22: full $720.80 / 329 trades / Sharpe 7.200 / hit 71.43% /
maxDD $198.76 on the 42-question 6-week corpus).

Each cell is its own `hl-bt run` invocation, so per-cell vol_sampling_dt_seconds
varies safely: `_cli_commands.py` derives reference_resample_seconds from the
cell's config and `assert_hl_cadence_match` hard-raises on any drift (no silent
0-trade revert). ONE cadence per invocation, as the guard requires.

Phases (all on full + early + late windows):
  0. BASELINE : prod config => assert full $720.80 / 329 trades / Sharpe 7.200 /
                hit 71.43% / maxDD $198.76. Abort if mismatch.
  1. OFAT     : one axis at a time around the live anchor.
  2. JOINT    : favorite_threshold x edge_buffer x vol_lookback (3x3x3 = 27),
                frozen tte=43200, dt=5, msd2.5/esd1.5. Pick = best min(early,late).

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
WORKTREE = Path(__file__).resolve().parents[1]  # this worktree
DATA_ROOT = REPO_ROOT / "data"  # shared HL corpus
OUT_ROOT = REPO_ROOT / "data/sim/runs/v31-hl-binary-axes-2026-06-22"
OUTER_WORKERS, RUN_WORKERS = 3, 4

WINDOWS = {
    "full": ("2026-05-10", "2026-06-21"),
    "early": ("2026-05-10", "2026-05-31"),
    "late": ("2026-05-31", "2026-06-21"),
}

# CURRENT live HL binary config (config/strategy.yaml theta block, 2026-06-22:
# msd2.5/esd1.5 band).
PROD = {
    "vol_lookback_seconds": 900,
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
    "exit_safety_d": 1.5,
    "min_safety_d": 2.5,
    "topup_enabled": True,
    "topup_threshold_pct": 0.2,
    "topup_min_notional_usd": 11.0,
}
BASELINE_EXPECT = {
    "n_markets": 42,
    "n_trades": 329,
    "total_pnl_usd": 720.80,
    "sharpe": 7.200,
    "hit_rate": 0.7143,
    "max_drawdown_usd": 198.76,
}

# ---- Phase 1: OFAT (each = {**PROD, axis: val}) ----
OFAT = {
    "favorite_threshold=0.80": {"favorite_threshold": 0.80},
    "favorite_threshold=0.90": {"favorite_threshold": 0.90},
    "edge_buffer=0.0": {"edge_buffer": 0.0},
    "edge_buffer=0.005": {"edge_buffer": 0.005},
    "edge_buffer=0.01": {"edge_buffer": 0.01},
    "edge_buffer=0.03": {"edge_buffer": 0.03},
    "vol_lookback_seconds=600": {"vol_lookback_seconds": 600},
    "vol_lookback_seconds=1800": {"vol_lookback_seconds": 1800},
    "vol_lookback_seconds=3600": {"vol_lookback_seconds": 3600},
    "vol_sampling_dt_seconds=2": {"vol_sampling_dt_seconds": 2},
    "vol_sampling_dt_seconds=10": {"vol_sampling_dt_seconds": 10},
    "tte_max_seconds=28800": {"tte_max_seconds": 28800},  # 8h
    "tte_max_seconds=36000": {"tte_max_seconds": 36000},  # 10h
    "tte_max_seconds=50400": {"tte_max_seconds": 50400},  # 14h
    "tte_max_seconds=57600": {"tte_max_seconds": 57600},  # 16h
}

# ---- Phase 2: joint grid (binary load-bearing axes) ----
FAV = [0.80, 0.85, 0.90]
EB = [0.0, 0.01, 0.02]
VLB = [600, 900, 1800]


def make_configs() -> dict[str, dict]:
    cfgs: dict[str, dict] = {"prod_ref": dict(PROD)}
    for label, override in OFAT.items():
        cfgs[f"ofat::{label}"] = {**PROD, **override}
    for fav, eb, vlb in product(FAV, EB, VLB):
        label = f"fav{fav}_eb{eb}_vlb{vlb}"
        cfgs[label] = {**PROD, "favorite_threshold": fav, "edge_buffer": eb, "vol_lookback_seconds": vlb}
    return cfgs


def parse_report(path: Path) -> dict:
    text = path.read_text()

    def find(pat, cast=float, default=0):
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
        cmd = [
            "uv",
            "run",
            "hl-bt",
            "run",
            "--strategy",
            "v3_theta_harvester",
            "--data-source",
            "hl_hip4",
            "--config",
            str(cfg_path),
            "--out-dir",
            str(out_dir),
            "--start",
            start,
            "--end",
            end,
            "--kind",
            "binary",
            "--workers",
            str(RUN_WORKERS),
        ]
        env = {**os.environ, "HLBT_HL_DATA_ROOT": str(DATA_ROOT)}
        t0 = time.time()
        p = subprocess.run(cmd, cwd=WORKTREE, capture_output=True, text=True, env=env)
        if p.returncode != 0:
            print(f"[FAIL {name}] {p.stderr.strip()[-300:]}")
            return {"label": label, "window": window}
        print(f"[ok {name:40}] {time.time() - t0:5.0f}s")
    return {"label": label, "window": window, **(parse_report(rpt) if rpt.exists() else {})}


def check_baseline(idx: dict) -> bool:
    r = idx.get(("prod_ref", "full"), {})
    print("\n=== PHASE 0: prod baseline reproduction (binary, full corpus) ===")
    ok = True
    for k, exp in BASELINE_EXPECT.items():
        got = r.get(k, float("nan"))
        tol = 0.01 if isinstance(exp, float) else 0
        match = abs(got - exp) <= tol
        ok = ok and match
        print(f"  {k:18} got={got:>10}  expect={exp:>10}  {'OK' if match else '*** MISMATCH ***'}")
    print(f"  => baseline {'REPRODUCED' if ok else 'MISMATCH — investigate'}")
    return ok


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cfgs = make_configs()
    jobs = [{"label": l, "window": w, "params": p} for l, p in cfgs.items() for w in WINDOWS]
    print(
        f"{len(cfgs)} configs x {len(WINDOWS)} windows = {len(jobs)} runs "
        f"(outer={OUTER_WORKERS}, run_workers={RUN_WORKERS})"
    )
    results = []
    with ThreadPoolExecutor(max_workers=OUTER_WORKERS) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for f in as_completed(futs):
            results.append(f.result())
    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2))

    idx = {(r["label"], r["window"]): r for r in results}

    def pnl(l, w):
        return idx.get((l, w), {}).get("total_pnl_usd", float("nan"))

    def mkrow(label):
        rf = idx.get((label, "full"), {})
        pe, pl = pnl(label, "early"), pnl(label, "late")
        return {
            "label": label,
            "full": rf.get("total_pnl_usd", float("nan")),
            "sharpe": rf.get("sharpe", 0),
            "dd": rf.get("max_drawdown_usd", 0),
            "trades": rf.get("n_trades", 0),
            "hit": rf.get("hit_rate", 0),
            "early": pe,
            "late": pl,
            "worst_half": min(pe, pl),
        }

    def fmt_row(r):
        return (
            f"{r['label']:>26} ${r['full']:>8.2f} {r['sharpe']:>6.2f} ${r['dd']:>7.2f} "
            f"${r['early']:>8.2f} ${r['late']:>8.2f} ${r['worst_half']:>8.2f} "
            f"{r['trades']:>6} {r['hit']:>4.0%}"
        )

    hdr = (
        f"{'config':>26} {'full':>9} {'Shrp':>6} {'maxDD':>8} "
        f"{'early':>9} {'late':>9} {'worst':>9} {'trd':>6} {'hit':>5}"
    )

    baseline_ok = check_baseline(idx)
    pr = mkrow("prod_ref")

    # ---- Phase 1: OFAT ----
    print("\n" + "=" * 100)
    print(
        f"PHASE 1 — OFAT  (live anchor: full=${pr['full']:.2f} "
        f"worst-half=${pr['worst_half']:.2f} maxDD=${pr['dd']:.2f})  sorted by WORST-HALF"
    )
    print("-" * 100)
    print(hdr)
    print(fmt_row(pr))
    print("-" * 100)
    ofat_rows = sorted((mkrow(f"ofat::{l}") for l in OFAT), key=lambda r: r["worst_half"], reverse=True)
    for r in ofat_rows:
        r["label"] = r["label"].replace("ofat::", "")
        print(fmt_row(r))

    # ---- Phase 2: joint grid ----
    grid_labels = [f"fav{fav}_eb{eb}_vlb{vlb}" for fav, eb, vlb in product(FAV, EB, VLB)]
    grid_rows = sorted((mkrow(l) for l in grid_labels), key=lambda r: r["worst_half"], reverse=True)
    print("\n" + "=" * 100)
    print(
        "PHASE 2 — JOINT GRID  fav x edge_buffer x vol_lookback "
        "(FIXED tte=12h, dt=5, msd2.5/esd1.5)  sorted by WORST-HALF"
    )
    print("-" * 100)
    print(hdr)
    print(fmt_row(pr))
    print("-" * 100)
    for r in grid_rows:
        print(fmt_row(r))
    print("=" * 100)

    best = grid_rows[0]
    print(f"\nBASELINE {'OK' if baseline_ok else 'MISMATCH!!'}")
    print(f"ROBUST PICK (best worst-half across grid): {best['label']}")
    print(
        f"  full=${best['full']:.2f}  early=${best['early']:.2f}  late=${best['late']:.2f}  "
        f"worst-half=${best['worst_half']:.2f}  Sharpe={best['sharpe']:.2f}  maxDD=${best['dd']:.2f}  "
        f"trades={best['trades']}  hit={best['hit']:.0%}"
    )
    print(
        f"  vs prod: full ${best['full'] - pr['full']:+.2f}  "
        f"worst-half ${best['worst_half'] - pr['worst_half']:+.2f}  maxDD ${best['dd'] - pr['dd']:+.2f}"
    )
    print(f"results: {OUT_ROOT}")


if __name__ == "__main__":
    main()
