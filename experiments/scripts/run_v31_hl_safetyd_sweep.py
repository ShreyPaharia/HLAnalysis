#!/usr/bin/env python3
"""Independent walk-forward sweep of the v31/theta HL BINARY safety_d HYSTERESIS BAND.

Question (2026-06-22): the live HL binary band is min_safety_d=2.0 (entry floor) /
exit_safety_d=1.0 (exit trigger) — a 2x ratio. safety_d is ALREADY a z-score
(d = (ln(S/K)+drift)/(sigma*sqrt(tau))), so it is intrinsically vol-normalized;
the open question is whether the THRESHOLD pair is near-optimal or arbitrary, and
whether band WIDTH matters at all — before deciding if a vol-regime-conditional
threshold is worth building.

This sweeps min_safety_d x exit_safety_d INDEPENDENTLY with every other knob frozen
at the CURRENT live HL binary config (vlb=900, dt=5, msd2 revert of 2026-06-18).
Cells with min_safety_d < exit_safety_d are an inverted-band (churn) sanity contrast.

Harness mirrors run_v31_hl_binary_joint.py exactly (ThreadPool over `hl-bt run`,
parse report.md, walk-forward early/late split, rank by best WORST-HALF PnL).
Anchored on CURRENT live config => baseline is NOT the old $830.24; the prod_ref
cell IS the comparison anchor (validated 2026-06-22: full $635.13 / 431 trades /
Sharpe 4.293 / hit 64.29% / maxDD $343.45 on the 42-question 6-week corpus).

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
OUT_ROOT = REPO_ROOT / "data/sim/runs/v31-hl-safetyd-sweep-2026-06-22"
OUTER_WORKERS, RUN_WORKERS = 3, 5

WINDOWS = {
    "full":  ("2026-05-10", "2026-06-21"),
    "early": ("2026-05-10", "2026-05-31"),
    "late":  ("2026-05-31", "2026-06-21"),
}

# CURRENT live HL binary config (config/strategy.yaml theta block, 2026-06-22).
PROD = {
    "vol_lookback_seconds": 900, "vol_sampling_dt_seconds": 5, "vol_clip_min": 0.0,
    "vol_clip_max": 5.0, "edge_buffer": 0.02, "fee_taker": 0.0,
    "half_spread_assumption": 0.005, "drift_lookback_seconds": 3600, "drift_blend": 0.0,
    "favorite_threshold": 0.85, "tte_min_seconds": 0, "tte_max_seconds": 43200,
    "edge_max": None, "min_distance_pct": None, "min_bid_notional_usd": 0.0,
    "max_position_usd": 500.0, "stop_loss_pct": None, "exit_edge_threshold": 0.0,
    "time_stop_seconds": 0, "exit_take_profit_mode": True, "exit_fee": 0.0007,
    "exit_safety_d": 1.0, "min_safety_d": 2.0,
    "topup_enabled": True, "topup_threshold_pct": 0.2, "topup_min_notional_usd": 11.0,
}

# ---- The sweep: independent min_safety_d (entry floor) x exit_safety_d (exit) ----
EXIT_SD = [0.0, 0.5, 1.0, 1.5, 2.0]
MIN_SD = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0]


def make_configs() -> dict[str, dict]:
    cfgs: dict[str, dict] = {"prod_ref": dict(PROD)}
    for msd, esd in product(MIN_SD, EXIT_SD):
        label = f"msd{msd}_esd{esd}"
        cfgs[label] = {**PROD, "min_safety_d": msd, "exit_safety_d": esd}
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
    name = f"{label}__{window}".replace("/", "_")
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
        print(f"[ok {name:32}] {time.time()-t0:5.0f}s")
    return {"label": label, "window": window, **(parse_report(rpt) if rpt.exists() else {})}


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

    def mkrow(label):
        rf = idx.get((label, "full"), {})
        pe, pl = pnl(label, "early"), pnl(label, "late")
        return {"label": label, "full": rf.get("total_pnl_usd", float("nan")),
                "sharpe": rf.get("sharpe", 0), "dd": rf.get("max_drawdown_usd", 0),
                "trades": rf.get("n_trades", 0), "hit": rf.get("hit_rate", 0),
                "early": pe, "late": pl, "worst_half": min(pe, pl)}

    def fmt_row(r):
        return (f"{r['label']:>16} ${r['full']:>8.2f} {r['sharpe']:>6.2f} ${r['dd']:>7.2f} "
                f"${r['early']:>8.2f} ${r['late']:>8.2f} ${r['worst_half']:>8.2f} "
                f"{r['trades']:>6} {r['hit']:>4.0%}")

    hdr = (f"{'config':>16} {'full':>9} {'Shrp':>6} {'maxDD':>8} "
           f"{'early':>9} {'late':>9} {'worst':>9} {'trd':>6} {'hit':>5}")

    pr = mkrow("prod_ref")
    print("\n" + "=" * 96)
    print(f"PROD_REF (current live msd2.0/esd1.0): full=${pr['full']:.2f} "
          f"worst-half=${pr['worst_half']:.2f} maxDD=${pr['dd']:.2f}")

    # full grid sorted by worst-half
    labels = [f"msd{msd}_esd{esd}" for msd, esd in product(MIN_SD, EXIT_SD)]
    rows = sorted((mkrow(l) for l in labels), key=lambda r: r["worst_half"], reverse=True)
    print("=" * 96)
    print("SWEEP min_safety_d x exit_safety_d  (sorted by WORST-HALF PnL)")
    print("-" * 96)
    print(hdr)
    print(fmt_row(pr))
    print("-" * 96)
    for r in rows:
        print(fmt_row(r))
    print("=" * 96)

    # also dump a worst-half matrix (rows=msd, cols=esd)
    print("\nWORST-HALF PnL matrix  (rows=min_safety_d, cols=exit_safety_d)")
    print("        " + "".join(f"esd{e:>7}" for e in EXIT_SD))
    cell = {(r["label"]): r for r in rows}
    for msd in MIN_SD:
        line = f"msd{msd:<5}"
        for esd in EXIT_SD:
            v = cell.get(f"msd{msd}_esd{esd}", {}).get("worst_half", float("nan"))
            line += f"{v:>10.0f}"
        print(line)

    best = rows[0]
    print(f"\nBEST worst-half: {best['label']}  full=${best['full']:.2f} "
          f"early=${best['early']:.2f} late=${best['late']:.2f} "
          f"worst=${best['worst_half']:.2f} Sharpe={best['sharpe']:.2f} "
          f"maxDD=${best['dd']:.2f} trd={best['trades']} hit={best['hit']:.0%}")
    print(f"  vs prod: full ${best['full']-pr['full']:+.2f}  "
          f"worst-half ${best['worst_half']-pr['worst_half']:+.2f}  "
          f"maxDD ${best['dd']-pr['dd']:+.2f}")
    print(f"results: {OUT_ROOT}")


if __name__ == "__main__":
    main()
