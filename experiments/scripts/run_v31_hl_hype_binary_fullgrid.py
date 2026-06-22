#!/usr/bin/env python3
"""FULL cartesian grid sweep of v31/theta on HL HIP-4 **HYPE** priceBinary.

Follow-up to run_v31_hl_hype_binary_sweep.py (which was OAT around the BTC anchor).
HYPE is a new token, so the joint optimum can sit anywhere — this sweeps the FULL
7-axis cartesian product, not perturbations around BTC. Still n=6 (see the OAT doc):
a full grid does NOT cure small-n; it makes spurious "winners" MORE likely, so this
is exploratory characterisation, not a promotable tune.

Tractable via the shared-decode warm-chunk driver scripts/perf/resumable_run.py:
each question is decoded once and replayed against every config (config-inner), so
4608 configs over 6 questions costs ~6 decodes + 4608x6 cheap replays instead of
4608 cold `hl-bt run` starts. Runs ONE pass over the full window; early/late
walk-forward halves are split OFFLINE from per-question PnL (q0-2 early, q3-5 late).

Reference = HL perp (HYPE not on Binance), via the driver's new --underlying HYPE.

DO NOT change live config / DO NOT deploy. Analysis + recommendation only.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

REPO_ROOT = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis")
WORKTREE = Path(__file__).resolve().parents[2]   # worktree root (experiments/scripts/..)
DATA_ROOT = REPO_ROOT / "data"
OUT_ROOT = REPO_ROOT / "data/sim/runs/v31-hl-hype-binary-fullgrid-2026-06-22"
DRIVER = WORKTREE / "scripts/perf/resumable_run.py"

START, END = "2026-06-15", "2026-06-22"   # all 6 HYPE binaries
N_EARLY = 3                                # q0,q1,q2 (expiries 06-16/17/18) vs q3,q4,q5

# Non-swept theta fields (current live HL-binary block minus the 7 swept axes).
BASE = {
    "vol_clip_min": 0.0, "vol_clip_max": 5.0, "fee_taker": 0.0,
    "half_spread_assumption": 0.005, "drift_lookback_seconds": 3600, "drift_blend": 0.0,
    "tte_min_seconds": 0, "edge_max": None, "min_distance_pct": None,
    "min_bid_notional_usd": 0.0, "max_position_usd": 500.0, "stop_loss_pct": None,
    "exit_edge_threshold": 0.0, "time_stop_seconds": 0, "exit_take_profit_mode": True,
    "exit_fee": 0.0007, "topup_enabled": True, "topup_threshold_pct": 0.2,
    "topup_min_notional_usd": 11.0,
}

# The 7 swept axes — full cartesian product.
AXES = {
    "favorite_threshold": [0.75, 0.80, 0.85, 0.90],
    "edge_buffer": [0.0, 0.01, 0.02, 0.03],
    "vol_lookback_seconds": [900, 1800, 3600],
    "vol_sampling_dt_seconds": [5, 60],
    "tte_max_seconds": [21600, 43200, 86400],
    "min_safety_d": [1.5, 2.0, 2.5, 3.0],
    "exit_safety_d": [0.0, 1.0, 1.5, 2.0],
}
ORDER = list(AXES)


def cid(combo: dict) -> str:
    return (f"f{combo['favorite_threshold']}_eb{combo['edge_buffer']}_vl{combo['vol_lookback_seconds']}"
            f"_dt{combo['vol_sampling_dt_seconds']}_tte{combo['tte_max_seconds']}"
            f"_msd{combo['min_safety_d']}_esd{combo['exit_safety_d']}")


def gen_configs(limit: int | None) -> list[dict]:
    combos = [dict(zip(ORDER, vals)) for vals in product(*(AXES[a] for a in ORDER))]
    if limit:
        combos = combos[:limit]
    return combos


def write_grid(combos: list[dict]) -> Path:
    cfgs_dir = OUT_ROOT / "cfgs"
    cfgs_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for c in combos:
        ident = cid(c)
        p = cfgs_dir / f"{ident}.json"
        p.write_text(json.dumps({**BASE, **c}, indent=2))
        entries.append({"id": ident, "slot_config": str(p)})
    configs_path = OUT_ROOT / "grid_configs.json"
    configs_path.write_text(json.dumps(entries, indent=2))
    return configs_path


def run_driver(configs_path: Path, workers: int) -> int:
    env = {**os.environ, "HLBT_HL_DATA_ROOT": str(DATA_ROOT), "LOGURU_LEVEL": "ERROR"}
    cmd = [sys.executable, str(DRIVER),
           "--strategy", "v3_theta_harvester", "--underlying", "HYPE",
           "--kind", "binary", "--start", START, "--end", END,
           "--out-base", str(OUT_ROOT), "--configs", str(configs_path),
           "--workers", str(workers), "--chunk-size", "1", "--timeout", "5400"]
    print("driver:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=WORKTREE, env=env).returncode


def parse_q(report: Path) -> dict:
    pnl = ntr = None
    for line in report.read_text().splitlines():
        s = line.strip()
        if s.startswith("- total PnL:"):
            pnl = float(s.split("$", 1)[1].replace(",", ""))
        elif s.startswith("- trades:"):
            ntr = int(s.split(":", 1)[1])
    return {"pnl": pnl or 0.0, "trd": ntr or 0}


def summarise(pnls: list[float], trd: int) -> dict:
    from hlanalysis.backtest.runner.result import summarise_run
    s = summarise_run(pnls, n_trades=trd)
    return {"sharpe": s.sharpe, "dd": s.max_drawdown_usd}


def aggregate(combos: list[dict]) -> None:
    rows = []
    for c in combos:
        ident = cid(c)
        cdir = OUT_ROOT / ident
        qfiles = sorted(cdir.glob("q*/report.md"))
        if len(qfiles) < 6:
            continue
        per = [parse_q(f) for f in qfiles]
        pnls = [p["pnl"] for p in per]
        trd = sum(p["trd"] for p in per)
        early = sum(pnls[:N_EARLY]); late = sum(pnls[N_EARLY:])
        full = sum(pnls)
        nwin = sum(1 for p in pnls if p > 0)
        s = summarise(pnls, trd)
        rows.append({"id": ident, **c, "full": full, "early": early, "late": late,
                     "worst": min(early, late), "trd": trd, "nwin": nwin, **s})
    rows.sort(key=lambda r: r["worst"], reverse=True)
    (OUT_ROOT / "aggregate.json").write_text(json.dumps(rows, indent=2))
    print(f"\n=== FULL GRID: {len(rows)} complete configs (n=6, EXPLORATORY) ===")
    print(f"{'config':>56}{'full':>9}{'early':>8}{'late':>8}{'worst':>8}{'Shrp':>7}{'maxDD':>8}{'trd':>5}{'win':>4}")
    def p(r):
        print(f"{r['id']:>56}{r['full']:>9.2f}{r['early']:>8.2f}{r['late']:>8.2f}"
              f"{r['worst']:>8.2f}{r['sharpe']:>7.1f}{r['dd']:>8.2f}{r['trd']:>5}{r['nwin']:>4}")
    print("--- TOP 20 by worst-half ---")
    for r in rows[:20]:
        p(r)
    print("--- BOTTOM 5 ---")
    for r in rows[-5:]:
        p(r)
    pos = [r for r in rows if r["worst"] > 0]
    print(f"\nconfigs with worst-half > 0: {len(pos)}/{len(rows)}")
    print(f"results: {OUT_ROOT}/aggregate.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="probe: only first N configs")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    combos = gen_configs(a.limit)
    print(f"{len(combos)} configs x 6 questions (full cartesian: "
          f"{'x'.join(str(len(AXES[k])) for k in ORDER)})", flush=True)
    if not a.aggregate_only:
        configs_path = write_grid(combos)
        rc = run_driver(configs_path, a.workers)
        if rc != 0:
            print(f"[warn] driver rc={rc} (some chunks may have failed; aggregating what completed)")
    aggregate(combos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
