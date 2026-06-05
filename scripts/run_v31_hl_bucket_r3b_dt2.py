#!/usr/bin/env python3
"""Round-3b: dt=2 confirmation neighborhood on the FIXED engine (main @ 7881738).

Round 3 (old engine, walk-forward) found the bucket risk-adjusted optimum near
fav=0.80 / tte=8h / vol_lookback~2700-3600 with exit_safety_d=0 / edge_buffer=0.005,
and — surprise — dt=2 beat the round-2 dt=1 full-sample winner under walk-forward
(center cell: dt2 worst-half $433 / Sharpe 27 / DD $0 vs dt1 $340 / 22.8 / $3).

This locks dt=2 and sweeps the top neighborhood, on the FIXED engine. It also
re-runs two known round-3 cells as dt-path re-validation: the benchmark battery
only exercised dt=5, but the regression WAS a dt-handling bug, so we confirm
dt=1 and dt=2 reproduce the old-engine numbers to the cent.

Parity (serial == --workers) is confirmed, so we use --workers for speed.
"""
from __future__ import annotations
import json, os, re, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

REPO_ROOT = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis")
WORKTREE = REPO_ROOT / ".worktrees/now-we-have-enough-data-for-hl-binary-and-buckets--0eWFmE"  # fixed engine
DATA_ROOT = REPO_ROOT / "data"
OUT_ROOT = REPO_ROOT / "data/sim/runs/v31-hl-bucket-r3b-dt2-2026-06-05"
OUTER_WORKERS, RUN_WORKERS = 2, 6

WINDOWS = {"full": ("2026-05-06", "2026-06-04"),
           "early": ("2026-05-06", "2026-05-20"),
           "late": ("2026-05-20", "2026-06-04")}

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
# fixed knobs for the neighborhood (dt now LOCKED at 2)
FIXED = {"exit_safety_d": 0.0, "edge_buffer": 0.005, "vol_sampling_dt_seconds": 2}

FAV = [0.80, 0.85]
TTE = [28800, 36000]           # 8h / 10h
VLB = [2700, 3600]

# round-3 (old-engine) expectations for the dt re-validation cells
REVAL = {
    # label: (params, expected_full_pnl)
    "reval_dt1_fav0.8_tte8h_vlb3600": (
        {**PROD, "exit_safety_d": 0.0, "edge_buffer": 0.005, "vol_sampling_dt_seconds": 1,
         "favorite_threshold": 0.80, "tte_max_seconds": 28800, "vol_lookback_seconds": 3600},
        1140.52),
    "reval_dt2_fav0.8_tte8h_vlb2700": (
        {**PROD, "exit_safety_d": 0.0, "edge_buffer": 0.005, "vol_sampling_dt_seconds": 2,
         "favorite_threshold": 0.80, "tte_max_seconds": 28800, "vol_lookback_seconds": 2700},
        1135.72),
    "prod_ref": (dict(PROD), 412.34),
}


def make_configs():
    cfgs = {}
    for fav, tte, vlb in product(FAV, TTE, VLB):
        cfgs[f"fav{fav}_tte{tte//3600}h_vlb{vlb}_dt2"] = {
            **PROD, **FIXED, "favorite_threshold": fav,
            "tte_max_seconds": tte, "vol_lookback_seconds": vlb}
    for label, (params, _exp) in REVAL.items():
        cfgs[label] = params
    return cfgs


def parse_report(path):
    text = path.read_text()
    def find(pat, cast=float, default=0):
        m = re.search(pat, text); return cast(m.group(1).replace(',', '')) if m else default
    return {'n_trades': find(r'trades:\s+(\d+)', int),
            'total_pnl_usd': find(r'total PnL:\s+\$([-\d.,]+)'),
            'sharpe': find(r'Sharpe[^:]*:\s+([-\d.]+)'),
            'hit_rate': find(r'hit rate:\s+([\d.]+)%') / 100.0,
            'max_drawdown_usd': find(r'max drawdown:\s+\$([-\d.,]+)')}


def run_one(job):
    label, window, params = job["label"], job["window"], job["params"]
    name = f"{label}__{window}"
    cfg_path = OUT_ROOT / f"cfg_{name}.json"; out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True); cfg_path.write_text(json.dumps(params, indent=2))
    rpt = out_dir / "report.md"
    if not (rpt.exists() and rpt.stat().st_size > 100):
        s, e = WINDOWS[window]
        cmd = ["uv", "run", "hl-bt", "run", "--strategy", "v3_theta_harvester",
               "--data-source", "hl_hip4", "--config", str(cfg_path), "--out-dir", str(out_dir),
               "--start", s, "--end", e, "--kind", "bucket", "--workers", str(RUN_WORKERS)]
        p = subprocess.run(cmd, cwd=WORKTREE, capture_output=True, text=True,
                           env={**os.environ, "HLBT_HL_DATA_ROOT": str(DATA_ROOT)})
        if p.returncode != 0:
            print(f"[FAIL {name}] {p.stderr.strip()[-300:]}"); return {"label": label, "window": window}
    return {"label": label, "window": window, **(parse_report(rpt) if rpt.exists() else {})}


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cfgs = make_configs()
    jobs = [{"label": l, "window": w, "params": p} for l, p in cfgs.items() for w in WINDOWS]
    print(f"FIXED engine — {len(cfgs)} configs x {len(WINDOWS)} = {len(jobs)} runs")
    results = []
    with ThreadPoolExecutor(max_workers=OUTER_WORKERS) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for f in as_completed(futs): results.append(f.result())
    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2))
    idx = {(r["label"], r["window"]): r for r in results}
    def pnl(l, w): return idx.get((l, w), {}).get("total_pnl_usd", float("nan"))

    print("\n=== dt-path RE-VALIDATION (fixed engine must match old-engine round-3) ===")
    ok = True
    for label, (_p, exp) in REVAL.items():
        got = idx.get((label, "full"), {}).get("total_pnl_usd", float("nan"))
        match = abs(got - exp) < 0.01
        ok = ok and match
        print(f"  {label:34} full=${got:8.2f}  expect=${exp:8.2f}  {'MATCH' if match else '*** MISMATCH ***'}")
    print(f"  => dt-path {'CONFIRMED on fixed engine' if ok else 'BROKEN — investigate before trusting r3b'}")

    rows = []
    for label in cfgs:
        if label in REVAL and label != "prod_ref":
            continue
        rf = idx.get((label, "full"), {})
        pe, pl = pnl(label, "early"), pnl(label, "late")
        rows.append({"label": label, "full": rf.get("total_pnl_usd", float("nan")),
                     "sharpe": rf.get("sharpe", 0), "dd": rf.get("max_drawdown_usd", 0),
                     "trades": rf.get("n_trades", 0), "hit": rf.get("hit_rate", 0),
                     "early": pe, "late": pl, "worst_half": min(pe, pl)})
    grid = [r for r in rows if r["label"] != "prod_ref"]
    grid.sort(key=lambda r: r["worst_half"], reverse=True)
    print("\n" + "=" * 100)
    print(f"{'cell (dt=2, by worst-half)':>30} {'full':>8} {'Sharpe':>7} {'maxDD':>8} "
          f"{'early':>8} {'late':>8} {'worst':>8} {'trades':>6} {'hit':>5}")
    for r in grid + [next(x for x in rows if x['label'] == 'prod_ref')]:
        print(f"{r['label']:>30} ${r['full']:>6.2f} {r['sharpe']:>7.2f} ${r['dd']:>6.2f} "
              f"${r['early']:>6.2f} ${r['late']:>6.2f} ${r['worst_half']:>6.2f} {r['trades']:>6} {r['hit']:>4.0%}")
    print("=" * 100)
    best = grid[0]
    print(f"\nROUND-3b PICK: {best['label']}  worst-half=${best['worst_half']:.2f}  "
          f"full=${best['full']:.2f}  Sharpe={best['sharpe']:.2f}  DD=${best['dd']:.2f}")
    print(f"results: {OUT_ROOT}")


if __name__ == "__main__":
    main()
