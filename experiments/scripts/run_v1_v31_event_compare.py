#!/usr/bin/env python3
"""Event-cadence re-run of the v1 tuned winners vs the live v1 anchors AND the
live v31 (theta_harvester) configs, HL BTC, binary + bucket.

WHY
---
The 2026-06-24 tiered v1 tune (docs/research/2026-06-24-v1-hl-tiered-tune.md) ran
the inner `hl-bt run` cells at the DEFAULT fixed 60s scanner cadence (the harness
passed `--strategy` without `--scan-mode`, so `_cli_plumbing` defaulted to
fixed/60s). σ-sampling was the live dt=5, but the *decision* cadence was 60s, not
the live event-driven cadence (scan_min=0.2 / scan_max=2.0). That under-samples the
exit gates (`exit_safety_d`), so the absolute PnL and especially the binary
exit_safety_d=0-vs-1 call need confirming at event cadence.

This script re-runs, all at EVENT cadence (scan_min=0.2 / scan_max=2.0, mirroring
live + the v31 HL tunes):
  * binary: v1 LIVE, v1 TUNED (esd0 winner), v1 TUNED-CONS (esd1 variant), v31 LIVE
  * bucket: v1 LIVE, v1 TUNED (tte8h/λ0.85 winner), v31 LIVE
and prints one walk-forward (worst-half) comparison table per class.

Driver is single-strategy per invocation, so v1 and v31 cells run as separate
`resumable_run.py` calls into separate out-bases; aggregation reads both per class.

Analysis only. No config change, no deploy.

Usage
-----
    # run everything (4 driver invocations) then print the comparison:
    HLBT_HL_DATA_ROOT=../../data uv run python \
        experiments/scripts/run_v1_v31_event_compare.py --workers 6
    # just re-print from completed cells:
    ... --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis")
WORKTREE = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
DRIVER = WORKTREE / "scripts/perf/resumable_run.py"
OUT_ROOT = REPO_ROOT / "data/sim/runs/v1-v31-event-full-2026-06-24"
# True event cadence (scan floor 0.5s, ceiling 2.0s) — mirrors the live engine and
# the v31 HL driver tunes. This is now tractable after the hftbt_runner ndarray
# passthrough fix (the per-scan O(window) HL-bar tuple build that made fine cadence
# ~85x slower for v1 was removed; one v1 cell at event-0.5 dropped from ~30 min to
# ~21s). Run on the FULL corpus for direct comparability with the 60s tune.
START, END = "2026-05-06", "2026-06-24"
SCAN_MIN, SCAN_MAX = 0.5, 2.0

# ---- v1 (late_resolution) flat params ----
_V1_BASE = {
    "tte_min_seconds": 0, "distance_from_strike_usd_min": 0, "vol_max": 100,
    "max_position_usd": 300.0, "stop_loss_pct": None, "max_strike_distance_pct": 50,
    "min_recent_volume_usd": 100, "stale_data_halt_seconds": 30,
    "vol_sampling_dt_seconds": 5, "vol_estimator": "parkinson",
    "size_cap_max_dist_pct": 1.5, "size_cap_min_ask": 0.88,
    "use_bid_for_entry_gate": True, "min_bid_notional_usd": 25.0,
    "topup_enabled": True, "topup_threshold_pct": 0.2, "topup_min_notional_usd": 11.0,
    "fee_model": "flat", "fee_rate": 0.0,
}
V1 = {
    "binary": {
        "live": {**_V1_BASE, "tte_max_seconds": 7200, "price_extreme_threshold": 0.85,
                 "price_extreme_max": 0.99, "min_safety_d": 3.0, "exit_safety_d": 1.0,
                 "vol_lookback_seconds": 3600, "vol_ewma_lambda": 0.97, "size_cap_near_strike_pct": 1.0},
        # tuned winner (DD-disciplined): vlb5400 / esd0 / λ0.85 / msd3.0
        "tuned": {**_V1_BASE, "tte_max_seconds": 7200, "price_extreme_threshold": 0.85,
                  "price_extreme_max": 0.99, "min_safety_d": 3.0, "exit_safety_d": 0.0,
                  "vol_lookback_seconds": 5400, "vol_ewma_lambda": 0.85, "size_cap_near_strike_pct": 1.0},
        # conservative variant that KEEPS the mid-hold exit (esd1.0) — the cadence-
        # sensitive comparison: does esd=0 still beat esd=1 at event cadence?
        "tuned_cons": {**_V1_BASE, "tte_max_seconds": 7200, "price_extreme_threshold": 0.85,
                       "price_extreme_max": 0.99, "min_safety_d": 3.0, "exit_safety_d": 1.0,
                       "vol_lookback_seconds": 5400, "vol_ewma_lambda": 0.85, "size_cap_near_strike_pct": 1.0},
    },
    "bucket": {
        "live": {**_V1_BASE, "tte_max_seconds": 21600, "price_extreme_threshold": 0.85,
                 "price_extreme_max": 0.999, "min_safety_d": 3.0, "exit_safety_d": 0.0,
                 "vol_lookback_seconds": 3600, "vol_ewma_lambda": 0.97, "size_cap_near_strike_pct": 0.0},
        # tuned winner: tte8h / λ0.85
        "tuned": {**_V1_BASE, "tte_max_seconds": 28800, "price_extreme_threshold": 0.85,
                  "price_extreme_max": 0.999, "min_safety_d": 3.0, "exit_safety_d": 0.0,
                  "vol_lookback_seconds": 3600, "vol_ewma_lambda": 0.85, "size_cap_near_strike_pct": 0.0},
    },
}

# ---- v31 (theta_harvester) flat params — CURRENT LIVE (config/strategy.yaml 2026-06-24) ----
_V31_BIN = {
    "vol_lookback_seconds": 1800, "vol_sampling_dt_seconds": 5, "vol_clip_min": 0.0,
    "vol_clip_max": 5.0, "edge_buffer": 0.01, "fee_taker": 0.0,
    "half_spread_assumption": 0.005, "drift_lookback_seconds": 3600, "drift_blend": 0.0,
    "favorite_threshold": 0.80, "tte_min_seconds": 0, "tte_max_seconds": 43200,
    "edge_max": None, "min_distance_pct": None, "min_bid_notional_usd": 0.0,
    "max_position_usd": 500.0, "stop_loss_pct": None, "exit_edge_threshold": 0.0,
    "time_stop_seconds": 0, "exit_take_profit_mode": True, "exit_fee": 0.0007,
    "exit_safety_d": 1.5, "min_safety_d": 2.5, "topup_enabled": True,
    "topup_threshold_pct": 0.2, "topup_min_notional_usd": 11.0,
}
V31 = {
    "binary": {"live": dict(_V31_BIN)},
    "bucket": {"live": {**_V31_BIN, "favorite_threshold": 0.85, "edge_buffer": 0.02,
                        "exit_safety_d": 1.0, "min_safety_d": 2.0, "vol_lookback_seconds": 2700,
                        "vol_sampling_dt_seconds": 2, "tte_max_seconds": 28800,
                        "exit_spread_hold": 0.04}},
}

STRATEGY_KEY = {"v1": "v1_late_resolution", "v31": "v3_theta_harvester"}
CONFIGS = {"v1": V1, "v31": V31}


def base(kind: str, strat: str) -> Path:
    return OUT_ROOT / kind / strat


def write_and_run(kind: str, strat: str, workers: int) -> int:
    cfgs = CONFIGS[strat][kind]
    b = base(kind, strat)
    b.mkdir(parents=True, exist_ok=True)
    cells = []
    for cid, params in cfgs.items():
        cdir = b / cid
        cdir.mkdir(parents=True, exist_ok=True)
        pj = cdir / "params.json"
        pj.write_text(json.dumps(params, indent=2))
        cells.append({"id": cid, "slot_config": str(pj)})
    configs_path = b / "configs.json"
    configs_path.write_text(json.dumps(cells, indent=2))
    cmd = [
        sys.executable, str(DRIVER),
        "--strategy", STRATEGY_KEY[strat],
        "--kind", kind, "--start", START, "--end", END,
        "--out-base", str(b), "--configs", str(configs_path),
        "--scan-min", str(SCAN_MIN), "--scan-max", str(SCAN_MAX),  # event cadence
        "--workers", str(workers), "--chunk-size", "4",
    ]
    env = {**os.environ, "HLBT_HL_DATA_ROOT": str(DATA_ROOT)}
    print(f"[run] {kind}/{strat}: {len(cells)} cells, event {SCAN_MIN}/{SCAN_MAX}s cadence", flush=True)
    return subprocess.run(cmd, cwd=WORKTREE, env=env).returncode


def aggregate() -> None:
    import hlanalysis.backtest.data.hl_hip4 as hl
    from hlanalysis.backtest.runner.result import summarise_run

    ds = hl.HLHip4DataSource(str(DATA_ROOT))
    for kind in ("binary", "bucket"):
        klass = "priceBinary" if kind == "binary" else "priceBucket"
        descs = list(ds.discover(start=START, end=END, kinds=(klass,), underlying="BTC"))
        starts = sorted(d.start_ts_ns for d in descs)
        split_ns = starts[len(starts) // 2]
        early = {i for i, d in enumerate(descs) if d.start_ts_ns < split_ns}
        late = {i for i, d in enumerate(descs) if d.start_ts_ns >= split_ns}

        rows = []
        for strat in ("v1", "v31"):
            for cid in CONFIGS[strat][kind]:
                cdir = base(kind, strat) / cid
                per_q: dict[int, tuple[float, int]] = {}
                for qd in cdir.glob("q[0-9]*"):
                    rpt = qd / "report.md"
                    if not rpt.exists():
                        continue
                    idx = int("".join(c for c in qd.name if c.isdigit()))
                    pnl = tr = 0
                    for line in rpt.read_text().splitlines():
                        if line.startswith("- total PnL:"):
                            pnl = float(line.split("$", 1)[1].replace(",", ""))
                        elif line.strip().startswith("- trades:"):
                            tr = int(line.split(":", 1)[1].strip().replace(",", ""))
                    per_q[idx] = (pnl, tr)
                if not per_q:
                    continue
                full = [per_q[i][0] for i in sorted(per_q)]
                full_tr = sum(t for _, t in per_q.values())
                e = [per_q[i][0] for i in sorted(per_q) if i in early]
                l = [per_q[i][0] for i in sorted(per_q) if i in late]
                sf = summarise_run(full, full_tr)
                se = summarise_run(e, sum(per_q[i][1] for i in per_q if i in early)) if e else None
                sl = summarise_run(l, sum(per_q[i][1] for i in per_q if i in late)) if l else None
                ev = se.total_pnl_usd if se else float("nan")
                lv = sl.total_pnl_usd if sl else float("nan")
                rows.append({
                    "name": f"{strat}_{cid}", "full": sf.total_pnl_usd, "sharpe": sf.sharpe,
                    "maxdd": sf.max_drawdown_usd, "trades": sf.n_trades, "hit": sf.hit_rate,
                    "n": sf.n_markets, "early": ev, "late": lv, "worst_half": min(ev, lv),
                })

        (OUT_ROOT / f"{kind}_compare.json").write_text(json.dumps(rows, indent=2))
        hdr = (f"{'config':>22} {'full':>9} {'Shrp':>6} {'maxDD':>8} "
               f"{'early':>9} {'late':>9} {'worst':>9} {'trd':>5} {'hit':>5} {'n':>3}")
        print("\n" + "=" * 100)
        print(f"HL {kind.upper()} @ EVENT cadence (scan {SCAN_MIN}/{SCAN_MAX}) — {START}..{END}, n={len(descs)} "
              f"(early={len(early)}/late={len(late)}) — sorted by WORST-HALF")
        print("-" * 100)
        print(hdr)
        for r in sorted(rows, key=lambda r: r["worst_half"], reverse=True):
            print(f"{r['name']:>22} ${r['full']:>8.2f} {r['sharpe']:>6.2f} ${r['maxdd']:>7.2f} "
                  f"${r['early']:>8.2f} ${r['late']:>8.2f} ${r['worst_half']:>8.2f} "
                  f"{r['trades']:>5} {r['hit']:>4.0%} {r['n']:>3}")
        print("=" * 100)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    if not args.aggregate_only:
        for kind in ("binary", "bucket"):
            for strat in ("v1", "v31"):
                write_and_run(kind, strat, args.workers)
    aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
