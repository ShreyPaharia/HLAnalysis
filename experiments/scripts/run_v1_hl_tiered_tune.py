#!/usr/bin/env python3
"""Tiered (coarse -> fine) walk-forward tune of **v1** (`v1_late_resolution`) on
**HL BTC**, for **binary AND bucket**, driven by the resumable warm-chunk driver
(`scripts/perf/resumable_run.py`).

Why this script exists
----------------------
The v31/theta HL tunes have a driver harness (run_v31_hl_binary_axes_sweep.py +
aggregate_v31_hl_binary_axes.py). v1 did not. This is the v1 equivalent: it
generates per-cell flat param JSONs (v1 builds from a flat dict via
`@register("v1_late_resolution")`), runs every (config x question) cell through
the warm-chunk driver (one bundle decode per question shared across all configs),
then aggregates per-question PnL into a walk-forward worst-half ranking.

Tiered method
-------------
* **coarse**: OFAT (one axis at a time) around the CURRENT LIVE v1 anchor for the
  class, on the six load-bearing v1 axes. Cheap under the driver. Read which axes
  move WORST-HALF PnL -> that is the "direction".
* **fine**: a joint grid over the axes coarse flagged, defined in FINE_GRID after
  inspecting coarse. Same driver, same aggregation.

Ranking is by **worst-half PnL** = min(early, late) over a date-split walk-forward,
the overfit-resistant criterion the v31 tunes use. The live anchor cell
(`prod_ref`) is always included as the comparison baseline.

Corpus: HL BTC recorded data, full available span. dt=5 held FIXED across all
cells (v1+v31 share the BTC feed and move dt in lockstep; not a tune axis here).

Analysis only -- writes NOTHING to config/strategy.yaml, deploys nothing.

Usage
-----
    # coarse tier, binary:
    HLBT_HL_DATA_ROOT=../../data uv run python \
        experiments/scripts/run_v1_hl_tiered_tune.py --kind binary --tier coarse --workers 6
    # coarse tier, bucket:
    ... --kind bucket --tier coarse --workers 6
    # fine tier (after editing FINE_GRID below from coarse direction):
    ... --kind binary --tier fine --workers 6
    # re-aggregate completed cells without re-running:
    ... --kind binary --tier coarse --aggregate-only
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
WORKTREE = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
DRIVER = WORKTREE / "scripts/perf/resumable_run.py"
OUT_ROOT = REPO_ROOT / "data/sim/runs/v1-hl-tiered-event-2026-06-24"
# Tune at the LIVE event cadence (scan floor 0.5s / ceiling 2.0s). The live engine
# runs event-driven 0.2/2.0 (runtime.py:712 _scan_loop); 0.5/2.0 is a ~2.5x-cheaper
# proxy that still re-evaluates exits on book updates. The original 60s-fixed tune
# (v1-hl-tiered-2026-06-24) did NOT match live and its optima did not transfer
# (bucket tte8h reversed; binary esd=0 was an artifact). Tractable after the
# hftbt_runner ndarray fix (~21s/cell). Validate the winner at the exact 0.2/2.0.
SCAN_MIN, SCAN_MAX = 0.5, 2.0

START, END = "2026-05-06", "2026-06-24"

# --- live v1 anchors (config/strategy.yaml, 2026-06-24) -------------------
# Flat dicts mirroring build_v1_late_resolution(params). dt=5 fixed.
_BASE = {
    "tte_min_seconds": 0,
    "distance_from_strike_usd_min": 0,
    "vol_max": 100,
    "max_position_usd": 300.0,
    "stop_loss_pct": None,
    "max_strike_distance_pct": 50,
    "min_recent_volume_usd": 100,
    "stale_data_halt_seconds": 30,
    "vol_sampling_dt_seconds": 5,
    "vol_estimator": "parkinson",
    "size_cap_max_dist_pct": 1.5,
    "size_cap_min_ask": 0.88,
    "use_bid_for_entry_gate": True,
    "min_bid_notional_usd": 25.0,
    "topup_enabled": True,
    "topup_threshold_pct": 0.2,
    "topup_min_notional_usd": 11.0,
    "fee_model": "flat",
    "fee_rate": 0.0,
}
PROD = {
    "binary": {
        **_BASE,
        "tte_max_seconds": 7200,          # 2h
        "price_extreme_threshold": 0.85,
        "price_extreme_max": 0.99,
        "min_safety_d": 3.0,
        "exit_safety_d": 1.0,
        "vol_lookback_seconds": 3600,
        "vol_ewma_lambda": 0.97,
        "size_cap_near_strike_pct": 1.0,  # near-strike low-ask cap ON for binary
    },
    "bucket": {
        **_BASE,
        "tte_max_seconds": 21600,         # 6h
        "price_extreme_threshold": 0.85,
        "price_extreme_max": 0.999,
        "min_safety_d": 3.0,
        "exit_safety_d": 0.0,             # mid-hold disabled on buckets
        "vol_lookback_seconds": 3600,
        "vol_ewma_lambda": 0.97,
        "size_cap_near_strike_pct": 0.0,  # disabled on buckets (live)
    },
}

# --- coarse OFAT axes (off-anchor values per class) -----------------------
# Tuned for EVENT cadence: at 0.5/2.0 exits fire on every book burst, so unlike the
# 60s tune we ALSO probe SHORTER tte (longer holds churned worse at fine cadence)
# and exit_safety_d in BOTH directions (the mid-hold exit is more protective at
# event cadence — esd=0 was a 60s artifact).
COARSE_AXES = {
    "binary": {
        "tte_max_seconds": [3600, 5400, 14400, 28800],  # 1h, 1.5h, 4h, 8h (anchor 2h)
        "min_safety_d": [2.0, 4.0],                      # anchor 3.0
        "exit_safety_d": [0.0, 0.5, 2.0],                # anchor 1.0 (probe both sides)
        "vol_lookback_seconds": [1800, 5400, 7200],      # anchor 3600
        "price_extreme_threshold": [0.80, 0.90],         # anchor 0.85
        "vol_ewma_lambda": [0.0, 0.85],                  # anchor 0.97
    },
    "bucket": {
        "tte_max_seconds": [10800, 14400, 28800, 43200],  # 3h, 4h, 8h, 12h (anchor 6h)
        "min_safety_d": [2.0, 4.0],                       # anchor 3.0
        "exit_safety_d": [1.0, 2.0],                      # anchor 0.0 (mid-hold ON helps at fine cadence)
        "vol_lookback_seconds": [1800, 5400, 7200],       # anchor 3600
        "price_extreme_threshold": [0.80, 0.90],          # anchor 0.85
        "vol_ewma_lambda": [0.0, 0.85],                   # anchor 0.97
    },
}

# --- fine joint grid (filled in AFTER reading coarse direction) -----------
# Map axis -> list of values; the script takes the cartesian product, with all
# non-listed axes pinned to the live anchor. Leave empty until coarse is read.
FINE_GRID: dict[str, dict[str, list]] = {
    # Binary direction (EVENT-cadence coarse 0.5/2.0, 2026-06-24): live LOSES
    # (worst-half -$108). vol_lookback wants LONGER (3600 -$108 → 5400 $7 → 7200
    # $68) and exit_safety_d wants HIGHER (1.0 -$108 → 2.0 $34/maxDD$162; esd=0
    # has $24 worst-half but maxDD $300 — the mid-hold exit is protective at fine
    # cadence, the OPPOSITE of the 60s tune). lambda=0.85 helps (-$26 vs -$108).
    # Refine vol_lookback × exit_safety_d × lambda around (7200, 2.0, 0.85).
    # 3*3*2 = 18 cells; min_safety_d pinned at live 3.0 (2.0/4.0 both worse).
    "binary": {
        "vol_lookback_seconds": [6300, 7200, 8100],
        "exit_safety_d": [1.5, 2.0, 2.5],
        "vol_ewma_lambda": [0.85, 0.97],
    },
    # Bucket direction (EVENT-cadence coarse 0.5/2.0): the ONLY good change is
    # tte_max 6h→8h (6h $236 → 8h $348/Sharpe25.8/maxDD$0, 12h -$215). vol_lookback
    # inert; exit_safety_d MUST stay 0 (1.0 -$455, 2.0 -$921); and lambda=0.85 is
    # TOXIC here (-$93) — KEEP live lambda=0.97. So the fine grid refines tte_max
    # ONLY around 8h (lambda is NOT swept — held at the live 0.97). 4 cells.
    "bucket": {
        "tte_max_seconds": [25200, 28800, 32400, 36000],
    },
}


def make_configs(kind: str, tier: str) -> dict[str, dict]:
    anchor = PROD[kind]
    cfgs: dict[str, dict] = {"prod_ref": dict(anchor)}
    if tier == "coarse":
        for axis, vals in COARSE_AXES[kind].items():
            for v in vals:
                cfgs[f"ofat::{axis}={v}"] = {**anchor, axis: v}
    else:  # fine
        grid = FINE_GRID[kind]
        if not grid:
            sys.exit(f"FINE_GRID['{kind}'] is empty — fill it in from coarse direction first.")
        axes = list(grid.keys())
        for combo in product(*(grid[a] for a in axes)):
            override = dict(zip(axes, combo, strict=True))
            label = "_".join(f"{a}{val}" for a, val in override.items())
            cfgs[label] = {**anchor, **override}
    return cfgs


def out_base(kind: str, tier: str) -> Path:
    return OUT_ROOT / kind / tier


def write_cells(kind: str, tier: str, cfgs: dict[str, dict]) -> Path:
    base = out_base(kind, tier)
    base.mkdir(parents=True, exist_ok=True)
    cells = []
    for cid, params in cfgs.items():
        cdir = base / cid.replace("::", "__").replace("/", "_")
        cdir.mkdir(parents=True, exist_ok=True)
        pj = cdir / "params.json"
        pj.write_text(json.dumps(params, indent=2))
        cells.append({"id": cid.replace("::", "__").replace("/", "_"), "slot_config": str(pj)})
    configs_path = base / "configs.json"
    configs_path.write_text(json.dumps(cells, indent=2))
    return configs_path


def run_driver(kind: str, tier: str, configs_path: Path, workers: int) -> int:
    base = out_base(kind, tier)
    cmd = [
        sys.executable, str(DRIVER),
        "--strategy", "v1_late_resolution",
        "--kind", kind,
        "--start", START, "--end", END,
        "--out-base", str(base),
        "--configs", str(configs_path),
        "--scan-min", str(SCAN_MIN), "--scan-max", str(SCAN_MAX),  # live event cadence
        "--workers", str(workers),
        "--chunk-size", "4",
        "--timeout", "5400",
    ]
    env = {**os.environ, "HLBT_HL_DATA_ROOT": str(DATA_ROOT)}
    print(f"[driver] {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=WORKTREE, env=env).returncode


# --- walk-forward aggregation --------------------------------------------

def _descriptors():
    import hlanalysis.backtest.data.hl_hip4 as hl
    ds = hl.HLHip4DataSource(str(DATA_ROOT))
    return ds


def aggregate(kind: str, tier: str) -> None:
    from hlanalysis.backtest.runner.result import summarise_run

    ds = _descriptors()
    klass = "priceBinary" if kind == "binary" else "priceBucket"
    descs = list(ds.discover(start=START, end=END, kinds=(klass,), underlying="BTC"))
    starts = sorted(d.start_ts_ns for d in descs)
    split_ns = starts[len(starts) // 2]  # median split -> balanced halves
    early_idx = {i for i, d in enumerate(descs) if d.start_ts_ns < split_ns}
    late_idx = {i for i, d in enumerate(descs) if d.start_ts_ns >= split_ns}

    base = out_base(kind, tier)
    cfg_ids = [c["id"] for c in json.loads((base / "configs.json").read_text())]

    def parse_q(rpt: Path) -> float:
        pnl = 0.0
        for line in rpt.read_text().splitlines():
            if line.startswith("- total PnL:"):
                pnl = float(line.split("$", 1)[1].replace(",", ""))
        return pnl

    rows = []
    for cid in cfg_ids:
        cdir = base / cid
        per_q: dict[int, float] = {}
        per_q_tr: dict[int, int] = {}
        for qd in cdir.glob("q[0-9]*"):
            rpt = qd / "report.md"
            if not rpt.exists():
                continue
            idx = int("".join(ch for ch in qd.name if ch.isdigit()))
            pnl = tr = 0
            for line in rpt.read_text().splitlines():
                if line.startswith("- total PnL:"):
                    pnl = float(line.split("$", 1)[1].replace(",", ""))
                elif line.strip().startswith("- trades:"):
                    tr = int(line.split(":", 1)[1].strip().replace(",", ""))
            per_q[idx] = pnl
            per_q_tr[idx] = tr
        if not per_q:
            continue
        full = [per_q[i] for i in sorted(per_q)]
        full_tr = sum(per_q_tr.values())
        early = [per_q[i] for i in sorted(per_q) if i in early_idx]
        late = [per_q[i] for i in sorted(per_q) if i in late_idx]
        sf = summarise_run(full, full_tr)
        se = summarise_run(early, sum(per_q_tr[i] for i in per_q if i in early_idx)) if early else None
        sl = summarise_run(late, sum(per_q_tr[i] for i in per_q if i in late_idx)) if late else None
        e = se.total_pnl_usd if se else float("nan")
        l = sl.total_pnl_usd if sl else float("nan")
        rows.append({
            "config": cid, "full": sf.total_pnl_usd, "sharpe": sf.sharpe,
            "maxdd": sf.max_drawdown_usd, "trades": sf.n_trades, "hit": sf.hit_rate,
            "n": sf.n_markets, "early": e, "late": l, "worst_half": min(e, l),
        })

    (base / "aggregate.json").write_text(json.dumps(rows, indent=2))
    pr = next((r for r in rows if r["config"] == "prod_ref"), None)

    def fmt(r, tag=""):
        return (f"{r['config']:>30} ${r['full']:>8.2f} {r['sharpe']:>6.2f} ${r['maxdd']:>7.2f} "
                f"${r['early']:>8.2f} ${r['late']:>8.2f} ${r['worst_half']:>8.2f} "
                f"{r['trades']:>5} {r['hit']:>4.0%}{tag}")

    hdr = (f"{'config':>30} {'full':>9} {'Shrp':>6} {'maxDD':>8} "
           f"{'early':>9} {'late':>9} {'worst':>9} {'trd':>5} {'hit':>5}")
    print("\n" + "=" * 104)
    print(f"v1 HL {kind.upper()} {tier.upper()} — sorted by WORST-HALF PnL "
          f"(corpus {START}..{END}, n={len(descs)}, split early={len(early_idx)}/late={len(late_idx)})")
    if pr:
        print(f"LIVE anchor: full=${pr['full']:.2f} worst-half=${pr['worst_half']:.2f} "
              f"maxDD=${pr['maxdd']:.2f} Sharpe={pr['sharpe']:.2f} trades={pr['trades']} hit={pr['hit']:.0%}")
    print("-" * 104)
    print(hdr)
    if pr:
        print(fmt(pr, "  <- LIVE"))
    print("-" * 104)
    for r in sorted(rows, key=lambda r: r["worst_half"], reverse=True):
        if r["config"] == "prod_ref":
            continue
        tag = "  *beats-live*" if pr and r["worst_half"] > pr["worst_half"] else ""
        print(fmt(r, tag))
    print("=" * 104)
    print(f"aggregate JSON: {base / 'aggregate.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["binary", "bucket"], required=True)
    ap.add_argument("--tier", choices=["coarse", "fine"], required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    cfgs = make_configs(args.kind, args.tier)
    if not args.aggregate_only:
        configs_path = write_cells(args.kind, args.tier, cfgs)
        print(f"{len(cfgs)} configs for {args.kind}/{args.tier}", flush=True)
        rc = run_driver(args.kind, args.tier, configs_path, args.workers)
        if rc != 0:
            print(f"[warn] driver returned {rc} (some cells may have failed); aggregating completed cells")
    aggregate(args.kind, args.tier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
