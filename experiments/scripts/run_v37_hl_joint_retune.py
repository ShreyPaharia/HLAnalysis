#!/usr/bin/env python3
"""v3.7 JOINT RETUNE: HL HIP-4 cadence × lookback × alpha grid.

Follow-up to the v3.7 cadence spike (`run_v37_hl_1s_sampling.py`). That spike
tuned `momentum_mr_lookback_min=30` and `momentum_mr_alpha_tilt=1.0` at the
*old* 60s cadence, then held them fixed while sweeping cadence — but at finer
dt the same 30-min window holds 60x more samples, so the optimal lookback and
tilt almost certainly move with cadence.

This runner sweeps all three axes jointly on the HL HIP-4 BTC binary corpus:

    vol_sampling_dt_seconds   ∈ {3, 5, 7, 10}
    momentum_mr_lookback_min  ∈ {5, 10, 15, 30}
    momentum_mr_alpha_tilt    ∈ {0.75, 1.0, 1.5}

= 48 cells, plus 3 anchor baselines:
    base_v31_dt60   — v3.1 (momentum_mr OFF) at dt=60          (~$266.59)
    base_v36_dt60   — v3.6 universal (lb30/a1.0) at dt=60       (~$267)
    t13_best_dt5    — T13 winner (dt=5, lb30, a1.0)            (~$316.26)

Everything else stays at the v3.6 universal HL config (indicator=ma_sigma,
mode=tilt, jr_trust=True). Loader reference-resample period auto-couples to
`vol_sampling_dt_seconds` via the run CLI — no extra flag needed.

SHIP CRITERIA (stricter than the loose 4-criteria gate, see task spec): a cell
ships only if it is a strict Pareto improvement over T13 best on ALL of
(PnL >, Sharpe >, maxDD <). Robustness: the winning (lookback, alpha) combo
must land in the top-5 by PnL at TWO consecutive dt values.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data")
OUT_ROOT = WORKTREE / "data" / "sim" / "runs" / "v3-7-hl-joint-retune-2026-05-29"

# T13 winner — the cell to beat (from summeries/v37_hl_1s_sampling_2026_05_28.md).
T13_BEST = {
    "label": "t13_best_dt5",
    "total_pnl_usd": 316.26,
    "sharpe": 18.60,
    "max_drawdown_usd": 26.28,
    "n_trades": 60,
    "hit_rate": 0.889,
}

# v3.6 universal HL config = v3.1-final HL params + jr-tilt-ma_sigma-lb30-a100.
# See [[v31_final_state_2026_05_23]] (HL does NOT take the PM fav=0.9/eb=0.03).
BASE = {
    "vol_lookback_seconds": 3600,
    "vol_sampling_dt_seconds": 60,
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
    "min_bid_notional_usd": 10.0,
    "max_position_usd": 200.0,
    "stop_loss_pct": None,
    "exit_edge_threshold": 0.0,
    "take_profit_price": None,
    "time_stop_seconds": 0,
    "gamma_lambda": 0.0,
    "exit_take_profit_mode": True,
    "exit_fee": 0.0007,
    "exit_safety_d": 1.0,
    "topup_enabled": True,
    "edge_buffer_tau_ref_seconds": 43200,
    # v3.6 universal — jr-tilt-ma_sigma-lb30-a100
    "momentum_mr_enabled": True,
    "momentum_mr_indicator": "ma_sigma",
    "momentum_mr_mode": "tilt",
    "momentum_mr_lookback_min": 30,
    "momentum_mr_alpha_tilt": 1.0,
    "momentum_mr_jr_trust_weight": True,
}

DT_AXIS = [3, 5, 7, 10]
LOOKBACK_AXIS = [5, 10, 15, 30]
ALPHA_AXIS = [0.75, 1.0, 1.5]


def _alpha_tag(alpha: float) -> str:
    """0.75 -> a075, 1.0 -> a100, 1.5 -> a150."""
    return f"a{round(alpha * 100):03d}"


def _parse_report(path: Path) -> dict:
    text = path.read_text()

    def find(pat: str, cast=float, default=0):
        m = re.search(pat, text)
        if not m:
            return default
        return cast(m.group(1).replace(",", ""))

    return {
        "n_markets": find(r"questions:\s+(\d+)", int),
        "n_trades": find(r"trades:\s+(\d+)", int),
        "total_pnl_usd": find(r"total PnL:\s+\$([-\d.,]+)"),
        "sharpe": find(r"Sharpe[^:]*:\s+([-\d.]+)"),
        "hit_rate": find(r"hit rate:\s+([\d.]+)%") / 100.0,
        "max_drawdown_usd": find(r"max drawdown:\s+\$([-\d.,]+)"),
    }


def run_cell(label: str, params: dict) -> dict:
    out_dir = OUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps(params, indent=2))

    report = out_dir / "report.md"
    if report.exists() and report.stat().st_size > 100:
        r = _parse_report(report)
        print(
            f"[skip] {label} (cached) : PnL=${r.get('total_pnl_usd', 0):.2f} "
            f"trades={r.get('n_trades', 0)} sharpe={r.get('sharpe', 0):.2f} "
            f"maxDD=${r.get('max_drawdown_usd', 0):.2f}"
        )
        return r

    cmd = [
        "uv", "run", "hl-bt", "run",
        "--strategy", "v3_5_momentum_mr",
        "--data-source", "hl_hip4",
        "--config", str(cfg_path),
        "--out-dir", str(out_dir),
        "--start", "2026-05-06", "--end", "2026-05-28",
        "--kind", "binary",
    ]
    # Quiet the per-tick DEBUG spam (topup_skip etc.) — report.md is written
    # regardless of log level; this just saves formatting thousands of lines.
    env = {
        **os.environ,
        "HLBT_HL_DATA_ROOT": str(DATA_ROOT),
        "LOGURU_LEVEL": "WARNING",
    }
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=WORKTREE, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"[FAIL {label}] rc={proc.returncode}")
        print(proc.stderr[-3000:])
        raise SystemExit(1)
    r = _parse_report(report) if report.exists() else {}
    print(
        f"[ok] {label} ({dt:.1f}s) : "
        f"PnL=${r.get('total_pnl_usd', 0):.2f} trades={r.get('n_trades', 0)} "
        f"hit={r.get('hit_rate', 0):.1%} sharpe={r.get('sharpe', 0):.2f} "
        f"maxDD=${r.get('max_drawdown_usd', 0):.2f}"
    )
    return r


def build_cells() -> list[tuple[str, dict, dict]]:
    """Return [(label, params, meta), ...] for baselines + 48-cell grid."""
    cells: list[tuple[str, dict, dict]] = []

    # --- anchor baselines (all at dt=60) ---
    v31 = {**BASE, "vol_sampling_dt_seconds": 60, "momentum_mr_enabled": False}
    cells.append(("base_v31_dt60", v31, {"kind": "baseline", "dt": 60}))

    v36 = {**BASE, "vol_sampling_dt_seconds": 60}
    cells.append(("base_v36_dt60", v36, {"kind": "baseline", "dt": 60}))

    t13 = {
        **BASE,
        "vol_sampling_dt_seconds": 5,
        "momentum_mr_lookback_min": 30,
        "momentum_mr_alpha_tilt": 1.0,
    }
    cells.append(("t13_best_dt5", t13, {"kind": "baseline", "dt": 5}))

    # --- 48-cell joint grid ---
    for dt_s in DT_AXIS:
        for lb in LOOKBACK_AXIS:
            for alpha in ALPHA_AXIS:
                label = f"dt{dt_s}-lb{lb}-{_alpha_tag(alpha)}"
                params = {
                    **BASE,
                    "vol_sampling_dt_seconds": dt_s,
                    "momentum_mr_lookback_min": lb,
                    "momentum_mr_alpha_tilt": alpha,
                }
                meta = {"kind": "grid", "dt": dt_s, "lb": lb, "alpha": alpha}
                cells.append((label, params, meta))

    return cells


def _pareto_vs_t13(r: dict) -> bool:
    """Strict Pareto improvement over T13 best on PnL, Sharpe, maxDD."""
    return (
        r.get("total_pnl_usd", 0) > T13_BEST["total_pnl_usd"]
        and r.get("sharpe", 0) > T13_BEST["sharpe"]
        and r.get("max_drawdown_usd", 1e9) < T13_BEST["max_drawdown_usd"]
    )


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cells = build_cells()
    print(f"Running {len(cells)} cells "
          f"({sum(1 for _, _, m in cells if m['kind'] == 'grid')} grid + "
          f"{sum(1 for _, _, m in cells if m['kind'] == 'baseline')} baselines)\n")

    results: list[dict] = []
    for label, params, meta in cells:
        r = run_cell(label, params)
        results.append({"label": label, "meta": meta, "metrics": r})

    # ---- summary table ----
    print()
    print(f"{'cell':>18} {'PnL':>9} {'trades':>7} {'hit':>7} {'sharpe':>7} {'maxDD':>9} {'pareto':>7}")
    for row in results:
        r = row["metrics"]
        pareto = "YES" if (row["meta"]["kind"] == "grid" and _pareto_vs_t13(r)) else ""
        print(
            f"{row['label']:>18} ${r.get('total_pnl_usd', 0):>7.2f} "
            f"{r.get('n_trades', 0):>7} {r.get('hit_rate', 0):>6.1%} "
            f"{r.get('sharpe', 0):>7.2f} ${r.get('max_drawdown_usd', 0):>7.2f} {pareto:>7}"
        )

    # ---- Pareto winners ----
    grid = [row for row in results if row["meta"]["kind"] == "grid"]
    pareto = [row for row in grid if _pareto_vs_t13(row["metrics"])]
    print(f"\nStrict Pareto improvements over T13 best "
          f"(PnL>${T13_BEST['total_pnl_usd']}, Sharpe>{T13_BEST['sharpe']}, "
          f"maxDD<${T13_BEST['max_drawdown_usd']}): {len(pareto)}")
    for row in sorted(pareto, key=lambda x: -x["metrics"].get("total_pnl_usd", 0)):
        r = row["metrics"]
        print(f"  {row['label']:>18}  PnL=${r['total_pnl_usd']:.2f} "
              f"Sharpe={r['sharpe']:.2f} maxDD=${r['max_drawdown_usd']:.2f}")

    # ---- robustness: top-5 by PnL per dt; flag (lb,alpha) combos that are
    #      top-5 at two consecutive dt values ----
    top5_by_dt: dict[int, set] = {}
    for dt_s in DT_AXIS:
        rows = sorted(
            (row for row in grid if row["meta"]["dt"] == dt_s),
            key=lambda x: -x["metrics"].get("total_pnl_usd", 0),
        )[:5]
        top5_by_dt[dt_s] = {(row["meta"]["lb"], row["meta"]["alpha"]) for row in rows}

    print("\nTop-5 (lb, alpha) combos per dt:")
    for dt_s in DT_AXIS:
        combos = sorted(top5_by_dt[dt_s])
        print(f"  dt={dt_s}: {combos}")

    robust: dict[tuple, list[int]] = {}
    for i in range(len(DT_AXIS) - 1):
        d1, d2 = DT_AXIS[i], DT_AXIS[i + 1]
        for combo in top5_by_dt[d1] & top5_by_dt[d2]:
            robust.setdefault(combo, [])
            for d in (d1, d2):
                if d not in robust[combo]:
                    robust[combo].append(d)
    print("\n(lb, alpha) combos top-5 at TWO CONSECUTIVE dt values:")
    if robust:
        for combo, dts in sorted(robust.items()):
            print(f"  lb={combo[0]}, alpha={combo[1]}: top-5 at dt={sorted(dts)}")
    else:
        print("  (none — every top-5 combo is single-dt; sampling noise)")

    out_json = OUT_ROOT / "summary.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nfull results -> {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
