#!/usr/bin/env python3
"""Variable (vol-scaled) TTE entry window — HL HIP-4 re-validation on the latest model.

Re-runs the "dynamic TTE as a function of σ" idea (memory:
backtester_polymarket_direction — "lower vol → enter earlier, higher vol →
require closer to resolution") on TOP of the CURRENT prod v1 config
(Parkinson σ, dt=5, min_safety_d=3.0, λ=0.97) and compares against the fixed
tte_max=7200 prod baseline.

Mechanism (LateResolutionConfig.vol_scaled_tte_*, off by default — fixed path
bit-identical):
    tte_max_eff = tte_max_seconds * (ref_sigma / σ) ** exponent,
                  clamped to [0, ceiling].

Cells:
  - fixed control: vol-scaling OFF, tte_max ∈ {1h, 2h(prod), 4h, 12h, 24h}.
  - vol-scaled:    base tte_max=7200, ceiling=24h, ref_sigma × exponent grid.

Corpus: HL HIP-4 2026-05-06 → 2026-05-28, kind=both, fee 0.00035, 5bps slippage —
same as scripts/run_v1_cadence_hl.py so numbers are comparable.
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
OUT_ROOT = WORKTREE / "data" / "sim" / "runs" / "variable-tte-2026-05-30"

# Current prod v1 HL priceBinary slot (config/strategy.yaml `late_resolution`).
BASE = {
    "tte_min_seconds": 0,
    "tte_max_seconds": 7200,
    "price_extreme_threshold": 0.85,
    "price_extreme_max": 0.99,
    "min_safety_d": 3.0,
    "vol_lookback_seconds": 3600,
    "exit_safety_d": 1.0,
    "vol_ewma_lambda": 0.97,
    "vol_estimator": "parkinson",
    "vol_sampling_dt_seconds": 5,
    "distance_from_strike_usd_min": 0,
    "vol_max": 100,
    "max_position_usd": 300,
    "stop_loss_pct": None,
    "size_cap_near_strike_pct": 1.0,
    "size_cap_max_dist_pct": 1.5,
    "size_cap_min_ask": 0.88,
    "use_bid_for_entry_gate": True,
    "min_bid_notional_usd": 25.0,
}


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


def run_cell(label: str, overrides: dict) -> dict:
    params = {**BASE, **overrides}
    out_dir = OUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps(params, indent=2))

    report = out_dir / "report.md"
    if report.exists() and report.stat().st_size > 100:
        print(f"[skip] {label} (cached)")
        return _parse_report(report)

    cmd = [
        "uv", "run", "hl-bt", "run",
        "--strategy", "v1_late_resolution",
        "--data-source", "hl_hip4",
        "--config", str(cfg_path),
        "--out-dir", str(out_dir),
        "--start", "2026-05-06", "--end", "2026-05-28",
        "--kind", "both",
        "--fee-taker", "0.00035",
        "--slippage-bps", "5.0",
    ]
    env = {**os.environ, "HLBT_HL_DATA_ROOT": str(DATA_ROOT)}
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


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cells: list[tuple[str, dict]] = []

    # --- Fixed-window control (vol-scaling OFF) ---
    for hrs, secs in [(1, 3600), (2, 7200), (4, 14400), (12, 43200), (24, 86400)]:
        cells.append((f"fixed_{hrs}h", {"tte_max_seconds": secs}))

    # --- Vol-scaled window (ON): base 2h, ceiling 24h ---
    for ref in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
        for k in (1.0, 2.0):
            ref_lbl = f"{ref:.0e}".replace("-0", "-").replace("e", "e")
            cells.append((
                f"vscale_ref{ref_lbl}_k{k:g}",
                {
                    "tte_max_seconds": 7200,
                    "vol_scaled_tte_enabled": True,
                    "vol_scaled_tte_ref_sigma": ref,
                    "vol_scaled_tte_exponent": k,
                    "vol_scaled_tte_ceiling_seconds": 86400,
                },
            ))

    results: list[tuple[str, dict]] = []
    for label, ov in cells:
        results.append((label, run_cell(label, ov)))

    print()
    hdr = f"{'cell':>22} {'PnL':>9} {'trades':>7} {'hit':>7} {'sharpe':>7} {'maxDD':>9}"
    print(hdr)
    for label, r in results:
        print(
            f"{label:>22} ${r.get('total_pnl_usd', 0):>7.2f} "
            f"{r.get('n_trades', 0):>7} {r.get('hit_rate', 0):>6.1%} "
            f"{r.get('sharpe', 0):>7.2f} ${r.get('max_drawdown_usd', 0):>7.2f}"
        )

    out_json = OUT_ROOT / "sweep_summary.json"
    out_json.write_text(json.dumps(
        [{"label": l, "metrics": r} for l, r in results], indent=2))
    print(f"\nfull results → {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
