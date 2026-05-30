#!/usr/bin/env python3
"""Variable (vol-scaled) TTE entry window — v3.1 (theta_harvester) HL re-validation.

Companion to scripts/run_variable_tte_hl.py (which did v1). v3.1 differs:
  - entry is EDGE-gated (favorite_threshold + edge_buffer), not min_safety_d;
  - σ enters via the GBM p_model and is ANNUALIZED (~0.3–1.5 for BTC), so the
    vol_scaled_tte_ref_sigma grid is on that scale (not v1's per-bar 1e-4);
  - prod is already at tte_max=12h (v1 was 2h), and fee is config-driven
    (fee_taker=0, exit_fee=0.0007 via exit_take_profit_mode).

Baseline = current prod v3.1 HL theta block (sample_std σ, dt=5, tte=12h). The
v3.6 momentum tilt is NOT in prod, so it is omitted here.

Cells:
  - fixed control: vol-scaling OFF, tte_max ∈ {2h, 4h, 12h(prod), 24h}.
  - vol-scaled:    base 12h, ceiling 24h, ref_sigma × exponent grid.

Corpus: HL HIP-4 2026-05-06 → 2026-05-28, kind=both, 5bps slippage (matches
the mid-hold-tte-stack v3.1 methodology that produced the documented $177).
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
OUT_ROOT = WORKTREE / "data" / "sim" / "runs" / "variable-tte-v31-2026-05-31"

# Current prod v3.1 HL theta block (config/strategy.yaml `theta_harvester`),
# minus the (research-only, not-shipped) v3.6 momentum tilt.
BASE = {
    "vol_lookback_seconds": 3600,
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
    "exit_safety_d": 1.0,
    "topup_enabled": True,
    "topup_threshold_pct": 0.2,
    "topup_min_notional_usd": 11.0,
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
        "--strategy", "v3_theta_harvester",
        "--data-source", "hl_hip4",
        "--config", str(cfg_path),
        "--out-dir", str(out_dir),
        "--start", "2026-05-06", "--end", "2026-05-28",
        "--kind", "both",
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

    for hrs, secs in [(2, 7200), (4, 14400), (12, 43200), (24, 86400)]:
        cells.append((f"fixed_{hrs}h", {"tte_max_seconds": secs}))

    for ref in (0.3, 0.5, 0.8, 1.2):
        for k in (1.0, 2.0):
            cells.append((
                f"vscale_ref{ref:g}_k{k:g}",
                {
                    "tte_max_seconds": 43200,
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
    print(f"{'cell':>22} {'PnL':>9} {'trades':>7} {'hit':>7} {'sharpe':>7} {'maxDD':>9}")
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
