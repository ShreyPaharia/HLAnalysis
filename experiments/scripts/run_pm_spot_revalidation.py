#!/usr/bin/env python3
"""σ re-validation: PM v1 (late_resolution) reference = Binance SPOT vs PERP BBO.

Gate for the live spot-reference change (PM price/σ/strike now Binance spot).
Holds the v1_pm prod slot config FIXED (mirrors run_v1_cadence_pm.py BASE) and
varies only (reference product_type ∈ {perp, spot}) × (vol_sampling_dt ∈ {60, 5}).

Acceptance: spot dt=5 (the deployed live cadence) is within noise of / better
than the perp baseline. Spot is the instrument PM actually settles on, so it is
the correct series; this confirms the σ tuning carries over (basis cancels in
returns, but we verify rather than assume).

Window 2026-05-06..2026-05-29 (BBO overlap). Corpus ~18-22 markets — suggestive,
not load-bearing (same caveat as the v3.7 PM cadence sweeps).
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
PM_CACHE = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data/sim")
OUT_ROOT = WORKTREE / "data" / "sim" / "runs" / "pm-spot-revalidation-2026-06-01"

# v1_pm prod slot (config/strategy.yaml late_resolution PM, priceBinary).
# Identical to run_v1_cadence_pm.py BASE so this is comparable to that baseline.
BASE = {
    "tte_min_seconds": 0,
    "tte_max_seconds": 86400,
    "price_extreme_threshold": 0.85,
    "price_extreme_max": 0.99,
    "distance_from_strike_usd_min": 0,
    "vol_max": 100,
    "stop_loss_pct": None,
    "max_position_usd": 300,
    "min_safety_d": 1.0,
    "vol_lookback_seconds": 3600,
    "exit_safety_d": 1.0,
    "vol_ewma_lambda": 0.85,
    "size_cap_near_strike_pct": 1.0,
    "size_cap_max_dist_pct": 1.5,
    "size_cap_min_ask": 0.88,
    "fee_model": "pm_binary",
    "fee_rate": 0.07,
    "vol_sampling_dt_seconds": 60,
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


def run_cell(label: str, dt_s: int, product_type: str) -> dict:
    params = {**BASE, "vol_sampling_dt_seconds": dt_s}
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
        "--data-source", "polymarket",
        "--config", str(cfg_path),
        "--out-dir", str(out_dir),
        "--start", "2026-05-06", "--end", "2026-05-29",
        "--kind", "binary",
        "--fee-model", "pm_binary", "--fee-rate", "0.07",
        "--pm-reference-source", "binance_bbo",
        "--pm-binance-bbo-product-type", product_type,
    ]
    env = {**os.environ, "HLBT_PM_CACHE_ROOT": str(PM_CACHE)}
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=WORKTREE, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"[FAIL {label}] rc={proc.returncode}")
        print(proc.stderr[-3000:])
        raise SystemExit(1)
    r = _parse_report(report) if report.exists() else {}
    print(
        f"[ok] {label} ({dt:.1f}s, {product_type} dt={dt_s}s) : "
        f"PnL=${r.get('total_pnl_usd', 0):.2f} trades={r.get('n_trades', 0)} "
        f"hit={r.get('hit_rate', 0):.1%} maxDD=${r.get('max_drawdown_usd', 0):.2f}"
    )
    return r


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cells = [
        ("perp_dt60", 60, "perp"),
        ("spot_dt60", 60, "spot"),
        ("perp_dt5", 5, "perp"),
        ("spot_dt5", 5, "spot"),
    ]
    results: list[tuple[str, int, str, dict]] = []
    for label, dt_s, pt in cells:
        results.append((label, dt_s, pt, run_cell(label, dt_s, pt)))

    print()
    print(f"{'cell':>10} {'dt(s)':>6} {'ref':>6}  {'PnL':>9} {'trades':>7} {'hit':>7} {'sharpe':>7} {'maxDD':>9}")
    for label, dt_s, pt, r in results:
        print(
            f"{label:>10} {dt_s:>6} {pt:>6}  ${r.get('total_pnl_usd', 0):>7.2f} "
            f"{r.get('n_trades', 0):>7} {r.get('hit_rate', 0):>6.1%} "
            f"{r.get('sharpe', 0):>7.2f} ${r.get('max_drawdown_usd', 0):>7.2f}"
        )

    out_json = OUT_ROOT / "summary.json"
    out_json.write_text(json.dumps(
        [{"label": l, "dt_seconds": d, "product_type": pt, "metrics": r}
         for l, d, pt, r in results], indent=2))
    print(f"\nfull results → {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
