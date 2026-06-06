#!/usr/bin/env python3
"""v1 (late_resolution) PM BTC Up/Down reference-sampling cadence sweep.

Companion to scripts/run_v1_cadence_hl.py. Mirrors run_v37_pm_1s_sampling.py:
sub-minute cadence on PM can only be measured via the `binance_bbo` reference
source (klines are native 1m and carry no sub-minute info). Sweeps
vol_sampling_dt_seconds ∈ {60, 5, 1} on the BBO overlap window plus a
klines_dt60 sanity baseline.

Config = v1 PM prod slot (config/strategy.yaml `late_resolution` PM block,
fee_model pm_binary / fee_rate 0.07). Held FIXED; only the loader cadence varies.

NOTE: v1 has NO cadence-aware sigma scaling (see run_v1_cadence_hl.py docstring).
Corpus is ~18-22 markets (BBO window) — suggestive, not load-bearing, per the
v3.7 PM companion sweep caveats.
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
OUT_ROOT = WORKTREE / "data" / "sim" / "runs" / "v1-cadence-pm-2026-05-29"

# v1 PM prod slot (config/strategy.yaml `late_resolution` PM, priceBinary).
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


def run_cell(label: str, dt_s: int, ref_source: str) -> dict:
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
        "--pm-reference-source", ref_source,
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
        f"[ok] {label} ({dt:.1f}s, ref={ref_source}) : "
        f"PnL=${r.get('total_pnl_usd', 0):.2f} trades={r.get('n_trades', 0)} "
        f"hit={r.get('hit_rate', 0):.1%} maxDD=${r.get('max_drawdown_usd', 0):.2f}"
    )
    return r


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cells = [
        ("klines_dt60", 60, "klines"),
        ("bbo_dt60", 60, "binance_bbo"),
        ("bbo_dt5", 5, "binance_bbo"),
        ("bbo_dt1", 1, "binance_bbo"),
    ]
    results: list[tuple[str, int, str, dict]] = []
    for label, dt_s, rs in cells:
        results.append((label, dt_s, rs, run_cell(label, dt_s, rs)))

    print()
    print(f"{'cell':>14} {'dt(s)':>6} {'ref':>11}  {'PnL':>9} {'trades':>7} {'hit':>7} {'sharpe':>7} {'maxDD':>9}")
    for label, dt_s, rs, r in results:
        print(
            f"{label:>14} {dt_s:>6} {rs:>11}  ${r.get('total_pnl_usd', 0):>7.2f} "
            f"{r.get('n_trades', 0):>7} {r.get('hit_rate', 0):>6.1%} "
            f"{r.get('sharpe', 0):>7.2f} ${r.get('max_drawdown_usd', 0):>7.2f}"
        )

    out_json = OUT_ROOT / "summary.json"
    out_json.write_text(json.dumps(
        [{"label": l, "dt_seconds": d, "ref_source": rs, "metrics": r}
         for l, d, rs, r in results], indent=2))
    print(f"\nfull results → {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
