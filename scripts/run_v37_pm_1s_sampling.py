#!/usr/bin/env python3
"""v3.7 SPIKE — companion to run_v37_hl_1s_sampling.py: PM cadence on the
BBO-overlap window.

Runs the v3.6 universal config on PM BTC binaries that resolve in the
Binance-perp BBO tick window (2026-05-06 → 2026-05-28). Sweeps
vol_sampling_dt_seconds ∈ {60, 10, 5, 1} using the new `binance_bbo`
reference source. Also runs a kline-reference baseline at dt=60 for sanity
(no cadence effect there — klines are native 1m).

Important: this corpus is ~18 markets, much smaller than the 12-month
walk-forward (300 markets) we tune on. Treat results as suggestive, not
load-bearing. The 12-month tune still uses the kline path.
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
OUT_ROOT = WORKTREE / "data" / "sim" / "runs" / "v3-7-pm-1s-sampling-2026-05-28"

# v3.6 universal PM config (jr-tilt-ma_sigma-lb30-a100) — PM anchor from
# scripts/run_v36_sweep.py:ANCHOR + momentum_mr block from the summary.
BASE = {
    "vol_lookback_seconds": 3600,
    "vol_sampling_dt_seconds": 60,
    "vol_clip_min": 0.0,
    "vol_clip_max": 5.0,
    "fee_taker": 0.0,
    "half_spread_assumption": 0.005,
    "drift_lookback_seconds": 3600,
    "drift_blend": 0.0,
    "edge_max": None,
    "min_distance_pct": None,
    "topup_enabled": False,
    "min_bid_notional_usd": 10.0,
    "favorite_threshold": 0.90,
    "edge_buffer": 0.03,
    "exit_safety_d": 1.0,
    "exit_take_profit_mode": True,
    "exit_fee": 0.0007,
    "tte_min_seconds": 0,
    "tte_max_seconds": 86400,
    "max_position_usd": 200.0,
    "stop_loss_pct": None,
    "exit_edge_threshold": 0.0,
    "take_profit_price": None,
    "time_stop_seconds": 0,
    "gamma_lambda": 0.0,
    "momentum_mr_enabled": True,
    "momentum_mr_indicator": "ma_sigma",
    "momentum_mr_mode": "tilt",
    "momentum_mr_lookback_min": 30,
    "momentum_mr_alpha_tilt": 1.0,
    "momentum_mr_jr_trust_weight": True,
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
        "--strategy", "v3_5_momentum_mr",
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
        # Sanity: kline baseline at dt=60 on this short window.
        ("klines_dt60", 60, "klines"),
        # BBO sweep
        ("bbo_dt60", 60, "binance_bbo"),
        ("bbo_dt10", 10, "binance_bbo"),
        ("bbo_dt5", 5, "binance_bbo"),
        ("bbo_dt1", 1, "binance_bbo"),
    ]
    results: list[tuple[str, int, str, dict]] = []
    for label, dt_s, rs in cells:
        r = run_cell(label, dt_s, rs)
        results.append((label, dt_s, rs, r))

    print()
    print(
        f"{'cell':>14} {'dt(s)':>6} {'ref':>11}  {'PnL':>9} {'trades':>7} {'hit':>7} {'sharpe':>7} {'maxDD':>9}"
    )
    for label, dt_s, rs, r in results:
        print(
            f"{label:>14} {dt_s:>6} {rs:>11}  ${r.get('total_pnl_usd', 0):>7.2f} "
            f"{r.get('n_trades', 0):>7} {r.get('hit_rate', 0):>6.1%} "
            f"{r.get('sharpe', 0):>7.2f} ${r.get('max_drawdown_usd', 0):>7.2f}"
        )

    out_json = OUT_ROOT / "summary.json"
    out_json.write_text(
        json.dumps(
            [{"label": l, "dt_seconds": d, "ref_source": rs, "metrics": r}
             for l, d, rs, r in results],
            indent=2,
        )
    )
    print(f"\nfull results → {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
