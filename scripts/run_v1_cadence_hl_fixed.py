#!/usr/bin/env python3
"""v1 (late_resolution) HL HIP-4 cadence sweep — CADENCE-AWARE code.

Mirrors scripts/run_v37_hl_1s_sampling.py but for the v1 strategy, to answer
whether v1 should move to sub-minute reference sampling in lockstep with v31
(theta_harvester). Both live slots read reference_symbol="BTC" and share one
bucketed mark history, so the engine refuses to start if they disagree on
vol_sampling_dt_seconds. This sweep validates v1 first.

Config = v1 HL prod binary slot (config/strategy.yaml `late_resolution`):
fav=0.85 / extreme_max=0.99 / min_safety_d=1.0 / exit_safety_d=1.0 /
vol_ewma_lambda=0.85 / size_cap / bid-gate. Held FIXED; only
`vol_sampling_dt_seconds` is swept {60, 5, 1}. The CLI auto-couples the loader's
reference-resample period to that value (cli.py: hl_reference_resample_seconds).

NOTE: v1 has NO cadence-aware sigma scaling. LateResolutionConfig has no
vol_sampling_dt_seconds field; the strategy hardcodes a 60s bar assumption in
two places (n_keep = vol_lookback_seconds // 60; tte_min = tte_s / 60.0). So
this sweep measures what happens to v1 when fed denser reference bars while its
sigma math still assumes 60s spacing — exactly the live behaviour if the loader
cadence were flipped without porting v1's sigma math. See the writeup.
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
OUT_ROOT = WORKTREE / "data" / "sim" / "runs" / "v1-cadence-hl-fixed-2026-05-30"

# v1 HL prod binary slot (config/strategy.yaml `late_resolution` priceBinary).
BASE = {
    "tte_min_seconds": 0,
    "tte_max_seconds": 7200,
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
    "use_bid_for_entry_gate": True,
    "min_bid_notional_usd": 25.0,
    # Loader reference-resample period (CLI couples this to the loader). v1's
    # strategy sigma math does NOT read this — see module docstring.
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


def run_cell(label: str, dt_s: int) -> dict:
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
        f"hit={r.get('hit_rate', 0):.1%} maxDD=${r.get('max_drawdown_usd', 0):.2f}"
    )
    return r


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cells = [("dt60", 60), ("dt5", 5), ("dt1", 1)]
    results: list[tuple[str, int, dict]] = []
    for label, dt_s in cells:
        results.append((label, dt_s, run_cell(label, dt_s)))

    print()
    print(f"{'cell':>8} {'dt(s)':>6} {'PnL':>9} {'trades':>7} {'hit':>7} {'sharpe':>7} {'maxDD':>9}")
    for label, dt_s, r in results:
        print(
            f"{label:>8} {dt_s:>6} ${r.get('total_pnl_usd', 0):>7.2f} "
            f"{r.get('n_trades', 0):>7} {r.get('hit_rate', 0):>6.1%} "
            f"{r.get('sharpe', 0):>7.2f} ${r.get('max_drawdown_usd', 0):>7.2f}"
        )

    out_json = OUT_ROOT / "summary.json"
    out_json.write_text(json.dumps(
        [{"label": l, "dt_seconds": d, "metrics": r} for l, d, r in results], indent=2))
    print(f"\nfull results → {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
