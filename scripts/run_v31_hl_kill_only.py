#!/usr/bin/env python3
"""HL kill-only ablation: keep prod fav/edge_buffer, ablate one dead-weight
knob at a time. Confirms each kill is safe on HL."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

REPO_ROOT = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis")
WORKTREE = REPO_ROOT / ".worktrees/do-an-analysis-of-the-final-strategies-and-see-if--53OYTm"
DATA_ROOT = REPO_ROOT / "data"
OUT_ROOT = REPO_ROOT / "data/sim/runs/v3-1-hl-kill-only-2026-05-23"

PROD_HL = {
    "vol_lookback_seconds": 3600,
    "vol_sampling_dt_seconds": 60,
    "vol_clip_min": 0.05,
    "vol_clip_max": 5.0,
    "edge_buffer": 0.02,
    "fee_taker": 0.0,
    "half_spread_assumption": 0.005,
    "drift_lookback_seconds": 3600,
    "drift_blend": 0.0,
    "favorite_threshold": 0.85,
    "tte_min_seconds": 0,
    "tte_max_seconds": 43200,
    "edge_max": 0.20,
    "min_distance_pct": None,
    "min_bid_notional_usd": 0.0,
    "max_position_usd": 200.0,
    "stop_loss_pct": None,
    "exit_edge_threshold": 0.0,
    "take_profit_price": None,
    "time_stop_seconds": 0,
    "gamma_lambda": 0.0,
    "exit_take_profit_mode": True,
    "exit_fee": 0.0007,
    "exit_safety_d": 1.0,
}

KILLS = {
    "prod":               PROD_HL,
    "no_edge_max":        {**PROD_HL, "edge_max": None},
    "no_vol_clip_min":    {**PROD_HL, "vol_clip_min": 0.0},
    "no_topup":           {**PROD_HL, "topup_enabled": False},
    "all_kills":          {**PROD_HL, "edge_max": None, "vol_clip_min": 0.0, "topup_enabled": False},
}


def parse_report(path: Path) -> dict:
    import re
    text = path.read_text()
    def find(pat, cast=float, default=0):
        m = re.search(pat, text)
        if not m: return default
        return cast(m.group(1).replace(',', ''))
    return {
        'n_markets': find(r'questions:\s+(\d+)', int),
        'n_trades': find(r'trades:\s+(\d+)', int),
        'total_pnl_usd': find(r'total PnL:\s+\$([-\d.,]+)'),
        'sharpe': find(r'Sharpe[^:]*:\s+([-\d.]+)'),
        'hit_rate': find(r'hit rate:\s+([\d.]+)%') / 100.0,
        'max_drawdown_usd': find(r'max drawdown:\s+\$([-\d.,]+)'),
    }


def run_config(name: str, params: dict, kind: str) -> dict:
    out_dir = OUT_ROOT / f"{name}_{kind}"
    cfg_path = OUT_ROOT / f"cfg_{name}_{kind}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(params, indent=2))
    report = out_dir / "report.md"
    if report.exists() and report.stat().st_size > 100:
        return parse_report(report)
    cmd = [
        "uv", "run", "hl-bt", "run",
        "--strategy", "v3_theta_harvester",
        "--data-source", "hl_hip4",
        "--config", str(cfg_path),
        "--out-dir", str(out_dir),
        "--start", "2026-05-01", "--end", "2026-05-25",
        "--kind", kind,
    ]
    env = {**os.environ, "HLBT_HL_DATA_ROOT": str(DATA_ROOT)}
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=WORKTREE, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"[FAIL {name}/{kind}] rc={proc.returncode}\n{proc.stderr[-1000:]}")
        return {}
    r = parse_report(report)
    print(f"[ok] {name}/{kind} in {dt:.1f}s : PnL=${r.get('total_pnl_usd',0):.2f} trades={r.get('n_trades',0)}")
    return r


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    for kind in ("binary", "bucket"):
        for name, params in KILLS.items():
            results[(name, kind)] = run_config(name, params, kind)
    print()
    print(f"{'config':>16} {'kind':>7} {'PnL':>10} {'Sharpe':>7} {'maxDD':>8} {'trades':>8} {'hit':>7}")
    for (name, kind), r in results.items():
        print(f"{name:>16} {kind:>7} ${r.get('total_pnl_usd',0):>8.2f} {r.get('sharpe',0):>7.2f} ${r.get('max_drawdown_usd',0):>6.2f} {r.get('n_trades',0):>8} {r.get('hit_rate',0):>6.1%}")


if __name__ == "__main__":
    main()
