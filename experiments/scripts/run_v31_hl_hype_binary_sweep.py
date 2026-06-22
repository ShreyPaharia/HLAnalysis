#!/usr/bin/env python3
"""First walk-forward sweep of v31/theta on HL HIP-4 **HYPE** priceBinary markets.

Context (2026-06-22): the recorder began ingesting HYPE perps + above/below binary
outcome markets ~2026-06-15 (record-only, no live HYPE slot). This establishes a
FIRST tuned HYPE-binary characterisation. HYPE is NOT on Binance, so the reference
price/σ MUST come from HL's own perp (ref_source=hl_perp, the CLI default) — there
is no binance feed for it.

CRITICAL: the settled HYPE binary corpus is tiny (n=6, expiries 2026-06-16 → 06-21).
A 6-question walk-forward split is statistically meaningless; this sweep is
EXPLORATORY characterisation, NOT a promotable tune. The underpowered n IS the
headline. State n + span prominently in the writeup.

Anchor = the current live BTC-tuned HL binary theta block (config/strategy.yaml as
of 2026-06-22: msd2.5/esd1.5, vlb900, dt5, fav0.85, eb0.02, tte43200, max_pos500,
topup on). Params almost certainly do NOT transfer across underlyings — this tunes
HYPE independently and compares to the anchor.

Harness mirrors run_v31_hl_safetyd_sweep.py (ThreadPool over `hl-bt run`, parse
report.md, walk-forward early/late split, rank by best WORST-HALF PnL), with
`--underlying HYPE` added (requires the SourceConfig.discover_underlying fix so
spawn/serial workers re-map non-BTC question_ids).

DO NOT change live config / DO NOT deploy. Analysis + recommendation only.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

REPO_ROOT = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis")
WORKTREE = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
OUT_ROOT = REPO_ROOT / "data/sim/runs/v31-hl-hype-binary-2026-06-22"
OUTER_WORKERS, RUN_WORKERS = 3, 4
UNDERLYING = "HYPE"

# 3/3 walk-forward split. Expiries: 06-16,17,18 (early) | 06-19,20,21 (late).
WINDOWS = {
    "full":  ("2026-06-15", "2026-06-22"),
    "early": ("2026-06-15", "2026-06-19"),
    "late":  ("2026-06-19", "2026-06-22"),
}

# Current live HL BINARY theta block (config/strategy.yaml, 2026-06-22). This IS
# the anchor / reference cell. NB: the goal-stated anchor (vlb1800/fav0.80/eb0.01)
# differs slightly from the real committed config (vlb900/fav0.85/eb0.02) — we
# anchor on the actual config (source of truth) and note the discrepancy.
PROD = {
    "vol_lookback_seconds": 900, "vol_sampling_dt_seconds": 5, "vol_clip_min": 0.0,
    "vol_clip_max": 5.0, "edge_buffer": 0.02, "fee_taker": 0.0,
    "half_spread_assumption": 0.005, "drift_lookback_seconds": 3600, "drift_blend": 0.0,
    "favorite_threshold": 0.85, "tte_min_seconds": 0, "tte_max_seconds": 43200,
    "edge_max": None, "min_distance_pct": None, "min_bid_notional_usd": 0.0,
    "max_position_usd": 500.0, "stop_loss_pct": None, "exit_edge_threshold": 0.0,
    "time_stop_seconds": 0, "exit_take_profit_mode": True, "exit_fee": 0.0007,
    "exit_safety_d": 1.5, "min_safety_d": 2.5,
    "topup_enabled": True, "topup_threshold_pct": 0.2, "topup_min_notional_usd": 11.0,
}

# Safety band 2D grid (the primary axis, directly comparable to the BTC band sweep).
EXIT_SD = [0.0, 1.0, 1.5, 2.0]
MIN_SD = [0.0, 1.5, 2.0, 2.5, 3.0]

# One-axis-at-a-time sweeps around the anchor (other knobs frozen at PROD).
OAT = {
    "favorite_threshold": [0.75, 0.80, 0.90],
    "edge_buffer": [0.0, 0.01, 0.03],
    "vol_lookback_seconds": [1800, 3600],
    "vol_sampling_dt_seconds": [60],
    "tte_max_seconds": [21600, 86400],
}


def make_configs() -> dict[str, dict]:
    cfgs: dict[str, dict] = {"prod_ref": dict(PROD)}
    for msd, esd in product(MIN_SD, EXIT_SD):
        cfgs[f"msd{msd}_esd{esd}"] = {**PROD, "min_safety_d": msd, "exit_safety_d": esd}
    for axis, vals in OAT.items():
        for v in vals:
            cfgs[f"{axis}={v}"] = {**PROD, axis: v}
    return cfgs


def parse_report(path: Path) -> dict:
    text = path.read_text()
    def find(pat, cast=float, default=0):
        m = re.search(pat, text)
        return cast(m.group(1).replace(',', '')) if m else default
    return {
        'n_markets': find(r'questions:\s+(\d+)', int),
        'n_trades': find(r'trades:\s+(\d+)', int),
        'total_pnl_usd': find(r'total PnL:\s+\$([-\d.,]+)'),
        'sharpe': find(r'Sharpe[^:]*:\s+([-\d.]+)'),
        'hit_rate': find(r'hit rate:\s+([\d.]+)%') / 100.0,
        'max_drawdown_usd': find(r'max drawdown:\s+\$([-\d.,]+)'),
    }


def run_one(job: dict) -> dict:
    label, window, params = job["label"], job["window"], job["params"]
    name = f"{label}__{window}".replace("/", "_")
    cfg_path = OUT_ROOT / f"cfg_{name}.json"
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(params, indent=2))
    rpt = out_dir / "report.md"
    if not (rpt.exists() and rpt.stat().st_size > 100):
        start, end = WINDOWS[window]
        cmd = ["uv", "run", "hl-bt", "run", "--strategy", "v3_theta_harvester",
               "--data-source", "hl_hip4", "--config", str(cfg_path),
               "--out-dir", str(out_dir), "--start", start, "--end", end,
               "--kind", "binary", "--underlying", UNDERLYING,
               "--workers", str(RUN_WORKERS)]
        env = {**os.environ, "HLBT_HL_DATA_ROOT": str(DATA_ROOT)}
        t0 = time.time()
        p = subprocess.run(cmd, cwd=WORKTREE, capture_output=True, text=True, env=env)
        if p.returncode != 0:
            print(f"[FAIL {name}] {p.stderr.strip()[-300:]}")
            return {"label": label, "window": window}
        print(f"[ok {name:36}] {time.time()-t0:5.0f}s")
    return {"label": label, "window": window, **(parse_report(rpt) if rpt.exists() else {})}


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cfgs = make_configs()
    jobs = [{"label": l, "window": w, "params": p}
            for l, p in cfgs.items() for w in WINDOWS]
    print(f"{len(cfgs)} configs x {len(WINDOWS)} windows = {len(jobs)} runs "
          f"(outer={OUTER_WORKERS}, run_workers={RUN_WORKERS}) underlying={UNDERLYING}")
    results = []
    with ThreadPoolExecutor(max_workers=OUTER_WORKERS) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for f in as_completed(futs):
            results.append(f.result())
    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2))

    idx = {(r["label"], r["window"]): r for r in results}
    def pnl(l, w): return idx.get((l, w), {}).get("total_pnl_usd", float("nan"))

    def mkrow(label):
        rf = idx.get((label, "full"), {})
        pe, pl = pnl(label, "early"), pnl(label, "late")
        return {"label": label, "full": rf.get("total_pnl_usd", float("nan")),
                "sharpe": rf.get("sharpe", 0), "dd": rf.get("max_drawdown_usd", 0),
                "trades": rf.get("n_trades", 0), "hit": rf.get("hit_rate", 0),
                "early": pe, "late": pl, "worst_half": min(pe, pl)}

    def fmt_row(r):
        return (f"{r['label']:>22} ${r['full']:>8.2f} {r['sharpe']:>7.2f} ${r['dd']:>7.2f} "
                f"${r['early']:>8.2f} ${r['late']:>8.2f} ${r['worst_half']:>8.2f} "
                f"{r['trades']:>5} {r['hit']:>4.0%}")

    hdr = (f"{'config':>22} {'full':>9} {'Shrp':>7} {'maxDD':>8} "
           f"{'early':>9} {'late':>9} {'worst':>9} {'trd':>5} {'hit':>5}")

    pr = mkrow("prod_ref")
    all_labels = [l for l in cfgs if l != "prod_ref"]
    rows = sorted((mkrow(l) for l in all_labels), key=lambda r: r["worst_half"], reverse=True)
    print("\n" + "=" * 104)
    print(f"ANCHOR prod_ref (BTC-tuned msd2.5/esd1.5 vlb900 dt5 fav0.85 eb0.02): "
          f"full=${pr['full']:.2f} worst-half=${pr['worst_half']:.2f} maxDD=${pr['dd']:.2f} "
          f"trd={pr['trades']} hit={pr['hit']:.0%}")
    print("=" * 104)
    print("FULL SWEEP (sorted by WORST-HALF PnL) — n=6, EXPLORATORY ONLY")
    print("-" * 104)
    print(hdr)
    print(fmt_row(pr))
    print("-" * 104)
    for r in rows:
        print(fmt_row(r))
    print("=" * 104)

    print("\nWORST-HALF PnL matrix  (rows=min_safety_d, cols=exit_safety_d)")
    print("        " + "".join(f"esd{e:>7}" for e in EXIT_SD))
    cell = {r["label"]: r for r in rows}
    for msd in MIN_SD:
        line = f"msd{msd:<5}"
        for esd in EXIT_SD:
            v = cell.get(f"msd{msd}_esd{esd}", {}).get("worst_half", float("nan"))
            line += f"{v:>10.0f}"
        print(line)

    best = rows[0]
    print(f"\nBEST worst-half: {best['label']}  full=${best['full']:.2f} "
          f"early=${best['early']:.2f} late=${best['late']:.2f} "
          f"worst=${best['worst_half']:.2f} Sharpe={best['sharpe']:.2f} "
          f"maxDD=${best['dd']:.2f} trd={best['trades']} hit={best['hit']:.0%}")
    print(f"results: {OUT_ROOT}")


if __name__ == "__main__":
    main()
