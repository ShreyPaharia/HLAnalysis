"""Aggregate a walk-forward tuning sweep across splits.

Reads results.jsonl (one row per (params, split)) and groups by params, then
reports per-config aggregates:
  - n_splits
  - profitable_splits (= count of splits with positive total_pnl_usd)
  - total_pnl_usd      (sum across splits)
  - mean_sharpe        (mean across splits)
  - mean_hit_rate
  - n_trades_total
  - max_drawdown_usd   (max across splits)

By default sorts by (profitable_splits desc, total_pnl_usd desc). Useful for
size_cap sweeps where the goal is "doesn't break in any split AND adds PnL".

Usage:
    uv run python scripts/analyze_sweep.py --run-dir data/sim/tuning/v1-size-cap-pm \
        [--sort pnl|profitable|sharpe] [--top 20] \
        [--baseline 'size_cap_near_strike_pct=0.0']
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _params_key(p: dict) -> tuple:
    return tuple(sorted(p.items()))


def _params_human(p: dict, only: list[str] | None = None) -> str:
    keys = only or sorted(p.keys())
    return " ".join(f"{k}={p[k]}" for k in keys if k in p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sort", choices=["pnl", "profitable", "sharpe"], default="profitable")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument(
        "--focus",
        nargs="*",
        default=None,
        help="Param keys to show in per-row dump (omit = all keys).",
    )
    ap.add_argument(
        "--baseline-keys",
        nargs="*",
        default=None,
        help="If set, treat rows matching baseline_key=value pairs as baseline and "
        "highlight delta vs them.",
    )
    args = ap.parse_args()

    rd = Path(args.run_dir)
    rows = [
        json.loads(l)
        for l in (rd / "results.jsonl").read_text().splitlines()
        if l.strip()
    ]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[_params_key(r["params"])].append(r)

    aggs: list[dict] = []
    for key, rs in groups.items():
        params = dict(key)
        pnls = [r["summary"]["total_pnl_usd"] for r in rs]
        sharpes = [r["summary"]["sharpe"] for r in rs]
        hits = [r["summary"]["hit_rate"] for r in rs]
        trades = [r["summary"]["n_trades"] for r in rs]
        dds = [r["summary"]["max_drawdown_usd"] for r in rs]
        aggs.append({
            "params": params,
            "n_splits": len(rs),
            "profitable_splits": sum(1 for p in pnls if p > 0),
            "total_pnl_usd": sum(pnls),
            "mean_sharpe": sum(sharpes) / len(sharpes) if sharpes else 0.0,
            "mean_hit_rate": sum(hits) / len(hits) if hits else 0.0,
            "n_trades_total": sum(trades),
            "max_drawdown_usd": max(dds) if dds else 0.0,
            "per_split_pnl": pnls,
        })

    if args.sort == "pnl":
        aggs.sort(key=lambda a: a["total_pnl_usd"], reverse=True)
    elif args.sort == "sharpe":
        aggs.sort(key=lambda a: a["mean_sharpe"], reverse=True)
    else:
        aggs.sort(key=lambda a: (a["profitable_splits"], a["total_pnl_usd"]), reverse=True)

    # Identify baseline (first matching row from baseline-keys filter).
    baseline_pnl: float | None = None
    if args.baseline_keys:
        kv = {}
        for token in args.baseline_keys:
            k, _, v = token.partition("=")
            try:
                kv[k] = float(v) if "." in v or v.replace("-", "").isdigit() else v
            except ValueError:
                kv[k] = v
        for a in aggs:
            if all(a["params"].get(k) == v for k, v in kv.items()):
                baseline_pnl = a["total_pnl_usd"]
                break

    print(f"\n=== Sweep aggregates: {len(aggs)} unique configs, sort={args.sort} ===")
    if baseline_pnl is not None:
        print(f"baseline (matched first {args.baseline_keys}): total_pnl_usd=${baseline_pnl:.2f}")
    print()
    hdr = (
        f"{'#':>3}  {'prof':>4}  {'splits':>6}  {'total$':>10}  {'Δbase':>9}  "
        f"{'sharpe':>7}  {'hit':>5}  {'trades':>6}  {'maxDD':>8}   params"
    )
    print(hdr)
    print("-" * len(hdr))
    for i, a in enumerate(aggs[: args.top]):
        delta = (
            f"${a['total_pnl_usd'] - baseline_pnl:+.2f}"
            if baseline_pnl is not None
            else "n/a"
        )
        print(
            f"{i+1:>3}  {a['profitable_splits']:>4}  {a['n_splits']:>6}  "
            f"${a['total_pnl_usd']:>9.2f}  {delta:>9}  "
            f"{a['mean_sharpe']:>7.3f}  {a['mean_hit_rate']:>5.1%}  "
            f"{a['n_trades_total']:>6}  ${a['max_drawdown_usd']:>7.2f}   "
            f"{_params_human(a['params'], args.focus)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
