"""Read tuning results.jsonl and print the best param tuples by OOS Sharpe.

Usage: uv run python scripts/best_params.py <run_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: best_params.py <run_dir>", file=sys.stderr)
        sys.exit(2)
    run_dir = Path(sys.argv[1])
    log = run_dir / "results.jsonl"
    if not log.exists():
        print(f"no results.jsonl at {log}", file=sys.stderr)
        sys.exit(2)

    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    if not rows:
        print("empty results", file=sys.stderr)
        sys.exit(2)

    # Aggregate per param tuple across walk-forward splits: pool per-market PnL
    # across splits to get a clean OOS Sharpe and total PnL per tuple.
    by_params: dict[str, dict] = {}
    for r in rows:
        key = json.dumps(r["params"], sort_keys=True)
        bucket = by_params.setdefault(key, {"params": r["params"], "splits": []})
        bucket["splits"].append(r)

    aggregated = []
    for key, bucket in by_params.items():
        pnls = [s["summary"]["total_pnl_usd"] for s in bucket["splits"]]
        n_trades = sum(s["summary"]["n_trades"] for s in bucket["splits"])
        n_markets = sum(s["summary"]["n_markets"] for s in bucket["splits"])
        # Sharpe of per-split totals (proxy: across-split stability)
        if len(pnls) >= 2:
            mu = sum(pnls) / len(pnls)
            var = sum((p - mu) ** 2 for p in pnls) / (len(pnls) - 1)
            split_sharpe = (mu / (var ** 0.5)) if var > 0 else 0.0
        else:
            split_sharpe = bucket["splits"][0]["summary"]["sharpe"]
        avg_sharpe = sum(s["summary"]["sharpe"] for s in bucket["splits"]) / len(bucket["splits"])
        aggregated.append({
            "params": bucket["params"],
            "n_splits": len(bucket["splits"]),
            "total_pnl_usd": sum(pnls),
            "avg_sharpe": avg_sharpe,
            "split_sharpe": split_sharpe,
            "n_trades": n_trades,
            "n_markets": n_markets,
        })

    aggregated.sort(key=lambda x: x["avg_sharpe"], reverse=True)

    print(f"# Tuning aggregated results — {len(aggregated)} param tuples × {aggregated[0]['n_splits']} splits each\n")
    print("## Top 10 by average OOS Sharpe across splits\n")
    for i, a in enumerate(aggregated[:10], 1):
        print(f"### #{i}  avg_sharpe={a['avg_sharpe']:.3f}  total_pnl=${a['total_pnl_usd']:.2f}  n_trades={a['n_trades']}  n_markets={a['n_markets']}")
        print(f"params: {json.dumps(a['params'])}")
        print()

    print("## Top 10 by total PnL\n")
    for i, a in enumerate(sorted(aggregated, key=lambda x: x["total_pnl_usd"], reverse=True)[:10], 1):
        print(f"### #{i}  total_pnl=${a['total_pnl_usd']:.2f}  avg_sharpe={a['avg_sharpe']:.3f}  n_trades={a['n_trades']}")
        print(f"params: {json.dumps(a['params'])}")
        print()


if __name__ == "__main__":
    main()
