#!/usr/bin/env python3
"""Aggregate tuning results.jsonl by params (across splits) and print table."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


def main(run_dir: str, group_keys: list[str]) -> None:
    p = Path(run_dir) / "results.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = tuple(r["params"].get(k) for k in group_keys)
        buckets[key].append(r)
    print("|" + " | ".join(group_keys) + " | n_splits | sum_pnl | mean_pnl | mean_sharpe | mean_hit | mean_dd |")
    print("|" + "|".join(["---"] * (len(group_keys) + 6)) + "|")
    out = []
    for key, items in buckets.items():
        s = [it["summary"] for it in items]
        total = sum(x["total_pnl_usd"] for x in s)
        mp = total / len(s)
        msh = mean(x["sharpe"] for x in s)
        mhit = mean(x["hit_rate"] for x in s)
        mdd = mean(x["max_drawdown_usd"] for x in s)
        out.append((key, len(items), total, mp, msh, mhit, mdd))
    out.sort(key=lambda x: -x[2])  # by sum_pnl desc
    for key, n, total, mp, msh, mhit, mdd in out:
        keystr = " | ".join(str(v) for v in key)
        print(f"| {keystr} | {n} | {total:.2f} | {mp:.2f} | {msh:.3f} | {mhit:.2%} | {mdd:.2f} |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","))
