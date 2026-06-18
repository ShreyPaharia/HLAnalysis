#!/usr/bin/env python3
"""Aggregate a resumable_run bucket sweep into the baseline-vs-hold_msd2 report.

Reads each per-(config, question) report dir under ``--out-base`` and computes,
per config: total PnL, annualized Sharpe, hit rate, max drawdown, trade count,
the full per-question PnL list incl. the worst-5 tail, and the churn
(entries=buy fills / exits=sell fills per question). Sharpe/hit/maxDD reuse the
repo's ``summarise_run`` so the numbers match ``hl-bt`` conventions.

    python scripts/perf/aggregate_bucket_sweep.py --out-base /tmp/bucket_sweep/out
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

from hlanalysis.backtest.runner.result import summarise_run

_PNL_RE = re.compile(r"total PnL:\s*\$?(-?[\d,]+\.?\d*)")


def _pnl(report_md: Path) -> float | None:
    if not report_md.exists():
        return None
    for line in report_md.read_text().splitlines():
        m = _PNL_RE.search(line)
        if m:
            return float(m.group(1).replace(",", ""))
    return None


def _fill_sides(fills_parquet: Path) -> tuple[int, int]:
    """(n_entries, n_exits) = (buy fills, sell fills), excluding hedge legs."""
    if not fills_parquet.exists():
        return 0, 0
    t = pq.read_table(fills_parquet)
    if t.num_rows == 0:
        return 0, 0
    df = t.to_pandas()
    if "is_hedge" in df.columns:
        df = df[~df["is_hedge"].fillna(False)]
    buys = int((df["side"] == "buy").sum())
    sells = int((df["side"] == "sell").sum())
    return buys, sells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-base", required=True)
    args = ap.parse_args()
    out_base = Path(args.out_base)
    cfg_ids = [c["id"] for c in json.loads((out_base / "_configs.json").read_text())]

    for cid in cfg_ids:
        cdir = out_base / cid
        rows = []
        for qd in sorted(cdir.glob("q*")):
            pnl = _pnl(qd / "report.md")
            if pnl is None:
                continue
            ent, ext = _fill_sides(qd / "fills.parquet")
            rows.append((qd.name, pnl, ent, ext))
        pnls = [r[1] for r in rows]
        n_entries = sum(r[2] for r in rows)
        n_exits = sum(r[3] for r in rows)
        s = summarise_run(pnls, n_entries + n_exits)
        worst5 = sorted(rows, key=lambda r: r[1])[:5]
        n = len(rows) or 1
        print(f"\n===== {cid}  (n={len(rows)} questions) =====")
        print(f"  total PnL     : ${s.total_pnl_usd:,.2f}")
        print(f"  Sharpe (ann)  : {s.sharpe:.2f}")
        print(f"  hit rate      : {s.hit_rate * 100:.1f}%")
        print(f"  max drawdown  : ${s.max_drawdown_usd:,.2f}")
        print(f"  fills (trades): {n_entries + n_exits}  (entries={n_entries} exits={n_exits})")
        print(f"  churn / q     : entries={n_entries / n:.2f}  exits={n_exits / n:.2f}")
        print("  worst-5 questions:")
        for qid, pnl, ent, ext in worst5:
            print(f"    {qid}: ${pnl:>9.2f}  (entries={ent} exits={ext})")
        print(f"  per-question PnL: {[round(p, 2) for _, p, _, _ in rows]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
