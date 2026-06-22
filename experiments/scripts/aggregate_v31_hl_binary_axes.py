#!/usr/bin/env python3
"""Aggregate the resumable-driver per-(config,question) outputs of the v31 HL
binary axes sweep into a walk-forward ranked table.

The driver (scripts/perf/resumable_run.py) ran each question in ISOLATION
(--max-markets 1, no cross-market caps) under --start 2026-05-10. σ warmup is a
trailing lookback (vol_lookback ≤ 3600s) taken before each question opens, so it
is independent of the window start given ample lead — therefore a question's
per-question PnL is IDENTICAL whether computed in the full run or a separate
early/late window run. We exploit that: run the FULL window once, then split the
per-question PnL by question open date (< 2026-05-31 → early, ≥ → late) to get the
walk-forward halves EXACTLY (validated below against the 2026-06-22 separate-run
numbers: prod_ref early $543.16 / late $177.64).

Metrics reuse summarise_run so Sharpe/maxDD/hit match the monolithic report to the
cent (validated: prod_ref full $720.83 vs monolithic $720.80). Ranks every cell by
WORST-HALF PnL — the overfit-resistant walk-forward criterion. Also concatenates
per-config full-window fills for the tail-stress script.

Analysis only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from hlanalysis.backtest.runner.result import summarise_run

BASE = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data/sim/runs/v31-axes-driver")
FULL = BASE / "full"
# Authoritative walk-forward split: the hl_hip4 source's own discovery for each
# window (matches the 2026-06-22 separate-run partition exactly: 21 early / 21
# late). Written by the discovery step; maps question_id -> half.
_WIN = json.loads(Path("/tmp/window_ids.json").read_text())
EARLY_IDS = set(_WIN["early"])
LATE_IDS = set(_WIN["late"])

# expected from the 2026-06-22 separate-window safetyd sweep (prod_ref = live band)
EXPECT_PRODREF = {"full": 720.80, "early": 543.16, "late": 177.64, "maxdd": 198.76, "sharpe": 7.200, "trades": 329}


def parse_q(report: Path) -> tuple[float, int]:
    pnl = trades = 0
    for line in report.read_text().splitlines():
        if line.startswith("- total PnL:"):
            pnl = float(line.split("$", 1)[1].replace(",", ""))
        elif line.strip().startswith("- trades:"):
            trades = int(line.split(":", 1)[1].strip().replace(",", ""))
    return pnl, trades


def q_question_id(qdir: Path) -> str:
    """The question_id for this per-question dir, from fills/<Qid>.parquet (or the
    diagnostics sibling). The driver names the inner parquet by question_id."""
    for sub in ("fills", "diagnostics"):
        files = list((qdir / sub).glob("Q*.parquet"))
        if files:
            return files[0].stem
    return ""


def config_rows(config_id: str) -> dict:
    cfg_dir = FULL / config_id
    qdirs = sorted(cfg_dir.glob("q[0-9]*"), key=lambda p: int(re.sub(r"\D", "", p.name)))
    per_q = []  # (q_index, qid, pnl, trades)
    for q in qdirs:
        rpt = q / "report.md"
        if not rpt.exists():
            continue
        pnl, tr = parse_q(rpt)
        per_q.append((int(re.sub(r"\D", "", q.name)), q_question_id(q), pnl, tr))
    if not per_q:
        return {}
    per_q.sort(key=lambda x: x[0])  # discovery/date order
    full_pnl = [p for _, _, p, _ in per_q]
    full_tr = sum(t for _, _, _, t in per_q)
    early = [(p, t) for _, qid, p, t in per_q if qid in EARLY_IDS]
    late = [(p, t) for _, qid, p, t in per_q if qid in LATE_IDS]
    sf = summarise_run(full_pnl, full_tr)
    se = summarise_run([p for p, _ in early], sum(t for _, t in early)) if early else None
    sl = summarise_run([p for p, _ in late], sum(t for _, t in late)) if late else None
    return {
        "config": config_id,
        "full": sf.total_pnl_usd,
        "sharpe": sf.sharpe,
        "maxdd": sf.max_drawdown_usd,
        "trades": sf.n_trades,
        "hit": sf.hit_rate,
        "n_markets": sf.n_markets,
        "early": se.total_pnl_usd if se else float("nan"),
        "late": sl.total_pnl_usd if sl else float("nan"),
        "n_early": se.n_markets if se else 0,
        "n_late": sl.n_markets if sl else 0,
        "worst_half": min(se.total_pnl_usd if se else float("nan"), sl.total_pnl_usd if sl else float("nan")),
    }


def concat_fills(config_id: str) -> None:
    cfg_dir = FULL / config_id
    if (cfg_dir / "fills.parquet").exists():
        return
    parts = [
        pd.read_parquet(q / "fills.parquet") for q in sorted(cfg_dir.glob("q[0-9]*")) if (q / "fills.parquet").exists()
    ]
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(cfg_dir / "fills.parquet")


def main():
    config_ids = sorted(p.name for p in FULL.iterdir() if p.is_dir())
    rows = [r for cid in config_ids if (r := config_rows(cid))]
    for cid in config_ids:
        concat_fills(cid)
    (BASE / "axes_aggregate.json").write_text(json.dumps(rows, indent=2))

    pr = next((r for r in rows if r["config"] == "prod_ref"), None)

    # ---- validation against 2026-06-22 separate-run numbers ----
    print("\n=== VALIDATION: prod_ref offline-split vs 2026-06-22 separate-run ===")
    if pr:
        for k, exp in EXPECT_PRODREF.items():
            got = pr[k]
            ok = abs(got - exp) <= (0.5 if k in ("full", "early", "late", "maxdd") else 0.05)
            print(
                f"  {k:7} got={got:>10.2f}  expect={exp:>10.2f}  "
                f"n_early={pr['n_early']} n_late={pr['n_late']}  {'OK' if ok else '*** CHECK ***'}"
            )

    def fmt(r, tag=""):
        return (
            f"{r['config']:>34} ${r['full']:>8.2f} {r['sharpe']:>6.2f} "
            f"${r['maxdd']:>7.2f} ${r['early']:>8.2f} ${r['late']:>8.2f} "
            f"${r['worst_half']:>8.2f} {r['trades']:>5} {r['hit']:>4.0%}{tag}"
        )

    hdr = (
        f"{'config':>34} {'full':>9} {'Shrp':>6} {'maxDD':>8} "
        f"{'early':>9} {'late':>9} {'worst':>9} {'trd':>5} {'hit':>5}"
    )

    print(f"\n{'=' * 112}")
    if pr:
        print(
            f"PROD_REF (LIVE msd2.5/esd1.5): full=${pr['full']:.2f} "
            f"worst-half=${pr['worst_half']:.2f} maxDD=${pr['maxdd']:.2f} "
            f"Sharpe={pr['sharpe']:.2f} trades={pr['trades']} hit={pr['hit']:.0%}"
        )
    print("=" * 112)
    print("v31 HL BINARY AXES SWEEP — all cells sorted by WORST-HALF PnL (corpus 2026-05-10..06-21, n=42)")
    print("-" * 112)
    print(hdr)
    if pr:
        print(fmt(pr, "  <- LIVE"))
    print("-" * 112)
    for r in sorted(rows, key=lambda r: r["worst_half"], reverse=True):
        if r["config"] == "prod_ref":
            continue
        tag = "  *beats-live*" if pr and r["worst_half"] > pr["worst_half"] else ""
        print(fmt(r, tag))
    print("=" * 112)
    print(f"aggregate JSON: {BASE / 'axes_aggregate.json'}")


if __name__ == "__main__":
    main()
