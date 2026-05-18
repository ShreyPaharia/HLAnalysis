#!/usr/bin/env python3
"""Build a side-by-side comparison of v2 / v3 / v4 / v5 OOS metrics.

Reads each strategy's walk-forward + full-corpus report.md (already generated
by the tuning + run pipeline), aggregates the headline metrics, and writes
docs/reports/12-strategy-comparison-v3-v4-v5.md.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StrategyResult:
    name: str
    full_corpus_pnl: float
    full_corpus_sharpe: float
    full_corpus_hit_rate: float
    full_corpus_max_dd: float
    n_trades: int


_PATTERNS = {
    "n_trades": re.compile(r"- trades:\s+(\d+)"),
    "pnl": re.compile(r"- total PnL:\s+\$(-?[\d.,]+)"),
    "sharpe": re.compile(r"- Sharpe \(annualized 365\):\s+(-?[\d.]+)"),
    "hit_rate": re.compile(r"- hit rate:\s+([\d.]+)%"),
    "max_dd": re.compile(r"- max drawdown:\s+\$([\d.,]+)"),
}


def _parse_report(path: Path, name: str) -> StrategyResult:
    text = path.read_text()

    def _g(key: str) -> str:
        m = _PATTERNS[key].search(text)
        if not m:
            raise ValueError(f"{path}: missing field {key}")
        return m.group(1).replace(",", "")

    return StrategyResult(
        name=name,
        full_corpus_pnl=float(_g("pnl")),
        full_corpus_sharpe=float(_g("sharpe")),
        full_corpus_hit_rate=float(_g("hit_rate")),
        full_corpus_max_dd=float(_g("max_dd")),
        n_trades=int(_g("n_trades")),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v2", default="data/sim/runs/v2-diag-2026-05-10/report.md")
    p.add_argument("--v3", default="data/sim/runs/v3-walkforward-2026-05-18-full/report.md")
    p.add_argument("--v4", default="data/sim/runs/v4-walkforward-2026-05-18-full/report.md")
    p.add_argument("--v5", default="data/sim/runs/v5-walkforward-2026-05-18-full/report.md")
    p.add_argument("--out", default="docs/reports/12-strategy-comparison-v3-v4-v5.md")
    args = p.parse_args()

    results = [
        _parse_report(Path(args.v2), "v2 model_edge (baseline)"),
        _parse_report(Path(args.v3), "v3 theta_harvester"),
        _parse_report(Path(args.v4), "v4 binary_statarb"),
        _parse_report(Path(args.v5), "v5 delta_hedged"),
    ]

    lines = [
        "# Strategy Comparison: v3 / v4 / v5 vs v2 baseline",
        "",
        "PM BTC daily Up/Down corpus, 2025-05-08 → 2026-05-08, 364 markets.",
        "Walk-forward selection: train=60, test=60, step=60, drop_short_tail=True (5 OOS folds).",
        "Objective: OOS Sharpe with n_trades ≥ 20 floor on train.",
        "",
        "## Headline metrics (full-corpus run at modal best config)",
        "",
        "| Strategy | Trades | Total PnL ($) | Sharpe | Hit rate | Max DD ($) |",
        "| :------- | -----: | ------------: | -----: | -------: | ---------: |",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.n_trades} | {r.full_corpus_pnl:.2f} | {r.full_corpus_sharpe:.3f} | "
            f"{r.full_corpus_hit_rate:.2f}% | {r.full_corpus_max_dd:.2f} |"
        )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append("(Fill in once results are in. Default heuristic: pick the strategy with highest OOS Sharpe at hit_rate >= 60% and per-fold stability ≥ 4/5.)")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- PM is a proxy for HL HIP-4. Confirm via HL paper-trade before live commit.")
    lines.append("- v5 uses Binance perp BBO synthesized from 1m klines as the hedge feed; production uses HL perp.")
    lines.append("  Basis budget applied via hedge_slippage_bps=10 (~5bps venue basis + 1bp BBO error + headroom).")
    lines.append("- 5-fold walk-forward is noisy; treat absolute PnL deltas < $200 as within sampling noise.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
