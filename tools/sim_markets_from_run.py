"""Emit the sim per-market hand-off JSON from a backtest run (SHR-90).

Bridges a ``hl-bt run`` output directory (``fills.parquet`` +
``diagnostics.parquet``) into the ``SimMarket`` record array that
``tools/sim_fidelity_report.py`` consumes, so a slot backtest can be reconciled
against the live engine state DB and its PnL residual attributed to
input-skew / execution / unmodeled-halt.

One record per traded market (keyed by HL ``symbol``):

    {"question_idx": int, "symbol": str, "realized_pnl": float,
     "traded": true, "n_fills": int,
     "sigma": float|null, "reference_price": float|null}

``sigma`` / ``reference_price`` are the sim's evaluate() inputs at the FIRST
entry decision for the market (mirroring how the live journal rollup takes the
first decision's inputs), so the report can detect input-skew. Reconcile with
``sim_fidelity_report --by symbol`` because the backtest and live engine
disagree on question_idx (buckets) and venue fills carry question_idx=-1.

Import-safe (no work at import time); lives under ``tools/`` which CLAUDE.md
notes is imported as a module by tests — keep it that way.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _entry_inputs_by_question(diag) -> dict[int, tuple[float | None, float | None]]:
    """Representative (σ, reference_price) per question_idx: the FIRST entry
    decision with a populated σ (hold/exit rows carry NaN σ)."""
    out: dict[int, tuple[float | None, float | None]] = {}
    if diag is None or len(diag) == 0:
        return out
    import pandas as pd

    d = diag.sort_values("ts_ns")
    if "action" in d.columns:
        d = d[d["action"] == "enter"]
    if "sigma" in d.columns:
        d = d[d["sigma"].notna()]
    for qi, g in d.groupby("question_idx"):
        r = g.iloc[0]
        sigma = float(r["sigma"]) if "sigma" in g.columns and pd.notna(r["sigma"]) else None
        ref = (
            float(r["ref_price"])
            if "ref_price" in g.columns and pd.notna(r["ref_price"])
            else None
        )
        out[int(qi)] = (sigma, ref)
    return out


def build_sim_markets(fills, diag) -> list[dict[str, Any]]:
    """Build the SimMarket record list from a run's fills + diagnostics frames.

    ``fills`` must carry ``symbol``, ``question_idx``, ``realized_pnl_at_settle``
    and (optionally) ``is_hedge``; hedge legs are excluded from the strategy
    fill count. ``diag`` supplies the entry-decision σ / reference_price."""
    if "is_hedge" in fills.columns:
        fills = fills[~fills["is_hedge"].astype(bool)]
    inputs = _entry_inputs_by_question(diag)
    out: list[dict[str, Any]] = []
    for symbol, g in fills.groupby("symbol"):
        qi = int(g["question_idx"].iloc[0])
        sigma, ref = inputs.get(qi, (None, None))
        out.append({
            "question_idx": qi,
            "symbol": str(symbol),
            "realized_pnl": float(g["realized_pnl_at_settle"].iloc[0]),
            "traded": True,
            "n_fills": int(len(g)),
            "sigma": sigma,
            "reference_price": ref,
        })
    return out


def sim_markets_from_out_dir(out_dir: Path | str) -> list[dict[str, Any]]:
    """Read ``fills.parquet`` (+ ``diagnostics.parquet`` if present) from a
    backtest out-dir and build the SimMarket records."""
    import pandas as pd

    out_dir = Path(out_dir)
    fills = pd.read_parquet(out_dir / "fills.parquet")
    diag_path = out_dir / "diagnostics.parquet"
    diag = pd.read_parquet(diag_path) if diag_path.exists() else None
    return build_sim_markets(fills, diag)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="hl-bt run output directory")
    p.add_argument("--out", required=True, help="write the SimMarket JSON array here")
    args = p.parse_args(argv)
    markets = sim_markets_from_out_dir(args.run_dir)
    Path(args.out).write_text(json.dumps(markets, indent=2))
    print(f"wrote {len(markets)} sim markets → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
