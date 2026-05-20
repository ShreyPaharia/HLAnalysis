"""Analyse the exit_safety_d sweep.

For each variant in data/sim/runs/v3.1-exit-safety-d-2026-05-21/<corpus>/exit_safety_d_*:
  - read fills.parquet → PnL, hit-rate, max DD, trade count, exit-reason histogram
  - read diagnostics.parquet → exit-reason histogram (the reduce_only exits show
    in fills as side flip; the strategy's exit_reason flows through the OrderIntent
    to the router and shows up in the alert layer, NOT in fills.parquet today —
    so we approximate from diagnostics.parquet `info=exit_*` rows).
  - join with market_meta + question_meta to classify HL fills into regime
    {binary, edge_bucket, middle_bucket}

Outputs aggregate tables and an exit-reason histogram per (corpus, variant).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd

DATA = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data")
RUNS = Path("data/sim/runs/v3.1-exit-safety-d-2026-05-21")

QMETA = DATA / "venue=hyperliquid/product_type=prediction_binary/mechanism=clob/event=question_meta"
MMETA = DATA / "venue=hyperliquid/product_type=prediction_binary/mechanism=clob/event=market_meta"


def hl_regime_lookup():
    """Return question_id → (klass, named_outcome_idxs_list), market_symbol → outcome_idx."""
    qd = ds.dataset(str(QMETA), format="parquet", partitioning="hive")
    qdf = qd.to_table(columns=["symbol", "named_outcome_idxs", "keys", "values"]).to_pandas()
    q_meta: dict[str, tuple[str, list[int]]] = {}
    for _, r in qdf.iterrows():
        kv = dict(zip(r["keys"], r["values"]))
        klass = kv.get("class", "")
        outcomes = [int(x) for x in r["named_outcome_idxs"]]
        q_meta[r["symbol"]] = (klass, outcomes)

    md = ds.dataset(str(MMETA), format="parquet", partitioning="hive")
    mdf = md.to_table(columns=["symbol", "keys", "values"]).to_pandas()
    m_outcome: dict[str, int] = {}
    for _, r in mdf.iterrows():
        kv = dict(zip(r["keys"], r["values"]))
        try:
            m_outcome[r["symbol"]] = int(kv.get("outcome_idx", "-1"))
        except (TypeError, ValueError):
            continue
    return q_meta, m_outcome


def classify_fill(qid: str, leg_sym: str, q_meta, m_outcome) -> str:
    if qid not in q_meta:
        return "unknown"
    klass, outcomes = q_meta[qid]
    if klass == "priceBinary":
        return "binary"
    if klass == "priceBucket":
        out_idx = m_outcome.get(leg_sym, -1)
        try:
            pos = outcomes.index(out_idx)
        except ValueError:
            return "unknown"
        n = len(outcomes)
        if pos == 0 or pos == n - 1:
            return "edge_bucket"
        return "middle_bucket"
    return "unknown"


def summarise_run(out_dir: Path, hl: bool, classifier_args) -> dict:
    fills_path = out_dir / "fills.parquet"
    if not fills_path.exists():
        return {"missing": True}

    df = pq.read_table(fills_path).to_pandas()
    if df.empty:
        return {"trades": 0, "pnl": 0.0, "sharpe": 0.0, "hit": 0.0, "max_dd": 0.0}

    # Per-position PnL: realized_pnl_at_settle is constant per question for v3.1.
    # We compute hit-rate at the question level (unique non-zero PnL per question_id).
    per_q = df.groupby("question_id")["realized_pnl_at_settle"].first().sort_index()
    # max DD on cumulative equity by question (in chrono order).
    df = df.sort_values("ts_ns")
    per_q_ts = df.groupby("question_id")["ts_ns"].min().to_dict()
    qids_order = sorted(per_q.index, key=lambda q: per_q_ts.get(q, 0))
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for q in qids_order:
        cum += per_q[q]
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    pnl = float(per_q.sum())
    n_q = int(len(per_q))
    n_trades = int(len(df))
    hit = float((per_q > 0).mean()) if n_q > 0 else 0.0
    if n_q > 1:
        mu = per_q.mean()
        sd = per_q.std(ddof=1)
        sharpe = float(mu / sd * math.sqrt(365.0)) if sd > 0 else 0.0
    else:
        sharpe = 0.0

    out: dict = {
        "trades": n_trades,
        "positions": n_q,
        "pnl": pnl,
        "sharpe": sharpe,
        "hit": hit,
        "max_dd": max_dd,
    }

    if hl:
        q_meta, m_outcome = classifier_args
        regimes = df.apply(
            lambda r: classify_fill(r["question_id"], r["symbol"], q_meta, m_outcome),
            axis=1,
        )
        df["regime"] = regimes
        # Per-question regime: take the FIRST entry's regime (multi-leg switches
        # are rare; pre-merge a separate column for per-leg PnL would need a
        # FIFO accounting pass).
        per_q_regime = df.sort_values("ts_ns").groupby("question_id")["regime"].first()
        merged = pd.DataFrame({"regime": per_q_regime, "pnl": per_q})
        by_regime = merged.groupby("regime")["pnl"].agg(["sum", "count", "mean"]).to_dict("index")
        out["regime"] = {
            k: {"pnl": float(v["sum"]), "n": int(v["count"]),
                "per_trade": float(v["mean"])}
            for k, v in by_regime.items()
        }

    # Exit-reason histogram from diagnostics. v3.1 emits Diagnostic(level="info",
    # message="exit_safety_d" | "exit_edge" | "exit_settlement" | ...). The
    # parquet writer flattens the diagnostic message into the `reason` column.
    diag_path = out_dir / "diagnostics.parquet"
    if diag_path.exists():
        dd = pq.read_table(diag_path).to_pandas()
        if "reason" in dd.columns:
            er = dd[dd["reason"].astype(str).str.startswith("exit_")]
            out["exit_reasons"] = {k: int(v) for k, v in er["reason"].value_counts().to_dict().items()}
        else:
            out["exit_reasons"] = {}
    return out


def main() -> int:
    print("loading HL meta...", file=sys.stderr)
    hl_args = hl_regime_lookup()

    results: dict[str, dict[str, dict]] = {"hl": {}, "pm": {}}

    for corpus in ["hl", "pm"]:
        cdir = RUNS / corpus
        if not cdir.exists():
            continue
        for variant in sorted(cdir.iterdir()):
            if not variant.is_dir():
                continue
            tag = variant.name
            try:
                results[corpus][tag] = summarise_run(variant, hl=(corpus == "hl"), classifier_args=hl_args)
                print(f"  {corpus}/{tag}: pnl=${results[corpus][tag].get('pnl', 0.0):+.2f}", file=sys.stderr)
            except Exception as e:
                results[corpus][tag] = {"error": str(e)}
                print(f"  {corpus}/{tag}: ERROR {e}", file=sys.stderr)

    out_path = RUNS / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"summary → {out_path}", file=sys.stderr)
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
