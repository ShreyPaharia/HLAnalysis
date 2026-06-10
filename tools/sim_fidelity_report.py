"""Standing sim-vs-live fidelity report (SHR-90 / Spec 3 T10).

The standing regression gate for the sim-fidelity program. Given one or more
live engine state DBs (the SHR-83 trade journal + venue fill mirror + settlement
table) and the sim's per-market run outputs, it reconciles realized PnL per
settled market, attributes the residual to **input-skew / execution /
unmodeled-halt**, writes a machine-readable report + human summary, appends the
monitored number (``residual_ratio``) to a time series, and exits non-zero when
the sim is no longer trustworthy — so CI / a cron can gate on it.

The sim side is consumed as a JSON array of per-market records (a small, stable
hand-off so this tool never has to run the backtest itself — it is a pure
read-only consumer). Each record:

    {"question_idx": int, "symbol": str, "realized_pnl": float,
     "traded": bool, "n_fills": int,
     "sigma": float | null, "reference_price": float | null}

``sigma`` / ``reference_price`` are the sim's evaluate() inputs; supplying them
lets the report detect input-skew (a decision divergence the sim's MarketState
caused), otherwise a one-sided market's residual falls back to execution.

This module is import-safe (no work at import time) and lives under ``tools/``,
which CLAUDE.md notes is imported as a module by tests — keep it that way.

Run:
    uv run python tools/sim_fidelity_report.py \
        --db /opt/hl-recorder/data/engine/v1/state.db \
        --db /opt/hl-recorder/data/engine/v31/state.db \
        --sim sim_markets.json \
        --json fidelity_report.json \
        --timeseries fidelity_history.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from hlanalysis.parity.sources import (
    LiveMarket,
    SimMarket,
    load_live_markets_from_db,
    reconcile_markets,
)
from hlanalysis.parity.validation import (
    DEFAULT_RESIDUAL_RATIO_THRESHOLD,
    FidelityReport,
    build_report,
    format_report,
)


def load_sim_markets_from_json(path: Path | str) -> list[SimMarket]:
    """Parse the sim per-market hand-off JSON into :class:`SimMarket` records."""
    raw = json.loads(Path(path).read_text())
    out: list[SimMarket] = []
    for r in raw:
        out.append(SimMarket(
            question_idx=int(r["question_idx"]),
            symbol=str(r["symbol"]),
            realized_pnl=float(r.get("realized_pnl", 0.0) or 0.0),
            traded=bool(r.get("traded", (r.get("n_fills", 0) or 0) > 0)),
            sigma=_opt_float(r.get("sigma")),
            reference_price=_opt_float(r.get("reference_price")),
            n_fills=int(r.get("n_fills", 0) or 0),
        ))
    return out


def _opt_float(v) -> float | None:
    return None if v is None else float(v)


def load_live_markets(
    db_paths: Sequence[Path | str], *, key: str = "question_idx",
) -> list[LiveMarket]:
    """Load + merge live per-market records across one or more slot state DBs.

    ``key`` selects the rollup grain (``"question_idx"`` or ``"symbol"``); the
    merge across DBs keys on the same attribute. A market is normally unique to a
    slot; if it recurs across DBs the last DB wins (slots don't share markets in
    practice, so this is defensive)."""
    merged: dict = {}
    for p in db_paths:
        for m in load_live_markets_from_db(p, key=key):
            merged[getattr(m, key)] = m
    return [merged[k] for k in sorted(merged, key=str)]


def generate_report(
    db_paths: Sequence[Path | str],
    sim_markets: Sequence[SimMarket],
    *,
    run_ts_ns: int,
    residual_ratio_threshold: float = DEFAULT_RESIDUAL_RATIO_THRESHOLD,
    sigma_rel_tol: float | None = None,
    ref_rel_tol: float | None = None,
    key: str = "question_idx",
) -> FidelityReport:
    """Reconcile the live DB(s) against the sim markets into a fidelity report.
    ``key`` selects the join grain — use ``"symbol"`` for backtest reconciliation
    (sim/live disagree on question_idx; venue fills carry -1)."""
    live = load_live_markets(db_paths, key=key)
    kw = {"key": key}
    if sigma_rel_tol is not None:
        kw["sigma_rel_tol"] = sigma_rel_tol
    if ref_rel_tol is not None:
        kw["ref_rel_tol"] = ref_rel_tol
    markets = reconcile_markets(live, list(sim_markets), **kw)
    return build_report(
        markets, run_ts_ns=run_ts_ns,
        residual_ratio_threshold=residual_ratio_threshold,
    )


def append_timeseries(path: Path | str, report: FidelityReport) -> None:
    """Append the report's one-line monitoring datapoint to a JSONL time series,
    creating the file (and parents) if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(report.timeseries_row()) + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--db", action="append", default=[], dest="dbs", required=True,
        help="engine state DB path (repeatable, one per slot)",
    )
    p.add_argument(
        "--sim", required=True,
        help="sim per-market hand-off JSON (array of market records)",
    )
    p.add_argument("--json", default=None, help="write machine-readable report here")
    p.add_argument(
        "--timeseries", default=None,
        help="append the monitored datapoint to this JSONL time series",
    )
    p.add_argument(
        "--threshold", type=float, default=DEFAULT_RESIDUAL_RATIO_THRESHOLD,
        help="residual-ratio gate threshold (default %(default)s)",
    )
    p.add_argument(
        "--run-ts-ns", type=int, default=None,
        help="run timestamp in ns (default: now)",
    )
    p.add_argument(
        "--sigma-rel-tol", type=float, default=None,
        help="relative σ tolerance for input-skew detection",
    )
    p.add_argument(
        "--ref-rel-tol", type=float, default=None,
        help="relative reference-price tolerance for input-skew detection",
    )
    p.add_argument(
        "--by", choices=["question_idx", "symbol"], default="question_idx",
        help="market join key. Use 'symbol' to reconcile a backtest (sim/live "
        "disagree on question_idx; venue fills carry -1). Default question_idx.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    run_ts_ns = args.run_ts_ns if args.run_ts_ns is not None else time.time_ns()
    sim_markets = load_sim_markets_from_json(args.sim)
    report = generate_report(
        args.dbs, sim_markets,
        run_ts_ns=run_ts_ns,
        residual_ratio_threshold=args.threshold,
        sigma_rel_tol=args.sigma_rel_tol,
        ref_rel_tol=args.ref_rel_tol,
        key=args.by,
    )
    print(format_report(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nwrote {args.json}")
    if args.timeseries:
        append_timeseries(args.timeseries, report)
        print(f"appended datapoint to {args.timeseries}")
    # Gate: non-zero when the sim is no longer trustworthy so CI / cron can fail.
    return 0 if report.trustworthy else 1


if __name__ == "__main__":
    sys.exit(main())
