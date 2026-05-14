"""``hl-bt`` command line entrypoint for the v2 backtester.

Subcommands:

- ``strategies`` — list registered strategy ids.
- ``run``        — execute one strategy config across all questions a data source
                   discovers in [start, end), writing report.md + diagnostics +
                   fills to --out-dir.
- ``fetch``      — populate the data-source cache (polymarket only).
- ``tune``       — walk-forward parallel grid sweep.
- ``trace``      — per-question diagnostic trace plot.
- ``report``     — re-render a tuning report from results.jsonl.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Callable, Sequence

from loguru import logger

# Importing the strategy package triggers strategy auto-registration: each
# strategy module's tail calls `@register(...)` at import time, so this
# populates the registry before any `build()` call.
import hlanalysis.strategy  # noqa: F401

from .core.data_source import DataSource, QuestionDescriptor
from .core.registry import build as build_strategy
from .core.registry import ids as registry_ids
from .report import write_single_run_report, write_tuning_report
from .runner.hftbt_runner import RunConfig, run_one_question
from .runner.result import summarise_run

# Environment knobs read by the data-source factories. Workers spawned by
# `hl-bt tune` reach the same caches through the same env vars.
_ENV_PM_CACHE = "HLBT_PM_CACHE_ROOT"
_ENV_HL_DATA = "HLBT_HL_DATA_ROOT"


# ---------------------------------------------------------------------------
# Data source resolution
# ---------------------------------------------------------------------------

def _resolve_data_source(name: str, *, cache_root: str | None = None) -> DataSource:
    """Map a CLI data-source name to a concrete DataSource instance.

    Sources live behind lazy imports so a missing dependency in one source
    (e.g. PM's requests-based client) doesn't take down the CLI. `cache_root`
    overrides the env-var default for PM (`HLBT_PM_CACHE_ROOT`) and the
    data-root default for HL HIP-4 (`HLBT_HL_DATA_ROOT`).
    """
    if name == "synthetic":
        from .data.synthetic import (
            SyntheticDataSource,
            make_default_binary_question,
        )

        ds = SyntheticDataSource()
        ds.add_question(make_default_binary_question())
        return ds
    if name == "polymarket":
        from .data.polymarket import PolymarketDataSource

        root = cache_root or os.environ.get(_ENV_PM_CACHE, "data/sim")
        return PolymarketDataSource(cache_root=Path(root))
    if name == "hl_hip4":
        from .data.hl_hip4 import HLHip4DataSource

        root = cache_root or os.environ.get(_ENV_HL_DATA, "data")
        return HLHip4DataSource(data_root=Path(root))
    raise SystemExit(f"Unknown --data-source: {name}")


# Module-level zero-arg factories used by `hl-bt tune` workers. The dotted name
# is passed through ProcessPoolExecutor and the worker re-imports + calls.
# Cache locations come from env vars set by the parent process before spawn.

def make_polymarket_source() -> "DataSource":
    from .data.polymarket import PolymarketDataSource

    root = os.environ.get(_ENV_PM_CACHE, "data/sim")
    return PolymarketDataSource(cache_root=Path(root))


def make_hl_hip4_source() -> "DataSource":
    from .data.hl_hip4 import HLHip4DataSource

    root = os.environ.get(_ENV_HL_DATA, "data")
    return HLHip4DataSource(data_root=Path(root))


def _factory_dotted_for(name: str) -> str:
    if name == "polymarket":
        return "hlanalysis.backtest.cli.make_polymarket_source"
    if name == "hl_hip4":
        return "hlanalysis.backtest.cli.make_hl_hip4_source"
    raise SystemExit(f"--data-source {name!r} not supported by `tune`")


# ---------------------------------------------------------------------------
# strike resolution helper
# ---------------------------------------------------------------------------

def _strike_for_synthetic(q: QuestionDescriptor) -> float:
    """Best-effort strike resolution for the synthetic source.

    The synthetic source carries strike on the SyntheticQuestion record. Other
    sources will encode strike either in the descriptor's ``leg_symbols`` /
    ``klass`` metadata or via ``question_view`` — for now the runner just
    needs *a* number for the strategy's ``QuestionView.strike`` field, and
    strategies (v1/v2) interpret strike relative to ``reference_price``, so a
    sentinel of 0 works during smoke runs.
    """
    return 0.0


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_strategies(_args: argparse.Namespace) -> int:
    ids_ = registry_ids()
    if not ids_:
        print("(no strategies registered)")
    for sid in ids_:
        print(sid)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    params = json.loads(Path(args.config).read_text())
    strategy = _build_strategy_for_cli(args.strategy, params)

    data_source = _resolve_data_source(args.data_source, cache_root=args.cache_root)
    start = args.start or ""
    end = args.end or ""
    descriptors = list(data_source.discover(start=start, end=end))
    if args.max_markets is not None:
        descriptors = descriptors[args.skip_markets : args.skip_markets + args.max_markets]
    elif args.skip_markets:
        descriptors = descriptors[args.skip_markets :]
    if not descriptors:
        logger.error(
            f"Data source '{args.data_source}' returned no questions for [{start}, {end})"
        )
        return 2
    logger.info(
        f"Running '{args.strategy}' on {len(descriptors)} question(s) "
        f"from data source '{args.data_source}'"
    )

    run_cfg = RunConfig(
        scanner_interval_seconds=args.scanner_interval_seconds,
        tick_size=args.tick_size,
        lot_size=args.lot_size,
        slippage_bps=args.slippage_bps,
        fee_taker=args.fee_taker,
        book_depth_assumption=args.depth,
    )

    strike_fn = _strike_for_data_source(args.data_source)

    out_dir = Path(args.out_dir)
    diag_dir = out_dir / "diagnostics"
    fills_dir = out_dir / "fills"
    per_q_pnl: list[float] = []
    outcomes: list[str] = []
    n_trades = 0
    for q in descriptors:
        res = run_one_question(
            strategy,
            data_source,
            q,
            run_cfg,
            diagnostics_dir=diag_dir,
            fills_dir=fills_dir,
            strike=strike_fn(q),
        )
        per_q_pnl.append(res.realized_pnl_usd or 0.0)
        n_trades += len(res.fills)
        outcomes.append(data_source.resolved_outcome(q))

    summary = summarise_run(per_q_pnl, n_trades=n_trades)
    write_single_run_report(
        out_dir=out_dir,
        strategy_name=args.strategy,
        config_summary=params,
        summary=summary,
        descriptors=descriptors,
        per_question_pnl=per_q_pnl,
        outcomes=outcomes,
        fills_dir=fills_dir,
        fee_taker=args.fee_taker,
        slippage_bps=args.slippage_bps,
    )
    # Concatenate per-question parquets into a single top-level
    # diagnostics.parquet / fills.parquet for downstream tooling.
    _concat_parquets(diag_dir, out_dir / "diagnostics.parquet")
    _concat_parquets(fills_dir, out_dir / "fills.parquet")
    logger.info(f"Report → {out_dir}/report.md")
    return 0


def _build_strategy_for_cli(strategy_id: str, params: dict):
    if strategy_id in registry_ids():
        return build_strategy(strategy_id, params)
    # Allow a smoke-test escape hatch: ``dummy`` strategy emits one ENTER then
    # holds. Used by tests/integration/test_backtest_synthetic_smoke.py.
    if strategy_id == "_dummy_enter_yes":
        from .data.synthetic import build_dummy_enter_strategy

        return build_dummy_enter_strategy(params)
    raise SystemExit(
        f"Unknown --strategy: {strategy_id}. "
        f"Registered: {registry_ids()}. (Task E wires real strategies.)"
    )


def _strike_for_data_source(name: str) -> Callable[[QuestionDescriptor], float]:
    if name == "synthetic":
        return _strike_for_synthetic
    # PM and HL sources will return strike via question_view; for now reuse 0.
    return lambda _q: 0.0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Populate the polymarket cache (Gamma + CLOB + Binance klines)."""
    if args.data_source != "polymarket":
        raise SystemExit(f"fetch supports --data-source polymarket only, got {args.data_source!r}")

    from .data.polymarket import PolymarketDataSource

    cache_root = Path(args.cache_root or os.environ.get(_ENV_PM_CACHE, "data/sim"))
    ds = PolymarketDataSource(cache_root=cache_root)
    descs = ds.fetch_and_cache(
        start=args.start,
        end=args.end,
        kind=args.kind,
        min_trades=args.min_trades,
        min_volume_usd=args.min_volume_usd,
        refresh=args.refresh,
    )
    logger.info(f"PM cache populated: {len(descs)} question(s) in [{args.start}, {args.end})")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    """Walk-forward parallel grid sweep over the configured data source."""
    from .tuning import load_tuning_yaml, run_tuning_parallel

    tcfg = load_tuning_yaml(Path(args.grid))
    grid = tcfg.grids.get(args.strategy)
    if grid is None:
        raise SystemExit(
            f"No grid defined for strategy {args.strategy!r} in {args.grid}. "
            f"Known grid keys: {sorted(tcfg.grids)}"
        )

    # Pass cache locations to workers via env (ProcessPoolExecutor inherits env).
    if args.cache_root:
        if args.data_source == "polymarket":
            os.environ[_ENV_PM_CACHE] = str(args.cache_root)
        elif args.data_source == "hl_hip4":
            os.environ[_ENV_HL_DATA] = str(args.cache_root)

    data_source = _resolve_data_source(args.data_source, cache_root=args.cache_root)
    descriptors = list(data_source.discover(start=args.start or "", end=args.end or ""))
    if args.max_markets is not None:
        descriptors = descriptors[args.skip_markets : args.skip_markets + args.max_markets]
    elif args.skip_markets:
        descriptors = descriptors[args.skip_markets :]
    if not descriptors:
        raise SystemExit(
            f"No questions discovered for data-source {args.data_source!r} "
            f"in [{args.start}, {args.end})"
        )

    run_cfg = RunConfig(
        scanner_interval_seconds=args.scanner_interval_seconds,
        tick_size=args.tick_size,
        lot_size=args.lot_size,
        slippage_bps=args.slippage_bps,
        fee_taker=args.fee_taker,
        book_depth_assumption=args.depth,
    )

    out_dir = Path(args.out_dir) / args.run_id
    run_meta = tcfg.run if isinstance(tcfg.run, dict) else {}
    rows = list(run_tuning_parallel(
        strategy_id=args.strategy,
        grid=grid,
        data_source_factory_dotted=_factory_dotted_for(args.data_source),
        descriptors=descriptors,
        run_cfg=run_cfg,
        train=int(run_meta.get("train_markets", 60)),
        test=int(run_meta.get("test_markets", 15)),
        step=int(run_meta.get("step_markets", 15)),
        out_dir=out_dir,
        n_workers=args.workers,
    ))
    write_tuning_report(out_dir=out_dir, strategy_name=args.strategy, rows=rows, top_k=args.top_k)
    logger.info(f"Tuning report → {out_dir}/report.md ({len(rows)} cells)")
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    from .plots.per_market_trace import plot_market_trace

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "traces" / f"{args.question_id}.html"
    result = plot_market_trace(args.question_id, run_dir, out)
    if result is None:
        logger.error(f"No diagnostics for question {args.question_id} in {run_dir}")
        return 2
    logger.info(f"Trace → {result}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    log_path = run_dir / "results.jsonl"
    if not log_path.exists():
        raise SystemExit(f"No results.jsonl at {log_path}")
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    write_tuning_report(out_dir=run_dir, strategy_name=args.strategy, rows=rows, top_k=args.top_k)
    logger.info(f"Report re-rendered → {run_dir}/report.md ({len(rows)} cells)")
    return 0


def _concat_parquets(in_dir: Path, out_path: Path) -> None:
    if not in_dir.exists():
        return
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = []
    for p in sorted(in_dir.glob("*.parquet")):
        try:
            tables.append(pq.read_table(p))
        except Exception:
            continue
    if not tables:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.concat_tables(tables), out_path)


# ---------------------------------------------------------------------------
# argparse plumbing
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hl-bt")
    sp = p.add_subparsers(dest="cmd", required=True)

    ps = sp.add_parser("strategies", help="List registered strategy ids")
    ps.set_defaults(func=cmd_strategies)

    pr = sp.add_parser("run", help="Run one strategy config across discovered questions")
    pr.add_argument("--strategy", required=True)
    pr.add_argument(
        "--data-source",
        required=True,
        choices=["synthetic", "polymarket", "hl_hip4"],
    )
    pr.add_argument("--config", required=True, help="JSON file of param dict")
    pr.add_argument("--out-dir", required=True)
    pr.add_argument("--start", default=None)
    pr.add_argument("--end", default=None)
    pr.add_argument("--scanner-interval-seconds", type=int, default=60)
    pr.add_argument("--slippage-bps", type=float, default=5.0)
    pr.add_argument("--fee-taker", type=float, default=0.0)
    pr.add_argument("--tick-size", type=float, default=0.001)
    pr.add_argument("--lot-size", type=float, default=1.0)
    pr.add_argument("--depth", type=float, default=10_000.0)
    pr.add_argument("--skip-markets", type=int, default=0)
    pr.add_argument("--max-markets", type=int, default=None)
    pr.add_argument(
        "--cache-root",
        default=None,
        help="Override cache/data root (env: HLBT_PM_CACHE_ROOT / HLBT_HL_DATA_ROOT).",
    )
    pr.set_defaults(func=cmd_run)

    pf = sp.add_parser("fetch", help="Populate polymarket cache from Gamma + CLOB")
    pf.add_argument(
        "--data-source",
        choices=["polymarket"],
        default="polymarket",
        help="Only polymarket supports fetch; HL HIP-4 is consumed from the recorder directly.",
    )
    pf.add_argument("--start", required=True, help="ISO date YYYY-MM-DD")
    pf.add_argument("--end", required=True, help="ISO date YYYY-MM-DD")
    pf.add_argument("--kind", choices=["binary", "bucket", "both"], default="both")
    pf.add_argument("--cache-root", default=None, help="Override env HLBT_PM_CACHE_ROOT")
    pf.add_argument("--min-trades", type=int, default=30)
    pf.add_argument("--min-volume-usd", type=float, default=1000.0)
    pf.add_argument("--refresh", action="store_true")
    pf.set_defaults(func=cmd_fetch)

    pt = sp.add_parser("tune", help="Walk-forward parallel grid sweep")
    pt.add_argument("--strategy", required=True)
    pt.add_argument(
        "--data-source",
        required=True,
        choices=["polymarket", "hl_hip4"],
        help="Synthetic is not supported by tune.",
    )
    pt.add_argument("--grid", required=True, help="YAML file with `grids.<strategy>` + `run.*`")
    pt.add_argument("--out-dir", default="data/sim/tuning")
    pt.add_argument("--run-id", required=True)
    pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--cache-root", default=None)
    pt.add_argument("--start", default=None)
    pt.add_argument("--end", default=None)
    pt.add_argument("--scanner-interval-seconds", type=int, default=60)
    pt.add_argument("--slippage-bps", type=float, default=5.0)
    pt.add_argument("--fee-taker", type=float, default=0.0)
    pt.add_argument("--tick-size", type=float, default=0.001)
    pt.add_argument("--lot-size", type=float, default=1.0)
    pt.add_argument("--depth", type=float, default=10_000.0)
    pt.add_argument("--skip-markets", type=int, default=0)
    pt.add_argument("--max-markets", type=int, default=None)
    pt.add_argument("--top-k", type=int, default=10)
    pt.set_defaults(func=cmd_tune)

    ptr = sp.add_parser("trace", help="Per-question diagnostic trace plot")
    ptr.add_argument("--run-dir", required=True)
    ptr.add_argument("--question-id", required=True)
    ptr.add_argument("--out", default=None, help="Defaults to <run-dir>/traces/<question-id>.html")
    ptr.set_defaults(func=cmd_trace)

    prp = sp.add_parser("report", help="Re-render tuning report from results.jsonl")
    prp.add_argument("--strategy", required=True)
    prp.add_argument("--run-dir", required=True)
    prp.add_argument("--top-k", type=int, default=10)
    prp.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
