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
from .runner.parallel import run_questions_parallel
from .runner.result import summarise_run

# Environment knobs read by the data-source factories. Workers spawned by
# `hl-bt tune` reach the same caches through the same env vars.
_ENV_PM_CACHE = "HLBT_PM_CACHE_ROOT"
_ENV_HL_DATA = "HLBT_HL_DATA_ROOT"
_ENV_PM_FLAVOR = "HLBT_PM_FLAVOR"  # propagated to tune workers via env

_PM_FLAVORS: dict[str, dict[str, str]] = {
    "btc_updown": {
        "reference_symbol": "BTC",
        "series_slug": "btc-up-or-down-daily",
        "klines_subdir": "btc_klines",
    },
    "wti_updown": {
        "reference_symbol": "WTI",
        "series_slug": "oil-daily-up-or-down",
        "klines_subdir": "wti_klines",
    },
}


# ---------------------------------------------------------------------------
# Data source resolution
# ---------------------------------------------------------------------------

def _resolve_data_source(
    name: str,
    *,
    cache_root: str | None = None,
    ref_source: str | None = None,
    pm_flavor: str | None = None,
    hl_reference_resample_seconds: int | None = None,
    pm_reference_source: str | None = None,
    pm_reference_resample_seconds: int | None = None,
    pm_book_source: str | None = None,
    pm_binance_bbo_product_type: str | None = None,
) -> DataSource:
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
        flavor = pm_flavor or os.environ.get(_ENV_PM_FLAVOR, "btc_updown")
        if flavor not in _PM_FLAVORS:
            raise SystemExit(f"Unknown --pm-flavor: {flavor!r}. Choices: {sorted(_PM_FLAVORS)}")
        # Optional cadence-sweep knobs. Default preserves the 1m-klines path
        # used by the 12-month tune — only the BBO-overlap window probe sets
        # these to "binance_bbo" + sub-minute dt.
        prs = pm_reference_source or "klines"
        prrs = int(pm_reference_resample_seconds) if pm_reference_resample_seconds else 60
        pbs = pm_book_source or "synthetic"
        pbpt = pm_binance_bbo_product_type or "perp"
        return PolymarketDataSource(
            cache_root=Path(root),
            reference_source=prs,  # type: ignore[arg-type]
            reference_resample_seconds=prrs,
            book_source=pbs,  # type: ignore[arg-type]
            binance_bbo_product_type=pbpt,  # type: ignore[arg-type]
            **_PM_FLAVORS[flavor],
        )
    if name == "hl_hip4":
        from .data.hl_hip4 import HLHip4DataSource

        root = cache_root or os.environ.get(_ENV_HL_DATA, "data")
        rs = ref_source or "hl_perp"
        # Keep the loader's reference resample period coupled to the strategy's
        # vol_sampling_dt_seconds (passed via the run config). Defaults to 60s.
        rrs = int(hl_reference_resample_seconds) if hl_reference_resample_seconds else 60
        return HLHip4DataSource(
            data_root=Path(root),
            ref_source=rs,  # type: ignore[arg-type]
            reference_resample_seconds=rrs,
        )
    if name == "pm_nba":
        from .data.pm_nba import PolymarketNBADataSource

        root = cache_root or os.environ.get("HLBT_PM_NBA_CACHE_ROOT", "data/sim/pm_nba")
        return PolymarketNBADataSource(cache_root=Path(root))
    raise SystemExit(f"Unknown --data-source: {name}")


# Module-level zero-arg factories used by `hl-bt tune` workers. The dotted name
# is passed through ProcessPoolExecutor and the worker re-imports + calls.
# Cache locations come from env vars set by the parent process before spawn.

def make_polymarket_source() -> "DataSource":
    from .data.polymarket import PolymarketDataSource

    root = os.environ.get(_ENV_PM_CACHE, "data/sim")
    flavor = os.environ.get(_ENV_PM_FLAVOR, "btc_updown")
    return PolymarketDataSource(cache_root=Path(root), **_PM_FLAVORS[flavor])


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


def _extract_hedge_config(params: dict) -> tuple[dict, dict | None]:
    """Extract hedge_* keys from params and return (strategy_params, hedge_run_cfg_dict).

    Strategy params have the CLI-only hedge keys removed (tick sizes, fee rates,
    slippage) so the strategy factory never sees them. ``hedge_symbol`` and
    ``rebalance_*`` stay in params because the v5 strategy factory reads them.

    Returns ``None`` for ``hedge_run_cfg_dict`` when ``hedge_enabled`` is falsy.
    """
    params = dict(params)  # shallow copy; don't mutate the caller's dict
    hedge_enabled = bool(params.pop("hedge_enabled", False))
    # hedge_symbol stays in params for the strategy factory; we also read it here.
    hedge_symbol = str(params.get("hedge_symbol", "BTC-PERP"))
    # These are CLI-only tuning knobs — strip them from strategy params.
    hedge_tick_size = float(params.pop("hedge_tick_size", 0.1))
    hedge_lot_size = float(params.pop("hedge_lot_size", 0.001))
    hedge_slippage_bps = float(params.pop("hedge_slippage_bps", 10.0))
    hedge_fee_bps = float(params.pop("hedge_fee_bps", 1.0))
    # half_spread_bps is hedge-data specific (BinancePerpKlinesSource param), not a strategy param.
    hedge_half_spread_bps = float(params.pop("hedge_half_spread_bps", 1.0))
    if not hedge_enabled:
        return params, None
    return params, dict(
        hedge_enabled=True,
        hedge_symbol=hedge_symbol,
        hedge_tick_size=hedge_tick_size,
        hedge_lot_size=hedge_lot_size,
        hedge_slippage_bps=hedge_slippage_bps,
        hedge_fee_bps=hedge_fee_bps,
        _half_spread_bps=hedge_half_spread_bps,  # extra; consumed by _build_hedge_source only
    )


def _build_hedge_source(hedge_data_path: str | None, hedge_cfg: dict | None):
    """Build a BinancePerpKlinesSource if hedge is enabled and a path is given."""
    if hedge_cfg is None or not hedge_data_path:
        return None
    from .data.binance_perp import BinancePerpKlinesSource

    return BinancePerpKlinesSource(
        path=Path(hedge_data_path),
        symbol=hedge_cfg["hedge_symbol"],
        half_spread_bps=hedge_cfg.get("_half_spread_bps", 1.0),
    )


def _data_source_dotted(name: str) -> str:
    """Return the zero-arg factory dotted path for a data source name.

    Reuses the same resolver as `_factory_dotted_for` (used by tune), extended
    to support sources that only `run` needs (synthetic, pm_nba).
    """
    if name in ("polymarket", "hl_hip4"):
        return _factory_dotted_for(name)
    # For synthetic / pm_nba — fall back to a dotted path that the worker can
    # reconstruct.  Synthetic is rarely used in multi-worker runs but must not
    # crash the arg-path.  Workers re-discover via wide date windows so synthetic
    # rebuilds an identical question set.
    if name == "pm_nba":
        return "hlanalysis.backtest.cli.make_pm_nba_source"
    # synthetic: not parallelisable in practice (in-memory source), but still
    # needs a path so workers can reconstruct it.
    return "hlanalysis.backtest.cli.make_synthetic_source"


def _hedge_data_path_for(args: argparse.Namespace) -> str | None:
    """Extract the hedge data path from parsed CLI args (None when not provided)."""
    return getattr(args, "hedge_data_path", None) or None


def _hedge_half_spread_for(args: argparse.Namespace) -> float:
    """Extract hedge half-spread bps from parsed CLI args."""
    return float(getattr(args, "hedge_half_spread_bps", 1.0))


# Zero-arg factories for additional data sources (used by parallel workers).
def make_pm_nba_source() -> "DataSource":
    import os
    from .data.pm_nba import PolymarketNBADataSource

    root = os.environ.get("HLBT_PM_NBA_CACHE_ROOT", "data/sim/pm_nba")
    return PolymarketNBADataSource(cache_root=Path(root))


def make_synthetic_source() -> "DataSource":
    from .data.synthetic import SyntheticDataSource, make_default_binary_question

    ds = SyntheticDataSource()
    ds.add_question(make_default_binary_question())
    return ds


def cmd_run(args: argparse.Namespace) -> int:
    params = json.loads(Path(args.config).read_text())

    # --rebuild-cache: force event-array rebuild process-wide (inherited by
    # spawn workers via the environment).
    if getattr(args, "rebuild_cache", False):
        os.environ["HLBT_REBUILD_CACHE"] = "1"

    # Extract hedge config from params before passing to the strategy factory.
    # This keeps the strategy factory clean (it only sees binary knobs).
    params, hedge_cfg = _extract_hedge_config(params)

    # Propagate cache/data-root to workers via env (mirrors cmd_tune so spawned
    # worker factories pick up the same root as the parent process).
    if args.cache_root:
        if args.data_source == "polymarket":
            os.environ[_ENV_PM_CACHE] = str(args.cache_root)
        elif args.data_source == "hl_hip4":
            os.environ[_ENV_HL_DATA] = str(args.cache_root)
    if args.data_source == "polymarket":
        os.environ[_ENV_PM_FLAVOR] = getattr(args, "pm_flavor", "btc_updown") or "btc_updown"

    strategy = _build_strategy_for_cli(args.strategy, params)

    hedge_source = _build_hedge_source(args.hedge_data_path, hedge_cfg)

    data_source = _resolve_data_source(
        args.data_source,
        cache_root=args.cache_root,
        ref_source=getattr(args, "ref_source", None),
        pm_flavor=getattr(args, "pm_flavor", None),
        # Keep the HL reference downsampler in lockstep with the strategy's
        # vol_sampling_dt_seconds — same param, same source-of-truth.
        hl_reference_resample_seconds=int(params.get("vol_sampling_dt_seconds", 60)),
        pm_reference_source=getattr(args, "pm_reference_source", None),
        pm_reference_resample_seconds=int(params.get("vol_sampling_dt_seconds", 60)),
        pm_book_source=getattr(args, "pm_book_source", None),
        pm_binance_bbo_product_type=getattr(args, "pm_binance_bbo_product_type", None),
    )
    start = args.start or ""
    end = args.end or ""
    discover_kwargs: dict = {}
    kind = getattr(args, "kind", "both")
    if kind != "both":
        if args.data_source == "polymarket":
            discover_kwargs["kind"] = kind
        elif args.data_source == "hl_hip4":
            discover_kwargs["kinds"] = (
                "priceBinary" if kind == "binary" else "priceBucket",
            )
    descriptors = list(data_source.discover(start=start, end=end, **discover_kwargs))
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

    # Build RunConfig — include hedge fields when enabled.
    run_cfg_kwargs: dict = dict(
        scanner_interval_seconds=args.scanner_interval_seconds,
        tick_size=args.tick_size,
        lot_size=args.lot_size,
        slippage_bps=args.slippage_bps,
        fee_taker=args.fee_taker,
        fee_model=args.fee_model,
        fee_rate=args.fee_rate,
        book_depth_assumption=args.depth,
    )
    if hedge_cfg is not None:
        run_cfg_kwargs["hedge_enabled"] = hedge_cfg["hedge_enabled"]
        run_cfg_kwargs["hedge_symbol"] = hedge_cfg["hedge_symbol"]
        run_cfg_kwargs["hedge_tick_size"] = hedge_cfg["hedge_tick_size"]
        run_cfg_kwargs["hedge_lot_size"] = hedge_cfg["hedge_lot_size"]
        run_cfg_kwargs["hedge_slippage_bps"] = hedge_cfg["hedge_slippage_bps"]
        run_cfg_kwargs["hedge_fee_bps"] = hedge_cfg["hedge_fee_bps"]
    run_cfg = RunConfig(**run_cfg_kwargs)

    strike_fn = _strike_for_data_source(args.data_source)

    out_dir = Path(args.out_dir)
    diag_dir = out_dir / "diagnostics"
    fills_dir = out_dir / "fills"

    n_workers = max(1, int(getattr(args, "workers", 1)))
    results = run_questions_parallel(
        descriptors=descriptors,
        strategy_id=args.strategy,
        params=params,
        run_cfg=run_cfg,
        data_source_dotted=_data_source_dotted(args.data_source),
        diagnostics_dir=diag_dir,
        fills_dir=fills_dir,
        strike_for=strike_fn,
        hedge_data_path=_hedge_data_path_for(args),
        hedge_half_spread_bps=_hedge_half_spread_for(args),
        n_workers=n_workers,
    )
    per_q_pnl = [r.realized_pnl_usd for r in results]
    n_trades = sum(r.n_fills for r in results)
    outcomes = [r.outcome for r in results]

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
    # Single source of truth shared with the pool workers (registry + the
    # ``_dummy_enter_yes`` smoke escape hatch).
    from .runner.parallel import build_strategy_for_run

    return build_strategy_for_run(strategy_id, params)


def _strike_for_data_source(name: str) -> Callable[[QuestionDescriptor], float]:
    if name == "synthetic":
        return _strike_for_synthetic
    # PM and HL sources will return strike via question_view; for now reuse 0.
    return lambda _q: 0.0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Populate the polymarket cache: PM markets/trades (Gamma + CLOB) plus a
    coupled fetch of the Binance spot klines covering those markets, so strike
    resolution never lags the market cache (SHR-54)."""
    if args.data_source != "polymarket":
        raise SystemExit(f"fetch supports --data-source polymarket only, got {args.data_source!r}")

    from .data.polymarket import PolymarketDataSource

    cache_root = Path(args.cache_root or os.environ.get(_ENV_PM_CACHE, "data/sim"))
    fcfg = _PM_FLAVORS[args.pm_flavor]
    ds = PolymarketDataSource(cache_root=cache_root, **fcfg)
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

    # --rebuild-cache: force event-array rebuild once; spawn workers inherit it.
    if getattr(args, "rebuild_cache", False):
        os.environ["HLBT_REBUILD_CACHE"] = "1"

    # Pass cache locations to workers via env (ProcessPoolExecutor inherits env).
    if args.cache_root:
        if args.data_source == "polymarket":
            os.environ[_ENV_PM_CACHE] = str(args.cache_root)
        elif args.data_source == "hl_hip4":
            os.environ[_ENV_HL_DATA] = str(args.cache_root)

    if args.data_source == "polymarket":
        os.environ[_ENV_PM_FLAVOR] = args.pm_flavor

    data_source = _resolve_data_source(
        args.data_source,
        cache_root=args.cache_root,
        ref_source=getattr(args, "ref_source", None),
        pm_flavor=getattr(args, "pm_flavor", None),
    )
    discover_kwargs: dict = {}
    if args.data_source == "polymarket" and args.kind != "both":
        discover_kwargs["kind"] = args.kind
    descriptors = list(data_source.discover(
        start=args.start or "", end=args.end or "", **discover_kwargs,
    ))
    if args.max_markets is not None:
        descriptors = descriptors[args.skip_markets : args.skip_markets + args.max_markets]
    elif args.skip_markets:
        descriptors = descriptors[args.skip_markets :]
    if not descriptors:
        raise SystemExit(
            f"No questions discovered for data-source {args.data_source!r} "
            f"in [{args.start}, {args.end})"
        )

    # Extract hedge config from the grid's fixed_params section if present.
    # The grid YAML may carry hedge_* keys in a top-level ``fixed_params`` block
    # alongside the grid sweep keys, OR the grid values may be length-1 lists.
    # Here we pull hedge_* from args (injected via --hedge-* flags at tune time).
    hedge_cfg_dict: dict | None = None
    if args.hedge_data_path:
        hedge_cfg_dict = dict(
            hedge_enabled=True,
            hedge_symbol=args.hedge_symbol,
            hedge_tick_size=args.hedge_tick_size,
            hedge_lot_size=args.hedge_lot_size,
            hedge_slippage_bps=args.hedge_slippage_bps,
            hedge_fee_bps=args.hedge_fee_bps,
            _half_spread_bps=args.hedge_half_spread_bps,
        )
        # Pass hedge data path to workers via env so they can rebuild the source.
        os.environ["HLBT_HEDGE_DATA_PATH"] = str(args.hedge_data_path)

    run_cfg_kwargs: dict = dict(
        scanner_interval_seconds=args.scanner_interval_seconds,
        tick_size=args.tick_size,
        lot_size=args.lot_size,
        slippage_bps=args.slippage_bps,
        fee_taker=args.fee_taker,
        fee_model=args.fee_model,
        fee_rate=args.fee_rate,
        book_depth_assumption=args.depth,
    )
    if hedge_cfg_dict is not None:
        run_cfg_kwargs["hedge_enabled"] = True
        run_cfg_kwargs["hedge_symbol"] = hedge_cfg_dict["hedge_symbol"]
        run_cfg_kwargs["hedge_tick_size"] = hedge_cfg_dict["hedge_tick_size"]
        run_cfg_kwargs["hedge_lot_size"] = hedge_cfg_dict["hedge_lot_size"]
        run_cfg_kwargs["hedge_slippage_bps"] = hedge_cfg_dict["hedge_slippage_bps"]
        run_cfg_kwargs["hedge_fee_bps"] = hedge_cfg_dict["hedge_fee_bps"]
    run_cfg = RunConfig(**run_cfg_kwargs)

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
        hedge_data_path=args.hedge_data_path,
        hedge_half_spread_bps=float(getattr(args, "hedge_half_spread_bps", 1.0)),
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
        choices=["synthetic", "polymarket", "hl_hip4", "pm_nba"],
    )
    pr.add_argument("--config", required=True, help="JSON file of param dict")
    pr.add_argument("--out-dir", required=True)
    pr.add_argument("--start", default=None)
    pr.add_argument("--end", default=None)
    pr.add_argument("--scanner-interval-seconds", type=int, default=60)
    pr.add_argument("--slippage-bps", type=float, default=5.0)
    pr.add_argument("--fee-taker", type=float, default=0.0)
    pr.add_argument(
        "--fee-model",
        choices=["flat", "pm_binary"],
        default="flat",
        help="Binary-leg fee model. 'flat' uses --fee-taker as constant %% of "
        "notional (HL/synthetic). 'pm_binary' uses Polymarket's curve "
        "fee = qty * --fee-rate * p * (1-p).",
    )
    pr.add_argument(
        "--fee-rate",
        type=float,
        default=0.07,
        help="feeRate for --fee-model pm_binary. PM crypto = 0.07; sports = 0.03; "
        "politics/tech/finance/mentions = 0.04; econ/culture/weather = 0.05.",
    )
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
    pr.add_argument(
        "--pm-flavor",
        choices=["btc_updown", "wti_updown"],
        default="btc_updown",
        help="(polymarket only) Which PM series + reference asset. "
        "btc_updown: BTC 'Up or Down Daily' (default). "
        "wti_updown: WTI 'Oil Daily Up or Down'.",
    )
    pr.add_argument(
        "--hedge-data-path",
        default=None,
        help="Path to Binance perp/spot kline JSON for the hedge leg (v5 only). "
        "When omitted, hedge_enabled in --config is ignored.",
    )
    pr.add_argument(
        "--kind",
        choices=["binary", "bucket", "both"],
        default="both",
        help="Filter discovery to this question class. "
        "For hl_hip4: binary→priceBinary, bucket→priceBucket. "
        "For polymarket: pass-through.",
    )
    pr.add_argument(
        "--ref-source",
        choices=["hl_perp", "binance_perp"],
        default="hl_perp",
        help="(hl_hip4 only) Reference-price feed for σ + p_model. "
        "`hl_perp` reads HL BBO/mark; `binance_perp` reads Binance perp BBO/mark.",
    )
    pr.add_argument(
        "--pm-reference-source",
        choices=["klines", "binance_bbo"],
        default="klines",
        help="(polymarket only) Reference-feed source. `klines` (default) reads "
        "cached 1m Binance klines (12-month corpus). `binance_bbo` reads "
        "recorded Binance BBO ticks and buckets to vol_sampling_dt_seconds "
        "(BBO-overlap window only).",
    )
    pr.add_argument(
        "--pm-binance-bbo-product-type",
        choices=["perp", "spot"],
        default="perp",
        dest="pm_binance_bbo_product_type",
        help="(polymarket binance_bbo only) Binance product type for the BBO "
        "reference feed. `perp` (default) reads Binance PERP BBO ticks. `spot` "
        "reads Binance SPOT BBO ticks, matching PM's settlement instrument "
        "(Binance SPOT 1m close). Spot ticks use local_recv_ts since Binance "
        "SPOT bookTicker does not provide exchange_ts.",
    )
    pr.add_argument(
        "--pm-book-source",
        choices=["synthetic", "recorded"],
        default="synthetic",
        help="(polymarket only) Fill-book source. `synthetic` (default) builds "
        "a flat 1-level book per trade print + `1−p` parity. `recorded` feeds "
        "the real multi-level L2 `book_snapshot` parquet per leg (HL parity; "
        "coverage from 2026-05-27).",
    )
    pr.add_argument("--workers", type=int, default=1,
                    help="Parallel worker processes for independent markets "
                         "(default 1 = serial). Use up to #cores for big runs.")
    pr.add_argument("--rebuild-cache", action="store_true",
                    help="Ignore cached event arrays and rebuild them (then "
                         "repopulate the cache).")
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
    pf.add_argument(
        "--pm-flavor",
        choices=["btc_updown", "wti_updown"],
        default="btc_updown",
        help="(polymarket only) Which PM series + reference asset. "
        "btc_updown: BTC 'Up or Down Daily' (default). "
        "wti_updown: WTI 'Oil Daily Up or Down'.",
    )
    pf.add_argument("--min-trades", type=int, default=30)
    pf.add_argument("--min-volume-usd", type=float, default=1000.0)
    pf.add_argument("--refresh", action="store_true")
    pf.set_defaults(func=cmd_fetch)

    pt = sp.add_parser("tune", help="Walk-forward parallel grid sweep")
    pt.add_argument("--strategy", required=True)
    pt.add_argument(
        "--data-source",
        required=True,
        choices=["polymarket", "hl_hip4", "pm_nba"],
        help="Synthetic is not supported by tune.",
    )
    pt.add_argument("--grid", required=True, help="YAML file with `grids.<strategy>` + `run.*`")
    pt.add_argument("--out-dir", default="data/sim/tuning")
    pt.add_argument("--run-id", required=True)
    pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--cache-root", default=None)
    pt.add_argument(
        "--pm-flavor",
        choices=["btc_updown", "wti_updown"],
        default="btc_updown",
        help="(polymarket only) Which PM series + reference asset. "
        "btc_updown: BTC 'Up or Down Daily' (default). "
        "wti_updown: WTI 'Oil Daily Up or Down'.",
    )
    pt.add_argument("--start", default=None)
    pt.add_argument("--end", default=None)
    pt.add_argument("--scanner-interval-seconds", type=int, default=60)
    pt.add_argument("--slippage-bps", type=float, default=5.0)
    pt.add_argument("--fee-taker", type=float, default=0.0)
    pt.add_argument(
        "--fee-model",
        choices=["flat", "pm_binary"],
        default="flat",
        help="Binary-leg fee model. See `run` help for details.",
    )
    pt.add_argument("--fee-rate", type=float, default=0.07)
    pt.add_argument("--tick-size", type=float, default=0.001)
    pt.add_argument("--lot-size", type=float, default=1.0)
    pt.add_argument("--depth", type=float, default=10_000.0)
    pt.add_argument("--skip-markets", type=int, default=0)
    pt.add_argument("--max-markets", type=int, default=None)
    pt.add_argument("--top-k", type=int, default=10)
    pt.add_argument("--rebuild-cache", action="store_true",
                    help="Ignore cached event arrays and rebuild them once "
                         "(then reuse across the sweep).")
    pt.add_argument(
        "--kind",
        choices=["binary", "bucket", "both"],
        default="both",
        help="(polymarket) Filter discovery to this question kind. "
        "Avoids mixing binary and bucket markets in one walk-forward.",
    )
    # Hedge leg flags (v5_delta_hedged only; safe to pass for other strategies
    # since hedge_enabled defaults to False when --hedge-data-path is omitted).
    pt.add_argument(
        "--hedge-data-path",
        default=None,
        help="Path to Binance perp/spot kline JSON for the hedge leg (v5 only).",
    )
    pt.add_argument("--hedge-symbol", default="BTC-PERP")
    pt.add_argument("--hedge-tick-size", type=float, default=0.1)
    pt.add_argument("--hedge-lot-size", type=float, default=0.001)
    pt.add_argument("--hedge-slippage-bps", type=float, default=15.0)
    pt.add_argument("--hedge-fee-bps", type=float, default=1.0)
    pt.add_argument("--hedge-half-spread-bps", type=float, default=1.0)
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
