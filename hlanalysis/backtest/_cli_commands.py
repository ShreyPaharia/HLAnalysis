"""``hl-bt`` subcommand handlers.

Each ``cmd_*`` function is the ``func`` set on a subparser via
``set_defaults(func=...)``.  The handlers import heavy dependencies lazily so
the CLI stays fast on ``--help``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from loguru import logger

from ._cli_plumbing import (
    _ENV_PM_CACHE,
    _build_hedge_source,
    _build_strategy_for_cli,
    _concat_parquets,
    _extract_hedge_config,
    _hedge_data_path_for,
    _hedge_half_spread_for,
    _load_run_params,
    _resolve_reference_warmup_seconds,
    _run_config_from_args,
    _source_config_from_args,
    _strike_for_data_source,
    assert_hl_cadence_match,
)
from .core.registry import ids as registry_ids
from .report import write_single_run_report, write_tuning_report
from .runner.parallel import run_questions_parallel
from .runner.result import summarise_run


def cmd_strategies(_args: argparse.Namespace) -> int:
    ids_ = registry_ids()
    if not ids_:
        print("(no strategies registered)")
    for sid in ids_:
        print(sid)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from ..marketdata.decision_input import from_backtest_params

    params = _load_run_params(args)

    # Event-array cache is default-ON. --fresh/--no-cache disables it for this
    # invocation; --rebuild-cache forces a one-time rebuild (still repopulates).
    # --cache-event-arrays is now a no-op (kept for back-compat). These are
    # genuine runtime toggles (not source-construction), so they still ride to
    # spawn workers via the inherited environment.
    if getattr(args, "no_cache", False):
        os.environ["HLBT_NO_CACHE"] = "1"
    if getattr(args, "rebuild_cache", False):
        os.environ["HLBT_REBUILD_CACHE"] = "1"

    # Extract hedge config from params before passing to the strategy factory.
    # This keeps the strategy factory clean (it only sees binary knobs).
    params, hedge_cfg = _extract_hedge_config(params)

    # Built for its validation side effect (raises on a bad hedge path); the
    # actual source is rebuilt from the picklable SourceConfig below.
    _hedge_source = _build_hedge_source(args.hedge_data_path, hedge_cfg)

    # ONE source-construction path: the SourceConfig is used to build the
    # in-process source here AND shipped (picklable) to subprocess workers, so
    # both build identically. Couple the reference downsampler to the strategy's
    # vol_sampling_dt_seconds — same param, same source-of-truth.
    # SHR-92: auto-derive reference_warmup_seconds from vol_lookback_seconds
    # (max across classes) so σ is warm at every market open. The
    # --reference-warmup-seconds flag overrides; 0 explicitly disables.
    # SHR-97: resolve source/ticks from the shared DecisionInputConfig so the
    # backtest derives live-faithful defaults (mark + raw) instead of hard-coding
    # non-live defaults (bbo + bars). CLI flags remain as explicit A/B overrides.
    _cli_warmup = getattr(args, "reference_warmup_seconds", None)
    _warmup = _resolve_reference_warmup_seconds(
        params, data_source=args.data_source, cli_override=_cli_warmup
    )
    _resolved = from_backtest_params(params, track_default_source="mark")
    source_config = _source_config_from_args(
        args,
        reference_resample_seconds=int(params.get("vol_sampling_dt_seconds", 60)),
        reference_warmup_seconds=_warmup,
        resolved=_resolved,
    )
    assert_hl_cadence_match(source_config, params)
    data_source = source_config.build()
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

    run_cfg = _run_config_from_args(args, hedge_cfg)

    strike_fn = _strike_for_data_source(args.data_source)

    out_dir = Path(args.out_dir)
    diag_dir = out_dir / "diagnostics"
    fills_dir = out_dir / "fills"

    n_workers = max(1, int(getattr(args, "workers", 1)))
    # Build the strategy once; the in-process path uses this + the already-built
    # `data_source` DIRECTLY. Subprocess workers (--workers>1) rebuild the source
    # from the SAME ``source_config`` — no env side-channel can drift the two
    # apart (closes the worker-factory config-drop bug class).
    strategy = _build_strategy_for_cli(args.strategy, params)
    results = run_questions_parallel(
        descriptors=descriptors,
        strategy_id=args.strategy,
        params=params,
        run_cfg=run_cfg,
        source_config=source_config,
        diagnostics_dir=diag_dir,
        fills_dir=fills_dir,
        strike_for=strike_fn,
        hedge_data_path=_hedge_data_path_for(args),
        hedge_half_spread_bps=_hedge_half_spread_for(args),
        n_workers=n_workers,
        data_source=data_source,
        strategy=strategy,
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
        # R5: stamp the shared slot config_sig when running from a live slot so
        # the backtest report hash equals make engine-diag's hash for the same
        # slot (enables "is this sim comparable to the live slot?" verification).
        slot_strategy_cfg=getattr(args, "_slot_strategy_cfg", None),
    )
    # Concatenate per-question parquets into a single top-level
    # diagnostics.parquet / fills.parquet for downstream tooling.
    _concat_parquets(diag_dir, out_dir / "diagnostics.parquet")
    _concat_parquets(fills_dir, out_dir / "fills.parquet")
    logger.info(f"Report → {out_dir}/report.md")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Populate the polymarket cache: PM markets/trades (Gamma + CLOB) plus a
    coupled fetch of the Binance spot klines covering those markets, so strike
    resolution never lags the market cache (SHR-54)."""
    if args.data_source != "polymarket":
        raise SystemExit(f"fetch supports --data-source polymarket only, got {args.data_source!r}")

    from .core.source_config import PM_FLAVORS
    from .data.polymarket import PolymarketDataSource

    cache_root = Path(args.cache_root or os.environ.get(_ENV_PM_CACHE, "data/sim"))
    fcfg = PM_FLAVORS[args.pm_flavor]
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
    from ..marketdata.decision_input import from_backtest_params
    from .tuning import load_tuning_yaml, run_tuning_parallel

    tcfg = load_tuning_yaml(Path(args.grid))
    grid = tcfg.grids.get(args.strategy)
    if grid is None:
        raise SystemExit(
            f"No grid defined for strategy {args.strategy!r} in {args.grid}. "
            f"Known grid keys: {sorted(tcfg.grids)}"
        )

    # Event-array cache is default-ON (same as cmd_run). --cache-event-arrays
    # is a no-op kept for back-compat; --no-cache/--fresh disables it;
    # --rebuild-cache forces a one-time rebuild.  These are genuine runtime
    # toggles (not source-construction), so spawn workers still inherit them
    # via the environment.
    if getattr(args, "no_cache", False):
        os.environ["HLBT_NO_CACHE"] = "1"
    if getattr(args, "rebuild_cache", False):
        os.environ["HLBT_REBUILD_CACHE"] = "1"

    # ONE source-construction path: build the discover source here and ship the
    # SAME picklable SourceConfig to workers. The reference-resample cadence is a
    # sweepable param (vol_sampling_dt_seconds), so workers override it per cell;
    # the discover pass here doesn't depend on the cadence, so 60 is fine.
    # SHR-92: auto-derive warmup from the grid's fixed vol_lookback_seconds if
    # present, or from the max across the sweep range. Workers carry the warmup
    # in the SourceConfig and apply it when building their per-cell source.
    # The discover pass doesn't depend on warmup (no reference data loaded), so
    # any value is safe here; workers derive the correct warmup from the grid params.
    _cli_warmup_tune = getattr(args, "reference_warmup_seconds", None)
    # SHR-97: build live-faithful defaults from the resolver. The discover pass
    # doesn't depend on source/ticks, but the SourceConfig is shipped to workers;
    # baking mark+raw here ensures workers start from the live-faithful baseline
    # (they override dt per cell via with_reference_resample but keep ref_event +
    # ref_ticks from the shipped config). Use an empty params dict so the
    # resolver returns the track defaults (mark + raw, the live HL defaults).
    _resolved_tune = from_backtest_params({}, track_default_source="mark")
    # For the discover source, use 0 (warmup irrelevant for discover).
    # The per-cell warmup is derived from cell params in _run_one_cell.
    source_config = _source_config_from_args(
        args,
        reference_resample_seconds=60,
        reference_warmup_seconds=0,
        resolved=_resolved_tune,
    )
    data_source = source_config.build()
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

    run_cfg = _run_config_from_args(args, hedge_cfg_dict)

    out_dir = Path(args.out_dir) / args.run_id
    run_meta = tcfg.run if isinstance(tcfg.run, dict) else {}
    rows = list(run_tuning_parallel(
        strategy_id=args.strategy,
        grid=grid,
        source_config=source_config,
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
