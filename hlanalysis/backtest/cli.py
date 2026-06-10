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

from .core.data_source import QuestionDescriptor
from .core.registry import ids as registry_ids
from .core.source_config import PM_FLAVORS, SourceConfig
from .report import write_single_run_report, write_tuning_report
from .runner.hftbt_runner import RunConfig, run_one_question
from .runner.parallel import run_questions_parallel
from .runner.result import summarise_run
from ..marketdata.decision_input import DecisionInputConfig, from_backtest_params

# Cache/data-root location defaults. These are user-facing "where is my data"
# knobs (documented on --cache-root): the parent resolves them once and bakes
# the concrete path into the SourceConfig, so workers read NO environment. They
# are NOT the source-construction side-channel — every construction knob now
# travels inside the picklable SourceConfig in the work tuple.
_ENV_PM_CACHE = "HLBT_PM_CACHE_ROOT"
_ENV_HL_DATA = "HLBT_HL_DATA_ROOT"
_ENV_PM_NBA_CACHE = "HLBT_PM_NBA_CACHE_ROOT"


# ---------------------------------------------------------------------------
# Data source resolution
# ---------------------------------------------------------------------------

def _source_config_from_args(
    args: argparse.Namespace,
    *,
    reference_resample_seconds: int = 60,
    reference_warmup_seconds: int = 0,
    resolved: DecisionInputConfig | None = None,
) -> SourceConfig:
    """Build the picklable :class:`SourceConfig` from parsed CLI args.

    ``cache_root`` keeps its documented env default (resolved HERE in the
    parent); every other construction knob is read straight off ``args``. The
    resulting config is used both to build the in-process source and to ship to
    spawn workers — one construction path, no env side-channel.

    SHR-97: when ``resolved`` is provided (a ``DecisionInputConfig`` built from
    ``from_backtest_params``), it supplies the live-faithful defaults for
    ``hl_ref_event`` and ``hl_ref_ticks``. The CLI flags act as explicit
    overrides: when the user passes ``--ref-event`` or ``--reference-ticks``
    (non-None), that value wins; when omitted (None), the config-derived value
    is used. This makes the default backtest wiring live-faithful (mark + raw)
    while keeping the flags as A/B override knobs.
    """
    name = args.data_source
    cache_root = getattr(args, "cache_root", None)
    if name == "polymarket":
        return SourceConfig(
            kind="polymarket",
            cache_root=cache_root or os.environ.get(_ENV_PM_CACHE),
            pm_flavor=getattr(args, "pm_flavor", None) or "btc_updown",
            pm_reference_source=getattr(args, "pm_reference_source", None) or "klines",
            pm_book_source=getattr(args, "pm_book_source", None) or "synthetic",
            pm_binance_bbo_product_type=(
                getattr(args, "pm_binance_bbo_product_type", None) or "perp"
            ),
            pm_liquidity_profile_path=getattr(args, "pm_liquidity_profile", None),
            reference_resample_seconds=reference_resample_seconds,
        )
    if name == "hl_hip4":
        # SHR-97: derive live-faithful defaults from the resolver when provided;
        # explicit CLI flags override. ref_event default was "bbo" (non-live);
        # reference_ticks default was "bars" (non-live). Both now resolve to the
        # config-derived live-faithful value (mark + raw) when not explicitly set.
        _cli_ref_event = getattr(args, "ref_event", None)
        _cli_ref_ticks = getattr(args, "reference_ticks", None)
        _default_ref_event = (resolved.reference_source if resolved is not None else "mark")
        _default_ref_ticks = (resolved.reference_ticks if resolved is not None else "raw")
        return SourceConfig(
            kind="hl_hip4",
            cache_root=cache_root or os.environ.get(_ENV_HL_DATA),
            hl_ref_source=getattr(args, "ref_source", None) or "hl_perp",
            hl_ref_event=_cli_ref_event or _default_ref_event,
            hl_ref_ticks=_cli_ref_ticks or _default_ref_ticks,
            reference_resample_seconds=reference_resample_seconds,
            reference_warmup_seconds=reference_warmup_seconds,
        )
    if name == "pm_nba":
        return SourceConfig(
            kind="pm_nba",
            cache_root=cache_root or os.environ.get(_ENV_PM_NBA_CACHE),
        )
    if name == "synthetic":
        return SourceConfig(kind="synthetic", cache_root=cache_root)
    raise SystemExit(f"Unknown --data-source: {name}")


def _derive_reference_warmup_seconds(params: dict, *, data_source: str) -> int:
    """Auto-derive the reference warm-up window from ``params``.

    For ``hl_hip4`` sources, returns the maximum ``vol_lookback_seconds`` across
    all param classes so the reference buffer is warm enough for any class's
    lookback at market open. For other sources (PM, synthetic), returns 0 because
    PM derives σ differently and doesn't benefit from HL-style BBO warm-up.

    Handles both:
    - flat params dict: ``{"vol_lookback_seconds": 900}``
    - per-class nested dict: ``{"binary": {"vol_lookback_seconds": 3600},
                                 "bucket": {"vol_lookback_seconds": 900}}``
    """
    if data_source != "hl_hip4":
        return 0
    # Collect all vol_lookback_seconds values from flat or nested params.
    values: list[int] = []
    for v in params.values():
        if isinstance(v, dict):
            # Nested per-class params.
            if "vol_lookback_seconds" in v:
                values.append(int(v["vol_lookback_seconds"]))
        elif isinstance(v, (int, float)):
            pass  # handled below for flat case
    # Flat case: top-level key.
    if "vol_lookback_seconds" in params:
        flat_val = params["vol_lookback_seconds"]
        if isinstance(flat_val, (int, float)):
            values.append(int(flat_val))
    return max(values) if values else 0


def _resolve_reference_warmup_seconds(
    params: dict,
    *,
    data_source: str,
    cli_override: int | None,
) -> int:
    """Resolve the effective reference warm-up window.

    Priority (highest first):
    1. ``cli_override`` when not None — the ``--reference-warmup-seconds`` flag was
       explicitly passed (including 0 to disable).
    2. Auto-derived from ``params`` via :func:`_derive_reference_warmup_seconds`.
    """
    if cli_override is not None:
        return int(cli_override)
    return _derive_reference_warmup_seconds(params, data_source=data_source)


def assert_hl_cadence_match(source_config: "SourceConfig", params: dict) -> None:
    """Raise ``ValueError`` if an hl_hip4 source's resample period disagrees with
    the strategy's ``vol_sampling_dt_seconds``.

    The reference OHLC resample period MUST equal ``vol_sampling_dt_seconds`` so
    the backtest evaluates σ / safety_d / p_model at the same cadence as the live
    engine.  A silent mismatch (e.g. resample=60 while dt=5 is live) makes the
    sim untestable against the real cadence.

    Per-class dt note: binary and bucket use different cadences (dt=5 vs dt=2);
    run them as separate ``hl-bt run`` invocations — one cadence per invocation.

    Polymarket and pm_nba sources are exempt (they derive cadence differently and
    do not gate on this param for correctness).
    """
    if source_config.kind != "hl_hip4":
        return
    if "vol_sampling_dt_seconds" not in params:
        return
    expected = int(params["vol_sampling_dt_seconds"])
    actual = source_config.reference_resample_seconds
    if actual != expected:
        raise ValueError(
            f"hl_hip4 reference_resample_seconds={actual} does not match "
            f"strategy vol_sampling_dt_seconds={expected}. "
            "The σ-resample cadence must equal vol_sampling_dt_seconds so the "
            "backtest evaluates at the live cadence. Pass "
            f"reference_resample_seconds={expected} when building SourceConfig."
        )


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


def _run_config_from_args(
    args: argparse.Namespace, hedge_cfg: dict | None
) -> RunConfig:
    """Assemble a :class:`RunConfig` from the shared run-config CLI args.

    Single builder used by both ``run`` and ``tune`` (they declare the same
    fee/tick/lot/slippage args via :func:`_add_run_config_args`). ``hedge_cfg``
    is the hedge dict produced by ``_extract_hedge_config`` (``run``) or the
    ``--hedge-*`` flags (``tune``); ``None`` means hedging is off.
    """
    kwargs: dict = dict(
        scanner_interval_seconds=args.scanner_interval_seconds,
        tick_size=args.tick_size,
        lot_size=args.lot_size,
        slippage_bps=args.slippage_bps,
        fee_taker=args.fee_taker,
        fee_model=args.fee_model,
        fee_rate=args.fee_rate,
        book_depth_assumption=args.depth,
        order_latency_ms=args.order_latency_ms,
        scan_mode=getattr(args, "scan_mode", "fixed"),
        scan_min_interval_seconds=getattr(args, "scan_min_interval_seconds", 0.2),
        scan_max_interval_seconds=getattr(args, "scan_max_interval_seconds", 2.0),
    )
    if hedge_cfg is not None:
        kwargs["hedge_enabled"] = hedge_cfg["hedge_enabled"]
        kwargs["hedge_symbol"] = hedge_cfg["hedge_symbol"]
        kwargs["hedge_tick_size"] = hedge_cfg["hedge_tick_size"]
        kwargs["hedge_lot_size"] = hedge_cfg["hedge_lot_size"]
        kwargs["hedge_slippage_bps"] = hedge_cfg["hedge_slippage_bps"]
        kwargs["hedge_fee_bps"] = hedge_cfg["hedge_fee_bps"]
    return RunConfig(**kwargs)


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


def _hedge_data_path_for(args: argparse.Namespace) -> str | None:
    """Extract the hedge data path from parsed CLI args (None when not provided)."""
    return getattr(args, "hedge_data_path", None) or None


def _hedge_half_spread_for(args: argparse.Namespace) -> float:
    """Extract hedge half-spread bps from parsed CLI args."""
    return float(getattr(args, "hedge_half_spread_bps", 1.0))


def _load_run_params(args: argparse.Namespace) -> dict:
    """Resolve the run's strategy params from either a live slot or a JSON file.

    SHR-99: ``--slot <alias>`` sources params from the live ``strategy.yaml``
    via the single converter (``backtest_params_from_slot``), so a sim run is
    config-faithful by construction — no hand-reconstructed JSON to drift. The
    converter sets ``args.strategy`` to the matching registry id. ``--config``
    remains the explicit JSON path for ad-hoc params. The two are mutually
    exclusive.
    """
    if getattr(args, "slot", None):
        if getattr(args, "config", None):
            raise SystemExit("--slot and --config are mutually exclusive")
        from ..engine.config import load_strategies_config
        from .slot_config import backtest_params_from_slot

        cfgs = load_strategies_config(Path(args.slot_config))
        by_alias = {c.account_alias: c for c in cfgs.strategies}
        if args.slot not in by_alias:
            raise SystemExit(
                f"--slot {args.slot!r} not found in {args.slot_config}; "
                f"available: {sorted(by_alias)}"
            )
        strategy_id, params = backtest_params_from_slot(
            by_alias[args.slot], klass=getattr(args, "slot_class", None)
        )
        args.strategy = strategy_id
        return params
    if not getattr(args, "config", None) or not getattr(args, "strategy", None):
        raise SystemExit("provide --slot, or both --strategy and --config")
    return json.loads(Path(args.config).read_text())


def cmd_run(args: argparse.Namespace) -> int:
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

    hedge_source = _build_hedge_source(args.hedge_data_path, hedge_cfg)

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
    from .tuning import load_tuning_yaml, run_tuning_parallel

    tcfg = load_tuning_yaml(Path(args.grid))
    grid = tcfg.grids.get(args.strategy)
    if grid is None:
        raise SystemExit(
            f"No grid defined for strategy {args.strategy!r} in {args.grid}. "
            f"Known grid keys: {sorted(tcfg.grids)}"
        )

    # Event-array cache is opt-in (default OFF). --rebuild-cache implies it.
    # These are genuine runtime toggles (not source-construction), so spawn
    # workers still inherit them via the environment.
    if getattr(args, "cache_event_arrays", False) or getattr(args, "rebuild_cache", False):
        os.environ["HLBT_CACHE_EVENT_ARRAYS"] = "1"
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


def _add_run_config_args(parser: argparse.ArgumentParser) -> None:
    """Declare the run-config args shared verbatim by ``run`` and ``tune``
    (fee / tick / lot / slippage / scan-cadence). Consumed by
    :func:`_run_config_from_args`."""
    parser.add_argument("--scanner-interval-seconds", type=int, default=60)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--fee-taker", type=float, default=0.0)
    parser.add_argument(
        "--fee-model",
        choices=["flat", "pm_binary"],
        default="flat",
        help="Binary-leg fee model. 'flat' uses --fee-taker as constant %% of "
        "notional (HL/synthetic). 'pm_binary' uses Polymarket's curve "
        "fee = qty * --fee-rate * p * (1-p).",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.07,
        help="feeRate for --fee-model pm_binary. PM crypto = 0.07; sports = 0.03; "
        "politics/tech/finance/mentions = 0.04; econ/culture/weather = 0.05.",
    )
    parser.add_argument("--tick-size", type=float, default=0.001)
    parser.add_argument("--lot-size", type=float, default=1.0)
    parser.add_argument(
        "--depth",
        type=float,
        default=None,
        help="Optional explicit fill-size cap in lots (SHR-79). Default=None "
        "(unlimited — the real recorded book governs fill size via "
        "partial_fill_exchange). Use e.g. --depth 500 to hard-cap fills.",
    )
    parser.add_argument(
        "--order-latency-ms",
        type=float,
        default=50.0,
        help="Constant order round-trip latency in milliseconds (SHR-79). "
        "Default=50 ms (empirical HL HIP-4 median). Set to 0 for legacy "
        "zero-latency behaviour.",
    )
    # SHR-95: event-driven scan mode (mirrors live engine cadence).
    parser.add_argument(
        "--scan-mode",
        choices=["fixed", "event"],
        default="fixed",
        dest="scan_mode",
        help="Scan cadence mode. 'fixed' (default) evaluates every "
        "--scanner-interval-seconds (legacy behaviour, back-compat). "
        "'event' evaluates on each book/reference update, clamped between "
        "--scan-min-interval-seconds (floor) and --scan-max-interval-seconds "
        "(ceiling), mirroring the live engine's event-driven cadence.",
    )
    parser.add_argument(
        "--scan-min-interval-seconds",
        type=float,
        default=0.2,
        dest="scan_min_interval_seconds",
        help="(event scan mode only) Minimum time between consecutive scans "
        "in seconds. Mirrors live scan_min_interval_seconds=0.2. Default=0.2.",
    )
    parser.add_argument(
        "--scan-max-interval-seconds",
        type=float,
        default=2.0,
        dest="scan_max_interval_seconds",
        help="(event scan mode only) Maximum time between consecutive scans "
        "(idle-backoff ceiling) in seconds. Mirrors live scan_max_interval_seconds=2.0. "
        "Default=2.0.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hl-bt")
    sp = p.add_subparsers(dest="cmd", required=True)

    ps = sp.add_parser("strategies", help="List registered strategy ids")
    ps.set_defaults(func=cmd_strategies)

    pr = sp.add_parser("run", help="Run one strategy config across discovered questions")
    pr.add_argument("--strategy", default=None,
                    help="Registry strategy id. Required unless --slot is given "
                         "(--slot derives it from the live config).")
    pr.add_argument(
        "--data-source",
        required=True,
        choices=["synthetic", "polymarket", "hl_hip4", "pm_nba"],
    )
    pr.add_argument("--config", default=None,
                    help="JSON file of param dict. Mutually exclusive with --slot.")
    pr.add_argument(
        "--slot", default=None,
        help="(SHR-99) Run from a live strategy.yaml slot by account_alias "
             "(e.g. v31). Sources the EXACT live decision config via the engine "
             "config builders — no hand-written JSON to drift. Sets --strategy.",
    )
    pr.add_argument(
        "--slot-config", default="config/strategy.yaml", dest="slot_config",
        help="Path to the live strategy config for --slot (default "
             "config/strategy.yaml).",
    )
    pr.add_argument(
        "--slot-class", default=None, dest="slot_class",
        help="(--slot only) Select a per-class override config (e.g. priceBucket) "
             "matching the engine's per-question.klass resolution. Omit for the "
             "slot default. Run one class per invocation (one σ cadence).",
    )
    pr.add_argument("--out-dir", required=True)
    pr.add_argument("--start", default=None)
    pr.add_argument("--end", default=None)
    _add_run_config_args(pr)
    pr.add_argument("--skip-markets", type=int, default=0)
    pr.add_argument("--max-markets", type=int, default=None)
    pr.add_argument(
        "--cache-root",
        default=None,
        help="Override cache/data root (env: HLBT_PM_CACHE_ROOT / HLBT_HL_DATA_ROOT).",
    )
    pr.add_argument(
        "--pm-flavor",
        choices=sorted(PM_FLAVORS),
        default="btc_updown",
        help="(polymarket only) Which PM series + reference asset to load. "
        "Default: btc_updown.",
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
        "--ref-event",
        choices=["bbo", "mark"],
        default=None,
        help="(hl_hip4 only) Which reference price stream feeds σ/p_model. "
        "`mark` (live-faithful default, SHR-97) matches the LIVE engine "
        "(reference_sigma_source default = mark). `bbo` reads bid/ask mid "
        "(bounces more → higher σ). Default: config-derived (mark).",
    )
    pr.add_argument(
        "--reference-ticks",
        choices=["bars", "raw"],
        default=None,
        dest="reference_ticks",
        help="(hl_hip4 only) Reference-tick mode (SHR-93). `raw` (live-faithful "
        "default, SHR-97) emits one event per recorded tick and lets the shared "
        "MarketState bucket them, making last_mark the instantaneous raw tick "
        "price (live-parity). `bars` pre-buckets raw ticks into OHLC bars "
        "(legacy A/B override). Default: config-derived (raw).",
    )
    pr.add_argument(
        "--pm-reference-source",
        choices=["klines", "binance_bbo", "klines_1s"],
        default="klines",
        help="(polymarket only) Reference-feed source. `klines` (default) reads "
        "cached 1m Binance klines (12-month corpus). `binance_bbo` reads "
        "recorded Binance BBO ticks and buckets to vol_sampling_dt_seconds "
        "(BBO-overlap window only). `klines_1s` pulls genuine Binance 1s klines "
        "(cached under btc_klines_1s/) and buckets to vol_sampling_dt_seconds — "
        "the on-demand-klines counterpart to binance_bbo for the ref-equivalence "
        "experiment.",
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
    pr.add_argument(
        "--pm-liquidity-profile",
        default=None,
        dest="pm_liquidity_profile",
        help="(polymarket synthetic mode only) Path to a JSON liquidity-profile "
        "produced by scripts/calibrate_pm_liquidity.py. When supplied, the "
        "synthetic book builder uses per-price-bucket half_spread/depth from "
        "the profile instead of the flat 0.005/10000 defaults.",
    )
    pr.add_argument(
        "--reference-warmup-seconds",
        type=int,
        default=None,
        dest="reference_warmup_seconds",
        help="(hl_hip4 only) Reference warm-up prefix in seconds (SHR-92). "
        "When > 0, pre-loads reference rows from [start - N, start) into "
        "MarketState so σ is warm at market open. Default: auto-derived from "
        "max(vol_lookback_seconds) across param classes. Pass 0 to disable.",
    )
    pr.add_argument("--workers", type=int, default=1,
                    help="Parallel worker processes for independent markets "
                         "(default 1 = serial). Use up to #cores for big runs.")
    pr.add_argument("--no-cache", "--fresh", dest="no_cache", action="store_true",
                    help="Disable the event-array cache for this run (it is "
                         "default-ON) and force a fresh build. Use when you "
                         "suspect a stale/poisoned cached entry.")
    pr.add_argument("--cache-event-arrays", action="store_true",
                    help="(deprecated, no-op) The event-array cache is now "
                         "default-ON; this flag is kept for back-compat.")
    pr.add_argument("--rebuild-cache", action="store_true",
                    help="Ignore cached event arrays and rebuild them once "
                         "(then repopulate the cache).")
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
        choices=sorted(PM_FLAVORS),
        default="btc_updown",
        help="(polymarket only) Which PM series + reference asset to load. "
        "Default: btc_updown.",
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
        choices=sorted(PM_FLAVORS),
        default="btc_updown",
        help="(polymarket only) Which PM series + reference asset to load. "
        "Default: btc_updown.",
    )
    pt.add_argument("--start", default=None)
    pt.add_argument("--end", default=None)
    _add_run_config_args(pt)
    pt.add_argument("--skip-markets", type=int, default=0)
    pt.add_argument("--max-markets", type=int, default=None)
    pt.add_argument("--top-k", type=int, default=10)
    pt.add_argument("--no-cache", "--fresh", dest="no_cache", action="store_true",
                    help="Disable the event-array cache for this sweep (it is "
                         "default-ON) and force fresh builds.")
    pt.add_argument("--cache-event-arrays", action="store_true",
                    help="(deprecated, no-op) The event-array cache is now "
                         "default-ON; kept for back-compat.")
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
    pt.add_argument(
        "--pm-liquidity-profile",
        default=None,
        dest="pm_liquidity_profile",
        help="(polymarket synthetic mode only) Path to a JSON liquidity-profile "
        "produced by scripts/calibrate_pm_liquidity.py. When supplied, the "
        "synthetic book builder uses per-price-bucket half_spread/depth from "
        "the profile instead of the flat 0.005/10000 defaults.",
    )
    pt.add_argument(
        "--pm-reference-source",
        choices=["klines", "binance_bbo", "klines_1s"],
        default="klines",
        help="(polymarket only) Reference-feed source for the sweep. `klines` "
        "(default) = cached 1m Binance klines. `klines_1s` = genuine Binance 1s "
        "klines (cached under <asset>_klines_1s/) bucketed to the per-cell "
        "vol_sampling_dt_seconds — use with a dt<60 grid for live-parity. "
        "`binance_bbo` = recorded Binance BBO ticks.",
    )
    pt.add_argument(
        "--pm-book-source",
        choices=["synthetic", "recorded"],
        default="synthetic",
        help="(polymarket only) Fill-book source for the sweep. `synthetic` "
        "(default) builds a flat 1-level book per trade print + `1−p` parity "
        "(calibratable via --pm-liquidity-profile). `recorded` feeds real L2.",
    )
    pt.add_argument(
        "--reference-warmup-seconds",
        type=int,
        default=None,
        dest="reference_warmup_seconds",
        help="(hl_hip4 only) Reference warm-up prefix in seconds (SHR-92). "
        "Override the per-cell auto-derivation from vol_lookback_seconds. "
        "Pass 0 to disable warm-up entirely.",
    )
    pt.add_argument(
        "--reference-ticks",
        choices=["bars", "raw"],
        default=None,
        dest="reference_ticks",
        help="(hl_hip4 only) Reference-tick mode (SHR-93). `raw` (live-faithful "
        "default, SHR-97) emits one event per recorded tick for live-parity. "
        "`bars` pre-buckets raw ticks — the legacy A/B override. "
        "Default: config-derived (raw).",
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
