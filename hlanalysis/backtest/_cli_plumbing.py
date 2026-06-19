"""Shared plumbing for the ``hl-bt`` CLI.

Data-source construction helpers, shared config utilities, and the
miscellaneous small helpers consumed by both argument-parsing code and
command handlers.  This module has **no** argparse or command-handler code —
those live in ``_cli_args.py`` and ``_cli_commands.py`` respectively.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path

from ..marketdata.decision_input import DecisionInputConfig
from .core.data_source import QuestionDescriptor
from .core.source_config import SourceConfig

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
            pm_binance_bbo_product_type=(getattr(args, "pm_binance_bbo_product_type", None) or "perp"),
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
        _default_ref_event = resolved.reference_source if resolved is not None else "mark"
        _default_ref_ticks = resolved.reference_ticks if resolved is not None else "raw"
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


def assert_hl_cadence_match(source_config: SourceConfig, params: dict) -> None:
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
# Strike resolution helper
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
# Hedge config helpers
# ---------------------------------------------------------------------------


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


def _sim_risk_caps_from_args(args: argparse.Namespace):
    """Build a SimRiskCaps from cap args stashed by a --slot run, else None.

    None preserves bit-identical behaviour for non-slot / ad-hoc runs (the
    sim applies no inventory or daily-loss cap, as before).
    """
    inv = getattr(args, "sim_max_inventory_usd", None)
    conc = getattr(args, "sim_max_concurrent_positions", None)
    dlc = getattr(args, "sim_daily_loss_cap_usd", None)
    if inv is None and conc is None and dlc is None:
        return None
    from .halt_replay import SimRiskCaps

    return SimRiskCaps(
        daily_loss_cap_usd=dlc,
        daily_window_start_hour_utc=int(getattr(args, "sim_daily_window_start_hour_utc", 0) or 0),
        max_total_inventory_usd=inv,
        max_concurrent_positions=conc,
    )


def _run_config_from_args(args: argparse.Namespace, hedge_cfg: dict | None):
    """Assemble a :class:`RunConfig` from the shared run-config CLI args.

    Single builder used by both ``run`` and ``tune`` (they declare the same
    fee/tick/lot/slippage args via :func:`_add_run_config_args`). ``hedge_cfg``
    is the hedge dict produced by ``_extract_hedge_config`` (``run``) or the
    ``--hedge-*`` flags (``tune``); ``None`` means hedging is off.
    """
    from .runner.hftbt_runner import RunConfig

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
        # None (no --scan-mode and not a --slot run) → legacy 'fixed'.
        scan_mode=getattr(args, "scan_mode", None) or "fixed",
        scan_min_interval_seconds=getattr(args, "scan_min_interval_seconds", 0.2),
        scan_max_interval_seconds=getattr(args, "scan_max_interval_seconds", 2.0),
        min_inter_order_seconds=getattr(args, "min_inter_order_seconds", 0.0),
        ioc_fleeting_persistence_seconds=getattr(args, "ioc_fleeting_persistence_seconds", 0.0),
        sim_risk_caps=_sim_risk_caps_from_args(args),
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

    ``--slot-config-asof <YYYY-MM-DD>`` sources the config from the last git
    commit touching ``config/strategy.yaml`` on or before that date. Mutually
    exclusive with a non-default ``--slot-config``.
    """
    if getattr(args, "slot", None):
        if getattr(args, "config", None):
            raise SystemExit("--slot and --config are mutually exclusive")

        # --slot-config-asof: resolve config from git history.
        asof = getattr(args, "slot_config_asof", None)
        if asof is not None:
            _default_slot_config = "config/strategy.yaml"
            if args.slot_config != _default_slot_config:
                raise SystemExit("--slot-config-asof and --slot-config are mutually exclusive")
            from .config_asof import _repo_root_from_file, resolve_config_asof

            repo_root = _repo_root_from_file(__file__)
            try:
                commit_hash, tmp_path = resolve_config_asof(asof, repo_root)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            from ..engine.config import load_strategies_config as _lsc
            from ..engine.config import strategy_config_sig

            _preview_cfgs = _lsc(tmp_path)
            _preview_by_alias = {c.account_alias: c for c in _preview_cfgs.strategies}
            if args.slot in _preview_by_alias:
                _preview_sig = strategy_config_sig(_preview_by_alias[args.slot])
            else:
                _preview_sig = "(slot not found in asof config)"
            import subprocess as _sp

            _commit_date = _sp.run(
                ["git", "log", "-1", "--format=%ci", commit_hash],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            print(
                f"[slot-config-asof] resolved commit: {commit_hash[:12]}  "
                f"date: {_commit_date}  config_sig({args.slot}): {_preview_sig}"
            )
            # Replace args.slot_config with the resolved temp-file path so the
            # standard load_strategies_config call below reads the historical config.
            args.slot_config = str(tmp_path)
            # Record the temp path so cmd_run can delete it after use.
            args._slot_config_asof_tmp = str(tmp_path)

        from ..engine.config import load_strategies_config
        from .slot_config import backtest_params_from_slot

        cfgs = load_strategies_config(Path(args.slot_config))
        by_alias = {c.account_alias: c for c in cfgs.strategies}
        if args.slot not in by_alias:
            raise SystemExit(f"--slot {args.slot!r} not found in {args.slot_config}; available: {sorted(by_alias)}")
        slot_cfg = by_alias[args.slot]
        # Mirror the live engine's scan cadence so a slot run evaluates intraday
        # exits (exit_safety_d / exit_edge) at the SAME granularity the engine
        # runs. The legacy 60s `fixed` scan misses sub-minute exit triggers and
        # badly misvalidates a slot (e.g. v31's mid-hold exits fire minutes late
        # in fixed mode). Default to event-driven, with the floor/ceiling taken
        # from the slot's live GlobalRiskConfig. An explicit --scan-mode wins
        # (A/B override); `None` means "not passed".
        if getattr(args, "scan_mode", None) is None:
            args.scan_mode = "event"
            args.scan_min_interval_seconds = slot_cfg.global_.scan_min_interval_seconds
            args.scan_max_interval_seconds = slot_cfg.global_.scan_max_interval_seconds
        # Lift the slot's live risk envelope into the sim so a slot run/tune
        # enforces the SAME inventory + daily-loss caps the engine does. Without
        # this the sim over-enters notional live's RiskGate would have blocked
        # (the $100 burn-in inventory cap binds at 2 positions live), making sim
        # PnL over-optimistic — exactly the figure a sizing decision must trust.
        # Consumed by _run_config_from_args → RunConfig.sim_risk_caps.
        args.sim_max_inventory_usd = slot_cfg.global_.max_total_inventory_usd
        args.sim_max_concurrent_positions = slot_cfg.global_.max_concurrent_positions
        args.sim_daily_loss_cap_usd = slot_cfg.global_.daily_loss_cap_usd
        args.sim_daily_window_start_hour_utc = slot_cfg.global_.daily_window_start_hour_utc
        strategy_id, params = backtest_params_from_slot(slot_cfg, klass=getattr(args, "slot_class", None))
        args.strategy = strategy_id
        # Stash the StrategyConfig on args so cmd_run can stamp the shared
        # config sig (strategy_config_sig) in the report without a second YAML
        # load. Non-slot runs leave this unset; cmd_run checks with getattr.
        args._slot_strategy_cfg = slot_cfg
        return params
    if not getattr(args, "config", None) or not getattr(args, "strategy", None):
        raise SystemExit("provide --slot, or both --strategy and --config")
    return json.loads(Path(args.config).read_text())


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
