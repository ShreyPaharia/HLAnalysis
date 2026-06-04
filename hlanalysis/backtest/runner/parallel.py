"""Shared process-pool helpers for backtests.

`run_questions_parallel` fans independent questions across worker processes;
`reconstruct_source` / `build_hedge_source` are the worker-side builders shared
with the tuning driver so the reconstruction logic lives in one place.
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

from .hftbt_runner import RunConfig, run_one_question
from ..core.data_source import QuestionDescriptor


def reconstruct_source(data_source_dotted: str):
    """Re-import + call a zero-arg factory named by a dotted path (worker side)."""
    mod_name, _, attr = data_source_dotted.rpartition(".")
    return getattr(importlib.import_module(mod_name), attr)()


def build_hedge_source(run_cfg: RunConfig, hedge_data_path: str | None,
                       hedge_half_spread_bps: float):
    """Build the hedge source in a worker, or None when hedging is off."""
    if not (run_cfg.hedge_enabled and hedge_data_path):
        return None
    from ..data.binance_perp import BinancePerpKlinesSource
    return BinancePerpKlinesSource(
        path=Path(hedge_data_path), symbol=run_cfg.hedge_symbol,
        half_spread_bps=hedge_half_spread_bps,
    )


@dataclass(frozen=True)
class QResult:
    idx: int
    realized_pnl_usd: float
    n_fills: int
    outcome: str


def build_strategy_for_run(strategy_id: str, params: dict):
    """Worker-safe strategy construction shared by the CLI and pool workers.

    Mirrors the CLI's strategy resolution: registry-constructible strategies
    plus the ``_dummy_enter_yes`` smoke-test escape hatch (not in the registry).
    Lives here (not cli.py) so spawn workers can call it without importing the
    CLI module.
    """
    # Importing strategy triggers auto-registration of all real strategies.
    import hlanalysis.strategy  # noqa: F401
    from ..core.registry import build, ids

    if strategy_id in ids():
        return build(strategy_id, params)
    if strategy_id == "_dummy_enter_yes":
        from ..data.synthetic import build_dummy_enter_strategy
        return build_dummy_enter_strategy(params)
    raise ValueError(
        f"Unknown --strategy: {strategy_id}. Registered: {ids()}."
    )


def _run_question_worker(args: tuple) -> QResult:
    (idx, q_id, strike, strategy_id, params, run_cfg_kwargs, data_source_dotted,
     diag_dir, fills_dir, hedge_data_path, hedge_half_spread_bps) = args

    data_source = reconstruct_source(data_source_dotted)
    run_cfg = RunConfig(**run_cfg_kwargs)
    strategy = build_strategy_for_run(strategy_id, params)

    # Wide window: cache-driven sources (PM) filter by end_ts ∈ [start,end);
    # re-discover everything, then map id → descriptor (mirrors tuning worker).
    all_descs = list(data_source.discover(start="1970-01-01", end="2999-12-31"))
    q = next((d for d in all_descs if d.question_id == q_id), None)
    if q is None:
        raise RuntimeError(f"worker could not re-map question_id {q_id!r}")

    hedge_source = build_hedge_source(run_cfg, hedge_data_path, hedge_half_spread_bps)
    hedge_events = None
    if hedge_source is not None:
        hedge_events = list(hedge_source.book_events(
            start_ts_ns=q.start_ts_ns, end_ts_ns=q.end_ts_ns))

    res = run_one_question(
        strategy,
        data_source,
        q,
        run_cfg,
        diagnostics_dir=Path(diag_dir) if diag_dir else None,
        fills_dir=Path(fills_dir) if fills_dir else None,
        strike=strike,
        hedge_events=hedge_events,
    )
    return QResult(idx, res.realized_pnl_usd or 0.0, len(res.fills),
                   data_source.resolved_outcome(q))


def run_questions_parallel(
    *,
    descriptors: list[QuestionDescriptor],
    strategy_id: str,
    params: dict,
    run_cfg: RunConfig,
    data_source_dotted: str,
    diagnostics_dir: Path | None,
    fills_dir: Path | None,
    strike_for: Callable[[QuestionDescriptor], float],
    hedge_data_path: str | None,
    hedge_half_spread_bps: float,
    n_workers: int,
) -> list[QResult]:
    """Run each descriptor's backtest, returning QResult in INPUT order.

    n_workers <= 1 runs in-process (clean tracebacks, cheap for tiny runs).
    """
    run_cfg_kwargs = asdict(run_cfg)
    work = [
        (i, q.question_id, strike_for(q), strategy_id, params, run_cfg_kwargs,
         data_source_dotted, str(diagnostics_dir) if diagnostics_dir else None,
         str(fills_dir) if fills_dir else None, hedge_data_path, hedge_half_spread_bps)
        for i, q in enumerate(descriptors)
    ]
    if n_workers <= 1:
        return sorted((_run_question_worker(w) for w in work), key=lambda r: r.idx)
    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=mp.get_context("spawn")) as ex:
        results = list(ex.map(_run_question_worker, work))
    return sorted(results, key=lambda r: r.idx)


__all__ = [
    "QResult",
    "reconstruct_source",
    "build_hedge_source",
    "run_questions_parallel",
]
