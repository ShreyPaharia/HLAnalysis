"""Grid + walk-forward parallel tuning. Strategy lookup goes through the registry.

The grid YAML is keyed by strategy id under the top-level ``grids`` key:

```yaml
grids:
  v1_late_resolution:
    tte_min_seconds: [60, 300]
    ...
  v2_model_edge:
    ...
run:
  train_markets: 60
  test_markets: 15
  step_markets: 15
```
"""
from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml

from .core.data_source import DataSource, QuestionDescriptor
from .core.registry import build as build_strategy
from .core.source_config import SourceConfig
from .runner.hftbt_runner import RunConfig, run_one_question
from .runner.parallel import parent_package_root, worker_path_init
from .runner.result import RunSummary, summarise_run
from .runner.walkforward import walk_forward_splits


# ---------------------------------------------------------------------------
# YAML grid config
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TuningConfig:
    grids: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)


def load_tuning_yaml(path: Path) -> TuningConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return TuningConfig(
        grids=raw.get("grids", {}),
        run=raw.get("run", {}),
    )


# ---------------------------------------------------------------------------
# Grid iteration
# ---------------------------------------------------------------------------

def iter_grid(grid: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# ---------------------------------------------------------------------------
# Single-process driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TuningCellResult:
    params: dict[str, Any]
    train_ids: list[str]
    test_ids: list[str]
    summary: RunSummary


def _cell_key(params: dict[str, Any], test_ids: list[str]) -> tuple:
    return (tuple(sorted(params.items())), tuple(test_ids))


def _load_completed_cells(log_path: Path) -> set[tuple]:
    if not log_path.exists():
        return set()
    completed: set[tuple] = set()
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        params = row.get("params") or {}
        test_ids = row.get("test_ids") or []
        completed.add(_cell_key(params, test_ids))
    return completed


def run_tuning(
    *,
    strategy_id: str,
    grid: dict[str, list[Any]],
    data_source: DataSource,
    descriptors: list[QuestionDescriptor],
    run_cfg: RunConfig,
    train: int,
    test: int,
    step: int,
    out_dir: Path,
    strike_for: Callable[[QuestionDescriptor], float] = lambda _q: 0.0,
) -> Iterator[TuningCellResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = list(
        walk_forward_splits(
            descriptors, train=train, test=test, step=step, drop_short_tail=True
        )
    )
    log_path = out_dir / "results.jsonl"
    completed = _load_completed_cells(log_path)
    with log_path.open("a") as f:
        for params in iter_grid(grid):
            strategy = build_strategy(strategy_id, params)
            for tr, te in splits:
                test_ids = [q.question_id for q in te]
                if _cell_key(params, test_ids) in completed:
                    continue
                pnl: list[float] = []
                n_trades = 0
                for q in te:
                    res = run_one_question(
                        strategy,
                        data_source,
                        q,
                        run_cfg,
                        strike=strike_for(q),
                    )
                    pnl.append(res.realized_pnl_usd or 0.0)
                    n_trades += len(res.fills)
                summary = summarise_run(pnl, n_trades=n_trades)
                cell = TuningCellResult(
                    params=params,
                    train_ids=[q.question_id for q in tr],
                    test_ids=test_ids,
                    summary=summary,
                )
                f.write(
                    json.dumps(
                        {
                            "params": params,
                            "n_train": len(tr),
                            "n_test": len(te),
                            "train_ids": cell.train_ids,
                            "test_ids": cell.test_ids,
                            "summary": asdict(summary),
                        }
                    )
                    + "\n"
                )
                f.flush()
                yield cell


# ---------------------------------------------------------------------------
# Parallel driver
# ---------------------------------------------------------------------------


def _set_inproc_memo_worker_env(n_workers: int) -> None:
    """Tell spawn workers how many of them share the box so each self-limits its
    in-proc bundle memo to total_budget / n_workers (SHR-71).

    Each spawn worker holds its OWN module-global memo, so a per-process-only
    bound multiplies to N × per-process aggregate RAM. Setting this env BEFORE
    the ProcessPoolExecutor spawns means children inherit it (spawn copies the
    parent environment) and ``_inproc_max_bytes`` divides the total budget by it.
    """
    os.environ["HLBT_INPROC_BUNDLE_MEMO_WORKERS"] = str(max(1, int(n_workers)))


def _run_one_cell(args: tuple) -> dict:
    (
        strategy_id,
        params,
        tr_ids,
        te_ids_with_strikes,
        run_cfg_kwargs,
        source_config,
        hedge_data_path,
        hedge_half_spread_bps,
    ) = args

    from .runner.parallel import build_hedge_source

    # The reference resample period MUST track this cell's vol_sampling_dt_seconds
    # (a per-cell grid param) — otherwise the source reverts to the default 60s
    # while the strategy annualizes at the cell's dt, inflating sigma and gating
    # every tick (the dt=5 regression). The picklable SourceConfig carries every
    # other construction knob already; only the cadence is per-cell, so override
    # it here from THIS cell's params before building.
    _dt = int(params.get("vol_sampling_dt_seconds", 60))
    source_config = source_config.with_reference_resample(_dt)

    # SHR-92: reference warm-up prefix. If the shipped SourceConfig already has
    # a non-zero warmup (e.g. set via --reference-warmup-seconds by the operator),
    # keep it. Otherwise auto-derive from this cell's vol_lookback_seconds so the
    # warm-up window covers the full lookback regardless of which param cell wins.
    # This mirrors how with_reference_resample overrides the cadence per cell.
    if source_config.reference_warmup_seconds == 0:
        from .cli import _derive_reference_warmup_seconds
        _warmup = _derive_reference_warmup_seconds(params, data_source="hl_hip4")
        if _warmup > 0:
            source_config = source_config.with_reference_warmup(_warmup)

    # A sweep replays each question across many param cells; the built
    # event-array bundle is param-independent, so memoize it in-process to skip
    # cache_key (file stat) + npz inflate on every cell after the first.
    # setdefault so an operator can still force it off with HLBT_INPROC_BUNDLE_MEMO=0.
    os.environ.setdefault("HLBT_INPROC_BUNDLE_MEMO", "1")

    data_source = source_config.build()

    strategy = build_strategy(strategy_id, params)
    run_cfg = RunConfig(**run_cfg_kwargs)

    # Build hedge source lazily (one per worker process, re-used across questions).
    # We ship the file path rather than the parsed list to avoid pickle overhead.
    hedge_source = build_hedge_source(run_cfg, hedge_data_path, hedge_half_spread_bps)

    pnl: list[float] = []
    n_trades = 0
    # Use a wide window so cache-driven sources (e.g. PolymarketDataSource,
    # which filters by `end_ts_ns ∈ [start, end)`) actually return all
    # cached questions, not an empty set. The parent process already applied
    # the user's --start/--end during the initial discover; workers just
    # need to re-map question_id → descriptor across all cached entries.
    all_descs = list(data_source.discover(start="1970-01-01", end="2999-12-31"))
    for q_id, strike in te_ids_with_strikes:
        match = next(
            (d for d in all_descs if d.question_id == q_id),
            None,
        )
        if match is None:
            continue
        hedge_events = None
        if hedge_source is not None:
            hedge_events = list(
                hedge_source.book_events(
                    start_ts_ns=match.start_ts_ns, end_ts_ns=match.end_ts_ns
                )
            )
        res = run_one_question(
            strategy, data_source, match, run_cfg,
            strike=strike, hedge_events=hedge_events,
        )
        pnl.append(res.realized_pnl_usd or 0.0)
        n_trades += len(res.fills)
    summary = summarise_run(pnl, n_trades=n_trades)
    return {
        "params": params,
        "n_train": len(tr_ids),
        "n_test": len(te_ids_with_strikes),
        "train_ids": list(tr_ids),
        "test_ids": [t[0] for t in te_ids_with_strikes],
        "summary": asdict(summary),
    }


def run_tuning_parallel(
    *,
    strategy_id: str,
    grid: dict[str, list[Any]],
    source_config: SourceConfig,
    descriptors: list[QuestionDescriptor],
    run_cfg: RunConfig,
    train: int,
    test: int,
    step: int,
    out_dir: Path,
    n_workers: int,
    strike_for: Callable[[QuestionDescriptor], float] = lambda _q: 0.0,
    hedge_data_path: str | None = None,
    hedge_half_spread_bps: float = 1.0,
) -> Iterator[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = list(
        walk_forward_splits(
            descriptors, train=train, test=test, step=step, drop_short_tail=True
        )
    )
    log_path = out_dir / "results.jsonl"
    completed = _load_completed_cells(log_path)
    run_cfg_kwargs = asdict(run_cfg)

    work: list[tuple] = []
    for params in iter_grid(grid):
        for tr, te in splits:
            te_with_strikes = [(q.question_id, strike_for(q)) for q in te]
            te_ids = [t[0] for t in te_with_strikes]
            if _cell_key(params, te_ids) in completed:
                continue
            work.append(
                (
                    strategy_id,
                    params,
                    tuple(q.question_id for q in tr),
                    tuple(te_with_strikes),
                    run_cfg_kwargs,
                    source_config,
                    hedge_data_path,
                    hedge_half_spread_bps,
                )
            )

    # Make the per-worker in-proc bundle memo budget worker-aware BEFORE spawning
    # so each child self-limits to total/n_workers (SHR-71 OOM guard).
    _set_inproc_memo_worker_env(n_workers)

    with log_path.open("a") as f, ProcessPoolExecutor(
        max_workers=n_workers, mp_context=mp.get_context("spawn"),
        initializer=worker_path_init, initargs=(parent_package_root(),),
    ) as ex:
        for row in ex.map(_run_one_cell, work):
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            yield row


__all__ = [
    "TuningConfig",
    "load_tuning_yaml",
    "iter_grid",
    "TuningCellResult",
    "run_tuning",
    "run_tuning_parallel",
]
