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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml

from .core.data_source import DataSource, QuestionDescriptor
from .core.registry import build as build_strategy
from .runner.hftbt_runner import RunConfig, run_one_question
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


def _run_one_cell(args: tuple) -> dict:
    (
        strategy_id,
        params,
        tr_ids,
        te_ids_with_strikes,
        run_cfg_kwargs,
        data_source_dotted,
    ) = args

    import importlib

    mod_name, _, attr = data_source_dotted.rpartition(".")
    ds_factory = getattr(importlib.import_module(mod_name), attr)
    data_source = ds_factory()

    strategy = build_strategy(strategy_id, params)
    run_cfg = RunConfig(**run_cfg_kwargs)
    pnl: list[float] = []
    n_trades = 0
    for q_id, strike in te_ids_with_strikes:
        # discover() returned QuestionDescriptors; the worker re-discovers and
        # matches by question_id.
        match = next(
            (d for d in data_source.discover(start="", end="") if d.question_id == q_id),
            None,
        )
        if match is None:
            continue
        res = run_one_question(strategy, data_source, match, run_cfg, strike=strike)
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
    data_source_factory_dotted: str,
    descriptors: list[QuestionDescriptor],
    run_cfg: RunConfig,
    train: int,
    test: int,
    step: int,
    out_dir: Path,
    n_workers: int,
    strike_for: Callable[[QuestionDescriptor], float] = lambda _q: 0.0,
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
                    data_source_factory_dotted,
                )
            )

    with log_path.open("a") as f, ProcessPoolExecutor(
        max_workers=n_workers, mp_context=mp.get_context("spawn")
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
