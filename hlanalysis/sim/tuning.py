from __future__ import annotations

import importlib
import itertools
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from hlanalysis.strategy.base import Strategy

from .data.binance_klines import Kline
from .data.schemas import PMMarket, PMTrade
from .fills import FillModelConfig
from .metrics import RunSummary, summarise_run
from .runner import RunnerConfig, run_one_market
from .walkforward import walk_forward_splits


def iter_grid(grid: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


@dataclass(frozen=True, slots=True)
class TuningJob:
    market: PMMarket
    klines: list[Kline]
    trades: list[PMTrade]
    day_open_btc: float


@dataclass(frozen=True, slots=True)
class TuningCellResult:
    params: dict[str, Any]
    train_ids: list[str]
    test_ids: list[str]
    summary: RunSummary


def _cell_key(params: dict[str, Any], test_ids: list[str]) -> tuple:
    """Stable key for resumability — identifies a (param tuple, OOS test set) cell."""
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
    grid: dict[str, list[Any]],
    strategy_factory: Callable[[dict[str, Any]], Strategy],
    runner_cfg_factory: Callable[[dict[str, Any]], RunnerConfig],
    jobs: list[TuningJob],
    train: int,
    test: int,
    step: int,
    out_dir: Path,
) -> Iterator[TuningCellResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # drop partial tail for uniform OOS window sizes (Sharpe stats stability)
    splits = list(walk_forward_splits(jobs, train=train, test=test, step=step, drop_short_tail=True))
    log_path = out_dir / "results.jsonl"
    completed = _load_completed_cells(log_path)
    with log_path.open("a") as f:
        for params in iter_grid(grid):
            strat = strategy_factory(params)
            base_rcfg = runner_cfg_factory(params)
            for tr_jobs, te_jobs in splits:
                test_ids = [j.market.condition_id for j in te_jobs]
                if _cell_key(params, test_ids) in completed:
                    continue
                pnl_per_market: list[float] = []
                n_trades = 0
                for job in te_jobs:
                    job_cfg = RunnerConfig(
                        scanner_interval_seconds=base_rcfg.scanner_interval_seconds,
                        fill_model=base_rcfg.fill_model,
                        synthetic_half_spread=base_rcfg.synthetic_half_spread,
                        synthetic_depth=base_rcfg.synthetic_depth,
                        day_open_btc=job.day_open_btc,
                    )
                    res = run_one_market(strat, job.market, job.klines, job.trades, job_cfg)
                    pnl_per_market.append(res.realized_pnl_usd or 0.0)
                    n_trades += len(res.fills)
                summary = summarise_run(pnl_per_market, n_trades=n_trades)
                cell = TuningCellResult(
                    params=params,
                    train_ids=[j.market.condition_id for j in tr_jobs],
                    test_ids=[j.market.condition_id for j in te_jobs],
                    summary=summary,
                )
                f.write(json.dumps({
                    "params": params,
                    "n_train": len(tr_jobs), "n_test": len(te_jobs),
                    "train_ids": cell.train_ids,
                    "test_ids":  cell.test_ids,
                    "summary": asdict(summary),
                }) + "\n")
                f.flush()
                yield cell


def _resolve_dotted(name: str):
    mod, _, attr = name.rpartition(".")
    return getattr(importlib.import_module(mod), attr)


def _run_one_split_for_params(args: tuple) -> dict:
    (params, factory_dotted, train_jobs, test_jobs,
     scanner_dt, fill_cfg_kwargs, half_spread, depth) = args
    strategy_factory = _resolve_dotted(factory_dotted)
    strat = strategy_factory(params)
    fill_cfg = FillModelConfig(**fill_cfg_kwargs)
    pnl_per_market: list[float] = []
    n_trades = 0
    for job in test_jobs:
        rcfg = RunnerConfig(
            scanner_interval_seconds=scanner_dt,
            fill_model=fill_cfg,
            synthetic_half_spread=half_spread,
            synthetic_depth=depth,
            day_open_btc=job.day_open_btc,
        )
        res = run_one_market(strat, job.market, job.klines, job.trades, rcfg)
        pnl_per_market.append(res.realized_pnl_usd or 0.0)
        n_trades += len(res.fills)
    summary = summarise_run(pnl_per_market, n_trades=n_trades)
    return {
        "params": params,
        "n_train": len(train_jobs), "n_test": len(test_jobs),
        "summary": asdict(summary),
        "train_ids": [j.market.condition_id for j in train_jobs],
        "test_ids":  [j.market.condition_id for j in test_jobs],
    }


def run_tuning_parallel(
    *,
    grid: dict[str, list[Any]],
    strategy_factory: Callable[[dict[str, Any]], Strategy],
    runner_cfg_factory: Callable[[dict[str, Any]], RunnerConfig],
    jobs: list[TuningJob],
    train: int, test: int, step: int,
    out_dir: Path, n_workers: int,
) -> Iterator[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = list(walk_forward_splits(jobs, train=train, test=test, step=step, drop_short_tail=True))
    factory_dotted = f"{strategy_factory.__module__}.{strategy_factory.__name__}"

    base_rcfg = runner_cfg_factory({})
    fc = base_rcfg.fill_model
    fill_cfg_kwargs = {
        "slippage_bps": fc.slippage_bps,
        "fee_taker": fc.fee_taker,
        "book_depth_assumption": fc.book_depth_assumption,
    }

    log_path = out_dir / "results.jsonl"
    completed = _load_completed_cells(log_path)

    work = []
    for params in iter_grid(grid):
        for tr, te in splits:
            test_ids = [j.market.condition_id for j in te]
            if _cell_key(params, test_ids) in completed:
                continue
            work.append((
                params, factory_dotted, tr, te,
                base_rcfg.scanner_interval_seconds, fill_cfg_kwargs,
                base_rcfg.synthetic_half_spread, base_rcfg.synthetic_depth,
            ))

    with log_path.open("a") as f, ProcessPoolExecutor(
        max_workers=n_workers, mp_context=mp.get_context("spawn")
    ) as ex:
        for row in ex.map(_run_one_split_for_params, work):
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            yield row
