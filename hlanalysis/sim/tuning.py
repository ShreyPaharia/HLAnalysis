from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from hlanalysis.strategy.base import Strategy

from .data.binance_klines import Kline
from .data.schemas import PMMarket, PMTrade
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
    splits = list(walk_forward_splits(jobs, train=train, test=test, step=step, drop_short_tail=True))
    log_path = out_dir / "results.jsonl"
    with log_path.open("a") as f:
        for params in iter_grid(grid):
            strat = strategy_factory(params)
            base_rcfg = runner_cfg_factory(params)
            for tr_jobs, te_jobs in splits:
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
                    "summary": asdict(summary),
                }) + "\n")
                f.flush()
                yield cell
