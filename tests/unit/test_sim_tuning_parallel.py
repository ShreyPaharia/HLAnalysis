from __future__ import annotations

from hlanalysis.sim.data.binance_klines import Kline
from hlanalysis.sim.data.schemas import PMMarket, PMTrade
from hlanalysis.sim.fills import FillModelConfig
from hlanalysis.sim.runner import RunnerConfig
from hlanalysis.sim.tuning import TuningJob, run_tuning_parallel
from hlanalysis.sim.v2_factory import build_v2_strategy_from_params


def _make_jobs(n: int) -> list[TuningJob]:
    out: list[TuningJob] = []
    for i in range(n):
        m = PMMarket(
            condition_id=f"c{i}", yes_token_id="Y", no_token_id="N",
            start_ts_ns=0, end_ts_ns=86_400_000_000_000,
            resolved_outcome="yes", total_volume_usd=1000.0, n_trades=10,
        )
        klines = [
            Kline(ts_ns=j * 60_000_000_000, open=100_000 + j, high=100_000 + j,
                  low=100_000 + j, close=100_000 + j + 1, volume=0)
            for j in range(10)
        ]
        trades = [PMTrade(ts_ns=5 * 60_000_000_000, token_id="Y", side="buy", price=0.6, size=10)]
        out.append(TuningJob(market=m, klines=klines, trades=trades, day_open_btc=100_000.0))
    return out


def test_parallel_run_produces_one_row_per_grid_cell_per_split(tmp_path):
    grid = {
        "edge_buffer": [0.02, 0.5],
        "stop_loss_pct": [10, None],
        "vol_lookback_seconds": [3600],
        "drift_lookback_seconds": [0],
    }
    jobs = _make_jobs(20)
    rcfg = RunnerConfig(
        scanner_interval_seconds=60,
        fill_model=FillModelConfig(slippage_bps=0.0, fee_taker=0.02, book_depth_assumption=1000.0),
        synthetic_half_spread=0.005, synthetic_depth=1000.0,
        day_open_btc=100_000.0,
    )
    rows = list(run_tuning_parallel(
        grid=grid,
        strategy_factory=build_v2_strategy_from_params,
        runner_cfg_factory=lambda p: rcfg,
        jobs=jobs, train=10, test=5, step=5,
        out_dir=tmp_path, n_workers=2,
    ))
    # 4 grid cells × 2 splits = 8 rows
    assert len(rows) == 8
