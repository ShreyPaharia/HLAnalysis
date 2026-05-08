from __future__ import annotations

import json
from pathlib import Path

import pytest

from hlanalysis.sim.data.binance_klines import Kline
from hlanalysis.sim.data.schemas import PMMarket, PMTrade
from hlanalysis.sim.fills import FillModelConfig
from hlanalysis.sim.runner import RunnerConfig, run_one_market
from hlanalysis.strategy.model_edge import ModelEdgeConfig, ModelEdgeStrategy

FX = Path("tests/fixtures/sim_smoke")


@pytest.mark.skipif(not (FX / "market.json").exists(), reason="smoke fixture not captured")
def test_smoke_one_market_end_to_end():
    market = PMMarket.model_validate_json((FX / "market.json").read_text())
    trades = [PMTrade(**t) for t in json.loads((FX / "trades.json").read_text())]
    klines = [Kline(**k) for k in json.loads((FX / "klines.json").read_text())]
    if not klines:
        pytest.skip("no klines in fixture")
    day_open = klines[0].open

    strat = ModelEdgeStrategy(ModelEdgeConfig(
        vol_lookback_seconds=14400, vol_sampling_dt_seconds=60,
        vol_clip_min=0.05, vol_clip_max=3.0,
        edge_buffer=0.02, fee_taker=0.02, half_spread_assumption=0.005,
        stop_loss_pct=10.0,
    ))
    cfg = RunnerConfig(
        scanner_interval_seconds=60,
        fill_model=FillModelConfig(slippage_bps=5.0, fee_taker=0.02, book_depth_assumption=10_000.0),
        synthetic_half_spread=0.005, synthetic_depth=10_000.0,
        day_open_btc=day_open,
    )
    res = run_one_market(strat, market, klines, trades, cfg)
    assert res.n_decisions > 0
    assert res.realized_pnl_usd is not None
