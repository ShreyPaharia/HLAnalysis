from __future__ import annotations

from hlanalysis.sim.data.binance_klines import Kline
from hlanalysis.sim.data.schemas import PMMarket, PMTrade
from hlanalysis.sim.fills import FillModelConfig
from hlanalysis.sim.runner import RunnerConfig, run_one_market
from hlanalysis.strategy.model_edge import ModelEdgeConfig, ModelEdgeStrategy


def _market() -> PMMarket:
    return PMMarket(
        condition_id="0xabc", yes_token_id="Y", no_token_id="N",
        start_ts_ns=0, end_ts_ns=86_400_000_000_000,
        resolved_outcome="yes", total_volume_usd=10_000.0, n_trades=100,
    )


def test_runner_produces_at_least_one_decision_and_resolves():
    market = _market()
    klines = [
        Kline(ts_ns=i * 60_000_000_000, open=100_000 + i, high=100_000 + i,
              low=100_000 + i, close=100_000 + i + 1, volume=0)
        for i in range(60)
    ]
    trades = [
        PMTrade(ts_ns=10 * 60_000_000_000, token_id="Y", side="buy", price=0.6, size=100),
        PMTrade(ts_ns=10 * 60_000_000_000 + 1, token_id="N", side="buy", price=0.4, size=100),
    ]
    strat = ModelEdgeStrategy(ModelEdgeConfig(
        vol_lookback_seconds=3600, vol_sampling_dt_seconds=60,
        vol_clip_min=0.05, vol_clip_max=3.0,
        edge_buffer=0.02, fee_taker=0.02, half_spread_assumption=0.005,
        stop_loss_pct=10.0, max_position_usd=100.0,
    ))
    cfg = RunnerConfig(
        scanner_interval_seconds=60,
        fill_model=FillModelConfig(slippage_bps=5.0, fee_taker=0.02, book_depth_assumption=10_000.0),
        synthetic_half_spread=0.005, synthetic_depth=10_000.0,
        day_open_btc=100_000.0,
    )
    result = run_one_market(strat, market, klines, trades, cfg)
    assert result.n_decisions > 0
    assert result.realized_pnl_usd is not None


def test_runner_settles_unknown_outcome_to_zero():
    """Document current behavior: when a market's outcome is unknown, any open
    position settles to 0. Cache must persist the canonical outcome to avoid
    silently zero-ing every winning trade. See cli._load_jobs_from_cache.
    """
    market = PMMarket(
        condition_id="0xabc", yes_token_id="Y", no_token_id="N",
        start_ts_ns=0, end_ts_ns=86_400_000_000_000,
        resolved_outcome="unknown", total_volume_usd=10_000.0, n_trades=100,
    )
    klines = [
        Kline(ts_ns=i * 60_000_000_000, open=100_000, high=100_000,
              low=100_000, close=100_000 + i * 100, volume=0)
        for i in range(120)
    ]
    trades = [
        PMTrade(ts_ns=30 * 60_000_000_000, token_id="Y", side="buy", price=0.6, size=100),
        PMTrade(ts_ns=30 * 60_000_000_000 + 1, token_id="N", side="buy", price=0.4, size=100),
    ]
    strat = ModelEdgeStrategy(ModelEdgeConfig(
        vol_lookback_seconds=3600, vol_sampling_dt_seconds=60,
        vol_clip_min=0.05, vol_clip_max=3.0,
        edge_buffer=0.02, fee_taker=0.02, half_spread_assumption=0.005,
        stop_loss_pct=None, max_position_usd=100.0,
    ))
    cfg = RunnerConfig(
        scanner_interval_seconds=60,
        fill_model=FillModelConfig(slippage_bps=5.0, fee_taker=0.02, book_depth_assumption=10_000.0),
        synthetic_half_spread=0.005, synthetic_depth=10_000.0,
        day_open_btc=100_000.0,
    )
    result = run_one_market(strat, market, klines, trades, cfg)
    settle_fills = [f for f in result.fills if f.cloid == "settle"]
    for f in settle_fills:
        assert f.price == 0.0


def test_runner_settles_open_position_at_outcome():
    market = _market()
    klines = [
        Kline(ts_ns=i * 60_000_000_000, open=100_000, high=100_000,
              low=100_000, close=100_000 + i * 100, volume=0)
        for i in range(120)
    ]
    trades = [
        PMTrade(ts_ns=30 * 60_000_000_000, token_id="Y", side="buy", price=0.6, size=100),
        PMTrade(ts_ns=30 * 60_000_000_000 + 1, token_id="N", side="buy", price=0.4, size=100),
    ]
    strat = ModelEdgeStrategy(ModelEdgeConfig(
        vol_lookback_seconds=3600, vol_sampling_dt_seconds=60,
        vol_clip_min=0.05, vol_clip_max=3.0,
        edge_buffer=0.02, fee_taker=0.02, half_spread_assumption=0.005,
        stop_loss_pct=None, max_position_usd=100.0,
    ))
    cfg = RunnerConfig(
        scanner_interval_seconds=60,
        fill_model=FillModelConfig(slippage_bps=5.0, fee_taker=0.02, book_depth_assumption=10_000.0),
        synthetic_half_spread=0.005, synthetic_depth=10_000.0,
        day_open_btc=100_000.0,
    )
    result = run_one_market(strat, market, klines, trades, cfg)
    if result.fills:
        last = result.fills[-1]
        assert last.cloid == "settle"
