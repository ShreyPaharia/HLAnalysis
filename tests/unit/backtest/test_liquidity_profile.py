from hlanalysis.backtest.data._synthetic_l2 import LiquidityProfile, trade_to_l2


def test_profile_lookup_by_price_bucket():
    prof = LiquidityProfile(bucket_width=0.5, half_spread=[0.02, 0.004],
                            depth=[2000.0, 8000.0],
                            global_half_spread=0.01, global_depth=5000.0)
    assert prof.half_spread_at(0.3) == 0.02
    assert prof.depth_at(0.3) == 2000.0
    assert prof.half_spread_at(0.9) == 0.004
    assert prof.depth_at(0.9) == 8000.0


def test_profile_empty_bucket_falls_back_to_global():
    prof = LiquidityProfile(bucket_width=0.5, half_spread=[None, 0.004],
                            depth=[None, 8000.0],
                            global_half_spread=0.01, global_depth=5000.0)
    assert prof.half_spread_at(0.3) == 0.01
    assert prof.depth_at(0.3) == 5000.0


def test_trade_to_l2_uses_profile_when_supplied():
    prof = LiquidityProfile(bucket_width=0.5, half_spread=[0.02, 0.004],
                            depth=[2000.0, 8000.0],
                            global_half_spread=0.01, global_depth=5000.0)
    s = trade_to_l2(ts_ns=1, token_id="t", price=0.9,
                    half_spread=0.005, depth=10000.0, profile=prof)
    assert s.ask_px == 0.9 + 0.004
    assert s.ask_sz == 8000.0


def test_trade_to_l2_bit_identical_without_profile():
    s = trade_to_l2(ts_ns=1, token_id="t", price=0.9, half_spread=0.005, depth=10000.0)
    assert s.ask_px == 0.905 and s.bid_px == 0.895
    assert s.ask_sz == 10000.0 and s.bid_sz == 10000.0


import json
from pathlib import Path
from hlanalysis.backtest.core.source_config import SourceConfig
from hlanalysis.backtest.data.polymarket import PolymarketDataSource


def _write_profile(p: Path) -> Path:
    p.write_text(json.dumps({
        "bucket_width": 0.5, "half_spread": [0.02, 0.004], "depth": [2000.0, 8000.0],
        "global_half_spread": 0.01, "global_depth": 5000.0}))
    return p


def test_datasource_loads_profile(tmp_path):
    prof = _write_profile(tmp_path / "prof.json")
    ds = PolymarketDataSource(cache_root=tmp_path, liquidity_profile_path=str(prof))
    assert ds._liquidity_profile is not None
    assert ds._liquidity_profile.half_spread_at(0.9) == 0.004


def test_sourceconfig_carries_profile_path(tmp_path):
    prof = _write_profile(tmp_path / "prof.json")
    cfg = SourceConfig(kind="polymarket", cache_root=str(tmp_path),
                       pm_liquidity_profile_path=str(prof))
    ds = cfg.build()
    assert ds._liquidity_profile is not None


def test_datasource_no_profile_is_none(tmp_path):
    ds = PolymarketDataSource(cache_root=tmp_path)
    assert ds._liquidity_profile is None
