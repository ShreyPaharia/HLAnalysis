"""End-to-end PM binary smoke + decision-equivalence regression.

Drives the real backtest runner (``run_one_question``) on the captured PM
binary fixture and asserts realized P&L matches the baseline captured from
the legacy `hlanalysis.sim.runner.run_one_market` before that module was
deleted.

Baseline captured 2026-05-11 against the same fixture (commit ef2edab,
before the v2 backtester integration pass). The new runner uses hftbacktest's
matching engine; on this fixture it deviates from the legacy fill model by
less than 1e-4 USD per market, well within ±$0.01.
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hlanalysis.backtest.data.polymarket import PolymarketDataSource
from hlanalysis.backtest.runner.hftbt_runner import RunConfig, run_one_question
from hlanalysis.strategy.model_edge import ModelEdgeConfig, ModelEdgeStrategy

FX = Path("tests/fixtures/pm/binary")
HALF_SPREAD = 0.005
DEPTH = 10_000.0
FEE_TAKER = 0.02
SLIPPAGE_BPS = 5.0
SCAN_INTERVAL_S = 60

# Baseline captured 2026-05-11 from sim.runner.run_one_market on this fixture
# with the config below, before `hlanalysis/sim/` was deleted in the v2
# integration pass.
BASELINE_PNL_USD = -148.116015
TOLERANCE_USD = 0.01


def _strategy() -> ModelEdgeStrategy:
    return ModelEdgeStrategy(ModelEdgeConfig(
        vol_lookback_seconds=14400, vol_sampling_dt_seconds=60,
        vol_clip_min=0.05, vol_clip_max=3.0,
        edge_buffer=0.02, fee_taker=FEE_TAKER, half_spread_assumption=HALF_SPREAD,
        stop_loss_pct=10.0,
    ))


def _populate_cache(
    cache_root: Path, market_json: str, trades_json: str, klines_json: str,
) -> str:
    cache_root.mkdir(parents=True, exist_ok=True)
    market_dict = json.loads(market_json)
    trades = json.loads(trades_json)
    klines = json.loads(klines_json)
    cond_id = market_dict["condition_id"]
    manifest = {cond_id: {
        "kind": "binary",
        "n_rows": len(trades),
        "last_pull_ts_ns": int(market_dict["end_ts_ns"]),
        "market": market_dict,
    }}
    (cache_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pm_trades = cache_root / "pm_trades"
    pm_trades.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "ts_ns":    [t["ts_ns"] for t in trades],
        "token_id": [t["token_id"] for t in trades],
        "side":     [t["side"] for t in trades],
        "price":    [t["price"] for t in trades],
        "size":     [t["size"] for t in trades],
    }), pm_trades / f"{cond_id}.parquet")
    klines_dir = cache_root / "btc_klines"
    klines_dir.mkdir(parents=True, exist_ok=True)
    (klines_dir / "fixture.json").write_text(json.dumps(klines))
    return cond_id


@pytest.mark.skipif(not (FX / "market.json").exists(), reason="PM fixture not captured")
def test_pm_binary_smoke_decision_equivalence(tmp_path: Path) -> None:
    cond_id = _populate_cache(
        tmp_path / "cache",
        (FX / "market.json").read_text(),
        (FX / "trades.json").read_text(),
        (FX / "klines.json").read_text(),
    )
    klines = json.loads((FX / "klines.json").read_text())
    if not klines:
        pytest.skip("no klines in fixture")
    day_open = klines[0]["open"]

    src = PolymarketDataSource(
        cache_root=tmp_path / "cache",
        half_spread=HALF_SPREAD,
        depth=DEPTH,
    )
    descs = src.discover(start="1970-01-01", end="2999-12-31", kind="binary")
    assert len(descs) == 1 and descs[0].question_id == cond_id

    run_cfg = RunConfig(
        scanner_interval_seconds=SCAN_INTERVAL_S,
        tick_size=0.001, lot_size=1.0,
        slippage_bps=SLIPPAGE_BPS, fee_taker=FEE_TAKER,
        book_depth_assumption=DEPTH,
    )
    res = run_one_question(_strategy(), src, descs[0], run_cfg, strike=float(day_open))
    pnl = res.realized_pnl_usd or 0.0
    assert abs(pnl - BASELINE_PNL_USD) <= TOLERANCE_USD, (
        f"PM binary P&L drift: new runner={pnl:.4f}, baseline={BASELINE_PNL_USD:.4f}, "
        f"|Δ|={abs(pnl - BASELINE_PNL_USD):.6f} > tolerance {TOLERANCE_USD}"
    )
