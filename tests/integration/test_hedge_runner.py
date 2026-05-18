"""Smoke test: runner with hedge_enabled fills both binary and hedge intents
emitted by a stub strategy."""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hlanalysis.backtest.data.binance_perp import BinancePerpKlinesSource
from hlanalysis.backtest.data.polymarket import PolymarketDataSource
from hlanalysis.backtest.runner.hftbt_runner import RunConfig, run_one_question
from hlanalysis.strategy.base import Strategy
from hlanalysis.strategy.types import (
    Action,
    Decision,
    OrderIntent,
    QuestionView,
)

FX = Path("tests/fixtures/pm/binary")

HEDGE_SYMBOL = "BTC-PERP"


class _DualLegStubStrategy(Strategy):
    """Emits one binary BUY and one hedge SELL on the very first tick where both
    books are present, then HOLDs. Used to verify the runner routes intents to
    the right leg and records is_hedge correctly."""

    name = "dual_leg_stub"

    def __init__(self, hedge_symbol: str) -> None:
        self.hedge_symbol = hedge_symbol
        self._fired = False

    def evaluate(self, **kwargs) -> Decision:
        if self._fired:
            return Decision(action=Action.HOLD, diagnostics=())
        books = kwargs["books"]
        q: QuestionView = kwargs["question"]
        if q.yes_symbol not in books or self.hedge_symbol not in books:
            return Decision(action=Action.HOLD, diagnostics=())
        self._fired = True
        return Decision(
            action=Action.ENTER,
            intents=(
                OrderIntent(
                    question_idx=q.question_idx,
                    symbol=q.yes_symbol,
                    side="buy",
                    size=10.0,
                    limit_price=1.0,
                    cloid="bin-1",
                    time_in_force="ioc",
                ),
                OrderIntent(
                    question_idx=q.question_idx,
                    symbol=self.hedge_symbol,
                    side="sell",
                    size=0.01,
                    limit_price=0.0,
                    cloid="hedge-1",
                    time_in_force="ioc",
                ),
            ),
        )


def _populate_cache(
    cache_root: Path,
    market_json: str,
    trades_json: str,
    klines_json: str,
) -> str:
    """Populate the PM cache layout expected by PolymarketDataSource."""
    cache_root.mkdir(parents=True, exist_ok=True)
    market_dict = json.loads(market_json)
    trades = json.loads(trades_json)
    klines = json.loads(klines_json)
    cond_id = market_dict["condition_id"]
    manifest = {
        cond_id: {
            "kind": "binary",
            "n_rows": len(trades),
            "last_pull_ts_ns": int(market_dict["end_ts_ns"]),
            "market": market_dict,
        }
    }
    (cache_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pm_trades = cache_root / "pm_trades"
    pm_trades.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "ts_ns": [t["ts_ns"] for t in trades],
                "token_id": [t["token_id"] for t in trades],
                "side": [t["side"] for t in trades],
                "price": [t["price"] for t in trades],
                "size": [t["size"] for t in trades],
            }
        ),
        pm_trades / f"{cond_id}.parquet",
    )
    klines_dir = cache_root / "btc_klines"
    klines_dir.mkdir(parents=True, exist_ok=True)
    (klines_dir / "fixture.json").write_text(json.dumps(klines))
    return cond_id


@pytest.mark.skipif(not (FX / "market.json").exists(), reason="PM fixture not captured")
def test_runner_fills_both_binary_and_hedge_intents(tmp_path: Path) -> None:
    """End-to-end: feed a stub strategy + a synthetic hedge BBO stream;
    assert one binary fill and one hedge fill are recorded."""
    # --- Populate PM cache from fixture ---
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
        half_spread=0.005,
        depth=10_000.0,
    )
    descs = src.discover(start="1970-01-01", end="2999-12-31", kind="binary")
    assert len(descs) == 1 and descs[0].question_id == cond_id
    q = descs[0]

    # --- Build synthetic hedge klines covering the question window ---
    # One entry per minute from start to end; constant BTC price of 80_000.
    step_ns = 60 * 1_000_000_000  # 1 minute in nanoseconds
    hedge_klines = []
    ts = q.start_ts_ns
    while ts < q.end_ts_ns:
        hedge_klines.append(
            {
                "ts_ns": ts,
                "open": 80_000.0,
                "high": 80_000.0,
                "low": 80_000.0,
                "close": 80_000.0,
                "volume": 1.0,
            }
        )
        ts += step_ns

    hedge_klines_path = tmp_path / "btc_perp_klines.json"
    hedge_klines_path.write_text(json.dumps(hedge_klines))

    hedge_source = BinancePerpKlinesSource(
        path=hedge_klines_path,
        symbol=HEDGE_SYMBOL,
        half_spread_bps=1.0,
    )
    hedge_events = list(
        hedge_source.book_events(start_ts_ns=q.start_ts_ns, end_ts_ns=q.end_ts_ns)
    )
    assert len(hedge_events) > 0, "Expected hedge events from synthetic klines"

    # --- Build stub strategy and RunConfig ---
    strategy = _DualLegStubStrategy(hedge_symbol=HEDGE_SYMBOL)
    cfg = RunConfig(
        scanner_interval_seconds=60,
        tick_size=0.001,
        lot_size=1.0,
        slippage_bps=5.0,
        fee_taker=0.0,
        book_depth_assumption=10_000.0,
        hedge_enabled=True,
        hedge_symbol=HEDGE_SYMBOL,
        hedge_tick_size=0.1,
        hedge_lot_size=0.001,
        hedge_slippage_bps=10.0,
        hedge_fee_bps=1.0,
    )

    # --- Run ---
    result = run_one_question(
        strategy,
        src,
        q,
        cfg,
        strike=float(day_open),
        hedge_events=hedge_events,
    )

    hedge_fills = [f for f in result.fills if f.is_hedge]
    binary_fills = [f for f in result.fills if not f.is_hedge]

    assert len(hedge_fills) >= 1, (
        f"Expected at least one hedge fill; got fills: {result.fills}"
    )
    assert len(binary_fills) >= 1, (
        f"Expected at least one binary fill; got fills: {result.fills}"
    )

    # Verify hedge fill properties.
    hf = hedge_fills[0]
    assert hf.symbol == HEDGE_SYMBOL
    assert hf.is_hedge is True
    assert hf.price > 1.0, f"Hedge fill price {hf.price} should be BTC-scale, not clamped to [0,1]"
