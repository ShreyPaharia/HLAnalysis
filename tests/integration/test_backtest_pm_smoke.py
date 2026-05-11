"""End-to-end smoke for the new `PolymarketDataSource` on a captured binary
PM fixture.

Establishes a baseline P&L from the legacy `sim.runner.run_one_market` on the
same fixture data, then drives a minimal in-test replay loop over the new
`DataSource.events()` stream using the SAME strategy + fill model. Asserts
realized per-market P&L is within ±$0.01 of the baseline (§4 Task C
acceptance: "per-market P&L within ±$0.01 of the captured fixture's previous
values").

The in-test replay loop is intentionally minimal — Task A owns the real
runner. This test just proves the event stream produced by the new source
carries equivalent information.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hlanalysis.backtest.core.events import (
    BookSnapshot,
    ReferenceEvent,
    SettlementEvent,
    TradeEvent,
)
from hlanalysis.backtest.data.polymarket import PolymarketDataSource
from hlanalysis.sim.data.binance_klines import Kline
from hlanalysis.sim.data.schemas import PMMarket, PMTrade
from hlanalysis.sim.fills import Fill, FillModelConfig, simulate_fill
from hlanalysis.sim.market_state import SimMarketState
from hlanalysis.sim.runner import RunnerConfig, run_one_market
from hlanalysis.sim.synthetic_l2 import L2Snapshot
from hlanalysis.strategy.model_edge import ModelEdgeConfig, ModelEdgeStrategy
from hlanalysis.strategy.types import Action, Position

FX = Path("tests/fixtures/pm/binary")
HALF_SPREAD = 0.005
DEPTH = 10_000.0
FEE_TAKER = 0.02
SLIPPAGE_BPS = 5.0
SCAN_INTERVAL_S = 60


def _strategy() -> ModelEdgeStrategy:
    return ModelEdgeStrategy(ModelEdgeConfig(
        vol_lookback_seconds=14400, vol_sampling_dt_seconds=60,
        vol_clip_min=0.05, vol_clip_max=3.0,
        edge_buffer=0.02, fee_taker=FEE_TAKER, half_spread_assumption=HALF_SPREAD,
        stop_loss_pct=10.0,
    ))


def _runner_cfg(day_open_btc: float) -> RunnerConfig:
    return RunnerConfig(
        scanner_interval_seconds=SCAN_INTERVAL_S,
        fill_model=FillModelConfig(
            slippage_bps=SLIPPAGE_BPS, fee_taker=FEE_TAKER,
            book_depth_assumption=DEPTH,
        ),
        synthetic_half_spread=HALF_SPREAD, synthetic_depth=DEPTH,
        day_open_btc=day_open_btc,
    )


def _populate_cache(cache_root: Path, market: PMMarket, trades: list[PMTrade], klines: list[Kline]) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        market.condition_id: {
            "kind": "binary",
            "n_rows": len(trades),
            "last_pull_ts_ns": market.end_ts_ns,
            "market": market.model_dump(),
        },
    }
    (cache_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pm_trades = cache_root / "pm_trades"
    pm_trades.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "ts_ns":    [t.ts_ns for t in trades],
        "token_id": [t.token_id for t in trades],
        "side":     [t.side for t in trades],
        "price":    [t.price for t in trades],
        "size":     [t.size for t in trades],
    }), pm_trades / f"{market.condition_id}.parquet")
    klines_dir = cache_root / "btc_klines"
    klines_dir.mkdir(parents=True, exist_ok=True)
    (klines_dir / "fixture.json").write_text(json.dumps([asdict(k) for k in klines]))


def _replay_with_new_source(
    src: PolymarketDataSource, market: PMMarket, day_open_btc: float,
) -> float:
    """Drive the new DataSource event stream through the same strategy +
    fill model as the legacy runner. Returns realized P&L (sum of fill cash
    flows, including the synthetic settlement fill).
    """
    descs = src.discover(start="1970-01-01", end="2999-12-31", kind="binary")
    assert len(descs) == 1
    q = descs[0]
    cfg = _runner_cfg(day_open_btc)
    strat = _strategy()
    stop_loss_pct = strat.cfg.stop_loss_pct
    state = SimMarketState()
    pos: Position | None = None
    fills: list[Fill] = []
    last_scan_ns = q.start_ts_ns
    scan_ns = cfg.scanner_interval_seconds * 1_000_000_000

    # Per-leg outcome map (so we can settle whichever leg the position holds).
    per_leg_outcome: dict[str, str] = {}

    for ev in src.events(q):
        if isinstance(ev, BookSnapshot):
            bid_px, bid_sz = ev.bids[0]
            ask_px, ask_sz = ev.asks[0]
            state.apply_l2(L2Snapshot(
                ts_ns=ev.ts_ns, token_id=ev.symbol,
                bid_px=bid_px, bid_sz=bid_sz,
                ask_px=ask_px, ask_sz=ask_sz,
            ))
        elif isinstance(ev, TradeEvent):
            state.apply_trade_ts(ev.symbol, ev.ts_ns)
        elif isinstance(ev, ReferenceEvent):
            state.apply_kline(Kline(
                ts_ns=ev.ts_ns, open=ev.close, high=ev.high, low=ev.low,
                close=ev.close, volume=0.0,
            ))
        elif isinstance(ev, SettlementEvent):
            per_leg_outcome[ev.symbol] = ev.outcome
            # Settlement events all share ts_ns = end_ts_ns; skip strategy eval.
            continue

        if ev.ts_ns - last_scan_ns < scan_ns:
            continue
        qv = src.question_view(q, now_ns=ev.ts_ns, settled=False)
        # The new DataSource doesn't know day_open_btc; runner uses it as the strike.
        from dataclasses import replace
        qv = replace(qv, strike=day_open_btc)
        books = {sym: state.book(sym) for sym in q.leg_symbols}
        books = {k: v for k, v in books.items() if v is not None}
        recent_returns = state.recent_returns(now_ns=ev.ts_ns, lookback_seconds=86_400)
        recent_hl = state.recent_hl_bars(now_ns=ev.ts_ns, lookback_seconds=86_400)
        ref_close = state.latest_btc_close() or day_open_btc
        decision = strat.evaluate(
            question=qv, books=books, reference_price=float(ref_close),
            recent_returns=recent_returns, recent_volume_usd=0.0,
            position=pos, now_ns=ev.ts_ns,
            recent_hl_bars=recent_hl,
        )
        if decision.action == Action.ENTER and decision.intents:
            intent = decision.intents[0]
            book = state.book(intent.symbol)
            if book is not None:
                fill = simulate_fill(intent, book, cfg.fill_model)
                if fill.size > 0:
                    fills.append(fill)
                    stop_px = max(0.0, fill.price * (1.0 - stop_loss_pct / 100.0)) if stop_loss_pct else -1.0
                    pos = Position(
                        question_idx=qv.question_idx, symbol=intent.symbol,
                        qty=fill.size, avg_entry=fill.price,
                        stop_loss_price=stop_px, last_update_ts_ns=ev.ts_ns,
                    )
        elif decision.action == Action.EXIT and decision.intents:
            intent = decision.intents[0]
            book = state.book(intent.symbol)
            if book is not None:
                fill = simulate_fill(intent, book, cfg.fill_model)
                if fill.size > 0:
                    fills.append(fill)
                    pos = None
        last_scan_ns = ev.ts_ns

    # Settle whatever position is left at expiry using the per-leg outcome map.
    if pos is not None:
        outcome = per_leg_outcome.get(pos.symbol, "unknown")
        settle_px = 1.0 if outcome == "yes" else 0.0
        fills.append(Fill(
            cloid="settle", symbol=pos.symbol,
            side="sell" if pos.qty > 0 else "buy",
            price=settle_px, size=abs(pos.qty), fee=0.0, partial=False,
        ))

    realized = 0.0
    for f in fills:
        notional = f.price * f.size
        if f.side == "buy":
            realized += -(notional + f.fee)
        else:
            realized += notional - f.fee
    return realized


@pytest.mark.skipif(not (FX / "market.json").exists(), reason="smoke fixture not captured")
def test_pm_binary_smoke_parity_against_legacy_runner(tmp_path: Path) -> None:
    market = PMMarket.model_validate_json((FX / "market.json").read_text())
    trades = [PMTrade(**t) for t in json.loads((FX / "trades.json").read_text())]
    klines = [Kline(**k) for k in json.loads((FX / "klines.json").read_text())]
    if not klines:
        pytest.skip("no klines in fixture")
    day_open = klines[0].open

    # Baseline from the legacy runner.
    baseline = run_one_market(_strategy(), market, klines, trades, _runner_cfg(day_open))
    assert baseline.realized_pnl_usd is not None
    assert baseline.n_decisions > 0

    # New path through PolymarketDataSource.
    cache = tmp_path / "cache"
    _populate_cache(cache, market, trades, klines)
    src = PolymarketDataSource(
        cache_root=cache, half_spread=HALF_SPREAD, depth=DEPTH,
    )
    new_pnl = _replay_with_new_source(src, market, day_open_btc=day_open)

    # ±$0.01 per spec Task C acceptance.
    assert abs(new_pnl - baseline.realized_pnl_usd) < 0.01, (
        f"P&L drift: new={new_pnl} legacy={baseline.realized_pnl_usd}"
    )
