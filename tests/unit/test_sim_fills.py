# tests/unit/test_sim_fills.py
from __future__ import annotations

from hlanalysis.strategy.types import BookState, OrderIntent
from hlanalysis.sim.fills import FillModelConfig, simulate_fill


def _book() -> BookState:
    return BookState("t1", 0.49, 200, 0.51, 200, 0, 0)


def test_buy_fill_at_ask_plus_slippage_minus_fee():
    cfg = FillModelConfig(slippage_bps=10.0, fee_taker=0.02, book_depth_assumption=1000.0)
    intent = OrderIntent(
        question_idx=1, symbol="t1", side="buy",
        size=100.0, limit_price=0.51, cloid="hla-x", time_in_force="ioc",
    )
    fill = simulate_fill(intent, _book(), cfg)
    expected_px = 0.51 * (1 + 10 / 1e4)
    assert abs(fill.price - expected_px) < 1e-9
    assert fill.size == 100.0
    assert fill.fee == expected_px * 100.0 * 0.02


def test_sell_fill_at_bid_minus_slippage():
    cfg = FillModelConfig(slippage_bps=10.0, fee_taker=0.02, book_depth_assumption=1000.0)
    intent = OrderIntent(
        question_idx=1, symbol="t1", side="sell",
        size=50.0, limit_price=0.49, cloid="hla-y", time_in_force="ioc",
    )
    fill = simulate_fill(intent, _book(), cfg)
    expected_px = 0.49 * (1 - 10 / 1e4)
    assert abs(fill.price - expected_px) < 1e-9


def test_partial_fill_when_size_exceeds_assumed_depth():
    cfg = FillModelConfig(slippage_bps=0.0, fee_taker=0.0, book_depth_assumption=10.0)
    intent = OrderIntent(
        question_idx=1, symbol="t1", side="buy",
        size=100.0, limit_price=0.51, cloid="hla-z", time_in_force="ioc",
    )
    fill = simulate_fill(intent, _book(), cfg)
    assert fill.size == 10.0
    assert fill.partial
