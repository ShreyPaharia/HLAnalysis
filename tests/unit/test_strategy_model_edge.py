# tests/unit/test_strategy_model_edge.py
from __future__ import annotations

from hlanalysis.strategy.types import (
    Action, BookState, Position, QuestionView,
)
from hlanalysis.strategy.model_edge import ModelEdgeConfig, ModelEdgeStrategy


def _cfg(**overrides) -> ModelEdgeConfig:
    base = dict(
        vol_lookback_seconds=3600,
        vol_sampling_dt_seconds=60,
        vol_clip_min=0.05,
        vol_clip_max=3.0,
        edge_buffer=0.02,
        fee_taker=0.02,
        half_spread_assumption=0.005,
        stop_loss_pct=10.0,
    )
    base.update(overrides)
    return ModelEdgeConfig(**base)


def _q(**overrides) -> QuestionView:
    base = dict(
        question_idx=1,
        yes_symbol="YES",
        no_symbol="NO",
        strike=100_000.0,
        expiry_ns=86_400_000_000_000,
        underlying="BTC",
        klass="priceBinary",
        period="24h",
    )
    base.update(overrides)
    return QuestionView(**base)


def test_holds_when_settled_no_position():
    s = ModelEdgeStrategy(_cfg())
    out = s.evaluate(
        question=_q(settled=True),
        books={},
        reference_price=100_000.0,
        recent_returns=(),
        recent_volume_usd=0.0,
        position=None,
        now_ns=0,
    )
    assert out.action == Action.HOLD


def test_exits_when_settled_with_position():
    s = ModelEdgeStrategy(_cfg())
    pos = Position(question_idx=1, symbol="YES", qty=1.0, avg_entry=0.5,
                   stop_loss_price=0.0, last_update_ts_ns=0)
    out = s.evaluate(
        question=_q(settled=True),
        books={},
        reference_price=100_000.0,
        recent_returns=(),
        recent_volume_usd=0,
        position=pos,
        now_ns=0,
    )
    assert out.action == Action.EXIT
