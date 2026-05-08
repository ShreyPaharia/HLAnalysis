# hlanalysis/strategy/model_edge.py
from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm  # type: ignore[import-untyped]

from .base import Strategy
from .types import (
    Action, BookState, Decision, Diagnostic, OrderIntent, Position, QuestionView,
)


@dataclass(frozen=True, slots=True)
class ModelEdgeConfig:
    vol_lookback_seconds: int
    vol_sampling_dt_seconds: int
    vol_clip_min: float
    vol_clip_max: float
    edge_buffer: float
    fee_taker: float
    half_spread_assumption: float
    stop_loss_pct: float | None
    drift_lookback_seconds: int = 0
    drift_blend: float = 0.0
    max_position_usd: float = 100.0


_ANNUAL_SECONDS = 365.25 * 86400.0


class ModelEdgeStrategy(Strategy):
    name = "model_edge"

    def __init__(self, cfg: ModelEdgeConfig) -> None:
        self.cfg = cfg

    def evaluate(
        self,
        *,
        question: QuestionView,
        books: Mapping[str, BookState],
        reference_price: float,
        recent_returns: tuple[float, ...],
        recent_volume_usd: float,
        position: Position | None,
        now_ns: int,
    ) -> Decision:
        if question.settled:
            if position is not None:
                return Decision(
                    action=Action.EXIT,
                    diagnostics=(Diagnostic("info", "exit_settlement"),),
                )
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "settled"),))

        if position is not None:
            held = books.get(position.symbol)
            if held is not None and held.bid_px is not None and held.bid_px <= position.stop_loss_price:
                intent = OrderIntent(
                    question_idx=question.question_idx,
                    symbol=position.symbol,
                    side="sell" if position.qty > 0 else "buy",
                    size=abs(position.qty),
                    limit_price=held.bid_px,
                    cloid=f"hla-{uuid.uuid4()}",
                    time_in_force="ioc",
                    reduce_only=True,
                )
                return Decision(
                    action=Action.EXIT,
                    intents=(intent,),
                    diagnostics=(Diagnostic("warn", "exit_stop_loss"),),
                )
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "have_position"),))

        # Entry logic in subsequent tasks
        return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "skeleton"),))
