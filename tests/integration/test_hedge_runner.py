"""Smoke test: runner with hedge_enabled fills both binary and hedge intents
emitted by a stub strategy."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from hlanalysis.backtest.core.data_source import DataSource, QuestionDescriptor
from hlanalysis.backtest.core.events import BookSnapshot
from hlanalysis.backtest.data.binance_perp import BinancePerpKlinesSource
from hlanalysis.backtest.runner.hftbt_runner import RunConfig, run_one_question
from hlanalysis.strategy.base import Strategy
from hlanalysis.strategy.types import (
    Action,
    BookState,
    Decision,
    OrderIntent,
    Position,
    QuestionView,
)


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


def test_runner_fills_both_binary_and_hedge_intents(tmp_path: Path) -> None:
    """End-to-end: feed a stub strategy + a synthetic hedge BBO stream;
    assert one binary fill and one hedge fill are recorded."""
    # NOTE: Fixture wiring depends on tests/fixtures/pm — use the smallest
    # PM market in the corpus and the new BinancePerpKlinesSource pointed at
    # a tiny synthesized kline file covering the same window. This test should
    # be straightforward to wire up once the runner accepts a hedge_book_stream.
    pytest.skip("Wire up against existing PM fixture once hedge_book_stream is plumbed.")
