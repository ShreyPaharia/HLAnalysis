from __future__ import annotations

import pytest

from hlanalysis.strategy.types import BookState, Position, QuestionView


@pytest.fixture
def question() -> QuestionView:
    return QuestionView(
        question_idx=42,
        yes_symbol="@30",
        no_symbol="@31",
        strike=80_000.0,
        expiry_ns=2_000_000_000_000_000_000,  # far future
        underlying="BTC",
        klass="priceBinary",
        period="1h",
    )


@pytest.fixture
def book_yes_extreme() -> BookState:
    return BookState(
        symbol="@30",
        bid_px=0.94,
        bid_sz=50.0,
        ask_px=0.95,
        ask_sz=80.0,
        last_trade_ts_ns=1,
        last_l2_ts_ns=1,
    )


@pytest.fixture
def empty_position(question) -> Position | None:
    return None
