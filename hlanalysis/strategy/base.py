from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from .types import BookState, Decision, Position, QuestionView


class Strategy(ABC):
    """Pure-policy interface. Implementations must not perform IO.

    The engine (or sim) calls evaluate() per question per scan tick. The strategy
    inspects the question, the relevant book(s), the reference price, and any
    open position; it returns a Decision describing what to do (or HOLD).

    All inputs are immutable snapshots. All outputs are immutable dataclasses.
    """

    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        *,
        question: QuestionView,
        books: Mapping[str, BookState],   # symbol -> BookState; includes both legs + ref market
        reference_price: float,           # underlying spot/perp mark, e.g. BTC
        recent_returns: tuple[float, ...],  # log-returns over a fixed lookback for vol calc
        recent_volume_usd: float,         # last-hour notional volume on this question
        position: Position | None,
        now_ns: int,
    ) -> Decision: ...
