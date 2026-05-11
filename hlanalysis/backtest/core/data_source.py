"""DataSource protocol + QuestionDescriptor (spec §3.2)."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

from hlanalysis.strategy.types import QuestionView

from .events import MarketEvent


@dataclass(frozen=True, slots=True)
class QuestionDescriptor:
    """Stable identifier for a single tradable question/market across event streams.

    Carries the minimum needed to (a) recover the QuestionView at any ts_ns, and
    (b) tag fills + diagnostics for downstream reporting.
    """

    question_id: str
    question_idx: int
    start_ts_ns: int
    end_ts_ns: int
    leg_symbols: tuple[str, ...]
    klass: str
    underlying: str


class DataSource(Protocol):
    """A data source for one or more questions.

    Implementations: Polymarket (synthetic L2 derived from trades), HL HIP-4
    (real recorded L2). The runner is source-agnostic and consumes only this
    protocol.
    """

    name: str

    def discover(self, *, start: str, end: str, **filters: object) -> list[QuestionDescriptor]:
        """Discover questions resolving in [start, end). ISO date strings."""
        ...

    def events(self, q: QuestionDescriptor) -> Iterator[MarketEvent]:
        """Chronologically ordered events for question `q`.

        Includes per-leg book/trade events AND ReferenceEvents (e.g. BTC klines)
        for the question's lifetime. Must yield in nondecreasing ts_ns; the
        runner does not re-sort.
        """
        ...

    def question_view(
        self, q: QuestionDescriptor, *, now_ns: int, settled: bool
    ) -> QuestionView:
        """Snapshot the QuestionView the strategy sees at `now_ns`."""
        ...

    def resolved_outcome(
        self, q: QuestionDescriptor
    ) -> Literal["yes", "no", "unknown"]:
        """Final outcome; used for settlement P&L on positions still open at end_ts_ns."""
        ...


__all__ = ["QuestionDescriptor", "DataSource"]
