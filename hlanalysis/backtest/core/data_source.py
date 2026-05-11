"""DataSource protocol + QuestionDescriptor. Local mirror of §3.2.

See `events.py` for the rationale around mirroring. Task E drops this when
Task A's PR merges and re-points imports.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

from hlanalysis.strategy.types import QuestionView

from .events import MarketEvent


@dataclass(frozen=True, slots=True)
class QuestionDescriptor:
    """Stable identifier for a single tradable question/market across event
    streams. Carries the minimum needed to (a) recover the QuestionView at any
    ts_ns, and (b) tag fills + diagnostics for downstream reporting.
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

    Implementations: PM (synthetic L2), HL HIP-4 (recorded L2).
    """

    name: str

    def discover(self, *, start: str, end: str, **filters: object) -> list[QuestionDescriptor]:
        ...

    def events(self, q: QuestionDescriptor) -> Iterator[MarketEvent]:
        ...

    def question_view(self, q: QuestionDescriptor, *, now_ns: int, settled: bool) -> QuestionView:
        ...

    def resolved_outcome(self, q: QuestionDescriptor) -> Literal["yes", "no", "unknown"]:
        ...


__all__ = ["QuestionDescriptor", "DataSource"]
