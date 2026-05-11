"""Hand-crafted in-memory DataSource used by Task A's tests + the CLI smoke run.

This is NOT a production data source. It builds one binary question with a
deterministic event sequence so the runner + CLI can be exercised end-to-end
without depending on the PM or HL HIP-4 sources (Tasks B/C). The CLI exposes
it under ``--data-source synthetic``; see ``hlanalysis/backtest/cli.py``.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

from hlanalysis.strategy.types import QuestionView

from ..core.data_source import QuestionDescriptor
from ..core.events import (
    BookSnapshot,
    MarketEvent,
    ReferenceEvent,
    SettlementEvent,
    TradeEvent,
)
from ..core.question import build_question_view


@dataclass(frozen=True, slots=True)
class SyntheticQuestion:
    descriptor: QuestionDescriptor
    book_snapshots: tuple[BookSnapshot, ...]
    trades: tuple[TradeEvent, ...]
    reference_events: tuple[ReferenceEvent, ...]
    settlement_events: tuple[SettlementEvent, ...]
    outcome: Literal["yes", "no", "unknown"]
    strike: float


@dataclass(slots=True)
class SyntheticDataSource:
    """An in-memory ``DataSource`` carrying one or more pre-built questions."""

    name: str = "synthetic"
    questions: list[SyntheticQuestion] = field(default_factory=list)

    def add_question(self, sq: SyntheticQuestion) -> None:
        self.questions.append(sq)

    # ---- DataSource Protocol ---------------------------------------------

    def discover(
        self,
        *,
        start: str | None = None,
        end: str | None = None,
        **_filters: object,
    ) -> list[QuestionDescriptor]:
        return [sq.descriptor for sq in self.questions]

    def events(self, q: QuestionDescriptor) -> Iterator[MarketEvent]:
        sq = self._find(q)
        merged: list[MarketEvent] = []
        merged.extend(sq.book_snapshots)
        merged.extend(sq.trades)
        merged.extend(sq.reference_events)
        merged.extend(sq.settlement_events)
        merged.sort(key=lambda e: e.ts_ns)
        yield from merged

    def question_view(
        self,
        q: QuestionDescriptor,
        *,
        now_ns: int,
        settled: bool,
    ) -> QuestionView:
        sq = self._find(q)
        return build_question_view(
            q,
            now_ns=now_ns,
            strike=sq.strike,
            settled=settled,
            settled_side=sq.outcome if settled else None,
        )

    def resolved_outcome(
        self, q: QuestionDescriptor
    ) -> Literal["yes", "no", "unknown"]:
        return self._find(q).outcome

    # ---- helpers ---------------------------------------------------------

    def _find(self, q: QuestionDescriptor) -> SyntheticQuestion:
        for sq in self.questions:
            if sq.descriptor.question_id == q.question_id:
                return sq
        raise KeyError(f"Unknown synthetic question: {q.question_id}")


def make_default_binary_question(
    *,
    question_id: str = "synth-q-0",
    question_idx: int = 1,
    start_ts_ns: int = 0,
    duration_ns: int = 10 * 60 * 1_000_000_000,  # 10 min
    strike: float = 60_000.0,
    yes_symbol: str = "synth-yes",
    no_symbol: str = "synth-no",
    outcome: Literal["yes", "no", "unknown"] = "yes",
) -> SyntheticQuestion:
    """Build a small deterministic binary question for tests and CLI smoke.

    Layout:
    - 10 minute lifetime, single scan interval of 60s by default
    - 2 legs (yes, no) with periodic top-of-book snapshots; YES bid drifts up
      from 0.30 to 0.70, ask from 0.32 to 0.72
    - Three BTC reference klines (open/high/low/close near ``strike``)
    - One settlement event at end_ts_ns for the outcome leg
    """
    end_ts_ns = start_ts_ns + duration_ns
    descriptor = QuestionDescriptor(
        question_id=question_id,
        question_idx=question_idx,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
        leg_symbols=(yes_symbol, no_symbol),
        klass="priceBinary",
        underlying="BTC",
    )

    # Book snapshots: 11 snapshots evenly spaced; YES bid linearly rises 0.30→0.70.
    n = 11
    snapshots: list[BookSnapshot] = []
    for i in range(n):
        t = start_ts_ns + int(i * duration_ns / (n - 1))
        # Anchor first snapshot strictly *after* start so depth-clear events at
        # start_ts_ns can resolve first.
        if i == 0:
            t = start_ts_ns + 1_000_000  # +1ms
        bid_yes = 0.30 + 0.40 * i / (n - 1)
        ask_yes = bid_yes + 0.02
        snapshots.append(
            BookSnapshot(
                ts_ns=t,
                symbol=yes_symbol,
                bids=((round(bid_yes, 4), 200.0),),
                asks=((round(ask_yes, 4), 200.0),),
            )
        )
        bid_no = 1.0 - ask_yes
        ask_no = 1.0 - bid_yes
        snapshots.append(
            BookSnapshot(
                ts_ns=t,
                symbol=no_symbol,
                bids=((round(bid_no, 4), 200.0),),
                asks=((round(ask_no, 4), 200.0),),
            )
        )

    # A handful of synthetic trades on YES.
    trades: list[TradeEvent] = []
    for i in range(1, n, 2):
        t = start_ts_ns + int(i * duration_ns / (n - 1)) + 100_000
        trades.append(
            TradeEvent(
                ts_ns=t,
                symbol=yes_symbol,
                side="buy",
                price=round(0.30 + 0.40 * i / (n - 1) + 0.01, 4),
                size=5.0,
            )
        )

    # Reference klines aligned to scan ticks (1m).
    refs: list[ReferenceEvent] = []
    n_klines = max(3, duration_ns // (60 * 1_000_000_000))
    for i in range(int(n_klines)):
        t = start_ts_ns + int(i * 60 * 1_000_000_000)
        close = strike * (1.0 + 0.001 * (i - n_klines / 2))
        refs.append(
            ReferenceEvent(
                ts_ns=t,
                symbol="BTC",
                high=close * 1.001,
                low=close * 0.999,
                close=close,
            )
        )

    settle: list[SettlementEvent] = [
        SettlementEvent(ts_ns=end_ts_ns, question_idx=question_idx, outcome=outcome)
    ]

    return SyntheticQuestion(
        descriptor=descriptor,
        book_snapshots=tuple(snapshots),
        trades=tuple(trades),
        reference_events=tuple(refs),
        settlement_events=tuple(settle),
        outcome=outcome,
        strike=strike,
    )


# ---------------------------------------------------------------------------
# Dummy strategy used by the CLI smoke test (Task A acceptance criterion)
# ---------------------------------------------------------------------------

from collections.abc import Mapping  # noqa: E402

from hlanalysis.strategy.base import Strategy  # noqa: E402
from hlanalysis.strategy.types import (  # noqa: E402
    Action,
    BookState,
    Decision,
    Diagnostic,
    OrderIntent,
    Position,
)


class _DummyEnterStrategy(Strategy):
    """Test-only strategy: enter YES at the first scan tick, then HOLD.

    The CLI smoke test acceptance requires producing fills.parquet without the
    real ``v1_late_resolution`` strategy (which Task E registers). This strategy
    is small enough to live next to the synthetic data source.
    """

    name = "_dummy_enter_yes"

    def __init__(self, params: dict[str, object]):
        self.params = params
        self._fired = False
        self._size = float(params.get("size", 10.0))

    def evaluate(
        self,
        *,
        question,
        books: Mapping[str, BookState],
        reference_price: float,
        recent_returns: tuple[float, ...],
        recent_volume_usd: float,
        position: Position | None,
        now_ns: int,
        recent_hl_bars: tuple[tuple[float, float], ...] = (),
    ) -> Decision:
        diag = (Diagnostic(level="info", message="dummy-tick"),)
        if not self._fired and position is None and question.yes_symbol in books:
            self._fired = True
            book = books[question.yes_symbol]
            if book.ask_px is None:
                return Decision(action=Action.HOLD, diagnostics=diag)
            intent = OrderIntent(
                question_idx=question.question_idx,
                symbol=question.yes_symbol,
                side="buy",
                size=self._size,
                limit_price=min(1.0, book.ask_px + 0.01),
                cloid=f"hla-dummy-{now_ns}",
            )
            return Decision(
                action=Action.ENTER, intents=(intent,), diagnostics=diag
            )
        return Decision(action=Action.HOLD, diagnostics=diag)


def build_dummy_enter_strategy(params: dict[str, object]) -> _DummyEnterStrategy:
    return _DummyEnterStrategy(params)


__all__ = [
    "SyntheticDataSource",
    "SyntheticQuestion",
    "make_default_binary_question",
    "build_dummy_enter_strategy",
]
