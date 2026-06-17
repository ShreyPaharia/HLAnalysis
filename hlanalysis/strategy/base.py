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

    # Whether evaluate() actually reads ``recent_hl_bars``. Only range-based σ
    # estimators (Parkinson, Garman-Klass — used by late_resolution) need the
    # per-bar (high, low) side-channel; return-based strategies (theta_harvester)
    # ignore it. The runner/engine use this to SKIP materialising the HL-bar
    # tuple (a ~1350-element tuple-of-tuples rebuilt every scan tick — ~79% of
    # sim runtime at 1s cadence) for strategies that don't consume it. Default
    # True is the safe/legacy behaviour (always build it).
    consumes_hl_bars: bool = True

    @abstractmethod
    def evaluate(
        self,
        *,
        question: QuestionView,
        books: Mapping[str, BookState],  # symbol -> BookState; includes both legs + ref market
        reference_price: float,  # underlying spot/perp mark, e.g. BTC
        recent_returns: tuple[float, ...],  # log-returns over a fixed lookback for vol calc
        recent_volume_usd: float,  # last-hour notional volume on this question
        position: Position | None,
        now_ns: int,
        # Optional side-channel: (high, low) for each 1m kline in the lookback
        # window, in chronological order. Used by σ estimators that need range
        # data (Parkinson, Garman-Klass). Empty tuple is the default; strategies
        # that only need close-to-close returns may ignore this.
        recent_hl_bars: tuple[tuple[float, float], ...] = (),
    ) -> Decision: ...
