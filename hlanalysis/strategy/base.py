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

    def decision_lookback_seconds(self) -> int | None:
        """Seconds of ``recent_returns``/``recent_hl_bars`` history this strategy
        actually consumes per ``evaluate()``, or ``None`` to use the caller's
        default window.

        The backtest runner reads the σ/drift inputs over the RunConfig default
        (86_400s ≈ a full day), then re-tuples that array every scan tick — at
        dt=5 a 17 280-element array, the single biggest sim cost. A strategy that
        only ever slices the most-recent ``N`` samples can report ``N·dt`` here so
        the runner bounds the array to what matters. ``None`` (default) keeps the
        legacy full window — correct for range-σ strategies (late_resolution's
        Parkinson) until they declare their own bound.

        Mirrors the live scanner's ``_lookback_secs`` derivation. The runner
        provisions a safety margin over the reported value (the time-bounded
        window is re-sliced by COUNT downstream), so report the strategy's true
        consumption, not a padded value.
        """
        return None

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
