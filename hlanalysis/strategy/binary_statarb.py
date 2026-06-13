from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .base import Strategy
from .intents import make_entry_intent, make_exit_intent, round_size
from .types import (
    Action,
    BookState,
    Decision,
    Diagnostic,
    Position,
    QuestionView,
)


@dataclass(frozen=True, slots=True)
class BinaryStatArbConfig:
    lookback_seconds: int
    sampling_dt_seconds: int
    ewma_lambda: float
    z_entry: float
    z_exit: float
    mid_lo: float
    mid_hi: float
    max_position_usd: float
    stop_loss_pct: float | None
    time_stop_seconds: int
    fee_taker: float
    half_spread_assumption: float


@dataclass(slots=True)
class _State:
    """Per-question rolling state. Resets between questions (one instance per question)."""

    last_sample_ns: int = 0
    sample_count: int = 0
    ewma_mean: float = 0.0
    ewma_var: float = 0.0


class BinaryStatArbStrategy(Strategy):
    name = "binary_statarb"

    def __init__(self, cfg: BinaryStatArbConfig) -> None:
        self.cfg = cfg
        # Strategy is rebuilt per question by the runner, so single _State is fine.
        self._state = _State()

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
        recent_hl_bars: tuple[tuple[float, float], ...] = (),
    ) -> Decision:
        if question.settled:
            if position is not None:
                return Decision(action=Action.EXIT, diagnostics=(Diagnostic("info", "exit_settlement"),))
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "settled"),))

        yes = books.get(question.yes_symbol)
        no_ = books.get(question.no_symbol)
        if (
            yes is None
            or yes.bid_px is None
            or yes.ask_px is None
            or no_ is None
            or no_.bid_px is None
            or no_.ask_px is None
        ):
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "no_book"),))

        yes_mid = (yes.bid_px + yes.ask_px) / 2.0
        tau_s = (question.expiry_ns - now_ns) / 1e9

        # Sample at fixed cadence
        dt_ns = self.cfg.sampling_dt_seconds * 1_000_000_000
        if now_ns - self._state.last_sample_ns >= dt_ns:
            self._update_state(yes_mid)
            self._state.last_sample_ns = now_ns

        warmup_needed = max(2, self.cfg.lookback_seconds // self.cfg.sampling_dt_seconds)
        if self._state.sample_count < warmup_needed:
            return Decision(
                action=Action.HOLD,
                diagnostics=(Diagnostic("info", "warmup", (("count", str(self._state.sample_count)),)),),
            )

        std = math.sqrt(max(self._state.ewma_var, 1e-12))
        z = (yes_mid - self._state.ewma_mean) / std

        # Have-position: exit logic
        if position is not None:
            held = yes if position.symbol == question.yes_symbol else no_
            # Time stop (must flatten — strategy has no edge into resolution)
            if self.cfg.time_stop_seconds > 0 and tau_s < self.cfg.time_stop_seconds:
                return self._exit_intent(question, position, held, reason="exit_time_stop")
            # Hard stop
            if self.cfg.stop_loss_pct is not None and held.bid_px <= position.stop_loss_price:
                return self._exit_intent(question, position, held, reason="exit_stop_loss")
            # Reversion captured
            if abs(z) <= self.cfg.z_exit:
                return self._exit_intent(question, position, held, reason="exit_reversion")
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "hold_in_pos", (("z", f"{z:.3f}"),)),))

        # No position: entry logic
        if not (self.cfg.mid_lo <= yes_mid <= self.cfg.mid_hi):
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "mid_out_of_band"),))

        if z <= -self.cfg.z_entry:
            target_book, target_symbol = yes, question.yes_symbol
        elif z >= self.cfg.z_entry:
            target_book, target_symbol = no_, question.no_symbol
        else:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "no_signal", (("z", f"{z:.3f}"),)),))

        size = max(0.0, round_size(self.cfg.max_position_usd, target_book.ask_px))
        if size <= 0:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("warn", "size_zero"),))

        intent = make_entry_intent(
            question,
            symbol=target_symbol,
            size=size,
            limit_price=target_book.ask_px,
        )
        return Decision(
            action=Action.ENTER,
            intents=(intent,),
            diagnostics=(
                Diagnostic("info", "entry"),
                Diagnostic(
                    "info",
                    "z",
                    (
                        ("z", f"{z:.3f}"),
                        ("mean", f"{self._state.ewma_mean:.4f}"),
                        ("std", f"{std:.4f}"),
                    ),
                ),
            ),
        )

    def _update_state(self, x: float) -> None:
        # EWMA mean & variance (Welford-style for EWMA)
        lam = self.cfg.ewma_lambda
        if self._state.sample_count == 0:
            self._state.ewma_mean = x
            self._state.ewma_var = 0.0
        else:
            diff = x - self._state.ewma_mean
            self._state.ewma_mean = lam * self._state.ewma_mean + (1.0 - lam) * x
            self._state.ewma_var = lam * (self._state.ewma_var + (1.0 - lam) * diff * diff)
        self._state.sample_count += 1

    def _exit_intent(self, question: QuestionView, position: Position, held: BookState, *, reason: str) -> Decision:
        # NB: ``reason`` tags the diagnostic only — the intent keeps the legacy
        # empty ``exit_reason`` (this strategy predates intent-level reasons).
        intent = make_exit_intent(question, position, limit_price=held.bid_px)
        return Decision(
            action=Action.EXIT,
            intents=(intent,),
            diagnostics=(Diagnostic("info", reason),),
        )


from hlanalysis.backtest.core.registry import register  # noqa: E402


@register("v4_binary_statarb")
def build_v4_binary_statarb(params: dict) -> BinaryStatArbStrategy:
    cfg = BinaryStatArbConfig(
        lookback_seconds=int(params["lookback_seconds"]),
        sampling_dt_seconds=int(params.get("sampling_dt_seconds", 60)),
        ewma_lambda=float(params.get("ewma_lambda", 0.95)),
        z_entry=float(params["z_entry"]),
        z_exit=float(params["z_exit"]),
        mid_lo=float(params.get("mid_lo", 0.2)),
        mid_hi=float(params.get("mid_hi", 0.8)),
        max_position_usd=float(params.get("max_position_usd", 100.0)),
        stop_loss_pct=(float(params["stop_loss_pct"]) if params.get("stop_loss_pct") is not None else None),
        time_stop_seconds=int(params.get("time_stop_seconds", 600)),
        fee_taker=float(params.get("fee_taker", 0.0)),
        half_spread_assumption=float(params.get("half_spread_assumption", 0.005)),
    )
    return BinaryStatArbStrategy(cfg)
