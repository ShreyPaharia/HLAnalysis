from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .base import Strategy
from .types import (
    Action,
    BookState,
    Decision,
    Diagnostic,
    OrderIntent,
    Position,
    QuestionView,
)


@dataclass(frozen=True, slots=True)
class LateResolutionConfig:
    tte_min_seconds: int
    tte_max_seconds: int
    price_extreme_threshold: float    # winning-leg ask must be ≥ this (e.g. 0.95)
    distance_from_strike_usd_min: float
    vol_max: float                    # annualised stdev of log-returns ceiling
    max_position_usd: float
    stop_loss_pct: float              # absolute % drawdown at which the exit fires
    max_strike_distance_pct: float    # reject if |strike − BTC|/BTC > this
    min_recent_volume_usd: float
    stale_data_halt_seconds: int
    # Upper bound on entry ask. Histogram showed v1 entries at >0.99 contribute
    # zero PnL after the [0,1] fill clamp — they pay $1 and settle $1. Default
    # 1.0 disables the cap (existing behavior); set to ~0.99 to skip dead trades.
    price_extreme_max: float = 1.0
    # Joint vol+distance gate. safety_d = |ln(BTC/strike)| / (σ_1m * sqrt(tte_min)).
    # Higher = favorite more securely on its side of the strike given remaining time.
    # Default 0 disables (existing behavior); set positive to require min standard
    # deviations of "safety" before entering.
    min_safety_d: float = 0.0
    # Lookback for σ in the safety gate AND vol_max gate. Runner provides 24h of
    # 1m returns; we slice to the last N (= vol_lookback_seconds // 60). Shorter =
    # more reactive to current regime. Default 1800s (30min, 30 samples).
    vol_lookback_seconds: int = 1800


class LateResolutionStrategy(Strategy):
    """Phase 1 heuristic late-resolution arb on HIP-4 binaries.

    Entry: TTE in window AND winning leg's ask ≥ extreme threshold AND
           |strike − BTC| ≥ distance_min AND realized vol ≤ cap AND book health OK
           AND no existing position on this question.
    Side:  buy whichever leg BTC says wins, IOC at top-of-book ask.
    Exit:  handled by separate `evaluate_exit()` (Task 6) — settlement is engine-driven,
           stop-loss is enforced by risk gate continuously; strategy returns EXIT
           only as a soft signal alongside the engine's hard stop.
    """

    name = "late_resolution"

    def __init__(self, cfg: LateResolutionConfig) -> None:
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
        diags: list[Diagnostic] = []

        if question.settled:
            if position is not None:
                # Settlement-driven exit: no order needed; engine resolves PnL via venue.
                return Decision(
                    action=Action.EXIT,
                    intents=(),
                    diagnostics=(Diagnostic("info", "exit_settlement"),),
                )
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "settled"),))

        if position is not None:
            # Soft stop-loss signal. The risk gate is the authoritative enforcer
            # (Plan 1B); we mirror it here so logging/diagnostics line up.
            held_book = books.get(position.symbol)
            if held_book is not None and held_book.bid_px is not None:
                if held_book.bid_px <= position.stop_loss_price:
                    intent = OrderIntent(
                        question_idx=question.question_idx,
                        symbol=position.symbol,
                        side="sell" if position.qty > 0 else "buy",
                        size=abs(position.qty),
                        limit_price=held_book.bid_px,
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

        # 1) TTE
        tte_s = (question.expiry_ns - now_ns) / 1e9
        if not (self.cfg.tte_min_seconds <= tte_s <= self.cfg.tte_max_seconds):
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "tte_out_of_window", (("tte_s", f"{tte_s:.0f}"),)),
            ))

        # 2) Pick the leg with the best ask in [threshold, max] across all legs
        # of this question. Generalises to multi-outcome (priceBucket): we follow
        # market consensus per-leg rather than mapping a single strike.
        # For non-binary classes (priceBucket etc.) we restrict to YES legs only
        # — buying NO of one bucket is structurally betting against a single
        # bucket, but it's the SAME exposure as a combination of other YES legs
        # at worse prices. YES-only avoids that redundancy.
        legs = question.leg_symbols or (
            (question.yes_symbol, question.no_symbol) if question.yes_symbol else ()
        )
        if not legs:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "no_legs"),))
        eligible: tuple[str, ...]
        if question.klass == "priceBinary":
            eligible = legs  # both YES (idx 0) and NO (idx 1) tradable
        else:
            eligible = tuple(legs[i] for i in range(0, len(legs), 2))  # YES legs only

        stale_ns = self.cfg.stale_data_halt_seconds * 1_000_000_000
        best_symbol: str | None = None
        best_book: BookState | None = None
        best_ask = -1.0
        for sym in eligible:
            b = books.get(sym)
            if b is None or b.ask_px is None or b.bid_px is None:
                continue
            if now_ns - b.last_l2_ts_ns > stale_ns:
                continue
            if not (self.cfg.price_extreme_threshold <= b.ask_px <= self.cfg.price_extreme_max):
                continue
            if b.ask_px > best_ask:
                best_ask = b.ask_px
                best_symbol = sym
                best_book = b

        if best_book is None or best_symbol is None:
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "no_extreme_leg"),
            ))
        win = best_book
        win_symbol = best_symbol

        # Skipped strike-distance gate for multi-outcome compatibility — buckets
        # don't have a single strike. Binary callers can tighten via config if needed.

        # 6) Realized vol cap (sample stdev of log-returns; treat as raw, not annualised)
        # Slice recent_returns to last vol_lookback_seconds (runner provides 24h of 1m bars).
        if len(recent_returns) < 2:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "vol_insufficient_data"),))
        n_keep = max(2, self.cfg.vol_lookback_seconds // 60)
        returns_window = recent_returns[-n_keep:] if len(recent_returns) > n_keep else recent_returns
        if len(returns_window) < 2:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "vol_insufficient_data"),))
        vol = float(np.std(returns_window, ddof=1))
        if vol > self.cfg.vol_max:
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "vol_above_cap", (("vol", f"{vol:.4f}"),)),
            ))

        # 6b) Joint safety gate: how many σ is BTC from the strike given remaining
        # time? safety_d = |ln(BTC/strike)| / (σ_1m * sqrt(tte_min)). Skip when too
        # close to the flip line. Only meaningful for binaries with a real strike.
        if self.cfg.min_safety_d > 0.0 and question.klass == "priceBinary" and question.strike > 0 and reference_price > 0:
            tte_min = max(tte_s / 60.0, 1.0)
            sigma_window = vol * math.sqrt(tte_min)
            if sigma_window > 0:
                log_dist = abs(math.log(reference_price / question.strike))
                safety_d = log_dist / sigma_window
                if safety_d < self.cfg.min_safety_d:
                    return Decision(action=Action.HOLD, diagnostics=(
                        Diagnostic("info", "safety_d_below_min",
                                   (("d", f"{safety_d:.3f}"),)),
                    ))

        # 7) Recent-volume sanity (avoid dead questions)
        if recent_volume_usd < self.cfg.min_recent_volume_usd:
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "low_volume", (("vol_usd", f"{recent_volume_usd:.0f}")),),
            ))

        # 8) Build the IOC intent. Size = max_position_usd / ask_px, taking entry
        # cost as a proxy for notional. Risk gate caps this again.
        size = max(0.0, math.floor((self.cfg.max_position_usd / win.ask_px) * 100) / 100)
        if size <= 0:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("warn", "size_zero"),))

        # limit_price = price_extreme_max so realized fills never exceed the cap
        # even after slippage; sim caps fill at limit, the engine's risk gate uses
        # the same number as its hard upper bound on entry cost.
        intent = OrderIntent(
            question_idx=question.question_idx,
            symbol=win_symbol,
            side="buy",
            size=size,
            limit_price=self.cfg.price_extreme_max,
            cloid=f"hla-{uuid.uuid4()}",
            time_in_force="ioc",
        )
        return Decision(
            action=Action.ENTER,
            intents=(intent,),
            diagnostics=(Diagnostic("info", "entry"),),
        )
