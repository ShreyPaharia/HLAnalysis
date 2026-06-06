"""WP-driven near-resolution arb strategy for PM NBA single-game winners.

Replaces the v3.1 theta-harvester GBM reference model with a logistic
win-probability (WP) input. The strategy treats `reference_price` as P(home
team wins), looked up by the data source at each PBP event and interpolated
forward to the current tick.

Reused v3.1 mechanics (identical behaviour modulo the p_model source):
- favorite_threshold gate
- edge_buffer / edge_max gates
- pm_binary fee curve
- max_position_usd sizing
- stop_loss_pct, time_stop, edge_held exit

Disabled gates (NBA state is not scalar — these don't apply):
- σ / τ / drift (no GBM)
- LM jump gate
- exit_safety_d
- gamma_lambda path-variance penalty
- min_distance_pct near-strike hover

Why a separate class rather than a flag on ThetaHarvesterStrategy: keeping the
two implementations in separate files lets the GBM strategy stay byte-for-byte
identical to the existing PM/HL backtests (no risk of regressing v3.1 numbers
when iterating on the NBA path).
"""
from __future__ import annotations

from collections.abc import Mapping

from .base import Strategy
from .fee import fee_per_share
from .intents import make_entry_intent, make_exit_intent, round_size
from .theta_harvester import ThetaHarvesterConfig
from .types import (
    Action, BookState, Decision, Diagnostic, Position, QuestionView,
)


class NBAWinProbStrategy(Strategy):
    name = "nba_wp"

    def __init__(self, cfg: ThetaHarvesterConfig) -> None:
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
        recent_hl_bars: tuple[tuple[float, float], ...] = (),
    ) -> Decision:
        # A. Settlement always wins.
        if question.settled:
            if position is not None:
                return Decision(action=Action.EXIT,
                                diagnostics=(Diagnostic("info", "exit_settlement"),))
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "settled"),))

        # B. Need a valid WP probability.
        p_yes_home = float(reference_price)
        if not (0.0 < p_yes_home < 1.0):
            return Decision(action=Action.HOLD,
                            diagnostics=(Diagnostic("info", "wp_unavailable"),))

        # C. TTE — keep the gate; NBA games are < 4h so callers usually leave
        # the bounds wide. tau_s referenced for diagnostics only.
        tau_s = (question.expiry_ns - now_ns) / 1e9
        if tau_s <= 0:
            return Decision(action=Action.HOLD,
                            diagnostics=(Diagnostic("info", "tau_nonpositive"),))
        if not (self.cfg.tte_min_seconds <= tau_s <= self.cfg.tte_max_seconds):
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "tte_out_of_window", (("tte_s", f"{tau_s:.0f}"),)),
            ))

        # D. Existing position → exits.
        if position is not None:
            return self._evaluate_held(
                question=question, books=books, p_yes_home=p_yes_home,
                position=position, tau_s=tau_s,
            )

        # E. No position → entry.
        return self._evaluate_entry(
            question=question, books=books, p_yes_home=p_yes_home,
        )

    # -- entry ------------------------------------------------------------

    def _p_for_leg(self, question: QuestionView, sym: str, p_yes_home: float) -> float:
        if sym == question.yes_symbol:
            return p_yes_home
        if sym == question.no_symbol:
            return 1.0 - p_yes_home
        return 0.0

    def _fee_per_share(self, p: float) -> float:
        return fee_per_share(self.cfg, p, side="entry")

    def _evaluate_entry(
        self, *, question: QuestionView, books: Mapping[str, BookState],
        p_yes_home: float,
    ) -> Decision:
        legs = (question.yes_symbol, question.no_symbol)
        per_leg: list[tuple[str, float, float, BookState]] = []
        for sym in legs:
            book = books.get(sym)
            if book is None or book.ask_px is None:
                continue
            p = self._p_for_leg(question, sym, p_yes_home)
            fee = self._fee_per_share(p)
            edge = p - book.ask_px - fee - self.cfg.half_spread_assumption
            per_leg.append((sym, p, edge, book))

        if not per_leg:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "no_book"),))

        # Favorite-mid gate.
        if self.cfg.favorite_threshold > 0.0:
            def _mid(b: BookState) -> float:
                if b.bid_px is not None and b.ask_px is not None:
                    return (b.bid_px + b.ask_px) / 2.0
                return b.ask_px if b.ask_px is not None else (b.bid_px or 0.0)
            per_leg = [t for t in per_leg if _mid(t[3]) >= self.cfg.favorite_threshold]
            if not per_leg:
                return Decision(action=Action.HOLD,
                                diagnostics=(Diagnostic("info", "no_favorite"),))

        # Bid-notional sanity gate.
        if self.cfg.min_bid_notional_usd > 0.0:
            def _bid_ntl(b: BookState) -> float:
                return (b.bid_px or 0.0) * (b.bid_sz or 0.0)
            per_leg = [t for t in per_leg if _bid_ntl(t[3]) >= self.cfg.min_bid_notional_usd]
            if not per_leg:
                return Decision(action=Action.HOLD, diagnostics=(
                    Diagnostic("info", "bid_notional_too_thin"),
                ))

        chosen_sym, chosen_p, chosen_edge, chosen_book = max(per_leg, key=lambda t: t[2])

        diag = Diagnostic("info", "edge", (
            ("p_model", f"{p_yes_home:.4f}"),
            ("chosen_leg", chosen_sym),
            ("chosen_p", f"{chosen_p:.4f}"),
            ("chosen_edge", f"{chosen_edge:.4f}"),
        ))

        if chosen_edge <= self.cfg.edge_buffer:
            return Decision(action=Action.HOLD, diagnostics=(diag,))
        if self.cfg.edge_max is not None and chosen_edge >= self.cfg.edge_max:
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "edge_too_extreme",
                           (("edge", f"{chosen_edge:.4f}"),)), diag,
            ))

        size = max(0.0, round_size(self.cfg.max_position_usd, chosen_book.ask_px))
        if size <= 0:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("warn", "size_zero"), diag))

        intent = make_entry_intent(
            question, symbol=chosen_sym, size=size, limit_price=chosen_book.ask_px,
        )
        return Decision(
            action=Action.ENTER, intents=(intent,),
            diagnostics=(Diagnostic("info", "entry"), diag),
        )

    # -- held -------------------------------------------------------------

    def _evaluate_held(
        self, *, question: QuestionView, books: Mapping[str, BookState],
        p_yes_home: float, position: Position, tau_s: float,
    ) -> Decision:
        held = books.get(position.symbol)
        if held is None or held.bid_px is None or held.ask_px is None:
            return Decision(action=Action.HOLD,
                            diagnostics=(Diagnostic("info", "no_book_exit"),))

        # Hard stop.
        if self.cfg.stop_loss_pct is not None and held.bid_px <= position.stop_loss_price:
            return self._exit(question, position, held, reason="exit_stop_loss")

        # Time stop.
        if self.cfg.time_stop_seconds > 0 and tau_s < self.cfg.time_stop_seconds:
            return self._exit(question, position, held, reason="exit_time_stop")

        # Take-profit (price).
        if (self.cfg.take_profit_price is not None
                and held.bid_px >= position.avg_entry + self.cfg.take_profit_price):
            return self._exit(question, position, held, reason="exit_take_profit")

        # Edge-based exit.
        held_p = self._p_for_leg(question, position.symbol, p_yes_home)
        exit_fee = fee_per_share(self.cfg, held_p, side="exit")
        if self.cfg.exit_take_profit_mode:
            edge_held = held.bid_px - held_p - exit_fee
            should_exit = edge_held > self.cfg.exit_edge_threshold
        else:
            edge_held = held_p - held.bid_px - exit_fee
            should_exit = edge_held < self.cfg.exit_edge_threshold

        if should_exit:
            return self._exit(question, position, held, reason="exit_edge")

        return Decision(action=Action.HOLD, diagnostics=(
            Diagnostic("info", "hold", (
                ("edge_held", f"{edge_held:.4f}"),
                ("held_p", f"{held_p:.4f}"),
                ("tau_s", f"{tau_s:.0f}"),
            )),
        ))

    def _exit(self, q: QuestionView, pos: Position, held: BookState, *, reason: str) -> Decision:
        intent = make_exit_intent(q, pos, limit_price=held.bid_px, exit_reason=reason)
        return Decision(action=Action.EXIT, intents=(intent,),
                        diagnostics=(Diagnostic("info", reason),))


from hlanalysis.backtest.core.registry import register  # noqa: E402


@register("v31_pm_nba")
def build_v31_pm_nba(params: dict) -> NBAWinProbStrategy:
    cfg = ThetaHarvesterConfig(
        vol_lookback_seconds=int(params.get("vol_lookback_seconds", 300)),
        vol_sampling_dt_seconds=int(params.get("vol_sampling_dt_seconds", 60)),
        vol_clip_min=float(params.get("vol_clip_min", 0.05)),
        vol_clip_max=float(params.get("vol_clip_max", 3.0)),
        edge_buffer=float(params.get("edge_buffer", 0.03)),
        fee_taker=float(params.get("fee_taker", 0.0)),
        half_spread_assumption=float(params.get("half_spread_assumption", 0.0)),
        drift_lookback_seconds=int(params.get("drift_lookback_seconds", 0)),
        drift_blend=float(params.get("drift_blend", 0.0)),
        max_position_usd=float(params.get("max_position_usd", 100.0)),
        favorite_threshold=float(params.get("favorite_threshold", 0.9)),
        tte_min_seconds=int(params.get("tte_min_seconds", 0)),
        tte_max_seconds=int(params.get("tte_max_seconds", 10**9)),
        stop_loss_pct=(float(params["stop_loss_pct"]) if params.get("stop_loss_pct") is not None else None),
        exit_edge_threshold=float(params.get("exit_edge_threshold", 0.0)),
        take_profit_price=(float(params["take_profit_price"]) if params.get("take_profit_price") is not None else None),
        time_stop_seconds=int(params.get("time_stop_seconds", 0)),
        min_bid_notional_usd=float(params.get("min_bid_notional_usd", 0.0)),
        fee_model=str(params.get("fee_model", "pm_binary")),
        fee_rate=float(params.get("fee_rate", 0.03)),  # PM sports default
    )
    return NBAWinProbStrategy(cfg)
