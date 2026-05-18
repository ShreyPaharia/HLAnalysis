from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm  # type: ignore[import-untyped]

from ._numba.vol import sample_std_returns as _nb_sample_std
from .base import Strategy
from .types import (
    Action, BookState, Decision, Diagnostic, OrderIntent, Position, QuestionView,
)

_ANNUAL_SECONDS = 365.25 * 86400.0


@dataclass(frozen=True, slots=True)
class ThetaHarvesterConfig:
    # v2 entry knobs (copied; we deliberately do NOT import ModelEdgeConfig to
    # keep v3 standalone and let the two diverge over time)
    vol_lookback_seconds: int
    vol_sampling_dt_seconds: int
    vol_clip_min: float
    vol_clip_max: float
    edge_buffer: float
    fee_taker: float
    half_spread_assumption: float
    drift_lookback_seconds: int
    drift_blend: float
    max_position_usd: float
    favorite_threshold: float
    tte_min_seconds: int
    tte_max_seconds: int
    # v3 exit knobs
    stop_loss_pct: Optional[float]
    exit_edge_threshold: float          # exit when edge_held_side < this (typically <= 0)
    take_profit_price: Optional[float]  # exit when held_bid >= entry_px + this
    time_stop_seconds: int              # exit when tau_s < this; 0 disables


class ThetaHarvesterStrategy(Strategy):
    name = "theta_harvester"

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
        # Phase A: settlement
        if question.settled:
            if position is not None:
                return Decision(
                    action=Action.EXIT,
                    diagnostics=(Diagnostic("info", "exit_settlement"),),
                )
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "settled"),))

        # Phase B: tau / vol / mu — needed for both entry and exit
        tau_s = (question.expiry_ns - now_ns) / 1e9
        if tau_s <= 0:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "tau_nonpositive"),))
        tau_yr = tau_s / _ANNUAL_SECONDS

        sigma = self._sigma(recent_returns)
        if sigma is None:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "vol_insufficient_data"),))

        mu_eff = self._mu(recent_returns)

        # Phase C: have-position exit rules (the v3 addition)
        if position is not None:
            return self._evaluate_exits(
                question=question, books=books, reference_price=reference_price,
                sigma=sigma, mu_eff=mu_eff, tau_yr=tau_yr, tau_s=tau_s, position=position,
            )

        # Phase D: no-position entry (v2 logic)
        return self._evaluate_entry(
            question=question, books=books, reference_price=reference_price,
            sigma=sigma, mu_eff=mu_eff, tau_yr=tau_yr,
        )

    # -- helpers --

    def _sigma(self, recent_returns: tuple[float, ...]) -> float | None:
        n_keep = max(2, self.cfg.vol_lookback_seconds // self.cfg.vol_sampling_dt_seconds)
        window = recent_returns[-n_keep:]
        if len(window) < 2:
            return None
        raw = float(_nb_sample_std(np.asarray(window, dtype=np.float64)))
        ann = math.sqrt(_ANNUAL_SECONDS / float(self.cfg.vol_sampling_dt_seconds))
        sigma = max(self.cfg.vol_clip_min, min(self.cfg.vol_clip_max, raw * ann))
        return sigma if sigma > 0 else None

    def _mu(self, recent_returns: tuple[float, ...]) -> float:
        if self.cfg.drift_lookback_seconds <= 0 or self.cfg.drift_blend <= 0:
            return 0.0
        n_drift = max(1, self.cfg.drift_lookback_seconds // self.cfg.vol_sampling_dt_seconds)
        window = recent_returns[-n_drift:]
        per_sample = float(np.mean(window))
        ann = per_sample * (_ANNUAL_SECONDS / float(self.cfg.vol_sampling_dt_seconds))
        return self.cfg.drift_blend * ann

    def _p_model(self, *, reference_price: float, strike: float, sigma: float, mu_eff: float, tau_yr: float) -> tuple[float, float]:
        ln_sk = math.log(reference_price / strike)
        d = (ln_sk + (mu_eff - 0.5 * sigma ** 2) * tau_yr) / (sigma * math.sqrt(tau_yr))
        return float(norm.cdf(d)), ln_sk

    def _evaluate_entry(
        self, *, question: QuestionView, books: Mapping[str, BookState], reference_price: float, sigma: float, mu_eff: float, tau_yr: float,
    ) -> Decision:
        # TTE entry window
        tau_s = tau_yr * _ANNUAL_SECONDS
        if not (self.cfg.tte_min_seconds <= tau_s <= self.cfg.tte_max_seconds):
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "tte_out_of_window", (("tte_s", f"{tau_s:.0f}"),)),
            ))

        p_model, ln_sk = self._p_model(
            reference_price=reference_price, strike=question.strike,
            sigma=sigma, mu_eff=mu_eff, tau_yr=tau_yr,
        )

        yes = books.get(question.yes_symbol)
        no_ = books.get(question.no_symbol)
        if yes is None or yes.ask_px is None or no_ is None or no_.ask_px is None:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "no_book"),))

        edge_yes = p_model - yes.ask_px - self.cfg.fee_taker - self.cfg.half_spread_assumption
        edge_no = (1.0 - p_model) - no_.ask_px - self.cfg.fee_taker - self.cfg.half_spread_assumption

        if self.cfg.favorite_threshold > 0.0:
            yes_mid = (yes.bid_px + yes.ask_px) / 2.0 if yes.bid_px is not None else yes.ask_px
            no_mid = (no_.bid_px + no_.ask_px) / 2.0 if no_.bid_px is not None else no_.ask_px
            if yes_mid >= self.cfg.favorite_threshold:
                edge_no = -1e9
            elif no_mid >= self.cfg.favorite_threshold:
                edge_yes = -1e9
            else:
                return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "no_favorite"),))

        diag = Diagnostic("info", "edge", (
            ("p_model", f"{p_model:.4f}"),
            ("edge_yes", f"{edge_yes:.4f}"),
            ("edge_no", f"{edge_no:.4f}"),
            ("sigma", f"{sigma:.4f}"),
            ("tau_yr", f"{tau_yr:.12f}"),
            ("ln_sk", f"{ln_sk:.4f}"),
        ))

        if max(edge_yes, edge_no) <= self.cfg.edge_buffer:
            return Decision(action=Action.HOLD, diagnostics=(diag,))

        if edge_yes >= edge_no:
            target_book, target_symbol = yes, question.yes_symbol
        else:
            target_book, target_symbol = no_, question.no_symbol

        size = max(0.0, math.floor((self.cfg.max_position_usd / target_book.ask_px) * 100) / 100)
        if size <= 0:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("warn", "size_zero"), diag))

        intent = OrderIntent(
            question_idx=question.question_idx,
            symbol=target_symbol,
            side="buy",
            size=size,
            limit_price=target_book.ask_px,
            cloid=f"hla-{uuid.uuid4()}",
            time_in_force="ioc",
        )
        return Decision(
            action=Action.ENTER,
            intents=(intent,),
            diagnostics=(Diagnostic("info", "entry"), diag),
        )

    def _evaluate_exits(
        self, *, question: QuestionView, books: Mapping[str, BookState], reference_price: float, sigma: float, mu_eff: float, tau_yr: float, tau_s: float, position: Position,
    ) -> Decision:
        held = books.get(position.symbol)
        if held is None or held.bid_px is None:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "no_book_exit"),))

        # Rule 1: hard price stop (v2 legacy)
        if self.cfg.stop_loss_pct is not None and held.bid_px <= position.stop_loss_price:
            return self._exit_intent(question, position, held, reason="exit_stop_loss")

        # Rule 4: time stop (cheap)
        if self.cfg.time_stop_seconds > 0 and tau_s < self.cfg.time_stop_seconds:
            return self._exit_intent(question, position, held, reason="exit_time_stop")

        # Rule 3: price take-profit
        if self.cfg.take_profit_price is not None and held.bid_px >= position.avg_entry + self.cfg.take_profit_price:
            return self._exit_intent(question, position, held, reason="exit_take_profit")

        # Rule 2: edge-based exit (the heart of v3)
        p_model, _ = self._p_model(
            reference_price=reference_price, strike=question.strike,
            sigma=sigma, mu_eff=mu_eff, tau_yr=tau_yr,
        )
        # held_p = probability the held leg wins
        held_p = p_model if position.symbol == question.yes_symbol else (1.0 - p_model)
        edge_held = held_p - held.ask_px - self.cfg.fee_taker - self.cfg.half_spread_assumption
        if edge_held < self.cfg.exit_edge_threshold:
            return self._exit_intent(question, position, held, reason="exit_edge")

        return Decision(action=Action.HOLD, diagnostics=(
            Diagnostic("info", "hold", (
                ("edge_held", f"{edge_held:.4f}"),
                ("held_p", f"{held_p:.4f}"),
                ("tau_s", f"{tau_s:.0f}"),
            )),
        ))

    def _exit_intent(self, question: QuestionView, position: Position, held: BookState, *, reason: str) -> Decision:
        intent = OrderIntent(
            question_idx=question.question_idx,
            symbol=position.symbol,
            side="sell" if position.qty > 0 else "buy",
            size=abs(position.qty),
            limit_price=held.bid_px,  # type: ignore[arg-type]
            cloid=f"hla-{uuid.uuid4()}",
            time_in_force="ioc",
            reduce_only=True,
        )
        return Decision(
            action=Action.EXIT, intents=(intent,),
            diagnostics=(Diagnostic("info", reason),),
        )


from hlanalysis.backtest.core.registry import register  # noqa: E402


@register("v3_theta_harvester")
def build_v3_theta_harvester(params: dict) -> ThetaHarvesterStrategy:
    cfg = ThetaHarvesterConfig(
        vol_lookback_seconds=int(params["vol_lookback_seconds"]),
        vol_sampling_dt_seconds=int(params.get("vol_sampling_dt_seconds", 60)),
        vol_clip_min=float(params.get("vol_clip_min", 0.05)),
        vol_clip_max=float(params.get("vol_clip_max", 3.0)),
        edge_buffer=float(params["edge_buffer"]),
        fee_taker=float(params.get("fee_taker", 0.0)),
        half_spread_assumption=float(params.get("half_spread_assumption", 0.0)),
        drift_lookback_seconds=int(params.get("drift_lookback_seconds", 0)),
        drift_blend=float(params.get("drift_blend", 1.0 if int(params.get("drift_lookback_seconds", 0)) else 0.0)),
        max_position_usd=float(params.get("max_position_usd", 100.0)),
        favorite_threshold=float(params.get("favorite_threshold", 0.0)),
        tte_min_seconds=int(params.get("tte_min_seconds", 0)),
        tte_max_seconds=int(params.get("tte_max_seconds", 10**9)),
        stop_loss_pct=(float(params["stop_loss_pct"]) if params.get("stop_loss_pct") is not None else None),
        exit_edge_threshold=float(params["exit_edge_threshold"]),
        take_profit_price=(float(params["take_profit_price"]) if params.get("take_profit_price") is not None else None),
        time_stop_seconds=int(params.get("time_stop_seconds", 0)),
    )
    return ThetaHarvesterStrategy(cfg)
