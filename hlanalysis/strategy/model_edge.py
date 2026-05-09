# hlanalysis/strategy/model_edge.py
from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm  # type: ignore[import-untyped]

from .base import Strategy
from .types import (
    Action, BookState, Decision, Diagnostic, OrderIntent, Position, QuestionView,
)


@dataclass(frozen=True, slots=True)
class ModelEdgeConfig:
    vol_lookback_seconds: int
    vol_sampling_dt_seconds: int
    vol_clip_min: float
    vol_clip_max: float
    edge_buffer: float
    fee_taker: float
    half_spread_assumption: float
    stop_loss_pct: float | None
    drift_lookback_seconds: int = 0
    drift_blend: float = 0.0
    max_position_usd: float = 100.0
    # Favorite-only filter. 0.0 disables (consider both sides as before). At
    # 0.7: only bet the side the market favors at ≥70%, skip mid-market noise
    # (0.3 ≤ p ≤ 0.7). Mid is bid+ask)/2; if no clear favorite, HOLD.
    favorite_threshold: float = 0.0
    # TTE entry window (seconds). Hold to expiry, but only ENTER inside this band.
    # Defaults preserve previous behavior (no constraint).
    tte_min_seconds: int = 0           # skip entries when less than this remains
    tte_max_seconds: int = 10**9       # skip entries when more than this remains


_ANNUAL_SECONDS = 365.25 * 86400.0


class ModelEdgeStrategy(Strategy):
    name = "model_edge"

    def __init__(self, cfg: ModelEdgeConfig) -> None:
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
        if question.settled:
            if position is not None:
                return Decision(
                    action=Action.EXIT,
                    diagnostics=(Diagnostic("info", "exit_settlement"),),
                )
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "settled"),))

        if position is not None:
            held = books.get(position.symbol)
            if held is not None and held.bid_px is not None and held.bid_px <= position.stop_loss_price:
                intent = OrderIntent(
                    question_idx=question.question_idx,
                    symbol=position.symbol,
                    side="sell" if position.qty > 0 else "buy",
                    size=abs(position.qty),
                    limit_price=held.bid_px,
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

        # 1) τ in years + entry-window gate
        tau_s = (question.expiry_ns - now_ns) / 1e9
        if tau_s <= 0:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "tau_nonpositive"),))
        if not (self.cfg.tte_min_seconds <= tau_s <= self.cfg.tte_max_seconds):
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "tte_out_of_window", (("tte_s", f"{tau_s:.0f}"),)),
            ))
        tau_yr = tau_s / _ANNUAL_SECONDS

        # 2) σ from recent_returns sliced to lookback; annualize and clip
        n_keep = max(2, self.cfg.vol_lookback_seconds // self.cfg.vol_sampling_dt_seconds)
        returns_window = recent_returns[-n_keep:]
        if len(returns_window) < 2:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "vol_insufficient_data"),))
        sigma_raw = float(np.std(returns_window, ddof=1))
        ann_factor = math.sqrt(_ANNUAL_SECONDS / float(self.cfg.vol_sampling_dt_seconds))
        sigma = max(self.cfg.vol_clip_min, min(self.cfg.vol_clip_max, sigma_raw * ann_factor))
        if sigma <= 0:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "sigma_zero"),))

        # 3) Drift μ
        mu_eff = 0.0
        if self.cfg.drift_lookback_seconds > 0 and self.cfg.drift_blend > 0:
            n_drift = max(1, self.cfg.drift_lookback_seconds // self.cfg.vol_sampling_dt_seconds)
            window = recent_returns[-n_drift:]
            mu_per_sample = float(np.mean(window))
            mu_ann = mu_per_sample * (_ANNUAL_SECONDS / float(self.cfg.vol_sampling_dt_seconds))
            mu_eff = self.cfg.drift_blend * mu_ann

        # 4) p_model under GBM with optional drift
        # d uses Itô-corrected GBM: ln(S/K) + (μ - ½σ²)τ — matching physical measure
        ln_sk = math.log(reference_price / question.strike)
        d = (ln_sk + (mu_eff - 0.5 * sigma ** 2) * tau_yr) / (sigma * math.sqrt(tau_yr))
        p_model = float(norm.cdf(d))

        # 5) Need both legs' asks to compute edges
        yes = books.get(question.yes_symbol)
        no_ = books.get(question.no_symbol)
        if yes is None or yes.ask_px is None or no_ is None or no_.ask_px is None:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "no_book"),))

        edge_yes = p_model - yes.ask_px - self.cfg.fee_taker - self.cfg.half_spread_assumption
        edge_no  = (1.0 - p_model) - no_.ask_px - self.cfg.fee_taker - self.cfg.half_spread_assumption

        # Favorite-only filter: market mid above threshold determines the only
        # eligible side; mid-market markets (no clear favorite) are skipped.
        if self.cfg.favorite_threshold > 0.0:
            yes_mid = (yes.bid_px + yes.ask_px) / 2.0 if yes.bid_px is not None else yes.ask_px
            no_mid  = (no_.bid_px + no_.ask_px) / 2.0 if no_.bid_px is not None else no_.ask_px
            if yes_mid >= self.cfg.favorite_threshold:
                edge_no = -1e9   # disable contrarian NO bet
            elif no_mid >= self.cfg.favorite_threshold:
                edge_yes = -1e9  # disable contrarian YES bet
            else:
                return Decision(action=Action.HOLD, diagnostics=(
                    Diagnostic("info", "no_favorite", (
                        ("yes_mid", f"{yes_mid:.3f}"), ("no_mid", f"{no_mid:.3f}"),
                    )),
                ))

        diag_common = Diagnostic("info", "edge", (
            ("p_model", f"{p_model:.4f}"),
            ("edge_yes", f"{edge_yes:.4f}"),
            ("edge_no", f"{edge_no:.4f}"),
            ("sigma", f"{sigma:.4f}"),
            ("tau_yr", f"{tau_yr:.6f}"),
            ("ln_sk", f"{ln_sk:.4f}"),
        ))

        if max(edge_yes, edge_no) <= self.cfg.edge_buffer:
            return Decision(action=Action.HOLD, diagnostics=(diag_common,))

        if edge_yes >= edge_no:
            target_book = yes
            target_symbol = question.yes_symbol
        else:
            target_book = no_
            target_symbol = question.no_symbol

        size = max(0.0, math.floor((self.cfg.max_position_usd / target_book.ask_px) * 100) / 100)
        if size <= 0:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("warn", "size_zero"), diag_common))

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
            diagnostics=(Diagnostic("info", "entry"), diag_common),
        )
