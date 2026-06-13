"""v5 delta-hedged binary: v3 entries + a BTC perp hedge ledger.

Re-uses v3's entry/exit logic verbatim by delegation, and additionally emits
hedge OrderIntents on the configured hedge symbol whenever the binary position
delta differs from the hedge book's tracked delta by more than rebalance_threshold.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

from scipy.stats import norm  # type: ignore[import-untyped]

from .base import Strategy
from .theta_harvester import ThetaHarvesterConfig, ThetaHarvesterStrategy
from .types import (
    Action,
    BookState,
    Decision,
    Diagnostic,
    OrderIntent,
    Position,
    QuestionView,
)
from .vol import ANNUAL_SECONDS


def binary_delta(*, reference_price: float, strike: float, sigma: float, tau_yr: float, mu_eff: float) -> float:
    """Δ_YES = ∂P(YES wins)/∂S = φ(d) / (S σ √τ)."""
    if tau_yr <= 0 or sigma <= 0 or reference_price <= 0:
        return 0.0
    ln_sk = math.log(reference_price / strike)
    d = (ln_sk + (mu_eff - 0.5 * sigma**2) * tau_yr) / (sigma * math.sqrt(tau_yr))
    phi = float(norm.pdf(d))
    return phi / (reference_price * sigma * math.sqrt(tau_yr))


@dataclass(frozen=True, slots=True)
class DeltaHedgedConfig:
    # Inherit v3 binary knobs in full
    binary: ThetaHarvesterConfig
    # Hedge knobs
    hedge_symbol: str
    rebalance_interval_s: int  # 0 means rebalance every tick
    rebalance_threshold: float  # 0 means rebalance every tick (when interval elapsed)


@dataclass(slots=True)
class _HedgeState:
    last_rebalance_ns: int = 0
    hedge_qty_btc: float = 0.0


class DeltaHedgedStrategy(Strategy):
    name = "delta_hedged"

    def __init__(self, cfg: DeltaHedgedConfig) -> None:
        self.cfg = cfg
        self._binary = ThetaHarvesterStrategy(cfg.binary)
        self._hedge_state = _HedgeState()
        # Track which question this strategy instance last saw so we can reset
        # per-question state when the runner reuses one instance across markets.
        self._last_question_idx: int | None = None

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
        # Reset hedge ledger when the runner advances to a new question.
        if self._last_question_idx != question.question_idx:
            self._hedge_state = _HedgeState()
            self._last_question_idx = question.question_idx

        binary_decision = self._binary.evaluate(
            question=question,
            books=books,
            reference_price=reference_price,
            recent_returns=recent_returns,
            recent_volume_usd=recent_volume_usd,
            position=position,
            now_ns=now_ns,
            recent_hl_bars=recent_hl_bars,
        )

        # No binary position and no entry → nothing to hedge
        if position is None and binary_decision.action != Action.ENTER:
            return binary_decision

        # Compute target hedge delta
        target_delta_btc = self._target_delta(
            question=question,
            position=position,
            binary_decision=binary_decision,
            reference_price=reference_price,
            recent_returns=recent_returns,
            now_ns=now_ns,
        )

        # Decide whether to rebalance
        gap = target_delta_btc - self._hedge_state.hedge_qty_btc
        interval_elapsed = (
            self.cfg.rebalance_interval_s == 0
            or now_ns - self._hedge_state.last_rebalance_ns >= self.cfg.rebalance_interval_s * 1_000_000_000
        )
        rel = abs(gap) / max(abs(target_delta_btc), 1e-9)
        should_rebalance = interval_elapsed and (
            rel >= self.cfg.rebalance_threshold or self._hedge_state.hedge_qty_btc == 0.0
        )

        if not should_rebalance:
            return binary_decision

        hedge_book = books.get(self.cfg.hedge_symbol)
        if hedge_book is None or hedge_book.bid_px is None or hedge_book.ask_px is None:
            return binary_decision  # cannot hedge without quote

        side = "buy" if gap > 0 else "sell"
        limit_px = hedge_book.ask_px if side == "buy" else hedge_book.bid_px
        hedge_intent = OrderIntent(
            question_idx=question.question_idx,
            symbol=self.cfg.hedge_symbol,
            side=side,
            size=abs(gap),
            limit_price=limit_px,
            cloid=f"hla-hedge-{uuid.uuid4()}",
            time_in_force="ioc",
        )
        self._hedge_state.hedge_qty_btc = target_delta_btc
        self._hedge_state.last_rebalance_ns = now_ns

        # Combine binary intents (if any) with the hedge intent
        return Decision(
            action=binary_decision.action,
            intents=tuple(binary_decision.intents) + (hedge_intent,),
            diagnostics=tuple(binary_decision.diagnostics)
            + (
                Diagnostic(
                    "info",
                    "hedge_rebalance",
                    (
                        ("target_delta_btc", f"{target_delta_btc:.6f}"),
                        ("gap_btc", f"{gap:.6f}"),
                    ),
                ),
            ),
        )

    def _target_delta(
        self,
        *,
        question: QuestionView,
        position: Position | None,
        binary_decision: Decision,
        reference_price: float,
        recent_returns: tuple[float, ...],
        now_ns: int,
    ) -> float:
        """Return the BTC quantity needed to neutralize the BINARY position delta.

        Sign convention: a long YES at +Δ_bin per binary unit means we want to be SHORT
        BTC by Δ_bin × qty_binary, so we return a NEGATIVE BTC qty (short)."""
        # Determine effective binary position after this tick
        if position is not None:
            held_qty = position.qty
            held_symbol = position.symbol
        elif binary_decision.action == Action.ENTER and binary_decision.intents:
            i0 = binary_decision.intents[0]
            held_qty = i0.size
            held_symbol = i0.symbol
        else:
            return 0.0

        # Recompute model state to obtain sigma/tau/mu
        sigma = self._binary._sigma(recent_returns)
        if sigma is None:
            return 0.0
        tau_yr = (question.expiry_ns - now_ns) / 1e9 / ANNUAL_SECONDS
        if tau_yr <= 0:
            return 0.0
        mu_eff = self._binary._mu(recent_returns)
        delta_per_unit = binary_delta(
            reference_price=reference_price,
            strike=question.strike,
            sigma=sigma,
            tau_yr=tau_yr,
            mu_eff=mu_eff,
        )
        # NO leg's delta is the negative of YES's
        if held_symbol == question.no_symbol:
            delta_per_unit = -delta_per_unit
        # Long binary → short BTC by delta_per_unit × held_qty
        return -delta_per_unit * held_qty


from hlanalysis.backtest.core.registry import register  # noqa: E402


@register("v5_delta_hedged")
def build_v5_delta_hedged(params: dict) -> DeltaHedgedStrategy:
    binary_cfg = ThetaHarvesterConfig(
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
    cfg = DeltaHedgedConfig(
        binary=binary_cfg,
        hedge_symbol=str(params["hedge_symbol"]),
        rebalance_interval_s=int(params.get("rebalance_interval_s", 300)),
        rebalance_threshold=float(params.get("rebalance_threshold", 0.25)),
    )
    return DeltaHedgedStrategy(cfg)
