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


# ---------------------------------------------------------------------------
# Leg-aware helpers — handle both priceBinary and priceBucket question classes
# uniformly. Bucket layout mirrors v1 (late_resolution.py): for N thresholds we
# get N+1 outcomes × 2 legs each, alternating [yes_o0, no_o0, yes_o1, no_o1, ...].
# Helpers are intentionally duplicated rather than imported from v1 so the two
# strategies stay decoupled and can diverge over time.
# ---------------------------------------------------------------------------


def _kv_get(qv: QuestionView, key: str) -> str:
    for k, v in qv.kv:
        if k == key:
            return v
    return ""


def _winning_region(qv: QuestionView, symbol: str) -> tuple[Optional[float], Optional[float]]:
    """Return (lo, hi) such that the leg `symbol` wins iff underlying ∈ [lo, hi]
    at expiry. None on either side denotes ±∞.

    Binary YES wins above strike → (strike, None); NO wins at-or-below → (None, strike).
    Bucket (HL convention, YES at even leg index, NO at odd):
      outcome 0       → YES wins below thr[0]
      outcome 1..N-1  → YES wins inside (thr[i-1], thr[i])
      outcome N       → YES wins above thr[-1]
    NO of an edge bucket inverts to the opposite half-line. NO of a middle
    bucket is the union of two half-lines (non-contiguous) and is signaled
    by returning (None, None) — callers must skip such legs as no-edge.
    """
    if qv.klass == "priceBinary":
        if symbol == qv.yes_symbol:
            return (qv.strike, None)
        if symbol == qv.no_symbol:
            return (None, qv.strike)
        return (None, None)

    if qv.klass != "priceBucket" or not qv.leg_symbols or symbol not in qv.leg_symbols:
        return (None, None)

    thresholds_raw = _kv_get(qv, "priceThresholds")
    thr = [float(t) for t in thresholds_raw.split(",") if t.strip()]
    if not thr:
        return (None, None)

    idx = qv.leg_symbols.index(symbol)
    outcome_pos = idx // 2
    side_idx = idx % 2  # 0=YES, 1=NO

    if outcome_pos == 0:
        yes_lo: Optional[float] = None
        yes_hi: Optional[float] = thr[0]
    elif outcome_pos == len(thr):
        yes_lo, yes_hi = thr[-1], None
    elif 0 < outcome_pos < len(thr):
        yes_lo, yes_hi = thr[outcome_pos - 1], thr[outcome_pos]
    else:
        return (None, None)

    if side_idx == 0:
        return (yes_lo, yes_hi)
    # NO leg
    if yes_lo is None:
        return (yes_hi, None)
    if yes_hi is None:
        return (None, yes_lo)
    # Middle-bucket NO = union of two half-lines (non-contiguous). Treat as
    # ineligible for entry; v3 exit-side handling needs care here too — see
    # _p_leg_win_prob which returns 0 in this case.
    return (None, None)


def _p_leg_win_prob(
    *,
    reference_price: float,
    lo: Optional[float],
    hi: Optional[float],
    sigma: float,
    mu_eff: float,
    tau_yr: float,
) -> Optional[float]:
    """P(lo < S_T ≤ hi) under GBM with drift μ, Itô-corrected.

    Returns None when the leg has no contiguous winning region (e.g. NO of a
    middle bucket — caller must skip). For unbounded edges, +∞/−∞ map to the
    natural CDF limits.
    """
    if lo is None and hi is None:
        return None
    sigma_sqrt_tau = sigma * math.sqrt(tau_yr)
    drift = (mu_eff - 0.5 * sigma * sigma) * tau_yr

    def _N_above(k: float) -> float:
        """P(S_T > k) under GBM. With d = (ln(S/K) + drift) / (σ√τ), this is N(d)."""
        d = (math.log(reference_price / k) + drift) / sigma_sqrt_tau
        return float(norm.cdf(d))

    p_above_lo = 1.0 if lo is None else _N_above(lo)
    p_above_hi = 0.0 if hi is None else _N_above(hi)
    # P(lo < S_T ≤ hi) = P(S > lo) − P(S > hi)
    return max(0.0, p_above_lo - p_above_hi)


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
    # v3.1 entry upper-edge filter. Trade-level analysis of v3 PM-corpus run
    # surfaced an asymmetry: entries claiming edge >= 0.20 had hit rate 55%
    # (same as the rest) but with full-position wipeouts on the losers, netting
    # -$569 on 176 trades. Hypothesis: when GBM claims huge edge, the market
    # usually knows something the realized-vol model doesn't (event risk).
    # Filtering edge >= 0.20 lifted PnL +37% and cut max DD -84% on PM.
    # None disables the filter (preserves v3 baseline behavior).
    edge_max: Optional[float] = None


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

    def _evaluate_entry(
        self, *, question: QuestionView, books: Mapping[str, BookState], reference_price: float, sigma: float, mu_eff: float, tau_yr: float,
    ) -> Decision:
        # TTE entry window
        tau_s = tau_yr * _ANNUAL_SECONDS
        if not (self.cfg.tte_min_seconds <= tau_s <= self.cfg.tte_max_seconds):
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "tte_out_of_window", (("tte_s", f"{tau_s:.0f}"),)),
            ))

        # Determine candidate legs. For binaries we keep the historical
        # (yes_symbol, no_symbol) ordering so behavior is bit-for-bit unchanged
        # and existing tests stay green. For buckets we iterate leg_symbols
        # but restrict to YES legs only (even indices). The NO of bucket k is
        # structurally redundant with the union of YES legs of all other
        # buckets, and on small HL HIP-4 bucket corpora entering both YES of
        # bucket A and NO of bucket B can produce contradictory bets that
        # net to a guaranteed loss after fees. Mirrors v1 (late_resolution.py).
        legs: tuple[str, ...] = (
            question.leg_symbols
            if question.leg_symbols and question.klass != "priceBinary"
            else (question.yes_symbol, question.no_symbol)
        )
        if question.klass == "priceBucket" and legs:
            legs = tuple(legs[i] for i in range(0, len(legs), 2))
        if not legs:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "no_legs"),))

        # Per-leg edge computation. Skip legs without a quote, without a
        # contiguous winning region, or that fail the favorite gate. We retain
        # explicit (edge_yes, edge_no) bookkeeping for binary diagnostics so
        # downstream parsing (result.py:_parse_edge_fields) is unchanged.
        is_binary = question.klass == "priceBinary"
        per_leg: list[tuple[str, float, float, BookState]] = []  # (symbol, p_win, edge, book)
        for sym in legs:
            book = books.get(sym)
            if book is None or book.ask_px is None:
                continue
            lo, hi = _winning_region(question, sym)
            p_win = _p_leg_win_prob(
                reference_price=reference_price, lo=lo, hi=hi,
                sigma=sigma, mu_eff=mu_eff, tau_yr=tau_yr,
            )
            if p_win is None:
                continue  # NO leg of a middle bucket — no contiguous winning region
            edge = p_win - book.ask_px - self.cfg.fee_taker - self.cfg.half_spread_assumption
            per_leg.append((sym, p_win, edge, book))

        if not per_leg:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "no_book"),))

        # Favorite gate: require the leg's mid ≥ threshold. The binary case
        # naturally collapses to "exactly one side passes" because YES+NO=1.
        # Buckets: filter to legs whose mid ≥ threshold; if no leg passes,
        # HOLD with diagnostic.
        if self.cfg.favorite_threshold > 0.0:
            def _mid(b: BookState) -> float:
                if b.bid_px is not None and b.ask_px is not None:
                    return (b.bid_px + b.ask_px) / 2.0
                return b.ask_px if b.ask_px is not None else (b.bid_px or 0.0)
            per_leg = [t for t in per_leg if _mid(t[3]) >= self.cfg.favorite_threshold]
            if not per_leg:
                return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "no_favorite"),))

        # Pick the leg with the highest edge.
        chosen_sym, chosen_p, chosen_edge, chosen_book = max(per_leg, key=lambda t: t[2])

        # Build a diagnostic preserving the binary schema (p_model/edge_yes/edge_no)
        # so existing parquet writers and tests keep working unchanged.
        if is_binary:
            yes = books.get(question.yes_symbol)
            no_ = books.get(question.no_symbol)
            # Binary always has both legs at this point because per_leg is non-empty
            # (favorite gate already passed if active). p_yes = P(S>strike).
            p_yes_view = _p_leg_win_prob(
                reference_price=reference_price, lo=question.strike, hi=None,
                sigma=sigma, mu_eff=mu_eff, tau_yr=tau_yr,
            ) or 0.0
            edge_yes = (
                p_yes_view - (yes.ask_px if yes and yes.ask_px is not None else 1.0)
                - self.cfg.fee_taker - self.cfg.half_spread_assumption
            )
            edge_no = (
                (1.0 - p_yes_view) - (no_.ask_px if no_ and no_.ask_px is not None else 1.0)
                - self.cfg.fee_taker - self.cfg.half_spread_assumption
            )
            # Apply favorite-gate disabling sentinels so the diagnostic mirrors
            # the legacy behavior exactly (existing tests check exact values).
            if self.cfg.favorite_threshold > 0.0:
                if chosen_sym == question.yes_symbol:
                    edge_no = -1e9
                else:
                    edge_yes = -1e9
            ln_sk = math.log(reference_price / question.strike)
            diag = Diagnostic("info", "edge", (
                ("p_model", f"{p_yes_view:.4f}"),
                ("edge_yes", f"{edge_yes:.4f}"),
                ("edge_no", f"{edge_no:.4f}"),
                ("sigma", f"{sigma:.4f}"),
                ("tau_yr", f"{tau_yr:.12f}"),
                ("ln_sk", f"{ln_sk:.4f}"),
            ))
        else:
            # Bucket diagnostic. We keep the schema column names so the parquet
            # writer is uniform; edge_yes carries the chosen-leg edge, edge_no
            # gets a sentinel that downstream selection (max) ignores.
            diag = Diagnostic("info", "edge", (
                ("p_model", f"{chosen_p:.4f}"),
                ("edge_yes", f"{chosen_edge:.4f}"),
                ("edge_no", f"{-1e9:.4f}"),
                ("sigma", f"{sigma:.4f}"),
                ("tau_yr", f"{tau_yr:.12f}"),
                ("ln_sk", "0.0000"),
                ("chosen_leg", chosen_sym),
            ))

        if chosen_edge <= self.cfg.edge_buffer:
            return Decision(action=Action.HOLD, diagnostics=(diag,))

        if self.cfg.edge_max is not None and chosen_edge >= self.cfg.edge_max:
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "edge_too_extreme", (
                    ("edge", f"{chosen_edge:.4f}"),
                    ("edge_max", f"{self.cfg.edge_max:.4f}"),
                )),
                diag,
            ))

        size = max(0.0, math.floor((self.cfg.max_position_usd / chosen_book.ask_px) * 100) / 100)
        if size <= 0:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("warn", "size_zero"), diag))

        intent = OrderIntent(
            question_idx=question.question_idx,
            symbol=chosen_sym,
            side="buy",
            size=size,
            limit_price=chosen_book.ask_px,
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
        if held is None or held.bid_px is None or held.ask_px is None:
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

        # Rule 2: edge-based exit (the heart of v3) — leg-aware.
        # For binary, _winning_region returns the standard (strike, None) or
        # (None, strike) and the result is bit-for-bit identical to the old
        # p_model / (1-p_model) split.
        lo, hi = _winning_region(question, position.symbol)
        held_p = _p_leg_win_prob(
            reference_price=reference_price, lo=lo, hi=hi,
            sigma=sigma, mu_eff=mu_eff, tau_yr=tau_yr,
        )
        if held_p is None:
            # Middle-bucket NO with no contiguous winning region. Skip edge
            # check; rely on stop_loss / time_stop / settlement.
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "hold_no_region"),
            ))
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
        edge_max=(float(params["edge_max"]) if params.get("edge_max") is not None else None),
    )
    return ThetaHarvesterStrategy(cfg)
