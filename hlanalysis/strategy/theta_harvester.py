from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger
from scipy.stats import norm  # type: ignore[import-untyped]

from .base import Strategy
from .fee import fee_per_share
from .intents import make_entry_intent, make_exit_intent, round_size
from .regions import winning_region as _winning_region
from .topup import run_topup
from .types import (
    Action, BookState, Decision, Diagnostic, Position, QuestionView,
)
from .vol import ANNUAL_SECONDS, annualized_sigma, bipower_variation_sigma

_ANNUAL_SECONDS = ANNUAL_SECONDS


_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _phi(d: float) -> float:
    """Standard-normal density at ``d``: φ(d) = exp(−d²/2) / √(2π)."""
    return _INV_SQRT_2PI * math.exp(-0.5 * d * d)


def _p_leg_win_prob_and_phi(
    *,
    reference_price: float,
    lo: Optional[float],
    hi: Optional[float],
    sigma: float,
    mu_eff: float,
    tau_yr: float,
) -> Optional[tuple[float, float]]:
    """P(lo < S_T ≤ hi) under GBM (Itô-corrected) AND φ(d) at the nearest
    leg boundary in d-space.

    The φ(d) value is the per-share path-stdev of the held leg's fair value:
    std[Δp_t | hold for τ] ≈ φ(d) / S (S=1 for binary contracts), independent
    of τ. It is used as a path-variance / gamma risk premium term in the
    effective edge: effective_edge = edge − gamma_lambda · φ(d). At d=0 (S=K)
    φ peaks at 0.399; at |d|=2 it drops to 0.054.

    Returns None when the leg has no contiguous winning region (NO of a middle
    bucket — caller skips).
    """
    if lo is None and hi is None:
        return None
    sigma_sqrt_tau = sigma * math.sqrt(tau_yr)
    drift = (mu_eff - 0.5 * sigma * sigma) * tau_yr

    def _d(k: float) -> float:
        return (math.log(reference_price / k) + drift) / sigma_sqrt_tau

    d_lo = _d(lo) if lo is not None else None
    d_hi = _d(hi) if hi is not None else None

    p_above_lo = 1.0 if d_lo is None else float(norm.cdf(d_lo))
    p_above_hi = 0.0 if d_hi is None else float(norm.cdf(d_hi))
    p_win = max(0.0, p_above_lo - p_above_hi)

    # Gamma proxy at the closer-to-strike boundary — that's where path-variance
    # is highest. Conservative for middle buckets.
    if d_lo is not None and d_hi is not None:
        phi_d = max(_phi(d_lo), _phi(d_hi))
    elif d_lo is not None:
        phi_d = _phi(d_lo)
    else:
        phi_d = _phi(d_hi)  # type: ignore[arg-type]

    return (p_win, phi_d)


def _safety_d_for_region(
    *,
    reference_price: float,
    lo: Optional[float],
    hi: Optional[float],
    sigma: float,
    mu_eff: float,
    tau_yr: float,
) -> Optional[float]:
    """Signed σ-normalized distance from ``reference_price`` to the NEAREST
    adverse boundary of the leg's winning region. Positive when BTC is safely
    inside the winning region; negative once already on the losing side.

    Uses the same Itô-corrected d-machinery as ``_p_leg_win_prob_and_phi``:
    ``d(k) = (ln(S/k) + (μ_eff − ½σ²)·τ) / (σ√τ)`` with drift baked in.
    Standalone helper (no dependency on v1's numba kernel) so v3.1 can diverge.
    Returns ``None`` when neither bound is known (e.g. NO leg of a middle bucket)
    or when σ·√τ is non-positive.
    """
    if lo is None and hi is None:
        return None
    if sigma <= 0.0 or tau_yr <= 0.0:
        return None
    sigma_sqrt_tau = sigma * math.sqrt(tau_yr)
    drift = (mu_eff - 0.5 * sigma * sigma) * tau_yr

    def _d(k: float) -> float:
        return (math.log(reference_price / k) + drift) / sigma_sqrt_tau

    if lo is not None and hi is not None:
        return min(_d(lo), -_d(hi))
    if lo is not None:
        return _d(lo)
    return -_d(hi)  # type: ignore[arg-type]


def _p_leg_win_prob(
    *,
    reference_price: float,
    lo: Optional[float],
    hi: Optional[float],
    sigma: float,
    mu_eff: float,
    tau_yr: float,
) -> Optional[float]:
    """Back-compat thin wrapper around ``_p_leg_win_prob_and_phi`` for callers
    that only need the probability (tests, binary diagnostic edge_yes/edge_no).
    """
    res = _p_leg_win_prob_and_phi(
        reference_price=reference_price, lo=lo, hi=hi,
        sigma=sigma, mu_eff=mu_eff, tau_yr=tau_yr,
    )
    if res is None:
        return None
    return res[0]


def _jr_trust_weight(recent_returns: tuple[float, ...], lookback_min: int) -> float:
    """Jump fraction JR = max(0, (RV − BPV)/RV) over the last lookback_min returns.
    Returns the trust scalar `(1 − JR)` ∈ [0, 1]. 1.0 = pure continuous, 0.0 = pure jump."""
    if len(recent_returns) < max(lookback_min, 3):
        return 1.0
    arr = np.asarray(recent_returns[-lookback_min:], dtype=np.float64)
    rv = float(np.sum(arr * arr))
    if rv <= 0.0:
        return 1.0
    # Bipower variation: π/2 · Σ |r_i|·|r_{i-1}| (over consecutive pairs)
    abs_r = np.abs(arr)
    bpv = (np.pi / 2.0) * float(np.sum(abs_r[1:] * abs_r[:-1]))
    jr = max(0.0, min(1.0, (rv - bpv) / rv))
    return 1.0 - jr


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
    # Near-strike hover veto. When set, entries with
    # |reference_price − question.strike| / reference_price below this
    # fraction are vetoed. Only meaningful on priceBinary (the strike concept
    # is well-defined). PM corpus evidence: entries below 0.20% lose
    # -$7.68/entry across 57 entries (~2.7% of all v3.1 entries); the
    # 0.20–0.50% band immediately above is the *best* band (+$16.10/entry).
    # The sharp discontinuity concentrates v3.1's losses. None disables.
    min_distance_pct: Optional[float] = None
    # Bid-notional sanity gate. When > 0, entries where the chosen leg's
    # bid_px × bid_sz is below this USD floor are vetoed. Catches single-
    # share spoof bids that pass numeric thresholds with no real interest.
    # 0 disables (legacy behavior).
    min_bid_notional_usd: float = 0.0
    # Path-variance / gamma risk premium. The current edge formula treats
    # p_model as a deterministic estimate of expected payoff at expiry but
    # ignores that p_t = N(d_t) is itself a random walk under GBM. The std
    # of Δp_t over a hold of length τ is approximately φ(d) / S (the σ√τ
    # terms cancel) — independent of how long τ is or what σ we assume.
    # This term applies a continuous penalty: effective_edge = edge -
    # gamma_lambda · φ(d_nearest_boundary), which scales from ~0 far-from-
    # strike (|d|>2, φ≈0.05) to ~0.40·λ at the strike (φ(0)=0.399). Applied
    # symmetrically at entry (require more edge near strike) and exit (cut
    # held position sooner when path variance is high).
    # None disables (legacy behavior).
    gamma_lambda: Optional[float] = None
    # Strategy-side position topup. When a held position's notional (qty × ask)
    # is below max_position_usd by at least topup_threshold_pct, the strategy
    # re-runs ALL entry gates against the CURRENT state and, if the same leg is
    # still chosen, emits a second IOC ENTER intent sized to fill the shortfall.
    # Exit-eval always runs first — exit signals override topup. Recovers size
    # left on the table by HL IOC partial fills on thin HIP-4 books.
    topup_enabled: bool = True
    topup_threshold_pct: float = 0.2
    topup_min_notional_usd: float = 11.0
    # σ-normalized mid-hold distance exit. Computes the signed distance from
    # BTC to the leg's NEAREST adverse boundary in σ√τ units (the same Itô-
    # corrected d-statistic the entry edge uses). When safety_d < this threshold
    # we close the held leg IOC at bid BEFORE the bid collapses — mirrors v1's
    # exit_safety_d. The existing ``edge_held`` gate fires only AFTER the bid
    # has already moved adversely; this gate fires while the underlying is
    # drifting toward the boundary, catching long-TTE losers earlier.
    # 0.0 disables (legacy behavior). Typical values: 0.25-1.0 (σ-units).
    exit_safety_d: float = 0.0
    # v3.2-volclock: vol estimator selector. "sample_std" preserves the v3.1
    # baseline (rolling sample stdev, ddof=1). "bipower" swaps in the jump-
    # robust Barndorff-Nielsen bipower variation σ — single-bar wicks no longer
    # inflate σ, so the d-statistic stays large after a wick and the strategy
    # can enter on wick-driven mispricings rather than holding while σ_RV
    # decays. The τ·σ² term in the GBM d is unchanged in form; only σ is
    # estimated differently. Default keeps v3.1 byte-for-byte.
    vol_estimator: str = "sample_std"
    # v3.4-LMgate: Lee-Mykland (2008) post-edge jump gate. When set, entries
    # that pass the standard edge_buffer/edge_max gates must additionally have
    # |r_last|/√BV > lm_threshold — i.e., the most recent return looks like a
    # jump relative to the jump-robust BV vol. k≈4 is the standard threshold
    # in the literature. None disables. This is a TRULY τ-free entry trigger:
    # the gate compares a single recent return to recent BV, no τ involved.
    lm_threshold: Optional[float] = None
    # 2026-05-22: take-profit-style exit edge. When True, the held-position
    # exit gate flips from "is the alpha gone?" (held_p - bid - fee < 0) to
    # "is the bid offering above-fair premium net of fees?" (bid - held_p
    # - exit_fee > 0). The new framing avoids the transient-bid-spike
    # churn caused by the legacy ASK-symmetric formulation crossing zero
    # on noise. Default False preserves v3.1 byte-for-byte for backtest
    # reproducibility; toggle via config and validate on PM+HL before
    # promoting. See `exit_fee` for the per-fill exit cost.
    exit_take_profit_mode: bool = False
    # Exit-side fee in fraction (e.g. 0.0007 = 7 bps). Only consumed when
    # ``exit_take_profit_mode`` is True. Models the cost incurred when
    # closing a position at bid; lets us decouple the exit-cost assumption
    # from entry-side ``fee_taker`` (which can stay 0 if the venue subsidises
    # opens).
    exit_fee: float = 0.0007
    # Entry-side fee model:
    #   "flat"      — subtract ``fee_taker`` per share (legacy, HL).
    #   "pm_binary" — Polymarket curve: fee_per_share = fee_rate * p * (1-p).
    # PM's headline 7% taker rate is applied to the *expected loss* leg of the
    # binary, so the realized per-share cost peaks at p=0.5 (0.0175 at 7%)
    # and tapers to ~0 for deep favorites. Default "flat" keeps HL bit-identical.
    fee_model: str = "flat"
    fee_rate: float = 0.0
    # v3.5: momentum / mean-reversion (MR) gate or tilt on the favorite-side
    # entry rule. Default off → v3.1 behavior is preserved bit-for-bit.
    # When `enabled` and `mode == "gate"`: skip entries where the momentum_mr
    # regime is "mr" against the favorite side and |score| > tau_gate.
    # When `enabled` and `mode == "tilt"`: scale the effective edge_buffer
    # by (1 − alpha_tilt * score). Score is signed: + = aligned with favorite.
    # See hlanalysis/strategy/momentum_mr.py and
    # docs/specs/2026-05-28-v35-momentum-mr-design.md.
    momentum_mr_enabled: bool = False
    momentum_mr_indicator: str = "z_ret"      # "z_ret" | "rsi" | "ma_sigma" | "hurst_ou"
    momentum_mr_lookback_min: int = 15
    momentum_mr_mode: str = "gate"            # "gate" | "tilt"
    momentum_mr_tau_gate: float = 1.0
    momentum_mr_alpha_tilt: float = 0.5
    # v3.6: Jump-Ratio trust weight. When True (and momentum_mr_enabled),
    # shrink the indicator score by (1 - JR) where JR = (RV - BPV) / RV is the
    # jump fraction over the same lookback as the indicator. Throttles the tilt
    # when the underlying is gapping (BPV diverges from RV). Default off.
    momentum_mr_jr_trust_weight: bool = False
    # Vol-scaled (variable) TTE entry window. Mirrors late_resolution: when
    # enabled, the upper TTE bound scales with the (annualized) entry σ instead
    # of the fixed tte_max_seconds:
    #   tte_max_eff = tte_max_seconds * (vol_scaled_tte_ref_sigma / σ) ** exp,
    #   clamped to [0, vol_scaled_tte_ceiling_seconds].
    # Low vol → wider window (enter earlier); high vol → narrower window. σ here
    # is the same annualized vol the GBM edge uses, so vol_scaled_tte_ref_sigma
    # is on the annualized scale (~0.3–1.5 for BTC). Default OFF → the fixed cap
    # applies and this path is bit-identical to v3.1.
    vol_scaled_tte_enabled: bool = False
    vol_scaled_tte_ref_sigma: float = 0.0
    vol_scaled_tte_exponent: float = 1.0
    vol_scaled_tte_ceiling_seconds: int = 0


class ThetaHarvesterStrategy(Strategy):
    name = "theta_harvester"

    def __init__(
        self,
        cfg: ThetaHarvesterConfig,
        *,
        cfg_by_class: Mapping[str, ThetaHarvesterConfig] | None = None,
    ) -> None:
        # `cfg` is the default; `cfg_by_class` (built by the runtime from the
        # YAML `theta_overrides:` block) overrides per question.klass. The active
        # config swap happens inside evaluate() so every internal helper can keep
        # reading self.cfg unmodified. Mirrors LateResolutionStrategy.
        self.cfg = cfg
        self._default_cfg = cfg
        self._cfg_by_class: dict[str, ThetaHarvesterConfig] = (
            dict(cfg_by_class) if cfg_by_class else {}
        )

    def _cfg_for(self, question: QuestionView) -> ThetaHarvesterConfig:
        return self._cfg_by_class.get(question.klass, self._default_cfg)

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
        # Resolve and pin the per-class config for this evaluation. The body (and
        # every helper it calls) reads self.cfg directly, so swapping it is the
        # simplest correct way to honor per-class theta overrides. evaluate is
        # single-threaded sync; restoring in finally keeps self.cfg stable for
        # external readers (diagnostics, tests). No overrides → no swap.
        resolved_cfg = self._cfg_for(question)
        if resolved_cfg is self.cfg:
            return self._evaluate(
                question=question, books=books, reference_price=reference_price,
                recent_returns=recent_returns, recent_volume_usd=recent_volume_usd,
                position=position, now_ns=now_ns, recent_hl_bars=recent_hl_bars,
            )
        prev_cfg = self.cfg
        self.cfg = resolved_cfg
        try:
            return self._evaluate(
                question=question, books=books, reference_price=reference_price,
                recent_returns=recent_returns, recent_volume_usd=recent_volume_usd,
                position=position, now_ns=now_ns, recent_hl_bars=recent_hl_bars,
            )
        finally:
            self.cfg = prev_cfg

    def _evaluate(
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

        # Phase C: have-position. Exit gates run FIRST and ALWAYS WIN — if any
        # exit fires we close immediately, never top up. If exits HOLD and topup
        # is enabled, we consider adding to the existing position.
        if position is not None:
            exit_dec = self._evaluate_exits(
                question=question, books=books, reference_price=reference_price,
                sigma=sigma, mu_eff=mu_eff, tau_yr=tau_yr, tau_s=tau_s, position=position,
            )
            if exit_dec.action == Action.EXIT:
                return exit_dec
            if not self.cfg.topup_enabled:
                return exit_dec
            topup_dec = self._evaluate_topup(
                question=question, books=books, reference_price=reference_price,
                sigma=sigma, mu_eff=mu_eff, tau_yr=tau_yr, position=position,
                recent_returns=recent_returns,
            )
            if topup_dec.action == Action.ENTER:
                return topup_dec
            # Topup HOLD diagnostics layered on top of the exit's HOLD diags so
            # we surface both the "no exit" reason and the topup-skip reason.
            return Decision(
                action=Action.HOLD,
                diagnostics=exit_dec.diagnostics + topup_dec.diagnostics,
            )

        # Phase D: no-position entry (v2 logic)
        return self._evaluate_entry(
            question=question, books=books, reference_price=reference_price,
            sigma=sigma, mu_eff=mu_eff, tau_yr=tau_yr,
            recent_returns=recent_returns,
        )

    # -- helpers --

    def _sigma(self, recent_returns: tuple[float, ...]) -> float | None:
        n_keep = max(2, self.cfg.vol_lookback_seconds // self.cfg.vol_sampling_dt_seconds)
        window = recent_returns[-n_keep:]
        if len(window) < 2:
            return None
        arr = np.asarray(window, dtype=np.float64)
        sigma = annualized_sigma(
            arr,
            dt_seconds=self.cfg.vol_sampling_dt_seconds,
            estimator=self.cfg.vol_estimator,
            clip_min=self.cfg.vol_clip_min, clip_max=self.cfg.vol_clip_max,
        )
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
        self, *, question: QuestionView, books: Mapping[str, BookState],
        reference_price: float, sigma: float, mu_eff: float, tau_yr: float,
        recent_returns: tuple[float, ...] = (),
    ) -> Decision:
        # TTE entry window. With the vol-scaled window enabled, the static upper
        # bound is relaxed to the ceiling and the σ-dependent cap is applied
        # immediately below (σ is already known here). Flag off (default) → the
        # fixed cap applies and this path is bit-identical to v3.1.
        tau_s = tau_yr * _ANNUAL_SECONDS
        tte_upper = (
            float(self.cfg.vol_scaled_tte_ceiling_seconds)
            if self.cfg.vol_scaled_tte_enabled
            else float(self.cfg.tte_max_seconds)
        )
        if not (self.cfg.tte_min_seconds <= tau_s <= tte_upper):
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "tte_out_of_window", (("tte_s", f"{tau_s:.0f}"),)),
            ))
        if self.cfg.vol_scaled_tte_enabled:
            sigma_eff = max(sigma, 1e-9)
            tte_max_eff = min(
                self.cfg.tte_max_seconds * (
                    self.cfg.vol_scaled_tte_ref_sigma / sigma_eff
                ) ** self.cfg.vol_scaled_tte_exponent,
                float(self.cfg.vol_scaled_tte_ceiling_seconds),
            )
            if tau_s > tte_max_eff:
                return Decision(action=Action.HOLD, diagnostics=(
                    Diagnostic("info", "vol_scaled_tte_exceeded", (
                        ("tte_s", f"{tau_s:.0f}"),
                        ("tte_max_eff", f"{tte_max_eff:.0f}"),
                        ("sigma", f"{sigma:.5f}"),
                    )),
                ))

        # Near-strike hover veto. PM corpus shows entries below 0.20% lose
        # -$7.68/entry on average across 57 entries, while the 0.20-0.50%
        # band immediately above is the *best* band (+$16.10/entry). The
        # sharp discontinuity is consistent: at this distance the implied
        # probability is too close to 50/50 for the model's edge claim to
        # survive a coin-flip move. We compute the gate from priceBinary's
        # strike only — buckets have multiple boundaries and the "near a
        # bucket boundary" check would need a different formulation.
        if (
            self.cfg.min_distance_pct is not None
            and self.cfg.min_distance_pct > 0.0
            and question.klass == "priceBinary"
            and reference_price > 0
            and question.strike > 0
        ):
            dist_pct = abs(reference_price - question.strike) / reference_price
            if dist_pct < self.cfg.min_distance_pct:
                return Decision(action=Action.HOLD, diagnostics=(
                    Diagnostic("info", "near_strike_hover",
                               (("dist_pct", f"{dist_pct:.5f}"),
                                ("min_dist_pct", f"{self.cfg.min_distance_pct:.5f}"))),
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
        # tuples are (symbol, p_win, raw_edge, book, phi_d) — phi_d enters the
        # entry gate via the gamma-aware effective edge below; raw_edge stays
        # in the diagnostic so analytics see the unmodified GBM edge.
        per_leg: list[tuple[str, float, float, BookState, float]] = []
        for sym in legs:
            book = books.get(sym)
            if book is None or book.ask_px is None:
                continue
            lo, hi = _winning_region(question, sym)
            pp = _p_leg_win_prob_and_phi(
                reference_price=reference_price, lo=lo, hi=hi,
                sigma=sigma, mu_eff=mu_eff, tau_yr=tau_yr,
            )
            if pp is None:
                continue  # NO leg of a middle bucket — no contiguous winning region
            p_win, phi_d = pp
            fee = fee_per_share(self.cfg, p_win, side="entry")
            edge = p_win - book.ask_px - fee - self.cfg.half_spread_assumption
            per_leg.append((sym, p_win, edge, book, phi_d))

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

        # Bid-notional sanity gate. A leg with bid_px × bid_sz below the
        # configured floor probably has only spoof / penny-stake buying
        # interest — passing the mid threshold doesn't mean the price is real.
        # 0 disables.
        if self.cfg.min_bid_notional_usd > 0.0:
            def _bid_ntl(b: BookState) -> float:
                return (b.bid_px or 0.0) * (b.bid_sz or 0.0)
            per_leg = [t for t in per_leg if _bid_ntl(t[3]) >= self.cfg.min_bid_notional_usd]
            if not per_leg:
                return Decision(action=Action.HOLD, diagnostics=(
                    Diagnostic("info", "bid_notional_too_thin",
                               (("min_usd", f"{self.cfg.min_bid_notional_usd:.2f}"),)),
                ))

        # Pick the leg with the highest GAMMA-AWARE effective edge. For λ=None
        # (legacy) this collapses to picking on raw edge. For λ>0 we subtract
        # the per-leg path-variance penalty λ·φ(d) so a near-strike leg with
        # the same raw edge as a far-from-strike leg is correctly de-prioritised.
        gamma_lambda = self.cfg.gamma_lambda or 0.0
        chosen_sym, chosen_p, chosen_edge, chosen_book, chosen_phi = max(
            per_leg, key=lambda t: t[2] - gamma_lambda * t[4]
        )
        effective_edge = chosen_edge - gamma_lambda * chosen_phi

        # v3.5: momentum / MR gate — skip if regime == "mr" and aligned-signed
        # score < -tau_gate. Computed AFTER favorite is chosen so we know which
        # side to align to.
        if (
            self.cfg.momentum_mr_enabled
            and self.cfg.momentum_mr_mode == "gate"
        ):
            from hlanalysis.strategy.momentum_mr import momentum_mr_score
            fav_side = +1 if chosen_sym == question.yes_symbol else -1
            mm_score, mm_regime = momentum_mr_score(
                recent_returns=recent_returns,
                lookback_min=self.cfg.momentum_mr_lookback_min,
                indicator=self.cfg.momentum_mr_indicator,
                favorite_side=fav_side,
            )
            if self.cfg.momentum_mr_jr_trust_weight:
                trust = _jr_trust_weight(recent_returns, self.cfg.momentum_mr_lookback_min)
                mm_score = mm_score * trust
            else:
                trust = 1.0
            gate_diag_kv: list = [
                ("indicator", self.cfg.momentum_mr_indicator),
                ("score", f"{mm_score:.3f}"),
                ("regime", mm_regime),
                ("tau_gate", f"{self.cfg.momentum_mr_tau_gate:.3f}"),
                ("fav_side", str(fav_side)),
            ]
            if self.cfg.momentum_mr_jr_trust_weight:
                gate_diag_kv.append(("jr_trust", f"{trust:.3f}"))
            if mm_regime == "mr" and mm_score < -self.cfg.momentum_mr_tau_gate:
                return Decision(action=Action.HOLD, diagnostics=(
                    Diagnostic("info", "momentum_mr_gate", tuple(gate_diag_kv)),
                ))

        # Build the edge diagnostic. Binary questions keep the p_model /
        # edge_yes / edge_no schema (meaningful per-side); multi-leg buckets
        # emit native chosen_leg / chosen_edge fields instead (see the bucket
        # branch below).
        if is_binary:
            yes = books.get(question.yes_symbol)
            no_ = books.get(question.no_symbol)
            # Binary always has both legs at this point because per_leg is non-empty
            # (favorite gate already passed if active). p_yes = P(S>strike).
            p_yes_view = _p_leg_win_prob(
                reference_price=reference_price, lo=question.strike, hi=None,
                sigma=sigma, mu_eff=mu_eff, tau_yr=tau_yr,
            ) or 0.0
            fee_yes = fee_per_share(self.cfg, p_yes_view, side="entry")
            fee_no = fee_per_share(self.cfg, 1.0 - p_yes_view, side="entry")
            edge_yes = (
                p_yes_view - (yes.ask_px if yes and yes.ask_px is not None else 1.0)
                - fee_yes - self.cfg.half_spread_assumption
            )
            edge_no = (
                (1.0 - p_yes_view) - (no_.ask_px if no_ and no_.ask_px is not None else 1.0)
                - fee_no - self.cfg.half_spread_assumption
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
            # Bucket (multi-leg) diagnostic. Buckets have no binary YES/NO, so we
            # emit NATIVE fields — chosen_leg / chosen_edge — for the chosen leg
            # rather than abusing the binary schema with edge_no=-1e9 sentinels.
            # edge_yes is retained as a back-compat MIRROR of chosen_edge so the
            # fixed-schema diagnostics parquet (backtest/runner) and the entry
            # fill-meta reader keep their populated value without an off-schema
            # column; edge_no is intentionally omitted (no sentinel).
            diag = Diagnostic("info", "edge", (
                ("p_model", f"{chosen_p:.4f}"),
                ("chosen_leg", chosen_sym),
                ("chosen_edge", f"{chosen_edge:.4f}"),
                ("edge_yes", f"{chosen_edge:.4f}"),
                ("sigma", f"{sigma:.4f}"),
                ("tau_yr", f"{tau_yr:.12f}"),
                ("ln_sk", "0.0000"),
            ))

        # v3.5: momentum / MR tilt — scale the effective edge_buffer by
        # (1 - alpha_tilt * score). Aligned momentum (score > 0) lowers the
        # bar; MR against favorite (score < 0) raises it.
        effective_edge_buffer = self.cfg.edge_buffer
        if (
            self.cfg.momentum_mr_enabled
            and self.cfg.momentum_mr_mode == "tilt"
        ):
            from hlanalysis.strategy.momentum_mr import momentum_mr_score
            fav_side = +1 if chosen_sym == question.yes_symbol else -1
            mm_score, mm_regime = momentum_mr_score(
                recent_returns=recent_returns,
                lookback_min=self.cfg.momentum_mr_lookback_min,
                indicator=self.cfg.momentum_mr_indicator,
                favorite_side=fav_side,
            )
            if self.cfg.momentum_mr_jr_trust_weight:
                trust = _jr_trust_weight(recent_returns, self.cfg.momentum_mr_lookback_min)
                mm_score = mm_score * trust
            else:
                trust = 1.0
            effective_edge_buffer = self.cfg.edge_buffer * (
                1.0 - self.cfg.momentum_mr_alpha_tilt * mm_score
            )
            # Append a single tilt diagnostic alongside `diag` below.
            tilt_diag_kv: list = [
                ("indicator", self.cfg.momentum_mr_indicator),
                ("score", f"{mm_score:.3f}"),
                ("regime", mm_regime),
                ("eff_edge_buffer", f"{effective_edge_buffer:.5f}"),
                ("base_edge_buffer", f"{self.cfg.edge_buffer:.5f}"),
                ("fav_side", str(fav_side)),
            ]
            if self.cfg.momentum_mr_jr_trust_weight:
                tilt_diag_kv.append(("jr_trust", f"{trust:.3f}"))
            tilt_diag = Diagnostic("info", "momentum_mr_tilt", tuple(tilt_diag_kv))
        else:
            tilt_diag = None

        if effective_edge <= effective_edge_buffer:
            diags: tuple = (diag,)
            if tilt_diag is not None:
                diags = (tilt_diag,) + diags
            if gamma_lambda > 0.0 and chosen_edge > effective_edge_buffer:
                diags = (Diagnostic("info", "edge_after_gamma_below_buffer", (
                    ("raw_edge", f"{chosen_edge:.4f}"),
                    ("phi_d", f"{chosen_phi:.4f}"),
                    ("gamma_penalty", f"{gamma_lambda * chosen_phi:.4f}"),
                )),) + diags
            return Decision(action=Action.HOLD, diagnostics=diags)

        if self.cfg.edge_max is not None and chosen_edge >= self.cfg.edge_max:
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "edge_too_extreme", (
                    ("edge", f"{chosen_edge:.4f}"),
                    ("edge_max", f"{self.cfg.edge_max:.4f}"),
                )),
                diag,
            ))

        # v3.4-LMgate: Lee-Mykland (2008) post-edge jump filter — TRULY τ-free.
        # Stat = |r_last| / √(BV_per_sample), comparing the latest log-return
        # to the jump-robust per-sample σ over the same window. > threshold
        # ⇒ jump that the market may not have repriced yet. None disables.
        if self.cfg.lm_threshold is not None and len(recent_returns) > 0:
            n_keep = max(
                2, self.cfg.vol_lookback_seconds // self.cfg.vol_sampling_dt_seconds
            )
            window = recent_returns[-n_keep:]
            if len(window) < 2:
                return Decision(action=Action.HOLD, diagnostics=(
                    Diagnostic("info", "lm_no_returns"), diag,
                ))
            arr = np.asarray(window, dtype=np.float64)
            bv_per_sample = float(bipower_variation_sigma(arr))
            if bv_per_sample <= 0.0:
                return Decision(action=Action.HOLD, diagnostics=(
                    Diagnostic("info", "lm_bv_zero"), diag,
                ))
            lm_stat = abs(float(arr[-1])) / bv_per_sample
            if lm_stat < self.cfg.lm_threshold:
                return Decision(action=Action.HOLD, diagnostics=(
                    Diagnostic("info", "lm_gate_no_jump", (
                        ("lm_stat", f"{lm_stat:.3f}"),
                        ("lm_threshold", f"{self.cfg.lm_threshold:.3f}"),
                    )),
                    diag,
                ))

        size = max(0.0, round_size(self.cfg.max_position_usd, chosen_book.ask_px))
        if size <= 0:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("warn", "size_zero"), diag))

        intent = make_entry_intent(
            question, symbol=chosen_sym, size=size, limit_price=chosen_book.ask_px,
        )
        diags_out: tuple = (Diagnostic("info", "entry"), diag)
        if tilt_diag is not None:
            diags_out = (tilt_diag,) + diags_out
        return Decision(
            action=Action.ENTER,
            intents=(intent,),
            diagnostics=diags_out,
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
        # p_model / (1-p_model) split. We also pull φ(d_nearest) so the gamma
        # penalty applies symmetrically here: held positions with high path
        # variance are cut sooner than the raw-edge gate would suggest.
        lo, hi = _winning_region(question, position.symbol)

        # Rule 2.5 (v3.1.1+): σ-normalized mid-hold distance exit. Fires BEFORE
        # the bid collapses, catching cases where BTC has drifted close to the
        # adverse boundary while the held leg's bid is still stale-positive.
        # Mirrors v1's exit_safety_d but uses v3.1's Itô-corrected d-machinery
        # (drift-aware). Skipped when σ·√τ is non-positive or the leg has no
        # contiguous winning region (middle-bucket NO).
        if self.cfg.exit_safety_d > 0.0:
            safety_d = _safety_d_for_region(
                reference_price=reference_price, lo=lo, hi=hi,
                sigma=sigma, mu_eff=mu_eff, tau_yr=tau_yr,
            )
            if safety_d is not None and safety_d < self.cfg.exit_safety_d:
                intent = make_exit_intent(
                    question, position, limit_price=held.bid_px,
                    exit_reason="exit_safety_d",
                )
                return Decision(
                    action=Action.EXIT,
                    intents=(intent,),
                    diagnostics=(
                        Diagnostic("info", "exit_safety_d", (
                            ("exit_reason", "safety_d_below_threshold"),
                            ("exit_safety_d", f"{safety_d:.4f}"),
                            ("exit_threshold", f"{self.cfg.exit_safety_d:.4f}"),
                        )),
                    ),
                )

        pp = _p_leg_win_prob_and_phi(
            reference_price=reference_price, lo=lo, hi=hi,
            sigma=sigma, mu_eff=mu_eff, tau_yr=tau_yr,
        )
        if pp is None:
            # Middle-bucket NO with no contiguous winning region. Skip edge
            # check; rely on stop_loss / time_stop / settlement.
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "hold_no_region"),
            ))
        held_p, phi_d_held = pp
        # Exit-edge MUST reference the price we'd actually fill at — the BID —
        # not the ask of the held leg. The legacy ASK-based formulation has the
        # right semantic ("is the alpha gone") in symmetric markets but in HL
        # HIP-4 the ask spikes independently of the bid (stale top-of-book
        # quotes during settlement-adjacent drift). Each transient ask spike
        # crosses the exit threshold even though the real liquidation price
        # (bid) hasn't moved → exit at bid → ask normalises → re-enter →
        # repeat. This is the v3.1 analogue of the v1 stale-ask churn that
        # cost ~$120 on q#601 today. Using the bid: exit fires only when the
        # bid genuinely climbs above held_p (clean take-profit). v3.1 has
        # never had a stop-loss-style cut here (stop_loss_pct=null by design);
        # legacy callers wanting one should re-introduce it as a separate
        # rule, not piggy-back on edge_held with the wrong price.
        # Two formulations available; see ThetaHarvesterConfig.exit_take_profit_mode
        # for the rationale and how to A/B-test them.
        #   - legacy (mode=False): edge_held = held_p − bid − fee_taker;
        #     exit when this drops below exit_edge_threshold. Frames the
        #     question as "is the alpha gone if I hypothetically rebought at
        #     the bid?" — fires on bid noise close to fair value.
        #   - take-profit (mode=True): edge_held = bid − held_p − exit_fee;
        #     exit when this rises above exit_edge_threshold. Frames the
        #     question as "is the bid offering me an above-fair sell price
        #     net of the exit-side fee?" — only fires on genuine premium.
        # Exit-side fee per share. PM charges the curve fee on BOTH taker
        # sides (entry and exit) — under fee_model="pm_binary" we estimate
        # it at the model probability `held_p`, consistent with the entry
        # edge formulation. Under fee_model="flat" (HL slots, legacy):
        # take-profit mode uses the fixed `exit_fee`; legacy mode uses
        # `fee_taker` (the legacy formula models the gate as "if I
        # hypothetically rebought at the bid, what'd I pay?", so an entry
        # fee is the right comparable).
        exit_fee_per_share = fee_per_share(self.cfg, held_p, side="exit")
        if self.cfg.exit_take_profit_mode:
            edge_held = held.bid_px - held_p - exit_fee_per_share
            exit_now = edge_held > self.cfg.exit_edge_threshold
        else:
            edge_held = held_p - held.bid_px - exit_fee_per_share
            exit_now = edge_held < self.cfg.exit_edge_threshold
        # Gamma penalty is INTENTIONALLY NOT applied to the exit gate.
        # Empirical: symmetric gamma penalty triggers premature exits during
        # transient drift through near-strike regions that would otherwise
        # recover (HL 9-question test 2026-05-19, net -$3 to -$10/question).
        if exit_now:
            return self._exit_intent(question, position, held, reason="exit_edge")

        return Decision(action=Action.HOLD, diagnostics=(
            Diagnostic("info", "hold", (
                ("edge_held", f"{edge_held:.4f}"),
                ("held_p", f"{held_p:.4f}"),
                ("tau_s", f"{tau_s:.0f}"),
            )),
        ))

    def _evaluate_topup(
        self, *, question: QuestionView, books: Mapping[str, BookState],
        reference_price: float, sigma: float, mu_eff: float, tau_yr: float,
        position: Position,
        recent_returns: tuple[float, ...] = (),
    ) -> Decision:
        """Top up an under-filled held position.

        Returns ENTER when the held leg is still the favourite under current
        state AND the shortfall vs max_position_usd is large enough to justify
        an order ≥ topup_min_notional_usd. Otherwise returns HOLD with a
        skip-reason diagnostic. The router's post-exit cooldown is bypassed
        naturally — `_last_exit_ts` is 0 while a position is open.

        Shared body lives in ``strategy/topup.py``; theta supplies its entry
        evaluator and the no_book / not_needed branches (which, unlike v1,
        return HOLD ``topup_skip`` diagnostics rather than ``None``).
        """
        def _on_no_book() -> Decision:
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "topup_skip", (("reason", "no_book"),)),
            ))

        def _on_not_needed(current_ntl: float, target_ntl: float) -> Decision:
            # Demoted from info → debug 2026-05-22: fires every scan tick (~1/s)
            # while a position is held and the topup gate stays "not_needed",
            # drowning genuinely interesting events out of journalctl. The
            # current/target notionals are still attached to the Diagnostic
            # below, which feeds the scanner's gate-transition log
            # (gate_decisions.jsonl) and renders in the bus-event journal log.
            logger.debug(
                "topup_skip q={} sym={} reason=not_needed current_ntl=${:.2f} target_ntl=${:.2f}",
                question.question_idx, position.symbol, current_ntl, target_ntl,
            )
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "topup_skip", (
                    ("reason", "not_needed"),
                    ("current_ntl", f"{current_ntl:.2f}"),
                    ("target_ntl", f"{target_ntl:.2f}"),
                )),
            ))

        return run_topup(
            question=question, books=books, position=position,
            max_position_usd=self.cfg.max_position_usd,
            topup_threshold_pct=self.cfg.topup_threshold_pct,
            topup_min_notional_usd=self.cfg.topup_min_notional_usd,
            run_entry=lambda: self._evaluate_entry(
                question=question, books=books, reference_price=reference_price,
                sigma=sigma, mu_eff=mu_eff, tau_yr=tau_yr,
                recent_returns=recent_returns,
            ),
            on_no_book=_on_no_book,
            on_not_needed=_on_not_needed,
        )

    def _exit_intent(self, question: QuestionView, position: Position, held: BookState, *, reason: str) -> Decision:
        intent = make_exit_intent(
            question, position, limit_price=held.bid_px, exit_reason=reason,
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
        min_distance_pct=(
            float(params["min_distance_pct"])
            if params.get("min_distance_pct") is not None
            else None
        ),
        min_bid_notional_usd=float(params.get("min_bid_notional_usd", 0.0)),
        gamma_lambda=(
            float(params["gamma_lambda"])
            if params.get("gamma_lambda") is not None
            else None
        ),
        topup_enabled=bool(params.get("topup_enabled", True)),
        topup_threshold_pct=float(params.get("topup_threshold_pct", 0.2)),
        topup_min_notional_usd=float(params.get("topup_min_notional_usd", 11.0)),
        exit_safety_d=float(params.get("exit_safety_d", 0.0)),
        vol_estimator=str(params.get("vol_estimator", "sample_std")),
        lm_threshold=(
            float(params["lm_threshold"])
            if params.get("lm_threshold") is not None else None
        ),
        exit_take_profit_mode=bool(params.get("exit_take_profit_mode", False)),
        exit_fee=float(params.get("exit_fee", 0.0007)),
        fee_model=str(params.get("fee_model", "flat")),
        fee_rate=float(params.get("fee_rate", 0.0)),
        momentum_mr_enabled=bool(params.get("momentum_mr_enabled", False)),
        momentum_mr_indicator=str(params.get("momentum_mr_indicator", "z_ret")),
        momentum_mr_lookback_min=int(params.get("momentum_mr_lookback_min", 15)),
        momentum_mr_mode=str(params.get("momentum_mr_mode", "gate")),
        momentum_mr_tau_gate=float(params.get("momentum_mr_tau_gate", 1.0)),
        momentum_mr_alpha_tilt=float(params.get("momentum_mr_alpha_tilt", 0.5)),
        momentum_mr_jr_trust_weight=bool(params.get("momentum_mr_jr_trust_weight", False)),
        vol_scaled_tte_enabled=bool(params.get("vol_scaled_tte_enabled", False)),
        vol_scaled_tte_ref_sigma=float(params.get("vol_scaled_tte_ref_sigma", 0.0)),
        vol_scaled_tte_exponent=float(params.get("vol_scaled_tte_exponent", 1.0)),
        vol_scaled_tte_ceiling_seconds=int(params.get("vol_scaled_tte_ceiling_seconds", 0)),
    )
    return ThetaHarvesterStrategy(cfg)


@register("v3_2_volclock")
def build_v3_2_volclock(params: dict) -> ThetaHarvesterStrategy:
    """v3.2-volclock — v3.1 with the σ estimator swapped from rolling sample
    stdev to bipower variation (jump-robust). Identical formula otherwise.

    Goal: keep p_model regime-aware without letting individual wicks inflate σ
    and shrink the d-statistic. After a wick, σ_BV stays calm → d stays large
    → the strategy enters on the wick-driven mispricing instead of holding.

    All params flow through to ``build_v3_theta_harvester``; the only
    difference is the default ``vol_estimator`` flips to "bipower" (callers
    can still override).
    """
    params_with_default = dict(params)
    params_with_default.setdefault("vol_estimator", "bipower")
    return build_v3_theta_harvester(params_with_default)


@register("v3_4_lmgate")
def build_v3_4_lmgate(params: dict) -> ThetaHarvesterStrategy:
    """v3.4-LMgate — v3.2-volclock plus the Lee-Mykland jump gate.

    Adds a τ-free post-edge confirmation: |r_last|/√BV > k_jump (default 4.0).
    Restricts entries to moments when the most recent return looks like a
    jump under the no-jump null — i.e., the market hasn't yet repriced the
    move. Should be higher-precision-lower-recall than v3.2.
    """
    params_with_default = dict(params)
    params_with_default.setdefault("vol_estimator", "bipower")
    params_with_default.setdefault("lm_threshold", 4.0)
    return build_v3_theta_harvester(params_with_default)


@register("v3_5_momentum_mr")
def build_v3_5_momentum_mr(params: dict) -> ThetaHarvesterStrategy:
    """v3.5 — v3.1 final + momentum/MR gate or tilt on favorite-side entries.

    Defaults to v3.1 final state plus momentum_mr_enabled=True. Sweep params
    expose `momentum_mr_indicator`, `momentum_mr_lookback_min`,
    `momentum_mr_mode`, `momentum_mr_tau_gate`, `momentum_mr_alpha_tilt`.
    """
    params_with_default = dict(params)
    params_with_default.setdefault("momentum_mr_enabled", True)
    return build_v3_theta_harvester(params_with_default)
