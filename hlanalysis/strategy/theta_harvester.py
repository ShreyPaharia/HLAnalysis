from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger
from scipy.stats import norm  # type: ignore[import-untyped]

from ._numba.vol import bipower_variation_sigma as _nb_bipower
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
        if self.cfg.vol_estimator == "bipower":
            raw = float(_nb_bipower(arr))
        elif self.cfg.vol_estimator == "sample_std":
            raw = float(_nb_sample_std(arr))
        else:
            raise ValueError(f"Unknown vol_estimator: {self.cfg.vol_estimator!r}")
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
        self, *, question: QuestionView, books: Mapping[str, BookState],
        reference_price: float, sigma: float, mu_eff: float, tau_yr: float,
        recent_returns: tuple[float, ...] = (),
    ) -> Decision:
        # TTE entry window
        tau_s = tau_yr * _ANNUAL_SECONDS
        if not (self.cfg.tte_min_seconds <= tau_s <= self.cfg.tte_max_seconds):
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "tte_out_of_window", (("tte_s", f"{tau_s:.0f}"),)),
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
            edge = p_win - book.ask_px - self.cfg.fee_taker - self.cfg.half_spread_assumption
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

        if effective_edge <= self.cfg.edge_buffer:
            # Diagnose whether gamma penalty was the deciding factor — helps
            # post-hoc tuning of gamma_lambda.
            if gamma_lambda > 0.0 and chosen_edge > self.cfg.edge_buffer:
                return Decision(action=Action.HOLD, diagnostics=(
                    Diagnostic("info", "edge_after_gamma_below_buffer", (
                        ("raw_edge", f"{chosen_edge:.4f}"),
                        ("phi_d", f"{chosen_phi:.4f}"),
                        ("gamma_penalty", f"{gamma_lambda * chosen_phi:.4f}"),
                    )),
                    diag,
                ))
            return Decision(action=Action.HOLD, diagnostics=(diag,))

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
            bv_per_sample = float(_nb_bipower(arr))
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
                intent = OrderIntent(
                    question_idx=question.question_idx,
                    symbol=position.symbol,
                    side="sell" if position.qty > 0 else "buy",
                    size=abs(position.qty),
                    limit_price=held.bid_px,
                    cloid=f"hla-{uuid.uuid4()}",
                    time_in_force="ioc",
                    reduce_only=True,
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
        edge_held = held_p - held.bid_px - self.cfg.fee_taker
        # Gamma penalty is INTENTIONALLY NOT applied to the exit gate.
        # Empirical: symmetric gamma penalty triggers premature exits during
        # transient drift through near-strike regions that would otherwise
        # recover (HL 9-question test 2026-05-19, net -$3 to -$10/question).
        if edge_held < self.cfg.exit_edge_threshold:
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
        """
        held = books.get(position.symbol)
        if held is None or held.ask_px is None or held.ask_px <= 0:
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "topup_skip", (("reason", "no_book"),)),
            ))
        ask = held.ask_px
        current_ntl = abs(position.qty) * ask
        target_ntl = self.cfg.max_position_usd
        shortfall_ntl = target_ntl - current_ntl
        if shortfall_ntl < target_ntl * self.cfg.topup_threshold_pct:
            logger.info(
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

        # Re-run ALL entry gates against the current state. _evaluate_entry
        # already encodes every gate (TTE, near-strike, favorite, edge,
        # edge_max, bid notional, etc.) so we delegate rather than duplicate.
        entry_dec = self._evaluate_entry(
            question=question, books=books, reference_price=reference_price,
            sigma=sigma, mu_eff=mu_eff, tau_yr=tau_yr,
            recent_returns=recent_returns,
        )
        if entry_dec.action != Action.ENTER or not entry_dec.intents:
            failed_gate = (
                entry_dec.diagnostics[0].message
                if entry_dec.diagnostics else "unknown"
            )
            logger.info(
                "topup_skip q={} sym={} reason=gate_failed:{} current_ntl=${:.2f} target_ntl=${:.2f}",
                question.question_idx, position.symbol, failed_gate,
                current_ntl, target_ntl,
            )
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "topup_skip", (
                    ("reason", f"gate_failed:{failed_gate}"),
                )),
            ))
        candidate = entry_dec.intents[0]
        if candidate.symbol != position.symbol:
            logger.info(
                "topup_skip q={} sym={} reason=leg_changed chosen={} current_ntl=${:.2f} target_ntl=${:.2f}",
                question.question_idx, position.symbol, candidate.symbol,
                current_ntl, target_ntl,
            )
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "topup_skip", (
                    ("reason", "leg_changed"),
                    ("chosen", candidate.symbol),
                )),
            ))

        topup_size = math.floor((shortfall_ntl / ask) * 100) / 100
        topup_ntl = topup_size * ask
        if topup_ntl < self.cfg.topup_min_notional_usd:
            logger.info(
                "topup_skip q={} sym={} reason=below_min_notional topup_ntl=${:.2f} min=${:.2f}",
                question.question_idx, position.symbol, topup_ntl,
                self.cfg.topup_min_notional_usd,
            )
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "topup_skip", (
                    ("reason", "below_min_notional"),
                    ("topup_ntl", f"{topup_ntl:.2f}"),
                )),
            ))

        intent = OrderIntent(
            question_idx=question.question_idx,
            symbol=position.symbol,
            side="buy",
            size=topup_size,
            limit_price=ask,
            cloid=f"hla-{uuid.uuid4()}",
            time_in_force="ioc",
        )
        logger.info(
            "topup_emit q={} sym={} side=buy current_ntl=${:.2f} target_ntl=${:.2f} "
            "topup_size={:.2f} ask={:.5f}",
            question.question_idx, position.symbol, current_ntl, target_ntl,
            topup_size, ask,
        )
        return Decision(
            action=Action.ENTER, intents=(intent,),
            diagnostics=(
                Diagnostic("info", "topup_emit", (
                    ("current_ntl", f"{current_ntl:.2f}"),
                    ("target_ntl", f"{target_ntl:.2f}"),
                    ("topup_size", f"{topup_size:.2f}"),
                    ("ask", f"{ask:.5f}"),
                )),
            ),
        )

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
            exit_reason=reason,
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
