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


def _kv_get(qv: QuestionView, key: str) -> str:
    for k, v in qv.kv:
        if k == key:
            return v
    return ""


def _winning_region(
    qv: QuestionView, symbol: str
) -> tuple[float | None, float | None]:
    """Return (lo, hi) such that the leg `symbol` wins iff BTC ∈ [lo, hi] at
    expiry. ``None`` denotes an unbounded side (-∞ for lo, +∞ for hi).

    Binary YES wins above strike → (strike, None); NO wins at-or-below → (None, strike).
    Bucket layout (HL convention, N thresholds yield N+1 outcomes, 2 legs each;
    YES at even leg index, NO at odd):
      outcome 0       (lowest)  → (None, thr[0])
      outcome 1..N-1  (middle)  → (thr[i-1], thr[i])
      outcome N       (highest) → (thr[-1], None)
    NO legs invert the corresponding YES region (winning region splits into two
    half-lines for middle buckets — collapse to whichever half-line a strict
    inversion would yield; this helper instead returns the YES bucket bounds and
    callers compute safety_d against the NEAREST adverse boundary, which is
    correct for both YES and the half-line NO cases).
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

    # YES region for this outcome bucket.
    if outcome_pos == 0:
        yes_lo: float | None = None
        yes_hi: float | None = thr[0]
    elif outcome_pos == len(thr):
        yes_lo, yes_hi = thr[-1], None
    elif 0 < outcome_pos < len(thr):
        yes_lo, yes_hi = thr[outcome_pos - 1], thr[outcome_pos]
    else:
        return (None, None)

    if side_idx == 0:
        return (yes_lo, yes_hi)

    # NO of a single-sided bucket inverts to the opposite half-line.
    if yes_lo is None:
        return (yes_hi, None)
    if yes_hi is None:
        return (None, yes_lo)
    # NO of a middle bucket is the union of two half-lines; not a contiguous
    # winning region. Callers treat (None, None) as "no leg-level gate" and
    # fall back to non-safety exits (stop-loss, settlement). Buying NO of a
    # middle bucket is disallowed anyway by the YES-only entry path.
    return (None, None)


def _safety_d_for_region(
    *,
    ref_price: float,
    lo: float | None,
    hi: float | None,
    sigma_window: float,
    mu: float,
    tte_min: float,
    drift_aware: bool,
) -> float | None:
    """Signed safety distance (in σ-window units) from ``ref_price`` to the
    nearer adverse boundary of the leg's winning region.

    Returns ``None`` when neither boundary is known (e.g. NO leg of a middle
    bucket). Positive values mean BTC is safely inside the winning region;
    negative means already on the losing side.

    Drift adjustment shifts the distance by μ·τ in the adverse direction
    (positive μ raises safety for lower-bounded legs, lowers it for
    upper-bounded). Two-sided (middle bucket) drops drift because there is
    no single adverse direction.
    """
    if sigma_window <= 0:
        return None
    if lo is not None and hi is not None:
        d_unscaled = min(math.log(ref_price / lo), math.log(hi / ref_price))
        # drift dropped for two-sided regions.
        return d_unscaled / sigma_window
    if lo is not None and hi is None:
        d_unscaled = math.log(ref_price / lo)
        if drift_aware:
            d_unscaled += mu * tte_min
        return d_unscaled / sigma_window
    if lo is None and hi is not None:
        d_unscaled = math.log(hi / ref_price)
        if drift_aware:
            d_unscaled -= mu * tte_min
        return d_unscaled / sigma_window
    return None


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
    # Mid-hold exit threshold. Re-evaluates safety_d every tick while position is
    # open; exits IOC at bid when safety_d drops below this. 0 = disabled. Should
    # be lower than min_safety_d to avoid thrashing on the entry boundary.
    exit_safety_d: float = 0.0
    # Hard price-level stop: exit IOC if held leg's bid drops to/below this.
    # Catches sudden flips that safety_d misses because kline cadence is too slow
    # for sub-minute moves. 0 = disabled.
    exit_bid_floor: float = 0.0
    # When True, safety_d uses drift-corrected distance:
    #   d = sign(ln S/K) * (ln S/K + μ * τ) / (σ * √τ)
    # where μ = mean of returns_window. Captures the case where BTC is trending
    # toward the strike (drift adverse to the favorite). Default False preserves
    # the symmetric |ln S/K| formulation.
    drift_aware_d: bool = False
    # EWMA decay for the σ estimator used in the safety_d gates (both entry and
    # exit). σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t, seeded with σ²_0 = r²_0. Higher λ =
    # smoother (more weight on history). 0.0 disables EWMA and reverts to the
    # sample std (ddof=1) over returns_window — backward compatible.
    vol_ewma_lambda: float = 0.0
    # Auxiliary fast σ exit. Re-evaluates safety_d using a SHORTER vol window
    # (default 5min) to catch sudden vol spikes that the slower σ_1h misses
    # mid-hold. Layered ON TOP of exit_safety_d (σ_1h) — both gates active.
    # 0 = disabled. Should be ≤ exit_safety_d to trip earlier on fast spikes.
    exit_safety_d_5m: float = 0.0
    # Lookback for the fast σ above. Runner provides 1m bars; we slice to last
    # N = exit_vol_lookback_5m_seconds // 60 samples. Default 300s (5min, 5 bars).
    exit_vol_lookback_5m_seconds: int = 300
    # σ estimator selector for the safety_d (entry+exit) and vol_max gates.
    #   "stdev"     → sample std (ddof=1) of close-to-close log returns (legacy).
    #   "parkinson" → range-based estimator using bar (H, L):
    #       σ²_bar = (ln(H/L))² / (4 ln 2);  σ²_window = mean(σ²_bar).
    # Parkinson is ~5× more sample-efficient than close-to-close std on the same
    # bar cadence. When selected, vol_ewma_lambda still applies (EWMA on per-bar
    # σ²) and vol_max compares against the Parkinson σ (per-bar units, same
    # scale as 1m close-to-close std). Both entry and exit safety_d gates use
    # the same estimator.
    vol_estimator: str = "stdev"
    # Position-size scaling by safety_d. Applied AFTER the entry gate passes
    # (safety_d already ≥ min_safety_d); only modulates the size, never the
    # gate decision. Maps safety_d ∈ [min_safety_d, 3.0+] → scale ∈
    # [size_min_fraction, 1.0]. Final size = floor(max_position_usd * scale / ask).
    #   "fixed"          → no scaling (legacy; scale = 1.0).
    #   "linear_safety"  → scale = (d − min_safety_d) / (3.0 − min_safety_d).
    #   "sqrt_safety"    → scale = sqrt of the above (less aggressive ramp).
    # Both clamped to [size_min_fraction, 1.0]. size_min_fraction floors the
    # smallest bet so we don't trade ~0 size on borderline-safe entries.
    size_scaling: str = "fixed"
    size_min_fraction: float = 0.25


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

    # 1 / (4 ln 2): Parkinson's normalisation constant for (ln H/L)² → variance.
    _PARK_K = 1.0 / (4.0 * math.log(2.0))

    @staticmethod
    def _ewma_std(returns: tuple[float, ...] | np.ndarray, lam: float) -> float:
        """Recursive EWMA stdev of returns. σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t,
        seeded σ²_0 = r²_0. Caller must ensure len(returns) >= 1."""
        var = float(returns[0]) ** 2
        for r in returns[1:]:
            rf = float(r)
            var = lam * var + (1.0 - lam) * rf * rf
        return math.sqrt(var)

    @classmethod
    def _parkinson_per_bar_var(
        cls, hl_bars: tuple[tuple[float, float], ...] | np.ndarray
    ) -> list[float]:
        """Per-bar Parkinson variance estimates: σ²_i = (ln(H_i/L_i))² / (4 ln 2).
        Returns an empty list if no bars have H > L > 0."""
        out: list[float] = []
        for h, l in hl_bars:
            hf = float(h)
            lf = float(l)
            if hf > 0 and lf > 0 and hf >= lf:
                ln_hl = math.log(hf / lf) if lf > 0 else 0.0
                out.append(cls._PARK_K * ln_hl * ln_hl)
        return out

    def _sigma_stdev(
        self, returns_window: tuple[float, ...] | np.ndarray
    ) -> float:
        if self.cfg.vol_ewma_lambda > 0.0:
            return self._ewma_std(returns_window, self.cfg.vol_ewma_lambda)
        return float(np.std(returns_window, ddof=1))

    def _sigma_parkinson(
        self, hl_window: tuple[tuple[float, float], ...] | np.ndarray
    ) -> float:
        """Window-level Parkinson σ. With EWMA: variance decays by λ across bars
        (same recursion as close-to-close), else arithmetic mean of per-bar σ²."""
        per_bar_var = self._parkinson_per_bar_var(hl_window)
        if not per_bar_var:
            return 0.0
        if self.cfg.vol_ewma_lambda > 0.0:
            var = per_bar_var[0]
            lam = self.cfg.vol_ewma_lambda
            for v in per_bar_var[1:]:
                var = lam * var + (1.0 - lam) * v
            return math.sqrt(max(var, 0.0))
        return math.sqrt(sum(per_bar_var) / len(per_bar_var))

    def _sigma(
        self,
        returns_window: tuple[float, ...] | np.ndarray,
        hl_window: tuple[tuple[float, float], ...] | np.ndarray = (),
    ) -> float:
        """σ estimator dispatched on cfg.vol_estimator. Falls back to stdev if
        the requested estimator's inputs are missing (e.g. Parkinson with no
        HL bars), preserving backward compatibility."""
        if self.cfg.vol_estimator == "parkinson" and len(hl_window) > 0:
            sig = self._sigma_parkinson(hl_window)
            if sig > 0:
                return sig
            # Fall through to stdev if Parkinson degenerates (all H==L).
        return self._sigma_stdev(returns_window)

    def _safety_d_for_leg(
        self,
        *,
        question: QuestionView,
        leg_symbol: str,
        ref_price: float,
        sigma_window: float,
        returns_window: tuple[float, ...] | np.ndarray,
        tte_min: float,
    ) -> float | None:
        """Leg-aware safety_d. Returns None when the leg has no contiguous
        winning region (e.g. NO of a middle bucket), σ_window is 0, or the
        helper inputs are non-positive — callers treat None as "gate skipped".
        """
        lo, hi = _winning_region(question, leg_symbol)
        if lo is None and hi is None:
            return None
        mu = float(np.mean(returns_window)) if len(returns_window) >= 2 else 0.0
        return _safety_d_for_region(
            ref_price=ref_price, lo=lo, hi=hi,
            sigma_window=sigma_window, mu=mu, tte_min=tte_min,
            drift_aware=self.cfg.drift_aware_d,
        )

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
            held_book = books.get(position.symbol)

            # Hard bid-level exit: catches sudden flips faster than safety_d can.
            if (
                self.cfg.exit_bid_floor > 0.0
                and held_book is not None
                and held_book.bid_px is not None
                and held_book.bid_px <= self.cfg.exit_bid_floor
            ):
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
                    diagnostics=(
                        Diagnostic("warn", "exit_bid_below_floor",
                                   (("bid", f"{held_book.bid_px:.4f}"),)),
                    ),
                )

            # Auxiliary fast σ_5m safety_d exit. Same shape as exit_safety_d but
            # uses a SHORTER vol window so sudden spikes (faster than σ_1h reacts)
            # trigger an earlier cut. Layered: this fires first; the σ_1h gate
            # below remains active as the slower confirmation.
            if (
                self.cfg.exit_safety_d_5m > 0.0
                and reference_price > 0
                and held_book is not None
                and held_book.bid_px is not None
                and held_book.bid_px > 0
                and len(recent_returns) >= 2
            ):
                tte_s_now_5m = (question.expiry_ns - now_ns) / 1e9
                if tte_s_now_5m > 0:
                    n_keep_5m = max(2, self.cfg.exit_vol_lookback_5m_seconds // 60)
                    rw_5m = (
                        recent_returns[-n_keep_5m:]
                        if len(recent_returns) > n_keep_5m
                        else recent_returns
                    )
                    hl_5m = (
                        recent_hl_bars[-n_keep_5m:]
                        if len(recent_hl_bars) > n_keep_5m
                        else recent_hl_bars
                    )
                    if len(rw_5m) >= 2:
                        vol_5m = self._sigma(rw_5m, hl_5m)
                        sigma_window_5m = vol_5m * math.sqrt(max(tte_s_now_5m / 60.0, 1.0))
                        safety_d_5m = self._safety_d_for_leg(
                            question=question, leg_symbol=position.symbol,
                            ref_price=reference_price,
                            sigma_window=sigma_window_5m, returns_window=rw_5m,
                            tte_min=tte_s_now_5m / 60.0,
                        )
                        if safety_d_5m is not None and safety_d_5m < self.cfg.exit_safety_d_5m:
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
                                diagnostics=(
                                    Diagnostic("warn", "exit_safety_d_5m_below_min",
                                               (("d", f"{safety_d_5m:.3f}"),)),
                                ),
                            )

            # Mid-hold safety_d exit: re-evaluate the same gate used at entry.
            # Fires before stop-loss so a regime-change flip can be cut at bid
            # rather than waiting for settle. Skipped near settlement (tte_s<=0)
            # because √tte_min would underflow and safety_d explodes.
            if (
                self.cfg.exit_safety_d > 0.0
                and reference_price > 0
                and held_book is not None
                and held_book.bid_px is not None
                and held_book.bid_px > 0
                and len(recent_returns) >= 2
            ):
                tte_s_now = (question.expiry_ns - now_ns) / 1e9
                if tte_s_now > 0:
                    n_keep = max(2, self.cfg.vol_lookback_seconds // 60)
                    rw = recent_returns[-n_keep:] if len(recent_returns) > n_keep else recent_returns
                    hl = (
                        recent_hl_bars[-n_keep:]
                        if len(recent_hl_bars) > n_keep
                        else recent_hl_bars
                    )
                    if len(rw) >= 2:
                        vol_now = self._sigma(rw, hl)
                        sigma_window = vol_now * math.sqrt(max(tte_s_now / 60.0, 1.0))
                        safety_d_now = self._safety_d_for_leg(
                            question=question, leg_symbol=position.symbol,
                            ref_price=reference_price,
                            sigma_window=sigma_window, returns_window=rw,
                            tte_min=tte_s_now / 60.0,
                        )
                        if safety_d_now is not None and safety_d_now < self.cfg.exit_safety_d:
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
                                diagnostics=(
                                    Diagnostic("warn", "exit_safety_d_below_min",
                                               (("d", f"{safety_d_now:.3f}"),)),
                                ),
                            )

            # Soft stop-loss signal. The risk gate is the authoritative enforcer
            # (Plan 1B); we mirror it here so logging/diagnostics line up.
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
        hl_window = (
            recent_hl_bars[-n_keep:] if len(recent_hl_bars) > n_keep else recent_hl_bars
        )
        if len(returns_window) < 2:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "vol_insufficient_data"),))
        vol = self._sigma(returns_window, hl_window)
        if vol > self.cfg.vol_max:
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "vol_above_cap", (("vol", f"{vol:.4f}"),)),
            ))

        # 6b) Joint safety gate: how many σ from the leg's winning region is BTC
        # given remaining time? For binary YES leg the region is (strike, +∞)
        # and d collapses to ln(BTC/strike) / (σ * sqrt(tte_min)) — the legacy
        # binary formula. For priceBucket legs the region is the bucket's
        # (lo, hi); d uses the nearer adverse boundary. None = no leg-level
        # gate available (rare); treat as skip.
        # Side effect: when computed, `safety_d_entry` is retained for size scaling.
        safety_d_entry: float | None = None
        if self.cfg.min_safety_d > 0.0 and reference_price > 0:
            tte_min = max(tte_s / 60.0, 1.0)
            sigma_window = vol * math.sqrt(tte_min)
            safety_d = self._safety_d_for_leg(
                question=question, leg_symbol=win_symbol,
                ref_price=reference_price,
                sigma_window=sigma_window, returns_window=returns_window,
                tte_min=tte_min,
            )
            if safety_d is not None:
                if safety_d < self.cfg.min_safety_d:
                    return Decision(action=Action.HOLD, diagnostics=(
                        Diagnostic("info", "safety_d_below_min",
                                   (("d", f"{safety_d:.3f}"),)),
                    ))
                safety_d_entry = safety_d

        # 7) Recent-volume sanity (avoid dead questions)
        if recent_volume_usd < self.cfg.min_recent_volume_usd:
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "low_volume", (("vol_usd", f"{recent_volume_usd:.0f}")),),
            ))

        # 8) Build the IOC intent. Size = size_usd / max(ask, limit_price). The
        # risk gate computes notional = size * limit_price; with limit_price set
        # to price_extreme_max (typically 1.0) and ask < 1.0, dividing by ask
        # alone overshoots the cap by cents and the gate vetoes every order.
        # Optional safety_d-aware scaling: bigger size on safer setups, floor
        # at size_min_fraction so we never effectively trade zero. Only applies
        # when min_safety_d > 0 (so safety_d_entry is available) and scaling != "fixed".
        scale = 1.0
        if (
            self.cfg.size_scaling != "fixed"
            and safety_d_entry is not None
            and self.cfg.min_safety_d > 0.0
        ):
            d_span = 3.0 - self.cfg.min_safety_d
            if d_span > 0:
                raw = (safety_d_entry - self.cfg.min_safety_d) / d_span
                if self.cfg.size_scaling == "sqrt_safety":
                    raw = math.sqrt(max(raw, 0.0))
                # else "linear_safety" uses raw as-is.
                scale = min(1.0, max(self.cfg.size_min_fraction, raw))
        size_usd = self.cfg.max_position_usd * scale
        sizing_px = max(win.ask_px, self.cfg.price_extreme_max)
        size = max(0.0, math.floor((size_usd / sizing_px) * 100) / 100)
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
