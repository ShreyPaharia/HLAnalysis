from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from loguru import logger

from ._numba.safety import safety_d_for_region_core as _nb_safety_d_core
from ._numba.vol import (
    ewma_std as _nb_ewma_std,
    parkinson_sigma_window as _nb_parkinson_sigma_window,
    sample_std_returns as _nb_sample_std,
)
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


def _as_f64(x) -> np.ndarray:
    """Coerce ``tuple[float, ...] | list | ndarray`` to a contiguous float64
    1-D array. Cheap O(N) Python-loop-free conversion at the JIT boundary."""
    if isinstance(x, np.ndarray) and x.dtype == np.float64 and x.flags["C_CONTIGUOUS"]:
        return x
    return np.asarray(x, dtype=np.float64)


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

    Thin Optional wrapper around the JIT'd
    ``_numba.safety.safety_d_for_region_core``: encodes ``None`` bounds as
    flag/value pairs, then maps the JIT's NaN sentinel back to ``None``.
    """
    has_lo = lo is not None
    has_hi = hi is not None
    val = _nb_safety_d_core(
        float(ref_price),
        has_lo,
        float(lo) if has_lo else 0.0,
        has_hi,
        float(hi) if has_hi else 0.0,
        float(sigma_window),
        float(mu),
        float(tte_min),
        bool(drift_aware),
    )
    if math.isnan(val):
        return None
    return float(val)


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
    # Targeted size cap for catastrophic near-strike low-ask entries. When non-zero
    # AND chosen leg ask < size_cap_min_ask AND BTC is within
    # size_cap_max_dist_pct% of the chosen leg's nearest adverse boundary
    # (strike for binary YES; nearest of (lo, hi) for bucket legs),
    # multiply final size by (1 − size_cap_near_strike_pct). Default 0 disables.
    size_cap_near_strike_pct: float = 0.0
    # Distance threshold (in % of boundary) at or below which the cap kicks in.
    size_cap_max_dist_pct: float = 1.5
    # Ask ceiling below which the cap kicks in (favorites that are still cheap
    # enough to flip catastrophically).
    size_cap_min_ask: float = 0.88
    # Entry-gate price source. When True, the favorite-leg filter checks bid
    # against price_extreme_threshold instead of ask. Sizing and IOC limit
    # still use ask, so fills happen at-or-below ask as before — only the
    # *entry signal* changes.
    #
    # Why bid: on thin HL HIP-4 books, a stale "sell at $0.999" sign on the
    # ask can persist for hours (it costs the seller nothing). A stale bid at
    # the same level is a free option to any seller and gets lifted in
    # milliseconds. So the bid is a much more reliable indicator that the
    # market actually values the leg at that level. PM tuning (the 0.85
    # threshold) was on tight-spread books where bid≈ask, so this change
    # tightens the gate on HL without affecting the PM backtest.
    use_bid_for_entry_gate: bool = False
    # Veto entries when the best-bid notional (bid_px × bid_sz) is below this
    # USD floor. Catches single-share "spoof" bids that pass a numeric
    # bid-price threshold but represent no real buying interest. Default 0
    # disables (legacy behavior); set to ~$10–20 for a meaningful filter.
    min_bid_notional_usd: float = 0.0
    # Strategy-side position topup. When a held position's notional (qty × ask)
    # is below max_position_usd by at least topup_threshold_pct (e.g. 20%), the
    # strategy re-runs all entry gates against the CURRENT state and, if the
    # same leg is still the favourite, emits a second IOC ENTER intent sized to
    # fill the shortfall. Exit-eval always runs first — an exit signal overrides
    # any topup. Designed to recover the size shortfall left by HL IOC partial
    # fills on thin HIP-4 books. Set topup_enabled=False to disable.
    topup_enabled: bool = True
    topup_threshold_pct: float = 0.2
    topup_min_notional_usd: float = 11.0


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

    def __init__(
        self,
        cfg: LateResolutionConfig,
        *,
        cfg_by_class: Mapping[str, LateResolutionConfig] | None = None,
    ) -> None:
        # `cfg` is the default; `cfg_by_class` (built by the runtime from the
        # YAML allowlist) overrides per question.klass. The active config swap
        # happens inside evaluate() so internal helpers can keep reading
        # self.cfg unmodified. See _cfg_for / evaluate.
        self.cfg = cfg
        self._default_cfg = cfg
        self._cfg_by_class: dict[str, LateResolutionConfig] = (
            dict(cfg_by_class) if cfg_by_class else {}
        )

    def _cfg_for(self, question: QuestionView) -> LateResolutionConfig:
        return self._cfg_by_class.get(question.klass, self._default_cfg)

    # 1 / (4 ln 2): kept for backward reference (consumers used to reach
    # for it before the JIT helpers landed). The Parkinson constant now lives
    # in `_numba.vol`.
    _PARK_K = 1.0 / (4.0 * math.log(2.0))

    @staticmethod
    def _ewma_std(returns: tuple[float, ...] | np.ndarray, lam: float) -> float:
        """Recursive EWMA stdev of returns. Delegates to the JIT'd helper;
        Python signature preserved for callers/tests."""
        return float(_nb_ewma_std(_as_f64(returns), float(lam)))

    def _sigma_stdev(
        self, returns_window: tuple[float, ...] | np.ndarray
    ) -> float:
        arr = _as_f64(returns_window)
        if self.cfg.vol_ewma_lambda > 0.0:
            return float(_nb_ewma_std(arr, float(self.cfg.vol_ewma_lambda)))
        return float(_nb_sample_std(arr))

    def _sigma_parkinson(
        self, hl_window: tuple[tuple[float, float], ...] | np.ndarray
    ) -> float:
        """Window-level Parkinson σ via JIT helper. With EWMA: variance decays
        by λ across bars; else arithmetic mean of per-bar σ²."""
        if isinstance(hl_window, np.ndarray):
            hl_arr = np.ascontiguousarray(hl_window, dtype=np.float64)
            if hl_arr.size == 0:
                return 0.0
            highs = hl_arr[:, 0]
            lows = hl_arr[:, 1]
        else:
            n = len(hl_window)
            if n == 0:
                return 0.0
            highs = np.empty(n, dtype=np.float64)
            lows = np.empty(n, dtype=np.float64)
            for i, (h, l) in enumerate(hl_window):
                highs[i] = h
                lows[i] = l
        return float(
            _nb_parkinson_sigma_window(highs, lows, float(self.cfg.vol_ewma_lambda))
        )

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
        # Resolve and pin the per-class config for this evaluation. The body
        # (and every internal helper it calls) reads self.cfg directly, so the
        # swap is the simplest correct way to honor allowlist match-specific
        # gate overrides. Strategy.evaluate is single-threaded sync — restoring
        # in finally keeps self.cfg stable for external readers (tests, diags).
        resolved_cfg = self._cfg_for(question)
        if resolved_cfg is self.cfg:
            return self._evaluate_dispatch(
                question=question, books=books, reference_price=reference_price,
                recent_returns=recent_returns, recent_volume_usd=recent_volume_usd,
                position=position, now_ns=now_ns, recent_hl_bars=recent_hl_bars,
            )
        prev_cfg = self.cfg
        self.cfg = resolved_cfg
        try:
            return self._evaluate_dispatch(
                question=question, books=books, reference_price=reference_price,
                recent_returns=recent_returns, recent_volume_usd=recent_volume_usd,
                position=position, now_ns=now_ns, recent_hl_bars=recent_hl_bars,
            )
        finally:
            self.cfg = prev_cfg

    def _evaluate_dispatch(
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
                    exit_reason="exit_bid_below_floor",
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
                                exit_reason="exit_safety_d_5m",
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
                                exit_reason="exit_safety_d",
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
                        exit_reason="exit_stop_loss",
                    )
                    return Decision(
                        action=Action.EXIT,
                        intents=(intent,),
                        diagnostics=(Diagnostic("warn", "exit_stop_loss"),),
                    )
            # No exit gate fired. If topup is enabled, see if we should add to
            # the held position to recover from an IOC partial-fill shortfall.
            # Exit-eval already ran above and chose not to fire — topup is safe.
            if self.cfg.topup_enabled:
                topup_dec = self._evaluate_topup(
                    question=question, books=books, reference_price=reference_price,
                    recent_returns=recent_returns, recent_volume_usd=recent_volume_usd,
                    now_ns=now_ns, recent_hl_bars=recent_hl_bars, position=position,
                )
                if topup_dec is not None:
                    return topup_dec
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "have_position"),))

        return self._evaluate_entry(
            question=question, books=books, reference_price=reference_price,
            recent_returns=recent_returns, recent_volume_usd=recent_volume_usd,
            now_ns=now_ns, recent_hl_bars=recent_hl_bars,
        )

    def _evaluate_entry(
        self, *,
        question: QuestionView,
        books: Mapping[str, BookState],
        reference_price: float,
        recent_returns: tuple[float, ...],
        recent_volume_usd: float,
        now_ns: int,
        recent_hl_bars: tuple[tuple[float, float], ...] = (),
    ) -> Decision:
        """Run all entry gates and return the corresponding Decision.

        Shared between the no-position path (called from `evaluate`) and the
        held-position topup path (called from `_evaluate_topup`) so the gate
        flow is identical — a topup must clear every gate a fresh entry would.
        """
        diags: list[Diagnostic] = []

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
        best_gate_px = -1.0
        # On HL HIP-4 the ask quote is structurally unreliable (lonely sellers
        # leave stale high asks that persist for hours). The bid is harder to
        # fake — any stale-high bid is a free option that gets lifted in
        # milliseconds. When `use_bid_for_entry_gate` is set, we gate on bid
        # and require a non-trivial bid notional. Sizing and IOC limit still
        # use ask (we still take the ask to enter), so this changes the
        # *entry signal*, not the *execution price*. PM corpus tunings of
        # price_extreme_threshold were calibrated on tight-spread books where
        # bid ≈ ask, so flipping this flag is a net-tighten on HL only.
        gate_min = self.cfg.price_extreme_threshold
        gate_max = self.cfg.price_extreme_max
        for sym in eligible:
            b = books.get(sym)
            if b is None or b.ask_px is None or b.bid_px is None:
                continue
            if now_ns - b.last_l2_ts_ns > stale_ns:
                continue
            gate_px = b.bid_px if self.cfg.use_bid_for_entry_gate else b.ask_px
            if not (gate_min <= gate_px <= gate_max):
                continue
            # Stale-ask sanity cap. When gating on bid (use_bid_for_entry_gate),
            # the ASK is unchecked — a stale-high ask of 0.999 with a real bid of
            # 0.95 would pass the bid gate but fill against the stale price if
            # we then set limit_price=ask. Reject these explicitly so the
            # subsequent limit_price=ask is bounded by price_extreme_max in
            # the same way the legacy limit_price=price_extreme_max was.
            if self.cfg.use_bid_for_entry_gate and b.ask_px > gate_max:
                continue
            # Bid-notional sanity gate. A bid of 0.85 × 1 share passes a
            # bid_px filter but is essentially a 85¢ stake — no real buying
            # interest. We require bid notional ≥ floor; default 0 disables.
            if self.cfg.min_bid_notional_usd > 0.0:
                bid_notional = (b.bid_px or 0.0) * (b.bid_sz or 0.0)
                if bid_notional < self.cfg.min_bid_notional_usd:
                    continue
            if gate_px > best_gate_px:
                best_gate_px = gate_px
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

        # Targeted size cap for catastrophic near-strike low-ask entries. Applied
        # AFTER safety_d-based scaling so it stacks multiplicatively. Only fires
        # when ALL three conditions hold: cap enabled (pct > 0), chosen leg's
        # ask is below the favorite ceiling, AND BTC is within the configured
        # % of the leg's nearest adverse boundary. Bucket NO legs of the middle
        # outcome have no contiguous winning region (lo=hi=None) → skip.
        if self.cfg.size_cap_near_strike_pct > 0.0:
            lo, hi = _winning_region(question, win_symbol)
            nearest: float | None = None
            denom: float | None = None
            if lo is not None and hi is not None:
                d_lo = abs(reference_price - lo)
                d_hi = abs(reference_price - hi)
                if d_lo <= d_hi:
                    nearest, denom = d_lo, lo
                else:
                    nearest, denom = d_hi, hi
            elif lo is not None:
                nearest, denom = abs(reference_price - lo), lo
            elif hi is not None:
                nearest, denom = abs(reference_price - hi), hi
            if nearest is not None and denom is not None and denom > 0:
                dist_pct = (nearest / denom) * 100.0
                if (
                    dist_pct < self.cfg.size_cap_max_dist_pct
                    and win.ask_px < self.cfg.size_cap_min_ask
                ):
                    scale *= 1.0 - self.cfg.size_cap_near_strike_pct
                    if scale < 0.0:
                        scale = 0.0
                    diags.append(
                        Diagnostic(
                            "info",
                            "size_cap_near_strike",
                            (
                                ("dist_pct", f"{dist_pct:.3f}"),
                                ("ask", f"{win.ask_px:.3f}"),
                                ("scale_after", f"{scale:.3f}"),
                            ),
                        )
                    )

        size_usd = self.cfg.max_position_usd * scale
        # Both size and limit are pinned to top-of-book ask. Earlier behaviour
        # set limit_price=price_extreme_max (e.g. 0.99) so the IOC could walk
        # the book up to the cap; that's how v1 entries used to fill multiple
        # ask levels in one shot. New behaviour mirrors v3.1: limit_price=ask
        # consumes ONLY the top-of-book level and lets _evaluate_topup recover
        # any shortfall on the next tick. The stale-ask sanity cap inside the
        # gate loop above already rejects entries where ask > price_extreme_max,
        # so the protection that the old limit ceiling provided is preserved.
        size = max(0.0, math.floor((size_usd / win.ask_px) * 100) / 100)
        if size <= 0:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("warn", "size_zero"),))

        intent = OrderIntent(
            question_idx=question.question_idx,
            symbol=win_symbol,
            side="buy",
            size=size,
            limit_price=win.ask_px,
            cloid=f"hla-{uuid.uuid4()}",
            time_in_force="ioc",
        )
        return Decision(
            action=Action.ENTER,
            intents=(intent,),
            diagnostics=(Diagnostic("info", "entry"), *diags),
        )

    def _evaluate_topup(
        self, *,
        question: QuestionView,
        books: Mapping[str, BookState],
        reference_price: float,
        recent_returns: tuple[float, ...],
        recent_volume_usd: float,
        now_ns: int,
        recent_hl_bars: tuple[tuple[float, float], ...],
        position: Position,
    ) -> Decision | None:
        """Top up an under-filled held position.

        Returns:
          - ENTER decision when the held leg is still the favourite under
            current state AND the shortfall is large enough to justify an
            order ≥ topup_min_notional_usd.
          - HOLD decision (with skip-reason diagnostic) when a topup check
            fired but was rejected.
          - None when no topup attempt was warranted (lets the caller fall
            through to its legacy "have_position" HOLD diagnostic so the
            scanner log shape is unchanged for non-topup ticks).
        """
        held = books.get(position.symbol)
        if held is None or held.ask_px is None or held.ask_px <= 0:
            return None
        ask = held.ask_px
        current_ntl = abs(position.qty) * ask
        target_ntl = self.cfg.max_position_usd
        shortfall_ntl = target_ntl - current_ntl
        if shortfall_ntl < target_ntl * self.cfg.topup_threshold_pct:
            return None  # routine "fully sized" tick — keep legacy diagnostic

        # Re-run ALL entry gates against the current state.
        entry_dec = self._evaluate_entry(
            question=question, books=books, reference_price=reference_price,
            recent_returns=recent_returns, recent_volume_usd=recent_volume_usd,
            now_ns=now_ns, recent_hl_bars=recent_hl_bars,
        )
        if entry_dec.action != Action.ENTER or not entry_dec.intents:
            failed_gate = (
                entry_dec.diagnostics[0].message
                if entry_dec.diagnostics else "unknown"
            )
            logger.debug(
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
            logger.debug(
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
            logger.debug(
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


from hlanalysis.backtest.core.registry import register  # noqa: E402


@register("v1_late_resolution")
def build_v1_late_resolution(params: dict) -> LateResolutionStrategy:
    cfg = LateResolutionConfig(
        tte_min_seconds=int(params["tte_min_seconds"]),
        tte_max_seconds=int(params["tte_max_seconds"]),
        price_extreme_threshold=float(params["price_extreme_threshold"]),
        distance_from_strike_usd_min=float(params["distance_from_strike_usd_min"]),
        vol_max=float(params["vol_max"]),
        max_position_usd=float(params.get("max_position_usd", 100.0)),
        stop_loss_pct=float(params["stop_loss_pct"]) if params["stop_loss_pct"] is not None else 1e9,
        max_strike_distance_pct=float(params.get("max_strike_distance_pct", 50.0)),
        min_recent_volume_usd=float(params.get("min_recent_volume_usd", 0.0)),
        stale_data_halt_seconds=int(params.get("stale_data_halt_seconds", 86400)),
        price_extreme_max=float(params.get("price_extreme_max", 1.0)),
        min_safety_d=float(params.get("min_safety_d", 0.0)),
        vol_lookback_seconds=int(params.get("vol_lookback_seconds", 1800)),
        exit_safety_d=float(params.get("exit_safety_d", 0.0)),
        exit_bid_floor=float(params.get("exit_bid_floor", 0.0)),
        drift_aware_d=bool(params.get("drift_aware_d", False)),
        vol_ewma_lambda=float(params.get("vol_ewma_lambda", 0.0)),
        exit_safety_d_5m=float(params.get("exit_safety_d_5m", 0.0)),
        exit_vol_lookback_5m_seconds=int(params.get("exit_vol_lookback_5m_seconds", 300)),
        vol_estimator=str(params.get("vol_estimator", "stdev")),
        size_scaling=str(params.get("size_scaling", "fixed")),
        size_min_fraction=float(params.get("size_min_fraction", 0.25)),
        size_cap_near_strike_pct=float(params.get("size_cap_near_strike_pct", 0.0)),
        size_cap_max_dist_pct=float(params.get("size_cap_max_dist_pct", 1.5)),
        size_cap_min_ask=float(params.get("size_cap_min_ask", 0.88)),
        use_bid_for_entry_gate=bool(params.get("use_bid_for_entry_gate", False)),
        min_bid_notional_usd=float(params.get("min_bid_notional_usd", 0.0)),
        topup_enabled=bool(params.get("topup_enabled", True)),
        topup_threshold_pct=float(params.get("topup_threshold_pct", 0.2)),
        topup_min_notional_usd=float(params.get("topup_min_notional_usd", 11.0)),
    )
    return LateResolutionStrategy(cfg)
