"""Pydantic model, frozen dataclass, and backtest-registry builder for the
theta_harvester strategy family.

All three were originally in ``theta_harvester.py``.  They are extracted here
so that config-layer code (``engine/config.py``) can import ``ThetaHarvesterParams``
without pulling in the full strategy class and its heavy numpy/scipy imports,
and so the params/builder definitions live in a focused module.

All symbols defined here are re-exported from ``theta_harvester.py`` so the
existing import surface is fully preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    # Only needed for type annotations in builder return types — not at runtime.
    from .theta_harvester import ThetaHarvesterStrategy


class ThetaHarvesterParams(BaseModel):
    """Pydantic model declaring every *optional* theta_harvester knob with its
    canonical default.  This is the SINGLE SOURCE OF TRUTH for those defaults:

    * ``ThetaParams`` (engine/config.py) inherits from this model — adding
      a knob here automatically makes it settable in the live YAML ``theta:``
      block with no separate declaration on ``ThetaParams``.
    * ``ThetaHarvesterConfig`` (the frozen dataclass below) carries the same
      field with the same default for the runtime strategy — enforced by
      ``tests/unit/test_theta_config_parity.py``.

    Adding one new optional knob therefore requires exactly two edits:
      1. Add the field here (with its canonical default).
      2. Add the same field to ThetaHarvesterConfig below.
    ThetaParams automatically inherits step 1; the parity test enforces step 2.

    Fields NOT declared here (they stay on ThetaParams directly):
    * Required-in-dataclass but defaulted-in-YAML fields: vol_lookback_seconds,
      vol_sampling_dt_seconds, vol_clip_min, vol_clip_max, edge_buffer,
      fee_taker, half_spread_assumption, drift_lookback_seconds, drift_blend,
      favorite_threshold, exit_edge_threshold, take_profit_price,
      time_stop_seconds. These have no dataclass default — ThetaParams must
      always supply them; they live on ThetaParams only, not here.
    * Allowlist-sourced fields: max_position_usd, tte_min_seconds,
      tte_max_seconds, stop_loss_pct. The live builder reads these from the
      strategy's ``defaults:`` block, not the ``theta:`` block.
    """

    model_config = ConfigDict(frozen=True)

    # === optional: entry edge filters ===
    edge_max: float | None = None
    min_distance_pct: float | None = None
    min_bid_notional_usd: float = 0.0
    gamma_lambda: float | None = None
    # === optional: position topup ===
    topup_enabled: bool = True
    topup_threshold_pct: float = 0.2
    topup_min_notional_usd: float = 11.0
    # === optional: exit extras / vol estimator / jump gate ===
    exit_safety_d: float = 0.0
    # Dwell/confirmation filter on the σ-normalized exit_safety_d soft-exit.
    # The exit only fires after safety_d has stayed below exit_safety_d for this
    # many CONSECUTIVE evaluate() scans — filtering single-tick reference
    # excursions that revert and settle in-band (the v31 bucket whipsaw, e.g.
    # leg #6460 sold @0.36 3min pre-settlement then settled @1.0). 1 == current
    # behavior (exit on the first breach), so the default is bit-identical. The
    # counter resets the moment safety_d recovers above threshold; the HARD
    # stop_loss/time_stop/take_profit/exit_edge/settlement paths are unaffected.
    exit_safety_d_dwell_scans: int = Field(default=1, ge=1)
    # Entry-side safety_d floor (mirrors v1's min_safety_d). Forms a hysteresis
    # band with exit_safety_d to stop enter→exit→re-enter churn. 0 disables.
    min_safety_d: float = 0.0
    vol_estimator: str = "sample_std"
    lm_threshold: float | None = None
    exit_take_profit_mode: bool = False
    exit_fee: float = 0.0007
    # === optional: fee model ===
    fee_model: str = "flat"
    fee_rate: float = 0.0
    # === optional: momentum / mean-reversion ===
    momentum_mr_enabled: bool = False
    momentum_mr_indicator: str = "z_ret"
    momentum_mr_lookback_min: int = 15
    momentum_mr_mode: str = "gate"
    momentum_mr_tau_gate: float = 1.0
    momentum_mr_alpha_tilt: float = 0.5
    momentum_mr_jr_trust_weight: bool = False
    # === SHR-102: bucket doom-loop fix (flag-gated, OFF BY DEFAULT) ===
    entry_spread_gate: bool = False
    exit_spread_hold: float = 0.0
    # === one-sided-quote guard (flag-gated, OFF BY DEFAULT) ===
    # When True, the favorite gate requires a two-sided quote (both bid_px and
    # ask_px present) on the chosen leg. A one-sided ask-only quote (e.g. stale
    # ask at 0.99 with no bid) can pass the favorite_threshold on the ask alone
    # — the _mid fallback returns ask_px when bid is absent. When this flag is
    # True, ask-only legs are excluded from the favorite-gate comparison instead,
    # preventing a stale high ask from triggering a spurious entry. Default False
    # preserves the current behavior exactly (bit-identical live when off).
    require_two_sided_entry: bool = False
    # === v31-improvement-eval 2026-06-13: favorite-band tilt (Card E FLB) ===
    # Upper bound on the favorite leg's mid. When set, the favorite gate keeps
    # only legs with mid in [favorite_threshold, favorite_max]. Card E found
    # favorites in mid∈[0.80,0.95] are underpriced ~6-9pp, but above ~0.95 the
    # premium thins. None disables (bit-identical). Applies to binary + bucket.
    favorite_max: float | None = None
    # === v31-improvement-eval 2026-06-13: vol-regime sizing (Card F) ===
    # When True, scale the entry clip by the realized-σ regime. σ >= threshold →
    # max_position_usd *= high_mult; else *= low_mult. Defaults are bit-identical.
    vol_regime_sizing: bool = False
    vol_regime_sigma_threshold: float | None = None
    vol_regime_low_mult: float = 1.0
    vol_regime_high_mult: float = 1.0
    # === v31-improvement-eval 2026-06-13: lead-lag micro-veto (Card C) ===
    # Pure downside-protection entry veto: skip entry when the most-recent
    # per-sample reference return is a sharp move ADVERSE to the chosen favorite
    # (|r_last|/σ_per_sample > k), catching the brief window where the binary
    # mid is stale after a perp move. None disables. Typical k: 3-5.
    leadlag_veto_k: float | None = None


@dataclass(frozen=True, slots=True)
class ThetaHarvesterConfig:
    # Fields are ordered required-first then defaulted (a dataclass constraint),
    # so a logical section (e.g. exit) is split across that boundary: its
    # required triggers live in the block below and its optional extras in the
    # defaulted block. Section banners mark the contiguous groups in place;
    # unifying them would require reordering across the required/default split.
    #
    # === required: vol / entry / fees / drift / sizing (copied from v2; we
    # deliberately do NOT import ModelEdgeConfig so v3 can diverge) ===
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
    # === required: exit triggers ===
    stop_loss_pct: float | None
    exit_edge_threshold: float  # exit when edge_held_side < this (typically <= 0)
    take_profit_price: float | None  # exit when held_bid >= entry_px + this
    time_stop_seconds: int  # exit when tau_s < this; 0 disables
    # === optional: entry edge filters ===
    # v3.1 entry upper-edge filter. Trade-level analysis of v3 PM-corpus run
    # surfaced an asymmetry: entries claiming edge >= 0.20 had hit rate 55%
    # (same as the rest) but with full-position wipeouts on the losers, netting
    # -$569 on 176 trades. Hypothesis: when GBM claims huge edge, the market
    # usually knows something the realized-vol model doesn't (event risk).
    # Filtering edge >= 0.20 lifted PnL +37% and cut max DD -84% on PM.
    # None disables the filter (preserves v3 baseline behavior).
    edge_max: float | None = None
    # Near-strike hover veto. When set, entries with
    # |reference_price − question.strike| / reference_price below this
    # fraction are vetoed. Only meaningful on priceBinary (the strike concept
    # is well-defined). PM corpus evidence: entries below 0.20% lose
    # -$7.68/entry across 57 entries (~2.7% of all v3.1 entries); the
    # 0.20–0.50% band immediately above is the *best* band (+$16.10/entry).
    # The sharp discontinuity concentrates v3.1's losses. None disables.
    min_distance_pct: float | None = None
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
    gamma_lambda: float | None = None
    # === optional: position topup ===
    # Strategy-side position topup. When a held position's notional (qty × ask)
    # is below max_position_usd by at least topup_threshold_pct, the strategy
    # re-runs ALL entry gates against the CURRENT state and, if the same leg is
    # still chosen, emits a second IOC ENTER intent sized to fill the shortfall.
    # Exit-eval always runs first — exit signals override topup. Recovers size
    # left on the table by HL IOC partial fills on thin HIP-4 books.
    topup_enabled: bool = True
    topup_threshold_pct: float = 0.2
    topup_min_notional_usd: float = 11.0
    # === optional: exit extras / vol estimator / jump gate (added incrementally) ===
    # σ-normalized mid-hold distance exit. Computes the signed distance from
    # BTC to the leg's NEAREST adverse boundary in σ√τ units (the same Itô-
    # corrected d-statistic the entry edge uses). When safety_d < this threshold
    # we close the held leg IOC at bid BEFORE the bid collapses — mirrors v1's
    # exit_safety_d. The existing ``edge_held`` gate fires only AFTER the bid
    # has already moved adversely; this gate fires while the underlying is
    # drifting toward the boundary, catching long-TTE losers earlier.
    # 0.0 disables (legacy behavior). Typical values: 0.25-1.0 (σ-units).
    exit_safety_d: float = 0.0
    # Dwell/confirmation count for the exit_safety_d soft-exit. The exit fires
    # only after safety_d < exit_safety_d holds for this many CONSECUTIVE scans.
    # 1 == legacy (exit on first breach; bit-identical). >1 filters single-tick
    # reference excursions that revert. Counter resets when safety_d recovers.
    exit_safety_d_dwell_scans: int = 1
    # Entry-side safety_d floor — the gate v1 (late_resolution) has but theta
    # historically lacked. Requires the candidate leg's σ√τ distance from its
    # nearest adverse boundary to be ≥ this before entering. Paired with a
    # lower exit_safety_d it forms a hysteresis band [exit_safety_d, min_safety_d]:
    # enter only when safely distant, exit when it drifts close, HOLD in between.
    # This breaks the post-exit_safety_d re-entry churn (a cut at safety_d<1.0
    # immediately re-bought on edge) while preserving the protective exit.
    # Should be > exit_safety_d to avoid thrashing on the boundary. 0 disables.
    min_safety_d: float = 0.0
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
    lm_threshold: float | None = None
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
    # === optional: fee model ===
    fee_model: str = "flat"
    fee_rate: float = 0.0
    # === optional: momentum / mean-reversion ===
    # v3.5: momentum / mean-reversion (MR) gate or tilt on the favorite-side
    # entry rule. Default off → v3.1 behavior is preserved bit-for-bit.
    # When `enabled` and `mode == "gate"`: skip entries where the momentum_mr
    # regime is "mr" against the favorite side and |score| > tau_gate.
    # When `enabled` and `mode == "tilt"`: scale the effective edge_buffer
    # by (1 − alpha_tilt * score). Score is signed: + = aligned with favorite.
    # See hlanalysis/strategy/momentum_mr.py and
    # docs/specs/2026-05-28-v35-momentum-mr-design.md.
    momentum_mr_enabled: bool = False
    momentum_mr_indicator: str = "z_ret"  # "z_ret" | "rsi" | "ma_sigma" | "hurst_ou"
    momentum_mr_lookback_min: int = 15
    momentum_mr_mode: str = "gate"  # "gate" | "tilt"
    momentum_mr_tau_gate: float = 1.0
    momentum_mr_alpha_tilt: float = 0.5
    # v3.6: Jump-Ratio trust weight. When True (and momentum_mr_enabled),
    # shrink the indicator score by (1 - JR) where JR = (RV - BPV) / RV is the
    # jump fraction over the same lookback as the indicator. Throttles the tilt
    # when the underlying is gapping (BPV diverges from RV). Default off.
    momentum_mr_jr_trust_weight: bool = False
    # === SHR-102: bucket doom-loop fix (flag-gated, OFF BY DEFAULT) ===
    # (a) Dynamic entry spread gate. When True, entry is suppressed when the
    # live book's half-spread (ask−bid)/2 of the chosen leg exceeds the net
    # fair-value edge budget: (p_win − mid − fee − edge_buffer). This is a
    # FAIR/MID-REFERENCED check, not limit-referenced — the existing
    # max_slippage_pct gate is limit-referenced and is a no-op on the doom loop
    # (limit == wide bid → slip ≈ 0). The dynamic gate auto-skips persistently-
    # wide illiquid bucket books while still trading the same market when its
    # book tightens to a tradeable window. False preserves pre-SHR-102 behavior.
    entry_spread_gate: bool = False
    # (b)/(c) Exit hold-to-settle. When > 0, suppress exit_safety_d and
    # exit_edge liquidations on a held leg whose book half-spread exceeds this
    # threshold — hold the position to settlement instead of crossing a spread
    # that won't tighten in time. The hard stop-loss gate ALWAYS fires regardless
    # of this setting (never disabled by exit_spread_hold). 0.0 disables (legacy
    # behavior). Typical value for HL bucket legs: 0.05–0.10.
    exit_spread_hold: float = 0.0
    # === one-sided-quote guard (flag-gated, OFF BY DEFAULT) ===
    # When True, the favorite gate requires a two-sided quote (both bid_px and
    # ask_px present) on the chosen leg. A one-sided ask-only quote (e.g. stale
    # ask at 0.99 with no bid) can pass the favorite_threshold on the ask alone
    # — the _mid fallback returns ask_px when bid is absent. When this flag is
    # True, ask-only legs are excluded from the favorite-gate comparison instead,
    # preventing a stale high ask from triggering a spurious entry. Default False
    # preserves the current behavior exactly (bit-identical live when off).
    require_two_sided_entry: bool = False
    # === v31-improvement-eval 2026-06-13: favorite-band tilt (Card E FLB) ===
    # Upper bound on the favorite leg's mid. When set, the favorite gate keeps
    # only legs with mid in [favorite_threshold, favorite_max]. Card E found
    # favorites in mid∈[0.80,0.95] are underpriced ~6-9pp, but above ~0.95 the
    # premium thins (deep favorites carry little edge over fee+half-spread and a
    # fat left tail). None disables the cap (bit-identical). Applies to both
    # binary (caps the chosen favorite) and bucket (caps the favorite band leg).
    favorite_max: float | None = None
    # === v31-improvement-eval 2026-06-13: vol-regime sizing (Card F) ===
    # When True, scale the entry clip by the prevailing realized-σ regime: the
    # strategy's annualized σ (same estimator the d-statistic uses) is compared
    # to vol_regime_sigma_threshold — σ >= threshold → max_position_usd *=
    # vol_regime_high_mult; σ < threshold → *= vol_regime_low_mult. Card F found
    # a +14% ann vol-risk-premium and r=0.53 between open-2h σ and theta edge, so
    # theta-harvest is paid more in high-σ regimes. Defaults (False, mults=1.0)
    # are bit-identical. The clip is still bounded by the engine inventory cap,
    # so prefer low_mult<1.0 (de-risk calm regimes) over high_mult>1.0.
    vol_regime_sizing: bool = False
    vol_regime_sigma_threshold: float | None = None
    vol_regime_low_mult: float = 1.0
    vol_regime_high_mult: float = 1.0
    # === v31-improvement-eval 2026-06-13: lead-lag micro-veto (Card C) ===
    # Pure downside-protection entry veto. Card C found the binary mid tracks the
    # perp with a ~5s half-life (sub-5s lag), so immediately after a sharp
    # adverse reference move the binary mid is briefly STALE (hasn't repriced
    # down), making the favorite ask look artificially attractive. When set, veto
    # entry if the most-recent per-sample reference return is a sharp move
    # ADVERSE to the chosen favorite, in per-sample σ units:
    #   |r_last| / σ_per_sample > leadlag_veto_k  AND  the move is adverse.
    # Binary: adverse = down for a YES favorite, up for a NO favorite.
    # Bucket : adverse = any sharp move (|r_last| crosses a band; mid stale).
    # None disables (bit-identical). Typical k: 3-5 (jump-sized moves only).
    leadlag_veto_k: float | None = None


from hlanalysis.backtest.core.registry import register  # noqa: E402

# Single source of truth for optional-knob defaults in the backtest builder.
# build_v3_theta_harvester uses _D.<field> instead of repeating literal defaults
# so changing a default in ThetaHarvesterParams automatically propagates here.
_D = ThetaHarvesterParams()


@register("v3_theta_harvester")
def build_v3_theta_harvester(params: dict) -> ThetaHarvesterStrategy:
    # Required fields (no default in ThetaHarvesterParams — must be in params):
    #   vol_lookback_seconds, edge_buffer, stop_loss_pct, exit_edge_threshold,
    #   take_profit_price.
    # Semi-required fields (not in ThetaHarvesterParams, but have backtest
    #   legacy defaults kept as literals here):
    #   vol_sampling_dt_seconds, vol_clip_min/max, fee_taker,
    #   half_spread_assumption, drift_lookback_seconds, drift_blend,
    #   max_position_usd, favorite_threshold, tte_min/max_seconds,
    #   time_stop_seconds.
    # Optional-knob fields (canonical defaults live in _D = ThetaHarvesterParams()):
    #   all fields declared in ThetaHarvesterParams.  Any new optional knob
    #   added to ThetaHarvesterParams automatically gets the right default here
    #   without touching this function — that is the single-source guarantee.
    from .theta_harvester import ThetaHarvesterStrategy  # noqa: PLC0415

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
        # --- optional knobs: defaults derived from _D (ThetaHarvesterParams) ---
        edge_max=(float(params["edge_max"]) if params.get("edge_max") is not None else _D.edge_max),
        min_distance_pct=(
            float(params["min_distance_pct"]) if params.get("min_distance_pct") is not None else _D.min_distance_pct
        ),
        min_bid_notional_usd=float(params.get("min_bid_notional_usd", _D.min_bid_notional_usd)),
        gamma_lambda=(float(params["gamma_lambda"]) if params.get("gamma_lambda") is not None else _D.gamma_lambda),
        topup_enabled=bool(params.get("topup_enabled", _D.topup_enabled)),
        topup_threshold_pct=float(params.get("topup_threshold_pct", _D.topup_threshold_pct)),
        topup_min_notional_usd=float(params.get("topup_min_notional_usd", _D.topup_min_notional_usd)),
        exit_safety_d=float(params.get("exit_safety_d", _D.exit_safety_d)),
        exit_safety_d_dwell_scans=int(params.get("exit_safety_d_dwell_scans", _D.exit_safety_d_dwell_scans)),
        min_safety_d=float(params.get("min_safety_d", _D.min_safety_d)),
        vol_estimator=str(params.get("vol_estimator", _D.vol_estimator)),
        lm_threshold=(float(params["lm_threshold"]) if params.get("lm_threshold") is not None else _D.lm_threshold),
        exit_take_profit_mode=bool(params.get("exit_take_profit_mode", _D.exit_take_profit_mode)),
        exit_fee=float(params.get("exit_fee", _D.exit_fee)),
        fee_model=str(params.get("fee_model", _D.fee_model)),
        fee_rate=float(params.get("fee_rate", _D.fee_rate)),
        momentum_mr_enabled=bool(params.get("momentum_mr_enabled", _D.momentum_mr_enabled)),
        momentum_mr_indicator=str(params.get("momentum_mr_indicator", _D.momentum_mr_indicator)),
        momentum_mr_lookback_min=int(params.get("momentum_mr_lookback_min", _D.momentum_mr_lookback_min)),
        momentum_mr_mode=str(params.get("momentum_mr_mode", _D.momentum_mr_mode)),
        momentum_mr_tau_gate=float(params.get("momentum_mr_tau_gate", _D.momentum_mr_tau_gate)),
        momentum_mr_alpha_tilt=float(params.get("momentum_mr_alpha_tilt", _D.momentum_mr_alpha_tilt)),
        momentum_mr_jr_trust_weight=bool(params.get("momentum_mr_jr_trust_weight", _D.momentum_mr_jr_trust_weight)),
        entry_spread_gate=bool(params.get("entry_spread_gate", _D.entry_spread_gate)),
        exit_spread_hold=float(params.get("exit_spread_hold", _D.exit_spread_hold)),
        require_two_sided_entry=bool(params.get("require_two_sided_entry", _D.require_two_sided_entry)),
        favorite_max=(float(params["favorite_max"]) if params.get("favorite_max") is not None else _D.favorite_max),
        vol_regime_sizing=bool(params.get("vol_regime_sizing", _D.vol_regime_sizing)),
        vol_regime_sigma_threshold=(
            float(params["vol_regime_sigma_threshold"])
            if params.get("vol_regime_sigma_threshold") is not None
            else _D.vol_regime_sigma_threshold
        ),
        vol_regime_low_mult=float(params.get("vol_regime_low_mult", _D.vol_regime_low_mult)),
        vol_regime_high_mult=float(params.get("vol_regime_high_mult", _D.vol_regime_high_mult)),
        leadlag_veto_k=(
            float(params["leadlag_veto_k"]) if params.get("leadlag_veto_k") is not None else _D.leadlag_veto_k
        ),
    )
    return ThetaHarvesterStrategy(cfg)


@register("v3_2_volclock")
def build_v3_2_volclock(params: dict) -> ThetaHarvesterStrategy:
    """v3.2-volclock — v3.1 with the σ estimator swapped from rolling sample
    stdev to bipower variation (jump-robust). Identical formula otherwise.

    Goal: keep p_model regime-aware without letting individual wicks inflate σ
    and shrink the d-statistic. After a wick, σ_BV stays calm → d stays large
    → the strategy can enter on the wick-driven mispricing instead of holding.

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
