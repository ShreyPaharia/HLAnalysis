from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from loguru import logger

# Re-exports from sub-modules so the existing import surface is fully preserved.
from ._theta_math import (  # noqa: F401
    _ANNUAL_SECONDS,
    _INV_SQRT_2PI,
    _jr_trust_weight,
    _p_leg_win_prob,
    _p_leg_win_prob_and_phi,
    _phi,
    _safety_d_for_region,
)
from .base import Strategy
from .fee import fee_per_share
from .intents import make_entry_intent, make_exit_intent, round_size
from .regions import winning_region as _winning_region
from .theta_params import (  # noqa: F401
    _D,
    ThetaHarvesterConfig,
    ThetaHarvesterParams,
    build_v3_2_volclock,
    build_v3_4_lmgate,
    build_v3_5_momentum_mr,
    build_v3_theta_harvester,
)
from .topup import run_topup
from .types import (
    Action,
    BookState,
    Decision,
    Diagnostic,
    Position,
    QuestionView,
)
from .vol import annualized_sigma, bipower_variation_sigma


class ThetaHarvesterStrategy(Strategy):
    name = "theta_harvester"
    # theta derives σ from recent_returns only; it never reads recent_hl_bars
    # (the param is accepted for the shared evaluate() contract but unused). Tell
    # the runner/engine to skip building the HL-bar tuple — see Strategy.consumes_hl_bars.
    consumes_hl_bars = False

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
        self._cfg_by_class: dict[str, ThetaHarvesterConfig] = dict(cfg_by_class) if cfg_by_class else {}

    def _cfg_for(self, question: QuestionView) -> ThetaHarvesterConfig:
        return self._cfg_by_class.get(question.klass, self._default_cfg)

    def decision_lookback_seconds(self) -> int | None:
        """Max seconds of ``recent_returns`` theta consumes, across the default
        cfg AND every per-class override (so no class is under-provisioned).

        Per-cfg consumption is the largest of the count-based tail slices the
        evaluator takes:
          * σ / LM gate     → ``vol_lookback_seconds`` (``[-vol_lookback//dt:]``)
          * drift μ         → ``drift_lookback_seconds`` (``[-drift_lookback//dt:]``)
          * momentum/MR gate→ ``momentum_mr_lookback_min`` SAMPLES, i.e.
                              ``momentum_mr_lookback_min · vol_sampling_dt_seconds``
                              seconds — only when ``momentum_mr_enabled``.
        The lead-lag veto reads only ``recent_returns[-1]`` (covered by any of
        the above). See Strategy.decision_lookback_seconds.
        """
        need = 0
        for c in (self._default_cfg, *self._cfg_by_class.values()):
            n = max(int(c.vol_lookback_seconds), int(c.drift_lookback_seconds))
            if c.momentum_mr_enabled:
                n = max(n, int(c.momentum_mr_lookback_min) * int(c.vol_sampling_dt_seconds))
            need = max(need, n)
        return need or None

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
                question=question,
                books=books,
                reference_price=reference_price,
                recent_returns=recent_returns,
                recent_volume_usd=recent_volume_usd,
                position=position,
                now_ns=now_ns,
                recent_hl_bars=recent_hl_bars,
            )
        prev_cfg = self.cfg
        self.cfg = resolved_cfg
        try:
            return self._evaluate(
                question=question,
                books=books,
                reference_price=reference_price,
                recent_returns=recent_returns,
                recent_volume_usd=recent_volume_usd,
                position=position,
                now_ns=now_ns,
                recent_hl_bars=recent_hl_bars,
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
                question=question,
                books=books,
                reference_price=reference_price,
                sigma=sigma,
                mu_eff=mu_eff,
                tau_yr=tau_yr,
                tau_s=tau_s,
                position=position,
            )
            if exit_dec.action == Action.EXIT:
                return exit_dec
            if not self.cfg.topup_enabled:
                return exit_dec
            topup_dec = self._evaluate_topup(
                question=question,
                books=books,
                reference_price=reference_price,
                sigma=sigma,
                mu_eff=mu_eff,
                tau_yr=tau_yr,
                position=position,
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
            question=question,
            books=books,
            reference_price=reference_price,
            sigma=sigma,
            mu_eff=mu_eff,
            tau_yr=tau_yr,
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
            clip_min=self.cfg.vol_clip_min,
            clip_max=self.cfg.vol_clip_max,
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
        self,
        *,
        question: QuestionView,
        books: Mapping[str, BookState],
        reference_price: float,
        sigma: float,
        mu_eff: float,
        tau_yr: float,
        recent_returns: tuple[float, ...] = (),
    ) -> Decision:
        # TTE entry window — fixed [tte_min_seconds, tte_max_seconds] bound.
        tau_s = tau_yr * _ANNUAL_SECONDS
        if not (self.cfg.tte_min_seconds <= tau_s <= float(self.cfg.tte_max_seconds)):
            return Decision(
                action=Action.HOLD, diagnostics=(Diagnostic("info", "tte_out_of_window", (("tte_s", f"{tau_s:.0f}"),)),)
            )

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
                return Decision(
                    action=Action.HOLD,
                    diagnostics=(
                        Diagnostic(
                            "info",
                            "near_strike_hover",
                            (("dist_pct", f"{dist_pct:.5f}"), ("min_dist_pct", f"{self.cfg.min_distance_pct:.5f}")),
                        ),
                    ),
                )

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
                reference_price=reference_price,
                lo=lo,
                hi=hi,
                sigma=sigma,
                mu_eff=mu_eff,
                tau_yr=tau_yr,
            )
            if pp is None:
                continue  # NO leg of a middle bucket — no contiguous winning region
            p_win, phi_d = pp
            # Entry-side safety_d floor (hysteresis with exit_safety_d). Skip a
            # leg whose σ√τ distance to its nearest adverse boundary is below the
            # floor — this blocks the immediate re-buy after an exit_safety_d cut
            # (price still near the strike) that drives the v31 churn loop.
            if self.cfg.min_safety_d > 0.0:
                entry_safety_d = _safety_d_for_region(
                    reference_price=reference_price,
                    lo=lo,
                    hi=hi,
                    sigma=sigma,
                    mu_eff=mu_eff,
                    tau_yr=tau_yr,
                )
                if entry_safety_d is not None and entry_safety_d < self.cfg.min_safety_d:
                    continue
            fee = fee_per_share(self.cfg, p_win, side="entry")
            edge = p_win - book.ask_px - fee - self.cfg.half_spread_assumption
            per_leg.append((sym, p_win, edge, book, phi_d))

        if not per_leg:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "no_book"),))

        # Favorite gate: require the leg's mid ≥ threshold. The binary case
        # naturally collapses to "exactly one side passes" because YES+NO=1.
        # Buckets: filter to legs whose mid ≥ threshold; if no leg passes,
        # HOLD with diagnostic.
        # When require_two_sided_entry is True, ask-only legs are excluded from
        # the comparison entirely — a one-sided stale ask (e.g. 0.99) cannot
        # pass the threshold on the ask alone. Default False preserves legacy
        # _mid fallback behavior (bit-identical when off).
        if self.cfg.favorite_threshold > 0.0 or self.cfg.favorite_max is not None:

            def _mid(b: BookState) -> float:
                if b.bid_px is not None and b.ask_px is not None:
                    return (b.bid_px + b.ask_px) / 2.0
                if self.cfg.require_two_sided_entry:
                    return 0.0  # one-sided quote fails the favorite gate
                return b.ask_px if b.ask_px is not None else (b.bid_px or 0.0)

            if self.cfg.favorite_threshold > 0.0:
                per_leg = [t for t in per_leg if _mid(t[3]) >= self.cfg.favorite_threshold]
            # v31-improvement-eval (Card E): upper-bound the favorite band — deep
            # favorites above favorite_max carry little net edge over fee+half-
            # spread and a fat left tail. None → no cap (bit-identical).
            if self.cfg.favorite_max is not None:
                per_leg = [t for t in per_leg if _mid(t[3]) <= self.cfg.favorite_max]
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
                return Decision(
                    action=Action.HOLD,
                    diagnostics=(
                        Diagnostic(
                            "info", "bid_notional_too_thin", (("min_usd", f"{self.cfg.min_bid_notional_usd:.2f}"),)
                        ),
                    ),
                )

        # Pick the leg with the highest GAMMA-AWARE effective edge. For λ=None
        # (legacy) this collapses to picking on raw edge. For λ>0 we subtract
        # the per-leg path-variance penalty λ·φ(d) so a near-strike leg with
        # the same raw edge as a far-from-strike leg is correctly de-prioritised.
        gamma_lambda = self.cfg.gamma_lambda or 0.0
        chosen_sym, chosen_p, chosen_edge, chosen_book, chosen_phi = max(
            per_leg, key=lambda t: t[2] - gamma_lambda * t[4]
        )
        effective_edge = chosen_edge - gamma_lambda * chosen_phi

        # v31-improvement-eval (Card C): lead-lag micro-veto. Pure downside
        # protection — the binary mid tracks the perp with a ~5s half-life, so
        # right after a sharp ADVERSE reference move the favorite ask is briefly
        # stale-high. Veto when the latest per-sample reference return is a
        # jump-sized adverse move (|z| > k). None disables (bit-identical).
        if self.cfg.leadlag_veto_k is not None and recent_returns:
            r_last = float(recent_returns[-1])
            sigma_per_sample = sigma * math.sqrt(self.cfg.vol_sampling_dt_seconds / _ANNUAL_SECONDS)
            if sigma_per_sample > 0.0:
                z = r_last / sigma_per_sample
                fav_side = +1 if chosen_sym == question.yes_symbol else -1
                # Binary: adverse = move against the favorite (down for YES, up
                # for NO). Bucket: any jump-sized move destabilizes the band.
                adverse = (z * fav_side < -self.cfg.leadlag_veto_k) if is_binary else (abs(z) > self.cfg.leadlag_veto_k)
                if adverse:
                    return Decision(
                        action=Action.HOLD,
                        diagnostics=(
                            Diagnostic(
                                "info",
                                "leadlag_veto",
                                (
                                    ("z", f"{z:.3f}"),
                                    ("k", f"{self.cfg.leadlag_veto_k:.3f}"),
                                    ("fav_side", str(fav_side)),
                                ),
                            ),
                        ),
                    )

        # SHR-102 (a): dynamic entry spread gate. When enabled, compare the
        # chosen leg's live half-spread to the net fair-value edge budget. The
        # check is FAIR/MID-REFERENCED (not limit-referenced) as required by
        # the order-mechanics note in the research doc: the existing slippage
        # gate is limit-referenced (limit == wide bid → slip ≈ 0 → no-op).
        #
        # gate fires when: live_half_spread > (p_win − mid − fee) − edge_buffer
        # equivalently: the round-trip spread cost > the gross fair-value edge
        # above the buffer. This auto-skips illiquid bucket books while still
        # entering during their tradeable (tight-spread) windows.
        if self.cfg.entry_spread_gate:
            _bid = chosen_book.bid_px
            _ask = chosen_book.ask_px
            if _bid is not None and _ask is not None and _ask > _bid:
                live_half_spread = (_ask - _bid) / 2.0
                mid = (_bid + _ask) / 2.0
                fee_entry = fee_per_share(self.cfg, chosen_p, side="entry")
                fair_edge = chosen_p - mid - fee_entry
                edge_budget = fair_edge - self.cfg.edge_buffer
                if live_half_spread > edge_budget:
                    return Decision(
                        action=Action.HOLD,
                        diagnostics=(
                            Diagnostic(
                                "info",
                                "entry_spread_too_wide",
                                (
                                    ("half_spread", f"{live_half_spread:.4f}"),
                                    ("edge_budget", f"{edge_budget:.4f}"),
                                    ("fair_edge", f"{fair_edge:.4f}"),
                                ),
                            ),
                        ),
                    )

        # v3.5: momentum / MR gate — skip if regime == "mr" and aligned-signed
        # score < -tau_gate. Computed AFTER favorite is chosen so we know which
        # side to align to.
        if self.cfg.momentum_mr_enabled and self.cfg.momentum_mr_mode == "gate":
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
                return Decision(
                    action=Action.HOLD, diagnostics=(Diagnostic("info", "momentum_mr_gate", tuple(gate_diag_kv)),)
                )

        # Build the edge diagnostic. Binary questions keep the p_model /
        # edge_yes / edge_no schema (meaningful per-side); multi-leg buckets
        # emit native chosen_leg / chosen_edge fields instead (see the bucket
        # branch below).
        if is_binary:
            yes = books.get(question.yes_symbol)
            no_ = books.get(question.no_symbol)
            # Binary always has both legs at this point because per_leg is non-empty
            # (favorite gate already passed if active). p_yes = P(S>strike).
            p_yes_view = (
                _p_leg_win_prob(
                    reference_price=reference_price,
                    lo=question.strike,
                    hi=None,
                    sigma=sigma,
                    mu_eff=mu_eff,
                    tau_yr=tau_yr,
                )
                or 0.0
            )
            fee_yes = fee_per_share(self.cfg, p_yes_view, side="entry")
            fee_no = fee_per_share(self.cfg, 1.0 - p_yes_view, side="entry")
            edge_yes = (
                p_yes_view
                - (yes.ask_px if yes and yes.ask_px is not None else 1.0)
                - fee_yes
                - self.cfg.half_spread_assumption
            )
            edge_no = (
                (1.0 - p_yes_view)
                - (no_.ask_px if no_ and no_.ask_px is not None else 1.0)
                - fee_no
                - self.cfg.half_spread_assumption
            )
            # Apply favorite-gate disabling sentinels so the diagnostic mirrors
            # the legacy behavior exactly (existing tests check exact values).
            if self.cfg.favorite_threshold > 0.0:
                if chosen_sym == question.yes_symbol:
                    edge_no = -1e9
                else:
                    edge_yes = -1e9
            ln_sk = math.log(reference_price / question.strike)
            diag = Diagnostic(
                "info",
                "edge",
                (
                    ("p_model", f"{p_yes_view:.4f}"),
                    ("edge_yes", f"{edge_yes:.4f}"),
                    ("edge_no", f"{edge_no:.4f}"),
                    ("sigma", f"{sigma:.4f}"),
                    ("tau_yr", f"{tau_yr:.12f}"),
                    ("ln_sk", f"{ln_sk:.4f}"),
                ),
            )
        else:
            # Bucket (multi-leg) diagnostic. Buckets have no binary YES/NO, so we
            # emit NATIVE fields — chosen_leg / chosen_edge — for the chosen leg
            # rather than abusing the binary schema with edge_no=-1e9 sentinels.
            # edge_yes is retained as a back-compat MIRROR of chosen_edge so the
            # fixed-schema diagnostics parquet (backtest/runner) and the entry
            # fill-meta reader keep their populated value without an off-schema
            # column; edge_no is intentionally omitted (no sentinel).
            diag = Diagnostic(
                "info",
                "edge",
                (
                    ("p_model", f"{chosen_p:.4f}"),
                    ("chosen_leg", chosen_sym),
                    ("chosen_edge", f"{chosen_edge:.4f}"),
                    ("edge_yes", f"{chosen_edge:.4f}"),
                    ("sigma", f"{sigma:.4f}"),
                    ("tau_yr", f"{tau_yr:.12f}"),
                    ("ln_sk", "0.0000"),
                ),
            )

        # v3.5: momentum / MR tilt — scale the effective edge_buffer by
        # (1 - alpha_tilt * score). Aligned momentum (score > 0) lowers the
        # bar; MR against favorite (score < 0) raises it.
        effective_edge_buffer = self.cfg.edge_buffer
        if self.cfg.momentum_mr_enabled and self.cfg.momentum_mr_mode == "tilt":
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
            effective_edge_buffer = self.cfg.edge_buffer * (1.0 - self.cfg.momentum_mr_alpha_tilt * mm_score)
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
                diags = (
                    Diagnostic(
                        "info",
                        "edge_after_gamma_below_buffer",
                        (
                            ("raw_edge", f"{chosen_edge:.4f}"),
                            ("phi_d", f"{chosen_phi:.4f}"),
                            ("gamma_penalty", f"{gamma_lambda * chosen_phi:.4f}"),
                        ),
                    ),
                ) + diags
            return Decision(action=Action.HOLD, diagnostics=diags)

        if self.cfg.edge_max is not None and chosen_edge >= self.cfg.edge_max:
            return Decision(
                action=Action.HOLD,
                diagnostics=(
                    Diagnostic(
                        "info",
                        "edge_too_extreme",
                        (
                            ("edge", f"{chosen_edge:.4f}"),
                            ("edge_max", f"{self.cfg.edge_max:.4f}"),
                        ),
                    ),
                    diag,
                ),
            )

        # v3.4-LMgate: Lee-Mykland (2008) post-edge jump filter — TRULY τ-free.
        # Stat = |r_last| / √(BV_per_sample), comparing the latest log-return
        # to the jump-robust per-sample σ over the same window. > threshold
        # ⇒ jump that the market may not have repriced yet. None disables.
        if self.cfg.lm_threshold is not None and len(recent_returns) > 0:
            n_keep = max(2, self.cfg.vol_lookback_seconds // self.cfg.vol_sampling_dt_seconds)
            window = recent_returns[-n_keep:]
            if len(window) < 2:
                return Decision(
                    action=Action.HOLD,
                    diagnostics=(
                        Diagnostic("info", "lm_no_returns"),
                        diag,
                    ),
                )
            arr = np.asarray(window, dtype=np.float64)
            bipower_sigma = float(bipower_variation_sigma(arr))
            if bipower_sigma <= 0.0:
                return Decision(
                    action=Action.HOLD,
                    diagnostics=(
                        Diagnostic("info", "lm_bv_zero"),
                        diag,
                    ),
                )
            lm_stat = abs(float(arr[-1])) / bipower_sigma
            if lm_stat < self.cfg.lm_threshold:
                return Decision(
                    action=Action.HOLD,
                    diagnostics=(
                        Diagnostic(
                            "info",
                            "lm_gate_no_jump",
                            (
                                ("lm_stat", f"{lm_stat:.3f}"),
                                ("lm_threshold", f"{self.cfg.lm_threshold:.3f}"),
                            ),
                        ),
                        diag,
                    ),
                )

        # v31-improvement-eval (Card F): vol-regime sizing. Scale the clip by the
        # realized-σ regime (theta-harvest is paid more in high-σ regimes). The
        # resulting clip is still bounded by the engine inventory cap. Defaults
        # (disabled / mults=1.0) are bit-identical.
        clip = self.cfg.max_position_usd
        if self.cfg.vol_regime_sizing and self.cfg.vol_regime_sigma_threshold is not None:
            mult = (
                self.cfg.vol_regime_high_mult
                if sigma >= self.cfg.vol_regime_sigma_threshold
                else self.cfg.vol_regime_low_mult
            )
            clip = clip * mult
        size = max(0.0, round_size(clip, chosen_book.ask_px))
        if size <= 0:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("warn", "size_zero"), diag))

        intent = make_entry_intent(
            question,
            symbol=chosen_sym,
            size=size,
            limit_price=chosen_book.ask_px,
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
        self,
        *,
        question: QuestionView,
        books: Mapping[str, BookState],
        reference_price: float,
        sigma: float,
        mu_eff: float,
        tau_yr: float,
        tau_s: float,
        position: Position,
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

        # SHR-102 (b)/(c): exit hold-to-settle. When enabled, suppress
        # exit_safety_d and exit_edge liquidations on a held leg whose book is
        # persistently wide — hold to settlement instead of crossing a spread
        # that is unlikely to tighten before resolution. The hard stop-loss
        # (Rule 1 above) ALWAYS fires; this gate only suppresses model-driven
        # soft exits. Time-stop and take-profit are also unaffected.
        #
        # Condition: exit_spread_hold > 0 AND live half-spread > threshold.
        _spread_hold_active = False
        if self.cfg.exit_spread_hold > 0.0 and held.bid_px is not None and held.ask_px is not None:
            held_half_spread = (held.ask_px - held.bid_px) / 2.0
            if held_half_spread > self.cfg.exit_spread_hold:
                _spread_hold_active = True

        # Rule 2.5 (v3.1.1+): σ-normalized mid-hold distance exit. Fires BEFORE
        # the bid collapses, catching cases where BTC has drifted close to the
        # adverse boundary while the held leg's bid is still stale-positive.
        # Mirrors v1's exit_safety_d but uses v3.1's Itô-corrected d-machinery
        # (drift-aware). Skipped when σ·√τ is non-positive or the leg has no
        # contiguous winning region (middle-bucket NO).
        # Skipped when _spread_hold_active (SHR-102 hold-to-settle).
        if self.cfg.exit_safety_d > 0.0 and not _spread_hold_active:
            safety_d = _safety_d_for_region(
                reference_price=reference_price,
                lo=lo,
                hi=hi,
                sigma=sigma,
                mu_eff=mu_eff,
                tau_yr=tau_yr,
            )
            if safety_d is not None and safety_d < self.cfg.exit_safety_d:
                intent = make_exit_intent(
                    question,
                    position,
                    limit_price=held.bid_px,
                    exit_reason="exit_safety_d",
                )
                return Decision(
                    action=Action.EXIT,
                    intents=(intent,),
                    diagnostics=(
                        Diagnostic(
                            "info",
                            "exit_safety_d",
                            (
                                ("exit_reason", "safety_d_below_threshold"),
                                ("exit_safety_d", f"{safety_d:.4f}"),
                                ("exit_threshold", f"{self.cfg.exit_safety_d:.4f}"),
                            ),
                        ),
                    ),
                )

        pp = _p_leg_win_prob_and_phi(
            reference_price=reference_price,
            lo=lo,
            hi=hi,
            sigma=sigma,
            mu_eff=mu_eff,
            tau_yr=tau_yr,
        )
        if pp is None:
            # Middle-bucket NO with no contiguous winning region. Skip edge
            # check; rely on stop_loss / time_stop / settlement.
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "hold_no_region"),))
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
        # SHR-102: when _spread_hold_active, suppress exit_edge (hold-to-settle).
        if exit_now and not _spread_hold_active:
            return self._exit_intent(question, position, held, reason="exit_edge")

        hold_diags: tuple = (
            Diagnostic(
                "info",
                "hold",
                (
                    ("edge_held", f"{edge_held:.4f}"),
                    ("held_p", f"{held_p:.4f}"),
                    ("tau_s", f"{tau_s:.0f}"),
                ),
            ),
        )
        if _spread_hold_active:
            held_half_spread_val = (held.ask_px - held.bid_px) / 2.0  # type: ignore[operator]
            hold_diags = (
                Diagnostic(
                    "info",
                    "hold_spread_too_wide",
                    (
                        ("half_spread", f"{held_half_spread_val:.4f}"),
                        ("exit_spread_hold", f"{self.cfg.exit_spread_hold:.4f}"),
                    ),
                ),
            ) + hold_diags
        return Decision(action=Action.HOLD, diagnostics=hold_diags)

    def _evaluate_topup(
        self,
        *,
        question: QuestionView,
        books: Mapping[str, BookState],
        reference_price: float,
        sigma: float,
        mu_eff: float,
        tau_yr: float,
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
            return Decision(
                action=Action.HOLD, diagnostics=(Diagnostic("info", "topup_skip", (("reason", "no_book"),)),)
            )

        def _on_not_needed(current_ntl: float, target_ntl: float) -> Decision:
            # Demoted from info → debug 2026-05-22: fires every scan tick (~1/s)
            # while a position is held and the topup gate stays "not_needed",
            # drowning genuinely interesting events out of journalctl. The
            # current/target notionals are still attached to the Diagnostic
            # below, which feeds the scanner's gate-transition log
            # (gate_decisions.jsonl) and renders in the bus-event journal log.
            logger.debug(
                "topup_skip q={} sym={} reason=not_needed current_ntl=${:.2f} target_ntl=${:.2f}",
                question.question_idx,
                position.symbol,
                current_ntl,
                target_ntl,
            )
            return Decision(
                action=Action.HOLD,
                diagnostics=(
                    Diagnostic(
                        "info",
                        "topup_skip",
                        (
                            ("reason", "not_needed"),
                            ("current_ntl", f"{current_ntl:.2f}"),
                            ("target_ntl", f"{target_ntl:.2f}"),
                        ),
                    ),
                ),
            )

        return run_topup(
            question=question,
            books=books,
            position=position,
            max_position_usd=self.cfg.max_position_usd,
            topup_threshold_pct=self.cfg.topup_threshold_pct,
            topup_min_notional_usd=self.cfg.topup_min_notional_usd,
            run_entry=lambda: self._evaluate_entry(
                question=question,
                books=books,
                reference_price=reference_price,
                sigma=sigma,
                mu_eff=mu_eff,
                tau_yr=tau_yr,
                recent_returns=recent_returns,
            ),
            on_no_book=_on_no_book,
            on_not_needed=_on_not_needed,
        )

    def _exit_intent(self, question: QuestionView, position: Position, held: BookState, *, reason: str) -> Decision:
        intent = make_exit_intent(
            question,
            position,
            limit_price=held.bid_px,
            exit_reason=reason,
        )
        return Decision(
            action=Action.EXIT,
            intents=(intent,),
            diagnostics=(Diagnostic("info", reason),),
        )
