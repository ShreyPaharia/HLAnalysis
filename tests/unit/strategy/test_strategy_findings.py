"""Tests for strategy/model correctness findings.

Covers four findings:
  SHR-59 (#20) — strike-distance gates parsed but never enforced in v1
  #36           — drift_aware ignores drift on two-sided (bucket-middle) regions
  #24           — GBM kernel has no sigma_sqrt_tau zero-guard
  #42           — bv_per_sample misnaming (pure rename, behaviour-preserving)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hlanalysis.strategy._theta_math import _p_leg_win_prob_and_phi
from hlanalysis.strategy.late_resolution import (
    LateResolutionConfig,
    LateResolutionStrategy,
    _safety_d_for_region,
)
from hlanalysis.strategy.types import Action, BookState, QuestionView
from hlanalysis.strategy.vol import bipower_variation_sigma


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _book(symbol: str, ask: float, bid: float, *, ts_ns: int) -> BookState:
    return BookState(
        symbol=symbol,
        bid_px=bid,
        bid_sz=100.0,
        ask_px=ask,
        ask_sz=100.0,
        last_trade_ts_ns=ts_ns,
        last_l2_ts_ns=ts_ns,
    )


def _binary_q(
    *,
    strike: float = 80_000.0,
    expiry_ns: int,
) -> QuestionView:
    return QuestionView(
        question_idx=1,
        yes_symbol="@Y",
        no_symbol="@N",
        strike=strike,
        expiry_ns=expiry_ns,
        underlying="BTC",
        klass="priceBinary",
        period="1d",
        leg_symbols=("@Y", "@N"),
    )


def _v1_cfg(**overrides) -> LateResolutionConfig:
    base = dict(
        tte_min_seconds=60,
        tte_max_seconds=86_400,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=500.0,
        vol_max=1.0,
        max_position_usd=100.0,
        stop_loss_pct=100.0,
        max_strike_distance_pct=5.0,
        min_recent_volume_usd=0.0,
        stale_data_halt_seconds=86400,
        min_safety_d=0.0,
    )
    base.update(overrides)
    return LateResolutionConfig(**base)


def _evaluate(
    strategy: LateResolutionStrategy,
    *,
    question: QuestionView,
    books: dict,
    reference_price: float,
    now_ns: int,
    recent_returns: tuple = tuple([0.0001] * 60),
):
    return strategy.evaluate(
        question=question,
        books=books,
        reference_price=reference_price,
        recent_returns=recent_returns,
        recent_volume_usd=50_000.0,
        position=None,
        now_ns=now_ns,
    )


# ===========================================================================
# SHR-59 (#20): Strike-distance gates enforced in v1._evaluate_entry
# ===========================================================================


class TestStrikeDistanceGatesSHR59:
    """distance_from_strike_usd_min and max_strike_distance_pct must be
    enforced in _evaluate_entry, not silently skipped."""

    now: int = 10_000_000_000_000

    @property
    def expiry(self) -> int:
        return self.now + 3600 * 1_000_000_000  # 1h TTE, comfortably in window

    def _books(self, ask: float = 0.97, ts_offset: int = 100) -> dict:
        ts = self.now - ts_offset
        return {
            "@Y": _book("@Y", ask=ask, bid=ask - 0.01, ts_ns=ts),
            "@N": _book("@N", ask=0.04, bid=0.03, ts_ns=ts),
        }

    def test_beyond_max_strike_distance_pct_is_vetoed(self) -> None:
        """Entry must be HELD when |strike - ref| / ref * 100 > max_strike_distance_pct.

        ref=90_000, strike=80_000 → distance=10_000/90_000*100 ≈ 11.1% > 5% cap.
        """
        q = _binary_q(strike=80_000.0, expiry_ns=self.expiry)
        books = self._books()
        s = LateResolutionStrategy(_v1_cfg(max_strike_distance_pct=5.0, distance_from_strike_usd_min=0.0))
        d = _evaluate(s, question=q, books=books, reference_price=90_000.0, now_ns=self.now)
        assert d.action is Action.HOLD, (
            f"Expected HOLD (strike too far from ref), got {d.action}; diags={d.diagnostics}"
        )
        assert any("strike_too_far" in diag.message for diag in d.diagnostics), (
            f"Expected strike_too_far diagnostic; got {d.diagnostics}"
        )

    def test_within_max_strike_distance_pct_is_allowed(self) -> None:
        """Entry must proceed when distance % is within the cap.

        ref=80_300, strike=80_000 → distance=300/80_300*100 ≈ 0.37% < 5% cap.
        """
        q = _binary_q(strike=80_000.0, expiry_ns=self.expiry)
        books = self._books()
        s = LateResolutionStrategy(_v1_cfg(max_strike_distance_pct=5.0, distance_from_strike_usd_min=0.0))
        d = _evaluate(s, question=q, books=books, reference_price=80_300.0, now_ns=self.now)
        assert d.action is Action.ENTER, f"Expected ENTER (within distance cap), got {d.action}; diags={d.diagnostics}"

    def test_below_min_usd_distance_is_vetoed(self) -> None:
        """Entry must be HELD when |strike - ref| < distance_from_strike_usd_min.

        strike=80_000, ref=80_200 → |distance|=$200 < $500 min.
        """
        q = _binary_q(strike=80_000.0, expiry_ns=self.expiry)
        books = self._books()
        s = LateResolutionStrategy(_v1_cfg(distance_from_strike_usd_min=500.0, max_strike_distance_pct=50.0))
        d = _evaluate(s, question=q, books=books, reference_price=80_200.0, now_ns=self.now)
        assert d.action is Action.HOLD, f"Expected HOLD (too close to strike), got {d.action}; diags={d.diagnostics}"
        assert any("strike_too_close" in diag.message for diag in d.diagnostics), (
            f"Expected strike_too_close diagnostic; got {d.diagnostics}"
        )

    def test_above_min_usd_distance_is_allowed(self) -> None:
        """Entry must proceed when |strike - ref| >= distance_from_strike_usd_min.

        strike=80_000, ref=80_600 → |distance|=$600 >= $500 min.
        """
        q = _binary_q(strike=80_000.0, expiry_ns=self.expiry)
        books = self._books()
        s = LateResolutionStrategy(_v1_cfg(distance_from_strike_usd_min=500.0, max_strike_distance_pct=50.0))
        d = _evaluate(s, question=q, books=books, reference_price=80_600.0, now_ns=self.now)
        assert d.action is Action.ENTER, f"Expected ENTER (sufficient distance), got {d.action}; diags={d.diagnostics}"

    def test_disabled_distance_gates_do_not_veto(self) -> None:
        """distance_from_strike_usd_min=0 and max_strike_distance_pct=100 → gates inactive."""
        q = _binary_q(strike=80_000.0, expiry_ns=self.expiry)
        books = self._books()
        s = LateResolutionStrategy(_v1_cfg(distance_from_strike_usd_min=0.0, max_strike_distance_pct=100.0))
        d = _evaluate(s, question=q, books=books, reference_price=80_300.0, now_ns=self.now)
        assert d.action is Action.ENTER, f"Expected ENTER (gates disabled), got {d.action}; diags={d.diagnostics}"

    def test_bucket_leg_too_far_from_nearest_boundary_is_vetoed(self) -> None:
        """For priceBucket, the gate uses the nearest winning-region boundary.

        Bucket leg @42 (middle YES) wins iff 77_991 < BTC <= 81_174.
        ref=90_000 → nearest boundary = 81_174.
        distance = |90_000 - 81_174| / 90_000 * 100 ≈ 9.8% > 5% cap → HOLD.
        """
        now = self.now
        bucket_q = QuestionView(
            question_idx=2,
            yes_symbol="@40",
            no_symbol="@41",
            strike=float("nan"),
            expiry_ns=now + 3600 * 1_000_000_000,
            underlying="BTC",
            klass="priceBucket",
            period="1d",
            leg_symbols=("@40", "@41", "@42", "@43", "@44", "@45"),
            kv=(("class", "priceBucket"), ("underlying", "BTC"), ("priceThresholds", "77991,81174")),
        )
        ts = now - 100
        books = {
            "@40": _book("@40", ask=0.04, bid=0.03, ts_ns=ts),
            "@42": _book("@42", ask=0.97, bid=0.96, ts_ns=ts),
            "@44": _book("@44", ask=0.04, bid=0.03, ts_ns=ts),
        }
        s = LateResolutionStrategy(_v1_cfg(max_strike_distance_pct=5.0, distance_from_strike_usd_min=0.0))
        d = _evaluate(s, question=bucket_q, books=books, reference_price=90_000.0, now_ns=now)
        assert d.action is Action.HOLD, f"Expected HOLD (bucket leg too far), got {d.action}; diags={d.diagnostics}"


# ===========================================================================
# #36: drift_aware must incorporate drift on two-sided regions
# ===========================================================================


class TestDriftAwareTwoSidedRegion:
    """_safety_d_for_region with a two-sided (lo, hi) region must apply drift
    to both bounds when drift_aware=True. Previously the two-sided branch
    returned early without reaching the drift-aware code."""

    def test_two_sided_drift_aware_off_baseline(self) -> None:
        """With drift_aware=False, two-sided safety_d is pure geometry."""
        ref = 79_500.0
        lo, hi = 77_991.0, 81_174.0
        d = _safety_d_for_region(
            ref_price=ref,
            lo=lo,
            hi=hi,
            sigma_window=1.0,
            mu=0.1,
            tte_min=10.0,
            drift_aware=False,
        )
        assert d is not None
        expected = min(math.log(ref / lo), math.log(hi / ref))
        assert math.isclose(d, expected, rel_tol=1e-9), f"expected {expected}, got {d}"

    def test_two_sided_drift_aware_on_shifts_d(self) -> None:
        """With drift_aware=True, drift must change the two-sided safety_d.

        Positive mu (upward drift) pushes ref toward hi (upper boundary) →
        d toward hi shrinks while d toward lo grows. The net minimum safety_d
        should differ from the no-drift case.
        """
        ref = 79_500.0
        lo, hi = 77_991.0, 81_174.0
        mu = 0.005
        tte_min = 100.0

        d_no_drift = _safety_d_for_region(
            ref_price=ref,
            lo=lo,
            hi=hi,
            sigma_window=1.0,
            mu=mu,
            tte_min=tte_min,
            drift_aware=False,
        )
        d_drift = _safety_d_for_region(
            ref_price=ref,
            lo=lo,
            hi=hi,
            sigma_window=1.0,
            mu=mu,
            tte_min=tte_min,
            drift_aware=True,
        )
        assert d_no_drift is not None
        assert d_drift is not None
        # Drift MUST change the result
        assert not math.isclose(d_no_drift, d_drift, rel_tol=1e-6), (
            f"drift_aware=True and drift_aware=False produced the same value {d_drift}; "
            "drift is not being applied to the two-sided branch"
        )

    def test_two_sided_drift_lowers_d_when_approaching_upper_bound(self) -> None:
        """Positive drift (trending toward hi) reduces safety distance from hi.

        When ref is at the geometric midpoint, positive mu makes hi-distance
        tighten more than lo-distance grows. The min(d_lo_adj, d_hi_adj)
        should be smaller than without drift.
        """
        lo, hi = 79_000.0, 81_000.0
        ref = math.sqrt(lo * hi)  # geometric midpoint → ln(ref/lo) == ln(hi/ref)

        mu = 0.01
        tte_min = 50.0

        d_no_drift = _safety_d_for_region(
            ref_price=ref,
            lo=lo,
            hi=hi,
            sigma_window=1.0,
            mu=mu,
            tte_min=tte_min,
            drift_aware=False,
        )
        d_drift = _safety_d_for_region(
            ref_price=ref,
            lo=lo,
            hi=hi,
            sigma_window=1.0,
            mu=mu,
            tte_min=tte_min,
            drift_aware=True,
        )
        assert d_no_drift is not None
        assert d_drift is not None
        # Positive drift toward hi → hi-side safety shrinks → min shrinks
        assert d_drift < d_no_drift, f"Expected drift to reduce safety_d (d_drift={d_drift} < d_no_drift={d_no_drift})"

    def test_one_sided_drift_aware_unchanged(self) -> None:
        """One-sided (lo, None) drift handling must remain identical to before."""
        d = _safety_d_for_region(
            ref_price=80_300.0,
            lo=80_000.0,
            hi=None,
            sigma_window=1.0,
            mu=0.001,
            tte_min=10.0,
            drift_aware=True,
        )
        expected = math.log(80_300.0 / 80_000.0) + 0.001 * 10.0
        assert d is not None
        assert math.isclose(d, expected, rel_tol=1e-9)

    def test_drift_aware_off_two_sided_unchanged_by_nonzero_mu(self) -> None:
        """When drift_aware=False, nonzero mu must not affect two-sided result."""
        ref, lo, hi = 79_500.0, 77_991.0, 81_174.0
        d0 = _safety_d_for_region(
            ref_price=ref,
            lo=lo,
            hi=hi,
            sigma_window=1.0,
            mu=0.0,
            tte_min=10.0,
            drift_aware=False,
        )
        d1 = _safety_d_for_region(
            ref_price=ref,
            lo=lo,
            hi=hi,
            sigma_window=1.0,
            mu=99.9,
            tte_min=10.0,
            drift_aware=False,
        )
        assert d0 is not None and d1 is not None
        assert math.isclose(d0, d1, rel_tol=1e-9), "drift_aware=False must not use mu in two-sided branch"


# ===========================================================================
# #24: GBM kernel sigma_sqrt_tau zero-guard
# ===========================================================================


class TestGBMKernelZeroGuard:
    """_p_leg_win_prob_and_phi must not produce NaN when called with sigma=0 or tau=0."""

    def test_sigma_zero_returns_finite(self) -> None:
        """sigma=0 → sigma_sqrt_tau=0 → would divide-by-zero without guard."""
        result = _p_leg_win_prob_and_phi(
            reference_price=80_000.0,
            lo=79_000.0,
            hi=None,
            sigma=0.0,
            mu_eff=0.0,
            tau_yr=1.0 / 365.0,
        )
        assert result is not None, "Expected a tuple, got None"
        p_win, phi_d = result
        assert math.isfinite(p_win), f"p_win is not finite: {p_win}"
        assert math.isfinite(phi_d), f"phi_d is not finite: {phi_d}"

    def test_tau_zero_returns_finite(self) -> None:
        """tau_yr=0 → sigma_sqrt_tau=0 → would divide-by-zero without guard."""
        result = _p_leg_win_prob_and_phi(
            reference_price=80_000.0,
            lo=79_000.0,
            hi=None,
            sigma=0.01,
            mu_eff=0.0,
            tau_yr=0.0,
        )
        assert result is not None, "Expected a tuple, got None"
        p_win, phi_d = result
        assert math.isfinite(p_win), f"p_win is not finite: {p_win}"
        assert math.isfinite(phi_d), f"phi_d is not finite: {phi_d}"

    def test_both_zero_returns_finite(self) -> None:
        """sigma=0 AND tau_yr=0 must both be handled without NaN."""
        result = _p_leg_win_prob_and_phi(
            reference_price=80_000.0,
            lo=79_000.0,
            hi=None,
            sigma=0.0,
            mu_eff=0.0,
            tau_yr=0.0,
        )
        assert result is not None
        p_win, phi_d = result
        assert math.isfinite(p_win)
        assert math.isfinite(phi_d)

    def test_normal_inputs_unaffected(self) -> None:
        """Guard must not change normal-case output.

        Use ref close to lo boundary so p_win is in (0, 1).
        ref=79_100, lo=79_000, sigma=0.1, tau=1/365 → d ≈ small positive → p_win < 1.
        """
        result = _p_leg_win_prob_and_phi(
            reference_price=79_100.0,
            lo=79_000.0,
            hi=None,
            sigma=0.1,
            mu_eff=0.0,
            tau_yr=1.0 / 365.0,
        )
        assert result is not None
        p_win, phi_d = result
        assert 0.0 < p_win < 1.0
        assert phi_d > 0.0

    def test_two_sided_sigma_zero(self) -> None:
        """Two-sided region (lo, hi) with sigma=0 must also return finite values."""
        result = _p_leg_win_prob_and_phi(
            reference_price=80_000.0,
            lo=79_000.0,
            hi=81_000.0,
            sigma=0.0,
            mu_eff=0.0,
            tau_yr=1.0 / 365.0,
        )
        assert result is not None
        p_win, phi_d = result
        assert math.isfinite(p_win)
        assert math.isfinite(phi_d)


# ===========================================================================
# #42: bv_per_sample rename to bipower_sigma (behaviour-preserving pure rename)
# ===========================================================================


class TestBipowerSigmaRename:
    """The rename of bv_per_sample → bipower_sigma must be pure — no numeric
    change. We validate indirectly by checking the LM gate still fires on an
    obvious jump and stays off on a quiet window."""

    def _build_theta_strategy(self, **extra):
        from hlanalysis.backtest.core.registry import build as build_strategy

        params = dict(
            vol_lookback_seconds=3600,
            vol_sampling_dt_seconds=60,
            edge_buffer=0.02,
            fee_taker=0.0,
            half_spread_assumption=0.0,
            stop_loss_pct=None,
            max_position_usd=100.0,
            favorite_threshold=0.0,
            tte_min_seconds=0,
            tte_max_seconds=10**9,
            exit_edge_threshold=-0.01,
            take_profit_price=None,
            time_stop_seconds=0,
            drift_lookback_seconds=0,
            drift_blend=0.0,
        )
        params.update(extra)
        return build_strategy("v3_theta_harvester", params)

    def _theta_qv(self, *, expiry_ns: int) -> QuestionView:
        return QuestionView(
            question_idx=0,
            yes_symbol="YES",
            no_symbol="NO",
            strike=80_000.0,
            expiry_ns=expiry_ns,
            underlying="BTC",
            klass="priceBinary",
            period="1d",
            settled=False,
            kv=(),
        )

    def _theta_book(self, symbol: str, *, bid: float, ask: float) -> BookState:
        return BookState(
            symbol=symbol,
            bid_px=bid,
            bid_sz=100.0,
            ask_px=ask,
            ask_sz=100.0,
            last_trade_ts_ns=0,
            last_l2_ts_ns=0,
        )

    def test_lm_gate_holds_on_no_jump(self) -> None:
        """LM gate uses bipower_sigma (formerly bv_per_sample). When returns are
        quiet (last return << bipower σ relative to threshold), the gate HOLDs
        with 'lm_gate_no_jump'. This confirms the rename did not break the path."""
        strat = self._build_theta_strategy(
            lm_threshold=3.0,
        )
        now = 10**18
        expiry = now + 600 * 10**9
        qv = self._theta_qv(expiry_ns=expiry)
        books = {
            "YES": self._theta_book("YES", bid=0.95, ask=0.97),
            "NO": self._theta_book("NO", bid=0.03, ask=0.05),
        }

        # All returns quiet including the last one → LM stat is small → no_jump
        # bipower_sigma ≈ 0.001, lm_stat = 0.001/0.001 ≈ 1.0 < 3.0 threshold
        quiet = tuple([0.001] * 60)

        d_no_jump = strat.evaluate(
            question=qv,
            books=books,
            reference_price=85_000.0,
            recent_returns=quiet,
            recent_volume_usd=0.0,
            position=None,
            now_ns=now,
        )
        # LM gate should produce lm_gate_no_jump HOLD
        assert d_no_jump.action is Action.HOLD
        assert any("lm_gate_no_jump" in dg.message for dg in d_no_jump.diagnostics), (
            f"Expected lm_gate_no_jump diagnostic; got {d_no_jump.diagnostics}"
        )

    def test_bipower_sigma_value_matches_formula(self) -> None:
        """bipower_variation_sigma formula: sqrt((π/2)·mean(|r_i|·|r_{i+1}|))."""
        arr = np.array([0.01, -0.02, 0.03, -0.01, 0.02], dtype=np.float64)
        abs_r = np.abs(arr)
        expected_bpv = (np.pi / 2.0) * float(np.mean(abs_r[:-1] * abs_r[1:]))
        expected_sigma = math.sqrt(expected_bpv)
        got = bipower_variation_sigma(arr)
        assert math.isclose(got, expected_sigma, rel_tol=1e-9), (
            f"bipower_variation_sigma: expected {expected_sigma}, got {got}"
        )
