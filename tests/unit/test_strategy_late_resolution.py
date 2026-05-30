from __future__ import annotations

import dataclasses

import pytest

from hlanalysis.strategy.types import Action, Decision, Diagnostic, OrderIntent


def test_decision_is_frozen():
    d = Decision(action=Action.HOLD, intents=(), diagnostics=(Diagnostic("info", "noop"),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.action = Action.ENTER  # type: ignore[misc]


def test_order_intent_signed_size_is_required():
    intent = OrderIntent(
        question_idx=42,
        symbol="@30",
        side="buy",
        size=10.0,
        limit_price=0.95,
        cloid="hla-test",
        time_in_force="ioc",
    )
    assert intent.size > 0
    assert intent.side in ("buy", "sell")


from hlanalysis.strategy.base import Strategy


def test_strategy_abc_cannot_be_instantiated_directly():
    import pytest as _p
    with _p.raises(TypeError):
        Strategy()  # type: ignore[abstract]


import math

from hlanalysis.strategy.late_resolution import (
    LateResolutionConfig,
    LateResolutionStrategy,
)
from hlanalysis.strategy.types import Action, BookState, QuestionView


def _cfg(**overrides) -> LateResolutionConfig:
    base = dict(
        tte_min_seconds=60,
        tte_max_seconds=1800,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200.0,
        vol_max=0.5,
        max_position_usd=100.0,
        stop_loss_pct=10.0,
        max_strike_distance_pct=10.0,
        min_recent_volume_usd=1000.0,
        stale_data_halt_seconds=5,
    )
    base.update(overrides)
    return LateResolutionConfig(**base)


def _ref_book(symbol: str, ask: float, bid: float, *, ts_ns: int = 1_000) -> BookState:
    return BookState(
        symbol=symbol,
        bid_px=bid,
        bid_sz=100.0,
        ask_px=ask,
        ask_sz=100.0,
        last_trade_ts_ns=ts_ns,
        last_l2_ts_ns=ts_ns,
    )


def _q(strike: float = 80_000.0, expiry_ns: int = 0) -> QuestionView:
    return QuestionView(
        question_idx=42,
        yes_symbol="@30",
        no_symbol="@31",
        strike=strike,
        expiry_ns=expiry_ns,
        underlying="BTC",
        klass="priceBinary",
        period="1h",
    )


# --- Entry: happy path (BTC > strike by margin, YES under 1.0, in window, low vol) ---


def test_entry_yes_when_btc_above_strike_and_yes_book_extreme():
    now = 10_000_000_000_000  # ns
    expiry = now + 600 * 1_000_000_000  # 10 min TTE
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.06, bid=0.04, ts_ns=now - 100),
    }
    s = LateResolutionStrategy(_cfg())
    d = s.evaluate(
        question=q,
        books=books,
        reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60),  # ~very low vol
        recent_volume_usd=5_000.0,
        position=None,
        now_ns=now,
    )
    assert d.action is Action.ENTER
    assert len(d.intents) == 1
    intent = d.intents[0]
    assert intent.symbol == "@30"
    assert intent.side == "buy"
    assert intent.time_in_force == "ioc"
    # limit_price is the live top-of-book ask. v1 entries no longer walk the
    # book up to price_extreme_max — they consume only the top level (same as
    # v3.1 entries and the topup intent). The stale-ask sanity cap in the
    # gate loop preserves the protection the old ceiling provided.
    assert math.isclose(intent.limit_price, 0.96, rel_tol=1e-9)


def test_entry_no_when_btc_below_strike_and_no_book_extreme():
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.06, bid=0.04, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.96, bid=0.95, ts_ns=now - 100),
    }
    s = LateResolutionStrategy(_cfg())
    d = s.evaluate(
        question=q,
        books=books,
        reference_price=79_700.0,
        recent_returns=tuple([0.0001] * 60),
        recent_volume_usd=5_000.0,
        position=None,
        now_ns=now,
    )
    assert d.action is Action.ENTER
    assert d.intents[0].symbol == "@31"


# --- Hold: each gate, one at a time ---


def test_hold_when_tte_too_long():
    now = 10_000_000_000_000
    expiry = now + 3600 * 1_000_000_000  # 1h TTE > 30 min cap
    q = _q(expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.06, bid=0.04, ts_ns=now - 100),
    }
    s = LateResolutionStrategy(_cfg())
    d = s.evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.HOLD


def test_hold_when_tte_too_short():
    now = 10_000_000_000_000
    expiry = now + 30 * 1_000_000_000  # 30s TTE < 60s floor
    q = _q(expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.06, bid=0.04, ts_ns=now - 100),
    }
    d = LateResolutionStrategy(_cfg()).evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.HOLD


def test_hold_when_winning_leg_not_extreme():
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q()
    books = {
        # YES book is winning side (BTC > strike), but ask=0.80 < 0.95 threshold
        "@30": _ref_book("@30", ask=0.80, bid=0.78, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.22, bid=0.20, ts_ns=now - 100),
    }
    d = LateResolutionStrategy(_cfg()).evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.HOLD


def test_hold_when_distance_below_min():
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0)
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.04, bid=0.03, ts_ns=now - 100),
    }
    d = LateResolutionStrategy(_cfg()).evaluate(
        question=q, books=books,
        reference_price=80_050.0,  # only $50 above strike < $200 min
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.HOLD


def test_hold_when_realized_vol_above_cap():
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q()
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.04, bid=0.03, ts_ns=now - 100),
    }
    high_vol_returns = tuple([0.05, -0.05] * 30)  # huge swings
    d = LateResolutionStrategy(_cfg()).evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=high_vol_returns, recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.HOLD


def test_hold_when_book_stale():
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q()
    stale_ts = now - 10 * 1_000_000_000  # 10s old > 5s halt
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=stale_ts),
        "@31": _ref_book("@31", ask=0.04, bid=0.03, ts_ns=stale_ts),
    }
    d = LateResolutionStrategy(_cfg()).evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.HOLD


def test_hold_when_position_already_held():
    from hlanalysis.strategy.types import Position
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q()
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.04, bid=0.03, ts_ns=now - 100),
    }
    pos = Position(
        question_idx=q.question_idx, symbol="@30", qty=10.0, avg_entry=0.95,
        stop_loss_price=0.855, last_update_ts_ns=now - 1_000_000,
    )
    d = LateResolutionStrategy(_cfg()).evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=pos, now_ns=now,
    )
    # No re-entry; exit logic runs separately (Task 6)
    assert d.action is not Action.ENTER


def test_exit_signal_when_question_is_settled():
    from hlanalysis.strategy.types import Position
    now = 10_000_000_000_000
    q = QuestionView(
        question_idx=42, yes_symbol="@30", no_symbol="@31",
        strike=80_000.0, expiry_ns=now - 1_000_000,
        underlying="BTC", klass="priceBinary", period="1h",
        settled=True, settled_side="yes",
    )
    books = {
        "@30": _ref_book("@30", ask=1.0, bid=1.0, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.0, bid=0.0, ts_ns=now - 100),
    }
    pos = Position(
        question_idx=q.question_idx, symbol="@30", qty=10.0, avg_entry=0.95,
        stop_loss_price=0.855, last_update_ts_ns=now - 1_000_000,
    )
    d = LateResolutionStrategy(_cfg()).evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=pos, now_ns=now,
    )
    assert d.action is Action.EXIT
    # Settlement-driven exit: zero intents — engine marks the position closed
    # at the venue's settlement value, no order needed.
    assert d.intents == ()


def test_exit_intent_when_price_below_stop_loss():
    from hlanalysis.strategy.types import Position
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.84, bid=0.83, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.16, bid=0.15, ts_ns=now - 100),
    }
    pos = Position(
        question_idx=q.question_idx, symbol="@30", qty=10.0, avg_entry=0.95,
        stop_loss_price=0.855, last_update_ts_ns=now - 1_000_000,
    )
    d = LateResolutionStrategy(_cfg()).evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=pos, now_ns=now,
    )
    assert d.action is Action.EXIT
    assert len(d.intents) == 1
    intent = d.intents[0]
    assert intent.symbol == "@30"
    assert intent.side == "sell"
    assert intent.reduce_only is True
    assert intent.time_in_force == "ioc"


# --- _winning_region: per-leg (lo, hi) derivation for binary + bucket ---

from hlanalysis.strategy.late_resolution import _winning_region  # noqa: E402


def _binary_qv() -> QuestionView:
    return QuestionView(
        question_idx=1, yes_symbol="@30", no_symbol="@31",
        strike=80_000.0, expiry_ns=0,
        underlying="BTC", klass="priceBinary", period="1d",
        leg_symbols=("@30", "@31"),
    )


def _bucket_qv() -> QuestionView:
    # 2 thresholds → 3 outcomes (lowest, middle, highest), 2 legs each.
    return QuestionView(
        question_idx=2, yes_symbol="@40", no_symbol="@41",
        strike=float("nan"), expiry_ns=0,
        underlying="BTC", klass="priceBucket", period="1d",
        leg_symbols=("@40", "@41", "@42", "@43", "@44", "@45"),
        kv=(("class", "priceBucket"), ("underlying", "BTC"),
            ("priceThresholds", "77991,81174")),
    )


def test_winning_region_binary_yes_is_strike_to_inf():
    lo, hi = _winning_region(_binary_qv(), "@30")
    assert lo == 80_000.0 and hi is None


def test_winning_region_binary_no_is_neg_inf_to_strike():
    lo, hi = _winning_region(_binary_qv(), "@31")
    assert lo is None and hi == 80_000.0


def test_winning_region_bucket_lowest_yes_is_below_first_threshold():
    lo, hi = _winning_region(_bucket_qv(), "@40")
    assert lo is None and hi == 77_991.0


def test_winning_region_bucket_middle_yes_is_between_thresholds():
    lo, hi = _winning_region(_bucket_qv(), "@42")
    assert lo == 77_991.0 and hi == 81_174.0


def test_winning_region_bucket_highest_yes_is_above_last_threshold():
    lo, hi = _winning_region(_bucket_qv(), "@44")
    assert lo == 81_174.0 and hi is None


def test_winning_region_bucket_no_legs_invert_yes_region():
    # NO leg of the lowest bucket wins when BTC >= 77_991 → region is [thr[0], +∞).
    lo, hi = _winning_region(_bucket_qv(), "@41")
    assert lo == 77_991.0 and hi is None
    # NO of the highest bucket wins when BTC <= 81_174 → (-∞, thr[-1]].
    lo, hi = _winning_region(_bucket_qv(), "@45")
    assert lo is None and hi == 81_174.0


def test_winning_region_unknown_symbol_returns_none_none():
    lo, hi = _winning_region(_bucket_qv(), "@999")
    assert lo is None and hi is None


# --- _safety_d_for_region: leg-aware safety distance in σ-units ---

from hlanalysis.strategy.late_resolution import _safety_d_for_region  # noqa: E402


def test_safety_d_lower_bounded_favorable():
    # Binary YES: lo=strike, hi=None. BTC above strike → positive d.
    d = _safety_d_for_region(
        ref_price=80_300.0, lo=80_000.0, hi=None,
        sigma_window=1.0, mu=0.0, tte_min=0.0, drift_aware=False,
    )
    assert d is not None
    assert math.isclose(d, math.log(80_300.0 / 80_000.0), rel_tol=1e-9)


def test_safety_d_upper_bounded_favorable():
    # Binary NO: lo=None, hi=strike. BTC below strike → positive d.
    d = _safety_d_for_region(
        ref_price=79_700.0, lo=None, hi=80_000.0,
        sigma_window=1.0, mu=0.0, tte_min=0.0, drift_aware=False,
    )
    assert d is not None
    assert math.isclose(d, math.log(80_000.0 / 79_700.0), rel_tol=1e-9)


def test_safety_d_lower_bounded_adverse_is_negative():
    # YES leg held but BTC dropped below strike: d < 0 → exit_safety_d fires.
    d = _safety_d_for_region(
        ref_price=79_700.0, lo=80_000.0, hi=None,
        sigma_window=1.0, mu=0.0, tte_min=0.0, drift_aware=False,
    )
    assert d is not None
    assert d < 0


def test_safety_d_middle_bucket_uses_nearer_boundary():
    # ref in middle of bucket; d = min(ln(ref/lo), ln(hi/ref)) / σ.
    ref = 79_500.0
    lo, hi = 77_991.0, 81_174.0
    expected = min(math.log(ref / lo), math.log(hi / ref))
    d = _safety_d_for_region(
        ref_price=ref, lo=lo, hi=hi,
        sigma_window=1.0, mu=0.0, tte_min=0.0, drift_aware=False,
    )
    assert d is not None
    assert math.isclose(d, expected, rel_tol=1e-9)


def test_safety_d_middle_bucket_adverse_when_outside_region():
    # ref below lo → ln(ref/lo) < 0 → min < 0.
    d = _safety_d_for_region(
        ref_price=76_000.0, lo=77_991.0, hi=81_174.0,
        sigma_window=1.0, mu=0.0, tte_min=0.0, drift_aware=False,
    )
    assert d is not None
    assert d < 0


def test_safety_d_drift_lower_bounded_adds_mu_tau():
    # (lo, None) adverse direction is downward; positive μ adds to safety.
    d = _safety_d_for_region(
        ref_price=80_300.0, lo=80_000.0, hi=None,
        sigma_window=1.0, mu=0.001, tte_min=10.0, drift_aware=True,
    )
    expected = math.log(80_300.0 / 80_000.0) + 0.001 * 10.0
    assert d is not None
    assert math.isclose(d, expected, rel_tol=1e-9)


def test_safety_d_drift_upper_bounded_subtracts_mu_tau():
    # (None, hi) adverse direction is upward; positive μ subtracts from safety.
    d = _safety_d_for_region(
        ref_price=79_700.0, lo=None, hi=80_000.0,
        sigma_window=1.0, mu=0.001, tte_min=10.0, drift_aware=True,
    )
    expected = math.log(80_000.0 / 79_700.0) - 0.001 * 10.0
    assert d is not None
    assert math.isclose(d, expected, rel_tol=1e-9)


def test_safety_d_drift_ignored_for_middle_bucket():
    # Two-sided region has no single adverse direction → drift dropped.
    d_no_drift = _safety_d_for_region(
        ref_price=79_500.0, lo=77_991.0, hi=81_174.0,
        sigma_window=1.0, mu=0.0, tte_min=0.0, drift_aware=False,
    )
    d_with_drift = _safety_d_for_region(
        ref_price=79_500.0, lo=77_991.0, hi=81_174.0,
        sigma_window=1.0, mu=0.01, tte_min=100.0, drift_aware=True,
    )
    assert d_no_drift is not None and d_with_drift is not None
    assert math.isclose(d_no_drift, d_with_drift, rel_tol=1e-9)


def test_safety_d_unbounded_returns_none():
    # NO leg of a middle bucket → caller passed (None, None). No gate available.
    d = _safety_d_for_region(
        ref_price=80_000.0, lo=None, hi=None,
        sigma_window=1.0, mu=0.0, tte_min=0.0, drift_aware=False,
    )
    assert d is None


# --- Bucket entry: safety_d gate fires on bucket legs (was binary-only) ---


def _bucket_q(expiry_ns: int) -> QuestionView:
    # 2 thresholds → 3 outcomes; legs are (lowest YES, lowest NO, middle YES,
    # middle NO, highest YES, highest NO).
    return QuestionView(
        question_idx=99, yes_symbol="@40", no_symbol="@41",
        strike=float("nan"), expiry_ns=expiry_ns,
        underlying="BTC", klass="priceBucket", period="1d",
        leg_symbols=("@40", "@41", "@42", "@43", "@44", "@45"),
        kv=(("class", "priceBucket"), ("underlying", "BTC"),
            ("priceThresholds", "77991,81174")),
    )


def _bucket_cfg(**overrides) -> LateResolutionConfig:
    base = dict(
        tte_min_seconds=60,
        tte_max_seconds=3600,
        price_extreme_threshold=0.90,
        distance_from_strike_usd_min=0.0,
        vol_max=0.5,
        max_position_usd=100.0,
        stop_loss_pct=10.0,
        max_strike_distance_pct=50.0,
        min_recent_volume_usd=0.0,
        stale_data_halt_seconds=5,
        price_extreme_max=0.995,
        min_safety_d=1.5,
        vol_lookback_seconds=3600,
    )
    base.update(overrides)
    return LateResolutionConfig(**base)


def test_bucket_entry_picks_middle_yes_leg_when_safe():
    # BTC sitting in the middle bucket, low vol → safety_d easily clears 1.5.
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _bucket_q(expiry)
    # Only the middle YES leg is above the price-extreme threshold.
    books = {
        "@40": _ref_book("@40", ask=0.04, bid=0.03, ts_ns=now - 100),
        "@41": _ref_book("@41", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@42": _ref_book("@42", ask=0.95, bid=0.94, ts_ns=now - 100),
        "@43": _ref_book("@43", ask=0.05, bid=0.04, ts_ns=now - 100),
        "@44": _ref_book("@44", ask=0.04, bid=0.03, ts_ns=now - 100),
        "@45": _ref_book("@45", ask=0.96, bid=0.95, ts_ns=now - 100),
    }
    d = LateResolutionStrategy(_bucket_cfg()).evaluate(
        question=q, books=books,
        reference_price=79_500.0,
        recent_returns=tuple([0.0001, -0.0001] * 30),
        recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.ENTER
    assert d.intents[0].symbol == "@42"  # middle YES


def test_bucket_entry_blocked_by_safety_d_when_btc_near_boundary():
    # BTC near the upper boundary of the middle bucket + high vol → safety_d < 1.5.
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _bucket_q(expiry)
    books = {
        "@40": _ref_book("@40", ask=0.04, bid=0.03, ts_ns=now - 100),
        "@41": _ref_book("@41", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@42": _ref_book("@42", ask=0.95, bid=0.94, ts_ns=now - 100),
        "@43": _ref_book("@43", ask=0.05, bid=0.04, ts_ns=now - 100),
        "@44": _ref_book("@44", ask=0.04, bid=0.03, ts_ns=now - 100),
        "@45": _ref_book("@45", ask=0.96, bid=0.95, ts_ns=now - 100),
    }
    d = LateResolutionStrategy(_bucket_cfg()).evaluate(
        question=q, books=books,
        reference_price=81_100.0,  # only ~$74 below upper bound 81_174
        recent_returns=tuple([0.005, -0.005] * 30),  # high vol
        recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.HOLD


# --- Bucket exit: exit_safety_d fires when BTC drifts out of held bucket ---


def test_bucket_exit_safety_d_fires_when_btc_exits_middle_bucket():
    from hlanalysis.strategy.types import Position
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _bucket_q(expiry)
    books = {
        "@42": _ref_book("@42", ask=0.92, bid=0.91, ts_ns=now - 100),
        "@43": _ref_book("@43", ask=0.10, bid=0.08, ts_ns=now - 100),
    }
    pos = Position(
        question_idx=q.question_idx, symbol="@42", qty=10.0, avg_entry=0.95,
        stop_loss_price=0.855, last_update_ts_ns=now - 1_000_000,
    )
    cfg = _bucket_cfg(exit_safety_d=1.0)
    # BTC well above the upper bucket boundary 81_174 → middle-bucket safety_d
    # is strongly negative → exit fires.
    d = LateResolutionStrategy(cfg).evaluate(
        question=q, books=books,
        reference_price=82_500.0,
        recent_returns=tuple([0.0001, -0.0001] * 30),
        recent_volume_usd=5_000.0,
        position=pos, now_ns=now,
    )
    assert d.action is Action.EXIT
    assert d.intents[0].symbol == "@42"
    assert d.intents[0].side == "sell"
    assert d.intents[0].reduce_only is True


# --- Entry size respects the risk gate's notional = size * limit_price cap ---


def test_entry_notional_never_exceeds_max_position_usd():
    # ask just below the limit cap is the bug regime: floor(size_usd/ask) on
    # 0.01 contracts yields ~100.10 contracts, then size * limit_price (1.0) =
    # 100.10 > cap 100, and the risk gate vetoes every order in a tight loop.
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.99899, bid=0.99, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.02, bid=0.01, ts_ns=now - 100),
    }
    # price_extreme_max=1.0 (default) — matches prod config.
    cfg = _cfg(
        price_extreme_threshold=0.95,
        max_position_usd=100.0,
        min_recent_volume_usd=0.0,
    )
    d = LateResolutionStrategy(cfg).evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.ENTER
    intent = d.intents[0]
    notional = intent.size * intent.limit_price
    assert notional <= cfg.max_position_usd, (
        f"notional {notional} would trip risk gate cap {cfg.max_position_usd}"
    )


# --- Targeted near-strike low-ask size cap ---


def _cap_books(yes_ask: float, no_ask: float, ts_ns: int) -> dict[str, BookState]:
    return {
        "@30": _ref_book("@30", ask=yes_ask, bid=yes_ask - 0.01, ts_ns=ts_ns),
        "@31": _ref_book("@31", ask=no_ask, bid=no_ask - 0.01, ts_ns=ts_ns),
    }


def test_size_cap_disabled_by_default_no_change_in_size():
    # Near-strike low-ask entry with default cap (pct=0) should produce the
    # full size: floor(100 / 0.86 * 100)/100 = 116.27 contracts. Notional
    # = 116.27 * 0.86 = $99.99, within the $100 cap.
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = _cap_books(yes_ask=0.86, no_ask=0.14, ts_ns=now - 100)
    cfg = _cfg(price_extreme_threshold=0.80, min_recent_volume_usd=0.0)
    d = LateResolutionStrategy(cfg).evaluate(
        question=q, books=books, reference_price=80_100.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.ENTER
    intent = d.intents[0]
    assert intent.symbol == "@30"
    assert math.isclose(intent.size, 116.27, abs_tol=0.01)


def test_size_cap_halves_size_on_near_strike_low_ask_entry():
    # BTC at 80_100, strike 80_000 → dist_pct = 0.125% < 1.5%.
    # YES ask 0.86 < 0.88 (min_ask). pct=0.5 → scale 0.5 → size 50.
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = _cap_books(yes_ask=0.86, no_ask=0.14, ts_ns=now - 100)
    cfg = _cfg(
        price_extreme_threshold=0.80,
        min_recent_volume_usd=0.0,
        size_cap_near_strike_pct=0.5,
        size_cap_max_dist_pct=1.5,
        size_cap_min_ask=0.88,
    )
    d = LateResolutionStrategy(cfg).evaluate(
        question=q, books=books, reference_price=80_100.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.ENTER
    intent = d.intents[0]
    assert intent.symbol == "@30"
    # Capped size_usd = 50, sizing_px = ask 0.86 → floor(50/0.86 * 100)/100 = 58.13.
    assert math.isclose(intent.size, 58.13, abs_tol=0.01)


def test_size_cap_does_not_apply_when_ask_above_min_ask():
    # Same near-strike setup but ask 0.95 ≥ 0.88 → cap does NOT apply.
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = _cap_books(yes_ask=0.95, no_ask=0.05, ts_ns=now - 100)
    cfg = _cfg(
        price_extreme_threshold=0.80,
        min_recent_volume_usd=0.0,
        size_cap_near_strike_pct=0.5,
        size_cap_max_dist_pct=1.5,
        size_cap_min_ask=0.88,
    )
    d = LateResolutionStrategy(cfg).evaluate(
        question=q, books=books, reference_price=80_100.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.ENTER
    intent = d.intents[0]
    assert intent.symbol == "@30"
    # ask 0.95 → floor(100/0.95 * 100)/100 = 105.26.
    assert math.isclose(intent.size, 105.26, abs_tol=0.01)
    # No size_cap diagnostic emitted.
    assert not any(d_.message == "size_cap_near_strike" for d_ in d.diagnostics)


def test_size_cap_does_not_apply_when_far_from_strike():
    # BTC at 81_500 → dist_pct = 1.875% > 1.5% → cap does NOT apply.
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = _cap_books(yes_ask=0.86, no_ask=0.14, ts_ns=now - 100)
    cfg = _cfg(
        price_extreme_threshold=0.80,
        min_recent_volume_usd=0.0,
        size_cap_near_strike_pct=0.5,
        size_cap_max_dist_pct=1.5,
        size_cap_min_ask=0.88,
    )
    d = LateResolutionStrategy(cfg).evaluate(
        question=q, books=books, reference_price=81_500.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.ENTER
    intent = d.intents[0]
    assert intent.symbol == "@30"
    # ask 0.86 → floor(100/0.86 * 100)/100 = 116.27.
    assert math.isclose(intent.size, 116.27, abs_tol=0.01)
    assert not any(d_.message == "size_cap_near_strike" for d_ in d.diagnostics)


def test_size_cap_works_on_no_leg_for_binary():
    # BTC just below strike → NO wins. NO ask 0.86 < 0.88 → cap applies → size halved.
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = _cap_books(yes_ask=0.14, no_ask=0.86, ts_ns=now - 100)
    cfg = _cfg(
        price_extreme_threshold=0.80,
        min_recent_volume_usd=0.0,
        size_cap_near_strike_pct=0.5,
        size_cap_max_dist_pct=1.5,
        size_cap_min_ask=0.88,
    )
    d = LateResolutionStrategy(cfg).evaluate(
        question=q, books=books, reference_price=79_900.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.ENTER
    intent = d.intents[0]
    assert intent.symbol == "@31"
    # Capped size_usd = 50, ask 0.86 → floor(50/0.86 * 100)/100 = 58.13.
    assert math.isclose(intent.size, 58.13, abs_tol=0.01)


def test_size_cap_emits_diagnostic_when_active():
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = _cap_books(yes_ask=0.86, no_ask=0.14, ts_ns=now - 100)
    cfg = _cfg(
        price_extreme_threshold=0.80,
        min_recent_volume_usd=0.0,
        size_cap_near_strike_pct=0.5,
        size_cap_max_dist_pct=1.5,
        size_cap_min_ask=0.88,
    )
    d = LateResolutionStrategy(cfg).evaluate(
        question=q, books=books, reference_price=80_100.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.ENTER
    cap_diags = [d_ for d_ in d.diagnostics if d_.message == "size_cap_near_strike"]
    assert len(cap_diags) == 1
    kv = dict(cap_diags[0].fields)
    assert "dist_pct" in kv and "ask" in kv and "scale_after" in kv


# --- 2026-05-19 churn fixes: bid-based gate, bid-notional sanity ---


def test_ask_based_gate_still_default_when_use_bid_for_entry_gate_false():
    """Regression guard: without the new flag, late_resolution should behave
    exactly as before — gate on ask, accept the stale-quote pattern that
    burned production today."""
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    # Wide-spread book: ask 0.999 (the "stale" quote), bid 0.55. The legacy
    # ask-based gate sees ask >= 0.95 → enter. The new bid-based gate would
    # reject because bid < 0.95.
    books = {
        "@30": _ref_book("@30", ask=0.999, bid=0.55, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.06, bid=0.04, ts_ns=now - 100),
    }
    s = LateResolutionStrategy(_cfg())  # flag default False
    d = s.evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.ENTER  # legacy behavior preserved


def test_bid_based_gate_rejects_wide_spread_stale_ask():
    """The key fix: with use_bid_for_entry_gate=True, a wide-spread book
    where ask is stale-high but bid has caved should NOT trigger an entry.
    Exactly the regime that drove the 7-cycle churn on #601 today."""
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.999, bid=0.55, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.06, bid=0.04, ts_ns=now - 100),
    }
    s = LateResolutionStrategy(_cfg(use_bid_for_entry_gate=True))
    d = s.evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.HOLD
    assert any(diag.message == "no_extreme_leg" for diag in d.diagnostics)


def test_bid_based_gate_allows_tight_spread_real_favourite():
    """Symmetric: when both sides are at the favourite level (a real, deep
    market consensus), the bid-based gate should still allow entry. The
    threshold semantics are preserved on tight books — only the stale-ask
    artefact is filtered."""
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = {
        # Tight market at the favourite level — bid=0.96, ask=0.97. The
        # bid-based gate sees bid >= 0.95 → enter.
        "@30": _ref_book("@30", ask=0.97, bid=0.96, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.04, bid=0.03, ts_ns=now - 100),
    }
    s = LateResolutionStrategy(_cfg(use_bid_for_entry_gate=True))
    d = s.evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.ENTER
    assert d.intents[0].symbol == "@30"


def test_min_bid_notional_blocks_spoof_size_1_bid():
    """A bid of 0.95 × 1 share passes the numeric bid threshold but is a 95¢
    stake — clearly not real buying interest. The min_bid_notional_usd gate
    must reject it."""
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    spoof_book = BookState(
        symbol="@30", bid_px=0.95, bid_sz=1.0, ask_px=0.97, ask_sz=100.0,
        last_trade_ts_ns=now - 100, last_l2_ts_ns=now - 100,
    )
    books = {
        "@30": spoof_book,
        "@31": _ref_book("@31", ask=0.04, bid=0.03, ts_ns=now - 100),
    }
    s = LateResolutionStrategy(_cfg(
        use_bid_for_entry_gate=True, min_bid_notional_usd=10.0,
    ))
    d = s.evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    assert d.action is Action.HOLD


def test_stale_ask_cap_rejects_ask_above_price_extreme_max_when_using_bid_gate():
    """With use_bid_for_entry_gate=true, a stale-high ask (0.9995) can pair
    with a fresh bid (0.95) — the bid passes the gate but the IOC would now
    fill at the stale 0.9995 ask (limit_price=ask). The explicit ask cap
    introduced alongside the top-of-book limit change rejects this case,
    preserving the protection the old limit_price=price_extreme_max gave."""
    from hlanalysis.strategy.types import BookState
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    # YES has fresh bid 0.95 (in [0.85, 0.99]) but stale-high ask 0.9995 > 0.99.
    yes_book = BookState(
        symbol="@30", bid_px=0.95, bid_sz=100.0, ask_px=0.9995, ask_sz=50.0,
        last_trade_ts_ns=now - 100, last_l2_ts_ns=now - 100,
    )
    books = {
        "@30": yes_book,
        "@31": _ref_book("@31", ask=0.04, bid=0.03, ts_ns=now - 100),
    }
    s = LateResolutionStrategy(_cfg(
        use_bid_for_entry_gate=True,
        price_extreme_threshold=0.85,
        price_extreme_max=0.99,
    ))
    d = s.evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=None, now_ns=now,
    )
    # Neither leg eligible (YES rejected by ask cap, NO ask 0.04 < 0.85 threshold).
    assert d.action is Action.HOLD
    assert any(diag.message == "no_extreme_leg" for diag in d.diagnostics)


# ---------------------------------------------------------------------------
# Position-topup tests (2026-05-20): partial-fill recovery on thin HL HIP-4
# books. Symmetric with theta_harvester's topup suite but uses v1's gates.
# ---------------------------------------------------------------------------


def _topup_cfg(**over):
    """v1 cfg with stops/safety/cooldowns relaxed so the entry gates don't
    interfere with isolated topup scenarios."""
    base = dict(
        tte_min_seconds=60,
        tte_max_seconds=1800,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200.0,
        vol_max=0.5,
        max_position_usd=100.0,
        stop_loss_pct=1e9,            # disabled — focus on topup
        max_strike_distance_pct=10.0,
        min_recent_volume_usd=1000.0,
        stale_data_halt_seconds=5,
        min_safety_d=0.0,             # disable safety_d for clarity
        exit_safety_d=0.0,            # no exit_safety_d; isolate topup
    )
    base.update(over)
    return LateResolutionConfig(**base)


def test_topup_emits_enter_when_held_under_target():
    """Position $48 ntl on $100 target → 52% shortfall ≥ 20% threshold; all
    v1 gates pass → ENTER with limit_price=current_ask."""
    from hlanalysis.strategy.types import Position
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.06, bid=0.04, ts_ns=now - 100),
    }
    # qty 50 × ask 0.96 = $48 ntl on $100 target → 52% shortfall.
    pos = Position(
        question_idx=q.question_idx, symbol="@30", qty=50.0, avg_entry=0.96,
        stop_loss_price=0.0, last_update_ts_ns=now - 1_000_000,
    )
    s = LateResolutionStrategy(_topup_cfg())
    d = s.evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=pos, now_ns=now,
    )
    assert d.action is Action.ENTER
    assert d.intents[0].symbol == "@30"
    assert d.intents[0].side == "buy"
    # shortfall $52 / ask 0.96 = floor(52/0.96 * 100) / 100 = 54.16 contracts.
    assert math.isclose(d.intents[0].size, 54.16, abs_tol=0.01)
    assert math.isclose(d.intents[0].limit_price, 0.96, abs_tol=1e-9)
    assert any(diag.message == "topup_emit" for diag in d.diagnostics)


def test_topup_holds_when_shortfall_below_threshold():
    """Held $96 ntl on $100 target → 4% shortfall < 20% threshold → HOLD
    with the legacy have_position diag (no topup attempt made)."""
    from hlanalysis.strategy.types import Position
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.06, bid=0.04, ts_ns=now - 100),
    }
    pos = Position(
        question_idx=q.question_idx, symbol="@30", qty=100.0, avg_entry=0.96,
        stop_loss_price=0.0, last_update_ts_ns=now - 1_000_000,
    )
    s = LateResolutionStrategy(_topup_cfg())
    d = s.evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=pos, now_ns=now,
    )
    assert d.action is Action.HOLD
    # The topup attempt was skipped pre-gate-check; legacy have_position diag.
    assert any(diag.message == "have_position" for diag in d.diagnostics)


def test_topup_holds_when_topup_notional_below_min():
    """max_position_usd=12, held qty 4 × 0.96 = $3.84 → shortfall ~$8 (67%)
    triggers attempt; topup_size = floor(8.16/0.96*100)/100 = 8.50 contracts →
    $8.16 ntl < $11 floor → HOLD with below_min_notional skip reason."""
    from hlanalysis.strategy.types import Position
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.06, bid=0.04, ts_ns=now - 100),
    }
    pos = Position(
        question_idx=q.question_idx, symbol="@30", qty=4.0, avg_entry=0.96,
        stop_loss_price=0.0, last_update_ts_ns=now - 1_000_000,
    )
    s = LateResolutionStrategy(_topup_cfg(max_position_usd=12.0))
    d = s.evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=pos, now_ns=now,
    )
    assert d.action is Action.HOLD
    skip = next(diag for diag in d.diagnostics if diag.message == "topup_skip")
    assert any(("reason", "below_min_notional") == kv for kv in skip.fields)


def test_topup_holds_when_entry_gate_now_fails():
    """Recent vol crosses vol_max ceiling — entry gate fails on topup attempt."""
    from hlanalysis.strategy.types import Position
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.06, bid=0.04, ts_ns=now - 100),
    }
    pos = Position(
        question_idx=q.question_idx, symbol="@30", qty=30.0, avg_entry=0.96,
        stop_loss_price=0.0, last_update_ts_ns=now - 1_000_000,
    )
    s = LateResolutionStrategy(_topup_cfg(vol_max=0.001))
    # Alternating returns produce non-zero stdev → vol exceeds tiny vol_max.
    spiky = tuple((0.05 if i % 2 == 0 else -0.05) for i in range(60))
    d = s.evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=spiky,
        recent_volume_usd=5_000.0,
        position=pos, now_ns=now,
    )
    assert d.action is Action.HOLD
    skip = next(diag for diag in d.diagnostics if diag.message == "topup_skip")
    reason_kv = next(kv for kv in skip.fields if kv[0] == "reason")
    assert reason_kv[1].startswith("gate_failed:")


def test_topup_holds_when_chosen_leg_differs_from_held():
    """Hold @30 (YES) under-filled, but @31 (NO) now has a higher gate price
    than @30 — entry picks @31; topup must NOT add to wrong leg.
    Set @31 ask=0.99 (above @30's 0.97) so entry's max-gate-price logic
    picks @31."""
    from hlanalysis.strategy.types import Position
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.97, bid=0.96, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.99, bid=0.98, ts_ns=now - 100),
    }
    pos = Position(
        question_idx=q.question_idx, symbol="@30", qty=30.0, avg_entry=0.97,
        stop_loss_price=0.0, last_update_ts_ns=now - 1_000_000,
    )
    s = LateResolutionStrategy(_topup_cfg())
    d = s.evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=pos, now_ns=now,
    )
    assert d.action is Action.HOLD
    skip = next(diag for diag in d.diagnostics if diag.message == "topup_skip")
    assert any(("reason", "leg_changed") == kv for kv in skip.fields)


def test_exit_takes_precedence_over_topup():
    """Stop-loss bid breach must close the position even when it's under-filled
    (which would otherwise trigger a topup)."""
    from hlanalysis.strategy.types import Position
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    # Bid below stop_loss_price triggers exit.
    books = {
        "@30": _ref_book("@30", ask=0.84, bid=0.83, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.16, bid=0.15, ts_ns=now - 100),
    }
    pos = Position(
        question_idx=q.question_idx, symbol="@30", qty=50.0, avg_entry=0.95,
        stop_loss_price=0.855, last_update_ts_ns=now - 1_000_000,
    )
    s = LateResolutionStrategy(_topup_cfg(stop_loss_pct=10.0))
    d = s.evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=pos, now_ns=now,
    )
    assert d.action is Action.EXIT
    assert d.intents[0].reduce_only is True


def test_topup_disabled_via_config_keeps_legacy_have_position():
    from hlanalysis.strategy.types import Position
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.06, bid=0.04, ts_ns=now - 100),
    }
    pos = Position(
        question_idx=q.question_idx, symbol="@30", qty=50.0, avg_entry=0.96,
        stop_loss_price=0.0, last_update_ts_ns=now - 1_000_000,
    )
    s = LateResolutionStrategy(_topup_cfg(topup_enabled=False))
    d = s.evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=pos, now_ns=now,
    )
    assert d.action is Action.HOLD
    assert any(diag.message == "have_position" for diag in d.diagnostics)
    assert not any(diag.message.startswith("topup_") for diag in d.diagnostics)


def test_topup_size_floors_to_two_decimals():
    """Sizing must use floor((shortfall/ask)*100)/100 — never overshoot target."""
    import math as _m
    from hlanalysis.strategy.types import Position
    now = 10_000_000_000_000
    expiry = now + 600 * 1_000_000_000
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.97, bid=0.96, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.04, bid=0.03, ts_ns=now - 100),
    }
    # held 50 × 0.97 = $48.5 ntl on $100 → shortfall $51.5; @0.97 →
    # floor(51.5/0.97 * 100)/100 = floor(5309.27)/100 = 53.09 contracts.
    pos = Position(
        question_idx=q.question_idx, symbol="@30", qty=50.0, avg_entry=0.97,
        stop_loss_price=0.0, last_update_ts_ns=now - 1_000_000,
    )
    s = LateResolutionStrategy(_topup_cfg())
    d = s.evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=tuple([0.0001] * 60), recent_volume_usd=5_000.0,
        position=pos, now_ns=now,
    )
    assert d.action is Action.ENTER
    assert _m.isclose(d.intents[0].size, 53.09, abs_tol=0.01)


# --- Per-class allowlist config overrides ---


def _q_bucket(expiry_ns: int) -> QuestionView:
    # Minimal priceBucket question; we only need TTE to be inside the bucket
    # window to prove the per-class TTE override fires. legs=() forces the
    # "no_legs" diagnostic on the *next* gate, which is what we assert on.
    return QuestionView(
        question_idx=99,
        yes_symbol="",
        no_symbol="",
        strike=0.0,
        expiry_ns=expiry_ns,
        underlying="BTC",
        klass="priceBucket",
        period="1d",
    )


def test_evaluate_uses_per_class_cfg_when_provided():
    """priceBucket questions must read tte_max from cfg_by_class['priceBucket'],
    not from the default cfg. Regression: build_late_resolution_config used to
    read only cfg.defaults, silently dropping per-class allowlist overrides."""
    default = _cfg(tte_max_seconds=1800)        # 30min — same as existing tests
    bucket = _cfg(tte_max_seconds=86400)        # 24h — what the YAML allowlist sets
    s = LateResolutionStrategy(default, cfg_by_class={"priceBucket": bucket})

    now = 10_000_000_000_000
    expiry_8h = now + 8 * 3600 * 1_000_000_000  # 8h TTE — outside default, inside bucket

    # priceBucket: TTE override applies → should NOT be tte_out_of_window.
    d_bucket = s.evaluate(
        question=_q_bucket(expiry_ns=expiry_8h), books={}, reference_price=80_000.0,
        recent_returns=(), recent_volume_usd=0.0, position=None, now_ns=now,
    )
    assert d_bucket.action is Action.HOLD
    assert not any(diag.message == "tte_out_of_window" for diag in d_bucket.diagnostics), (
        f"priceBucket should clear TTE gate with per-class override; got {d_bucket.diagnostics!r}"
    )

    # priceBinary at the same 8h TTE: no override → falls back to default tte_max=1800
    # → must still be tte_out_of_window.
    d_binary = s.evaluate(
        question=_q(strike=80_000.0, expiry_ns=expiry_8h), books={}, reference_price=80_000.0,
        recent_returns=(), recent_volume_usd=0.0, position=None, now_ns=now,
    )
    assert d_binary.action is Action.HOLD
    assert any(diag.message == "tte_out_of_window" for diag in d_binary.diagnostics)


def test_fee_model_defaults_preserve_legacy_hl_behavior():
    """fee_model defaults to 'flat' and fee_rate to 0.0 so existing HL slot
    configs (which don't set either) load bit-identical."""
    cfg = _cfg()
    assert cfg.fee_model == "flat"
    assert cfg.fee_rate == 0.0


def test_build_v1_late_resolution_threads_pm_binary_fee_fields():
    """v1 on Polymarket: the registered builder must propagate fee_model and
    fee_rate from params into LateResolutionConfig so a v1_pm slot can declare
    the PM fee curve in the same shape as v31_pm."""
    from hlanalysis.strategy.late_resolution import build_v1_late_resolution

    params = dict(
        tte_min_seconds=0,
        tte_max_seconds=7200,
        price_extreme_threshold=0.85,
        price_extreme_max=0.99,
        distance_from_strike_usd_min=0,
        vol_max=100,
        stop_loss_pct=None,
        max_position_usd=50.0,
        fee_model="pm_binary",
        fee_rate=0.07,
    )
    strat = build_v1_late_resolution(params)
    assert strat.cfg.fee_model == "pm_binary"
    assert strat.cfg.fee_rate == 0.07


def test_evaluate_restores_default_cfg_after_per_class_call():
    """Outside an evaluate() call, strategy.cfg must read as the originally-
    constructed default — never the last per-class swap."""
    default = _cfg(tte_max_seconds=1800)
    bucket = _cfg(tte_max_seconds=86400)
    s = LateResolutionStrategy(default, cfg_by_class={"priceBucket": bucket})

    now = 10_000_000_000_000
    expiry_8h = now + 8 * 3600 * 1_000_000_000
    s.evaluate(
        question=_q_bucket(expiry_ns=expiry_8h), books={}, reference_price=80_000.0,
        recent_returns=(), recent_volume_usd=0.0, position=None, now_ns=now,
    )
    assert s.cfg is default



# --- Cadence awareness: vol_sampling_dt_seconds scales sigma_window / lookback ---


def _alternating_returns(a: float, n: int) -> tuple[float, ...]:
    """n returns alternating +a, -a (mean 0, sample std ~= a). Any contiguous
    even-length slice stays balanced, so sigma is invariant to n_keep — this
    isolates the sqrt(tte_s/dt) scaling from the lookback-window slicing."""
    return tuple((a if i % 2 == 0 else -a) for i in range(n))


def test_vol_sampling_dt_seconds_defaults_to_60():
    # Backward-compat marker: legacy configs assume 60s bars.
    assert _cfg().vol_sampling_dt_seconds == 60


def test_entry_safety_gate_is_cadence_aware():
    # Identical inputs; only vol_sampling_dt_seconds differs. At dt=60 the
    # winning leg is ~0.48 sigma safe (>= min_safety_d=0.3) -> ENTER. At dt=5
    # the SAME per-bar sigma must be scaled over tte_s/5 bars (not tte_s/60),
    # making sigma_window ~sqrt(12) larger and safety_d ~sqrt(12) smaller
    # (~0.14 < 0.3) -> HOLD. With the 60s-hardcoded math this distinction is
    # invisible and dt=5 wrongly ENTERs.
    now = 10_000_000_000_000
    expiry = now + 3600 * 1_000_000_000  # 1h TTE
    q = _q(strike=80_000.0, expiry_ns=expiry)
    books = {
        "@30": _ref_book("@30", ask=0.96, bid=0.95, ts_ns=now - 100),
        "@31": _ref_book("@31", ask=0.06, bid=0.04, ts_ns=now - 100),
    }
    returns = _alternating_returns(0.001, 800)
    common = dict(
        tte_min_seconds=0,
        tte_max_seconds=7200,
        vol_max=100.0,
        min_safety_d=0.3,
        vol_lookback_seconds=3600,
        vol_ewma_lambda=0.0,
    )

    s60 = LateResolutionStrategy(_cfg(vol_sampling_dt_seconds=60, **common))
    d60 = s60.evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=returns, recent_volume_usd=5_000.0, position=None, now_ns=now,
    )
    assert d60.action is Action.ENTER

    s5 = LateResolutionStrategy(_cfg(vol_sampling_dt_seconds=5, **common))
    d5 = s5.evaluate(
        question=q, books=books, reference_price=80_300.0,
        recent_returns=returns, recent_volume_usd=5_000.0, position=None, now_ns=now,
    )
    assert d5.action is Action.HOLD
    assert any("safety_d_below_min" in dg.message for dg in d5.diagnostics)
