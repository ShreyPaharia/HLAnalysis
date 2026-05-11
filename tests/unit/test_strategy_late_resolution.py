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
    # limit_price is price_extreme_max (cap on entry cost), not the live ask;
    # sim/engine clamp fills to limit so realized fills never exceed the cap.
    assert math.isclose(intent.limit_price, 1.0, rel_tol=1e-9)


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
