"""Focused tests for the extracted shared strategy helpers.

These assert the consolidated helpers reproduce the behavior that was
previously inlined/duplicated across the strategy modules.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from hlanalysis.strategy.fee import fee_per_share
from hlanalysis.strategy.intents import (
    make_entry_intent,
    make_exit_intent,
    round_size,
)
from hlanalysis.strategy.regions import kv_get, winning_region
from hlanalysis.strategy.topup import run_topup
from hlanalysis.strategy.types import (
    Action,
    BookState,
    Decision,
    Diagnostic,
    Position,
    QuestionView,
)
from hlanalysis.strategy.vol import (
    ANNUAL_SECONDS,
    annualized_sigma,
    bipower_variation_sigma,
    sample_std_returns,
)


# --------------------------------------------------------------------------
# regions
# --------------------------------------------------------------------------
def _binary_qv() -> QuestionView:
    return QuestionView(
        question_idx=0,
        yes_symbol="@Y",
        no_symbol="@N",
        strike=80_000.0,
        expiry_ns=0,
        underlying="BTC",
        klass="priceBinary",
        period="1d",
        leg_symbols=("@Y", "@N"),
    )


def _bucket_qv() -> QuestionView:
    # 2 thresholds → 3 outcomes → 6 legs (YES/NO interleaved).
    return QuestionView(
        question_idx=0,
        yes_symbol="",
        no_symbol="",
        strike=float("nan"),
        expiry_ns=0,
        underlying="BTC",
        klass="priceBucket",
        period="1d",
        leg_symbols=("y0", "n0", "y1", "n1", "y2", "n2"),
        kv=(("priceThresholds", "77000,81000"),),
    )


def test_kv_get_found_missing_default():
    qv = _bucket_qv()
    assert kv_get(qv, "priceThresholds") == "77000,81000"
    assert kv_get(qv, "absent") == ""
    assert kv_get(qv, "absent", "fallback") == "fallback"


def test_winning_region_binary():
    qv = _binary_qv()
    assert winning_region(qv, "@Y") == (80_000.0, None)
    assert winning_region(qv, "@N") == (None, 80_000.0)
    assert winning_region(qv, "@Z") == (None, None)


def test_winning_region_bucket_edges_and_middle():
    qv = _bucket_qv()
    assert winning_region(qv, "y0") == (None, 77000.0)  # lowest YES
    assert winning_region(qv, "y1") == (77000.0, 81000.0)  # middle YES
    assert winning_region(qv, "y2") == (81000.0, None)  # highest YES
    assert winning_region(qv, "n0") == (77000.0, None)  # NO of lowest → half-line
    assert winning_region(qv, "n2") == (None, 81000.0)  # NO of highest → half-line
    assert winning_region(qv, "n1") == (None, None)  # NO of middle → non-contiguous


def _above_ladder_qv() -> QuestionView:
    """PM multi-strike ladder: 3 independent 'above X' binaries interleaved YES/NO."""
    legs = []
    for t in [78000, 80000, 82000]:
        legs += [f"y{t}", f"n{t}"]
    return QuestionView(
        question_idx=0,
        yes_symbol="",
        no_symbol="",
        strike=float("nan"),
        expiry_ns=0,
        underlying="BTC",
        klass="priceBucket",
        period="1d",
        leg_symbols=tuple(legs),
        kv=(("priceThresholds", "78000,80000,82000"), ("bucketLayout", "above_ladder")),
    )


def test_above_ladder_yes_legs_map_to_half_lines():
    qv = _above_ladder_qv()
    assert winning_region(qv, "y78000") == (78000.0, None)
    assert winning_region(qv, "y80000") == (80000.0, None)
    assert winning_region(qv, "y82000") == (82000.0, None)


def test_above_ladder_no_legs_map_to_lower_half_line():
    qv = _above_ladder_qv()
    assert winning_region(qv, "n78000") == (None, 78000.0)
    assert winning_region(qv, "n80000") == (None, 80000.0)
    assert winning_region(qv, "n82000") == (None, 82000.0)


def test_contiguous_bucket_unchanged_without_flag():
    """HL path (no bucketLayout kv) must be byte-identical to the original."""
    # Same leg_symbols and thresholds as above_ladder_qv but NO bucketLayout flag.
    legs = []
    for t in [78000, 80000, 82000]:
        legs += [f"y{t}", f"n{t}"]
    qv = QuestionView(
        question_idx=0,
        yes_symbol="",
        no_symbol="",
        strike=float("nan"),
        expiry_ns=0,
        underlying="BTC",
        klass="priceBucket",
        period="1d",
        leg_symbols=tuple(legs),
        kv=(("priceThresholds", "78000,80000,82000"),),  # no bucketLayout
    )
    # Contiguous convention: 3 thresholds → 4 outcome buckets, YES at even index.
    # y78000 = outcome 0 (lowest): (None, 78000); y80000 = outcome 1: (78000, 80000); y82000 = outcome 2: (80000, 82000)
    assert winning_region(qv, "y78000") == (None, 78000.0)
    assert winning_region(qv, "y80000") == (78000.0, 80000.0)
    assert winning_region(qv, "y82000") == (80000.0, 82000.0)


def test_binary_unchanged():
    qv = _binary_qv()
    assert winning_region(qv, "@Y") == (80_000.0, None)
    assert winning_region(qv, "@N") == (None, 80_000.0)
    assert winning_region(qv, "@Z") == (None, None)


# --------------------------------------------------------------------------
# vol
# --------------------------------------------------------------------------
def test_annual_seconds_value():
    assert ANNUAL_SECONDS == 365.25 * 86400.0


@pytest.mark.parametrize("n", [2, 30, 250])
def test_sample_std_matches_numpy(n):
    arr = np.random.default_rng(n).normal(0, 1e-3, n)
    assert math.isclose(sample_std_returns(arr), float(np.std(arr, ddof=1)), rel_tol=1e-12)


def test_sample_std_short_window_zero():
    assert sample_std_returns(np.array([0.01])) == 0.0
    assert sample_std_returns(np.array([])) == 0.0


def test_bipower_matches_formula():
    arr = np.random.default_rng(7).normal(0, 1e-3, 50)
    abs_r = np.abs(arr)
    expected = math.sqrt((math.pi / 2.0) * float(np.sum(abs_r[1:] * abs_r[:-1])) / (len(arr) - 1))
    assert math.isclose(bipower_variation_sigma(arr), expected, rel_tol=1e-12)


def test_bipower_short_and_zero():
    assert bipower_variation_sigma(np.array([0.01])) == 0.0
    assert bipower_variation_sigma(np.zeros(10)) == 0.0


def test_annualized_sigma_matches_manual_pipeline():
    arr = np.random.default_rng(1).normal(0, 1e-3, 40)
    dt = 60
    raw = sample_std_returns(arr)
    expected = max(0.05, min(3.0, raw * math.sqrt(ANNUAL_SECONDS / dt)))
    got = annualized_sigma(arr, dt_seconds=dt, estimator="sample_std", clip_min=0.05, clip_max=3.0)
    assert math.isclose(got, expected, rel_tol=1e-12)
    # bipower path
    raw_bp = bipower_variation_sigma(arr)
    exp_bp = max(0.05, min(3.0, raw_bp * math.sqrt(ANNUAL_SECONDS / dt)))
    got_bp = annualized_sigma(arr, dt_seconds=dt, estimator="bipower", clip_min=0.05, clip_max=3.0)
    assert math.isclose(got_bp, exp_bp, rel_tol=1e-12)


def test_annualized_sigma_unknown_estimator_raises():
    with pytest.raises(ValueError):
        annualized_sigma(np.array([0.0, 0.1]), dt_seconds=60, estimator="bogus", clip_min=0.0, clip_max=1.0)


# --------------------------------------------------------------------------
# intents
# --------------------------------------------------------------------------
def test_round_size_matches_legacy_floor():
    assert round_size(100.0, 0.97) == math.floor((100.0 / 0.97) * 100) / 100


def test_make_entry_intent_shape():
    qv = _binary_qv()
    it = make_entry_intent(qv, symbol="@Y", size=12.34, limit_price=0.97)
    assert it.side == "buy"
    assert it.size == 12.34
    assert it.limit_price == 0.97
    assert it.reduce_only is False
    assert it.exit_reason == ""
    assert it.time_in_force == "ioc"
    assert it.cloid.startswith("hla-")


def test_make_exit_intent_flips_side_and_flags():
    qv = _binary_qv()
    long_pos = Position(question_idx=0, symbol="@Y", qty=10.0, avg_entry=0.9, stop_loss_price=0.5, last_update_ts_ns=0)
    short_pos = Position(
        question_idx=0, symbol="@Y", qty=-10.0, avg_entry=0.9, stop_loss_price=0.5, last_update_ts_ns=0
    )
    it = make_exit_intent(qv, long_pos, limit_price=0.88, exit_reason="exit_stop_loss")
    assert it.side == "sell"
    assert it.size == 10.0
    assert it.reduce_only is True
    assert it.exit_reason == "exit_stop_loss"
    it2 = make_exit_intent(qv, short_pos, limit_price=0.88)
    assert it2.side == "buy"
    assert it2.exit_reason == ""  # default


# --------------------------------------------------------------------------
# fee
# --------------------------------------------------------------------------
def _fee_cfg(**kw):
    base = dict(fee_model="flat", fee_rate=0.0, fee_taker=0.0, exit_take_profit_mode=False, exit_fee=0.0007)
    base.update(kw)
    return SimpleNamespace(**base)


def test_fee_pm_binary_curve_both_sides():
    cfg = _fee_cfg(fee_model="pm_binary", fee_rate=0.07)
    for side in ("entry", "exit"):
        # exact float match to the legacy inline `fee_rate * p * (1 - p)`
        assert fee_per_share(cfg, 0.8, side=side) == 0.07 * 0.8 * (1.0 - 0.8)
    # mathematically symmetric in p (float ordering may differ by an ULP)
    assert math.isclose(
        fee_per_share(cfg, 0.8, side="entry"),
        fee_per_share(cfg, 0.2, side="entry"),
        rel_tol=1e-12,
    )


def test_fee_flat_entry_vs_exit_take_profit():
    # flat entry → fee_taker; flat exit with take_profit_mode → exit_fee.
    cfg = _fee_cfg(fee_taker=0.003, exit_take_profit_mode=True, exit_fee=0.0007)
    assert fee_per_share(cfg, 0.8, side="entry") == 0.003
    assert fee_per_share(cfg, 0.8, side="exit") == 0.0007
    # flat exit WITHOUT take_profit_mode → fee_taker
    cfg2 = _fee_cfg(fee_taker=0.003, exit_take_profit_mode=False)
    assert fee_per_share(cfg2, 0.8, side="exit") == 0.003


# --------------------------------------------------------------------------
# topup
# --------------------------------------------------------------------------
def _book(ask):
    return BookState(
        symbol="@Y", bid_px=ask - 0.01, bid_sz=100.0, ask_px=ask, ask_sz=100.0, last_trade_ts_ns=0, last_l2_ts_ns=0
    )


def _pos(qty):
    return Position(question_idx=0, symbol="@Y", qty=qty, avg_entry=0.5, stop_loss_price=0.0, last_update_ts_ns=0)


def _enter_decision():
    qv = _binary_qv()
    return Decision(action=Action.ENTER, intents=(make_entry_intent(qv, symbol="@Y", size=1.0, limit_price=0.5),))


def test_topup_no_book_uses_callback():
    qv = _binary_qv()
    sentinel = object()
    out = run_topup(
        question=qv,
        books={},
        position=_pos(10.0),
        max_position_usd=100.0,
        topup_threshold_pct=0.2,
        topup_min_notional_usd=11.0,
        run_entry=_enter_decision,
        on_no_book=lambda: sentinel,
        on_not_needed=lambda c, t: None,
    )
    assert out is sentinel


def test_topup_not_needed_uses_callback():
    qv = _binary_qv()
    # qty 200 @ ask 0.5 → current_ntl 100 == target → shortfall 0 < threshold.
    seen = {}

    def _not_needed(c, t):
        seen["c"], seen["t"] = c, t
        return None

    out = run_topup(
        question=qv,
        books={"@Y": _book(0.5)},
        position=_pos(200.0),
        max_position_usd=100.0,
        topup_threshold_pct=0.2,
        topup_min_notional_usd=11.0,
        run_entry=_enter_decision,
        on_no_book=lambda: None,
        on_not_needed=_not_needed,
    )
    assert out is None
    assert seen == {"c": 100.0, "t": 100.0}


def test_topup_emits_when_undersized_and_leg_unchanged():
    qv = _binary_qv()
    # qty 50 @ ask 0.5 → current 25, shortfall 75 (>20). topup_size = floor(75/0.5*100)/100=150
    out = run_topup(
        question=qv,
        books={"@Y": _book(0.5)},
        position=_pos(50.0),
        max_position_usd=100.0,
        topup_threshold_pct=0.2,
        topup_min_notional_usd=11.0,
        run_entry=_enter_decision,
        on_no_book=lambda: None,
        on_not_needed=lambda c, t: None,
    )
    assert out is not None and out.action == Action.ENTER
    assert out.intents[0].symbol == "@Y"
    assert out.intents[0].size == round_size(75.0, 0.5)


def test_topup_gate_failed_returns_skip():
    qv = _binary_qv()
    hold = Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "no_favorite"),))
    out = run_topup(
        question=qv,
        books={"@Y": _book(0.5)},
        position=_pos(50.0),
        max_position_usd=100.0,
        topup_threshold_pct=0.2,
        topup_min_notional_usd=11.0,
        run_entry=lambda: hold,
        on_no_book=lambda: None,
        on_not_needed=lambda c, t: None,
    )
    assert out is not None and out.action == Action.HOLD
    assert out.diagnostics[0].message == "topup_skip"
    assert out.diagnostics[0].fields[0] == ("reason", "gate_failed:no_favorite")


def test_topup_leg_changed_returns_skip():
    qv = _binary_qv()
    other = Decision(action=Action.ENTER, intents=(make_entry_intent(qv, symbol="@N", size=1.0, limit_price=0.5),))
    out = run_topup(
        question=qv,
        books={"@Y": _book(0.5)},
        position=_pos(50.0),
        max_position_usd=100.0,
        topup_threshold_pct=0.2,
        topup_min_notional_usd=11.0,
        run_entry=lambda: other,
        on_no_book=lambda: None,
        on_not_needed=lambda c, t: None,
    )
    assert out is not None and out.diagnostics[0].fields[0] == ("reason", "leg_changed")


def test_topup_below_min_notional_returns_skip():
    qv = _binary_qv()
    # shortfall just above threshold but topup_ntl below min: target 100, qty 178@0.5
    # current=89, shortfall=11 (>=20? no). Use threshold 0.1 so 11>=10 passes, min=20.
    out = run_topup(
        question=qv,
        books={"@Y": _book(0.5)},
        position=_pos(178.0),
        max_position_usd=100.0,
        topup_threshold_pct=0.1,
        topup_min_notional_usd=20.0,
        run_entry=_enter_decision,
        on_no_book=lambda: None,
        on_not_needed=lambda c, t: None,
    )
    assert out is not None and out.diagnostics[0].fields[0] == ("reason", "below_min_notional")
