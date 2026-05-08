from __future__ import annotations

import pytest

from hlanalysis.engine.config import (
    AllowlistEntry, GlobalRiskConfig, StrategyConfig,
)
from hlanalysis.engine.risk import RiskGate, RiskInputs
from hlanalysis.strategy.types import (
    BookState, OrderIntent, Position, QuestionView,
)


def _strategy_cfg() -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        max_position_usd=100, stop_loss_pct=10, tte_min_seconds=60,
        tte_max_seconds=1800, price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200, vol_max=0.5,
    )
    return StrategyConfig(
        name="late_resolution", paper_mode=True,
        allowlist=[entry], blocklist_question_idxs=[],
        defaults=entry,
        **{"global": GlobalRiskConfig(
            max_total_inventory_usd=500, max_concurrent_positions=5,
            daily_loss_cap_usd=200, max_strike_distance_pct=10,
            min_recent_volume_usd=1000, stale_data_halt_seconds=5,
            reconcile_interval_seconds=60,
        )},
    )


def _q(strike: float = 80_000.0, expiry_ns: int = 0,
       klass: str = "priceBinary", period: str = "1h") -> QuestionView:
    return QuestionView(
        question_idx=42, yes_symbol="@30", no_symbol="@31",
        strike=strike, expiry_ns=expiry_ns, underlying="BTC",
        klass=klass, period=period,
    )


def _intent(symbol: str = "@30", size: float = 100.0, price: float = 0.95) -> OrderIntent:
    return OrderIntent(
        question_idx=42, symbol=symbol, side="buy", size=size,
        limit_price=price, cloid="hla-test", time_in_force="ioc",
    )


def _inputs(**overrides) -> RiskInputs:
    base = dict(
        question=_q(expiry_ns=10_000_000_000_000_001 + 600_000_000_000),
        question_fields={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        reference_price=80_300.0,
        book=BookState(symbol="@30", bid_px=0.94, bid_sz=10.0, ask_px=0.95,
                       ask_sz=10.0, last_trade_ts_ns=10_000_000_000_000_000,
                       last_l2_ts_ns=10_000_000_000_000_000),
        recent_volume_usd=5_000.0,
        positions=[],
        live_orders_total_notional=0.0,
        realized_pnl_today=0.0,
        kill_switch_active=False,
        last_reconcile_ns=10_000_000_000_000_000,
        now_ns=10_000_000_000_000_001,
    )
    base.update(overrides)
    return RiskInputs(**base)


def _gate() -> RiskGate:
    return RiskGate(_strategy_cfg())


# 1. Allowlist match
def test_reject_when_allowlist_does_not_match():
    inp = _inputs(
        question=_q(klass="priceBucket", expiry_ns=10_000_000_000_000_001 + 600_000_000_000),
        question_fields={"class": "priceBucket", "underlying": "BTC", "period": "1h"},
    )
    v = _gate().check_pre_trade(_intent(), inp)
    assert v.approved is False and "allowlist" in v.reason


def test_reject_when_question_in_blocklist():
    cfg = _strategy_cfg().model_copy(update={"blocklist_question_idxs": [42]})
    inp = _inputs()
    v = RiskGate(cfg).check_pre_trade(_intent(), inp)
    assert v.approved is False and "blocklist" in v.reason


# 2. Per-position cap
def test_reject_when_intent_above_per_position_cap():
    v = _gate().check_pre_trade(_intent(size=10_000), _inputs())
    assert v.approved is False and "max_position_usd" in v.reason


# 3. Global inventory cap
def test_reject_when_global_inventory_exceeded():
    v = _gate().check_pre_trade(_intent(), _inputs(live_orders_total_notional=499))
    # 499 + 0.95*100 (=95) = 594 > 500
    assert v.approved is False and "max_total_inventory" in v.reason


# 4. Concurrent-positions cap
def test_reject_when_concurrent_positions_exceeded():
    poses = [
        Position(question_idx=i, symbol="@x", qty=1.0, avg_entry=0.95,
                 stop_loss_price=0.85, last_update_ts_ns=0)
        for i in range(5)
    ]
    v = _gate().check_pre_trade(_intent(), _inputs(positions=poses))
    assert v.approved is False and "max_concurrent" in v.reason


# 5. Daily loss cap (entries only)
def test_reject_entry_when_daily_loss_breached():
    v = _gate().check_pre_trade(_intent(), _inputs(realized_pnl_today=-300))
    assert v.approved is False and "daily_loss" in v.reason


def test_allow_exit_even_when_daily_loss_breached():
    from dataclasses import replace
    intent = replace(_intent(), reduce_only=True, side="sell")
    v = _gate().check_pre_trade(intent, _inputs(realized_pnl_today=-300))
    assert v.approved is True


# 6. Order size sanity
def test_reject_zero_size():
    from dataclasses import replace
    v = _gate().check_pre_trade(replace(_intent(), size=0), _inputs())
    assert v.approved is False and "size" in v.reason


# 7. TTE bounds
def test_reject_tte_out_of_window():
    far = _inputs().now_ns + 3600 * 1_000_000_000
    inp = _inputs(question=_q(expiry_ns=far))
    v = _gate().check_pre_trade(_intent(), inp)
    assert v.approved is False and "tte" in v.reason


# 8. Strike-proximity
def test_reject_when_strike_too_far_from_btc():
    inp = _inputs(reference_price=200_000.0, question=_q(strike=80_000.0,
                                                          expiry_ns=10_000_000_000_000_001 + 600_000_000_000))
    v = _gate().check_pre_trade(_intent(), inp)
    assert v.approved is False and "strike_distance" in v.reason


# 9. Min recent volume
def test_reject_when_recent_volume_below_min():
    v = _gate().check_pre_trade(_intent(), _inputs(recent_volume_usd=10.0))
    assert v.approved is False and "volume" in v.reason


# 10. Engine health (kill switch / reconcile / stale L2)
def test_reject_when_kill_switch_active():
    v = _gate().check_pre_trade(_intent(), _inputs(kill_switch_active=True))
    assert v.approved is False and "kill_switch" in v.reason


def test_reject_when_book_stale():
    inp = _inputs(book=BookState(
        symbol="@30", bid_px=0.94, bid_sz=10.0, ask_px=0.95, ask_sz=10.0,
        last_trade_ts_ns=0, last_l2_ts_ns=10_000_000_000_000_001 - 10 * 1_000_000_000,
    ))
    v = _gate().check_pre_trade(_intent(), inp)
    assert v.approved is False and "stale" in v.reason


# 11. No conflicting leg
def test_reject_when_holding_opposite_leg_of_same_question():
    pos = Position(question_idx=42, symbol="@31", qty=10.0, avg_entry=0.05,
                   stop_loss_price=0.045, last_update_ts_ns=0)
    v = _gate().check_pre_trade(_intent(symbol="@30"), _inputs(positions=[pos]))
    assert v.approved is False and "opposite" in v.reason


# 12. Settled-market refusal
def test_reject_when_question_settled():
    from dataclasses import replace
    q = replace(
        _q(expiry_ns=10_000_000_000_000_001 + 600_000_000_000),
        settled=True, settled_side="yes",
    )
    v = _gate().check_pre_trade(_intent(), _inputs(question=q))
    assert v.approved is False and "settled" in v.reason


# Happy path
def test_approve_when_all_checks_pass():
    v = _gate().check_pre_trade(_intent(), _inputs())
    assert v.approved is True


# Continuous helpers
def test_stop_loss_helper_detects_breach():
    pos = Position(question_idx=42, symbol="@30", qty=10.0, avg_entry=0.95,
                   stop_loss_price=0.86, last_update_ts_ns=0)
    book = BookState(symbol="@30", bid_px=0.85, bid_sz=10.0, ask_px=0.86,
                     ask_sz=10.0, last_trade_ts_ns=0, last_l2_ts_ns=0)
    breached = _gate().breached_stops([pos], {"@30": book})
    assert len(breached) == 1


def test_kill_switch_path_polled(tmp_path):
    p = tmp_path / "halt"
    assert _gate().kill_switch_active(p) is False
    p.write_text("halt")
    assert _gate().kill_switch_active(p) is True
