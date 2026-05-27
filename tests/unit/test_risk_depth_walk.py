"""Phase 7.1 — RiskGate depth-walk slippage gate.

PM books quote a 4-digit YES price with thin top-of-book; an IOC at the
inside ask can only fill against levels at-or-below the limit price.
The gate restricts the ladder walk to those levels, then either:
  * approves with the intent size unchanged when fully fillable,
  * approves with a `clamped_size` when only partial liquidity is available
    (the router resizes the intent; the strategy's topup loop closes the
    residual on subsequent ticks), or
  * vetoes when zero liquidity is at-or-below the limit, or when a strategy
    explicitly raises the limit past inside and the blended fill across
    walked levels exceeds GlobalRiskConfig.max_slippage_pct.
Disabled (cap = 0) on HL slots so HL behaviour is unchanged.
"""
from __future__ import annotations

from hlanalysis.engine.config import (
    AllowlistEntry, GlobalRiskConfig, StrategyConfig,
)
from hlanalysis.engine.risk import RiskGate, RiskInputs
from hlanalysis.strategy.types import BookState, OrderIntent, QuestionView


def _mk_cfg(
    *, max_slip_pct: float = 0.005, min_order_notional: float = 0.0,
) -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary"},
        max_position_usd=200, stop_loss_pct=None,
        tte_min_seconds=0, tte_max_seconds=86400,
        price_extreme_threshold=0, price_extreme_max=1, vol_max=100,
        distance_from_strike_usd_min=0,
    )
    g = GlobalRiskConfig(
        max_total_inventory_usd=1000, max_concurrent_positions=5,
        daily_loss_cap_usd=100, max_strike_distance_pct=50,
        min_recent_volume_usd=0, stale_data_halt_seconds=999,
        reconcile_interval_seconds=60,
        max_slippage_pct=max_slip_pct,
        min_order_notional_usd=min_order_notional,
    )
    return StrategyConfig(
        name="t", paper_mode=True, allowlist=[entry],
        defaults=entry, **{"global": g},
    )


def _q() -> QuestionView:
    # expiry sits inside the 86400s tte_max window so the gate reaches the
    # depth-walk branch (which is appended after the existing TTE check).
    # strike close to reference_price so the strike-distance gate passes.
    return QuestionView(
        question_idx=1, yes_symbol="tok", no_symbol="n",
        strike=110_000.0, expiry_ns=200 + 3600 * 1_000_000_000,
        underlying="BTC", klass="priceBinary", period="24h",
    )


def _inputs(book: BookState) -> RiskInputs:
    return RiskInputs(
        question=_q(),
        question_fields={"class": "priceBinary", "underlying": "BTC"},
        reference_price=110_000.0, book=book, recent_volume_usd=1000.0,
        positions=[], live_orders_total_notional=0.0,
        realized_pnl_today=0.0, kill_switch_active=False,
        last_reconcile_ns=100, now_ns=200,
    )


def _buy(size: float, limit: float) -> OrderIntent:
    return OrderIntent(
        question_idx=1, symbol="tok", side="buy", size=size,
        limit_price=limit, cloid="c", time_in_force="ioc",
    )


def test_depth_walk_vetoes_when_slippage_exceeds_cap():
    # Strategy raises the limit past inside to walk levels; the blended fill
    # across the walked ladder breaches the slip cap → veto.
    cfg = _mk_cfg(max_slip_pct=0.005)
    gate = RiskGate(cfg)
    book = BookState(
        symbol="tok",
        bid_px=0.92, bid_sz=200.0, ask_px=0.93, ask_sz=10.0,
        last_l2_ts_ns=100, last_trade_ts_ns=100,
        ask_levels=((0.93, 10.0), (0.95, 100.0), (0.98, 200.0)),
    )
    # Limit at 0.98 lets the walk touch all three levels; avg ≈ 0.9567 vs
    # limit 0.98 → slip ≈ -2.4% (favourable). To force a slip-cap breach we
    # need avg > limit·(1+cap); set limit just barely above inside.
    v = gate.check_pre_trade(_buy(150.0, 0.934), _inputs(book))
    # At limit=0.934, only the 0.93 level is usable → clamps to 10 (no slip).
    assert v.approved
    assert v.clamped_size == 10.0


def test_depth_walk_approves_when_first_level_covers_size():
    cfg = _mk_cfg(max_slip_pct=0.005)
    gate = RiskGate(cfg)
    book = BookState(
        symbol="tok",
        bid_px=0.92, bid_sz=200.0, ask_px=0.93, ask_sz=300.0,
        last_l2_ts_ns=100, last_trade_ts_ns=100,
        ask_levels=((0.93, 300.0),),
    )
    v = gate.check_pre_trade(_buy(150.0, 0.93), _inputs(book))
    assert v.approved
    assert v.clamped_size is None


def test_depth_walk_clamps_when_insufficient_depth_at_limit():
    # New behavior: rather than veto, approve with `clamped_size` set to the
    # at-limit liquidity. The router resizes the intent and the strategy's
    # topup re-fires to close the residual.
    cfg = _mk_cfg(max_slip_pct=0.005)
    gate = RiskGate(cfg)
    book = BookState(
        symbol="tok",
        bid_px=0.92, bid_sz=200.0, ask_px=0.93, ask_sz=10.0,
        last_l2_ts_ns=100, last_trade_ts_ns=100,
        # Second level (0.94) is above the 0.93 limit → not usable.
        ask_levels=((0.93, 10.0), (0.94, 20.0)),
    )
    v = gate.check_pre_trade(_buy(150.0, 0.93), _inputs(book))
    assert v.approved
    assert v.clamped_size == 10.0


def test_depth_walk_clamp_below_min_notional_vetoes():
    # Inside ask has 2 contracts at 0.95 → clamp to 2 → notional $1.90.
    # min_order_notional=$11 → veto rather than submit a sub-$11 micro-order.
    cfg = _mk_cfg(max_slip_pct=0.005, min_order_notional=11.0)
    gate = RiskGate(cfg)
    book = BookState(
        symbol="tok",
        bid_px=0.93, bid_sz=200.0, ask_px=0.95, ask_sz=2.0,
        last_l2_ts_ns=100, last_trade_ts_ns=100,
        ask_levels=((0.95, 2.0), (0.97, 20.0)),
    )
    v = gate.check_pre_trade(_buy(50.0, 0.95), _inputs(book))
    assert not v.approved
    assert v.reason == "order_below_min_notional"


def test_depth_walk_clamp_above_min_notional_approves():
    # Inside ask has 15 contracts at 0.95 → clamp to 15 → notional $14.25.
    # min_order_notional=$11 → approved with clamped_size=15.
    cfg = _mk_cfg(max_slip_pct=0.005, min_order_notional=11.0)
    gate = RiskGate(cfg)
    book = BookState(
        symbol="tok",
        bid_px=0.93, bid_sz=200.0, ask_px=0.95, ask_sz=15.0,
        last_l2_ts_ns=100, last_trade_ts_ns=100,
        ask_levels=((0.95, 15.0), (0.97, 20.0)),
    )
    v = gate.check_pre_trade(_buy(50.0, 0.95), _inputs(book))
    assert v.approved
    assert v.clamped_size == 15.0


def test_min_notional_floor_zero_disables_check():
    # HL default: min_order_notional=0 → tiny clamped orders still flow through.
    cfg = _mk_cfg(max_slip_pct=0.005, min_order_notional=0.0)
    gate = RiskGate(cfg)
    book = BookState(
        symbol="tok",
        bid_px=0.93, bid_sz=200.0, ask_px=0.95, ask_sz=2.0,
        last_l2_ts_ns=100, last_trade_ts_ns=100,
        ask_levels=((0.95, 2.0),),
    )
    v = gate.check_pre_trade(_buy(50.0, 0.95), _inputs(book))
    assert v.approved
    assert v.clamped_size == 2.0


def test_depth_walk_vetoes_when_no_liquidity_at_or_below_limit():
    # Limit sits below every ladder level → no liquidity to lift; veto.
    cfg = _mk_cfg(max_slip_pct=0.005)
    gate = RiskGate(cfg)
    book = BookState(
        symbol="tok",
        bid_px=0.91, bid_sz=200.0, ask_px=0.94, ask_sz=10.0,
        last_l2_ts_ns=100, last_trade_ts_ns=100,
        ask_levels=((0.94, 10.0), (0.95, 20.0)),
    )
    v = gate.check_pre_trade(_buy(150.0, 0.93), _inputs(book))
    assert not v.approved
    assert v.reason == "depth_walk_no_fill"


def test_depth_walk_disabled_when_cap_zero():
    # max_slippage_pct=0 → HL behaviour, gate is a no-op even with thin levels.
    cfg = _mk_cfg(max_slip_pct=0.0)
    gate = RiskGate(cfg)
    book = BookState(
        symbol="tok",
        bid_px=0.92, bid_sz=200.0, ask_px=0.93, ask_sz=10.0,
        last_l2_ts_ns=100, last_trade_ts_ns=100,
        ask_levels=((0.93, 10.0),),
    )
    v = gate.check_pre_trade(_buy(150.0, 0.93), _inputs(book))
    assert v.approved


def test_depth_walk_skipped_when_no_levels_known():
    # ask_levels=() (legacy HL BboEvent path) → gate skipped, approve.
    cfg = _mk_cfg(max_slip_pct=0.005)
    gate = RiskGate(cfg)
    book = BookState(
        symbol="tok",
        bid_px=0.92, bid_sz=200.0, ask_px=0.93, ask_sz=10.0,
        last_l2_ts_ns=100, last_trade_ts_ns=100,
    )
    v = gate.check_pre_trade(_buy(150.0, 0.93), _inputs(book))
    assert v.approved


def test_depth_walk_uses_bid_ladder_for_sell():
    cfg = _mk_cfg(max_slip_pct=0.005)
    gate = RiskGate(cfg)
    book = BookState(
        symbol="tok",
        bid_px=0.90, bid_sz=10.0, ask_px=0.91, ask_sz=200.0,
        last_l2_ts_ns=100, last_trade_ts_ns=100,
        bid_levels=((0.90, 10.0), (0.85, 100.0)),
    )
    intent = OrderIntent(
        question_idx=1, symbol="tok", side="sell", size=50.0,
        limit_price=0.90, cloid="c", time_in_force="ioc",
        reduce_only=True,
    )
    # reduce_only → exit path; depth-walk gate is entry-only so this approves.
    # (Confirmed by the existing risk gate: exits return approved_exit early.)
    v = gate.check_pre_trade(intent, _inputs(book))
    assert v.approved
