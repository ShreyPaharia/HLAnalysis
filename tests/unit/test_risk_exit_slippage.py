"""SHR-48 — Exit IOC slippage clamp.

Stop/exit orders must respect max_slippage_pct exactly like entries; the prior
`if is_exit: return approved_exit` short-circuit bypassed the depth-walk block
entirely, allowing a ~1 Hz stop-loss loop to walk a thin PM book to zero.

Three scenarios are tested:
  1. PM slot (max_slippage_pct=0.005, bid_levels populated): a sell exit whose
     size exceeds inside-bid depth must be CLAMPED to the at-budget fill, not
     approved at full size.
  2. HL slot (max_slippage_pct=0, no bid_levels): exit must pass through
     unclamped (no regression for HL where the feed doesn't populate levels).
  3. No-usable-levels veto: a sell exit where no bid level is at or above the
     limit price should return depth_walk_no_fill, not approved_exit.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from hlanalysis.engine.config import (
    AllowlistEntry,
    GlobalRiskConfig,
    StrategyConfig,
)
from hlanalysis.engine.risk import RiskGate, RiskInputs
from hlanalysis.strategy.types import BookState, OrderIntent, QuestionView


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mk_cfg(*, max_slip_pct: float, min_order_notional: float = 0.0) -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary"},
        max_position_usd=200,
        stop_loss_pct=None,
        tte_min_seconds=0,
        tte_max_seconds=86400,
        price_extreme_threshold=0,
        price_extreme_max=1,
        vol_max=100,
        distance_from_strike_usd_min=0,
    )
    g = GlobalRiskConfig(
        max_total_inventory_usd=1000,
        max_concurrent_positions=5,
        daily_loss_cap_usd=100,
        max_strike_distance_pct=50,
        min_recent_volume_usd=0,
        stale_data_halt_seconds=999,
        reconcile_interval_seconds=60,
        max_slippage_pct=max_slip_pct,
        min_order_notional_usd=min_order_notional,
    )
    return StrategyConfig(
        name="t",
        paper_mode=True,
        allowlist=[entry],
        defaults=entry,
        **{"global": g},
    )


def _q() -> QuestionView:
    return QuestionView(
        question_idx=1,
        yes_symbol="tok",
        no_symbol="n",
        strike=110_000.0,
        expiry_ns=200 + 3600 * 1_000_000_000,
        underlying="BTC",
        klass="priceBinary",
        period="24h",
    )


def _base_inputs(book: BookState) -> RiskInputs:
    return RiskInputs(
        question=_q(),
        question_fields={"class": "priceBinary", "underlying": "BTC"},
        reference_price=110_000.0,
        book=book,
        recent_volume_usd=1000.0,
        positions=[],
        live_orders_total_notional=0.0,
        realized_pnl_today=0.0,
        kill_switch_active=False,
        last_reconcile_ns=100,
        now_ns=200,
    )


def _exit_sell(size: float, limit: float) -> OrderIntent:
    """A reduce-only (exit) sell intent — matches the stop-loss IOC shape."""
    return OrderIntent(
        question_idx=1,
        symbol="tok",
        side="sell",
        size=size,
        limit_price=limit,
        cloid="c",
        time_in_force="ioc",
        reduce_only=True,
        exit_reason="stop_loss",
    )


# ---------------------------------------------------------------------------
# Test 1: PM slot — exit sell CLAMPED to at-budget depth (SHR-48 core)
# ---------------------------------------------------------------------------

def test_exit_sell_clamped_to_at_budget_depth():
    """Exit whose size > inside-bid depth on a PM book is approved with
    clamped_size equal to the available depth at-or-above the limit price.
    The second bid level is more than 0.5% below the top so the full walk
    would exceed the slip cap; instead we clamp to the first level only.
    """
    cfg = _mk_cfg(max_slip_pct=0.005)
    gate = RiskGate(cfg)
    # Inside bid: 0.88 @ 10 contracts. Second level: 0.87 (gap = 1.14% > 0.5%)
    book = BookState(
        symbol="tok",
        bid_px=0.88,
        bid_sz=10.0,
        ask_px=0.90,
        ask_sz=200.0,
        last_l2_ts_ns=100,
        last_trade_ts_ns=100,
        bid_levels=((0.88, 10.0), (0.87, 100.0)),
    )
    # Intent: sell 50 contracts at limit=0.88.
    # Only the 0.88 level is at-or-above limit=0.88; depth = 10.
    # Remaining = 40 → clamped_size should be 10.
    intent = _exit_sell(size=50.0, limit=0.88)
    v = gate.check_pre_trade(intent, _base_inputs(book))

    assert v.approved, f"Expected approved; got reason={v.reason!r}"
    assert v.clamped_size is not None, "Expected clamped_size to be set for thin-book exit"
    assert v.clamped_size < intent.size, (
        f"clamped_size={v.clamped_size} should be < intent.size={intent.size}"
    )
    assert v.clamped_size == 10.0


# ---------------------------------------------------------------------------
# Test 2: PM slot — exit sell APPROVED at full size (enough depth)
# ---------------------------------------------------------------------------

def test_exit_sell_approved_unclamped_when_depth_sufficient():
    """Exit whose full size is covered by a single bid level at-or-above limit
    is approved with clamped_size=None (no clamp needed)."""
    cfg = _mk_cfg(max_slip_pct=0.005)
    gate = RiskGate(cfg)
    book = BookState(
        symbol="tok",
        bid_px=0.88,
        bid_sz=200.0,
        ask_px=0.90,
        ask_sz=200.0,
        last_l2_ts_ns=100,
        last_trade_ts_ns=100,
        bid_levels=((0.88, 200.0),),
    )
    intent = _exit_sell(size=50.0, limit=0.88)
    v = gate.check_pre_trade(intent, _base_inputs(book))

    assert v.approved
    assert v.clamped_size is None


# ---------------------------------------------------------------------------
# Test 3: HL slot — no levels, slip cap = 0 → exit passes unclamped (no regression)
# ---------------------------------------------------------------------------

def test_exit_hl_slot_no_levels_approved_unclamped():
    """HL slots leave max_slippage_pct=0 and don't populate bid/ask levels.
    Exit must be approved unconditionally (no regression for HL)."""
    cfg = _mk_cfg(max_slip_pct=0.0)
    gate = RiskGate(cfg)
    # BboEvent-style: no bid_levels / ask_levels
    book = BookState(
        symbol="tok",
        bid_px=0.88,
        bid_sz=200.0,
        ask_px=0.90,
        ask_sz=200.0,
        last_l2_ts_ns=100,
        last_trade_ts_ns=100,
    )
    intent = _exit_sell(size=50.0, limit=0.88)
    v = gate.check_pre_trade(intent, _base_inputs(book))

    assert v.approved
    assert v.clamped_size is None


# ---------------------------------------------------------------------------
# Test 4: PM slot — exit with NO usable levels (all bids below limit) → veto
# ---------------------------------------------------------------------------

def test_exit_vetoed_when_no_usable_bid_levels():
    """If all bid levels are below the limit price (no at-or-above level),
    the exit is vetoed with depth_walk_no_fill.  A partial reduce against a
    stale book that the strategy already priced is better refused than walked
    to ruinous prices."""
    cfg = _mk_cfg(max_slip_pct=0.005)
    gate = RiskGate(cfg)
    # Limit=0.88 but all bids are below 0.88 → no usable level
    book = BookState(
        symbol="tok",
        bid_px=0.85,
        bid_sz=100.0,
        ask_px=0.90,
        ask_sz=200.0,
        last_l2_ts_ns=100,
        last_trade_ts_ns=100,
        bid_levels=((0.85, 100.0), (0.80, 200.0)),
    )
    intent = _exit_sell(size=50.0, limit=0.88)
    v = gate.check_pre_trade(intent, _base_inputs(book))

    assert not v.approved
    assert v.reason == "depth_walk_no_fill"


# ---------------------------------------------------------------------------
# Test 5: PM slot — exit ENTRY gates are still skipped (daily loss / inventory)
# ---------------------------------------------------------------------------

def test_exit_skips_daily_loss_gate():
    """Exits must still skip entry-only gates. An exit on a daily-loss-breached
    slot should be approved (modulo slippage), not rejected by the daily_loss_cap."""
    cfg = _mk_cfg(max_slip_pct=0.005)
    gate = RiskGate(cfg)
    book = BookState(
        symbol="tok",
        bid_px=0.88,
        bid_sz=200.0,
        ask_px=0.90,
        ask_sz=200.0,
        last_l2_ts_ns=100,
        last_trade_ts_ns=100,
        bid_levels=((0.88, 200.0),),
    )
    intent = _exit_sell(size=50.0, limit=0.88)
    # Breach the daily loss cap — this must NOT block the exit.
    inp = replace(_base_inputs(book), realized_pnl_today=-9999.0)
    v = gate.check_pre_trade(intent, inp)

    assert v.approved, f"Exit should bypass daily_loss_cap gate; got reason={v.reason!r}"


# ---------------------------------------------------------------------------
# Test 6: Regression — existing entry-side depth-walk behavior unchanged
# ---------------------------------------------------------------------------

def test_entry_buy_clamped_when_insufficient_ask_depth():
    """Regression: entry buys are still clamped by the depth-walk block."""
    cfg = _mk_cfg(max_slip_pct=0.005)
    gate = RiskGate(cfg)
    book = BookState(
        symbol="tok",
        bid_px=0.88,
        bid_sz=200.0,
        ask_px=0.90,
        ask_sz=10.0,
        last_l2_ts_ns=100,
        last_trade_ts_ns=100,
        ask_levels=((0.90, 10.0), (0.92, 20.0)),
    )
    # Entry buy (reduce_only=False), limit=0.90 → only 0.90 level usable; 10 < 50 → clamp
    intent = OrderIntent(
        question_idx=1,
        symbol="tok",
        side="buy",
        size=50.0,
        limit_price=0.90,
        cloid="c",
        time_in_force="ioc",
    )
    v = gate.check_pre_trade(intent, _base_inputs(book))
    assert v.approved
    assert v.clamped_size == 10.0
