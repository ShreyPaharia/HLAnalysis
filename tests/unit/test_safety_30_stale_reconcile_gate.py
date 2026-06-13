"""#30: Stale-reconcile pre-trade gate FAILS OPEN on startup.

Tests that last_reconcile_ns == 0 (never reconciled) vetoes entries with a
clear reason, while exits are always allowed, and once a reconcile timestamp is
set entries are allowed again.
"""

from __future__ import annotations

from hlanalysis.engine.config import (
    AllowlistEntry,
    GlobalRiskConfig,
    StrategyConfig,
)
from hlanalysis.engine.risk import RiskGate, RiskInputs
from hlanalysis.strategy.types import BookState, OrderIntent, QuestionView

NOW = 10_000_000_000_000_000


def _strategy_cfg() -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        max_position_usd=100,
        stop_loss_pct=10,
        tte_min_seconds=60,
        tte_max_seconds=1800,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200,
        vol_max=0.5,
    )
    return StrategyConfig(
        name="late_resolution",
        paper_mode=True,
        allowlist=[entry],
        blocklist_question_idxs=[],
        defaults=entry,
        **{
            "global": GlobalRiskConfig(
                max_total_inventory_usd=500,
                max_concurrent_positions=5,
                daily_loss_cap_usd=200,
                max_strike_distance_pct=10,
                min_recent_volume_usd=1000,
                stale_data_halt_seconds=5,
                reconcile_interval_seconds=60,
            )
        },
    )


def _q() -> QuestionView:
    return QuestionView(
        question_idx=42,
        yes_symbol="@30",
        no_symbol="@31",
        strike=80_000.0,
        expiry_ns=NOW + 600_000_000_000,
        underlying="BTC",
        klass="priceBinary",
        period="1h",
    )


def _book() -> BookState:
    return BookState(
        symbol="@30",
        bid_px=0.94,
        bid_sz=10.0,
        ask_px=0.95,
        ask_sz=10.0,
        last_trade_ts_ns=NOW,
        last_l2_ts_ns=NOW,
    )


def _entry_intent() -> OrderIntent:
    return OrderIntent(
        question_idx=42,
        symbol="@30",
        side="buy",
        size=100.0,
        limit_price=0.95,
        cloid="hla-test",
        time_in_force="ioc",
        reduce_only=False,
    )


def _exit_intent() -> OrderIntent:
    return OrderIntent(
        question_idx=42,
        symbol="@30",
        side="sell",
        size=100.0,
        limit_price=0.94,
        cloid="hla-test-exit",
        time_in_force="ioc",
        reduce_only=True,
    )


def _inputs(last_reconcile_ns: int, **overrides) -> RiskInputs:
    base = dict(
        question=_q(),
        question_fields={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        reference_price=80_300.0,
        book=_book(),
        recent_volume_usd=5_000.0,
        positions=[],
        live_orders_total_notional=0.0,
        realized_pnl_today=0.0,
        kill_switch_active=False,
        last_reconcile_ns=last_reconcile_ns,
        now_ns=NOW,
    )
    base.update(overrides)
    return RiskInputs(**base)


def test_entry_vetoed_when_never_reconciled():
    """#30: last_reconcile_ns=0 (never reconciled) must veto entries."""
    gate = RiskGate(_strategy_cfg())
    inp = _inputs(last_reconcile_ns=0)
    v = gate.check_pre_trade(_entry_intent(), inp)
    assert v.approved is False
    assert "reconcile" in v.reason


def test_exit_always_allowed_when_never_reconciled():
    """#30: exits must pass even when never reconciled (exits are always exempt)."""
    gate = RiskGate(_strategy_cfg())
    inp = _inputs(last_reconcile_ns=0)
    v = gate.check_pre_trade(_exit_intent(), inp)
    assert v.approved is True


def test_entry_allowed_after_reconcile_timestamp_set():
    """#30: once a reconcile has happened, entries should not be vetoed by the
    reconcile gate (other gates may still veto for other reasons)."""
    gate = RiskGate(_strategy_cfg())
    # Fresh reconcile (1 ms ago — well within the 2*60s stale window)
    inp = _inputs(last_reconcile_ns=NOW - 1_000_000)
    v = gate.check_pre_trade(_entry_intent(), inp)
    # The reconcile gate must not be the reason for veto.
    assert "reconcile" not in v.reason


def test_entry_still_vetoed_when_reconcile_stale():
    """Existing behaviour: stale (but non-zero) reconcile still vetoes entries."""
    gate = RiskGate(_strategy_cfg())
    # reconcile_interval = 60s; stale threshold = 2 * 60s = 120s
    stale_reconcile_ns = NOW - 130 * 1_000_000_000  # 130s ago
    inp = _inputs(last_reconcile_ns=stale_reconcile_ns)
    v = gate.check_pre_trade(_entry_intent(), inp)
    assert v.approved is False
    assert "reconcile" in v.reason
