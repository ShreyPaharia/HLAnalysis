"""Targeted TDD tests for four latent-risk fixes.

Fix 1 — risk.py ZeroDivisionError guard (book level with sz=0).
Fix 2 — engine/market_state.py one-sided BBO None guard.
Fix 3 — exec_types.py add "cancelled" to OrderAck.status Literal.
Fix 4 — unbounded _books / per-question cache eviction.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fix 1: ZeroDivisionError when the only book level has sz=0
# ---------------------------------------------------------------------------

from hlanalysis.engine.config import AllowlistEntry, GlobalRiskConfig, StrategyConfig
from hlanalysis.engine.risk import RiskGate, RiskInputs
from hlanalysis.strategy.types import BookState, OrderIntent, QuestionView


def _mk_cfg_slip(max_slip_pct: float = 0.01) -> StrategyConfig:
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
    )
    return StrategyConfig(
        name="t",
        paper_mode=True,
        allowlist=[entry],
        defaults=entry,
        **{"global": g},
    )


def _q1() -> QuestionView:
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


def _base_risk_inputs(book: BookState) -> RiskInputs:
    return RiskInputs(
        question=_q1(),
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


def test_depth_walk_zero_size_level_no_zerodivision():
    """Fix 1: book whose only level has sz=0 must not raise ZeroDivisionError.
    It should return the same 'no fillable size' outcome as an empty book.
    """
    cfg = _mk_cfg_slip(max_slip_pct=0.01)
    gate = RiskGate(cfg)
    # Only ask level has sz=0 — a degenerate book that can arrive from adapters
    # that publish a level entry with zero quantity before the size comes in.
    book = BookState(
        symbol="tok",
        bid_px=0.88,
        bid_sz=10.0,
        ask_px=0.90,
        ask_sz=0.0,
        last_l2_ts_ns=100,
        last_trade_ts_ns=100,
        ask_levels=((0.90, 0.0),),  # sz=0, the problematic case
    )
    intent = OrderIntent(
        question_idx=1,
        symbol="tok",
        side="buy",
        size=10.0,
        limit_price=0.90,
        cloid="c",
        time_in_force="ioc",
    )
    # Must not raise ZeroDivisionError.
    v = gate.check_pre_trade(intent, _base_risk_inputs(book))
    # Should be treated as "no fillable size" — same path as an empty book.
    # The depth-walk finds a level (price=0.90 <= limit=0.90) but sz=0, so
    # filled=0 → the guard kicks in → depth_walk_no_fill verdict.
    assert not v.approved
    assert v.reason == "depth_walk_no_fill"


def test_depth_walk_zero_size_exit_no_zerodivision():
    """Fix 1: same guard for an exit path (reduce_only=True)."""
    cfg = _mk_cfg_slip(max_slip_pct=0.01)
    gate = RiskGate(cfg)
    book = BookState(
        symbol="tok",
        bid_px=0.88,
        bid_sz=0.0,
        ask_px=0.90,
        ask_sz=10.0,
        last_l2_ts_ns=100,
        last_trade_ts_ns=100,
        bid_levels=((0.88, 0.0),),
    )
    intent = OrderIntent(
        question_idx=1,
        symbol="tok",
        side="sell",
        size=5.0,
        limit_price=0.88,
        cloid="c",
        time_in_force="ioc",
        reduce_only=True,
    )
    v = gate.check_pre_trade(intent, _base_risk_inputs(book))
    assert not v.approved
    assert v.reason == "depth_walk_no_fill"


# ---------------------------------------------------------------------------
# Fix 2: one-sided BBO None guard in engine/market_state.py
# ---------------------------------------------------------------------------

from hlanalysis.engine.market_state import MarketState as EngineMarketState
from hlanalysis.events import BboEvent, ProductType, Mechanism


def _bbo(symbol: str, bid_px, ask_px, ts: int = 1_000_000_000) -> BboEvent:
    return BboEvent(
        venue="binance",
        product_type=ProductType.SPOT,
        mechanism=Mechanism.CLOB,
        symbol=symbol,
        exchange_ts=ts,
        local_recv_ts=ts,
        bid_px=bid_px,
        bid_sz=1.0 if bid_px is not None else None,
        ask_px=ask_px,
        ask_sz=1.0 if ask_px is not None else None,
    )


def test_bbo_none_ask_does_not_raise_type_error():
    """Fix 2: BBO event with ask_px=None must not raise TypeError when computing
    the mid for a bbo-sourced reference symbol.  The mid update must be skipped
    and the previous last_mark retained.
    """
    ms = EngineMarketState()
    ms.set_reference_source("BTC_SPOT", "bbo")
    ms.set_reference_cadence("BTC_SPOT", sampling_dt_seconds=5)

    # First: a valid two-sided BBO to set a baseline last_mark.
    ms.apply(_bbo("BTC_SPOT", bid_px=100.0, ask_px=102.0, ts=1_000_000_000))
    first_mark = ms.last_mark("BTC_SPOT")
    assert first_mark is not None
    assert abs(first_mark - 101.0) < 1e-9  # mid = (100+102)/2

    # Now apply a one-sided BBO (ask_px=None) — must not raise.
    ms.apply(_bbo("BTC_SPOT", bid_px=100.0, ask_px=None, ts=2_000_000_000))

    # last_mark must be UNCHANGED (mid update was skipped).
    after_mark = ms.last_mark("BTC_SPOT")
    assert after_mark == first_mark, (
        f"last_mark changed from {first_mark} to {after_mark} on a partial BBO — mid update should have been skipped."
    )


def test_bbo_none_bid_does_not_raise_type_error():
    """Fix 2: BBO event with bid_px=None must also be handled gracefully."""
    ms = EngineMarketState()
    ms.set_reference_source("BTC_SPOT", "bbo")
    ms.set_reference_cadence("BTC_SPOT", sampling_dt_seconds=5)

    ms.apply(_bbo("BTC_SPOT", bid_px=100.0, ask_px=102.0, ts=1_000_000_000))
    first_mark = ms.last_mark("BTC_SPOT")

    ms.apply(_bbo("BTC_SPOT", bid_px=None, ask_px=102.0, ts=2_000_000_000))
    after_mark = ms.last_mark("BTC_SPOT")
    assert after_mark == first_mark


def test_bbo_mark_sourced_symbol_none_sides_no_error():
    """Fix 2: mark-sourced symbols (default) should be unaffected even if the
    BBO event has None sides — the BBO path for mark-sourced symbols doesn't
    compute a mid, so no change needed, but it must not crash either."""
    ms = EngineMarketState()
    # No set_reference_source → defaults to "mark"
    ms.apply(_bbo("BTC", bid_px=None, ask_px=102.0, ts=1_000_000_000))
    # No exception, last_mark is None (no MarkEvent fed)
    assert ms.last_mark("BTC") is None


# ---------------------------------------------------------------------------
# Fix 3: "cancelled" added to OrderAck.status Literal
# ---------------------------------------------------------------------------

from hlanalysis.engine.exec_types import OrderAck


def test_order_ack_status_cancelled_in_literal():
    """Fix 3: 'cancelled' must be listed in the OrderAck.status Literal so that
    pm_client.py:383's `status='cancelled'` is type-correct.

    Python dataclasses don't enforce Literal at runtime, so we inspect the
    type hint directly — this is the canonical red→green check for a Literal
    expansion fix.
    """
    import typing

    hints = typing.get_type_hints(OrderAck)
    status_literal = hints.get("status")
    assert status_literal is not None
    # get_args returns the values of a Literal, e.g. ("pending", "open", ...)
    allowed = typing.get_args(status_literal)
    assert "cancelled" in allowed, (
        f"'cancelled' is missing from OrderAck.status Literal; "
        f"current values: {allowed}. "
        f"pm_client.py constructs OrderAck(status='cancelled') — this Literal "
        f"mismatch causes mypy errors and misleads IDEs."
    )


def test_order_ack_status_cancelled_constructs():
    """Fix 3: OrderAck(status='cancelled', ...) must be constructable."""
    ack = OrderAck(
        cloid="test-cloid-123",
        venue_oid="venue-oid-456",
        status="cancelled",
    )
    assert ack.status == "cancelled"


def test_order_ack_existing_statuses_still_work():
    """Regression: existing statuses remain valid after adding 'cancelled'."""
    import typing

    hints = typing.get_type_hints(OrderAck)
    allowed = typing.get_args(hints["status"])
    for status in ("pending", "open", "filled", "rejected"):
        assert status in allowed, f"{status!r} dropped from OrderAck.status Literal"
        ack = OrderAck(cloid="c", venue_oid="v", status=status)
        assert ack.status == status


# ---------------------------------------------------------------------------
# Fix 4: _books eviction + Scanner/Router cache pruning
# ---------------------------------------------------------------------------

from hlanalysis.events import QuestionMetaEvent, SettlementEvent


def _make_question_meta(
    question_idx: int,
    yes_symbol: str,
    no_symbol: str,
    expiry_ns: int,
    ts: int,
) -> QuestionMetaEvent:
    from datetime import datetime, timezone

    expiry_str = datetime.fromtimestamp(expiry_ns / 1e9, tz=timezone.utc).strftime("%Y%m%d-%H%M")
    return QuestionMetaEvent(
        venue="hyperliquid",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="qmeta",
        exchange_ts=ts,
        local_recv_ts=ts,
        question_idx=question_idx,
        named_outcome_idxs=[0],
        keys=[
            "class",
            "underlying",
            "period",
            "expiry",
            "strike",
            "yes_token_id" if False else "class",
        ],  # build via kv
        values=["priceBinary", "BTC", "24h", expiry_str, "80000", "priceBinary"],
    )


def _q_meta_hl(
    question_idx: int,
    yes_symbol: str,
    no_symbol: str,
    expiry_ns: int,
    ts: int,
) -> QuestionMetaEvent:
    """Helper that produces a valid HL QuestionMetaEvent."""
    from datetime import datetime, timezone

    expiry_str = datetime.fromtimestamp(expiry_ns / 1e9, tz=timezone.utc).strftime("%Y%m%d-%H%M")
    return QuestionMetaEvent(
        venue="hyperliquid",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="qmeta",
        exchange_ts=ts,
        local_recv_ts=ts,
        question_idx=question_idx,
        named_outcome_idxs=[0],
        keys=["class", "underlying", "period", "expiry", "strike"],
        values=["priceBinary", "BTC", "24h", expiry_str, "80000"],
    )


def test_evict_settled_question_removes_books():
    """Fix 4: evict_settled_questions must also remove _books entries for
    symbols that belong ONLY to the evicted question.
    """
    NOW_NS = 10_000_000_000_000_000
    EXPIRY_NS = NOW_NS - 2 * 3600 * 1_000_000_000  # 2h ago (already expired)
    RETAIN_NS = 1 * 3600 * 1_000_000_000  # retain 1h after settlement

    ms = EngineMarketState()
    # Register a question — HL uses #<N> symbols derived from named_outcome_idxs=[0]
    ms.apply(
        _q_meta_hl(
            question_idx=99,
            yes_symbol="#0",
            no_symbol="#1",
            expiry_ns=EXPIRY_NS,
            ts=EXPIRY_NS - 3600_000_000_000,
        )
    )
    q = ms.question(99)
    assert q is not None

    # Feed BBO events for the question's leg symbols to populate _books
    yes_sym = q.yes_symbol  # "#0"
    no_sym = q.no_symbol  # "#1"
    for sym in (yes_sym, no_sym):
        if sym:
            ms.apply(_bbo(sym, bid_px=0.5, ask_px=0.6, ts=EXPIRY_NS - 1000))

    # Confirm books are populated
    assert ms.book(yes_sym) is not None, "yes_sym book should be present before eviction"

    # Mark the question settled
    ms.mark_question_settled(99)
    assert ms.question(99).settled

    # Evict — expiry is 2h ago, retain window is 1h → should evict
    count = ms.evict_settled_questions(now_ns=NOW_NS, retain_after_settle_ns=RETAIN_NS)
    assert count == 1, f"Expected 1 eviction, got {count}"
    assert ms.question(99) is None

    # The _books entries for the evicted question's leg symbols must be gone
    assert ms.book(yes_sym) is None, f"_books[{yes_sym!r}] still present after question 99 was evicted"


def test_evict_does_not_remove_shared_symbol_books():
    """Fix 4: a symbol shared by two questions must NOT be evicted when only
    one question is evicted (the other still references the symbol).

    Construct two questions that share a YES-leg symbol. Settle+evict only Q1.
    Q2 is still live — its shared symbol's book must remain in _books.
    """
    NOW_NS = 10_000_000_000_000_000
    EXPIRY_OLD_NS = NOW_NS - 2 * 3600 * 1_000_000_000  # 2h ago
    EXPIRY_NEW_NS = NOW_NS + 3600 * 1_000_000_000  # 1h from now (live)
    RETAIN_NS = 1 * 3600 * 1_000_000_000

    ms = EngineMarketState()
    ms.apply(
        _q_meta_hl(
            question_idx=100,
            yes_symbol="#0",
            no_symbol="#1",
            expiry_ns=EXPIRY_OLD_NS,
            ts=EXPIRY_OLD_NS - 3600_000_000_000,
        )
    )
    ms.apply(
        _q_meta_hl(
            question_idx=101,
            yes_symbol="#0",
            no_symbol="#2",
            expiry_ns=EXPIRY_NEW_NS,
            ts=NOW_NS - 1000,
        )
    )

    q100 = ms.question(100)
    shared_sym = q100.yes_symbol  # "#0" - shared between 100 and 101

    # Populate the shared symbol's book
    ms.apply(_bbo(shared_sym, bid_px=0.5, ask_px=0.6, ts=NOW_NS - 1000))
    assert ms.book(shared_sym) is not None

    # Mark only Q100 settled
    ms.mark_question_settled(100)
    count = ms.evict_settled_questions(now_ns=NOW_NS, retain_after_settle_ns=RETAIN_NS)
    assert count == 1

    # Q101 still live — the shared symbol's book must be retained
    assert ms.book(shared_sym) is not None, (
        f"Shared book for {shared_sym!r} must NOT be evicted while Q101 still references it"
    )


def _mk_scanner_cfg() -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC"},
        max_position_usd=100,
        stop_loss_pct=10,
        tte_min_seconds=0,
        tte_max_seconds=86400,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=0,
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
                max_strike_distance_pct=50,
                min_recent_volume_usd=0,
                stale_data_halt_seconds=999,
                reconcile_interval_seconds=60,
            )
        },
    )


def test_scanner_caches_pruned_on_eviction(tmp_path):
    """Fix 4: after a question is evicted from MarketState, Scanner's
    _tradeable_cache, _pm_strike_seen, and _last_logged_state must no longer
    contain that question_idx.

    Wiring choice: Scanner.prune(active_question_idxs) is called from within
    the scan loop's question-iteration step immediately after observing that a
    qidx is no longer in ms.all_questions() — i.e. self-pruning inside the
    scanner's owned scan loop, not relying on runtime.py.
    """
    from hlanalysis.engine.scanner import Scanner
    from hlanalysis.engine.state import StateDAL
    from hlanalysis.strategy.late_resolution import (
        LateResolutionConfig,
        LateResolutionStrategy,
    )
    from hlanalysis.events import MarkEvent

    NOW_NS = 10_000_000_000_000_000
    EXPIRY_NS = NOW_NS - 2 * 3600 * 1_000_000_000
    RETAIN_NS = 1 * 3600 * 1_000_000_000

    ms = EngineMarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=60)
    # Feed some mark events so the scanner can get a reference price
    for i in range(5):
        ts = NOW_NS - (5 - i) * 60_000_000_000
        ms.apply(
            MarkEvent(
                venue="hyperliquid",
                product_type=ProductType.PERP,
                mechanism=Mechanism.CLOB,
                symbol="BTC",
                exchange_ts=ts,
                local_recv_ts=ts,
                mark_px=80_000.0,
            )
        )

    ms.apply(
        _q_meta_hl(
            question_idx=200,
            yes_symbol="#0",
            no_symbol="#1",
            expiry_ns=EXPIRY_NS,
            ts=EXPIRY_NS - 3600_000_000_000,
        )
    )
    q200 = ms.question(200)
    for sym in (q200.yes_symbol, q200.no_symbol):
        if sym:
            ms.apply(_bbo(sym, bid_px=0.5, ask_px=0.6, ts=NOW_NS - 1000))

    cfg = _mk_scanner_cfg()
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    strategy = LateResolutionStrategy(
        LateResolutionConfig(
            tte_min_seconds=0,
            tte_max_seconds=86400,
            price_extreme_threshold=0.0,
            distance_from_strike_usd_min=0.0,
            vol_max=100.0,
            max_position_usd=100.0,
            stop_loss_pct=10.0,
            max_strike_distance_pct=50.0,
            min_recent_volume_usd=0.0,
            stale_data_halt_seconds=999,
        )
    )
    scanner = Scanner(
        strategy=strategy,
        cfg=cfg,
        market_state=ms,
        dal=dal,
        kill_switch_path=tmp_path / "kill",
        last_reconcile_ns=NOW_NS,
        reference_symbol="BTC",
    )

    # Run a scan so question 200 gets into the caches
    scanner.scan(now_ns=NOW_NS)
    # Manually inject into the caches (scan may not populate all if gates block)
    scanner._tradeable_cache[200] = True
    scanner._pm_strike_seen.add(200)
    scanner._last_logged_state[200] = ("HOLD", "no_mark")

    # Confirm they are populated
    assert 200 in scanner._tradeable_cache
    assert 200 in scanner._pm_strike_seen
    assert 200 in scanner._last_logged_state

    # Evict the question from MarketState
    ms.mark_question_settled(200)
    evicted = ms.evict_settled_questions(now_ns=NOW_NS, retain_after_settle_ns=RETAIN_NS)
    assert evicted == 1

    # Call prune directly (as would happen in the self-pruning scan loop)
    active_idxs = {q.question_idx for q in ms.all_questions()}
    scanner.prune(active_idxs)

    assert 200 not in scanner._tradeable_cache, "_tradeable_cache not pruned"
    assert 200 not in scanner._pm_strike_seen, "_pm_strike_seen not pruned"
    assert 200 not in scanner._last_logged_state, "_last_logged_state not pruned"


def test_router_cache_pruned_on_eviction(tmp_path):
    """Fix 4: Router._last_exit_ts and _tradeable_cache (if present) must not
    retain entries for evicted question_idxs after prune() is called.
    """
    from hlanalysis.engine.router import Router
    from hlanalysis.engine.risk import RiskGate
    from hlanalysis.engine.event_bus import EventBus
    from unittest.mock import MagicMock

    from hlanalysis.engine.state import StateDAL

    cfg = _mk_scanner_cfg()
    dal = StateDAL(tmp_path / "state.db")
    gate = RiskGate(cfg)
    bus = EventBus()
    exec_client = MagicMock()

    router = Router(
        dal=dal,
        gate=gate,
        bus=bus,
        exec_client=exec_client,
        strategy_cfg=cfg,
    )

    # Inject stale entries for question 300
    router._last_exit_ts[300] = 1_000_000
    router._consecutive_rejects[(300, "buy")] = 3
    router._last_reject_ts[(300, "buy")] = 1_000_000

    # Now prune question 300 (as if it was evicted from MarketState)
    active_idxs = {100, 200}  # 300 is absent → should be pruned
    router.prune(active_idxs)

    assert 300 not in router._last_exit_ts, "_last_exit_ts not pruned for evicted question"
    # reject-breaker dicts are keyed (qidx, side); also pruned
    assert (300, "buy") not in router._consecutive_rejects, "_consecutive_rejects not pruned"
    assert (300, "buy") not in router._last_reject_ts, "_last_reject_ts not pruned"

    # Retained entries for active questions must be unaffected
    router._last_exit_ts[100] = 999
    router.prune({100, 200})
    assert 100 in router._last_exit_ts
