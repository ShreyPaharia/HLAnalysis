"""#28: Concurrent-position / inventory caps are checked against a STALE
pre-fill snapshot taken once per tick.

When two questions both fire ENTER in the same scan tick with a 1-position cap,
only ONE entry intent must be emitted with an empty positions list. The second
entry intent must be emitted with an augmented positions list (containing the
first intent as a pending position) so the Router's risk gate sees the reduced
concurrent budget and vetoes it.

The fix is in scanner.py: as ENTER intents are emitted within a tick, the
scanner tracks in-tick pending positions and passes an augmented positions list
to RiskInputs for subsequent questions. The risk gate then sees the reduced
budget and vetoes the second entry.

We test at two levels:
1. RiskGate: with an augmented positions list the gate correctly blocks a second
   entry when cap=1 (unit test — gate logic correctness).
2. Scanner integration: seed two tradeable questions, AlwaysEnter strategy, cap=1.
   Verify the second ENTER decision's inputs.positions includes the first
   question's pending position — proving the budget was decremented.
"""
from __future__ import annotations

from hlanalysis.engine.config import (
    AllowlistEntry,
    GlobalRiskConfig,
    StrategyConfig,
)
from hlanalysis.engine.risk import RiskGate, RiskInputs
from hlanalysis.strategy.types import (
    Action,
    BookState,
    Decision,
    OrderIntent,
    Position,
    QuestionView,
)

# ── Unit-level: RiskGate concurrent cap logic ─────────────────────────────────

NOW = 10_000_000_000_000_000


def _strategy_cfg(max_concurrent: int = 1) -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        max_position_usd=200,
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
                max_total_inventory_usd=1000,
                max_concurrent_positions=max_concurrent,
                daily_loss_cap_usd=200,
                max_strike_distance_pct=10,
                min_recent_volume_usd=0,
                stale_data_halt_seconds=5,
                reconcile_interval_seconds=60,
            )
        },
    )


def _q(question_idx: int, yes_symbol: str) -> QuestionView:
    return QuestionView(
        question_idx=question_idx,
        yes_symbol=yes_symbol,
        no_symbol=yes_symbol + "_no",
        strike=80_000.0,
        expiry_ns=NOW + 600_000_000_000,
        underlying="BTC",
        klass="priceBinary",
        period="1h",
    )


def _book(symbol: str) -> BookState:
    return BookState(
        symbol=symbol,
        bid_px=0.94,
        bid_sz=10.0,
        ask_px=0.95,
        ask_sz=10.0,
        last_trade_ts_ns=NOW,
        last_l2_ts_ns=NOW,
    )


def _entry_intent(question_idx: int, symbol: str) -> OrderIntent:
    return OrderIntent(
        question_idx=question_idx,
        symbol=symbol,
        side="buy",
        size=100.0,
        limit_price=0.95,
        cloid=f"hla-test-{question_idx}",
        time_in_force="ioc",
        reduce_only=False,
    )


def _inputs(
    question_idx: int,
    symbol: str,
    positions: list[Position],
) -> RiskInputs:
    return RiskInputs(
        question=_q(question_idx, symbol),
        question_fields={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        reference_price=80_300.0,
        book=_book(symbol),
        recent_volume_usd=5_000.0,
        positions=positions,
        live_orders_total_notional=0.0,
        realized_pnl_today=0.0,
        kill_switch_active=False,
        last_reconcile_ns=NOW - 1_000_000,
        now_ns=NOW,
    )


def test_second_intent_blocked_when_first_approved_and_cap_is_one():
    """#28 gate-level: If the first ENTER is approved, the second must be blocked
    by concurrent cap when the scanner passes the augmented positions list.

    Simulates what the scanner must do: after approving the first intent,
    add a synthetic pending Position to the positions list for the second."""
    gate = RiskGate(_strategy_cfg(max_concurrent=1))

    # First intent (0 positions in DB — both see the same stale snapshot without fix)
    inp1 = _inputs(question_idx=42, symbol="@30", positions=[])
    v1 = gate.check_pre_trade(_entry_intent(42, "@30"), inp1)
    assert v1.approved is True, "First intent must be approved with empty DB"

    # Simulate scanner building a synthetic pending position for q42
    pending = Position(
        question_idx=42,
        symbol="@30",
        qty=100.0,
        avg_entry=0.95,
        stop_loss_price=0.0,
        last_update_ts_ns=NOW,
    )
    # Second intent sees augmented positions list (with the pending entry)
    inp2 = _inputs(question_idx=43, symbol="@32", positions=[pending])
    v2 = gate.check_pre_trade(_entry_intent(43, "@32"), inp2)
    assert v2.approved is False
    assert "concurrent" in v2.reason


def test_two_entries_both_allowed_when_cap_is_two():
    """#28 regression guard: when cap=2, two simultaneous entries must both pass."""
    gate = RiskGate(_strategy_cfg(max_concurrent=2))

    inp1 = _inputs(question_idx=42, symbol="@30", positions=[])
    v1 = gate.check_pre_trade(_entry_intent(42, "@30"), inp1)
    assert v1.approved is True

    pending = Position(
        question_idx=42,
        symbol="@30",
        qty=100.0,
        avg_entry=0.95,
        stop_loss_price=0.0,
        last_update_ts_ns=NOW,
    )
    inp2 = _inputs(question_idx=43, symbol="@32", positions=[pending])
    v2 = gate.check_pre_trade(_entry_intent(43, "@32"), inp2)
    assert v2.approved is True


# ── Scanner integration: two simultaneous ENTER, cap=1 ───────────────────────


def _make_scanner_with_two_questions(tmp_path, max_concurrent: int = 1):
    """Build a Scanner seeded with two tradeable binary questions and a strategy
    that fires ENTER for both.

    Returns (scanner, now_ns).
    """
    from datetime import datetime, timezone

    from hlanalysis.engine.market_state import MarketState
    from hlanalysis.engine.scanner import Scanner
    from hlanalysis.engine.state import StateDAL
    from hlanalysis.events import (
        BboEvent,
        MarkEvent,
        Mechanism,
        ProductType,
        QuestionMetaEvent,
    )
    from hlanalysis.strategy.base import Strategy

    class AlwaysEnterStrategy(Strategy):
        """Toy strategy that always fires ENTER with a 100-unit buy on the yes leg."""

        name = "always_enter"

        def evaluate(
            self,
            *,
            question,
            books,
            reference_price,
            recent_returns,
            recent_hl_bars=(),
            recent_volume_usd=0.0,
            position,
            now_ns,
        ):
            if position is not None:
                return Decision(action=Action.HOLD, intents=())
            yes_sym = question.yes_symbol
            if not yes_sym or yes_sym not in books:
                return Decision(action=Action.HOLD, intents=())
            intent = OrderIntent(
                question_idx=question.question_idx,
                symbol=yes_sym,
                side="buy",
                size=100.0,
                limit_price=0.95,
                cloid=f"hla-test-{question.question_idx}",
                time_in_force="ioc",
                reduce_only=False,
            )
            return Decision(action=Action.ENTER, intents=(intent,))

    now = NOW
    ms = MarketState()
    expiry_str = datetime.fromtimestamp(
        (now + 10 * 60 * 1_000_000_000) / 1e9, tz=timezone.utc
    ).strftime("%Y%m%d-%H%M")

    # Seed two questions with matching allowlist fields
    # q42 uses "#30"/"#31" as underlying symbols (match scanner's HL convention)
    # q43 uses "#32"/"#33"
    for q_idx, yes_sym, no_sym in [
        (42, "#30", "#31"),
        (43, "#32", "#33"),
    ]:
        ms.apply(
            QuestionMetaEvent(
                venue="hyperliquid",
                product_type=ProductType.PREDICTION_BINARY,
                mechanism=Mechanism.CLOB,
                symbol=f"qmeta-{q_idx}",
                exchange_ts=now - 60_000_000_000,
                local_recv_ts=now - 60_000_000_000,
                question_idx=q_idx,
                named_outcome_idxs=[3],
                keys=["class", "underlying", "period", "expiry", "strike"],
                values=["priceBinary", "BTC", "1h", expiry_str, "80000"],
            )
        )
        for is_yes, sym in [(True, yes_sym), (False, no_sym)]:
            ms.apply(
                BboEvent(
                    venue="hyperliquid",
                    product_type=ProductType.PREDICTION_BINARY,
                    mechanism=Mechanism.CLOB,
                    symbol=sym,
                    exchange_ts=now,
                    local_recv_ts=now,
                    bid_px=0.94 if is_yes else 0.04,
                    bid_sz=10.0,
                    ask_px=0.95 if is_yes else 0.05,
                    ask_sz=10.0,
                )
            )

    # Seed mark events for σ calculations (need at least a few)
    for i in range(8):
        ts = now - (8 - i) * 60_000_000_000
        ms.apply(
            MarkEvent(
                venue="hyperliquid",
                product_type=ProductType.PERP,
                mechanism=Mechanism.CLOB,
                symbol="BTC",
                exchange_ts=ts,
                local_recv_ts=ts,
                mark_px=80_300.0 + i * 0.01,
            )
        )

    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()

    cfg = _strategy_cfg(max_concurrent=max_concurrent)
    scanner = Scanner(
        strategy=AlwaysEnterStrategy(),
        cfg=cfg,
        market_state=ms,
        dal=dal,
        kill_switch_path=tmp_path / "halt",
        last_reconcile_ns=now - 1_000_000,  # fresh reconcile
    )
    return scanner, now


def test_scanner_second_entry_sees_pending_position_in_inputs(tmp_path):
    """#28: With two simultaneously-favored markets and cap=1, the scanner
    must pass an augmented positions list to the second ENTER decision so the
    Router's risk gate sees the pending slot from the first.

    The first ScannedDecision must have positions=[] (nothing pending yet).
    The second ScannedDecision must have positions containing the first
    question's pending entry (so concurrent cap fires when the Router checks).
    """
    scanner, now = _make_scanner_with_two_questions(tmp_path, max_concurrent=1)
    decisions = scanner.scan(now_ns=now)
    enter_decisions = [d for d in decisions if d.decision.action is Action.ENTER]

    # Both questions emit ENTER from the strategy — two ScannedDecisions
    assert len(enter_decisions) == 2, (
        f"Strategy fires ENTER for both questions; scanner must emit 2 ScannedDecisions. Got {len(enter_decisions)}"
    )

    # The first one sees an empty pending list (no prior in-tick entries)
    first = enter_decisions[0]
    first_q = first.inputs.question.question_idx
    db_positions_count = 0  # DB is empty in this test
    assert len(first.inputs.positions) == db_positions_count, (
        f"First entry should see {db_positions_count} positions (from DB), "
        f"got {len(first.inputs.positions)}"
    )

    # The second one must see the first question's pending position in its
    # inputs.positions list (so the concurrent cap check fires in the Router)
    second = enter_decisions[1]
    second_q = second.inputs.question.question_idx
    assert len(second.inputs.positions) == 1, (
        f"Second entry must see 1 pending position from first intent, "
        f"got {len(second.inputs.positions)}"
    )
    assert second.inputs.positions[0].question_idx == first_q, (
        f"Pending position must be from the first question ({first_q}), "
        f"got {second.inputs.positions[0].question_idx}"
    )


def test_scanner_second_entry_router_veto_with_cap_one(tmp_path):
    """#28: With cap=1, the Router must veto the second ENTER because the
    scanner passes an augmented positions list with the first pending intent.

    This is the end-to-end budget enforcement test.
    """
    from hlanalysis.engine.risk import RiskGate

    scanner, now = _make_scanner_with_two_questions(tmp_path, max_concurrent=1)
    decisions = scanner.scan(now_ns=now)
    enter_decisions = [d for d in decisions if d.decision.action is Action.ENTER]
    assert len(enter_decisions) == 2

    gate = RiskGate(_strategy_cfg(max_concurrent=1))

    verdicts = [
        gate.check_pre_trade(d.decision.intents[0], d.inputs)
        for d in enter_decisions
    ]
    # First must be approved; second must be vetoed (concurrent cap)
    assert verdicts[0].approved is True, f"First intent must pass, got {verdicts[0].reason}"
    assert verdicts[1].approved is False, f"Second intent must be vetoed, got {verdicts[1].reason}"
    assert "concurrent" in verdicts[1].reason


def test_scanner_cap2_both_entries_pass_gate(tmp_path):
    """#28 regression guard: with cap=2, both entries must pass the Router's
    gate even with the augmented positions list from the fix."""
    from hlanalysis.engine.risk import RiskGate

    scanner, now = _make_scanner_with_two_questions(tmp_path, max_concurrent=2)
    decisions = scanner.scan(now_ns=now)
    enter_decisions = [d for d in decisions if d.decision.action is Action.ENTER]
    assert len(enter_decisions) == 2

    gate = RiskGate(_strategy_cfg(max_concurrent=2))
    verdicts = [
        gate.check_pre_trade(d.decision.intents[0], d.inputs)
        for d in enter_decisions
    ]
    assert all(v.approved for v in verdicts), (
        f"Both intents must pass with cap=2: {[v.reason for v in verdicts]}"
    )
