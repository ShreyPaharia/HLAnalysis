"""Regression pin for the reject circuit-breaker (SHR-45).

The circuit-breaker already exists in Router._place (router.py lines 240-242 for
suppression, 267-292 for increment-on-reject, and _book_fill lines 347-350 for
reset-on-fill).  These tests exercise the REAL Router with a fake exec_client and
lock down the two load-bearing contracts:

  (a) Suppress: after `threshold` consecutive rejects for the same
      (question_idx, side), further handle() calls must NOT call
      exec_client.place — the place-call count must plateau at `threshold`.

  (b) Reset: after any fill on the same question (either side), the reject
      counter is cleared — a subsequent entry attempt places again.

The test uses reject_breaker_threshold=3 to keep the sequence short.
Note: the counter is shared across ALL error strings for a given (question, side)
— there is intentionally no per-error-class bucketing.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from hlanalysis.engine.config import (
    AllowlistEntry, GlobalRiskConfig, StrategyConfig,
)
from hlanalysis.engine.event_bus import EventBus
from hlanalysis.engine.exec_types import ClearinghouseState, OrderAck, PlaceRequest
from hlanalysis.engine.risk import RiskGate, RiskInputs
from hlanalysis.engine.router import Router
from hlanalysis.engine.state import Position, StateDAL
from hlanalysis.strategy.types import (
    Action, BookState, Decision, OrderIntent, QuestionView,
)

# ---------------------------------------------------------------------------
# Shared test-harness helpers (mirrored from test_router.py so this file is
# self-contained and can run in isolation).
# ---------------------------------------------------------------------------

_THRESHOLD = 3


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


def _q() -> QuestionView:
    return QuestionView(
        question_idx=42, yes_symbol="@30", no_symbol="@31",
        strike=80_000.0, expiry_ns=10_000_000_000_000_001 + 600_000_000_000,
        underlying="BTC", klass="priceBinary", period="1h",
    )


def _approval_inputs() -> RiskInputs:
    return RiskInputs(
        question=_q(),
        question_fields={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        reference_price=80_300.0,
        book=BookState(
            symbol="@30", bid_px=0.94, bid_sz=10.0, ask_px=0.95, ask_sz=10.0,
            last_trade_ts_ns=10_000_000_000_000_000,
            last_l2_ts_ns=10_000_000_000_000_000,
        ),
        recent_volume_usd=5_000.0,
        positions=[],
        live_orders_total_notional=0.0,
        realized_pnl_today=0.0,
        kill_switch_active=False,
        last_reconcile_ns=10_000_000_000_000_000,
        now_ns=10_000_000_000_000_001,
    )


def _held(dal: StateDAL, question_idx: int, symbol: str, qty: float) -> None:
    dal.upsert_position(Position(
        question_idx=question_idx, symbol=symbol, qty=qty, avg_entry=0.9,
        realized_pnl=0.0, last_update_ts_ns=1, stop_loss_price=-1.0,
    ))


def _exit_intent(cloid: str, question_idx: int = 42, symbol: str = "@30") -> OrderIntent:
    return OrderIntent(
        question_idx=question_idx, symbol=symbol, side="sell", size=10.0,
        limit_price=0.99, cloid=cloid, time_in_force="ioc", reduce_only=True,
        exit_reason="exit_safety_d",
    )


def _enter_intent(cloid: str, question_idx: int = 42, symbol: str = "@30") -> OrderIntent:
    return OrderIntent(
        question_idx=question_idx, symbol=symbol, side="buy", size=10.0,
        limit_price=0.95, cloid=cloid, time_in_force="ioc",
    )


# ---------------------------------------------------------------------------
# Fake exec_client implementations.
# ---------------------------------------------------------------------------

class _RejectingExec:
    """Always returns a rejected ack — models a permanently-unfillable order."""

    paper_mode = False

    def __init__(self) -> None:
        self.placed: list[PlaceRequest] = []

    def place(self, req: PlaceRequest) -> OrderAck:
        self.placed.append(req)
        return OrderAck(cloid=req.cloid, venue_oid="", status="rejected",
                        error="insufficient margin")

    def cancel(self, *, cloid: str, symbol: str) -> bool:
        return True

    def open_orders(self):
        return []

    def clearinghouse_state(self) -> ClearinghouseState:
        return ClearinghouseState(positions=(), account_value_usd=0.0)

    def user_fills(self, *, since_ts_ns: int = 0):
        return []

    def realized_pnl_since(self, since_ts_ns: int) -> float:
        return 0.0


class _ScriptedExec(_RejectingExec):
    """Returns a scripted sequence of statuses (one per call), then rejects.

    A "filled" status returns a real fill ack so _book_fill is invoked and the
    reject counter actually resets.
    """

    def __init__(self, statuses: list[str]) -> None:
        super().__init__()
        self._statuses = list(statuses)

    def place(self, req: PlaceRequest) -> OrderAck:
        self.placed.append(req)
        status = self._statuses.pop(0) if self._statuses else "rejected"
        if status == "filled":
            return OrderAck(cloid=req.cloid, venue_oid="v", status="filled",
                            fill_price=req.price, fill_size=req.size)
        return OrderAck(cloid=req.cloid, venue_oid="", status="rejected",
                        error="rej")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_suppress_after_threshold_consecutive_rejects(tmp_path):
    """Contract (a): place-call count must plateau at `threshold` once the
    (question_idx, side) breaker trips.

    Router._place (line ~241): if consecutive_rejects >= threshold → return early,
    skipping both the DB write and the exec_client.place call.
    """
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    _held(dal, 42, "@30", 10.0)  # position so reduce_only clamp doesn't fire

    client = _RejectingExec()
    cfg = _strategy_cfg()
    router = Router(
        dal=dal, gate=RiskGate(cfg), bus=EventBus(),
        exec_client=client, strategy_cfg=cfg,
        reject_breaker_threshold=_THRESHOLD,
    )

    # Fire 2× threshold attempts; only the first `_THRESHOLD` should place.
    total_attempts = _THRESHOLD * 2
    for i in range(total_attempts):
        await router.handle(
            Decision(action=Action.EXIT, intents=(_exit_intent(f"hla-sb{i}"),)),
            inputs=_approval_inputs(), now_ns=10 + i,
        )

    assert len(client.placed) == _THRESHOLD, (
        f"Expected place-call count to plateau at threshold={_THRESHOLD} "
        f"after {total_attempts} attempts; got {len(client.placed)}. "
        "Breaker suppression (router.py ~line 241) is broken."
    )


@pytest.mark.asyncio
async def test_reset_on_fill_restores_full_budget(tmp_path):
    """Contract (b): a successful fill on the same question clears the reject
    counter so a subsequent bout of rejects gets a fresh budget of `threshold`.

    Router._book_fill (lines ~347-350): filters out all keys where k[0] ==
    intent.question_idx, resetting both sides for that question.

    Sequence:
      rej, rej, FILL  → counter reset
      rej, rej, rej   → tripped again at threshold=3 (3 rejects total post-reset)
      (7th attempt)   → suppressed

    Expected place calls: 2 (pre-fill rejects) + 1 (fill) + 3 (post-reset rejects)
                        = 6 total; the 7th handle() call is suppressed.
    """
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    # Large qty so fills never flatten the position (reduce_only clamp stays inert).
    _held(dal, 42, "@30", 100.0)

    # rej, rej, fill, rej, rej, rej, (7th suppressed)
    client = _ScriptedExec(["rejected", "rejected", "filled",
                            "rejected", "rejected", "rejected", "rejected"])
    cfg = _strategy_cfg()
    router = Router(
        dal=dal, gate=RiskGate(cfg), bus=EventBus(),
        exec_client=client, strategy_cfg=cfg,
        reject_breaker_threshold=_THRESHOLD,
    )

    for i in range(7):
        await router.handle(
            Decision(action=Action.EXIT, intents=(_exit_intent(f"hla-rf{i}"),)),
            inputs=_approval_inputs(), now_ns=10 + i,
        )

    assert len(client.placed) == 6, (
        f"Expected 6 place calls (2 pre-fill + 1 fill + 3 post-reset) "
        f"with threshold={_THRESHOLD}; got {len(client.placed)}. "
        "Fill-driven reset (router.py _book_fill ~line 347) is broken."
    )


@pytest.mark.asyncio
async def test_breaker_scoped_per_question_side_does_not_gag_other_questions(tmp_path):
    """The breaker key is (question_idx, side); a tripped breaker on q=42/sell
    must not suppress placements on q=99/sell.

    Verifies the dict-key scoping in router.py (_consecutive_rejects keyed by
    (intent.question_idx, intent.side)).
    """
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    _held(dal, 42, "@30", 10.0)
    _held(dal, 99, "@40", 10.0)

    client = _RejectingExec()
    cfg = _strategy_cfg()
    router = Router(
        dal=dal, gate=RiskGate(cfg), bus=EventBus(),
        exec_client=client, strategy_cfg=cfg,
        reject_breaker_threshold=2,
    )

    # Trip q=42 breaker (threshold=2: 2 placed, 3rd suppressed).
    for i in range(3):
        await router.handle(
            Decision(action=Action.EXIT, intents=(_exit_intent(f"hla-q42-{i}", 42, "@30"),)),
            inputs=_approval_inputs(), now_ns=10 + i,
        )
    placed_after_q42 = len(client.placed)
    assert placed_after_q42 == 2, f"Expected 2 placed for q=42 at threshold=2; got {placed_after_q42}"

    # q=99 must still be placeable despite q=42 being tripped.
    q99_inputs = replace(
        _approval_inputs(),
        question=replace(_q(), question_idx=99, yes_symbol="@40"),
    )
    await router.handle(
        Decision(action=Action.EXIT, intents=(_exit_intent("hla-q99-0", 99, "@40"),)),
        inputs=q99_inputs, now_ns=20,
    )
    assert len(client.placed) == placed_after_q42 + 1, (
        "Exit on q=99 was wrongly suppressed by q=42's tripped breaker. "
        "Breaker scoping (router.py _consecutive_rejects keyed by (qidx, side)) is broken."
    )
