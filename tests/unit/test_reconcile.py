from __future__ import annotations

import pytest

from hlanalysis.engine.hl_client import (
    ClearinghouseState, OpenOrderRow, UserFillRow, VenuePosition,
)
from hlanalysis.engine.reconcile import ReconcileResult, Reconciler
from hlanalysis.engine.state import OpenOrder, Position, StateDAL


@pytest.fixture
def dal(tmp_path):
    d = StateDAL(tmp_path / "state.db")
    d.run_migrations()
    return d


def _row(cloid: str, symbol: str = "@30", price: float = 0.95,
         size: float = 10.0) -> OpenOrderRow:
    return OpenOrderRow(cloid=cloid, venue_oid=f"v-{cloid}", symbol=symbol,
                       side="buy", price=price, size=size, placed_ts_ns=1)


def _db_order(cloid: str, status: str = "open") -> OpenOrder:
    return OpenOrder(
        cloid=cloid, venue_oid=f"v-{cloid}", question_idx=42, symbol="@30",
        side="buy", price=0.95, size=10.0, status=status,
        placed_ts_ns=1, last_update_ts_ns=1, strategy_id="x",
    )


def test_clean_state_no_drift(dal):
    dal.upsert_order(_db_order("hla-1"))
    venue_open = [_row("hla-1")]
    venue_state = ClearinghouseState(positions=(), account_value_usd=0)
    r = Reconciler(dal, fills_lookup=lambda _cloid: [])
    res = r.run(venue_open=venue_open, venue_state=venue_state, now_ns=2)
    assert res.drift_events == []


def test_local_ghost_marks_cancelled_and_emits_drift(dal):
    dal.upsert_order(_db_order("hla-1"))
    r = Reconciler(dal, fills_lookup=lambda _cloid: [])
    res = r.run(venue_open=[], venue_state=ClearinghouseState(positions=(), account_value_usd=0), now_ns=2)
    assert dal.get_order("hla-1").status == "cancelled"
    assert any(d.case == "local_ghost" for d in res.drift_events)


def test_local_ghost_with_known_fill_marks_filled_and_replays(dal):
    dal.upsert_order(_db_order("hla-1"))
    fills = [UserFillRow(
        fill_id="f-1", cloid="hla-1", symbol="@30", side="buy",
        price=0.95, size=10.0, fee=0.05, ts_ns=1,
    )]
    r = Reconciler(dal, fills_lookup=lambda cloid: fills if cloid == "hla-1" else [])
    res = r.run(venue_open=[], venue_state=ClearinghouseState(positions=(), account_value_usd=0), now_ns=2)
    assert dal.get_order("hla-1").status == "filled"
    assert dal.fills_for_cloid("hla-1")  # replayed
    assert any(d.case == "state_mismatch" for d in res.drift_events)


def test_venue_orphan_emits_drift_and_caller_cancels(dal):
    venue_open = [_row("hla-orphan")]
    r = Reconciler(dal, fills_lookup=lambda _cloid: [])
    res = r.run(venue_open=venue_open, venue_state=ClearinghouseState(positions=(), account_value_usd=0), now_ns=2)
    assert any(d.case == "venue_orphan" and d.cloid == "hla-orphan" for d in res.drift_events)
    assert ("hla-orphan", "@30") in res.orphans_to_cancel


def test_state_mismatch_hl_wins(dal):
    dal.upsert_order(_db_order("hla-1"))
    # Same cloid, different price on venue
    res = Reconciler(dal, fills_lookup=lambda _: []).run(
        venue_open=[_row("hla-1", price=0.96)],
        venue_state=ClearinghouseState(positions=(), account_value_usd=0),
        now_ns=2,
    )
    assert dal.get_order("hla-1").price == 0.96
    assert any(d.case == "state_mismatch" for d in res.drift_events)


def test_position_mismatch_hl_wins(dal):
    dal.upsert_position(Position(
        question_idx=42, symbol="@30", qty=10.0, avg_entry=0.95,
        realized_pnl=0.0, last_update_ts_ns=1, stop_loss_price=0.855,
    ))
    venue_state = ClearinghouseState(
        positions=(VenuePosition(symbol="@30", qty=8.0, avg_entry=0.95, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(dal, fills_lookup=lambda _: [], symbol_to_question={"@30": 42}).run(
        venue_open=[], venue_state=venue_state, now_ns=2,
    )
    assert dal.get_position(42).qty == 8.0
    assert any(d.case == "position_mismatch" for d in res.drift_events)


def test_position_disappearance_drops_local_position(dal):
    dal.upsert_position(Position(
        question_idx=42, symbol="@30", qty=10.0, avg_entry=0.95,
        realized_pnl=0.0, last_update_ts_ns=1, stop_loss_price=0.855,
    ))
    res = Reconciler(dal, fills_lookup=lambda _: [], symbol_to_question={"@30": 42}).run(
        venue_open=[], venue_state=ClearinghouseState(positions=(), account_value_usd=0), now_ns=2,
    )
    assert dal.get_position(42) is None
    assert any(d.case == "position_mismatch" for d in res.drift_events)
