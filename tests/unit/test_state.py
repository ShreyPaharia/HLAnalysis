from __future__ import annotations

import pytest

from hlanalysis.engine.state import (
    OpenOrder, Position, Fill, Session, StateDAL,
)


@pytest.fixture
def dal(tmp_path):
    db_path = tmp_path / "state.db"
    d = StateDAL(db_path)
    d.run_migrations()
    return d


def test_migrations_creates_tables(dal):
    info = dal.applied_versions()
    assert "0001_initial" in info


def test_open_order_round_trip(dal):
    oo = OpenOrder(
        cloid="hla-1", venue_oid=None, question_idx=42, symbol="@30",
        side="buy", price=0.95, size=10.0, status="pending",
        placed_ts_ns=1, last_update_ts_ns=1, strategy_id="late_resolution",
    )
    dal.upsert_order(oo)
    found = dal.get_order("hla-1")
    assert found is not None
    assert found.status == "pending"
    dal.update_order_status("hla-1", status="open", venue_oid="0xv1", now_ns=2)
    again = dal.get_order("hla-1")
    assert again.status == "open" and again.venue_oid == "0xv1"


def test_open_orders_filter_by_status(dal):
    for i, status in enumerate(["pending", "open", "filled", "cancelled"]):
        dal.upsert_order(OpenOrder(
            cloid=f"hla-{i}", venue_oid=None, question_idx=42, symbol="@30",
            side="buy", price=0.95, size=1.0, status=status,
            placed_ts_ns=i, last_update_ts_ns=i, strategy_id="x",
        ))
    live = dal.live_orders()
    assert {o.cloid for o in live} == {"hla-0", "hla-1"}


def test_position_round_trip(dal):
    dal.upsert_position(Position(
        question_idx=42, symbol="@30", qty=10.0, avg_entry=0.95,
        realized_pnl=0.0, last_update_ts_ns=1, stop_loss_price=0.855,
    ))
    p = dal.get_position(42)
    assert p is not None and p.qty == 10.0


def test_fill_append_and_query(dal):
    dal.append_fill(Fill(
        fill_id="f-1", cloid="hla-1", question_idx=42, symbol="@30",
        side="buy", price=0.95, size=10.0, fee=0.05, ts_ns=1,
    ))
    fills = dal.fills_for_cloid("hla-1")
    assert len(fills) == 1
    assert fills[0].fill_id == "f-1"


def test_cloid_uniqueness_enforced(dal):
    base = dict(
        cloid="hla-dup", venue_oid=None, question_idx=42, symbol="@30",
        side="buy", price=0.95, size=10.0, status="pending",
        placed_ts_ns=1, last_update_ts_ns=1, strategy_id="x",
    )
    oo = OpenOrder(**base)
    dal.upsert_order(oo)
    # Second upsert with the same cloid must update, not duplicate
    oo2 = OpenOrder(**{**base, "price": 0.96})
    dal.upsert_order(oo2)
    assert dal.get_order("hla-dup").price == 0.96
