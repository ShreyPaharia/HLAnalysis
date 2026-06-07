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
    # applied_versions() reports the current Alembic head (a single revision),
    # so assert a head is recorded rather than a specific revision (which moves
    # with every new migration). The substantive check is that core tables exist.
    assert dal.applied_versions()
    import sqlite3
    with sqlite3.connect(dal.db_path) as conn:
        names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert {"position", "openorder", "fill"} <= names


def test_pm_strike_round_trip_and_upsert(dal):
    # PM up/down open-strikes are captured live at the market open; persisting
    # them lets the engine reload after a restart instead of skipping markets
    # whose open it missed.
    assert dal.get_pm_strike(1000126) is None
    dal.set_pm_strike(1000126, 73_500.0)
    assert dal.get_pm_strike(1000126) == 73_500.0
    # Idempotent upsert (re-capture / re-stamp must not error or duplicate).
    dal.set_pm_strike(1000126, 73_500.0)
    assert dal.get_pm_strike(1000126) == 73_500.0


def test_coin_klass_round_trip_and_upsert(dal):
    # SHR-77: HL outcome-share coins ("#N") are persisted with their market
    # class at QuestionMetaEvent ingest so the daily report can split venue
    # fills (which carry only "#N") by binary vs bucket.
    assert dal.coin_klass_map() == {}
    dal.set_coin_klass(coin="#150", klass="priceBinary", question_idx=1000015)
    dal.set_coin_klass(coin="#151", klass="priceBinary", question_idx=1000015)
    dal.set_coin_klass(coin="#160", klass="priceBucket", question_idx=16)
    assert dal.coin_klass_map() == {
        "#150": "priceBinary",
        "#151": "priceBinary",
        "#160": "priceBucket",
    }
    # Idempotent upsert (re-ingest of the same question must not error/duplicate).
    dal.set_coin_klass(coin="#150", klass="priceBinary", question_idx=1000015)
    assert dal.coin_klass_map()["#150"] == "priceBinary"


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


def test_settlement_persisted_idempotently_and_counted_in_realized_pnl(dal):
    """SHR-53/49: settlement realized PnL must be persisted (not just alerted)
    so the daily-loss gate sees it, and the two close paths (reconcile +
    router._close_settled) must not double-book — first writer per qidx wins."""
    assert dal.realized_pnl_since(0) == 0.0
    dal.record_settlement(question_idx=42, symbol="@30",
                          realized_pnl=-25.0, ts_ns=1_000)
    assert dal.realized_pnl_since(0) == -25.0
    # Second close path for the same qidx must NOT double-book — single row per
    # qidx, last authoritative write wins (the vanished path may write an
    # incomplete value first, then the settle event re-emits the real payout).
    dal.record_settlement(question_idx=42, symbol="@30",
                          realized_pnl=-30.0, ts_ns=1_001)
    assert dal.realized_pnl_since(0) == -30.0  # overwritten, not summed
    # A different settled question accumulates.
    dal.record_settlement(question_idx=43, symbol="@40",
                          realized_pnl=10.0, ts_ns=2_000)
    assert dal.realized_pnl_since(0) == -20.0
    # Window cutoff excludes earlier settlements.
    assert dal.realized_pnl_since(1_500) == 10.0


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
