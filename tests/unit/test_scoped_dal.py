"""Tests for StrategyScopedDAL — per-strategy isolation over a shared DB.

Covers:
  1. Positional scope isolation (two scopes, same CachedStateDAL, same account).
  2. Order scope isolation (live_orders).
  3. Fill / realized-PnL isolation (realized_pnl_since).
  4. Seen-question isolation.
  5. PM-strike isolation.
  6. Journal-row isolation.
  7. Scoped writes persist with correct strategy_id + account tags (raw DB check).
  8. Write-through correctness: a fresh CachedStateDAL over the same file returns
     the same per-strategy data that was written through the scoped DAL.
"""

from __future__ import annotations

import sqlite3
import time

from hlanalysis.engine.scoped_dal import StrategyScopedDAL
from hlanalysis.engine.state import (
    CachedStateDAL,
    Fill,
    OpenOrder,
    Position,
    StateDAL,
    TradeJournalRow,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_shared(tmp_path) -> CachedStateDAL:
    dal = CachedStateDAL(tmp_path / "state.db")
    dal.run_migrations()
    return dal


def _pos(qidx: int, qty: float, strategy_id: str = "") -> Position:
    return Position(
        strategy_id=strategy_id,
        question_idx=qidx,
        symbol=f"#{qidx}",
        qty=qty,
        avg_entry=0.90,
        realized_pnl=0.0,
        last_update_ts_ns=1,
        stop_loss_price=0.80,
    )


def _ord(cloid: str, strategy_id: str = "t") -> OpenOrder:
    return OpenOrder(
        cloid=cloid,
        venue_oid=None,
        question_idx=1,
        symbol="#1",
        side="buy",
        price=0.90,
        size=10.0,
        status="open",
        placed_ts_ns=1,
        last_update_ts_ns=1,
        strategy_id=strategy_id,
    )


def _fill(fill_id: str, strategy_id: str = "", account: str | None = None) -> Fill:
    return Fill(
        fill_id=fill_id,
        cloid="c1",
        question_idx=1,
        symbol="#1",
        side="buy",
        price=0.90,
        size=10.0,
        fee=0.01,
        ts_ns=time.time_ns(),
        closed_pnl=5.0,
        strategy_id=strategy_id or None,
        account=account,
    )


def _journal_row(cloid: str) -> TradeJournalRow:
    return TradeJournalRow(
        cloid=cloid,
        question_idx=1,
        decision_ts_ns=time.time_ns(),
        action="enter",
    )


# ---------------------------------------------------------------------------
# 1. Position scope isolation
# ---------------------------------------------------------------------------


def test_scopes_isolated_position(tmp_path):
    """A writes a position at qidx=5; B reads None for qidx=5."""
    shared = _make_shared(tmp_path)
    a = StrategyScopedDAL(shared, strategy_id="A", account="0xACC")
    b = StrategyScopedDAL(shared, strategy_id="B", account="0xACC")

    a.upsert_position(_pos(5, 10.0))

    assert a.get_position(5) is not None
    assert a.get_position(5).qty == 10.0
    assert b.get_position(5) is None  # B must not see A's row

    # all_positions is also scoped
    assert len(a.all_positions()) == 1
    assert len(b.all_positions()) == 0


def test_scopes_isolated_position_delete(tmp_path):
    """A deletes its position; B's (non-existent) position is unaffected."""
    shared = _make_shared(tmp_path)
    a = StrategyScopedDAL(shared, strategy_id="A", account="0xACC")
    b = StrategyScopedDAL(shared, strategy_id="B", account="0xACC")

    a.upsert_position(_pos(5, 10.0))
    b.upsert_position(_pos(5, 20.0))

    a.delete_position(5)

    assert a.get_position(5) is None
    assert b.get_position(5) is not None
    assert b.get_position(5).qty == 20.0


# ---------------------------------------------------------------------------
# 2. Order scope isolation
# ---------------------------------------------------------------------------


def test_scopes_isolated_orders(tmp_path):
    """live_orders() for A must not include B's orders."""
    shared = _make_shared(tmp_path)
    a = StrategyScopedDAL(shared, strategy_id="A", account="0xACC")
    b = StrategyScopedDAL(shared, strategy_id="B", account="0xACC")

    a.upsert_order(_ord("cloid-a", strategy_id=""))
    b.upsert_order(_ord("cloid-b", strategy_id=""))

    a_live = {o.cloid for o in a.live_orders()}
    b_live = {o.cloid for o in b.live_orders()}

    assert a_live == {"cloid-a"}
    assert b_live == {"cloid-b"}


# ---------------------------------------------------------------------------
# 3. Fill / realized-PnL isolation
# ---------------------------------------------------------------------------


def test_scopes_isolated_realized_pnl(tmp_path):
    """realized_pnl_since for A must not include B's fills."""
    shared = _make_shared(tmp_path)
    a = StrategyScopedDAL(shared, strategy_id="A", account="0xACC")
    b = StrategyScopedDAL(shared, strategy_id="B", account="0xACC")

    a.append_fill(_fill("fill-a"))
    b.append_fill(_fill("fill-b"))

    # closed_pnl=5.0, fee=0.01 → net 4.99 per fill
    pnl_a = a.realized_pnl_since(0)
    pnl_b = b.realized_pnl_since(0)

    # Each scope sees exactly one fill (its own)
    assert abs(pnl_a - 4.99) < 1e-9, f"unexpected pnl_a={pnl_a}"
    assert abs(pnl_b - 4.99) < 1e-9, f"unexpected pnl_b={pnl_b}"

    # fills_count is also scoped
    assert a.fills_count() == 1
    assert b.fills_count() == 1


# ---------------------------------------------------------------------------
# 4. Seen-question isolation
# ---------------------------------------------------------------------------


def test_scopes_isolated_seen_question(tmp_path):
    """A marking qidx=10 as seen must not affect B."""
    shared = _make_shared(tmp_path)
    a = StrategyScopedDAL(shared, strategy_id="A", account="0xACC")
    b = StrategyScopedDAL(shared, strategy_id="B", account="0xACC")

    assert not a.has_seen_question(10)
    assert not b.has_seen_question(10)

    a.mark_question_seen(10, now_ns=time.time_ns())

    assert a.has_seen_question(10)
    assert not b.has_seen_question(10)


# ---------------------------------------------------------------------------
# 5. PM-strike isolation
# ---------------------------------------------------------------------------


def test_scopes_isolated_pm_strike(tmp_path):
    """A's pm_strike for qidx=20 must not be visible to B."""
    shared = _make_shared(tmp_path)
    a = StrategyScopedDAL(shared, strategy_id="A", account="0xACC")
    b = StrategyScopedDAL(shared, strategy_id="B", account="0xACC")

    a.set_pm_strike(20, 73_500.0)

    assert a.get_pm_strike(20) == 73_500.0
    assert b.get_pm_strike(20) is None


# ---------------------------------------------------------------------------
# 6. Trade-journal isolation
# ---------------------------------------------------------------------------


def test_scopes_isolated_journal_rows(tmp_path):
    """A's journal row for cloid='x' must not be visible to B."""
    shared = _make_shared(tmp_path)
    a = StrategyScopedDAL(shared, strategy_id="A", account="0xACC")
    b = StrategyScopedDAL(shared, strategy_id="B", account="0xACC")

    a.add_journal_decision(_journal_row("cloid-x"))

    assert a.get_journal_row("cloid-x") is not None
    # B cannot see A's journal row (different strategy_id)
    assert b.get_journal_row("cloid-x") is None


def test_scopes_isolated_journal_delete(tmp_path):
    """B.delete_journal_decision must not remove A's row."""
    shared = _make_shared(tmp_path)
    a = StrategyScopedDAL(shared, strategy_id="A", account="0xACC")
    b = StrategyScopedDAL(shared, strategy_id="B", account="0xACC")

    a.add_journal_decision(_journal_row("shared-cloid"))
    # B tries to delete a row it doesn't own → must be a no-op
    b.delete_journal_decision("shared-cloid")

    assert a.get_journal_row("shared-cloid") is not None


# ---------------------------------------------------------------------------
# 7. Writes persist with correct tags (raw DB check)
# ---------------------------------------------------------------------------


def test_scope_writes_stamped_in_db(tmp_path):
    """After A writes, the raw DB row carries strategy_id='A' and account='0xACC'."""
    shared = _make_shared(tmp_path)
    a = StrategyScopedDAL(shared, strategy_id="A", account="0xACC")

    a.upsert_position(_pos(5, 10.0))
    a.append_fill(_fill("fill-tag-test"))

    db_path = str(tmp_path / "state.db")
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT strategy_id, account FROM position WHERE question_idx=5").fetchone()
        assert row is not None, "position row missing"
        assert row[0] == "A", f"strategy_id wrong: {row[0]}"
        assert row[1] == "0xACC", f"account wrong: {row[1]}"

        fill_row = conn.execute("SELECT strategy_id, account FROM fill WHERE fill_id='fill-tag-test'").fetchone()
        assert fill_row is not None, "fill row missing"
        assert fill_row[0] == "A", f"fill strategy_id wrong: {fill_row[0]}"
        assert fill_row[1] == "0xACC", f"fill account wrong: {fill_row[1]}"


def test_settlement_stamps_strategy_id(tmp_path):
    """record_settlement persists the correct strategy_id on the settlement row."""
    shared = _make_shared(tmp_path)
    a = StrategyScopedDAL(shared, strategy_id="A", account="0xACC")
    b = StrategyScopedDAL(shared, strategy_id="B", account="0xACC")

    a.record_settlement(question_idx=99, symbol="#99", realized_pnl=12.50, ts_ns=1)

    assert abs(a.settlement_pnl_since(0) - 12.50) < 1e-9
    assert abs(b.settlement_pnl_since(0)) < 1e-9  # B sees nothing

    with sqlite3.connect(str(tmp_path / "state.db")) as conn:
        row = conn.execute("SELECT strategy_id FROM settlement WHERE question_idx=99").fetchone()
        assert row is not None
        assert row[0] == "A"


# ---------------------------------------------------------------------------
# 8. Write-through correctness: fresh CachedStateDAL sees same data
# ---------------------------------------------------------------------------


def test_write_through_fresh_cached_dal(tmp_path):
    """After A writes via CachedStateDAL, a fresh CachedStateDAL sees the data."""
    shared = _make_shared(tmp_path)
    a = StrategyScopedDAL(shared, strategy_id="A", account="0xACC")

    a.upsert_position(_pos(7, 5.0))
    a.upsert_order(_ord("cloid-z"))

    # Fresh CachedStateDAL over the same file (simulates restart)
    fresh_shared = CachedStateDAL(tmp_path / "state.db")
    fresh_a = StrategyScopedDAL(fresh_shared, strategy_id="A", account="0xACC")
    fresh_b = StrategyScopedDAL(fresh_shared, strategy_id="B", account="0xACC")

    assert fresh_a.get_position(7) is not None
    assert fresh_a.get_position(7).qty == 5.0
    assert fresh_b.get_position(7) is None

    assert {o.cloid for o in fresh_a.live_orders()} == {"cloid-z"}
    assert list(fresh_b.live_orders()) == []


# ---------------------------------------------------------------------------
# 9. Global pass-throughs (migrations, prune_events) are not scoped
# ---------------------------------------------------------------------------


def test_run_migrations_delegates_to_base(tmp_path):
    """run_migrations() on a scoped DAL runs against the underlying base."""
    base = StateDAL(tmp_path / "state2.db")
    scoped = StrategyScopedDAL(base, strategy_id="X", account="0x0")
    # Should not raise; tables will be created
    scoped.run_migrations()
    assert scoped.applied_versions()


def test_prune_events_global(tmp_path):
    """prune_events() is global (no strategy filter) — it bounds the whole table."""
    shared = _make_shared(tmp_path)
    a = StrategyScopedDAL(shared, strategy_id="A", account="0xACC")
    b = StrategyScopedDAL(shared, strategy_id="B", account="0xACC")

    # Write one event each
    a.append_event(
        ts_ns=1,
        alias="v1",
        kind="entry",
        question_idx=1,
        reason=None,
        payload_json=None,
    )
    b.append_event(
        ts_ns=2,
        alias="v31",
        kind="entry",
        question_idx=2,
        reason=None,
        payload_json=None,
    )

    # Prune with max_rows=0 → removes everything globally
    a.prune_events(max_age_ns=0, max_rows=0)

    # Both slots' events are gone (global prune)
    assert a.events_since(0) == []
    assert b.events_since(0) == []
