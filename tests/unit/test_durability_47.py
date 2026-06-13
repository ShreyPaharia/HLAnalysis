"""Finding #47: Session_.halt_reason is absent from the SQLModel ORM class but
present in the DB DDL (0001_baseline). This means halt_reason written via
end_session() is silently dropped (only a Python attribute, never persisted).

Tests:
  - halt_reason round-trips through the ORM (write via DAL, read back equal)
  - Session_ model has the halt_reason field declared
  - end_session persists halt_reason to the DB column
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hlanalysis.engine.state import Session_, StateDAL


def _dal(tmp_path: Path) -> StateDAL:
    db_path = tmp_path / "state.db"
    d = StateDAL(db_path)
    d.run_migrations()
    return d


def test_session_model_has_halt_reason_field():
    """Session_ SQLModel must declare halt_reason so it's persisted, not just
    set as a transient Python attribute (finding #47)."""
    fields = Session_.model_fields
    assert "halt_reason" in fields, (
        "Session_ ORM model is missing halt_reason field; the DB column exists but the ORM silently drops writes to it"
    )


def test_halt_reason_round_trips_through_orm(tmp_path: Path):
    """Write a Session with halt_reason via end_session, then read it back
    with the ORM and assert halt_reason is equal (not None)."""
    dal = _dal(tmp_path)

    sid = "test-session-001"
    dal.start_session(sid, now_ns=1_000_000_000)
    dal.end_session(sid, now_ns=2_000_000_000, halt_reason="daily_loss_cap")

    # Read back via raw SQL to confirm the column was actually written
    with sqlite3.connect(dal.db_path) as conn:
        row = conn.execute("SELECT halt_reason FROM session WHERE session_id = ?", (sid,)).fetchone()
    assert row is not None, "session row not found after end_session"
    assert row[0] == "daily_loss_cap", (
        f"halt_reason not persisted: expected 'daily_loss_cap', got {row[0]!r}. "
        "Session_.halt_reason field may be missing from the ORM model (finding #47)"
    )


def test_halt_reason_none_when_not_set(tmp_path: Path):
    """A session ended without a halt_reason must have NULL in the column
    (additive-only change; existing behaviour is preserved)."""
    dal = _dal(tmp_path)

    sid = "test-session-002"
    dal.start_session(sid, now_ns=1_000_000_000)
    dal.end_session(sid, now_ns=2_000_000_000)  # no halt_reason

    with sqlite3.connect(dal.db_path) as conn:
        row = conn.execute("SELECT halt_reason FROM session WHERE session_id = ?", (sid,)).fetchone()
    assert row is not None
    assert row[0] is None, f"expected NULL halt_reason when not set, got {row[0]!r}"


def test_halt_reason_readable_via_orm(tmp_path: Path):
    """halt_reason must be readable via the ORM Session_ object, not just raw SQL."""
    from sqlmodel import select
    from sqlmodel import Session as _Session

    dal = _dal(tmp_path)
    sid = "test-session-003"
    dal.start_session(sid, now_ns=1_000_000_000)
    dal.end_session(sid, now_ns=2_000_000_000, halt_reason="stop_loss")

    with _Session(dal._engine) as s:
        row = s.get(Session_, sid)

    assert row is not None
    assert row.halt_reason == "stop_loss", (
        f"halt_reason not readable via ORM: expected 'stop_loss', got {row.halt_reason!r}"
    )
