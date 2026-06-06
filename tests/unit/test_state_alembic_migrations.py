"""CRITICAL: Alembic adoption must be byte-compatible with existing live DBs.

state.db files persist live trading state (orders/positions/fills/settlements).
Replacing the hand-written migration runner with Alembic MUST:

  * upgrade/stamp a pre-existing (pre-Alembic) DB with NO data loss and NO
    schema drift — existing production rows are untouched, the schema is left
    bit-identical, and the DB is simply stamped at the Alembic baseline; and
  * build a FRESH DB whose schema is byte-identical to the legacy one.

If these can't both hold, the DB change is unsafe for live data.

``LEGACY_*`` below is a FROZEN snapshot of the exact pre-Alembic production
schema (the six hand-written SQL migrations, applied in order — note the
``fill.closed_pnl`` column was added via ALTER, so it appears appended to the
stored CREATE TABLE text). DO NOT EDIT it to match new code; it is the contract
with databases already on disk in production.
"""
from __future__ import annotations

import sqlite3

from hlanalysis.engine.state import Fill, Position, StateDAL

# --- frozen pre-Alembic production schema -----------------------------------

LEGACY_DDL = """
CREATE TABLE schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at_ns INTEGER NOT NULL
);
CREATE TABLE openorder (
    cloid TEXT PRIMARY KEY,
    venue_oid TEXT,
    question_idx INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    size REAL NOT NULL,
    status TEXT NOT NULL,
    placed_ts_ns INTEGER NOT NULL,
    last_update_ts_ns INTEGER NOT NULL,
    strategy_id TEXT NOT NULL
);
CREATE INDEX idx_openorder_status ON openorder(status);
CREATE INDEX idx_openorder_question ON openorder(question_idx);
CREATE TABLE position (
    question_idx INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    qty REAL NOT NULL,
    avg_entry REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    last_update_ts_ns INTEGER NOT NULL,
    stop_loss_price REAL NOT NULL
);
CREATE TABLE fill (
    fill_id TEXT PRIMARY KEY,
    cloid TEXT NOT NULL,
    question_idx INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    size REAL NOT NULL,
    fee REAL NOT NULL,
    ts_ns INTEGER NOT NULL
);
CREATE INDEX idx_fill_cloid ON fill(cloid);
CREATE TABLE session (
    session_id TEXT PRIMARY KEY,
    started_ts_ns INTEGER NOT NULL,
    ended_ts_ns INTEGER,
    halt_reason TEXT
);
CREATE TABLE seen_question (
    question_idx INTEGER PRIMARY KEY,
    first_seen_ts_ns INTEGER NOT NULL
);
ALTER TABLE fill ADD COLUMN closed_pnl REAL NOT NULL DEFAULT 0.0;
CREATE TABLE pm_strike (
    question_idx INTEGER PRIMARY KEY,
    strike REAL NOT NULL
);
CREATE TABLE settlement (
    question_idx INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    realized_pnl REAL NOT NULL,
    ts_ns INTEGER NOT NULL
);
CREATE TABLE events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ns         INTEGER NOT NULL,
    alias         TEXT,
    kind          TEXT NOT NULL,
    question_idx  INTEGER,
    reason        TEXT,
    payload_json  TEXT
);
CREATE INDEX idx_events_alias_kind_ts
    ON events (alias, kind, ts_ns);
CREATE INDEX idx_events_question_ts
    ON events (question_idx, ts_ns);
"""

LEGACY_VERSIONS = [
    "0001_initial", "0002_seen_question", "0003_fill_closed_pnl",
    "0004_pm_strike", "0005_settlement", "0006_events",
]


def _build_legacy_db(path) -> None:
    """Materialize a DB exactly as the pre-Alembic runner would have left it."""
    conn = sqlite3.connect(path)
    conn.executescript(LEGACY_DDL)
    for v in LEGACY_VERSIONS:
        conn.execute(
            "INSERT INTO schema_migrations(version, applied_at_ns) VALUES (?, ?)",
            (v, 1),
        )
    conn.commit()
    conn.close()


def _app_schema(path) -> dict[str, str]:
    """name -> normalized CREATE sql for every app object, excluding the
    migration-bookkeeping tables (schema_migrations / alembic_version) and
    sqlite internals. This is the thing that must NOT drift."""
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' "
            "AND name NOT IN ('schema_migrations', 'alembic_version') "
            "AND sql IS NOT NULL "
            "ORDER BY name"
        ).fetchall()
    finally:
        conn.close()
    return {name: sql for name, sql in rows}


def _table_names(path) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    finally:
        conn.close()


def test_existing_db_upgrades_without_data_loss_or_schema_drift(tmp_path):
    db = tmp_path / "state.db"
    _build_legacy_db(db)

    # Seed live data spanning multiple tables, incl. the ALTER-added closed_pnl.
    seed = StateDAL(db)
    seed.upsert_position(Position(
        question_idx=42, symbol="@30", qty=10.0, avg_entry=0.95,
        realized_pnl=-3.0, last_update_ts_ns=111, stop_loss_price=0.85,
    ))
    seed.set_pm_strike(1000126, 73_500.0)
    seed.append_fill(Fill(
        fill_id="f-1", cloid="hla-1", question_idx=42, symbol="@30",
        side="buy", price=0.95, size=10.0, fee=0.05, ts_ns=222, closed_pnl=-2.5,
    ))
    seed.record_settlement(question_idx=42, symbol="@30",
                           realized_pnl=-5.0, ts_ns=333)

    schema_before = _app_schema(db)

    # Adopt Alembic on an existing DB: must stamp at baseline, NOT re-run DDL.
    dal = StateDAL(db)
    dal.run_migrations()

    # 1) No schema drift — every app object's CREATE text is unchanged.
    assert _app_schema(db) == schema_before

    # 2) Stamped at the Alembic baseline (version table now present).
    assert "alembic_version" in _table_names(db)

    # 3) No data loss.
    assert dal.get_position(42).qty == 10.0
    assert dal.get_position(42).realized_pnl == -3.0
    assert dal.get_pm_strike(1000126) == 73_500.0
    fills = dal.fills_for_cloid("hla-1")
    assert len(fills) == 1 and fills[0].closed_pnl == -2.5
    assert dal.settlement_pnl_since(0) == -5.0

    # 4) Idempotent — a second run is a no-op (already at head).
    dal.run_migrations()
    assert _app_schema(db) == schema_before


def test_fresh_db_schema_is_byte_identical_to_legacy(tmp_path):
    fresh = tmp_path / "fresh.db"
    StateDAL(fresh).run_migrations()

    legacy = tmp_path / "legacy.db"
    _build_legacy_db(legacy)

    # A from-scratch Alembic upgrade must reproduce the exact production schema.
    assert _app_schema(fresh) == _app_schema(legacy)
    # And it tracks state in alembic_version, not the old schema_migrations.
    assert "alembic_version" in _table_names(fresh)
    assert "schema_migrations" not in _table_names(fresh)
