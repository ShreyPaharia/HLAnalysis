"""baseline — exact pre-Alembic engine state.db schema

Revision ID: 0001_baseline
Revises:
Create Date: 2026-06-06

This baseline reproduces the EXACT schema produced by the six hand-written SQL
migrations (0001_initial .. 0006_events) applied in order. The statements are
replayed verbatim — including the ALTER that appends fill.closed_pnl — so a
fresh ``alembic upgrade head`` yields a sqlite_master byte-identical to a
pre-Alembic production DB. Existing DBs are STAMPED at this revision (not
re-run) by StateDAL.run_migrations, leaving their data and schema untouched.

DO NOT autogenerate or "tidy" these statements: byte-for-byte schema parity
with databases already on disk is the safety contract (see
tests/unit/test_state_alembic_migrations.py).
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


# Each statement's text matches the original migration files so SQLite stores
# the identical CREATE/ALTER text in sqlite_master.
_UPGRADE_STATEMENTS = [
    # --- 0001_initial (schema_migrations omitted — Alembic owns versioning) ---
    """CREATE TABLE openorder (
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
)""",
    "CREATE INDEX idx_openorder_status ON openorder(status)",
    "CREATE INDEX idx_openorder_question ON openorder(question_idx)",
    """CREATE TABLE position (
    question_idx INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    qty REAL NOT NULL,
    avg_entry REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    last_update_ts_ns INTEGER NOT NULL,
    stop_loss_price REAL NOT NULL
)""",
    """CREATE TABLE fill (
    fill_id TEXT PRIMARY KEY,
    cloid TEXT NOT NULL,
    question_idx INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    size REAL NOT NULL,
    fee REAL NOT NULL,
    ts_ns INTEGER NOT NULL
)""",
    "CREATE INDEX idx_fill_cloid ON fill(cloid)",
    """CREATE TABLE session (
    session_id TEXT PRIMARY KEY,
    started_ts_ns INTEGER NOT NULL,
    ended_ts_ns INTEGER,
    halt_reason TEXT
)""",
    # --- 0002_seen_question ---
    """CREATE TABLE seen_question (
    question_idx INTEGER PRIMARY KEY,
    first_seen_ts_ns INTEGER NOT NULL
)""",
    # --- 0003_fill_closed_pnl (ALTER → column appended to stored CREATE text) ---
    "ALTER TABLE fill ADD COLUMN closed_pnl REAL NOT NULL DEFAULT 0.0",
    # --- 0004_pm_strike ---
    """CREATE TABLE pm_strike (
    question_idx INTEGER PRIMARY KEY,
    strike REAL NOT NULL
)""",
    # --- 0005_settlement ---
    """CREATE TABLE settlement (
    question_idx INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    realized_pnl REAL NOT NULL,
    ts_ns INTEGER NOT NULL
)""",
    # --- 0006_events ---
    """CREATE TABLE events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ns         INTEGER NOT NULL,
    alias         TEXT,
    kind          TEXT NOT NULL,
    question_idx  INTEGER,
    reason        TEXT,
    payload_json  TEXT
)""",
    """CREATE INDEX idx_events_alias_kind_ts
    ON events (alias, kind, ts_ns)""",
    """CREATE INDEX idx_events_question_ts
    ON events (question_idx, ts_ns)""",
]

_DOWNGRADE_TABLES = [
    "events",
    "settlement",
    "pm_strike",
    "seen_question",
    "session",
    "fill",
    "position",
    "openorder",
]


def upgrade() -> None:
    for stmt in _UPGRADE_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    for table in _DOWNGRADE_TABLES:
        op.execute(f"DROP TABLE IF EXISTS {table}")
