"""trade_journal — durable per-decision/order live trade journal (SHR-83)

Revision ID: 0005_trade_journal
Revises: 0004_fill_source
Create Date: 2026-06-09

Adds the ``trade_journal`` table: ONE row per emitted order, keyed by the
account-stamped cloid and progressively populated across the
decision → send → fill → (reject) lifecycle. It captures the evaluate() inputs
(book-at-decision top-N levels, σ, recent_returns summary, recent_volume_usd,
reference price), the emitted Decision (action + diagnostics), the venue
send/fill timestamps + fill px/sz, the reject/veto reason, and the slot
halt-state at decision time (stale_data_halt / daily_loss_cap / reject_breaker /
restart_blocked).

One journal serves three Spec-3 consumers at once — decision-parity (sim vs
live), execution-latency calibration (decision_ts → send_ts → fill_ts), and
halt replay. Written off the hot path by ``engine.trade_journal.TradeJournal``
(best-effort; a write failure never blocks order submission).

A fresh DB creates the table here; pre-existing (baseline-stamped) DBs gain it
on upgrade. No rows to backfill — the journal populates forward as the engine
emits decisions.
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0005_trade_journal"
down_revision = "0004_fill_source"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """CREATE TABLE trade_journal (
    cloid                 TEXT PRIMARY KEY,
    question_idx          INTEGER NOT NULL,
    decision_ts_ns        INTEGER NOT NULL,
    action                TEXT NOT NULL,
    side                  TEXT,
    symbol                TEXT,
    intended_size         REAL,
    intended_price        REAL,
    reference_price       REAL,
    recent_volume_usd     REAL,
    sigma                 REAL,
    returns_summary_json  TEXT,
    book_json             TEXT,
    diagnostics_json      TEXT,
    halt_json             TEXT,
    send_ts_ns            INTEGER,
    fill_ts_ns            INTEGER,
    fill_px               REAL,
    fill_sz               REAL,
    reject_reason         TEXT
)"""
    )
    op.execute("CREATE INDEX idx_trade_journal_decision_ts ON trade_journal(decision_ts_ns)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_trade_journal_decision_ts")
    op.execute("DROP TABLE IF EXISTS trade_journal")
