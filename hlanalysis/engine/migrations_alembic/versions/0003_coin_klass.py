"""coin_klass — HL outcome coin ("#N") → market class map (SHR-77)

Revision ID: 0003_coin_klass
Revises: 0002_position_closed_qty
Create Date: 2026-06-08

Adds the ``coin_klass`` table so the daily desk report can split each HL slot's
outcome PnL + fill count by market type (priceBinary vs priceBucket). The venue
fill feed reports HIP-4 fills as coin "#N" with no class; the engine stamps this
table at QuestionMetaEvent ingest, where both the leg coins and the class are
known, giving a durable join keyed by the exact "#N" user_fills returns.

A fresh DB creates the table here; pre-existing (baseline-stamped) DBs gain it on
upgrade. Existing rows: none to backfill — the map populates forward as questions
are (re-)ingested; historical unmapped fills show as "unknown" in the report.
"""
from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0003_coin_klass"
down_revision = "0002_position_closed_qty"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """CREATE TABLE coin_klass (
    coin          TEXT PRIMARY KEY,
    klass         TEXT NOT NULL,
    question_idx  INTEGER NOT NULL
)"""
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS coin_klass")
