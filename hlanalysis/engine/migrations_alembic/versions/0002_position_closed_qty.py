"""position.closed_qty — cumulative closed quantity over a position's life

Revision ID: 0002_position_closed_qty
Revises: 0001_baseline
Create Date: 2026-06-07

Adds ``position.closed_qty`` (REAL NOT NULL DEFAULT 0.0). It accumulates the
absolute quantity closed on every partial reduce so the final close's Exit event
can report the TOTAL closed size paired with the cumulative realized PnL —
fixing the misleading ``qty=3 PnL=-$25.80`` alert where the last-lot qty was
shown against the whole-trade PnL.

Existing rows backfill to 0.0 (the default), which is correct: a position open
at migration time has, from the engine's accounting perspective, closed nothing
yet. The ALTER mirrors the ``0003_fill_closed_pnl`` pattern already folded into
the baseline, so the column appends to the stored CREATE text.
"""
from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0002_position_closed_qty"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE position ADD COLUMN closed_qty REAL NOT NULL DEFAULT 0.0"
    )


def downgrade() -> None:
    # SQLite gained ALTER TABLE ... DROP COLUMN in 3.35; the project's sqlite is
    # newer. Downgrade is provided for completeness/symmetry.
    op.execute("ALTER TABLE position DROP COLUMN closed_qty")
