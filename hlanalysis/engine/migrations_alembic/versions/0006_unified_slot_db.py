"""unified_slot_db — add account/strategy_id scoping columns to all state tables

Revision ID: 0006_unified_slot_db
Revises: 0005_trade_journal
Create Date: 2026-06-15

Prepares the engine state schema for the unified shared DB whose rows are tagged
(account, strategy_id).  This migration only alters the schema — it does NOT
update ORM models (Task 2) or add a scoped DAL (Task 3).

Changes per table
-----------------
* ``open_order``   — add ``account`` TEXT (NULL)  [strategy_id already exists]
* ``fill``         — add ``account``, ``strategy_id`` TEXT (NULL)
* ``trade_journal``— add ``account``, ``strategy_id`` TEXT (NULL); PK stays cloid
* ``position``     — add ``account``, ``strategy_id``; PK → (strategy_id, question_idx)
* ``seen_question``— add ``account``, ``strategy_id``; PK → (strategy_id, question_idx)
* ``pm_strike``    — add ``account``, ``strategy_id``; PK → (strategy_id, question_idx)
* ``settlement``   — add ``account``, ``strategy_id``; PK → (strategy_id, question_idx)
* ``coin_klass``   — add ``account``, ``strategy_id``; PK → (strategy_id, coin)
* ``events``       — add ``strategy_id`` TEXT (NULL); backfill from existing ``alias``

New columns default NULL — existing rows have no scope yet; later tasks or a
one-off data-migration script will stamp them.

SQLite cannot ALTER a primary key in place, so all composite-PK changes use
Alembic batch mode (``recreate="always"``).

Downgrade is best-effort: simple column drops where possible, batch-mode
recreation to restore the original single-column PK where the batch ALTER was
used.  Dropping a column added by ALTER is supported from SQLite 3.35+; the
project's SQLite satisfies that.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0006_unified_slot_db"
down_revision = "0005_trade_journal"
branch_labels = None
depends_on = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STR_NULL = sa.Column(sa.String(), nullable=True)


def _add_account_strategy(table: str) -> None:
    """Add nullable account + strategy_id columns to a table via raw ALTER.

    Used for tables whose PK does NOT change (simple column append).
    """
    op.execute(f"ALTER TABLE {table} ADD COLUMN account TEXT")
    op.execute(f"ALTER TABLE {table} ADD COLUMN strategy_id TEXT")


def _drop_account_strategy(table: str) -> None:
    """Drop account + strategy_id columns (downgrade helper)."""
    op.execute(f"ALTER TABLE {table} DROP COLUMN strategy_id")
    op.execute(f"ALTER TABLE {table} DROP COLUMN account")


# ---------------------------------------------------------------------------
# upgrade
# ---------------------------------------------------------------------------


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. open_order — strategy_id already exists; add account only.
    # ------------------------------------------------------------------
    op.execute("ALTER TABLE openorder ADD COLUMN account TEXT")

    # ------------------------------------------------------------------
    # 2. fill — add account + strategy_id (PK stays fill_id).
    # ------------------------------------------------------------------
    _add_account_strategy("fill")

    # ------------------------------------------------------------------
    # 3. trade_journal — add account + strategy_id (PK stays cloid).
    # ------------------------------------------------------------------
    _add_account_strategy("trade_journal")

    # ------------------------------------------------------------------
    # 4. position — add account + strategy_id; PK → (strategy_id, question_idx).
    #    SQLite cannot ALTER PK in place → batch mode with recreate="always".
    # ------------------------------------------------------------------
    with op.batch_alter_table("position", recreate="always") as b:
        b.add_column(sa.Column("account", sa.String(), nullable=True))
        b.add_column(sa.Column("strategy_id", sa.String(), nullable=True))
        b.create_primary_key("pk_position", ["strategy_id", "question_idx"])

    # ------------------------------------------------------------------
    # 5. seen_question — add account + strategy_id; PK → (strategy_id, question_idx).
    # ------------------------------------------------------------------
    with op.batch_alter_table("seen_question", recreate="always") as b:
        b.add_column(sa.Column("account", sa.String(), nullable=True))
        b.add_column(sa.Column("strategy_id", sa.String(), nullable=True))
        b.create_primary_key("pk_seen_question", ["strategy_id", "question_idx"])

    # ------------------------------------------------------------------
    # 6. pm_strike — add account + strategy_id; PK → (strategy_id, question_idx).
    # ------------------------------------------------------------------
    with op.batch_alter_table("pm_strike", recreate="always") as b:
        b.add_column(sa.Column("account", sa.String(), nullable=True))
        b.add_column(sa.Column("strategy_id", sa.String(), nullable=True))
        b.create_primary_key("pk_pm_strike", ["strategy_id", "question_idx"])

    # ------------------------------------------------------------------
    # 7. settlement — add account + strategy_id; PK → (strategy_id, question_idx).
    # ------------------------------------------------------------------
    with op.batch_alter_table("settlement", recreate="always") as b:
        b.add_column(sa.Column("account", sa.String(), nullable=True))
        b.add_column(sa.Column("strategy_id", sa.String(), nullable=True))
        b.create_primary_key("pk_settlement", ["strategy_id", "question_idx"])

    # ------------------------------------------------------------------
    # 8. coin_klass — add account + strategy_id; PK → (strategy_id, coin).
    # ------------------------------------------------------------------
    with op.batch_alter_table("coin_klass", recreate="always") as b:
        b.add_column(sa.Column("account", sa.String(), nullable=True))
        b.add_column(sa.Column("strategy_id", sa.String(), nullable=True))
        b.create_primary_key("pk_coin_klass", ["strategy_id", "coin"])

    # ------------------------------------------------------------------
    # 9. events — add strategy_id; backfill from existing alias column.
    # ------------------------------------------------------------------
    op.execute("ALTER TABLE events ADD COLUMN strategy_id TEXT")
    op.execute("UPDATE events SET strategy_id = alias")


# ---------------------------------------------------------------------------
# downgrade (best-effort)
# ---------------------------------------------------------------------------


def downgrade() -> None:
    # Undo in reverse order.

    # 9. events
    op.execute("ALTER TABLE events DROP COLUMN strategy_id")

    # 8. coin_klass — restore PK to single column (coin).
    with op.batch_alter_table("coin_klass", recreate="always") as b:
        b.drop_column("strategy_id")
        b.drop_column("account")
        b.create_primary_key("pk_coin_klass", ["coin"])

    # 7. settlement — restore PK to question_idx.
    with op.batch_alter_table("settlement", recreate="always") as b:
        b.drop_column("strategy_id")
        b.drop_column("account")
        b.create_primary_key("pk_settlement", ["question_idx"])

    # 6. pm_strike — restore PK to question_idx.
    with op.batch_alter_table("pm_strike", recreate="always") as b:
        b.drop_column("strategy_id")
        b.drop_column("account")
        b.create_primary_key("pk_pm_strike", ["question_idx"])

    # 5. seen_question — restore PK to question_idx.
    with op.batch_alter_table("seen_question", recreate="always") as b:
        b.drop_column("strategy_id")
        b.drop_column("account")
        b.create_primary_key("pk_seen_question", ["question_idx"])

    # 4. position — restore PK to question_idx.
    with op.batch_alter_table("position", recreate="always") as b:
        b.drop_column("strategy_id")
        b.drop_column("account")
        b.create_primary_key("pk_position", ["question_idx"])

    # 3. trade_journal — drop account + strategy_id (PK stays cloid).
    _drop_account_strategy("trade_journal")

    # 2. fill — drop account + strategy_id.
    _drop_account_strategy("fill")

    # 1. open_order — drop account.
    op.execute("ALTER TABLE openorder DROP COLUMN account")
