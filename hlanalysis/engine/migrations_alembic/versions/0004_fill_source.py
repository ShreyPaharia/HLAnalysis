"""fill.source — provenance of a Fill row (router-booked vs venue-mirrored)

Revision ID: 0004_fill_source
Revises: 0003_coin_klass
Create Date: 2026-06-08

Adds ``fill.source`` (TEXT NOT NULL DEFAULT 'router') so the local ledger can
distinguish two provenances of the same economic fill:

* ``'router'`` — written by ``Router._book_fill`` at ACK time with a
  cloid-derived ``fill_id``, ``fee=0`` (HL never surfaces fee on the ACK), and a
  *locally computed* ``closed_pnl``. Authoritative for PM (its local ledger is
  the only realized-PnL source) and a diagnostic trade-history for HL.
* ``'venue'`` — mirrored from HL ``user_fills`` by the reconcile loop, keyed by
  the venue ``tid`` with the venue's own ``closedPnl``/``fee``. For HL this is
  the authoritative realized ledger because HIP-4/bucket settlement payouts
  arrive ONLY as venue fills (``dir="Settlement"``) and never reach
  ``_book_fill`` — the gap that made the local ledger diverge from venue
  (SHR-74).

``realized_pnl_since`` prefers ``'venue'`` rows when any exist (HL, post-mirror),
otherwise sums all rows (PM, or a not-yet-mirrored HL slot). This makes the HL
local realized == venue by construction without disturbing the PM ledger or the
``_book_fill`` hot path (its rows are still written, just stamped 'router').

Existing rows backfill to 'router' (the default), which is correct: every Fill
row written before this migration came from ``_book_fill``. The one-time HL
ledger backfill (tools/backfill_hl_fill_ledger.py) wipes each HL slot's fill rows
and re-mirrors them as 'venue'.
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0004_fill_source"
down_revision = "0003_coin_klass"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE fill ADD COLUMN source TEXT NOT NULL DEFAULT 'router'")


def downgrade() -> None:
    op.execute("ALTER TABLE fill DROP COLUMN source")
