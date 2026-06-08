"""SHR-74: HL local realized ledger mirrors venue user_fills.

The local Fill table holds two provenances (Fill.source): 'router' rows from
Router._book_fill (cloid-keyed, fee=0, locally-computed closed_pnl) and 'venue'
rows mirrored from HL user_fills (tid-keyed, venue closedPnl/fee, including the
settlement-dir payouts that never reach _book_fill). realized_pnl_since must
prefer 'venue' rows when present so the HL figure equals venue truth, while a PM
slot (no 'venue' rows) keeps summing its authoritative 'router' ledger.
"""
from __future__ import annotations

from hlanalysis.engine.exec_types import UserFillRow
from hlanalysis.engine.state import (
    FILL_SOURCE_ROUTER, FILL_SOURCE_VENUE, Fill, StateDAL,
)


def _dal(tmp_path) -> StateDAL:
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    return dal


def _venue_fill(fill_id, symbol, closed_pnl=0.0, fee=0.0, ts_ns=1, size=10.0):
    return UserFillRow(
        fill_id=fill_id, cloid="", symbol=symbol, side="sell",
        price=0.9, size=size, fee=fee, ts_ns=ts_ns, closed_pnl=closed_pnl,
    )


def test_pm_slot_sums_router_rows_when_no_venue_rows(tmp_path):
    """A PM slot never gets mirrored venue rows, so realized_pnl_since sums its
    authoritative router-booked ledger (unchanged behaviour)."""
    dal = _dal(tmp_path)
    dal.append_fill(Fill(fill_id="hla-1", cloid="hla-1", question_idx=1,
                         symbol="0xtok", side="sell", price=0.9, size=10.0,
                         fee=0.1, ts_ns=5, closed_pnl=4.0))
    assert dal.realized_pnl_since(0) == 4.0 - 0.1


def test_venue_rows_preferred_over_router_rows(tmp_path):
    """Once any source='venue' row exists (HL post-mirror), the router rows for
    the same economic fills are excluded — no double-count, venue is truth."""
    dal = _dal(tmp_path)
    # The router booked a sell with a locally-computed -$5 and fee=0 ...
    dal.append_fill(Fill(fill_id="hla-9-100", cloid="hla-9", question_idx=1,
                         symbol="#10", side="sell", price=0.9, size=10.0,
                         fee=0.0, ts_ns=10, closed_pnl=-5.0,
                         source=FILL_SOURCE_ROUTER))
    # ... and the venue mirror booked the same trade fill (real fee) PLUS the
    # settlement-dir winning payout that _book_fill never saw.
    dal.mirror_venue_fills([
        _venue_fill("tid-trade", "#10", closed_pnl=-5.5, fee=0.2, ts_ns=10),
        _venue_fill("tid-settle", "#11", closed_pnl=47.7, fee=0.0, ts_ns=20),
    ])
    # Sum of venue rows only: (-5.5 - 0.2) + (47.7 - 0.0) = 42.0. The router
    # row's -5.0 is excluded.
    assert dal.realized_pnl_since(0) == (-5.5 - 0.2) + 47.7


def test_mirror_is_idempotent_and_outcome_only(tmp_path):
    """Re-mirroring books only newly-seen tids; non-'#' (perp/spot) fills are
    skipped so the outcome-only realized figure stays clean."""
    dal = _dal(tmp_path)
    fills = [
        _venue_fill("tid-1", "#10", closed_pnl=3.0, ts_ns=10),
        _venue_fill("tid-perp", "BTC", closed_pnl=999.0, ts_ns=10),  # skipped
    ]
    assert dal.mirror_venue_fills(fills) == 1          # only #10 booked
    assert dal.mirror_venue_fills(fills) == 0          # idempotent
    assert dal.realized_pnl_since(0) == 3.0            # perp excluded


def test_window_filter_applies_to_venue_rows(tmp_path):
    """The since_ts_ns window applies to the selected (venue) rows."""
    dal = _dal(tmp_path)
    dal.mirror_venue_fills([
        _venue_fill("tid-old", "#10", closed_pnl=2.0, ts_ns=100),
        _venue_fill("tid-new", "#10", closed_pnl=5.0, ts_ns=300),
    ])
    assert dal.realized_pnl_since(0) == 7.0
    assert dal.realized_pnl_since(200) == 5.0
