"""Tests for tools/backfill_hl_fill_ledger.py — core backfill_dal function.

The pure core (backfill_dal) takes a StateDAL + a list of UserFillRow objects so
it can be exercised without real config or venue connectivity.

Test plan
---------
1. Normal path: stale router Fill rows + a bogus Settlement row are wiped;
   venue '#'-symbol fills are mirrored; realized_pnl_since(0) == venue_realized.
2. --dry-run: nothing is written to the DB; before_realized unchanged; counts
   report what WOULD be changed.
3. Non-'#' venue fills are not mirrored (outcome-only).
4. Already-empty DB (no prior rows): backfill works cleanly (0-wipe, n-mirror).
5. Venue fills with no '#' symbols at all → after_realized == 0.0.
6. Multiple '#' fills: all are mirrored; realized_pnl_since matches sum.
"""
from __future__ import annotations

import pytest

from hlanalysis.engine.exec_types import UserFillRow
from hlanalysis.engine.state import (
    FILL_SOURCE_ROUTER,
    Fill,
    Settlement,
    StateDAL,
)
from tools.backfill_hl_fill_ledger import backfill_dal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dal(tmp_path, suffix="state.db") -> StateDAL:
    dal = StateDAL(tmp_path / suffix)
    dal.run_migrations()
    return dal


def _router_fill(fill_id: str, symbol: str = "#10", closed_pnl: float = 0.0) -> Fill:
    """A stale cloid-keyed 'router' Fill row (the old shape we need to wipe)."""
    return Fill(
        fill_id=fill_id,
        cloid=fill_id,
        question_idx=1,
        symbol=symbol,
        side="sell",
        price=0.9,
        size=10.0,
        fee=0.0,
        ts_ns=100,
        closed_pnl=closed_pnl,
        source=FILL_SOURCE_ROUTER,
    )


def _venue_row(
    fill_id: str,
    symbol: str = "#10",
    closed_pnl: float = 0.0,
    fee: float = 0.0,
    ts_ns: int = 200,
) -> UserFillRow:
    """A UserFillRow as returned by exec_client.user_fills()."""
    return UserFillRow(
        fill_id=fill_id,
        cloid="",
        symbol=symbol,
        side="sell",
        price=0.9,
        size=10.0,
        fee=fee,
        ts_ns=ts_ns,
        closed_pnl=closed_pnl,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_wipes_router_fills_and_settlement_then_mirrors(tmp_path):
    """Normal path: stale router rows + bogus Settlement row are removed; venue
    '#' fills are mirrored; realized_pnl_since == venue truth."""
    dal = _dal(tmp_path)

    # Seed stale router rows.
    dal.append_fill(_router_fill("router-1", "#10", closed_pnl=-5.0))
    dal.append_fill(_router_fill("router-2", "#11", closed_pnl=3.0))

    # Seed a bogus Settlement row (old shape).
    dal.record_settlement(question_idx=1, symbol="#10", realized_pnl=-99.0, ts_ns=50)

    before_pnl = dal.realized_pnl_since(0)
    # The router rows give (-5.0 - 0) + (3.0 - 0) = -2.0; Settlement adds -99.
    # (The exact value is less important than confirming it changes after backfill.)

    venue_fills = [
        _venue_row("tid-trade", "#10", closed_pnl=-5.5, fee=0.2, ts_ns=200),
        _venue_row("tid-settle", "#11", closed_pnl=47.7, fee=0.0, ts_ns=300),
    ]
    expected_venue_realized = (-5.5 - 0.2) + (47.7 - 0.0)

    result = backfill_dal(dal, venue_fills)

    # Summary fields
    assert result["before_realized"] == pytest.approx(before_pnl)
    assert result["venue_realized"] == pytest.approx(expected_venue_realized)
    assert result["after_realized"] == pytest.approx(expected_venue_realized)
    assert result["rows_mirrored"] == 2
    assert result["fills_wiped"] == 2
    assert result["settlements_wiped"] == 1

    # DB state: settlement table empty, fill table == venue mirror.
    from sqlmodel import Session, select
    with Session(dal._engine) as s:
        assert s.exec(select(Settlement)).all() == []
        fills = s.exec(select(Fill)).all()
    assert len(fills) == 2
    fill_ids = {f.fill_id for f in fills}
    assert fill_ids == {"tid-trade", "tid-settle"}
    assert all(f.source == "venue" for f in fills)

    # realized_pnl_since matches venue truth.
    assert dal.realized_pnl_since(0) == pytest.approx(expected_venue_realized)


def test_dry_run_writes_nothing(tmp_path):
    """--dry-run computes the summary but leaves the DB untouched."""
    dal = _dal(tmp_path)

    dal.append_fill(_router_fill("router-1", "#10", closed_pnl=7.0))
    dal.record_settlement(question_idx=1, symbol="#10", realized_pnl=50.0, ts_ns=50)

    before_pnl = dal.realized_pnl_since(0)

    venue_fills = [
        _venue_row("tid-1", "#10", closed_pnl=6.0, fee=0.1),
    ]
    result = backfill_dal(dal, venue_fills, dry_run=True)

    # Dry-run summary
    assert result["dry_run"] if "dry_run" in result else True  # key optional
    assert result["rows_mirrored"] == 0
    assert result["fills_wiped"] == 1        # what WOULD be wiped
    assert result["settlements_wiped"] == 1  # what WOULD be wiped
    assert result["after_realized"] == pytest.approx(before_pnl)  # unchanged

    # DB must be identical to before the call.
    from sqlmodel import Session, select
    with Session(dal._engine) as s:
        fills = s.exec(select(Fill)).all()
        settlements = s.exec(select(Settlement)).all()
    assert len(fills) == 1
    assert fills[0].fill_id == "router-1"
    assert len(settlements) == 1
    assert settlements[0].realized_pnl == pytest.approx(50.0)


def test_non_outcome_fills_not_mirrored(tmp_path):
    """Venue fills whose symbol does NOT start with '#' (perp, spot) are skipped."""
    dal = _dal(tmp_path)

    venue_fills = [
        _venue_row("tid-outcome", "#10", closed_pnl=20.0),
        _venue_row("tid-perp", "BTCUSDT", closed_pnl=999.0),
        _venue_row("tid-spot", "BTC", closed_pnl=111.0),
    ]
    result = backfill_dal(dal, venue_fills)

    assert result["rows_mirrored"] == 1
    assert result["venue_realized"] == pytest.approx(20.0)
    assert dal.realized_pnl_since(0) == pytest.approx(20.0)


def test_empty_db_backfill(tmp_path):
    """Backfill on a fresh DB with no prior rows works cleanly."""
    dal = _dal(tmp_path)

    venue_fills = [
        _venue_row("tid-a", "#10", closed_pnl=10.0, fee=0.5),
        _venue_row("tid-b", "#20", closed_pnl=5.0, fee=0.2),
    ]
    expected = (10.0 - 0.5) + (5.0 - 0.2)
    result = backfill_dal(dal, venue_fills)

    assert result["fills_wiped"] == 0
    assert result["settlements_wiped"] == 0
    assert result["rows_mirrored"] == 2
    assert result["after_realized"] == pytest.approx(expected)
    assert dal.realized_pnl_since(0) == pytest.approx(expected)


def test_no_outcome_venue_fills(tmp_path):
    """When ALL venue fills are non-'#', after_realized == 0.0."""
    dal = _dal(tmp_path)
    # Seed a stale router row that should be wiped.
    dal.append_fill(_router_fill("router-1", "#10", closed_pnl=100.0))

    venue_fills = [
        _venue_row("tid-perp", "BTCUSDT", closed_pnl=500.0),
    ]
    result = backfill_dal(dal, venue_fills)

    assert result["rows_mirrored"] == 0
    assert result["venue_realized"] == pytest.approx(0.0)
    assert result["after_realized"] == pytest.approx(0.0)
    assert result["fills_wiped"] == 1

    assert dal.realized_pnl_since(0) == pytest.approx(0.0)


def test_multiple_fills_realized_matches(tmp_path):
    """Σ (closed_pnl - fee) across multiple '#' fills matches realized_pnl_since."""
    dal = _dal(tmp_path)

    fills = [
        _venue_row("t1", "#10", closed_pnl=12.0, fee=0.3),
        _venue_row("t2", "#11", closed_pnl=-3.5, fee=0.1),
        _venue_row("t3", "#12", closed_pnl=50.0, fee=0.5),
        _venue_row("t4", "#13", closed_pnl=0.0, fee=0.05),   # open/neutral
    ]
    expected = (12.0 - 0.3) + (-3.5 - 0.1) + (50.0 - 0.5) + (0.0 - 0.05)
    result = backfill_dal(dal, fills)

    assert result["rows_mirrored"] == 4
    assert result["venue_realized"] == pytest.approx(expected)
    assert result["after_realized"] == pytest.approx(expected)
    assert dal.realized_pnl_since(0) == pytest.approx(expected)


def test_symbol_to_question_forwarded(tmp_path):
    """symbol_to_question mapping is forwarded to mirror_venue_fills so
    question_idx is set correctly on the inserted rows."""
    dal = _dal(tmp_path)

    venue_fills = [_venue_row("t1", "#10", closed_pnl=5.0)]
    sym_map = {"#10": 42}
    backfill_dal(dal, venue_fills, symbol_to_question=sym_map)

    from sqlmodel import Session, select
    with Session(dal._engine) as s:
        row = s.exec(select(Fill)).one()
    assert row.question_idx == 42


def test_idempotent_after_wipe_and_remirror(tmp_path):
    """Running backfill_dal twice on the same DB yields the same realized figure
    (the second run wipes the first run's venue rows and re-inserts them)."""
    dal = _dal(tmp_path)
    fills = [_venue_row("t1", "#10", closed_pnl=7.0, fee=0.1)]
    expected = 7.0 - 0.1

    result1 = backfill_dal(dal, fills)
    result2 = backfill_dal(dal, fills)

    assert result1["after_realized"] == pytest.approx(expected)
    assert result2["after_realized"] == pytest.approx(expected)
    # Second run wipes the one row inserted by the first run.
    assert result2["fills_wiped"] == 1
    assert result2["rows_mirrored"] == 1
