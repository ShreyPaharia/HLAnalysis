"""Tests for SHR-46: reconcile local-ghost branch must book a recovered lost-ACK
fill into the Position table (apply_position_changes=True) and must NOT book it
when in alert-only mode (apply_position_changes=False, i.e. PM live path).

Key design note: we deliberately pass symbol_to_question={} in the core tests
so the adopt-venue-orphan path at the bottom of Reconciler.run() has no symbol
mapping and cannot book the position. This isolates the local-ghost branch fix
— the position must come from there, not from the orphan-adoption fallback.
"""

from __future__ import annotations

import pytest

from hlanalysis.engine.hl_client import (
    ClearinghouseState, UserFillRow, VenuePosition,
)
from hlanalysis.engine.reconcile import Reconciler
from hlanalysis.engine.state import OpenOrder, StateDAL


@pytest.fixture
def dal(tmp_path):
    d = StateDAL(tmp_path / "state.db")
    d.run_migrations()
    return d


def _pending_order(cloid: str, qidx: int = 42, symbol: str = "@30") -> OpenOrder:
    """A locally-pending (open) buy order — the lost-ACK case."""
    return OpenOrder(
        cloid=cloid, venue_oid=f"v-{cloid}", question_idx=qidx, symbol=symbol,
        side="buy", price=0.60, size=100.0, status="open",
        placed_ts_ns=1, last_update_ts_ns=1, strategy_id="v31",
    )


def _fill(cloid: str, symbol: str = "@30", size: float = 100.0,
          price: float = 0.60) -> UserFillRow:
    return UserFillRow(
        fill_id=f"fill-{cloid}", cloid=cloid, symbol=symbol,
        side="buy", price=price, size=size, fee=0.10, ts_ns=2,
    )


def _venue_state_with_position(symbol: str = "@30", qty: float = 100.0,
                                avg_entry: float = 0.60) -> ClearinghouseState:
    return ClearinghouseState(
        positions=(VenuePosition(symbol=symbol, qty=qty,
                                 avg_entry=avg_entry, unrealized_pnl=0.0),),
        account_value_usd=1000.0,
    )


# ---------------------------------------------------------------------------
# Happy path: apply_position_changes=True (HL live path)
# symbol_to_question={} ensures the Position comes from the local-ghost fix,
# NOT from the fallback adopt-venue-orphan path.
# ---------------------------------------------------------------------------

def test_recovered_fill_books_position_in_apply_mode(dal):
    """Lost-ACK fill discovered by reconcile MUST create a Position row when
    apply_position_changes=True (HL).  Before SHR-46 the position was never
    upserted from the local-ghost branch (the adopt-orphan fallback at the
    bottom of run() only works when symbol_to_question has the mapping; this
    test passes no mapping to isolate the fix)."""
    cloid = "hla-v31-aaaa"
    qidx = 42
    dal.upsert_order(_pending_order(cloid, qidx=qidx, symbol="@30"))

    fills = [_fill(cloid, symbol="@30", size=100.0, price=0.60)]
    venue_state = _venue_state_with_position(symbol="@30", qty=100.0, avg_entry=0.60)

    r = Reconciler(
        dal,
        fills_lookup=lambda c: fills if c == cloid else [],
        symbol_to_question={},   # no mapping → adopt-orphan cannot help
        apply_position_changes=True,
    )
    r.run(venue_open=[], venue_state=venue_state, now_ns=3)

    pos = dal.get_position(qidx)
    assert pos is not None, "Position must be created after a recovered lost-ACK fill"
    assert pos.qty == 100.0
    assert pos.avg_entry == 0.60
    assert pos.symbol == "@30"


def test_recovered_fill_order_still_marked_filled_in_apply_mode(dal):
    """Existing behaviour — order marked filled + fill row replayed — must be
    preserved alongside the new Position upsert."""
    cloid = "hla-v31-bbbb"
    qidx = 43
    dal.upsert_order(_pending_order(cloid, qidx=qidx, symbol="@31"))

    fills = [_fill(cloid, symbol="@31", size=50.0, price=0.55)]
    venue_state = _venue_state_with_position(symbol="@31", qty=50.0, avg_entry=0.55)

    Reconciler(
        dal,
        fills_lookup=lambda c: fills if c == cloid else [],
        symbol_to_question={},   # no mapping → isolate local-ghost path
        apply_position_changes=True,
    ).run(venue_open=[], venue_state=venue_state, now_ns=3)

    assert dal.get_order(cloid).status == "filled"
    assert dal.fills_for_cloid(cloid), "Fill row must be replayed into DB"
    # Position must also exist (the new SHR-46 behavior).
    assert dal.get_position(qidx) is not None
    assert dal.get_position(qidx).qty == 50.0


def test_recovered_fill_no_position_if_venue_has_zero_qty(dal):
    """If the venue reports qty≈0 for the symbol (e.g. immediately settled after
    fill), we must NOT upsert a Position (mirrors the adopt-orphan guard)."""
    cloid = "hla-v31-cccc"
    qidx = 44
    dal.upsert_order(_pending_order(cloid, qidx=qidx, symbol="@32"))

    fills = [_fill(cloid, symbol="@32", size=100.0)]
    # Venue shows qty=0 for that symbol (already settled/closed).
    # symbol_to_question={} ensures the adopt-orphan path also cannot adopt.
    venue_state = ClearinghouseState(
        positions=(VenuePosition(symbol="@32", qty=0.0, avg_entry=0.60,
                                 unrealized_pnl=0.0),),
        account_value_usd=0,
    )

    Reconciler(
        dal,
        fills_lookup=lambda c: fills if c == cloid else [],
        symbol_to_question={},
        apply_position_changes=True,
    ).run(venue_open=[], venue_state=venue_state, now_ns=3)

    # No position row: venue has nothing real to adopt.
    assert dal.get_position(qidx) is None


def test_recovered_fill_no_position_if_symbol_absent_from_venue_state(dal):
    """If the fill is for a symbol the venue_state doesn't mention at all (rare
    race: position already settled before reconcile ran), we skip the upsert."""
    cloid = "hla-v31-dddd"
    qidx = 45
    dal.upsert_order(_pending_order(cloid, qidx=qidx, symbol="@33"))

    fills = [_fill(cloid, symbol="@33", size=100.0)]
    # Venue has no positions at all.
    venue_state = ClearinghouseState(positions=(), account_value_usd=0)

    Reconciler(
        dal,
        fills_lookup=lambda c: fills if c == cloid else [],
        symbol_to_question={},
        apply_position_changes=True,
    ).run(venue_open=[], venue_state=venue_state, now_ns=3)

    assert dal.get_position(qidx) is None


# ---------------------------------------------------------------------------
# Alert-only path: apply_position_changes=False (PM live path — unchanged)
# ---------------------------------------------------------------------------

def test_recovered_fill_does_not_book_position_in_alert_only_mode(dal):
    """PM live (apply_position_changes=False): fill is replayed + order marked
    filled, but the Position table must NOT be touched — PM position truth comes
    from the fill ledger + endDate settlement, not from reconcile injection."""
    cloid = "pm-v31-eeee"
    qidx = 50
    dal.upsert_order(_pending_order(cloid, qidx=qidx, symbol="pm-tok"))

    fills = [_fill(cloid, symbol="pm-tok", size=75.0, price=0.45)]
    venue_state = _venue_state_with_position(symbol="pm-tok", qty=75.0,
                                              avg_entry=0.45)

    Reconciler(
        dal,
        fills_lookup=lambda c: fills if c == cloid else [],
        symbol_to_question={},   # no mapping: also guards adopt-orphan path
        apply_position_changes=False,   # PM live: alert-only
    ).run(venue_open=[], venue_state=venue_state, now_ns=3)

    # Position must NOT be created in alert-only mode.
    assert dal.get_position(qidx) is None, (
        "PM alert-only mode must not create a Position from a recovered fill"
    )

    # But the order + fill bookkeeping still happens.
    assert dal.get_order(cloid).status == "filled"
    assert dal.fills_for_cloid(cloid), "Fill row must still be replayed"
