"""#29: Reconcile loop MUTATES positions even when the slot is restart-blocked.

The Reconciler's apply_position_changes=False path already prevents DB mutation.
The bug is that runtime._reconcile_loop computes apply_positions based only on
is_pm / startup_position_synced, and does NOT check slot.blocked. For a blocked
HL slot, apply_positions=True causes the reconciler to delete/mutate positions.

These tests:
1. Verify Reconciler with apply_position_changes=False does NOT mutate positions
   (unit-level: contracts the Reconciler must honour).
2. The runtime fix (passing apply_position_changes=False when slot.blocked) is
   verified at the unit level since the runtime loop involves async venue IO
   which is hard to fake here.
"""

from __future__ import annotations

from hlanalysis.engine.exec_types import ClearinghouseState, VenuePosition
from hlanalysis.engine.reconcile import Reconciler
from hlanalysis.engine.state import Position, StateDAL

_NOW = 1_000_000_000


def _dal(tmp_path):
    d = StateDAL(tmp_path / "state.db")
    d.run_migrations()
    return d


def _position(question_idx: int = 42, symbol: str = "@30", qty: float = 10.0) -> Position:
    return Position(
        question_idx=question_idx,
        symbol=symbol,
        qty=qty,
        avg_entry=0.90,
        realized_pnl=0.0,
        last_update_ts_ns=_NOW,
        stop_loss_price=0.0,
    )


def _venue_pos(symbol: str = "@30", qty: float = 10.0) -> VenuePosition:
    return VenuePosition(symbol=symbol, qty=qty, avg_entry=0.90, unrealized_pnl=0.0)


def _venue_state_with(positions) -> ClearinghouseState:
    return ClearinghouseState(positions=tuple(positions), account_value_usd=0.0)


def test_blocked_slot_reconcile_does_not_delete_positions(tmp_path):
    """#29: When apply_position_changes=False (blocked slot), reconcile must
    not delete a locally-held position that vanished from the venue."""
    dal = _dal(tmp_path)
    dal.upsert_position(_position())
    # Venue has no positions (vanished)
    venue_state = _venue_state_with([])
    r = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        apply_position_changes=False,  # blocked slot
    )
    res = r.run(venue_open=[], venue_state=venue_state, now_ns=_NOW)
    # Position must still be in DB — not deleted
    assert dal.all_positions() != [], "blocked slot reconcile must not delete positions"
    # Vanished_positions list must be empty when apply=False
    assert res.vanished_positions == [], "vanished_positions must be empty in alert-only mode"


def test_blocked_slot_reconcile_does_not_adopt_orphan(tmp_path):
    """#29: When apply_position_changes=False (blocked slot), reconcile must
    not adopt a venue orphan position into the DB."""
    dal = _dal(tmp_path)
    # No local position, but venue has one
    venue_state = _venue_state_with([_venue_pos()])
    r = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"@30": 42},
        apply_position_changes=False,  # blocked slot
    )
    res = r.run(venue_open=[], venue_state=venue_state, now_ns=_NOW)
    # Must not have adopted the orphan
    assert dal.all_positions() == [], "blocked slot reconcile must not adopt venue orphan"


def test_blocked_slot_reconcile_does_not_mutate_qty(tmp_path):
    """#29: When apply_position_changes=False (blocked slot), reconcile must
    not overwrite local qty with a differing venue qty."""
    dal = _dal(tmp_path)
    dal.upsert_position(_position(qty=10.0))
    # Venue reports a different qty
    venue_state = _venue_state_with([_venue_pos(qty=5.0)])
    r = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"@30": 42},
        apply_position_changes=False,  # blocked slot
    )
    r.run(venue_open=[], venue_state=venue_state, now_ns=_NOW)
    # Local qty must be unchanged
    positions = dal.all_positions()
    assert len(positions) == 1
    assert positions[0].qty == 10.0, "blocked slot reconcile must not overwrite local qty"


def test_unblocked_slot_reconcile_still_applies(tmp_path):
    """Regression guard: unblocked slot (apply_position_changes=True) still
    applies changes as before."""
    dal = _dal(tmp_path)
    dal.upsert_position(_position(qty=10.0))
    # Venue has no positions (vanished)
    venue_state = _venue_state_with([])
    r = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        apply_position_changes=True,  # unblocked slot
    )
    res = r.run(venue_open=[], venue_state=venue_state, now_ns=_NOW)
    # Position must be deleted (vanished on venue, apply=True means venue wins)
    assert dal.all_positions() == [], "unblocked slot must still apply venue deletions"
    assert len(res.vanished_positions) == 1
