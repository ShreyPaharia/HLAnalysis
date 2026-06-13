"""#26: Material position-qty drift ALERTS but never HALTS.

Tests that when qty drift exceeds the material threshold, the ReconcileResult
signals it via material_qty_drift=True, so the runtime can escalate to halt.
Small drift emits alert-only (drift event), no halt flag.
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


def test_small_qty_drift_is_alert_only(tmp_path):
    """#26: Drift smaller than the material threshold must set
    material_qty_drift=False (alert-only, no halt)."""
    dal = _dal(tmp_path)
    dal.upsert_position(_position(qty=10.0))
    # Venue differs by only 0.005 — below any sane material threshold
    venue_state = _venue_state_with([_venue_pos(qty=10.005)])
    r = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"@30": 42},
        apply_position_changes=True,
        material_drift_qty=1.0,  # 1-share material threshold
    )
    res = r.run(venue_open=[], venue_state=venue_state, now_ns=_NOW)
    assert res.material_qty_drift is False, "tiny drift must not set material_qty_drift"


def test_large_qty_drift_sets_material_flag(tmp_path):
    """#26: Drift larger than the material threshold must set
    material_qty_drift=True so the runtime can escalate to halt."""
    dal = _dal(tmp_path)
    dal.upsert_position(_position(qty=10.0))
    # Venue reports 15 shares — 5 shares off, well above a 1-share threshold
    venue_state = _venue_state_with([_venue_pos(qty=15.0)])
    r = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"@30": 42},
        apply_position_changes=True,
        material_drift_qty=1.0,  # 1-share material threshold
    )
    res = r.run(venue_open=[], venue_state=venue_state, now_ns=_NOW)
    assert res.material_qty_drift is True, "large drift must set material_qty_drift"


def test_material_drift_default_threshold(tmp_path):
    """#26: Default threshold is conservative so a 2-share discrepancy triggers halt."""
    dal = _dal(tmp_path)
    dal.upsert_position(_position(qty=10.0))
    # Venue is missing 2 shares
    venue_state = _venue_state_with([_venue_pos(qty=8.0)])
    r = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"@30": 42},
        apply_position_changes=True,
        # No material_drift_qty specified — uses default
    )
    res = r.run(venue_open=[], venue_state=venue_state, now_ns=_NOW)
    # With a 2-share diff and a sensible default threshold the flag must fire
    assert res.material_qty_drift is True


def test_no_drift_no_material_flag(tmp_path):
    """#26: When local and venue agree, material_qty_drift must be False."""
    dal = _dal(tmp_path)
    dal.upsert_position(_position(qty=10.0))
    venue_state = _venue_state_with([_venue_pos(qty=10.0)])
    r = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"@30": 42},
        apply_position_changes=True,
        material_drift_qty=1.0,
    )
    res = r.run(venue_open=[], venue_state=venue_state, now_ns=_NOW)
    assert res.material_qty_drift is False
