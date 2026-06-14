"""Tests for multi-strategy account attribution (Task 5).

These tests verify:
1. attribute_venue_positions partitions by owner, with an unattributed bucket.
2. Reconciler.sibling_symbols suppresses orphan alerts for a sibling strategy's
   venue positions and orders on a shared wallet.
3. Legacy (sibling_symbols=None / empty) behaviour is unchanged — the existing
   reconcile tests cover this; the tests here cross-check the boundary case.
"""

from __future__ import annotations

import pytest

from hlanalysis.engine.hl_client import (
    ClearinghouseState,
    OpenOrderRow,
    VenuePosition,
)
from hlanalysis.engine.reconcile import Reconciler, attribute_venue_positions
from hlanalysis.engine.state import StateDAL

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def dal(tmp_path):
    d = StateDAL(tmp_path / "state.db")
    d.run_migrations()
    return d


def _venue_pos(symbol: str, qty: float = 10.0) -> VenuePosition:
    return VenuePosition(symbol=symbol, qty=qty, avg_entry=0.90, unrealized_pnl=0.0)


def _open_order_row(cloid: str, symbol: str) -> OpenOrderRow:
    return OpenOrderRow(
        cloid=cloid,
        venue_oid=f"v-{cloid}",
        symbol=symbol,
        side="buy",
        price=0.90,
        size=10.0,
        placed_ts_ns=1,
    )


def _cs(positions=()) -> ClearinghouseState:
    return ClearinghouseState(positions=positions, account_value_usd=0)


# ---------------------------------------------------------------------------
# attribute_venue_positions helper
# ---------------------------------------------------------------------------


def test_attribute_venue_positions_partitions_by_owner():
    """Symbols claimed by distinct strategies land in their own bucket; an
    unclaimed symbol goes to the 'unattributed' bucket."""
    pos_a = _venue_pos("SYM_A")
    pos_b = _venue_pos("SYM_B")
    pos_unknown = _venue_pos("SYM_X")

    owners = {
        "strat_a": {"SYM_A"},
        "strat_b": {"SYM_B"},
    }
    result = attribute_venue_positions([pos_a, pos_b, pos_unknown], owners)

    assert result["strat_a"] == [pos_a]
    assert result["strat_b"] == [pos_b]
    assert result["unattributed"] == [pos_unknown]


def test_attribute_venue_positions_empty_wallet():
    """No venue positions → every bucket is empty."""
    owners = {"strat_a": {"SYM_A"}, "strat_b": set()}
    result = attribute_venue_positions([], owners)
    assert result["strat_a"] == []
    assert result["strat_b"] == []
    assert result["unattributed"] == []


def test_attribute_venue_positions_all_unattributed():
    """No strategy claims any symbol → everything is unattributed."""
    pos_x = _venue_pos("SYM_X")
    pos_y = _venue_pos("SYM_Y")
    result = attribute_venue_positions([pos_x, pos_y], {"strat_a": set()})
    assert result["strat_a"] == []
    assert result["unattributed"] == [pos_x, pos_y]


# ---------------------------------------------------------------------------
# Reconciler.sibling_symbols — position-level
# ---------------------------------------------------------------------------


def test_sibling_position_symbol_not_orphaned(dal):
    """Strategy A's Reconciler with sibling_symbols={'SYM_B'}: a venue position
    in SYM_B is owned by strategy B (same wallet) and must NOT generate an
    orphan alert or adoption for strategy A."""
    # Strategy A has no local positions.
    venue_state = _cs(positions=(_venue_pos("SYM_B"),))
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"SYM_B": 99},
        sibling_symbols={"SYM_B"},
    ).run(venue_open=[], venue_state=venue_state, now_ns=2)

    # No drift events at all for SYM_B.
    assert not any((d.detail or {}).get("symbol") == "SYM_B" for d in res.drift_events), (
        f"unexpected drift for sibling symbol: {res.drift_events}"
    )
    # Not adopted into strategy A's DB.
    assert dal.get_position(99) is None


def test_sibling_position_no_sibling_symbols_is_treated_as_orphan(dal):
    """Legacy path (sibling_symbols=None): the same venue position IS treated as
    an orphan (venue_orphan_alert_only / adopted_venue_orphan). This confirms
    the 1:1 behaviour is unchanged when sibling_symbols is not provided."""
    venue_state = _cs(positions=(_venue_pos("SYM_B"),))
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"SYM_B": 99},
        # No sibling_symbols — legacy default
    ).run(venue_open=[], venue_state=venue_state, now_ns=2)

    # In apply mode (default) the position is adopted + a drift event fires.
    assert dal.get_position(99) is not None
    assert any((d.detail or {}).get("resolution") == "adopted_venue_orphan" for d in res.drift_events)


def test_sibling_position_alert_only_mode_not_orphaned(dal):
    """sibling_symbols also suppresses the alert-only venue_orphan path (PM live
    mode, apply_position_changes=False)."""
    venue_state = _cs(positions=(_venue_pos("SYM_B"),))
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"SYM_B": 99},
        apply_position_changes=False,
        sibling_symbols={"SYM_B"},
    ).run(venue_open=[], venue_state=venue_state, now_ns=2)

    assert not any((d.detail or {}).get("symbol") == "SYM_B" for d in res.drift_events)
    assert dal.get_position(99) is None


def test_sibling_position_unrelated_symbol_still_alerts(dal):
    """Guard against over-suppression: a symbol NOT in sibling_symbols still
    triggers the normal orphan path."""
    venue_state = _cs(
        positions=(
            _venue_pos("SYM_B"),  # sibling → skipped
            _venue_pos("SYM_C"),  # not a sibling → orphan
        )
    )
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"SYM_B": 99, "SYM_C": 100},
        sibling_symbols={"SYM_B"},
    ).run(venue_open=[], venue_state=venue_state, now_ns=2)

    # SYM_B: no drift
    assert not any((d.detail or {}).get("symbol") == "SYM_B" for d in res.drift_events)
    # SYM_C: adopted orphan (apply mode default)
    assert dal.get_position(100) is not None


# ---------------------------------------------------------------------------
# Reconciler.sibling_symbols — order-level
# ---------------------------------------------------------------------------


def test_sibling_order_symbol_not_orphaned(dal):
    """Strategy A's Reconciler: a venue *order* in a sibling-owned symbol must
    not appear in orphans_to_cancel or generate a venue_orphan drift."""
    venue_order = _open_order_row("0xaaaa", "SYM_B")
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        sibling_symbols={"SYM_B"},
    ).run(
        venue_open=[venue_order],
        venue_state=_cs(),
        now_ns=2,
    )
    assert res.orphans_to_cancel == []
    assert not any(d.case == "venue_orphan" for d in res.drift_events)


def test_sibling_order_no_sibling_symbols_is_orphan(dal):
    """Legacy path: without sibling_symbols the same venue order IS an orphan."""
    venue_order = _open_order_row("0xaaaa", "SYM_B")
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        # no sibling_symbols
    ).run(
        venue_open=[venue_order],
        venue_state=_cs(),
        now_ns=2,
    )
    assert ("0xaaaa", "SYM_B") in res.orphans_to_cancel
    assert any(d.case == "venue_orphan" for d in res.drift_events)
