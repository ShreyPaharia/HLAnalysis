from __future__ import annotations

import pytest

from hlanalysis.engine.hl_client import (
    ClearinghouseState, OpenOrderRow, UserFillRow, VenuePosition,
)
from hlanalysis.engine.reconcile import ReconcileResult, Reconciler
from hlanalysis.engine.state import OpenOrder, Position, StateDAL


@pytest.fixture
def dal(tmp_path):
    d = StateDAL(tmp_path / "state.db")
    d.run_migrations()
    return d


def _row(cloid: str, symbol: str = "@30", price: float = 0.95,
         size: float = 10.0) -> OpenOrderRow:
    return OpenOrderRow(cloid=cloid, venue_oid=f"v-{cloid}", symbol=symbol,
                       side="buy", price=price, size=size, placed_ts_ns=1)


def _db_order(cloid: str, status: str = "open") -> OpenOrder:
    return OpenOrder(
        cloid=cloid, venue_oid=f"v-{cloid}", question_idx=42, symbol="@30",
        side="buy", price=0.95, size=10.0, status=status,
        placed_ts_ns=1, last_update_ts_ns=1, strategy_id="x",
    )


def test_clean_state_no_drift(dal):
    dal.upsert_order(_db_order("hla-1"))
    venue_open = [_row("hla-1")]
    venue_state = ClearinghouseState(positions=(), account_value_usd=0)
    r = Reconciler(dal, fills_lookup=lambda _cloid: [])
    res = r.run(venue_open=venue_open, venue_state=venue_state, now_ns=2)
    assert res.drift_events == []


def test_local_ghost_marks_cancelled_and_emits_drift(dal):
    dal.upsert_order(_db_order("hla-1"))
    r = Reconciler(dal, fills_lookup=lambda _cloid: [])
    res = r.run(venue_open=[], venue_state=ClearinghouseState(positions=(), account_value_usd=0), now_ns=2)
    assert dal.get_order("hla-1").status == "cancelled"
    assert any(d.case == "local_ghost" for d in res.drift_events)


def test_local_ghost_with_known_fill_marks_filled_and_replays(dal):
    dal.upsert_order(_db_order("hla-1"))
    fills = [UserFillRow(
        fill_id="f-1", cloid="hla-1", symbol="@30", side="buy",
        price=0.95, size=10.0, fee=0.05, ts_ns=1,
    )]
    r = Reconciler(dal, fills_lookup=lambda cloid: fills if cloid == "hla-1" else [])
    res = r.run(venue_open=[], venue_state=ClearinghouseState(positions=(), account_value_usd=0), now_ns=2)
    assert dal.get_order("hla-1").status == "filled"
    assert dal.fills_for_cloid("hla-1")  # replayed
    assert any(d.case == "state_mismatch" for d in res.drift_events)


def test_venue_orphan_emits_drift_and_caller_cancels(dal):
    venue_open = [_row("hla-orphan")]
    r = Reconciler(dal, fills_lookup=lambda _cloid: [])
    res = r.run(venue_open=venue_open, venue_state=ClearinghouseState(positions=(), account_value_usd=0), now_ns=2)
    assert any(d.case == "venue_orphan" and d.cloid == "hla-orphan" for d in res.drift_events)
    assert ("hla-orphan", "@30") in res.orphans_to_cancel


def test_state_mismatch_hl_wins(dal):
    dal.upsert_order(_db_order("hla-1"))
    # Same cloid, different price on venue
    res = Reconciler(dal, fills_lookup=lambda _: []).run(
        venue_open=[_row("hla-1", price=0.96)],
        venue_state=ClearinghouseState(positions=(), account_value_usd=0),
        now_ns=2,
    )
    assert dal.get_order("hla-1").price == 0.96
    assert any(d.case == "state_mismatch" for d in res.drift_events)


def test_position_mismatch_hl_wins(dal):
    dal.upsert_position(Position(
        question_idx=42, symbol="@30", qty=10.0, avg_entry=0.95,
        realized_pnl=0.0, last_update_ts_ns=1, stop_loss_price=0.855,
    ))
    venue_state = ClearinghouseState(
        positions=(VenuePosition(symbol="@30", qty=8.0, avg_entry=0.95, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(dal, fills_lookup=lambda _: [], symbol_to_question={"@30": 42}).run(
        venue_open=[], venue_state=venue_state, now_ns=2,
    )
    assert dal.get_position(42).qty == 8.0
    assert any(d.case == "position_mismatch" for d in res.drift_events)


def test_position_disappearance_drops_local_position(dal):
    dal.upsert_position(Position(
        question_idx=42, symbol="@30", qty=10.0, avg_entry=0.95,
        realized_pnl=0.0, last_update_ts_ns=1, stop_loss_price=0.855,
    ))
    res = Reconciler(dal, fills_lookup=lambda _: [], symbol_to_question={"@30": 42}).run(
        venue_open=[], venue_state=ClearinghouseState(positions=(), account_value_usd=0), now_ns=2,
    )
    assert dal.get_position(42) is None
    assert any(d.case == "position_mismatch" for d in res.drift_events)


def test_venue_orphan_position_is_adopted_into_local_db(dal):
    # HL has #551 with 500 shares but the local DB is empty (e.g. fills booked
    # before a restart that wiped the DB, or filled while the engine was
    # blind to HIP-4 spot balances). The reconciler must adopt the venue
    # position so the gate's caps and strategy's have_position branch see it,
    # otherwise the strategy re-fires entries the venue then rejects.
    venue_state = ClearinghouseState(
        positions=(VenuePosition(symbol="#551", qty=500.0, avg_entry=0.988,
                                  unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(
        dal, fills_lookup=lambda _: [], symbol_to_question={"#551": 10},
    ).run(venue_open=[], venue_state=venue_state, now_ns=2)

    booked = dal.get_position(10)
    assert booked is not None, "venue-orphan position was not adopted"
    assert booked.symbol == "#551"
    assert booked.qty == 500.0
    assert booked.avg_entry == 0.988
    assert any(
        d.case == "position_mismatch"
        and (d.detail or {}).get("resolution") == "adopted_venue_orphan"
        for d in res.drift_events
    )


def test_reconciler_matches_cloid_by_hex_tail_not_prefix(dal):
    # HL normalizes our `hla-v1-<hex>` cloid to `0x<hex>` on the venue side.
    # Without the hex-tail normalization in the reconciler, the join misses
    # every venue order: local rows look like `hla-v1-2626…` while venue rows
    # come back as `0x2626…`. The two-cloid views below describe the same
    # order; reconciler must see them as a match (no drift). The shared
    # venue_oid models the post-place-ack state — both sides agree on HL's oid.
    HEX = "2626f31a8d1348b5828826a8baef96c8"
    SHARED_OID = "venue-oid-7"
    dal.upsert_order(OpenOrder(
        cloid=f"hla-v1-{HEX}", venue_oid=SHARED_OID, question_idx=42, symbol="@30",
        side="buy", price=0.95, size=10.0, status="open",
        placed_ts_ns=1, last_update_ts_ns=1, strategy_id="x",
    ))
    venue_open = [OpenOrderRow(cloid=f"0x{HEX}", venue_oid=SHARED_OID, symbol="@30",
                                side="buy", price=0.95, size=10.0, placed_ts_ns=1)]
    res = Reconciler(dal, fills_lookup=lambda _: [], cloid_prefix="hla-v1-").run(
        venue_open=venue_open,
        venue_state=ClearinghouseState(positions=(), account_value_usd=0),
        now_ns=2,
    )
    # Same order, just different cloid forms — no drift.
    assert res.drift_events == []
    assert res.orphans_to_cancel == []


def test_reconciler_state_mismatch_preserves_local_cloid_after_hex_match(dal):
    # A field-level drift on a cross-form cloid pair must update the DB row
    # using the LOCAL cloid (DB primary key), not the venue's `0x<hex>` form,
    # otherwise upsert_order would insert a duplicate row.
    HEX = "deadbeefdeadbeefdeadbeefdeadbeef"
    SHARED_OID = "venue-oid-8"
    dal.upsert_order(OpenOrder(
        cloid=f"hla-v1-{HEX}", venue_oid=SHARED_OID, question_idx=42, symbol="@30",
        side="buy", price=0.95, size=10.0, status="open",
        placed_ts_ns=1, last_update_ts_ns=1, strategy_id="x",
    ))
    venue_open = [OpenOrderRow(cloid=f"0x{HEX}", venue_oid=SHARED_OID, symbol="@30",
                                side="buy", price=0.97, size=10.0, placed_ts_ns=1)]
    Reconciler(dal, fills_lookup=lambda _: [], cloid_prefix="hla-v1-").run(
        venue_open=venue_open,
        venue_state=ClearinghouseState(positions=(), account_value_usd=0),
        now_ns=2,
    )
    # The local row was updated in-place — no duplicate row under the 0x form.
    assert dal.get_order(f"hla-v1-{HEX}").price == 0.97
    assert dal.get_order(f"0x{HEX}") is None


def test_venue_orphan_position_without_mapping_does_not_corrupt_db(dal):
    # If we can't map symbol→question_idx (e.g. the question meta hasn't been
    # ingested yet) we must NOT insert a Position row with a guessed/sentinel
    # question_idx — that would corrupt the table. Just emit a drift event.
    venue_state = ClearinghouseState(
        positions=(VenuePosition(symbol="#999", qty=10.0, avg_entry=0.5,
                                  unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(
        dal, fills_lookup=lambda _: [], symbol_to_question={},
    ).run(venue_open=[], venue_state=venue_state, now_ns=2)

    # No new position rows created
    assert dal.all_positions() == []
    # But the drift event must surface so we can act on it
    assert any(
        d.case == "position_mismatch"
        and (d.detail or {}).get("symbol") == "#999"
        for d in res.drift_events
    )
