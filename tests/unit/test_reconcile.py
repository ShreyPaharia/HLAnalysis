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


def test_position_qty_mismatch_emits_drift_and_adopts_hl(dal):
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


def test_position_avg_entry_only_diff_silently_adopts_no_drift(dal):
    # Regression: HL computes avg_entry to more precision than we sent (e.g. we
    # placed @ 0.96190 but HL reports entryNtl/qty = 0.96199619). The 1e-9
    # tolerance trips every reconcile cycle, flooding Telegram. Since
    # avg_entry is display-only (PnL gate reads from HL directly), we silently
    # adopt HL's value without alerting.
    dal.upsert_position(Position(
        question_idx=42, symbol="@30", qty=15.0, avg_entry=0.96190,
        realized_pnl=0.0, last_update_ts_ns=1, stop_loss_price=0.855,
    ))
    venue_state = ClearinghouseState(
        positions=(VenuePosition(symbol="@30", qty=15.0, avg_entry=0.96199619,
                                  unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(dal, fills_lookup=lambda _: [], symbol_to_question={"@30": 42}).run(
        venue_open=[], venue_state=venue_state, now_ns=2,
    )
    # Local row was silently updated to HL's avg_entry…
    assert dal.get_position(42).avg_entry == 0.96199619
    # …but no drift event surfaces (would have spammed Telegram every cycle).
    assert not any(d.case == "position_mismatch" for d in res.drift_events)


def test_position_both_qty_and_avg_diff_emits_single_qty_drift(dal):
    # If both fields drift, we still only fire one drift event and it carries
    # the qty-diff detail (the load-bearing field). The local row is updated
    # to HL's qty AND avg_entry in the same upsert.
    dal.upsert_position(Position(
        question_idx=42, symbol="@30", qty=10.0, avg_entry=0.95,
        realized_pnl=0.0, last_update_ts_ns=1, stop_loss_price=0.855,
    ))
    venue_state = ClearinghouseState(
        positions=(VenuePosition(symbol="@30", qty=8.0, avg_entry=0.961,
                                  unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(dal, fills_lookup=lambda _: [], symbol_to_question={"@30": 42}).run(
        venue_open=[], venue_state=venue_state, now_ns=2,
    )
    assert dal.get_position(42).qty == 8.0
    assert dal.get_position(42).avg_entry == 0.961
    pos_drift = [d for d in res.drift_events if d.case == "position_mismatch"]
    assert len(pos_drift) == 1
    assert pos_drift[0].detail.get("hl_qty") == "8.0"
    assert pos_drift[0].detail.get("db_qty") == "10.0"


def test_positions_unknown_skips_position_reconciliation(dal):
    # When the venue position set couldn't be fetched (PM data-api error →
    # positions_known=False), the reconciler must NOT treat the empty set as
    # truth. Otherwise it would vanish-delete every live position on a transient
    # API blip.
    dal.upsert_position(Position(
        question_idx=42, symbol="@30", qty=10.0, avg_entry=0.95,
        realized_pnl=0.0, last_update_ts_ns=1, stop_loss_price=0.855,
    ))
    res = Reconciler(dal, fills_lookup=lambda _: [], symbol_to_question={"@30": 42}).run(
        venue_open=[],
        venue_state=ClearinghouseState(
            positions=(), account_value_usd=0, positions_known=False,
        ),
        now_ns=2,
    )
    assert dal.get_position(42) is not None  # not vanished
    assert res.vanished_positions == []


def test_alert_only_does_not_vanish_local_on_venue_absence(dal):
    # PM live (apply_position_changes=False): the data-api drops/flaps
    # positions, so venue-absence must NOT delete the local position or emit a
    # settlement. Only an informational drift; closure comes from fills + the
    # endDate gamma settlement.
    dal.upsert_position(Position(
        question_idx=42, symbol="tok", qty=10.0, avg_entry=0.95,
        realized_pnl=0.0, last_update_ts_ns=1, stop_loss_price=-1.0,
    ))
    res = Reconciler(
        dal, fills_lookup=lambda _: [], symbol_to_question={"tok": 42},
        apply_position_changes=False,
    ).run(venue_open=[],
          venue_state=ClearinghouseState(positions=(), account_value_usd=0),
          now_ns=2)
    assert dal.get_position(42) is not None       # not deleted
    assert res.vanished_positions == []           # no settlement triggered
    assert any(
        d.detail.get("resolution") == "venue_absent_alert_only"
        for d in res.drift_events
    )


def test_alert_only_keeps_local_qty_on_drift(dal):
    # PM live: a SELL reduced local to dust (0.006) but the data-api still
    # reports the pre-sell 51.536 (lag). Alert-only must KEEP the fresh local
    # qty (our fill ledger is truth) and only alert — overwriting it reverted
    # the fill and caused the 2026-06-02 double-exit.
    dal.upsert_position(Position(
        question_idx=42, symbol="tok", qty=0.006, avg_entry=0.97,
        realized_pnl=0.0, last_update_ts_ns=1, stop_loss_price=-1.0,
    ))
    venue = ClearinghouseState(
        positions=(VenuePosition(
            symbol="tok", qty=51.536, avg_entry=0.97, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(
        dal, fills_lookup=lambda _: [], symbol_to_question={"tok": 42},
        apply_position_changes=False,
    ).run(venue_open=[], venue_state=venue, now_ns=2)
    assert dal.get_position(42).qty == 0.006      # local kept, not reverted
    assert any(
        d.detail.get("resolution") == "qty_mismatch_alert_only"
        for d in res.drift_events
    )


def test_alert_only_does_not_adopt_venue_orphan(dal):
    # PM live: a venue position we don't track locally is NOT adopted (re-
    # adopting a just-closed position from the laggy data-api would resurrect
    # it). Only an alert. (Apply-mode adoption is covered separately.)
    venue = ClearinghouseState(
        positions=(VenuePosition(
            symbol="tok", qty=5.0, avg_entry=0.4, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(
        dal, fills_lookup=lambda _: [], symbol_to_question={"tok": 99},
        apply_position_changes=False,
    ).run(venue_open=[], venue_state=venue, now_ns=2)
    assert dal.get_position(99) is None           # not adopted
    assert any(
        d.detail.get("resolution") == "venue_orphan_alert_only"
        for d in res.drift_events
    )


def test_apply_mode_adopts_venue_orphan(dal):
    # Restart / HL (apply_position_changes=True, the default): a venue position
    # we don't track locally IS adopted into the DB.
    venue = ClearinghouseState(
        positions=(VenuePosition(
            symbol="tok", qty=5.0, avg_entry=0.4, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(
        dal, fills_lookup=lambda _: [], symbol_to_question={"tok": 99},
    ).run(venue_open=[], venue_state=venue, now_ns=2)
    assert dal.get_position(99) is not None
    assert dal.get_position(99).qty == 5.0
    assert any(
        d.detail.get("resolution") == "adopted_venue_orphan"
        for d in res.drift_events
    )


def test_position_disappearance_drops_local_position(dal):
    # A local position with no matching venue position is overwhelmingly a
    # HIP-4 settlement auto-close on HL. The reconciler surfaces it on
    # ``vanished_positions`` so the runtime can publish a settlement Exit
    # alert and mark the question settled in MarketState (suppressing the
    # stale-data halt the leg would otherwise trip on its now-silent book).
    # We deliberately do NOT emit a position_mismatch DRIFT here — the Exit
    # is the canonical alert; firing both would double-notify on every roll.
    dal.upsert_position(Position(
        question_idx=42, symbol="@30", qty=10.0, avg_entry=0.95,
        realized_pnl=1.25, last_update_ts_ns=1, stop_loss_price=0.855,
    ))
    res = Reconciler(dal, fills_lookup=lambda _: [], symbol_to_question={"@30": 42}).run(
        venue_open=[], venue_state=ClearinghouseState(positions=(), account_value_usd=0), now_ns=2,
    )
    assert dal.get_position(42) is None
    assert not any(d.case == "position_mismatch" for d in res.drift_events)
    assert len(res.vanished_positions) == 1
    qidx, sym, lp = res.vanished_positions[0]
    assert (qidx, sym) == (42, "@30")
    # Snapshot is taken BEFORE delete_position so qty/PnL survive for the
    # caller's Exit payload.
    assert lp.qty == 10.0
    assert lp.realized_pnl == 1.25


def test_zero_qty_venue_position_treated_as_vanished_on_pm(dal):
    # Unlike HL HIP-4 (which removes the position row entirely on settlement),
    # Polymarket leaves the venue position at qty=0 until redemption. The
    # reconciler must treat that as a vanished position so the caller's
    # _close_settled flow publishes the settlement Exit and deletes the local
    # row — otherwise the strategy keeps seeing a stale local position long
    # after the market resolved.
    dal.upsert_position(Position(
        question_idx=77, symbol="pm-0xabc", qty=10.0, avg_entry=0.55,
        realized_pnl=4.50, last_update_ts_ns=1, stop_loss_price=-1.0,
    ))
    venue_state = ClearinghouseState(
        positions=(VenuePosition(symbol="pm-0xabc", qty=0.0, avg_entry=0.55,
                                  unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(
        dal, fills_lookup=lambda _: [], symbol_to_question={"pm-0xabc": 77},
    ).run(venue_open=[], venue_state=venue_state, now_ns=2)

    assert dal.get_position(77) is None
    assert len(res.vanished_positions) == 1
    qidx, sym, lp = res.vanished_positions[0]
    assert (qidx, sym) == (77, "pm-0xabc")
    assert lp.qty == 10.0
    assert lp.realized_pnl == 4.50
    # No false-positive position_mismatch drift — the Exit is the canonical
    # alert (mirrors the HL vanish path).
    assert not any(d.case == "position_mismatch" for d in res.drift_events)


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
