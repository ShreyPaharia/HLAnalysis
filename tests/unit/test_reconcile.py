from __future__ import annotations

import pytest

from hlanalysis.engine.hl_client import (
    ClearinghouseState,
    OpenOrderRow,
    UserFillRow,
    VenuePosition,
)
from hlanalysis.engine.reconcile import ReconcileResult, Reconciler
from hlanalysis.engine.state import OpenOrder, Position, StateDAL


@pytest.fixture
def dal(tmp_path):
    d = StateDAL(tmp_path / "state.db")
    d.run_migrations()
    return d


def _row(cloid: str, symbol: str = "@30", price: float = 0.95, size: float = 10.0) -> OpenOrderRow:
    return OpenOrderRow(
        cloid=cloid, venue_oid=f"v-{cloid}", symbol=symbol, side="buy", price=price, size=size, placed_ts_ns=1
    )


def _db_order(cloid: str, status: str = "open") -> OpenOrder:
    return OpenOrder(
        cloid=cloid,
        venue_oid=f"v-{cloid}",
        question_idx=42,
        symbol="@30",
        side="buy",
        price=0.95,
        size=10.0,
        status=status,
        placed_ts_ns=1,
        last_update_ts_ns=1,
        strategy_id="x",
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
    fills = [
        UserFillRow(
            fill_id="f-1",
            cloid="hla-1",
            symbol="@30",
            side="buy",
            price=0.95,
            size=10.0,
            fee=0.05,
            ts_ns=1,
        )
    ]
    r = Reconciler(dal, fills_lookup=lambda cloid: fills if cloid == "hla-1" else [])
    res = r.run(venue_open=[], venue_state=ClearinghouseState(positions=(), account_value_usd=0), now_ns=2)
    assert dal.get_order("hla-1").status == "filled"
    assert dal.fills_for_cloid("hla-1")  # replayed
    drift = [d for d in res.drift_events if d.case == "state_mismatch"]
    assert drift
    # The drift must carry the order's question_idx (it's in hand via db_o) so
    # the Telegram alert isn't rendered with q=None and is attributable.
    assert drift[0].question_idx == 42


def test_local_ghost_fill_discovery_logs_unapplied_position_delta(dal):
    """When reconcile discovers a venue fill for a locally-live order it marks
    the order filled + replays the Fill row, but does NOT touch the position
    table. That gap is the #1 stale-position root-cause suspect (2026-06-04):
    a sell fill the router never booked leaves the position open forever. Log
    the net position delta the discovered fills represent + that it's
    order-level only, so the next occurrence is unambiguous."""
    from loguru import logger

    dal.upsert_order(_db_order("hla-1"))  # local buy 10 @ @30
    fills = [
        UserFillRow(
            fill_id="f-1",
            cloid="hla-1",
            symbol="@30",
            side="buy",
            price=0.95,
            size=10.0,
            fee=0.0,
            ts_ns=1,
        )
    ]
    r = Reconciler(dal, fills_lookup=lambda c: fills if c == "hla-1" else [])
    msgs: list[str] = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="INFO")
    try:
        r.run(venue_open=[], venue_state=ClearinghouseState(positions=(), account_value_usd=0), now_ns=2)
    finally:
        logger.remove(sink)
    audit = [m for m in msgs if "reconcile_fill_discovered" in m]
    assert audit, f"no reconcile_fill_discovered log; got {msgs}"
    joined = " ".join(audit)
    assert "hla-1" in joined and "net_delta=10" in joined, joined


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
    dal.upsert_position(
        Position(
            question_idx=42,
            symbol="@30",
            qty=10.0,
            avg_entry=0.95,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=0.855,
        )
    )
    venue_state = ClearinghouseState(
        positions=(VenuePosition(symbol="@30", qty=8.0, avg_entry=0.95, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(dal, fills_lookup=lambda _: [], symbol_to_question={"@30": 42}).run(
        venue_open=[],
        venue_state=venue_state,
        now_ns=2,
    )
    assert dal.get_position(42).qty == 8.0
    assert any(d.case == "position_mismatch" for d in res.drift_events)


def test_position_avg_entry_only_diff_silently_adopts_no_drift(dal):
    # Regression: HL computes avg_entry to more precision than we sent (e.g. we
    # placed @ 0.96190 but HL reports entryNtl/qty = 0.96199619). The 1e-9
    # tolerance trips every reconcile cycle, flooding Telegram. Since
    # avg_entry is display-only (PnL gate reads from HL directly), we silently
    # adopt HL's value without alerting.
    dal.upsert_position(
        Position(
            question_idx=42,
            symbol="@30",
            qty=15.0,
            avg_entry=0.96190,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=0.855,
        )
    )
    venue_state = ClearinghouseState(
        positions=(VenuePosition(symbol="@30", qty=15.0, avg_entry=0.96199619, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(dal, fills_lookup=lambda _: [], symbol_to_question={"@30": 42}).run(
        venue_open=[],
        venue_state=venue_state,
        now_ns=2,
    )
    # Local row was silently updated to HL's avg_entry…
    assert dal.get_position(42).avg_entry == 0.96199619
    # …but no drift event surfaces (would have spammed Telegram every cycle).
    assert not any(d.case == "position_mismatch" for d in res.drift_events)


def test_position_qty_subshare_rounding_diff_no_drift(dal):
    # Regression: PM share qty comes from two independent float sources — the
    # data-api `/positions` `size` (4dp, e.g. 56.1685) vs our fill ledger's
    # summed `takingAmount`. They agree to a fraction of a share but differ
    # below ~1e-3, which the old exact `>1e-9` check flagged as a
    # position_mismatch DRIFT every reconcile cycle, flooding Telegram. A real
    # fill discrepancy is ≥~1 share; sub-share rounding is not a mismatch.
    dal.upsert_position(
        Position(
            question_idx=42,
            symbol="tok",
            qty=56.16850001,
            avg_entry=0.89,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=-1.0,
        )
    )
    venue_state = ClearinghouseState(
        positions=(VenuePosition(symbol="tok", qty=56.1685, avg_entry=0.89, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"tok": 42},
        apply_position_changes=False,
    ).run(venue_open=[], venue_state=venue_state, now_ns=2)
    assert not any(d.case == "position_mismatch" for d in res.drift_events)


def test_position_both_qty_and_avg_diff_emits_single_qty_drift(dal):
    # If both fields drift, we still only fire one drift event and it carries
    # the qty-diff detail (the load-bearing field). The local row is updated
    # to HL's qty AND avg_entry in the same upsert.
    dal.upsert_position(
        Position(
            question_idx=42,
            symbol="@30",
            qty=10.0,
            avg_entry=0.95,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=0.855,
        )
    )
    venue_state = ClearinghouseState(
        positions=(VenuePosition(symbol="@30", qty=8.0, avg_entry=0.961, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(dal, fills_lookup=lambda _: [], symbol_to_question={"@30": 42}).run(
        venue_open=[],
        venue_state=venue_state,
        now_ns=2,
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
    dal.upsert_position(
        Position(
            question_idx=42,
            symbol="@30",
            qty=10.0,
            avg_entry=0.95,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=0.855,
        )
    )
    res = Reconciler(dal, fills_lookup=lambda _: [], symbol_to_question={"@30": 42}).run(
        venue_open=[],
        venue_state=ClearinghouseState(
            positions=(),
            account_value_usd=0,
            positions_known=False,
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
    dal.upsert_position(
        Position(
            question_idx=42,
            symbol="tok",
            qty=10.0,
            avg_entry=0.95,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=-1.0,
        )
    )
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"tok": 42},
        apply_position_changes=False,
    ).run(venue_open=[], venue_state=ClearinghouseState(positions=(), account_value_usd=0), now_ns=2)
    assert dal.get_position(42) is not None  # not deleted
    assert res.vanished_positions == []  # no settlement triggered
    assert any(d.detail.get("resolution") == "venue_absent_alert_only" for d in res.drift_events)


def test_alert_only_clears_dust_position_on_venue_absence(dal):
    # PM live: a closed round-trip leaves sub-precision dust behind. PM market
    # sells round the share amount to 2dp, so closing a 54.934055-share buy
    # sells 54.93 and strands 0.004055 shares. That dust is un-sellable (below
    # PM's min order size) AND un-reported (below the data-api's dust filter),
    # so the venue shows it absent forever. Without a dust floor the venue-
    # absent branch re-fires `venue_absent_alert_only` every reconcile cycle —
    # ~1 Telegram DRIFT/min, permanently (incident 2026-06-05, q729375628).
    # When the LOCAL qty is itself dust and the venue reports it absent, the
    # position is genuinely stranded-closed: clear the row so the flood stops,
    # emit one informational drift, and do NOT route a settlement Exit (the
    # round-trip PnL is already booked in the sell fill's closed_pnl).
    dal.upsert_position(
        Position(
            question_idx=42,
            symbol="tok",
            qty=0.00405500000000103,
            avg_entry=0.91,
            realized_pnl=2.1972,
            last_update_ts_ns=1,
            stop_loss_price=-1.0,
        )
    )
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"tok": 42},
        apply_position_changes=False,
    ).run(venue_open=[], venue_state=ClearinghouseState(positions=(), account_value_usd=0), now_ns=2)
    assert dal.get_position(42) is None  # dust row cleared
    assert res.vanished_positions == []  # no settlement Exit / re-book
    assert any(d.detail.get("resolution") == "dust_cleared" for d in res.drift_events)


def test_alert_only_keeps_whole_share_position_on_venue_absence(dal):
    # Boundary guard for the dust floor: a whole-share position the venue
    # transiently drops (data-api flap) must NOT be cleared — only a real,
    # below-min-order dust qty is. A 0.5-share position is far above the dust
    # floor, so it keeps the legacy keep-local + `venue_absent_alert_only`.
    dal.upsert_position(
        Position(
            question_idx=42,
            symbol="tok",
            qty=0.5,
            avg_entry=0.91,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=-1.0,
        )
    )
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"tok": 42},
        apply_position_changes=False,
    ).run(venue_open=[], venue_state=ClearinghouseState(positions=(), account_value_usd=0), now_ns=2)
    assert dal.get_position(42) is not None  # not cleared
    assert any(d.detail.get("resolution") == "venue_absent_alert_only" for d in res.drift_events)


def test_alert_only_clears_dust_position_when_venue_reports_stale_qty(dal):
    # PM live: a reduce-only SELL fills the 2dp-floored share amount and strands
    # sub-precision dust (0.0079 — the 2026-06-06 v31_pm wedge). The data-api
    # still reports the pre-sell qty (58.1279, lag). The OLD behaviour kept the
    # dust + re-fired `qty_mismatch_alert_only` every reconcile cycle forever.
    # Since the local qty is un-sellable dust, clear the row (NOT adopt the stale
    # venue qty — that would resurrect a closed position → 2026-06-02 double-exit).
    dal.upsert_position(
        Position(
            question_idx=42,
            symbol="tok",
            qty=0.0079,
            avg_entry=0.90,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=-1.0,
        )
    )
    venue = ClearinghouseState(
        positions=(VenuePosition(symbol="tok", qty=58.1279, avg_entry=0.90, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"tok": 42},
        apply_position_changes=False,
    ).run(venue_open=[], venue_state=venue, now_ns=2)
    assert dal.get_position(42) is None  # dust row cleared
    assert any(d.detail.get("resolution") == "dust_cleared" for d in res.drift_events)
    # NOT adopted as the stale venue qty (would resurrect a closed position).
    assert not any(d.detail.get("resolution") == "adopted_venue_orphan" for d in res.drift_events)


def test_alert_only_keeps_real_local_qty_on_drift(dal):
    # Boundary guard: a REAL (non-dust) local qty that drifts from the laggy
    # data-api must still be KEPT (our fill ledger is truth) and only alerted —
    # overwriting it reverted the fill and caused the 2026-06-02 double-exit.
    dal.upsert_position(
        Position(
            question_idx=42,
            symbol="tok",
            qty=30.0,
            avg_entry=0.97,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=-1.0,
        )
    )
    venue = ClearinghouseState(
        positions=(VenuePosition(symbol="tok", qty=51.536, avg_entry=0.97, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"tok": 42},
        apply_position_changes=False,
    ).run(venue_open=[], venue_state=venue, now_ns=2)
    assert dal.get_position(42).qty == 30.0  # local kept, not reverted
    assert any(d.detail.get("resolution") == "qty_mismatch_alert_only" for d in res.drift_events)


def test_alert_only_ack_vs_indexer_subshare_gap_no_drift(dal):
    # PM live: the BUY fill size we book (notional/limit → 55.5444) routinely
    # differs from the data-api `/positions` size (the settled on-chain balance,
    # 55.5523) by ~8e-3 shares — bigger than 4dp truncation. The old abs_tol of
    # 1e-3 flagged this as a `qty_mismatch_alert_only` on EVERY held PM position
    # every reconcile cycle. The widened tolerance treats sub-share ack-vs-
    # indexer divergence as noise (a real missed fill is ≥~1 share).
    dal.upsert_position(
        Position(
            question_idx=42,
            symbol="tok",
            qty=55.5444,
            avg_entry=0.90,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=-1.0,
        )
    )
    venue = ClearinghouseState(
        positions=(VenuePosition(symbol="tok", qty=55.5523, avg_entry=0.90, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"tok": 42},
        apply_position_changes=False,
    ).run(venue_open=[], venue_state=venue, now_ns=2)
    assert not any(d.case == "position_mismatch" for d in res.drift_events)


def test_alert_only_does_not_adopt_venue_orphan(dal):
    # PM live: a venue position we don't track locally is NOT adopted (re-
    # adopting a just-closed position from the laggy data-api would resurrect
    # it). Only an alert. (Apply-mode adoption is covered separately.)
    venue = ClearinghouseState(
        positions=(VenuePosition(symbol="tok", qty=5.0, avg_entry=0.4, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"tok": 99},
        apply_position_changes=False,
    ).run(venue_open=[], venue_state=venue, now_ns=2)
    assert dal.get_position(99) is None  # not adopted
    assert any(d.detail.get("resolution") == "venue_orphan_alert_only" for d in res.drift_events)


def test_apply_mode_adopts_venue_orphan(dal):
    # Restart / HL (apply_position_changes=True, the default): a venue position
    # we don't track locally IS adopted into the DB.
    venue = ClearinghouseState(
        positions=(VenuePosition(symbol="tok", qty=5.0, avg_entry=0.4, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"tok": 99},
    ).run(venue_open=[], venue_state=venue, now_ns=2)
    assert dal.get_position(99) is not None
    assert dal.get_position(99).qty == 5.0
    assert any(d.detail.get("resolution") == "adopted_venue_orphan" for d in res.drift_events)


def test_position_disappearance_drops_local_position(dal):
    # A local position with no matching venue position is overwhelmingly a
    # HIP-4 settlement auto-close on HL. The reconciler surfaces it on
    # ``vanished_positions`` so the runtime can publish a settlement Exit
    # alert and mark the question settled in MarketState (suppressing the
    # stale-data halt the leg would otherwise trip on its now-silent book).
    # We deliberately do NOT emit a position_mismatch DRIFT here — the Exit
    # is the canonical alert; firing both would double-notify on every roll.
    dal.upsert_position(
        Position(
            question_idx=42,
            symbol="@30",
            qty=10.0,
            avg_entry=0.95,
            realized_pnl=1.25,
            last_update_ts_ns=1,
            stop_loss_price=0.855,
        )
    )
    res = Reconciler(dal, fills_lookup=lambda _: [], symbol_to_question={"@30": 42}).run(
        venue_open=[],
        venue_state=ClearinghouseState(positions=(), account_value_usd=0),
        now_ns=2,
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
    dal.upsert_position(
        Position(
            question_idx=77,
            symbol="pm-0xabc",
            qty=10.0,
            avg_entry=0.55,
            realized_pnl=4.50,
            last_update_ts_ns=1,
            stop_loss_price=-1.0,
        )
    )
    venue_state = ClearinghouseState(
        positions=(VenuePosition(symbol="pm-0xabc", qty=0.0, avg_entry=0.55, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"pm-0xabc": 77},
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
        positions=(VenuePosition(symbol="#551", qty=500.0, avg_entry=0.988, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"#551": 10},
    ).run(venue_open=[], venue_state=venue_state, now_ns=2)

    booked = dal.get_position(10)
    assert booked is not None, "venue-orphan position was not adopted"
    assert booked.symbol == "#551"
    assert booked.qty == 500.0
    assert booked.avg_entry == 0.988
    assert any(
        d.case == "position_mismatch" and (d.detail or {}).get("resolution") == "adopted_venue_orphan"
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
    dal.upsert_order(
        OpenOrder(
            cloid=f"hla-v1-{HEX}",
            venue_oid=SHARED_OID,
            question_idx=42,
            symbol="@30",
            side="buy",
            price=0.95,
            size=10.0,
            status="open",
            placed_ts_ns=1,
            last_update_ts_ns=1,
            strategy_id="x",
        )
    )
    venue_open = [
        OpenOrderRow(
            cloid=f"0x{HEX}", venue_oid=SHARED_OID, symbol="@30", side="buy", price=0.95, size=10.0, placed_ts_ns=1
        )
    ]
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
    dal.upsert_order(
        OpenOrder(
            cloid=f"hla-v1-{HEX}",
            venue_oid=SHARED_OID,
            question_idx=42,
            symbol="@30",
            side="buy",
            price=0.95,
            size=10.0,
            status="open",
            placed_ts_ns=1,
            last_update_ts_ns=1,
            strategy_id="x",
        )
    )
    venue_open = [
        OpenOrderRow(
            cloid=f"0x{HEX}", venue_oid=SHARED_OID, symbol="@30", side="buy", price=0.97, size=10.0, placed_ts_ns=1
        )
    ]
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
        positions=(VenuePosition(symbol="#999", qty=10.0, avg_entry=0.5, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={},
    ).run(venue_open=[], venue_state=venue_state, now_ns=2)

    # No new position rows created
    assert dal.all_positions() == []
    # But the drift event must surface so we can act on it
    assert any(d.case == "position_mismatch" and (d.detail or {}).get("symbol") == "#999" for d in res.drift_events)


def test_settled_qidx_suppresses_venue_orphan_alert(dal):
    # Settlement→redemption gap (incident 2026-06-12 q700064348). After a PM
    # market settles, the engine deletes its local position at settlement, but
    # Polymarket keeps reporting the winning shares for ~15 min until it
    # auto-redeems them on-chain. Without this guard the reconciler fires a
    # `venue_orphan_alert_only` DRIFT every 15s cycle for that whole window.
    # A settled qidx is pending auto-redemption — expected, not drift: suppress
    # the alert AND don't adopt (adopting resurrects a position about to be
    # redeemed). The 6h RedemptionTimeout watchdog backstops a stuck redeem.
    venue = ClearinghouseState(
        positions=(VenuePosition(symbol="tok", qty=51.0102, avg_entry=0.98, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"tok": 700064348},
        apply_position_changes=False,
        settled_qidxs={700064348},
    ).run(venue_open=[], venue_state=venue, now_ns=2)
    assert not any(d.case == "position_mismatch" for d in res.drift_events)
    assert dal.get_position(700064348) is None  # not adopted


def test_settled_qidx_skips_apply_mode_adoption(dal):
    # Even in apply mode (restart / HL), a settled qidx must NOT be adopted from
    # the venue — the winning shares are pending auto-redemption and adopting
    # them resurrects a position that's about to vanish (then can't be sold).
    venue = ClearinghouseState(
        positions=(VenuePosition(symbol="tok", qty=51.0102, avg_entry=0.98, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"tok": 700064348},
        settled_qidxs={700064348},
    ).run(venue_open=[], venue_state=venue, now_ns=2)
    assert dal.get_position(700064348) is None
    assert not any(d.case == "position_mismatch" for d in res.drift_events)


def test_unsettled_qidx_still_alerts_venue_orphan(dal):
    # Guard against over-suppression: a venue orphan whose qidx is NOT settled
    # must still alert (the settled-suppression must be narrowly scoped).
    venue = ClearinghouseState(
        positions=(VenuePosition(symbol="tok", qty=5.0, avg_entry=0.4, unrealized_pnl=0.0),),
        account_value_usd=0,
    )
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"tok": 99},
        apply_position_changes=False,
        settled_qidxs={700064348},
    ).run(venue_open=[], venue_state=venue, now_ns=2)
    assert any(d.detail.get("resolution") == "venue_orphan_alert_only" for d in res.drift_events)


def test_settled_qidx_suppresses_venue_absent_alert(dal):
    # The mirror window: local still holds the settled position but the venue
    # has already dropped it (redeemed). In alert-only (PM) mode that normally
    # fires `venue_absent_alert_only` every cycle; suppress it for a settled
    # qidx (expected post-settlement, not drift).
    dal.upsert_position(
        Position(
            question_idx=700064348,
            symbol="tok",
            qty=51.0102,
            avg_entry=0.98,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=-1.0,
        )
    )
    venue = ClearinghouseState(positions=(), account_value_usd=0)
    res = Reconciler(
        dal,
        fills_lookup=lambda _: [],
        symbol_to_question={"tok": 700064348},
        apply_position_changes=False,
        settled_qidxs={700064348},
    ).run(venue_open=[], venue_state=venue, now_ns=2)
    assert not any(d.case == "position_mismatch" for d in res.drift_events)
