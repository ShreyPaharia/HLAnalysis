from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger

from .hl_client import (
    ClearinghouseState, OpenOrderRow, UserFillRow, _extract_cloid_hex32,
)
from .risk_events import ReconcileDrift
from .state import Fill, OpenOrder, Position, StateDAL


CLOID_PREFIX = "hla-"

# Position quantities are compared across two independently-sourced floats: the
# venue's reported size (HL szi / spot balance, or the PM data-api `/positions`
# `size` truncated to 4dp) versus our fill ledger's summed fill sizes. They
# express the same holding but disagree in the low decimals — a 56.1685-share
# PM position routinely differs from our accumulated 56.16850001 by sub-share
# rounding. An exact `>1e-9` check treats that noise as a position_mismatch and
# re-fires a DRIFT alert every reconcile cycle. A real fill discrepancy (a
# missed buy/sell) is on the order of whole shares, so we only flag a mismatch
# when the quantities differ by more than rounding noise. abs_tol clears 4dp
# truncation (≤5e-5) with margin; rel_tol keeps it scale-safe for large lots.
_QTY_MISMATCH_REL_TOL = 1e-4
_QTY_MISMATCH_ABS_TOL = 1e-3

# Dust floor for the alert-only venue-absent branch. PM market sells round the
# share amount to 2dp, so closing a non-round-2dp buy strands sub-precision dust
# (a 54.934055-share buy closes by selling 54.93, leaving 0.004055 shares). That
# residual is BOTH un-sellable (below PM's ~$1 min order size) and un-reported
# (below the data-api's dust filter), so the venue shows it absent forever and
# the alert-only branch re-fires `venue_absent_alert_only` every reconcile cycle
# (~1 Telegram DRIFT/min, permanently — incident 2026-06-05, q729375628). When
# the LOCAL qty is itself this small AND the venue reports it absent, the
# position is genuinely stranded-closed, not a data-api flap of a real holding:
# clear the row so the flood stops. The threshold sits above the max 2dp sell
# residual (≤5e-3) with margin and ~100x below a 1-share min order, so it can
# never swallow a real lagging position (which is whole shares). PM prices are
# ≤1.0, so 1e-2 shares is ≤$0.01 notional.
_DUST_QTY_ABS_TOL = 1e-2


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    drift_events: list[ReconcileDrift]
    # (cloid, symbol) pairs the caller should defensively cancel. Symbol must be
    # non-empty — HL's cancelByCloid rejects empty coin and the SDK silently
    # reports outer status='ok' even when the per-cancel statuses[] failed.
    orphans_to_cancel: list[tuple[str, str]]
    # Local positions that no longer exist on the venue — almost always means
    # the HIP-4 market settled and HL auto-closed the position. The caller is
    # expected to (a) mark the question settled in MarketState so subsequent
    # stale-data checks skip the now-silent leg, and (b) publish a settlement
    # Exit event so operators see a 🏁 alert instead of a generic DRIFT.
    # The list carries the pre-delete Position snapshot so the caller can
    # populate the Exit's qty/realized_pnl without re-reading the DB.
    vanished_positions: list[tuple[int, str, Position]]


class Reconciler:
    """Three-way merge between local DB, HL openOrders, HL clearinghouseState.

    Spec §5.2 + §5.5. Pure synchronous logic — no IO. The caller wires in:
      - `fills_lookup(cloid)` → list[UserFillRow] for resolving local-ghost cases
      - `symbol_to_question` → optional mapping from venue symbol to question_idx
        for position-row attribution.
      - `cloid_prefix` → retained for backward compatibility but no longer used
        for matching. HL normalizes the engine's `hla-vN-<hex32>` cloid to
        `0x<hex32>`, so the join is now done on the 32-hex tail
        (via `_extract_cloid_hex32`). Per-account isolation comes from
        `info.open_orders(account_address)` being address-scoped on the venue
        side and from the per-account `state.db` on the local side.
    """

    def __init__(
        self,
        dal: StateDAL,
        *,
        fills_lookup: Callable[[str], list[UserFillRow]],
        symbol_to_question: dict[str, int] | None = None,
        cloid_prefix: str = CLOID_PREFIX,
        account_alias: str = "",
        apply_position_changes: bool = True,
    ) -> None:
        self.dal = dal
        self.fills_lookup = fills_lookup
        self.symbol_to_question = symbol_to_question or {}
        self.cloid_prefix = cloid_prefix
        # When False, the positions pass DETECTS venue/local mismatches and
        # emits drift alerts but does NOT mutate the DB (no vanish, no
        # qty-overwrite, no adopt). Used for the PM live loop: PM's only
        # position-truth source is the data-api indexer, which lags our own
        # fills by seconds and flaps — so live we trust our fill ledger and only
        # alert on differences. Venue truth is APPLIED at restart (the
        # RestartDriftGate, to recover fills missed while down) and continuously
        # for HL (clearinghouseState is instant + authoritative). PM positions
        # close via our sell fills + the endDate-gated gamma SettlementEvent.
        self.apply_position_changes = apply_position_changes
        # Stamped onto every ReconcileDrift this Reconciler emits so alerts
        # carry the account that detected the drift.
        self.account_alias = account_alias

    def run(
        self,
        *,
        venue_open: list[OpenOrderRow],
        venue_state: ClearinghouseState,
        now_ns: int,
    ) -> ReconcileResult:
        drift: list[ReconcileDrift] = []
        orphans: list[tuple[str, str]] = []
        vanished: list[tuple[int, str, Position]] = []

        # --- orders ---
        # Match local↔venue by the 32-hex tail of the cloid. HL normalizes
        # the engine's `hla-v1-<hex32>` form to `0x<hex32>` on its side, so a
        # literal prefix filter would exclude every venue order. Normalizing
        # both sides to the bare hex makes the join work regardless of which
        # form HL hands back. The per-account isolation that the legacy
        # cloid_prefix filter was guarding is already provided by
        # info.open_orders(account_address) being address-scoped (and by the
        # per-account state.db on the local side).
        def _hex(c: str) -> str:
            return _extract_cloid_hex32(c)

        venue_by_hex = {_hex(r.cloid): r for r in venue_open}
        local_live = list(self.dal.live_orders())
        local_by_hex = {_hex(o.cloid): o for o in local_live}

        # local-ghost: in DB live, not on venue
        for cloid_hex, db_o in local_by_hex.items():
            if cloid_hex in venue_by_hex:
                continue
            # We pass the LOCAL cloid (the internal form) to fills_lookup so
            # downstream filters that key off the engine's cloid work; HL
            # returns its normalized form on the rows themselves and the
            # caller compares hex tails when matching.
            cloid = db_o.cloid
            fills = self.fills_lookup(cloid)
            if fills:
                # Mark filled, replay fills into DB
                for f in fills:
                    self.dal.append_fill(Fill(
                        fill_id=f.fill_id, cloid=cloid, question_idx=db_o.question_idx,
                        symbol=db_o.symbol, side=f.side, price=f.price, size=f.size,
                        fee=f.fee, ts_ns=f.ts_ns,
                    ))
                self.dal.update_order_status(cloid, status="filled", now_ns=now_ns)
                # Diagnostic (incident 2026-06-04, #1 root-cause suspect): this
                # path replays the Fill rows + marks the order filled but does
                # NOT apply the fills to the position table. If the router never
                # booked this fill (PM ack returned no fill info), the position
                # stays open by `net_delta` forever → endless re-exit loop. Log
                # the unapplied delta so the next occurrence is unambiguous.
                net_delta = sum(
                    (f.size if f.side == "buy" else -f.size) for f in fills
                )
                logger.info(
                    "reconcile_fill_discovered cloid={} qidx={} symbol={} "
                    "n_fills={} net_delta={:g} (order marked filled; position "
                    "table NOT auto-applied — verify router booked it)",
                    cloid, db_o.question_idx, db_o.symbol, len(fills), net_delta,
                )
                drift.append(ReconcileDrift(
                    ts_ns=now_ns, account_alias=self.account_alias, case="state_mismatch", cloid=cloid,
                    detail={"resolution": "filled_via_user_fills"},
                ))
            else:
                self.dal.update_order_status(cloid, status="cancelled", now_ns=now_ns)
                drift.append(ReconcileDrift(
                    ts_ns=now_ns, account_alias=self.account_alias, case="local_ghost", cloid=cloid,
                ))

        # venue-orphan: on venue, not in DB live. Orphan cloid is reported in
        # the venue's form (HL hex) — that's what cancel() needs to address it.
        for cloid_hex, vo in venue_by_hex.items():
            if cloid_hex in local_by_hex:
                continue
            orphans.append((vo.cloid, vo.symbol))
            drift.append(ReconcileDrift(
                ts_ns=now_ns, account_alias=self.account_alias, case="venue_orphan", cloid=vo.cloid,
                detail={"symbol": vo.symbol},
            ))

        # state-mismatch: both, fields differ — HL wins. Keyed by hex tail; we
        # carry the local cloid through to upsert_order so the DB row keeps its
        # internal identity.
        for cloid_hex, vo in venue_by_hex.items():
            db_o = local_by_hex.get(cloid_hex)
            if db_o is None:
                continue
            if (
                abs(db_o.price - vo.price) > 1e-9
                or abs(db_o.size - vo.size) > 1e-9
                or db_o.symbol != vo.symbol
                or db_o.venue_oid != vo.venue_oid
            ):
                self.dal.upsert_order(OpenOrder(
                    cloid=db_o.cloid, venue_oid=vo.venue_oid, question_idx=db_o.question_idx,
                    symbol=vo.symbol, side=db_o.side, price=vo.price, size=vo.size,
                    status="open", placed_ts_ns=db_o.placed_ts_ns,
                    last_update_ts_ns=now_ns, strategy_id=db_o.strategy_id,
                ))
                drift.append(ReconcileDrift(
                    ts_ns=now_ns, account_alias=self.account_alias, case="state_mismatch", cloid=db_o.cloid,
                    detail={"hl_price": f"{vo.price}", "db_price": f"{db_o.price}"},
                ))

        # --- positions ---
        # When the venue position set couldn't be fetched (PM data-api error),
        # we have no truth to reconcile against. Skipping is critical: treating
        # an unknown set as empty would vanish-delete every live position.
        if not venue_state.positions_known:
            return ReconcileResult(
                drift_events=drift,
                orphans_to_cancel=orphans,
                vanished_positions=vanished,
            )

        apply = self.apply_position_changes
        venue_by_symbol = {p.symbol: p for p in venue_state.positions}
        local_by_qidx = {p.question_idx: p for p in self.dal.all_positions()}

        for qidx, lp in local_by_qidx.items():
            vp = venue_by_symbol.get(lp.symbol)
            if vp is None or abs(vp.qty) < 1e-9:
                # Position absent on venue (or present at qty=0). On HL this is
                # a HIP-4 settlement auto-close: surface it as a
                # vanished_positions entry so the runtime publishes a settlement
                # Exit + marks the question settled (suppressing stale_data_halt
                # on the now-silent book). In alert-only mode (PM live) venue
                # absence is NOT trusted as a close — the data-api drops
                # positions transiently / lags our own sell fills — so we only
                # emit an informational drift and leave local state to the fill
                # ledger + the endDate gamma settlement.
                if apply:
                    vanished.append((qidx, lp.symbol, lp))
                    self.dal.delete_position(qidx)
                elif abs(lp.qty) <= _DUST_QTY_ABS_TOL:
                    # Stranded sub-precision dust (un-sellable + un-reported):
                    # delete the row so the alert-only branch stops re-firing
                    # forever. No vanished_positions entry — the round-trip PnL
                    # is already booked in the sell fill's closed_pnl, so routing
                    # a settlement Exit here would double-book it and surface a
                    # misleading 🏁 on un-sellable dust. One informational drift.
                    self.dal.delete_position(qidx)
                    drift.append(ReconcileDrift(
                        ts_ns=now_ns, account_alias=self.account_alias,
                        case="position_mismatch", question_idx=qidx,
                        detail={"resolution": "venue_absent_dust_cleared",
                                "symbol": lp.symbol, "db_qty": f"{lp.qty}"},
                    ))
                else:
                    drift.append(ReconcileDrift(
                        ts_ns=now_ns, account_alias=self.account_alias,
                        case="position_mismatch", question_idx=qidx,
                        detail={"resolution": "venue_absent_alert_only",
                                "symbol": lp.symbol, "db_qty": f"{lp.qty}"},
                    ))
                continue
            qty_diff = not math.isclose(
                vp.qty, lp.qty,
                rel_tol=_QTY_MISMATCH_REL_TOL, abs_tol=_QTY_MISMATCH_ABS_TOL,
            )
            avg_diff = abs(vp.avg_entry - lp.avg_entry) > 1e-9
            if qty_diff or avg_diff:
                # In apply mode the venue wins (HL is authoritative): adopt its
                # qty/avg. In alert-only mode we KEEP local (our fill ledger is
                # truth) and only alert — overwriting with a stale data-api qty
                # reverts our own fills (2026-06-02 double-exit).
                if apply:
                    # avg_entry is display-only (the daily-loss gate reads PnL
                    # from HL directly); qty is load-bearing (risk caps +
                    # have_position), so qty drift surfaces as an alert.
                    self.dal.upsert_position(Position(
                        question_idx=qidx, symbol=lp.symbol, qty=vp.qty,
                        avg_entry=vp.avg_entry, realized_pnl=lp.realized_pnl,
                        last_update_ts_ns=now_ns, stop_loss_price=lp.stop_loss_price,
                    ))
                if qty_diff:
                    detail = {"hl_qty": f"{vp.qty}", "db_qty": f"{lp.qty}"}
                    if not apply:
                        detail["resolution"] = "qty_mismatch_alert_only"
                    drift.append(ReconcileDrift(
                        ts_ns=now_ns, account_alias=self.account_alias,
                        case="position_mismatch", question_idx=qidx,
                        detail=detail,
                    ))

        # Position on venue we don't track locally. In apply mode (restart / HL)
        # adopt it into the DB so risk caps + the strategy's have_position HOLD
        # branch see it (without adoption the strategy re-fires entries). In
        # alert-only mode (PM live) we don't adopt from the laggy data-api —
        # re-adopting a just-closed position would resurrect it — so we only
        # alert. Stop-loss is disabled on adopted rows (no entry context).
        _STOP_DISABLED_SENTINEL = -1.0
        for sym, vp in venue_by_symbol.items():
            qidx = self.symbol_to_question.get(sym)
            if qidx is None:
                # No question mapping yet (e.g. meta event not ingested). We
                # cannot invent a question_idx — it's the position table's
                # primary key, and collisions would corrupt accounting. Emit
                # an unattributed drift event so the orphan is visible.
                drift.append(ReconcileDrift(
                    ts_ns=now_ns, account_alias=self.account_alias, case="position_mismatch", question_idx=0,
                    detail={"resolution": "venue_orphan_unattributed",
                            "symbol": sym, "qty": f"{vp.qty}"},
                ))
                continue
            if qidx in local_by_qidx:
                continue
            if apply:
                self.dal.upsert_position(Position(
                    question_idx=qidx, symbol=sym, qty=vp.qty,
                    avg_entry=vp.avg_entry, realized_pnl=0.0,
                    last_update_ts_ns=now_ns,
                    stop_loss_price=_STOP_DISABLED_SENTINEL,
                ))
            drift.append(ReconcileDrift(
                ts_ns=now_ns, account_alias=self.account_alias, case="position_mismatch", question_idx=qidx,
                detail={"resolution": "adopted_venue_orphan" if apply
                        else "venue_orphan_alert_only", "symbol": sym,
                        "qty": f"{vp.qty}", "avg_entry": f"{vp.avg_entry}"},
            ))

        return ReconcileResult(
            drift_events=drift,
            orphans_to_cancel=orphans,
            vanished_positions=vanished,
        )
