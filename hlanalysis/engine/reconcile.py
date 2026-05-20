from __future__ import annotations

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


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    drift_events: list[ReconcileDrift]
    # (cloid, symbol) pairs the caller should defensively cancel. Symbol must be
    # non-empty — HL's cancelByCloid rejects empty coin and the SDK silently
    # reports outer status='ok' even when the per-cancel statuses[] failed.
    orphans_to_cancel: list[tuple[str, str]]


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
    ) -> None:
        self.dal = dal
        self.fills_lookup = fills_lookup
        self.symbol_to_question = symbol_to_question or {}
        self.cloid_prefix = cloid_prefix
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
        venue_by_symbol = {p.symbol: p for p in venue_state.positions}
        local_by_qidx = {p.question_idx: p for p in self.dal.all_positions()}

        for qidx, lp in local_by_qidx.items():
            vp = venue_by_symbol.get(lp.symbol)
            if vp is None:
                # Position vanished from venue. Likely settled/closed during outage.
                self.dal.delete_position(qidx)
                drift.append(ReconcileDrift(
                    ts_ns=now_ns, account_alias=self.account_alias, case="position_mismatch", question_idx=qidx,
                    detail={"resolution": "deleted_local_position_not_on_venue"},
                ))
                continue
            qty_diff = abs(vp.qty - lp.qty) > 1e-9
            avg_diff = abs(vp.avg_entry - lp.avg_entry) > 1e-9
            if qty_diff or avg_diff:
                # avg_entry is display-only (the daily-loss gate reads PnL from
                # HL directly), so we silently adopt HL's value without firing
                # an alert. qty is load-bearing — it gates risk caps and the
                # strategy's have_position branch — so any qty drift must
                # surface as a drift event.
                self.dal.upsert_position(Position(
                    question_idx=qidx, symbol=lp.symbol, qty=vp.qty,
                    avg_entry=vp.avg_entry, realized_pnl=lp.realized_pnl,
                    last_update_ts_ns=now_ns, stop_loss_price=lp.stop_loss_price,
                ))
                if qty_diff:
                    drift.append(ReconcileDrift(
                        ts_ns=now_ns, account_alias=self.account_alias, case="position_mismatch", question_idx=qidx,
                        detail={"hl_qty": f"{vp.qty}", "db_qty": f"{lp.qty}"},
                    ))

        # Position on venue we don't track locally — adopt it into the DB so
        # the risk gate's caps and the strategy's `have_position` HOLD branch
        # both see it. Without adoption the strategy re-fires entries each
        # scan tick and the venue rejects them for insufficient quote balance.
        # Stop-loss is disabled on adopted rows: we have no entry context to
        # compute it from, and the alternative (exit-at-bid panic) would dump
        # a position we may want to hold to settlement. Settlement-driven
        # exits still work via _close_settled.
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
            self.dal.upsert_position(Position(
                question_idx=qidx, symbol=sym, qty=vp.qty,
                avg_entry=vp.avg_entry, realized_pnl=0.0,
                last_update_ts_ns=now_ns,
                stop_loss_price=_STOP_DISABLED_SENTINEL,
            ))
            drift.append(ReconcileDrift(
                ts_ns=now_ns, account_alias=self.account_alias, case="position_mismatch", question_idx=qidx,
                detail={"resolution": "adopted_venue_orphan", "symbol": sym,
                        "qty": f"{vp.qty}", "avg_entry": f"{vp.avg_entry}"},
            ))

        return ReconcileResult(drift_events=drift, orphans_to_cancel=orphans)
