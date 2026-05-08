from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger

from .hl_client import ClearinghouseState, OpenOrderRow, UserFillRow
from .risk_events import ReconcileDrift
from .state import Fill, OpenOrder, Position, StateDAL


CLOID_PREFIX = "hla-"


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    drift_events: list[ReconcileDrift]
    orphans_to_cancel: list[str]   # cloids the caller should defensively cancel


class Reconciler:
    """Three-way merge between local DB, HL openOrders, HL clearinghouseState.

    Spec §5.2 + §5.5. Pure synchronous logic — no IO. The caller wires in:
      - `fills_lookup(cloid)` → list[UserFillRow] for resolving local-ghost cases
      - `symbol_to_question` → optional mapping from venue symbol to question_idx
        for position-row attribution.
    """

    def __init__(
        self,
        dal: StateDAL,
        *,
        fills_lookup: Callable[[str], list[UserFillRow]],
        symbol_to_question: dict[str, int] | None = None,
    ) -> None:
        self.dal = dal
        self.fills_lookup = fills_lookup
        self.symbol_to_question = symbol_to_question or {}

    def run(
        self,
        *,
        venue_open: list[OpenOrderRow],
        venue_state: ClearinghouseState,
        now_ns: int,
    ) -> ReconcileResult:
        drift: list[ReconcileDrift] = []
        orphans: list[str] = []

        # --- orders ---
        venue_by_cloid = {r.cloid: r for r in venue_open if r.cloid.startswith(CLOID_PREFIX)}
        local_live = {o.cloid: o for o in self.dal.live_orders()}

        # local-ghost: in DB live, not on venue
        for cloid, db_o in local_live.items():
            if cloid in venue_by_cloid:
                continue
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
                    ts_ns=now_ns, case="state_mismatch", cloid=cloid,
                    detail={"resolution": "filled_via_user_fills"},
                ))
            else:
                self.dal.update_order_status(cloid, status="cancelled", now_ns=now_ns)
                drift.append(ReconcileDrift(
                    ts_ns=now_ns, case="local_ghost", cloid=cloid,
                ))

        # venue-orphan: on venue with our prefix, not in DB live
        for cloid, vo in venue_by_cloid.items():
            if cloid in local_live:
                continue
            orphans.append(cloid)
            drift.append(ReconcileDrift(
                ts_ns=now_ns, case="venue_orphan", cloid=cloid,
                detail={"symbol": vo.symbol},
            ))

        # state-mismatch: both, fields differ — HL wins
        for cloid, vo in venue_by_cloid.items():
            db_o = local_live.get(cloid)
            if db_o is None:
                continue
            if (
                abs(db_o.price - vo.price) > 1e-9
                or abs(db_o.size - vo.size) > 1e-9
                or db_o.symbol != vo.symbol
                or db_o.venue_oid != vo.venue_oid
            ):
                self.dal.upsert_order(OpenOrder(
                    cloid=cloid, venue_oid=vo.venue_oid, question_idx=db_o.question_idx,
                    symbol=vo.symbol, side=db_o.side, price=vo.price, size=vo.size,
                    status="open", placed_ts_ns=db_o.placed_ts_ns,
                    last_update_ts_ns=now_ns, strategy_id=db_o.strategy_id,
                ))
                drift.append(ReconcileDrift(
                    ts_ns=now_ns, case="state_mismatch", cloid=cloid,
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
                    ts_ns=now_ns, case="position_mismatch", question_idx=qidx,
                    detail={"resolution": "deleted_local_position_not_on_venue"},
                ))
                continue
            if abs(vp.qty - lp.qty) > 1e-9 or abs(vp.avg_entry - lp.avg_entry) > 1e-9:
                self.dal.upsert_position(Position(
                    question_idx=qidx, symbol=lp.symbol, qty=vp.qty,
                    avg_entry=vp.avg_entry, realized_pnl=lp.realized_pnl,
                    last_update_ts_ns=now_ns, stop_loss_price=lp.stop_loss_price,
                ))
                drift.append(ReconcileDrift(
                    ts_ns=now_ns, case="position_mismatch", question_idx=qidx,
                    detail={"hl_qty": f"{vp.qty}", "db_qty": f"{lp.qty}"},
                ))

        # Position on venue we don't track locally. Attribution uses
        # symbol_to_question; if we can't attribute, we still alert.
        for sym, vp in venue_by_symbol.items():
            qidx = self.symbol_to_question.get(sym)
            if qidx is None:
                continue
            if qidx in local_by_qidx:
                continue
            drift.append(ReconcileDrift(
                ts_ns=now_ns, case="position_mismatch", question_idx=qidx,
                detail={"resolution": "venue_orphan_position", "symbol": sym},
            ))

        return ReconcileResult(drift_events=drift, orphans_to_cancel=orphans)
