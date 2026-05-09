from __future__ import annotations

import time

from loguru import logger

from .event_bus import EventBus
from .hl_client import HLClient, PlaceRequest
from .risk import RiskGate, RiskInputs
from .risk_events import Entry, Exit, RiskVeto
from .state import OpenOrder, Position, StateDAL
from ..strategy.render import outcome_description, question_description
from ..strategy.types import Action, Decision, OrderIntent, QuestionView


class Router:
    """Decision → HL API call. Risk gate is the hard veto.

    Phase 1 single-shot: each intent becomes one place call. We do NOT manage
    queue position, partial fills, or order amends — IOC is fire-and-forget.
    """

    def __init__(
        self,
        *,
        dal: StateDAL,
        gate: RiskGate,
        bus: EventBus,
        hl: HLClient,
        strategy_id: str = "late_resolution",
    ) -> None:
        self.dal = dal
        self.gate = gate
        self.bus = bus
        self.hl = hl
        self.strategy_id = strategy_id

    async def handle(self, decision: Decision, *, inputs: RiskInputs, now_ns: int) -> None:
        if decision.action is Action.HOLD:
            return
        for intent in decision.intents:
            verdict = self.gate.check_pre_trade(intent, inputs)
            if not verdict.approved:
                await self.bus.publish(RiskVeto(
                    ts_ns=now_ns, reason=verdict.reason,
                    question_idx=intent.question_idx,
                    detail=verdict.detail or {},
                ))
                logger.info(
                    "risk veto cloid={} reason={} detail={}",
                    intent.cloid, verdict.reason, verdict.detail,
                )
                continue
            await self._place(intent, now_ns=now_ns, decision=decision, question=inputs.question)
        # Settlement-driven EXIT with no intents: book the close locally.
        if decision.action is Action.EXIT and not decision.intents:
            await self._close_settled(inputs.question.question_idx, now_ns=now_ns, question=inputs.question)

    async def _place(self, intent: OrderIntent, *, now_ns: int, decision: Decision, question: QuestionView | None = None) -> None:
        # 1. Persist pending row before the network call (spec §5.5 idempotency).
        self.dal.upsert_order(OpenOrder(
            cloid=intent.cloid, venue_oid=None, question_idx=intent.question_idx,
            symbol=intent.symbol, side=intent.side, price=intent.limit_price,
            size=intent.size, status="pending", placed_ts_ns=now_ns,
            last_update_ts_ns=now_ns, strategy_id=self.strategy_id,
        ))
        # 2. Send.
        ack = self.hl.place(PlaceRequest(
            cloid=intent.cloid, symbol=intent.symbol, side=intent.side,
            size=intent.size, price=intent.limit_price,
            reduce_only=intent.reduce_only, time_in_force=intent.time_in_force,
        ))
        # 3. Update DB row from ack.
        self.dal.update_order_status(intent.cloid, status=ack.status,
                                      venue_oid=ack.venue_oid, now_ns=now_ns)
        # 4. If filled, update Position + emit Entry/Exit.
        if ack.status == "filled" and ack.fill_size and ack.fill_price:
            await self._book_fill(intent, ack.fill_price, ack.fill_size, now_ns=now_ns, question=question)

    async def _book_fill(
        self, intent: OrderIntent, price: float, size: float, *, now_ns: int,
        question: QuestionView | None = None,
    ) -> None:
        q_desc = question_description(question) if question else ""
        o_desc = outcome_description(question, intent.symbol) if question else ""
        existing = self.dal.get_position(intent.question_idx)
        signed = size if intent.side == "buy" else -size
        if existing is None:
            self.dal.upsert_position(Position(
                question_idx=intent.question_idx, symbol=intent.symbol,
                qty=signed, avg_entry=price, realized_pnl=0.0,
                last_update_ts_ns=now_ns,
                # Stop-loss = entry * (1 - stop_loss_pct/100) for longs
                # (Plan 1C will look up the matched allowlist entry's stop_loss_pct.)
                stop_loss_price=price * 0.9,
            ))
            await self.bus.publish(Entry(
                ts_ns=now_ns, cloid=intent.cloid,
                question_idx=intent.question_idx, symbol=intent.symbol,
                side=intent.side, size=size, price=price,
                question_description=q_desc, outcome_description=o_desc,
            ))
        else:
            new_qty = existing.qty + signed
            if abs(new_qty) < 1e-9:
                # Closed
                realized = (price - existing.avg_entry) * existing.qty
                self.dal.delete_position(intent.question_idx)
                await self.bus.publish(Exit(
                    ts_ns=now_ns, question_idx=intent.question_idx,
                    symbol=intent.symbol, qty=existing.qty,
                    realized_pnl=realized + existing.realized_pnl,
                    reason="stop_loss" if intent.reduce_only else "manual",
                    question_description=q_desc, outcome_description=o_desc,
                ))
            else:
                avg = (existing.qty * existing.avg_entry + signed * price) / new_qty if new_qty else 0.0
                self.dal.upsert_position(Position(
                    question_idx=intent.question_idx, symbol=intent.symbol,
                    qty=new_qty, avg_entry=avg, realized_pnl=existing.realized_pnl,
                    last_update_ts_ns=now_ns, stop_loss_price=existing.stop_loss_price,
                ))

    async def _close_settled(self, question_idx: int, *, now_ns: int, question: QuestionView | None = None) -> None:
        p = self.dal.get_position(question_idx)
        if p is None:
            return
        # Settlement: caller has already confirmed the question settled. We
        # simply delete the local position; venue truth handles PnL.
        self.dal.delete_position(question_idx)
        q_desc = question_description(question) if question else ""
        o_desc = outcome_description(question, p.symbol) if question else ""
        await self.bus.publish(Exit(
            ts_ns=now_ns, question_idx=question_idx, symbol=p.symbol,
            qty=p.qty, realized_pnl=p.realized_pnl, reason="settlement",
            question_description=q_desc, outcome_description=o_desc,
        ))
