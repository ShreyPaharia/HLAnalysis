from __future__ import annotations

import time

from loguru import logger

from .config import StrategyConfig, match_question
from .event_bus import EventBus
from .hl_client import HLClient, PlaceRequest
from .risk import RiskGate, RiskInputs
from .risk_events import Entry, Exit, RiskVeto
from .state import OpenOrder, Position, StateDAL
from ..strategy.render import outcome_description, question_description
from ..strategy.types import Action, Decision, OrderIntent, QuestionView


# Mirrors hlanalysis.backtest.runner.hftbt_runner._STOP_DISABLED_SENTINEL — a
# negative price the risk gate's stop-loss check (px <= stop_loss_price for
# longs) can never trigger on, since real bid prices are ≥ 0.
_STOP_DISABLED_SENTINEL = -1.0


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
        strategy_cfg: StrategyConfig,
        strategy_id: str = "late_resolution",
    ) -> None:
        self.dal = dal
        self.gate = gate
        self.bus = bus
        self.hl = hl
        self.strategy_cfg = strategy_cfg
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
            await self._place(intent, now_ns=now_ns, decision=decision,
                              question=inputs.question,
                              question_fields=inputs.question_fields)
        # Settlement-driven EXIT with no intents: book the close locally.
        if decision.action is Action.EXIT and not decision.intents:
            await self._close_settled(inputs.question.question_idx, now_ns=now_ns, question=inputs.question)

    async def _place(
        self, intent: OrderIntent, *, now_ns: int, decision: Decision,
        question: QuestionView | None = None,
        question_fields: dict[str, str] | None = None,
    ) -> None:
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
        # Surface venue-side rejections. The ack.error is dropped on the floor
        # by upsert_order_status (no error column on the openorder table), so
        # without this log the rejection reason is invisible — symptoms are
        # "orders placed silently never fill", which is what bit us live with
        # the HYPE-short eating all margin and HIP-4 buys getting rejected.
        if ack.status == "rejected":
            logger.warning(
                "order rejected cloid={} symbol={} side={} size={} price={} err={}",
                intent.cloid, intent.symbol, intent.side, intent.size,
                intent.limit_price, ack.error or "<no_error_field>",
            )
        # 4. If filled, update Position + emit Entry/Exit.
        # Use `is not None` rather than truthy: a falsy 0.0 is a malformed ack
        # that should be logged as a problem, not silently treated as 'no fill'
        # which would create an unmanaged live position (DB filled, no Position).
        if ack.status == "filled":
            if ack.fill_size is None or ack.fill_price is None or ack.fill_size <= 0 or ack.fill_price <= 0:
                logger.warning(
                    "filled ack with missing/zero size or price; cloid={} size={} price={}",
                    intent.cloid, ack.fill_size, ack.fill_price,
                )
                return
            await self._book_fill(intent, ack.fill_price, ack.fill_size,
                                  now_ns=now_ns, question=question,
                                  question_fields=question_fields)

    def _stop_loss_price_for(
        self, fill_price: float, question_idx: int,
        question_fields: dict[str, str] | None,
    ) -> float:
        """Resolve the per-trade stop loss from the matched allowlist entry.

        Returns the price at which the risk gate should trigger a stop-loss
        exit, or _STOP_DISABLED_SENTINEL when the entry's `stop_loss_pct` is
        None (or no entry matches — defaults catch this in practice).
        """
        matched = None
        if question_fields:
            matched = match_question(
                self.strategy_cfg,
                question_idx=question_idx,
                fields=question_fields,
            )
        entry = matched or self.strategy_cfg.defaults
        pct = entry.stop_loss_pct
        if pct is None:
            return _STOP_DISABLED_SENTINEL
        return max(0.0, fill_price * (1.0 - pct / 100.0))

    async def _book_fill(
        self, intent: OrderIntent, price: float, size: float, *, now_ns: int,
        question: QuestionView | None = None,
        question_fields: dict[str, str] | None = None,
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
                stop_loss_price=self._stop_loss_price_for(
                    price, intent.question_idx, question_fields,
                ),
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
