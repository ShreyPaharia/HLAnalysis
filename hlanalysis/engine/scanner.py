from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import StrategyConfig, match_question
from .market_state import MarketState
from .risk import RiskInputs
from .state import StateDAL
from ..strategy.base import Strategy
from ..strategy.types import (
    Action, BookState, Decision, Position, QuestionView,
)


@dataclass(frozen=True, slots=True)
class ScannedDecision:
    decision: Decision
    inputs: RiskInputs


class Scanner:
    """Per-tick scanner. Walks all known questions, runs allowlist match,
    builds RiskInputs, calls strategy.evaluate(), returns the decisions.
    The runtime owns the timer; this class is pure synchronous logic.
    """

    def __init__(
        self,
        *,
        strategy: Strategy,
        cfg: StrategyConfig,
        market_state: MarketState,
        dal: StateDAL,
        kill_switch_path: Path,
        last_reconcile_ns: int,
        reference_symbol: str = "BTC",
    ) -> None:
        self.strategy = strategy
        self.cfg = cfg
        self.ms = market_state
        self.dal = dal
        self.kill_switch_path = kill_switch_path
        self.last_reconcile_ns = last_reconcile_ns
        self.ref_symbol = reference_symbol

    def scan(self, *, now_ns: int) -> list[ScannedDecision]:
        out: list[ScannedDecision] = []
        ref = self.ms.last_mark(self.ref_symbol)
        if ref is None:
            return out
        positions_db = self.dal.all_positions()
        positions_by_q = {p.question_idx: p for p in positions_db}
        all_positions_strategy = [
            self._db_pos_to_strategy(p) for p in positions_db
        ]
        live_orders = self.dal.live_orders()
        live_notional = sum(o.price * o.size for o in live_orders)
        # Daily loss: realized PnL since UTC midnight of `now_ns`
        midnight_ns = self._utc_midnight_ns(now_ns)
        realized_today = self.dal.realized_pnl_since(midnight_ns)
        kill = self.kill_switch_path.exists()

        for q in self.ms.all_questions():
            fields = {"class": q.klass, "underlying": q.underlying, "period": q.period}
            if match_question(self.cfg, question_idx=q.question_idx, fields=fields) is None:
                continue
            yes_book = self.ms.book(q.yes_symbol)
            no_book = self.ms.book(q.no_symbol)
            if yes_book is None and no_book is None:
                continue
            books: dict[str, BookState] = {}
            if yes_book is not None:
                books[q.yes_symbol] = yes_book
            if no_book is not None:
                books[q.no_symbol] = no_book

            db_pos = positions_by_q.get(q.question_idx)
            strat_pos = self._db_pos_to_strategy(db_pos) if db_pos else None
            decision = self.strategy.evaluate(
                question=q,
                books=books,
                reference_price=ref,
                recent_returns=self.ms.recent_returns(self.ref_symbol, n=32),
                recent_volume_usd=(
                    self.ms.recent_volume_usd(q.yes_symbol, now=now_ns)
                    + self.ms.recent_volume_usd(q.no_symbol, now=now_ns)
                ),
                position=strat_pos,
                now_ns=now_ns,
            )
            if decision.action is Action.HOLD:
                continue
            for intent in decision.intents:
                target = books.get(intent.symbol)
                if target is None:
                    continue
                inputs = RiskInputs(
                    question=q,
                    question_fields=fields,
                    reference_price=ref,
                    book=target,
                    recent_volume_usd=(
                        self.ms.recent_volume_usd(q.yes_symbol, now=now_ns)
                        + self.ms.recent_volume_usd(q.no_symbol, now=now_ns)
                    ),
                    positions=all_positions_strategy,
                    live_orders_total_notional=live_notional,
                    realized_pnl_today=realized_today,
                    kill_switch_active=kill,
                    last_reconcile_ns=self.last_reconcile_ns,
                    now_ns=now_ns,
                )
                out.append(ScannedDecision(
                    decision=Decision(action=decision.action, intents=(intent,),
                                       diagnostics=decision.diagnostics),
                    inputs=inputs,
                ))
            # EXIT with no intents (settlement) — pass through with a synthetic input
            if decision.action is Action.EXIT and not decision.intents:
                # Build a stub inputs (book is required; pick any leg we have)
                stub_book = next(iter(books.values()))
                inputs = RiskInputs(
                    question=q, question_fields=fields, reference_price=ref,
                    book=stub_book, recent_volume_usd=0.0,
                    positions=all_positions_strategy,
                    live_orders_total_notional=live_notional,
                    realized_pnl_today=realized_today, kill_switch_active=kill,
                    last_reconcile_ns=self.last_reconcile_ns, now_ns=now_ns,
                )
                out.append(ScannedDecision(decision=decision, inputs=inputs))
        return out

    @staticmethod
    def _db_pos_to_strategy(p: object | None) -> Position | None:
        if p is None:
            return None
        return Position(
            question_idx=p.question_idx, symbol=p.symbol, qty=p.qty,
            avg_entry=p.avg_entry, stop_loss_price=p.stop_loss_price,
            last_update_ts_ns=p.last_update_ts_ns,
        )

    @staticmethod
    def _utc_midnight_ns(now_ns: int) -> int:
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(now_ns / 1e9, tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return int(dt.timestamp() * 1_000_000_000)
