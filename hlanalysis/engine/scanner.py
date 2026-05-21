from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

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
        pnl_provider: Callable[[int], float] | None = None,
        gate_log_path: Path | None = None,
    ) -> None:
        self.strategy = strategy
        self.cfg = cfg
        self.ms = market_state
        self.dal = dal
        self.kill_switch_path = kill_switch_path
        self.last_reconcile_ns = last_reconcile_ns
        self.ref_symbol = reference_symbol
        # Source of truth for today's realized PnL. When None, falls back to
        # the local DAL — used in tests / paper-mode where there's no HL
        # connection. Live runtime wires this to HLClient.realized_pnl_since
        # so the daily-loss gate sees venue truth rather than the local DB
        # (which loses information on every position close).
        self._pnl_provider = pnl_provider
        # Structured gate-decision log. When set, the scanner writes a JSONL
        # line every time a (question_idx, decision_action, primary_diag_name)
        # tuple changes from the previous tick — i.e. on STATE TRANSITIONS,
        # not every scan. This gives operators a forensic trail of "which
        # gate was blocking each candidate" without flooding the disk at
        # the 1Hz scan rate. Used for the 48h forward test after introducing
        # the bid-gate / cooldown / near-strike filters.
        self.gate_log_path = gate_log_path
        self._last_logged_state: dict[int, tuple[str, str]] = {}

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
        # Daily loss: realized PnL since the configured daily window start
        # (default UTC midnight, set to 06:00 UTC to align with HL HIP-4
        # binary settlement). Prefer the injected provider (HL truth) over
        # the local DB; see _pnl_provider doc.
        midnight_ns = self._daily_window_start_ns(
            now_ns, hour=self.cfg.global_.daily_window_start_hour_utc,
        )
        if self._pnl_provider is not None:
            try:
                realized_today = self._pnl_provider(midnight_ns)
            except Exception:
                # If HL is unreachable, fall back to local DB rather than
                # halting the scan. The DAL value is a structural under-estimate
                # of losses today (see state.py), so this is the safer side to
                # err on: continue trading rather than freeze on a network blip.
                realized_today = self.dal.realized_pnl_since(midnight_ns)
        else:
            realized_today = self.dal.realized_pnl_since(midnight_ns)
        kill = self.kill_switch_path.exists()

        for q in self.ms.all_questions():
            fields = {"class": q.klass, "underlying": q.underlying, "period": q.period}
            if match_question(self.cfg, question_idx=q.question_idx, fields=fields) is None:
                continue
            # Multi-outcome support: feed every leg of the question to the
            # strategy so it can decide across all sides (priceBucket has 6 legs;
            # priceBinary has 2). Skip the question if no leg has a book yet.
            leg_syms = q.leg_symbols or (
                (q.yes_symbol, q.no_symbol) if q.yes_symbol else ()
            )
            books: dict[str, BookState] = {}
            for sym in leg_syms:
                b = self.ms.book(sym)
                if b is not None:
                    books[sym] = b
            if not books:
                continue

            db_pos = positions_by_q.get(q.question_idx)
            strat_pos = self._db_pos_to_strategy(db_pos) if db_pos else None
            volume_total = sum(
                self.ms.recent_volume_usd(sym, now=now_ns) for sym in leg_syms
            )
            decision = self.strategy.evaluate(
                question=q,
                books=books,
                reference_price=ref,
                recent_returns=self.ms.recent_returns(self.ref_symbol, n=32),
                recent_volume_usd=volume_total,
                position=strat_pos,
                now_ns=now_ns,
            )
            self._maybe_log_gate_transition(
                question=q, decision=decision, books=books, now_ns=now_ns,
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
                    recent_volume_usd=volume_total,
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

    def _maybe_log_gate_transition(
        self, *, question: QuestionView, decision: Decision,
        books: dict[str, BookState], now_ns: int,
    ) -> None:
        """Append a JSONL line when the (action, primary_diag) tuple changes
        for this question_idx vs the last scan tick.

        Steady-state ticks (same blocking reason for hours) produce nothing.
        Transitions (entered/exited/changed gate) produce one line. Disk-cheap
        and forensically useful — operators can answer "why didn't we enter
        on #601 at 04:47?" by tailing this file.
        """
        if self.gate_log_path is None:
            return
        # Reduce diagnostics to a stable identifier. Strategies emit one or
        # more Diagnostic objects per decision; the first one is the
        # human-readable label of why this scan tick ended where it did.
        diag_name = (
            decision.diagnostics[0].message if decision.diagnostics else ""
        )
        key = (decision.action.value, diag_name)
        prev = self._last_logged_state.get(question.question_idx)
        if prev == key:
            return
        self._last_logged_state[question.question_idx] = key
        # Snapshot book state for the leg the strategy actually chose. For
        # binary questions there's only one favorite path so this is moot,
        # but for buckets the strategy picks among N YES legs and emits
        # `chosen_leg=<sym>` in the diagnostic. Falling back to first-leg
        # for older diagnostics or missing books keeps legacy rows readable.
        chosen_sym: str | None = None
        for d in decision.diagnostics:
            for k, v in d.fields:
                if k == "chosen_leg":
                    chosen_sym = v
                    break
            if chosen_sym is not None:
                break
        sample_book: BookState | None = (
            books.get(chosen_sym) if chosen_sym else None
        ) or next(iter(books.values()), None)
        bid = sample_book.bid_px if sample_book else None
        ask = sample_book.ask_px if sample_book else None
        mid = (bid + ask) / 2.0 if (bid is not None and ask is not None) else None
        row = {
            "ts_ns": now_ns,
            "question_idx": question.question_idx,
            "klass": question.klass,
            "action": decision.action.value,
            "reason": diag_name,
            "bid_px": bid,
            "ask_px": ask,
            "mid_px": mid,
            "bid_sz": sample_book.bid_sz if sample_book else None,
            "ask_sz": sample_book.ask_sz if sample_book else None,
            "strike": question.strike if question.strike == question.strike else None,
            "diag_fields": [
                {"k": k, "v": v}
                for d in decision.diagnostics
                for (k, v) in d.fields
            ] or None,
        }
        try:
            self.gate_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.gate_log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError as e:
            # Best-effort logging — never block trading on a filesystem hiccup.
            logger.warning("gate log write failed path={} err={}", self.gate_log_path, e)

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
        """Backwards-compatible alias used by older callers / tests; equivalent
        to `_daily_window_start_ns(now_ns, hour=0)`. New callers should pass an
        explicit hour from GlobalRiskConfig.daily_window_start_hour_utc."""
        return Scanner._daily_window_start_ns(now_ns, hour=0)

    @staticmethod
    def _daily_window_start_ns(now_ns: int, *, hour: int) -> int:
        """Most-recent timestamp at HH:00:00 UTC at-or-before `now_ns`.

        If `now_ns` is already past today's HH:00 UTC, returns today's HH:00.
        Otherwise rolls back to yesterday's HH:00. This is the cutoff the
        daily-loss gate uses to query HL fills with — it must be a stable
        boundary so the same fill never appears in two consecutive windows.
        """
        from datetime import datetime, timezone, timedelta
        dt = datetime.fromtimestamp(now_ns / 1e9, tz=timezone.utc)
        boundary = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
        if dt < boundary:
            boundary = boundary - timedelta(days=1)
        return int(boundary.timestamp() * 1_000_000_000)
