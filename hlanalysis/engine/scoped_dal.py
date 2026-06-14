"""Per-strategy view over a shared StateDAL / CachedStateDAL.

A single state.db can now host rows for multiple strategy slots (the
0006_unified_slot_db migration adds strategy_id + account scoping columns
to every table). StrategyScopedDAL wraps a shared StateDAL and auto-injects
strategy_id + account into every read/write so a slot only ever touches its
own rows, without each slot needing to pass those kwargs explicitly at every
call site.

Usage::

    shared_dal = CachedStateDAL(db_path)
    shared_dal.run_migrations()

    slot_a = StrategyScopedDAL(shared_dal, strategy_id="v31", account="0xABCD")
    slot_b = StrategyScopedDAL(shared_dal, strategy_id="v1",  account="0xABCD")

    slot_a.upsert_position(p)      # stamps strategy_id="v31", account="0xABCD"
    slot_b.get_position(qidx)      # only sees v1 rows

The scoped DAL exposes every method that the engine touches (router, scanner,
reconciler, events_sink, trade_journal, diag, reconcile_report). Methods that
are genuinely global — run_migrations, applied_versions, prune_events,
prune_trade_journal, start_session, end_session, mirror_venue_fills — are
delegated straight through with no scope filter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .state import Fill, OpenOrder, Position, StateDAL, TradeJournalRow


class StrategyScopedDAL:
    """A per-strategy view over a shared StateDAL / CachedStateDAL.

    Auto-injects ``strategy_id`` and ``account`` into every read/write so a
    slot only ever touches its own rows in the unified state DB.

    Genuinely global operations (migrations, prune, session lifecycle,
    mirror_venue_fills) delegate straight through — they operate across the
    whole DB and must not be narrowed to one strategy's rows.

    ``db_path`` is a **virtual** per-strategy path whose *parent* is the
    per-strategy slot directory (``<shared_db.parent>/<strategy_id>/``).
    This lets existing code that computes sibling-file paths via
    ``Path(dal.db_path).parent / "<file>"`` (e.g. Router._cooldown_path)
    resolve those files into the per-strategy slot directory rather than the
    shared-DB root, preserving per-slot isolation for operational files.
    """

    def __init__(self, base: StateDAL, *, strategy_id: str, account: str) -> None:
        self._base = base
        self.strategy_id = strategy_id
        self.account = account
        # Virtual per-strategy path: <shared_db_parent>/<strategy_id>/<db_name>.
        # Parent = slot_dir_for(strategy_id) so sibling-file lookups
        # (exit_cooldowns.json, gate_decisions.jsonl) resolve into the slot dir.
        self.db_path: Path = base.db_path.parent / strategy_id / base.db_path.name

    # ------------------------------------------------------------------ #
    # Global (pass-through) operations
    # ------------------------------------------------------------------ #

    def run_migrations(self) -> None:
        self._base.run_migrations()

    def applied_versions(self) -> set[str]:
        return self._base.applied_versions()

    def start_session(self, session_id: str, *, now_ns: int) -> None:
        self._base.start_session(session_id, now_ns=now_ns)

    def end_session(self, session_id: str, *, now_ns: int, halt_reason: str | None = None) -> None:
        self._base.end_session(session_id, now_ns=now_ns, halt_reason=halt_reason)

    def prune_events(self, *, max_age_ns: int, max_rows: int) -> None:
        # global across the DB — not scoped by strategy to keep table bounded
        self._base.prune_events(max_age_ns=max_age_ns, max_rows=max_rows)

    def prune_trade_journal(self, *, max_age_ns: int, max_rows: int) -> None:
        # global across the DB — not scoped by strategy to keep table bounded
        self._base.prune_trade_journal(max_age_ns=max_age_ns, max_rows=max_rows)

    def mirror_venue_fills(
        self,
        fills: Any,
        *,
        symbol_to_question: dict[str, int] | None = None,
    ) -> int:
        # Venue-mirror runs cross-slot (HL user_fills are per-account, not
        # per-strategy). Delegate globally; the reconcile loop already filters
        # by symbol.
        return self._base.mirror_venue_fills(fills, symbol_to_question=symbol_to_question)

    # ------------------------------------------------------------------ #
    # Orders
    # ------------------------------------------------------------------ #

    def upsert_order(self, o: OpenOrder) -> None:
        self._base.upsert_order(o, strategy_id=self.strategy_id, account=self.account)

    def get_order(self, cloid: str) -> OpenOrder | None:
        return self._base.get_order(cloid, strategy_id=self.strategy_id)

    def update_order_status(
        self,
        cloid: str,
        *,
        status: str,
        venue_oid: str | None = None,
        now_ns: int,
    ) -> None:
        self._base.update_order_status(
            cloid,
            status=status,
            venue_oid=venue_oid,
            now_ns=now_ns,
            strategy_id=self.strategy_id,
        )

    def live_orders(self) -> list[OpenOrder]:
        return self._base.live_orders(strategy_id=self.strategy_id)

    # ------------------------------------------------------------------ #
    # Positions
    # ------------------------------------------------------------------ #

    def upsert_position(self, p: Position) -> None:
        self._base.upsert_position(p, strategy_id=self.strategy_id, account=self.account)

    def get_position(self, question_idx: int) -> Position | None:
        return self._base.get_position(question_idx, strategy_id=self.strategy_id)

    def all_positions(self) -> list[Position]:
        return self._base.all_positions(strategy_id=self.strategy_id)

    def delete_position(self, question_idx: int) -> None:
        self._base.delete_position(question_idx, strategy_id=self.strategy_id)

    # ------------------------------------------------------------------ #
    # Seen-questions
    # ------------------------------------------------------------------ #

    def has_seen_question(self, question_idx: int) -> bool:
        return self._base.has_seen_question(question_idx, strategy_id=self.strategy_id)

    def mark_question_seen(self, question_idx: int, *, now_ns: int) -> None:
        self._base.mark_question_seen(question_idx, now_ns=now_ns, strategy_id=self.strategy_id)

    # ------------------------------------------------------------------ #
    # PM open-strikes
    # ------------------------------------------------------------------ #

    def get_pm_strike(self, question_idx: int) -> float | None:
        return self._base.get_pm_strike(question_idx, strategy_id=self.strategy_id)

    def set_pm_strike(self, question_idx: int, strike: float) -> None:
        self._base.set_pm_strike(question_idx, strike, strategy_id=self.strategy_id)

    # ------------------------------------------------------------------ #
    # Coin → market-class map (SHR-77)
    # ------------------------------------------------------------------ #

    def set_coin_klass(self, *, coin: str, klass: str, question_idx: int) -> None:
        self._base.set_coin_klass(coin=coin, klass=klass, question_idx=question_idx, strategy_id=self.strategy_id)

    def coin_klass_map(self) -> dict[str, str]:
        return self._base.coin_klass_map(strategy_id=self.strategy_id)

    # ------------------------------------------------------------------ #
    # Settlements
    # ------------------------------------------------------------------ #

    def record_settlement(
        self,
        *,
        question_idx: int,
        symbol: str,
        realized_pnl: float,
        ts_ns: int,
    ) -> None:
        self._base.record_settlement(
            question_idx=question_idx,
            symbol=symbol,
            realized_pnl=realized_pnl,
            ts_ns=ts_ns,
            strategy_id=self.strategy_id,
            account=self.account,
        )

    def settlement_pnl_since(self, since_ts_ns: int) -> float:
        return self._base.settlement_pnl_since(since_ts_ns, strategy_id=self.strategy_id)

    # ------------------------------------------------------------------ #
    # Fills
    # ------------------------------------------------------------------ #

    def append_fill(self, f: Fill) -> bool:
        return self._base.append_fill(f, strategy_id=self.strategy_id, account=self.account)

    def fills_count(self) -> int:
        return self._base.fills_count(strategy_id=self.strategy_id)

    def fills_for_cloid(self, cloid: str) -> list[Fill]:
        return self._base.fills_for_cloid(cloid, strategy_id=self.strategy_id)

    # ------------------------------------------------------------------ #
    # Realized PnL
    # ------------------------------------------------------------------ #

    def realized_pnl_since(self, since_ts_ns: int) -> float:
        return self._base.realized_pnl_since(since_ts_ns, strategy_id=self.strategy_id)

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #

    def append_event(
        self,
        *,
        ts_ns: int,
        alias: str | None,
        kind: str,
        question_idx: int | None,
        reason: str | None,
        payload_json: str | None,
    ) -> None:
        self._base.append_event(
            ts_ns=ts_ns,
            alias=alias,
            kind=kind,
            question_idx=question_idx,
            reason=reason,
            payload_json=payload_json,
            strategy_id=self.strategy_id,
        )

    def events_since(self, since_ts_ns: int) -> list[dict[str, Any]]:
        return self._base.events_since(since_ts_ns, strategy_id=self.strategy_id)

    def reject_counts_since(self, since_ts_ns: int) -> list[dict[str, Any]]:
        return self._base.reject_counts_since(since_ts_ns, strategy_id=self.strategy_id)

    def events_for_question(self, question_idx: int) -> list[dict[str, Any]]:
        return self._base.events_for_question(question_idx, strategy_id=self.strategy_id)

    def last_event_by_kind(self, kind: str, *, alias: str | None = None) -> dict[str, Any] | None:
        return self._base.last_event_by_kind(kind, alias=alias, strategy_id=self.strategy_id)

    # ------------------------------------------------------------------ #
    # Trade journal (SHR-83)
    # ------------------------------------------------------------------ #

    def add_journal_decision(self, row: TradeJournalRow) -> bool:
        return self._base.add_journal_decision(row, strategy_id=self.strategy_id, account=self.account)

    def update_journal(self, cloid: str, **changes: Any) -> None:
        self._base.update_journal(cloid, strategy_id=self.strategy_id, **changes)

    def get_journal_row(self, cloid: str) -> TradeJournalRow | None:
        return self._base.get_journal_row(cloid, strategy_id=self.strategy_id)

    def delete_journal_decision(self, cloid: str) -> None:
        self._base.delete_journal_decision(cloid, strategy_id=self.strategy_id)
