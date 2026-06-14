from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import delete, func
from sqlmodel import Field, SQLModel, create_engine, select
from sqlmodel import Session as _Session

# Fill provenance (SHR-74). See Fill.source / 0004_fill_source.
FILL_SOURCE_ROUTER = "router"
FILL_SOURCE_VENUE = "venue"


class OpenOrder(SQLModel, table=True):
    cloid: str = Field(primary_key=True)
    venue_oid: str | None = None
    question_idx: int
    symbol: str
    side: str  # "buy" | "sell"
    price: float
    size: float
    status: str  # "pending" | "open" | "partially_filled" | "filled" | "cancelled" | "rejected"
    placed_ts_ns: int
    last_update_ts_ns: int
    strategy_id: str
    # 0006_unified_slot_db: account scoping column (nullable — existing rows unscoped).
    account: str | None = None


class Position(SQLModel, table=True):
    # 0006_unified_slot_db: composite PK (strategy_id, question_idx) so the same
    # question can be held independently by different strategy slots in a shared DB.
    strategy_id: str = Field(default="", primary_key=True)
    question_idx: int = Field(primary_key=True)
    symbol: str
    qty: float
    avg_entry: float
    realized_pnl: float
    last_update_ts_ns: int
    stop_loss_price: float
    # Cumulative absolute quantity closed (reduced) over this position's life.
    # Persisted so a close that fills across multiple partial reduces can report
    # the TOTAL closed size alongside the cumulative realized_pnl on its Exit
    # event (see marketdata.position_math.PositionState.closed_qty). Default 0.0
    # so pre-migration rows and fresh opens start at zero.
    closed_qty: float = 0.0
    # Account scoping column (nullable — existing per-slot rows have no account yet).
    account: str | None = None


class Fill(SQLModel, table=True):
    fill_id: str = Field(primary_key=True)
    cloid: str
    question_idx: int
    symbol: str
    side: str  # "buy" | "sell"
    price: float
    size: float
    fee: float
    ts_ns: int
    # Locally computed realized PnL on this fill (0 on opens, signed on
    # reduces): (fill_price − position.avg_entry) × closed_qty. Populated by
    # Router._book_fill for post-hoc audit; the daily-loss gate reads HL's
    # closedPnl directly via HLClient.hl_realized_pnl_since.
    closed_pnl: float = 0.0
    # Provenance (SHR-74). 'router' = booked by Router._book_fill (cloid-keyed,
    # fee=0, locally-computed closed_pnl); 'venue' = mirrored from HL user_fills
    # by the reconcile loop (tid-keyed, venue closedPnl/fee). realized_pnl_since
    # prefers 'venue' rows when present so the HL local ledger == venue (HIP-4
    # settlement payouts only ever arrive as venue fills). See 0004_fill_source.
    source: str = FILL_SOURCE_ROUTER
    # 0006_unified_slot_db: scoping columns (nullable — PK stays fill_id).
    strategy_id: str | None = None
    account: str | None = None


class Session_(SQLModel, table=True):
    __tablename__ = "session"
    session_id: str = Field(primary_key=True)
    started_ts_ns: int
    ended_ts_ns: int | None = None
    # Finding #47: halt_reason column exists in the DB DDL (0001_baseline) but
    # was absent from the ORM model, causing end_session() writes to be silently
    # dropped (a transient Python attribute, never SQL-bound). Adding it here
    # makes the field round-trip through the ORM. No migration is needed because
    # the column already exists in the schema.
    halt_reason: str | None = None


class SeenQuestion(SQLModel, table=True):
    """Tracks question_idxs the engine has notified about. Persists across
    restarts so NewQuestion alerts don't re-fire for already-known markets."""

    __tablename__ = "seen_question"
    # 0006_unified_slot_db: composite PK (strategy_id, question_idx).
    strategy_id: str = Field(default="", primary_key=True)
    question_idx: int = Field(primary_key=True)
    first_seen_ts_ns: int
    # Account scoping column (nullable — existing rows unscoped).
    account: str | None = None


class PmStrike(SQLModel, table=True):
    """Open-strike captured for a Polymarket up/down question. PM up/down
    markets have no static strike; the engine stamps it from the Binance SPOT
    1m candle close at strike_ref_ts_ns. Persisting it lets a restarted engine
    reuse the strike instead of skipping markets whose open it can no longer see."""

    __tablename__ = "pm_strike"
    # 0006_unified_slot_db: composite PK (strategy_id, question_idx).
    strategy_id: str = Field(default="", primary_key=True)
    question_idx: int = Field(primary_key=True)
    strike: float
    # Account scoping column (nullable — existing rows unscoped).
    account: str | None = None


class CoinKlass(SQLModel, table=True):
    """Maps an HL HIP-4 outcome-share coin ("#N") to its market class
    ("priceBinary" | "priceBucket"), written at QuestionMetaEvent ingest (SHR-77).

    The venue fill feed (user_fills) reports each HIP-4 fill as coin "#N" with a
    closedPnl but NO market type, and the engine `events` table is pruned and
    keyed by a different symbol representation — so neither can reliably classify
    historical fills. This table is the durable, forward-correct join: the engine
    already derives the leg coins ``f"#{10*outcome_idx + side_idx}"`` and the
    class from the question's metadata, so we persist that pair keyed by the exact
    coin user_fills returns. HL-only (PM fills are binary by construction)."""

    __tablename__ = "coin_klass"
    # 0006_unified_slot_db: composite PK (strategy_id, coin).
    strategy_id: str = Field(default="", primary_key=True)
    coin: str = Field(primary_key=True)
    klass: str
    question_idx: int
    # Account scoping column (nullable — existing rows unscoped).
    account: str | None = None


class Settlement(SQLModel, table=True):
    """Persisted realized PnL of a settled position (SHR-53). HIP-4 binaries
    close via settlement payouts, not HL fills, so this PnL was previously only
    alerted, never stored — leaving the daily-loss gate blind to it. Keyed by
    (strategy_id, question_idx) so the two close paths can't double-book per slot
    (first writer wins)."""

    __tablename__ = "settlement"
    # 0006_unified_slot_db: composite PK (strategy_id, question_idx).
    strategy_id: str = Field(default="", primary_key=True)
    question_idx: int = Field(primary_key=True)
    symbol: str
    realized_pnl: float
    ts_ns: int
    # Account scoping column (nullable — existing rows unscoped).
    account: str | None = None


class TradeJournalRow(SQLModel, table=True):
    """Durable per-decision/order trade journal (SHR-83).

    ONE row per emitted order, keyed by the (account-stamped) cloid and
    progressively populated across the decision → send → fill → (reject)
    lifecycle. Captures the evaluate() inputs (book-at-decision top-N, σ,
    recent_returns summary, recent_volume_usd, reference price), the emitted
    Decision, the venue send/fill timestamps + fill px/sz, the reject/veto
    reason, and the slot halt-state at decision time. This single journal serves
    three Spec-3 consumers at once: decision-parity (sim vs live), execution
    latency calibration (decision_ts → send_ts → fill_ts deltas), and halt
    replay (halt_json).

    Written off the hot path via ``trade_journal.TradeJournal`` (best-effort —
    a write failure is swallowed and never blocks order submission). The
    nullable columns are populated by later lifecycle hooks; the JSON blobs
    follow the ``events.payload_json`` precedent (free-form, query via the
    explicit columns)."""

    __tablename__ = "trade_journal"
    cloid: str = Field(primary_key=True)
    question_idx: int
    decision_ts_ns: int
    action: str  # "enter" | "exit"
    # 0006_unified_slot_db: scoping columns (nullable — PK stays cloid).
    strategy_id: str | None = None
    account: str | None = None
    side: str | None = None  # "buy" | "sell"
    symbol: str | None = None
    intended_size: float | None = None
    intended_price: float | None = None
    reference_price: float | None = None
    recent_volume_usd: float | None = None
    sigma: float | None = None
    returns_summary_json: str | None = None
    book_json: str | None = None
    diagnostics_json: str | None = None
    halt_json: str | None = None
    # Lifecycle timestamps + outcome — populated after the decision row exists.
    send_ts_ns: int | None = None
    fill_ts_ns: int | None = None
    fill_px: float | None = None
    fill_sz: float | None = None
    reject_reason: str | None = None


class Event(SQLModel, table=True):
    """Append-only log of every BusEvent published by the engine (Component 2).

    Written by _events_persist_loop in runtime.py — one row per published event.
    Retention is bounded by both age (max_age_ns) and row count (max_rows) via
    StateDAL.prune_events; prune is called periodically from the persist loop,
    not on every insert. The unbounded-_questions→OOM scar (hl_live_eval_2026_05_31)
    is why the row ceiling is non-optional.
    """

    __tablename__ = "events"
    id: int | None = Field(default=None, primary_key=True)
    ts_ns: int
    alias: str | None = None
    kind: str
    question_idx: int | None = None
    reason: str | None = None
    payload_json: str | None = None
    # 0006_unified_slot_db: strategy_id scoping column, backfilled from alias on
    # existing rows. Nullable — alias kept for backward compat.
    strategy_id: str | None = None


# Public alias for tests / external users.
Session = Session_  # noqa: F811


# Alembic migration environment (programmatic — no CLI step at runtime).
_ALEMBIC_DIR = Path(__file__).parent / "migrations_alembic"
# The single revision capturing the full pre-Alembic schema. Existing DBs are
# stamped here; fresh DBs upgrade to it (and beyond) from scratch.
_BASELINE_REVISION = "0001_baseline"
# Presence of this table without `alembic_version` marks a pre-Alembic DB
# created by the old hand-written SQL runner (0001 always creates it first).
_LEGACY_SENTINEL_TABLE = "openorder"


class StateDAL:
    """Thin DAL over sqlite via sqlmodel, with schema managed by Alembic.

    Migrations live in ``migrations_alembic/`` and are applied programmatically
    by ``run_migrations`` (no CLI step). A pre-existing (pre-Alembic) DB is
    STAMPED at the baseline rather than re-run, so live data and on-disk schema
    are left untouched; a fresh DB runs every revision from scratch and ends at
    the identical schema (see tests/unit/test_state_alembic_migrations.py).
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # WAL + fsync for durability (spec §5.5)
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        self._engine = create_engine(f"sqlite:///{db_path}", echo=False)

    # ---- migrations ----

    def _alembic_config(self) -> Config:
        cfg = Config()
        cfg.set_main_option("script_location", str(_ALEMBIC_DIR))
        cfg.set_main_option("sqlalchemy.url", f"sqlite:///{self.db_path}")
        return cfg

    def _existing_tables(self) -> set[str]:
        with sqlite3.connect(self.db_path) as conn:
            return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    def run_migrations(self) -> None:
        """Bring the DB to the Alembic head. Three cases:

        * already Alembic-managed (`alembic_version` present) → upgrade to head;
        * pre-Alembic production DB (old SQL runner left app tables but no
          `alembic_version`) → STAMP the baseline (its schema already equals the
          baseline; re-running DDL would error on existing tables and risk data),
          then upgrade to head for any post-baseline revisions;
        * fresh DB → run every revision from scratch.
        """
        cfg = self._alembic_config()
        tables = self._existing_tables()
        if "alembic_version" in tables:
            command.upgrade(cfg, "head")
        elif _LEGACY_SENTINEL_TABLE in tables:
            command.stamp(cfg, _BASELINE_REVISION)
            command.upgrade(cfg, "head")
        else:
            command.upgrade(cfg, "head")

    def applied_versions(self) -> set[str]:
        """Current Alembic head revision(s) recorded in the DB (empty before
        any migration runs)."""
        with self._engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            return set(ctx.get_current_heads())

    # ---- orders ----

    def upsert_order(self, o: OpenOrder, *, strategy_id: str = "", account: str | None = None) -> None:
        # Stamp scoping columns if callers provide them (StrategyScopedDAL path).
        # Values already set on the object take precedence when the kwargs are
        # empty-string / None so un-scoped direct calls stay bit-identical.
        if strategy_id:
            o.strategy_id = strategy_id
        if account is not None:
            o.account = account
        # expire_on_commit=False so `o` stays readable after commit (the cached
        # subclass reads its fields post-write without a re-SELECT).
        with _Session(self._engine, expire_on_commit=False) as s:
            existing = s.get(OpenOrder, o.cloid)
            if existing is None:
                s.add(o)
            else:
                for k, v in o.model_dump().items():
                    setattr(existing, k, v)
                s.add(existing)
            s.commit()

    def get_order(self, cloid: str, *, strategy_id: str = "") -> OpenOrder | None:
        with _Session(self._engine) as s:
            row = s.get(OpenOrder, cloid)
            if row is None:
                return None
            if strategy_id and row.strategy_id != strategy_id:
                return None
            return row

    def update_order_status(
        self,
        cloid: str,
        *,
        status: str,
        venue_oid: str | None = None,
        now_ns: int,
        strategy_id: str = "",
    ) -> None:
        with _Session(self._engine) as s:
            o = s.get(OpenOrder, cloid)
            if o is None:
                return
            if strategy_id and o.strategy_id != strategy_id:
                return
            o.status = status  # type: ignore[assignment]
            if venue_oid is not None:
                o.venue_oid = venue_oid
            o.last_update_ts_ns = now_ns
            s.add(o)
            s.commit()

    def live_orders(self, *, strategy_id: str = "") -> list[OpenOrder]:
        with _Session(self._engine) as s:
            stmt = select(OpenOrder).where(OpenOrder.status.in_(("pending", "open", "partially_filled")))
            if strategy_id:
                stmt = stmt.where(OpenOrder.strategy_id == strategy_id)
            return list(s.exec(stmt).all())

    # ---- positions ----

    def upsert_position(self, p: Position, *, strategy_id: str = "", account: str | None = None) -> None:
        # Stamp scoping columns when callers provide them (StrategyScopedDAL path).
        if strategy_id:
            p.strategy_id = strategy_id
        if account is not None:
            p.account = account
        # expire_on_commit=False so `p` stays readable after commit (the cached
        # subclass reads its fields post-write without a re-SELECT).
        with _Session(self._engine, expire_on_commit=False) as s:
            existing = s.get(Position, (p.strategy_id, p.question_idx))
            if existing is None:
                s.add(p)
            else:
                for k, v in p.model_dump().items():
                    setattr(existing, k, v)
                s.add(existing)
            s.commit()

    def get_position(self, question_idx: int, *, strategy_id: str = "") -> Position | None:
        with _Session(self._engine) as s:
            return s.get(Position, (strategy_id, question_idx))

    def all_positions(self, *, strategy_id: str = "") -> list[Position]:
        with _Session(self._engine) as s:
            stmt = select(Position)
            if strategy_id:
                stmt = stmt.where(Position.strategy_id == strategy_id)
            return list(s.exec(stmt).all())

    def delete_position(self, question_idx: int, *, strategy_id: str = "") -> None:
        with _Session(self._engine) as s:
            p = s.get(Position, (strategy_id, question_idx))
            if p is not None:
                s.delete(p)
                s.commit()

    def has_seen_question(self, question_idx: int, *, strategy_id: str = "") -> bool:
        with _Session(self._engine) as s:
            return s.get(SeenQuestion, (strategy_id, question_idx)) is not None

    def mark_question_seen(self, question_idx: int, *, now_ns: int, strategy_id: str = "") -> None:
        with _Session(self._engine) as s:
            if s.get(SeenQuestion, (strategy_id, question_idx)) is None:
                s.add(SeenQuestion(strategy_id=strategy_id, question_idx=question_idx, first_seen_ts_ns=now_ns))
                s.commit()

    # ---- PM open-strikes ----

    def get_pm_strike(self, question_idx: int, *, strategy_id: str = "") -> float | None:
        with _Session(self._engine) as s:
            row = s.get(PmStrike, (strategy_id, question_idx))
            return None if row is None else row.strike

    def set_pm_strike(self, question_idx: int, strike: float, *, strategy_id: str = "") -> None:
        with _Session(self._engine) as s:
            existing = s.get(PmStrike, (strategy_id, question_idx))
            if existing is None:
                s.add(PmStrike(strategy_id=strategy_id, question_idx=question_idx, strike=strike))
            else:
                existing.strike = strike
                s.add(existing)
            s.commit()

    # ---- coin → market-class map (SHR-77) ----

    def set_coin_klass(self, *, coin: str, klass: str, question_idx: int, strategy_id: str = "") -> None:
        """Persist (or refresh) the market class for an HL outcome coin "#N".

        Idempotent upsert: a question's QuestionMetaEvent is re-ingested on every
        engine restart, and its class never changes, so re-stamping the same pair
        must not error or duplicate."""
        with _Session(self._engine) as s:
            existing = s.get(CoinKlass, (strategy_id, coin))
            if existing is None:
                s.add(CoinKlass(strategy_id=strategy_id, coin=coin, klass=klass, question_idx=question_idx))
            else:
                existing.klass = klass
                existing.question_idx = question_idx
                s.add(existing)
            s.commit()

    def coin_klass_map(self, *, strategy_id: str = "") -> dict[str, str]:
        """All persisted coin("#N") → klass pairs, for the daily report's split."""
        with _Session(self._engine) as s:
            stmt = select(CoinKlass)
            if strategy_id:
                stmt = stmt.where(CoinKlass.strategy_id == strategy_id)
            return {r.coin: r.klass for r in s.exec(stmt).all()}

    # ---- settlements ----

    def record_settlement(
        self,
        *,
        question_idx: int,
        symbol: str,
        realized_pnl: float,
        ts_ns: int,
        strategy_id: str = "",
        account: str | None = None,
    ) -> None:
        """Persist a settled position's realized PnL, keyed by (strategy_id, question_idx)
        (SHR-53). Upsert (single row per slot/qidx pair) so the two close paths — the
        reconcile vanished-position path and router._close_settled — can't
        double-book: the daily-loss gate sums one row per settlement, never two.
        Last write wins, which is correct because the vanished-position path can
        fire BEFORE the settle event with an incomplete PnL (falls back to prior
        realized) and then re-emit the authoritative payout once settled_symbol
        is known; the authoritative value, written later, must overwrite."""
        with _Session(self._engine) as s:
            existing = s.get(Settlement, (strategy_id, question_idx))
            if existing is None:
                s.add(
                    Settlement(
                        strategy_id=strategy_id,
                        question_idx=question_idx,
                        symbol=symbol,
                        realized_pnl=realized_pnl,
                        ts_ns=ts_ns,
                        account=account,
                    )
                )
            else:
                existing.symbol = symbol
                existing.realized_pnl = realized_pnl
                existing.ts_ns = ts_ns
                if account is not None:
                    existing.account = account
                s.add(existing)
            s.commit()

    def settlement_pnl_since(self, since_ts_ns: int, *, strategy_id: str = "") -> float:
        with _Session(self._engine) as s:
            stmt = select(Settlement).where(Settlement.ts_ns >= since_ts_ns)
            if strategy_id:
                stmt = stmt.where(Settlement.strategy_id == strategy_id)
            rows = list(s.exec(stmt).all())
        return sum(r.realized_pnl for r in rows)

    # ---- fills ----

    def append_fill(self, f: Fill, *, strategy_id: str = "", account: str | None = None) -> bool:
        """Insert a Fill row if its fill_id is not already present. Returns True
        when a new row was inserted, False when it already existed (the dedup
        that lets the venue mirror run idempotently every reconcile cycle).

        strategy_id / account stamp the scoping columns when provided by
        StrategyScopedDAL; un-scoped direct calls (strategy_id="") are unchanged."""
        if strategy_id and not f.strategy_id:
            f.strategy_id = strategy_id
        if account is not None and f.account is None:
            f.account = account
        with _Session(self._engine) as s:
            existing = s.get(Fill, f.fill_id)
            if existing is None:
                s.add(f)
                s.commit()
                return True
        return False

    def mirror_venue_fills(
        self,
        fills,
        *,
        symbol_to_question: dict[str, int] | None = None,
        strategy_id: str | None = None,
        account: str | None = None,
    ) -> int:
        """Book HL venue user_fills into the local Fill table as source='venue'
        (SHR-74). Only outcome ("#") fills are mirrored — perp/spot legs are not
        strategy markets and would pollute the outcome-only realized figure that
        the daily report reconciles against. Keyed by the venue tid
        (``fill.fill_id``) and idempotent via append_fill, so calling it every
        reconcile cycle only books newly-seen fills (settlement-dir payouts
        included). Returns the count of rows newly inserted.

        ``fills`` is any iterable of objects with the UserFillRow shape
        (fill_id, cloid, symbol, side, price, size, fee, ts_ns, closed_pnl).

        ``strategy_id`` / ``account``: when provided (stamped by
        StrategyScopedDAL), each mirrored Fill row is tagged with the owning
        strategy so that ``realized_pnl_since(strategy_id=...)`` finds it. In
        the current 1:1 (one strategy per account) configuration, the owning
        strategy is the account's single strategy — multi-strategy attribution
        by symbol is a documented follow-up for when multiple strategies share
        an account. When ``strategy_id`` is None (direct / un-scoped call), Fill
        rows are written with no strategy tag (backward-compatible)."""
        sym_q = symbol_to_question or {}
        # Normalise: treat empty-string the same as None (no scope).
        _sid = strategy_id or None
        inserted = 0
        for f in fills:
            if not f.symbol.startswith("#"):
                continue
            if self.append_fill(
                Fill(
                    fill_id=f.fill_id,
                    cloid=f.cloid,
                    question_idx=sym_q.get(f.symbol, -1),
                    symbol=f.symbol,
                    side=f.side,
                    price=f.price,
                    size=f.size,
                    fee=f.fee,
                    ts_ns=f.ts_ns,
                    closed_pnl=f.closed_pnl,
                    source=FILL_SOURCE_VENUE,
                    strategy_id=_sid,
                    account=account,
                )
            ):
                inserted += 1
        return inserted

    def fills_count(self, *, strategy_id: str = "") -> int:
        """Total number of Fill rows (used by the daily report)."""
        with _Session(self._engine) as s:
            stmt = select(func.count()).select_from(Fill)
            if strategy_id:
                stmt = stmt.where(Fill.strategy_id == strategy_id)
            return int(s.exec(stmt).one())

    def fills_for_cloid(self, cloid: str, *, strategy_id: str = "") -> list[Fill]:
        with _Session(self._engine) as s:
            stmt = select(Fill).where(Fill.cloid == cloid).order_by(Fill.ts_ns)
            if strategy_id:
                stmt = stmt.where(Fill.strategy_id == strategy_id)
            return list(s.exec(stmt).all())

    # ---- session ----

    def start_session(self, session_id: str, *, now_ns: int) -> None:
        with _Session(self._engine) as s:
            s.add(Session_(session_id=session_id, started_ts_ns=now_ns))
            s.commit()

    def end_session(self, session_id: str, *, now_ns: int, halt_reason: str | None = None) -> None:
        # Finding #47: halt_reason is now persisted via the ORM (Session_ model
        # was updated to include the field). The column already existed in the DB
        # DDL (0001_baseline), so no migration is required — the field was simply
        # missing from the ORM class and silently discarded on commit.
        with _Session(self._engine) as s:
            row = s.get(Session_, session_id)
            if row is not None:
                row.ended_ts_ns = now_ns
                if halt_reason is not None:
                    row.halt_reason = halt_reason
                s.add(row)
                s.commit()

    # ---- events ----

    def append_event(
        self,
        *,
        ts_ns: int,
        alias: str | None,
        kind: str,
        question_idx: int | None,
        reason: str | None,
        payload_json: str | None,
        strategy_id: str | None = None,
    ) -> None:
        """Insert one event row. Thread-safe (each call opens its own session
        on the shared engine, same connect-per-call isolation as before)."""
        with _Session(self._engine) as s:
            s.add(
                Event(
                    ts_ns=ts_ns,
                    alias=alias,
                    kind=kind,
                    question_idx=question_idx,
                    reason=reason,
                    payload_json=payload_json,
                    strategy_id=strategy_id,
                )
            )
            s.commit()

    def events_since(self, since_ts_ns: int, *, strategy_id: str | None = None) -> list[dict[str, Any]]:
        """Return all events with ts_ns >= since_ts_ns, ordered by ts_ns asc.

        When strategy_id is provided (non-None), filter to that strategy. When
        None (default), return events for all strategies (global view).
        """
        with _Session(self._engine) as s:
            stmt = select(Event).where(Event.ts_ns >= since_ts_ns)
            if strategy_id is not None:
                stmt = stmt.where(Event.strategy_id == strategy_id)
            stmt = stmt.order_by(Event.ts_ns)
            rows = s.exec(stmt).all()
            return [r.model_dump() for r in rows]

    def reject_counts_since(self, since_ts_ns: int, *, strategy_id: str | None = None) -> list[dict[str, Any]]:
        """Group events by (kind, reason) since since_ts_ns with counts.

        Returns a list of dicts with keys: kind, reason, count, sample_payload.
        Covers all event kinds (not just order_rejected) so callers can aggregate
        risk_veto counts too. Rows ordered by count desc.

        When strategy_id is provided (non-None), filter to that strategy. When
        None (default), aggregate across all strategies (global view).
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if strategy_id is not None:
                rows = conn.execute(
                    """
                    SELECT kind, reason, COUNT(*) AS count,
                           MAX(payload_json) AS sample_payload
                    FROM events
                    WHERE ts_ns >= ? AND strategy_id = ?
                    GROUP BY kind, reason
                    ORDER BY count DESC
                    """,
                    (since_ts_ns, strategy_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT kind, reason, COUNT(*) AS count,
                           MAX(payload_json) AS sample_payload
                    FROM events
                    WHERE ts_ns >= ?
                    GROUP BY kind, reason
                    ORDER BY count DESC
                    """,
                    (since_ts_ns,),
                ).fetchall()
        return [dict(r) for r in rows]

    def events_for_question(self, question_idx: int, *, strategy_id: str | None = None) -> list[dict[str, Any]]:
        """Return all events for a given question_idx, ordered by ts_ns asc.

        When strategy_id is provided (non-None), filter to that strategy. When
        None (default), return events across all strategies (global view).
        """
        with _Session(self._engine) as s:
            stmt = select(Event).where(Event.question_idx == question_idx)
            if strategy_id is not None:
                stmt = stmt.where(Event.strategy_id == strategy_id)
            stmt = stmt.order_by(Event.ts_ns)
            rows = s.exec(stmt).all()
            return [r.model_dump() for r in rows]

    def last_event_by_kind(
        self, kind: str, *, alias: str | None = None, strategy_id: str | None = None
    ) -> dict[str, Any] | None:
        """Return the single most-recent event of the given kind.

        When alias is provided, filter to that slot. When strategy_id is
        provided (non-None), filter to that strategy. Returns None if no match.
        """
        stmt = select(Event).where(Event.kind == kind)
        if alias is not None:
            stmt = stmt.where(Event.alias == alias)
        if strategy_id is not None:
            stmt = stmt.where(Event.strategy_id == strategy_id)
        stmt = stmt.order_by(Event.ts_ns.desc()).limit(1)
        with _Session(self._engine) as s:
            row = s.exec(stmt).first()
        return row.model_dump() if row is not None else None

    def prune_events(self, *, max_age_ns: int, max_rows: int) -> None:
        """Delete events older than max_age_ns, then enforce max_rows ceiling.

        Age prune runs first (primary). Row-count prune is a burst backstop:
        if a reject storm or heartbeat flood writes more rows than the age
        prune removes, the oldest rows above the ceiling are deleted.
        Both bounds apply on every call — caller decides the frequency.
        """
        cutoff_ns = time.time_ns() - max_age_ns
        with _Session(self._engine) as s:
            # 1) Age prune
            s.exec(delete(Event).where(Event.ts_ns < cutoff_ns))
            # 2) Row-count ceiling: keep only the newest max_rows rows.
            # We count current rows; if count > max_rows, delete the oldest
            # (count - max_rows) rows. Using COUNT avoids the NULL-subquery
            # problem when OFFSET >= row count (LIMIT 1 OFFSET N returns NULL
            # when N >= count, causing `id <= NULL` to silently delete nothing
            # or everything depending on the DB).
            count = s.exec(select(func.count()).select_from(Event)).one()
            excess = count - max_rows
            if excess > 0:
                oldest_ids = select(Event.id).order_by(Event.id.asc()).limit(excess)
                s.exec(delete(Event).where(Event.id.in_(oldest_ids)))
            s.commit()

    # ---- trade journal (SHR-83) ----

    def add_journal_decision(
        self,
        row: TradeJournalRow,
        *,
        strategy_id: str = "",
        account: str | None = None,
    ) -> bool:
        """Insert a journal row if its cloid is not already present. Returns True
        when a new row was inserted, False when one already existed (cloids are
        unique per order, so a duplicate decision is a no-op — first write wins,
        matching append_fill's insert-once semantics).

        strategy_id / account stamp the scoping columns when provided by
        StrategyScopedDAL; un-scoped direct calls are unchanged."""
        if strategy_id and not row.strategy_id:
            row.strategy_id = strategy_id
        if account is not None and row.account is None:
            row.account = account
        with _Session(self._engine) as s:
            if s.get(TradeJournalRow, row.cloid) is not None:
                return False
            s.add(row)
            s.commit()
            return True

    def update_journal(self, cloid: str, *, strategy_id: str = "", **changes: Any) -> None:
        """Set the given columns on an existing journal row. No-op if the cloid
        has no decision row (a stray send/fill/reject is dropped rather than
        creating a partial row).

        When strategy_id is provided, only update the row if it belongs to that
        strategy (defensive guard for shared-DB usage)."""
        if not changes:
            return
        with _Session(self._engine) as s:
            row = s.get(TradeJournalRow, cloid)
            if row is None:
                return
            if strategy_id and row.strategy_id and row.strategy_id != strategy_id:
                return
            for k, v in changes.items():
                setattr(row, k, v)
            s.add(row)
            s.commit()

    def get_journal_row(self, cloid: str, *, strategy_id: str = "") -> TradeJournalRow | None:
        with _Session(self._engine) as s:
            row = s.get(TradeJournalRow, cloid)
            if row is None:
                return None
            if strategy_id and row.strategy_id and row.strategy_id != strategy_id:
                return None
            return row

    def delete_journal_decision(self, cloid: str, *, strategy_id: str = "") -> None:
        """Remove a journal row by cloid (no-op if absent). Used to drop a
        decision that was vetoed by a routine, high-frequency gate reason we
        don't retain — the row is inserted at decision time, then removed here
        once the suppressed veto is known (see TradeJournal.record_reject).

        When strategy_id is provided, only delete the row if it belongs to that
        strategy (defensive guard for shared-DB usage)."""
        with _Session(self._engine) as s:
            if strategy_id:
                row = s.get(TradeJournalRow, cloid)
                if row is None:
                    return
                if row.strategy_id and row.strategy_id != strategy_id:
                    return
                s.delete(row)
            else:
                s.exec(delete(TradeJournalRow).where(TradeJournalRow.cloid == cloid))
            s.commit()

    def prune_trade_journal(self, *, max_age_ns: int, max_rows: int) -> None:
        """Delete journal rows older than max_age_ns, then enforce max_rows.

        Mirrors ``prune_events``: age prune is primary (rows older than the
        retention window are dropped — they are archived in the daily S3
        state.db snapshot), and the row-count ceiling is a burst backstop for a
        high-fan-out slot that journals faster than the age prune removes. Rows
        are ordered by decision_ts_ns; the newest max_rows are kept."""
        cutoff_ns = time.time_ns() - max_age_ns
        with _Session(self._engine) as s:
            # 1) Age prune
            s.exec(delete(TradeJournalRow).where(TradeJournalRow.decision_ts_ns < cutoff_ns))
            # 2) Row-count ceiling: keep only the newest max_rows by decision_ts.
            count = s.exec(select(func.count()).select_from(TradeJournalRow)).one()
            excess = count - max_rows
            if excess > 0:
                oldest = select(TradeJournalRow.cloid).order_by(TradeJournalRow.decision_ts_ns.asc()).limit(excess)
                s.exec(delete(TradeJournalRow).where(TradeJournalRow.cloid.in_(oldest)))
            s.commit()

    # ---- realized pnl helpers ----

    def realized_pnl_since(self, since_ts_ns: int, *, strategy_id: str = "") -> float:
        """Local-DB realized PnL. Diagnostic fallback only — the live daily-loss
        gate now reads from HL (HLClient.realized_pnl_since) because the local
        fill table was historically empty on the happy path and closed
        positions were deleted before their realized PnL could be summed here.

        With the 0003 migration, Router._book_fill writes a Fill row on every
        venue fill (including closed_pnl on reduces), so this calc is finally
        meaningful for post-hoc analysis. The sum of (closed_pnl - fee) across
        the window's fills IS the authoritative windowed realized-from-trades
        figure — including the partial-reduce PnL of positions still open. We do
        NOT add the open positions' accumulated realized_pnl: that PnL is already
        in the Fill sum, so adding it would double-count any still-open position
        partially reduced inside the window (SHR-72).

        Source preference (SHR-74): when the slot has ANY source='venue' row
        (an HL slot whose ledger the reconcile loop mirrors from user_fills), we
        sum ONLY those — they carry the venue's own closedPnl/fee and include the
        HIP-4 settlement payouts that never reach _book_fill, so the figure
        equals venue truth by construction. The 'router' rows (cloid-keyed,
        fee=0, locally-computed) are then redundant diagnostics and excluded to
        avoid double-counting. A PM slot (or a not-yet-mirrored HL slot) has no
        'venue' rows, so this falls back to summing all rows — its 'router'
        ledger is authoritative.

        When strategy_id is provided, filter fills and settlements to that
        strategy (for shared-DB usage via StrategyScopedDAL).
        """
        with _Session(self._engine) as s:
            venue_stmt = select(func.count()).select_from(Fill).where(Fill.source == FILL_SOURCE_VENUE)
            if strategy_id:
                venue_stmt = venue_stmt.where(Fill.strategy_id == strategy_id)
            has_venue = s.exec(venue_stmt).one() > 0
            stmt = select(Fill).where(Fill.ts_ns >= since_ts_ns)
            if strategy_id:
                stmt = stmt.where(Fill.strategy_id == strategy_id)
            if has_venue:
                stmt = stmt.where(Fill.source == FILL_SOURCE_VENUE)
            fills = list(s.exec(stmt).all())
        # Settlement payouts are not fills (HIP-4 binaries close via settlement,
        # not HL trades), so without this the dominant PnL component of the
        # binary strategy is invisible here (SHR-53/49).
        return sum(getattr(f, "closed_pnl", 0.0) - f.fee for f in fills) + self.settlement_pnl_since(
            since_ts_ns, strategy_id=strategy_id
        )


class CachedStateDAL(StateDAL):
    """StateDAL with an in-memory write-through cache of positions + live orders.

    Why: the scan, stop-loss, and heartbeat loops read positions/orders many
    times per second; under an event-driven loop (P1) that read rate climbs.
    Those reads are all on the event-loop thread and `slot.dal` is the single
    handle to each state.db, so a cache layered here is coherent by construction
    — there is no out-of-band writer to invalidate it.

    Invariants:
      * Reads (all_positions/get_position/live_orders) serve from memory.
      * Writes go DB-FIRST (super().<write>()), THEN update the cache, so a failed
        commit raises before the cache mutates → DB and cache fail together.
      * The cache is lazily loaded from the DB on first access (AFTER run_migrations
        has created the tables) so restart recovery is automatic.
      * Off-loop aggregate reads (settlement_pnl_since / realized_pnl_since) are
        inherited UNCHANGED and keep reading the DB — they run on worker threads
        and must never touch the cache dicts. Same for the events-table helpers.
      * The cache is namespaced by strategy_id so multiple StrategyScopedDAL
        instances sharing this CachedStateDAL remain isolated in memory.
    """

    def __init__(self, db_path: Path) -> None:
        super().__init__(db_path)
        self._cache_lock = threading.Lock()
        # Per-strategy_id loaded flags and caches.
        self._loaded: dict[str, bool] = {}
        self._pos_cache: dict[str, dict[int, Position]] = {}  # strategy_id -> qidx -> Position
        self._order_cache: dict[str, dict[str, OpenOrder]] = {}  # strategy_id -> cloid -> OpenOrder

    def _ensure_loaded(self, strategy_id: str = "") -> None:
        # Fast path (no lock): already loaded for this strategy_id.
        if self._loaded.get(strategy_id):
            return
        with self._cache_lock:
            # Re-check under the lock; another thread may have loaded while we
            # were waiting.
            if self._loaded.get(strategy_id):
                return
            # Pull current truth from the DB once for this strategy_id. live_orders()
            # returns only the not-terminal statuses, which is all the cache needs.
            self._pos_cache[strategy_id] = {p.question_idx: p for p in super().all_positions(strategy_id=strategy_id)}
            self._order_cache[strategy_id] = {o.cloid: o for o in super().live_orders(strategy_id=strategy_id)}
            self._loaded[strategy_id] = True

    # ---- positions: cached reads, write-through writes ----

    def upsert_position(self, p: Position, *, strategy_id: str = "", account: str | None = None) -> None:
        sid = strategy_id or p.strategy_id
        self._ensure_loaded(sid)
        # DB first; the base writes with expire_on_commit=False so `p` stays
        # readable after the commit and can be cached directly.
        super().upsert_position(p, strategy_id=strategy_id, account=account)
        with self._cache_lock:
            self._pos_cache.setdefault(sid, {})[p.question_idx] = p

    def get_position(self, question_idx: int, *, strategy_id: str = "") -> Position | None:
        self._ensure_loaded(strategy_id)
        with self._cache_lock:
            return self._pos_cache.get(strategy_id, {}).get(question_idx)

    def all_positions(self, *, strategy_id: str = "") -> list[Position]:
        self._ensure_loaded(strategy_id)
        with self._cache_lock:
            return list(self._pos_cache.get(strategy_id, {}).values())

    def delete_position(self, question_idx: int, *, strategy_id: str = "") -> None:
        self._ensure_loaded(strategy_id)
        super().delete_position(question_idx, strategy_id=strategy_id)
        with self._cache_lock:
            self._pos_cache.get(strategy_id, {}).pop(question_idx, None)

    # ---- orders: cached reads, write-through writes ----

    def upsert_order(self, o: OpenOrder, *, strategy_id: str = "", account: str | None = None) -> None:
        sid = strategy_id or o.strategy_id
        self._ensure_loaded(sid)
        # DB first; the base writes with expire_on_commit=False so `o` stays
        # readable after the commit and can be cached directly.
        super().upsert_order(o, strategy_id=strategy_id, account=account)
        with self._cache_lock:
            self._order_cache.setdefault(sid, {})[o.cloid] = o

    def update_order_status(
        self,
        cloid: str,
        *,
        status: str,
        venue_oid: str | None = None,
        now_ns: int,
        strategy_id: str = "",
    ) -> None:
        self._ensure_loaded(strategy_id)
        super().update_order_status(
            cloid,
            status=status,
            venue_oid=venue_oid,
            now_ns=now_ns,
            strategy_id=strategy_id,
        )
        with self._cache_lock:
            o = self._order_cache.get(strategy_id, {}).get(cloid)
            if o is not None:
                o.status = status
                o.last_update_ts_ns = now_ns
                if venue_oid is not None:
                    o.venue_oid = venue_oid

    def live_orders(self, *, strategy_id: str = "") -> list[OpenOrder]:
        self._ensure_loaded(strategy_id)
        with self._cache_lock:
            return [
                o
                for o in self._order_cache.get(strategy_id, {}).values()
                if o.status in ("pending", "open", "partially_filled")
            ]
