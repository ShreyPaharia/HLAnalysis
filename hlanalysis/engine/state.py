from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import delete, func
from sqlmodel import Field, Session as _Session, SQLModel, create_engine, select


class OpenOrder(SQLModel, table=True):
    cloid: str = Field(primary_key=True)
    venue_oid: str | None = None
    question_idx: int
    symbol: str
    side: str   # "buy" | "sell"
    price: float
    size: float
    status: str  # "pending" | "open" | "partially_filled" | "filled" | "cancelled" | "rejected"
    placed_ts_ns: int
    last_update_ts_ns: int
    strategy_id: str


class Position(SQLModel, table=True):
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


class Fill(SQLModel, table=True):
    fill_id: str = Field(primary_key=True)
    cloid: str
    question_idx: int
    symbol: str
    side: str   # "buy" | "sell"
    price: float
    size: float
    fee: float
    ts_ns: int
    # Locally computed realized PnL on this fill (0 on opens, signed on
    # reduces): (fill_price − position.avg_entry) × closed_qty. Populated by
    # Router._book_fill for post-hoc audit; the daily-loss gate reads HL's
    # closedPnl directly via HLClient.hl_realized_pnl_since.
    closed_pnl: float = 0.0


class Session_(SQLModel, table=True):
    __tablename__ = "session"
    session_id: str = Field(primary_key=True)
    started_ts_ns: int
    ended_ts_ns: int | None = None


class SeenQuestion(SQLModel, table=True):
    """Tracks question_idxs the engine has notified about. Persists across
    restarts so NewQuestion alerts don't re-fire for already-known markets."""
    __tablename__ = "seen_question"
    question_idx: int = Field(primary_key=True)
    first_seen_ts_ns: int


class PmStrike(SQLModel, table=True):
    """Open-strike captured for a Polymarket up/down question. PM up/down
    markets have no static strike; the engine stamps it from the Binance SPOT
    1m candle close at strike_ref_ts_ns. Persisting it lets a restarted engine
    reuse the strike instead of skipping markets whose open it can no longer see."""
    __tablename__ = "pm_strike"
    question_idx: int = Field(primary_key=True)
    strike: float


class Settlement(SQLModel, table=True):
    """Persisted realized PnL of a settled position (SHR-53). HIP-4 binaries
    close via settlement payouts, not HL fills, so this PnL was previously only
    alerted, never stored — leaving the daily-loss gate blind to it. Keyed by
    question_idx so the two close paths can't double-book (first writer wins)."""
    __tablename__ = "settlement"
    question_idx: int = Field(primary_key=True)
    symbol: str
    realized_pnl: float
    ts_ns: int


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
            return {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

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

    def upsert_order(self, o: OpenOrder) -> None:
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

    def get_order(self, cloid: str) -> OpenOrder | None:
        with _Session(self._engine) as s:
            return s.get(OpenOrder, cloid)

    def update_order_status(
        self, cloid: str, *, status: str, venue_oid: str | None = None, now_ns: int,
    ) -> None:
        with _Session(self._engine) as s:
            o = s.get(OpenOrder, cloid)
            if o is None:
                return
            o.status = status  # type: ignore[assignment]
            if venue_oid is not None:
                o.venue_oid = venue_oid
            o.last_update_ts_ns = now_ns
            s.add(o)
            s.commit()

    def live_orders(self) -> list[OpenOrder]:
        with _Session(self._engine) as s:
            stmt = select(OpenOrder).where(
                OpenOrder.status.in_(("pending", "open", "partially_filled"))
            )
            return list(s.exec(stmt).all())

    # ---- positions ----

    def upsert_position(self, p: Position) -> None:
        # expire_on_commit=False so `p` stays readable after commit (the cached
        # subclass reads its fields post-write without a re-SELECT).
        with _Session(self._engine, expire_on_commit=False) as s:
            existing = s.get(Position, p.question_idx)
            if existing is None:
                s.add(p)
            else:
                for k, v in p.model_dump().items():
                    setattr(existing, k, v)
                s.add(existing)
            s.commit()

    def get_position(self, question_idx: int) -> Position | None:
        with _Session(self._engine) as s:
            return s.get(Position, question_idx)

    def all_positions(self) -> list[Position]:
        with _Session(self._engine) as s:
            return list(s.exec(select(Position)).all())

    def delete_position(self, question_idx: int) -> None:
        with _Session(self._engine) as s:
            p = s.get(Position, question_idx)
            if p is not None:
                s.delete(p)
                s.commit()

    def has_seen_question(self, question_idx: int) -> bool:
        with _Session(self._engine) as s:
            return s.get(SeenQuestion, question_idx) is not None

    def mark_question_seen(self, question_idx: int, *, now_ns: int) -> None:
        with _Session(self._engine) as s:
            if s.get(SeenQuestion, question_idx) is None:
                s.add(SeenQuestion(question_idx=question_idx, first_seen_ts_ns=now_ns))
                s.commit()

    # ---- PM open-strikes ----

    def get_pm_strike(self, question_idx: int) -> float | None:
        with _Session(self._engine) as s:
            row = s.get(PmStrike, question_idx)
            return None if row is None else row.strike

    def set_pm_strike(self, question_idx: int, strike: float) -> None:
        with _Session(self._engine) as s:
            existing = s.get(PmStrike, question_idx)
            if existing is None:
                s.add(PmStrike(question_idx=question_idx, strike=strike))
            else:
                existing.strike = strike
                s.add(existing)
            s.commit()

    # ---- settlements ----

    def record_settlement(
        self, *, question_idx: int, symbol: str, realized_pnl: float, ts_ns: int,
    ) -> None:
        """Persist a settled position's realized PnL, keyed by question_idx
        (SHR-53). Upsert (single row per qidx) so the two close paths — the
        reconcile vanished-position path and router._close_settled — can't
        double-book: the daily-loss gate sums one row per settlement, never two.
        Last write wins, which is correct because the vanished-position path can
        fire BEFORE the settle event with an incomplete PnL (falls back to prior
        realized) and then re-emit the authoritative payout once settled_symbol
        is known; the authoritative value, written later, must overwrite."""
        with _Session(self._engine) as s:
            existing = s.get(Settlement, question_idx)
            if existing is None:
                s.add(Settlement(
                    question_idx=question_idx, symbol=symbol,
                    realized_pnl=realized_pnl, ts_ns=ts_ns,
                ))
            else:
                existing.symbol = symbol
                existing.realized_pnl = realized_pnl
                existing.ts_ns = ts_ns
                s.add(existing)
            s.commit()

    def settlement_pnl_since(self, since_ts_ns: int) -> float:
        with _Session(self._engine) as s:
            rows = list(s.exec(
                select(Settlement).where(Settlement.ts_ns >= since_ts_ns)
            ).all())
        return sum(r.realized_pnl for r in rows)

    # ---- fills ----

    def append_fill(self, f: Fill) -> None:
        with _Session(self._engine) as s:
            existing = s.get(Fill, f.fill_id)
            if existing is None:
                s.add(f)
                s.commit()

    def fills_count(self) -> int:
        """Total number of Fill rows (used by the daily report)."""
        with _Session(self._engine) as s:
            return int(s.exec(select(func.count()).select_from(Fill)).one())

    def fills_for_cloid(self, cloid: str) -> list[Fill]:
        with _Session(self._engine) as s:
            stmt = select(Fill).where(Fill.cloid == cloid).order_by(Fill.ts_ns)
            return list(s.exec(stmt).all())

    # ---- session ----

    def start_session(self, session_id: str, *, now_ns: int) -> None:
        with _Session(self._engine) as s:
            s.add(Session_(session_id=session_id, started_ts_ns=now_ns))
            s.commit()

    def end_session(self, session_id: str, *, now_ns: int, halt_reason: str | None = None) -> None:
        with _Session(self._engine) as s:
            row = s.get(Session_, session_id)
            if row is not None:
                row.ended_ts_ns = now_ns
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
    ) -> None:
        """Insert one event row. Thread-safe (each call opens its own session
        on the shared engine, same connect-per-call isolation as before)."""
        with _Session(self._engine) as s:
            s.add(Event(
                ts_ns=ts_ns, alias=alias, kind=kind,
                question_idx=question_idx, reason=reason, payload_json=payload_json,
            ))
            s.commit()

    def events_since(self, since_ts_ns: int) -> list[dict[str, Any]]:
        """Return all events with ts_ns >= since_ts_ns, ordered by ts_ns asc."""
        with _Session(self._engine) as s:
            rows = s.exec(
                select(Event).where(Event.ts_ns >= since_ts_ns).order_by(Event.ts_ns)
            ).all()
            return [r.model_dump() for r in rows]

    def reject_counts_since(self, since_ts_ns: int) -> list[dict[str, Any]]:
        """Group events by (kind, reason) since since_ts_ns with counts.

        Returns a list of dicts with keys: kind, reason, count, sample_payload.
        Covers all event kinds (not just order_rejected) so callers can aggregate
        risk_veto counts too. Rows ordered by count desc.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
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

    def events_for_question(self, question_idx: int) -> list[dict[str, Any]]:
        """Return all events for a given question_idx, ordered by ts_ns asc."""
        with _Session(self._engine) as s:
            rows = s.exec(
                select(Event)
                .where(Event.question_idx == question_idx)
                .order_by(Event.ts_ns)
            ).all()
            return [r.model_dump() for r in rows]

    def last_event_by_kind(
        self, kind: str, *, alias: str | None = None
    ) -> dict[str, Any] | None:
        """Return the single most-recent event of the given kind.

        When alias is provided, filter to that slot. Returns None if no match.
        """
        stmt = select(Event).where(Event.kind == kind)
        if alias is not None:
            stmt = stmt.where(Event.alias == alias)
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
                oldest_ids = (
                    select(Event.id).order_by(Event.id.asc()).limit(excess)
                )
                s.exec(delete(Event).where(Event.id.in_(oldest_ids)))
            s.commit()

    # ---- realized pnl helpers ----

    def realized_pnl_since(self, since_ts_ns: int) -> float:
        """Local-DB realized PnL. Diagnostic fallback only — the live daily-loss
        gate now reads from HL (HLClient.realized_pnl_since) because the local
        fill table was historically empty on the happy path and closed
        positions were deleted before their realized PnL could be summed here.

        With the 0003 migration, Router._book_fill writes a Fill row on every
        venue fill (including closed_pnl on reduces), so this calc is finally
        meaningful for post-hoc analysis. We sum (closed_pnl - fee) across
        today's fills plus any residual realized_pnl on still-open positions
        (intra-cycle partial closes).
        """
        with _Session(self._engine) as s:
            stmt = select(Fill).where(Fill.ts_ns >= since_ts_ns)
            fills = list(s.exec(stmt).all())
        with _Session(self._engine) as s:
            positions = list(s.exec(select(Position)).all())
        # Settlement payouts are not fills (HIP-4 binaries close via settlement,
        # not HL trades), so without this the dominant PnL component of the
        # binary strategy is invisible here (SHR-53/49).
        # ⚠️ SHR-72: this DOUBLE-COUNTS the realized PnL of a position that has
        # been partially reduced but is still open — its partial-reduce PnL is in
        # BOTH the Fill.closed_pnl sum above AND the open-position realized_pnl
        # sum below. Fully-closed positions are fine (the row is deleted). Fix is
        # to drop the positions term now that _book_fill writes a Fill row with
        # closed_pnl on every fill. Tracked; fix before Phase-2 sizing.
        return (
            sum(getattr(f, "closed_pnl", 0.0) - f.fee for f in fills)
            + sum(p.realized_pnl for p in positions)
            + self.settlement_pnl_since(since_ts_ns)
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
    """

    def __init__(self, db_path: Path) -> None:
        super().__init__(db_path)
        self._loaded = False
        self._pos_cache: dict[int, Position] = {}
        self._order_cache: dict[str, OpenOrder] = {}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        # Pull current truth from the DB once. live_orders() returns only the
        # not-terminal statuses, which is all the cache ever needs to serve.
        self._pos_cache = {p.question_idx: p for p in super().all_positions()}
        self._order_cache = {o.cloid: o for o in super().live_orders()}
        self._loaded = True

    # ---- positions: cached reads, write-through writes ----

    def upsert_position(self, p: Position) -> None:
        self._ensure_loaded()
        # DB first; the base writes with expire_on_commit=False so `p` stays
        # readable after the commit and can be cached directly.
        super().upsert_position(p)
        self._pos_cache[p.question_idx] = p

    def get_position(self, question_idx: int) -> Position | None:
        self._ensure_loaded()
        return self._pos_cache.get(question_idx)

    def all_positions(self) -> list[Position]:
        self._ensure_loaded()
        return list(self._pos_cache.values())

    def delete_position(self, question_idx: int) -> None:
        self._ensure_loaded()
        super().delete_position(question_idx)
        self._pos_cache.pop(question_idx, None)

    # ---- orders: cached reads, write-through writes ----

    def upsert_order(self, o: OpenOrder) -> None:
        self._ensure_loaded()
        # DB first; the base writes with expire_on_commit=False so `o` stays
        # readable after the commit and can be cached directly.
        super().upsert_order(o)
        self._order_cache[o.cloid] = o

    def update_order_status(
        self, cloid: str, *, status: str, venue_oid: str | None = None, now_ns: int,
    ) -> None:
        self._ensure_loaded()
        super().update_order_status(
            cloid, status=status, venue_oid=venue_oid, now_ns=now_ns,
        )
        o = self._order_cache.get(cloid)
        if o is not None:
            o.status = status
            o.last_update_ts_ns = now_ns
            if venue_oid is not None:
                o.venue_oid = venue_oid

    def live_orders(self) -> list[OpenOrder]:
        self._ensure_loaded()
        return [
            o for o in self._order_cache.values()
            if o.status in ("pending", "open", "partially_filled")
        ]
