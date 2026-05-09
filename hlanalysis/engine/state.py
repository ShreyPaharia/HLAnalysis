from __future__ import annotations

import sqlite3
import time
from pathlib import Path

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


# Public alias for tests / external users.
Session = Session_  # noqa: F811


_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class StateDAL:
    """Thin DAL over sqlite via sqlmodel + raw sqlite3 for migrations.

    We do NOT use SQLModel.metadata.create_all() — the spec calls for explicit
    SQL migration files even in Phase 1, with versions tracked in `schema_migrations`.
    Alembic adopted in Phase 2 (spec §5.1).
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

    def run_migrations(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version TEXT PRIMARY KEY, applied_at_ns INTEGER NOT NULL)"
            )
            conn.commit()
            applied = {
                r[0] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()
            }
            for sql_path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
                version = sql_path.stem
                if version in applied:
                    continue
                conn.executescript(sql_path.read_text())
                conn.execute(
                    "INSERT INTO schema_migrations(version, applied_at_ns) VALUES (?, ?)",
                    (version, time.time_ns()),
                )
                conn.commit()

    def applied_versions(self) -> set[str]:
        with sqlite3.connect(self.db_path) as conn:
            return {
                r[0] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()
            }

    # ---- orders ----

    def upsert_order(self, o: OpenOrder) -> None:
        with _Session(self._engine) as s:
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
        with _Session(self._engine) as s:
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

    # ---- fills ----

    def append_fill(self, f: Fill) -> None:
        with _Session(self._engine) as s:
            existing = s.get(Fill, f.fill_id)
            if existing is None:
                s.add(f)
                s.commit()

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

    # ---- realized pnl helpers ----

    def realized_pnl_since(self, since_ts_ns: int) -> float:
        with _Session(self._engine) as s:
            stmt = select(Fill).where(Fill.ts_ns >= since_ts_ns)
            fills = list(s.exec(stmt).all())
        # Phase 1 uses fill.fee as the only material PnL component pre-settlement;
        # closed-position PnL is realized into Position.realized_pnl on close, and
        # we sum that here too.
        with _Session(self._engine) as s:
            positions = list(s.exec(select(Position)).all())
        return -sum(f.fee for f in fills) + sum(p.realized_pnl for p in positions)
