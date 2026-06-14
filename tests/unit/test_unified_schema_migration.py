"""Tests for 0006_unified_slot_db migration — composite PKs + account/strategy_id columns.

These tests deliberately avoid using the ORM models for tables being altered
(Position, SeenQuestion, etc.) because Task 2 adds the ORM-level fields; here
we operate via raw SQL so the test is self-contained.
"""

from __future__ import annotations

import sqlite3

from hlanalysis.engine.state import StateDAL


def test_position_composite_pk_allows_same_qidx_across_strategies(tmp_path):
    """After migration, position PK is (strategy_id, question_idx).

    Two rows with the same question_idx but different strategy_id must both
    commit without a UNIQUE constraint violation.  The table must also expose
    both `strategy_id` and `account` columns.
    """
    db = tmp_path / "u.db"
    dal = StateDAL(db)
    dal.run_migrations()

    conn = sqlite3.connect(db)
    try:
        # Insert two positions with the SAME question_idx, different strategy_id.
        conn.execute(
            "INSERT INTO position "
            "(strategy_id, account, question_idx, symbol, qty, avg_entry, "
            "realized_pnl, last_update_ts_ns, stop_loss_price, closed_qty) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("v31", "w1", 1001, "@30", 1.0, 0.85, 0.0, 1_000_000_000, 0.75, 0.0),
        )
        conn.execute(
            "INSERT INTO position "
            "(strategy_id, account, question_idx, symbol, qty, avg_entry, "
            "realized_pnl, last_update_ts_ns, stop_loss_price, closed_qty) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("v1", "w1", 1001, "@30", 2.0, 0.90, 0.0, 1_000_000_001, 0.80, 0.0),
        )
        conn.commit()  # must NOT raise

        rows = conn.execute("SELECT strategy_id, question_idx FROM position ORDER BY strategy_id").fetchall()
        assert len(rows) == 2
        assert {r[0] for r in rows} == {"v1", "v31"}

        # Confirm PRAGMA shows the new columns.
        col_names = {row[1] for row in conn.execute("PRAGMA table_info(position)").fetchall()}
        assert "strategy_id" in col_names
        assert "account" in col_names
    finally:
        conn.close()


def test_events_strategy_id_column_exists_after_migration(tmp_path):
    """After migration the `events` table must have a `strategy_id` column."""
    db = tmp_path / "ev.db"
    dal = StateDAL(db)
    dal.run_migrations()

    conn = sqlite3.connect(db)
    try:
        col_names = {row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()}
        assert "strategy_id" in col_names
    finally:
        conn.close()


def test_events_backfill_strategy_id_from_alias(tmp_path):
    """Rows pre-existing in `events` that have alias set must have their
    strategy_id backfilled after migration runs.

    We simulate this by inserting a row directly (bypassing the ORM) before
    running migrations, then asserting strategy_id == alias afterwards.
    """
    db = tmp_path / "ev_bf.db"
    # Build a fresh migrated-to-0005 DB by running only up to 0005.
    # We achieve this by first running all migrations (getting to 0005 head
    # before 0006 exists), but since 0006 is what we're testing we use a
    # workaround: build the DB at the 0005 head by using a StateDAL on a
    # fresh DB *without* 0006, then inserting a row, then running migrations
    # again to apply 0006.
    #
    # Because we can't easily run only up to 0005 here, we take a simpler
    # approach: run migrations to head (which includes 0006), insert a row
    # with alias set via raw SQL (no strategy_id), then verify that a second
    # run_migrations() is a no-op and strategy_id still equals alias for the
    # row we inserted with matching alias.
    #
    # Actually the cleanest test: insert with alias BEFORE migration so
    # the UPDATE backfill fires.  We do this by building the pre-0006 schema
    # manually using a raw sqlite3 connection, seeding a row, then calling
    # StateDAL.run_migrations() to apply 0006.

    # Build DB at 0005-head schema by running migrations on a fresh db,
    # then rolling back alembic_version to 0005 so run_migrations sees an
    # upgrade is needed.
    #
    # Simplest correct approach: build schema to 0005 via alembic,
    # seed an alias-only event row, then downgrade+upgrade OR just stamp
    # to 0005 and upgrade.  But we don't have easy access to partial
    # run_migrations here.
    #
    # Use the most portable approach: build a minimal SQLite state directly
    # at the 0005 head by replaying only the DDL we need, stamp alembic
    # to 0005, then call run_migrations to apply 0006.

    import sqlite3
    from pathlib import Path

    from alembic import command
    from alembic.config import Config

    _ALEMBIC_DIR = Path(__file__).parent.parent.parent / "hlanalysis" / "engine" / "migrations_alembic"

    # Build the DB at post-0005 schema by instantiating StateDAL (which goes
    # to head) then downgrade to 0005 via alembic command.
    dal0 = StateDAL(db)
    dal0.run_migrations()  # goes to head (0006 or whatever head is)

    # Downgrade to 0005_trade_journal so we can seed a pre-0006 row.
    cfg = Config()
    cfg.set_main_option("script_location", str(_ALEMBIC_DIR))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")
    command.downgrade(cfg, "0005_trade_journal")

    # Now insert an alias-only events row (strategy_id column doesn't exist yet).
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO events (ts_ns, alias, kind) VALUES (?, ?, ?)",
        (999_000_000_000, "v1", "heartbeat"),
    )
    conn.commit()
    conn.close()

    # Apply 0006 — the UPDATE backfill should stamp strategy_id = alias.
    dal1 = StateDAL(db)
    dal1.run_migrations()

    conn = sqlite3.connect(db)
    try:
        row = conn.execute("SELECT alias, strategy_id FROM events WHERE alias = 'v1'").fetchone()
        assert row is not None, "seeded event row not found"
        assert row[0] == "v1"
        assert row[1] == "v1", f"backfill failed: strategy_id={row[1]!r}, expected 'v1'"
    finally:
        conn.close()


def test_new_columns_nullable_on_fresh_db(tmp_path):
    """After migration all new account/strategy_id columns accept NULL (existing
    rows from a per-slot DB that hasn't been scoped yet)."""
    db = tmp_path / "null_test.db"
    dal = StateDAL(db)
    dal.run_migrations()

    conn = sqlite3.connect(db)
    try:
        # seen_question: insert with NULL strategy_id and account
        conn.execute(
            "INSERT INTO seen_question (strategy_id, account, question_idx, first_seen_ts_ns) VALUES (?, ?, ?, ?)",
            (None, None, 42, 1_000_000),
        )
        # pm_strike: same
        conn.execute(
            "INSERT INTO pm_strike (strategy_id, account, question_idx, strike) VALUES (?, ?, ?, ?)",
            (None, None, 42, 98000.0),
        )
        # settlement: same
        conn.execute(
            "INSERT INTO settlement (strategy_id, account, question_idx, symbol, realized_pnl, ts_ns) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (None, None, 42, "@30", 5.0, 2_000_000),
        )
        # fill: new columns nullable
        conn.execute(
            "INSERT INTO fill (fill_id, cloid, question_idx, symbol, side, price, "
            "size, fee, ts_ns, closed_pnl, source, account, strategy_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("f-null-1", "c-1", 42, "@30", "buy", 0.9, 1.0, 0.01, 3_000_000, 0.0, "router", None, None),
        )
        conn.commit()  # must not raise
    finally:
        conn.close()


def test_migration_idempotent(tmp_path):
    """Running run_migrations() twice must be a no-op (already at head)."""
    db = tmp_path / "idem.db"
    dal = StateDAL(db)
    dal.run_migrations()
    dal.run_migrations()  # second call must not raise or corrupt schema

    conn = sqlite3.connect(db)
    try:
        col_names = {row[1] for row in conn.execute("PRAGMA table_info(position)").fetchall()}
        assert "strategy_id" in col_names
        assert "account" in col_names
    finally:
        conn.close()


def test_seen_question_composite_pk(tmp_path):
    """seen_question PK becomes (strategy_id, question_idx)."""
    db = tmp_path / "sq.db"
    dal = StateDAL(db)
    dal.run_migrations()

    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT INTO seen_question (strategy_id, account, question_idx, first_seen_ts_ns) VALUES (?, ?, ?, ?)",
            ("v31", "w1", 5000, 1_000_000),
        )
        conn.execute(
            "INSERT INTO seen_question (strategy_id, account, question_idx, first_seen_ts_ns) VALUES (?, ?, ?, ?)",
            ("v1", "w1", 5000, 2_000_000),
        )
        conn.commit()  # same question_idx, different strategy_id → OK

        rows = conn.execute("SELECT strategy_id FROM seen_question ORDER BY strategy_id").fetchall()
        assert len(rows) == 2
    finally:
        conn.close()


def test_coin_klass_composite_pk(tmp_path):
    """coin_klass PK becomes (strategy_id, coin)."""
    db = tmp_path / "ck.db"
    dal = StateDAL(db)
    dal.run_migrations()

    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT INTO coin_klass (strategy_id, account, coin, klass, question_idx) VALUES (?, ?, ?, ?, ?)",
            ("v31", "w1", "#10", "priceBinary", 100),
        )
        conn.execute(
            "INSERT INTO coin_klass (strategy_id, account, coin, klass, question_idx) VALUES (?, ?, ?, ?, ?)",
            ("v1", "w1", "#10", "priceBinary", 100),
        )
        conn.commit()  # same coin, different strategy_id → OK

        rows = conn.execute("SELECT strategy_id FROM coin_klass ORDER BY strategy_id").fetchall()
        assert len(rows) == 2
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# ORM round-trip tests (Task 2) — verify SQLModel classes have correct fields
# ---------------------------------------------------------------------------
# These tests import the ORM models directly to ensure:
#   1. New fields (strategy_id, account) are present at the ORM layer.
#   2. Composite PKs on Position/PmStrike/Settlement/SeenQuestion/CoinKlass
#      match what migration 0006 creates (no SAWarning about PK mismatch).
#   3. Fill and TradeJournalRow carry nullable strategy_id/account.
#   4. Event carries a nullable strategy_id.


import warnings

from sqlmodel import Session as _Session
from sqlmodel import select

from hlanalysis.engine.state import (
    CoinKlass,
    Event,
    Fill,
    PmStrike,
    Position,
    SeenQuestion,
    Settlement,
    TradeJournalRow,
)


def _make_dal(tmp_path):
    db = tmp_path / "orm.db"
    dal = StateDAL(db)
    dal.run_migrations()
    return dal


def test_no_sa_warning_on_orm_ops_after_migration(tmp_path):
    """ORM operations on a FULLY MIGRATED DB must emit no SAWarning about PK
    mismatch.

    Note: the Alembic batch-alter step in 0006 necessarily fires SAWarning
    during the migration itself (the temp-table is created against the OLD
    single-PK schema before the PK change lands).  That warning is an
    inherent Alembic batch-alter side effect and cannot be suppressed at the
    ORM layer.

    What CAN be verified: after migration is complete, a fresh run_migrations()
    call (idempotent — already at head) and normal ORM insert/query operations
    must not produce any PK-mismatch SAWarning.  Before Task 2, this test
    would have failed because the ORM model still declared the old single
    primary_key, so every subsequent operation that triggers SQLAlchemy
    reflection would repeat the warning.
    """
    import sqlalchemy.exc

    db = tmp_path / "warn.db"
    dal = StateDAL(db)
    dal.run_migrations()  # first run — the batch-alter SAWarnings happen here (expected)

    # Second run: already at head, no DDL executes — must be warning-free.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dal.run_migrations()

    sa_pk_warnings = [
        x
        for x in w
        if issubclass(x.category, sqlalchemy.exc.SAWarning)
        and "primary_key" in str(x.message).lower()
        and "not matching" in str(x.message).lower()
    ]
    assert sa_pk_warnings == [], (
        f"Got {len(sa_pk_warnings)} SAWarning(s) about PK mismatch on idempotent run:\n"
        + "\n".join(str(x.message) for x in sa_pk_warnings)
    )

    # ORM insert/query on fully-migrated DB must also be warning-free.
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        p = Position(
            strategy_id="v1",
            question_idx=9999,
            symbol="@30",
            qty=1.0,
            avg_entry=0.85,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=0.75,
            closed_qty=0.0,
        )
        with _Session(dal._engine) as s:
            s.add(p)
            s.commit()
        with _Session(dal._engine) as s:
            rows = list(s.exec(select(Position).where(Position.question_idx == 9999)).all())
        assert len(rows) == 1

    orm_pk_warns = [
        x
        for x in w2
        if issubclass(x.category, sqlalchemy.exc.SAWarning)
        and "primary_key" in str(x.message).lower()
        and "not matching" in str(x.message).lower()
    ]
    assert orm_pk_warns == [], f"Got {len(orm_pk_warns)} SAWarning(s) during ORM ops on migrated DB:\n" + "\n".join(
        str(x.message) for x in orm_pk_warns
    )


def test_position_orm_composite_pk_roundtrip(tmp_path):
    """ORM: two Position rows with same question_idx, different strategy_id,
    both persist and are independently readable via s.get()."""
    dal = _make_dal(tmp_path)

    p1 = Position(
        strategy_id="v31",
        account="w1",
        question_idx=2001,
        symbol="@30",
        qty=5.0,
        avg_entry=0.85,
        realized_pnl=0.0,
        last_update_ts_ns=1_000_000_000,
        stop_loss_price=0.75,
        closed_qty=0.0,
    )
    p2 = Position(
        strategy_id="v1",
        account="w1",
        question_idx=2001,
        symbol="@30",
        qty=3.0,
        avg_entry=0.90,
        realized_pnl=0.0,
        last_update_ts_ns=1_000_000_001,
        stop_loss_price=0.80,
        closed_qty=0.0,
    )

    with _Session(dal._engine) as s:
        s.add(p1)
        s.add(p2)
        s.commit()

    with _Session(dal._engine) as s:
        rows = list(s.exec(select(Position).where(Position.question_idx == 2001)).all())
    assert len(rows) == 2
    strat_ids = {r.strategy_id for r in rows}
    assert strat_ids == {"v1", "v31"}


def test_pm_strike_orm_has_strategy_id_account(tmp_path):
    """ORM: PmStrike accepts strategy_id and account fields."""
    dal = _make_dal(tmp_path)

    ps = PmStrike(
        strategy_id="v31",
        account="w1",
        question_idx=3001,
        strike=98_000.0,
    )
    with _Session(dal._engine) as s:
        s.add(ps)
        s.commit()

    with _Session(dal._engine) as s:
        rows = list(s.exec(select(PmStrike).where(PmStrike.question_idx == 3001)).all())
    assert len(rows) == 1
    assert rows[0].strategy_id == "v31"
    assert rows[0].account == "w1"
    assert rows[0].strike == 98_000.0


def test_settlement_orm_has_strategy_id_account(tmp_path):
    """ORM: Settlement accepts strategy_id and account fields."""
    dal = _make_dal(tmp_path)

    st = Settlement(
        strategy_id="v1",
        account="w2",
        question_idx=4001,
        symbol="@30",
        realized_pnl=12.5,
        ts_ns=2_000_000_000,
    )
    with _Session(dal._engine) as s:
        s.add(st)
        s.commit()

    with _Session(dal._engine) as s:
        rows = list(s.exec(select(Settlement).where(Settlement.question_idx == 4001)).all())
    assert len(rows) == 1
    assert rows[0].strategy_id == "v1"
    assert rows[0].account == "w2"
    assert rows[0].realized_pnl == 12.5


def test_fill_orm_has_strategy_id_account(tmp_path):
    """ORM: Fill accepts nullable strategy_id and account fields."""
    dal = _make_dal(tmp_path)

    f = Fill(
        fill_id="f-orm-1",
        cloid="c-orm-1",
        question_idx=5001,
        symbol="@30",
        side="buy",
        price=0.85,
        size=10.0,
        fee=0.01,
        ts_ns=3_000_000_000,
        closed_pnl=0.0,
        source="router",
        strategy_id="v31",
        account="w1",
    )
    with _Session(dal._engine) as s:
        s.add(f)
        s.commit()

    with _Session(dal._engine) as s:
        row = s.get(Fill, "f-orm-1")
    assert row is not None
    assert row.strategy_id == "v31"
    assert row.account == "w1"


def test_trade_journal_row_orm_has_strategy_id_account(tmp_path):
    """ORM: TradeJournalRow accepts nullable strategy_id and account fields."""
    dal = _make_dal(tmp_path)

    tj = TradeJournalRow(
        cloid="tj-orm-1",
        question_idx=6001,
        decision_ts_ns=4_000_000_000,
        action="enter",
        strategy_id="v31",
        account="w1",
    )
    with _Session(dal._engine) as s:
        s.add(tj)
        s.commit()

    with _Session(dal._engine) as s:
        row = s.get(TradeJournalRow, "tj-orm-1")
    assert row is not None
    assert row.strategy_id == "v31"
    assert row.account == "w1"


def test_event_orm_has_strategy_id(tmp_path):
    """ORM: Event accepts nullable strategy_id field."""
    dal = _make_dal(tmp_path)

    ev = Event(
        ts_ns=5_000_000_000,
        alias="v31",
        kind="entry",
        question_idx=7001,
        strategy_id="v31",
    )
    with _Session(dal._engine) as s:
        s.add(ev)
        s.commit()

    with _Session(dal._engine) as s:
        rows = list(s.exec(select(Event).where(Event.question_idx == 7001)).all())
    assert len(rows) == 1
    assert rows[0].strategy_id == "v31"


def test_seen_question_orm_has_strategy_id_account(tmp_path):
    """ORM: SeenQuestion accepts strategy_id and account fields."""
    dal = _make_dal(tmp_path)

    sq = SeenQuestion(
        strategy_id="v1",
        account="w1",
        question_idx=8001,
        first_seen_ts_ns=6_000_000_000,
    )
    with _Session(dal._engine) as s:
        s.add(sq)
        s.commit()

    with _Session(dal._engine) as s:
        rows = list(s.exec(select(SeenQuestion).where(SeenQuestion.question_idx == 8001)).all())
    assert len(rows) == 1
    assert rows[0].strategy_id == "v1"
    assert rows[0].account == "w1"


def test_coin_klass_orm_has_strategy_id_account(tmp_path):
    """ORM: CoinKlass accepts strategy_id and account fields."""
    dal = _make_dal(tmp_path)

    ck = CoinKlass(
        strategy_id="v31",
        account="w1",
        coin="#10",
        klass="priceBinary",
        question_idx=9001,
    )
    with _Session(dal._engine) as s:
        s.add(ck)
        s.commit()

    with _Session(dal._engine) as s:
        rows = list(s.exec(select(CoinKlass).where(CoinKlass.coin == "#10")).all())
    assert len(rows) == 1
    assert rows[0].strategy_id == "v31"
    assert rows[0].account == "w1"
