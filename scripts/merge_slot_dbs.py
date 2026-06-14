#!/usr/bin/env python3
"""One-time offline migration: merge per-slot engine state DBs into a unified DB.

Run this ONCE with the engine stopped, during cutover to the shared-DB layout
(Task 7 of 9 in the unified-slot-db epic). It is safe to re-run: INSERT OR IGNORE
prevents duplicate rows — the composite PKs (strategy_id + question_idx / coin /
fill_id / cloid) ensure idempotency across slots and re-runs.

Source layout (per-slot, pre-unification):
    <engine_root>/<alias>/state.db   e.g. /opt/hl-recorder/data/engine/v1/state.db
    <engine_root>/state.db           legacy flat single-account path (alias = _root)

Target:
    A fresh or existing unified DB at --out, brought to the Alembic head schema
    (0006_unified_slot_db) via StateDAL.run_migrations() before any rows are copied.

Usage:
    python scripts/merge_slot_dbs.py \\
        --src /opt/hl-recorder/data/engine \\
        --out /opt/hl-recorder/data/engine/unified.db \\
        [--aliases v1,v31,v31_pm] \\
        [--map v1=0xABC,v31=0xDEF]
"""

from __future__ import annotations

import argparse
import datetime
import sqlite3
import sys
from pathlib import Path

# Bookkeeping tables that must NOT be copied (schema/migration metadata only,
# plus the internal merge-log table we add to the destination).
_SKIP_TABLES = frozenset({"alembic_version", "schema_migrations", "sqlite_sequence", "_merge_log"})

# Columns that are synthetic autoincrement PKs and must NOT be copied: we let
# the destination DB assign fresh IDs so rows from different source slots don't
# collide on their source-local id values. Currently only events.id is in this
# set — all other PKs are content-keyed (fill_id, cloid, question_idx, …).
_AUTOINCREMENT_PK_COLS: dict[str, str] = {"events": "id"}

# Internal merge-log table written to the destination to enable idempotency.
# Records (alias, src_path) pairs that have already been copied so a re-run
# with the same sources is a safe no-op. Not part of the engine schema — it
# is skipped during row-copy and treated as a bookkeeping table.
_MERGE_LOG_TABLE = "_merge_log"
_CREATE_MERGE_LOG = (
    f"CREATE TABLE IF NOT EXISTS {_MERGE_LOG_TABLE} "
    "(alias TEXT NOT NULL, src_path TEXT NOT NULL, merged_at TEXT NOT NULL, "
    "PRIMARY KEY (alias, src_path))"
)


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Return column names for *table* via PRAGMA table_info."""
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _data_tables(conn: sqlite3.Connection) -> list[str]:
    """Return all non-bookkeeping table names from *conn*."""
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    return [r[0] for r in rows if r[0] not in _SKIP_TABLES]


def _discover_sources(src_root: Path, aliases: list[str] | None) -> list[tuple[str, Path]]:
    """Return (alias, db_path) pairs for all discovered source DBs.

    Scans <src_root>/*/state.db (alias = parent dir name) and the legacy flat
    <src_root>/state.db (alias = '_root'). When *aliases* is non-empty, restricts
    to those aliases only.
    """
    sources: list[tuple[str, Path]] = []

    for slot_dir in sorted(src_root.iterdir()):
        if not slot_dir.is_dir():
            continue
        db = slot_dir / "state.db"
        if db.is_file():
            alias = slot_dir.name
            if aliases is None or alias in aliases:
                sources.append((alias, db))

    # Legacy flat single-account DB.
    flat = src_root / "state.db"
    if flat.is_file():
        if aliases is None or "_root" in aliases:
            sources.append(("_root", flat))

    return sources


def _merge_one(
    *,
    src_con: sqlite3.Connection,
    dst_con: sqlite3.Connection,
    alias: str,
    account: str,
) -> dict[str, int]:
    """Copy all data rows from *src_con* into *dst_con*, tagging each row with
    (strategy_id=alias, account=account) where those columns exist.

    Returns a dict of {table_name: rows_read} for the per-alias report.
    """
    rows_read: dict[str, int] = {}

    dst_tables_cols: dict[str, list[str]] = {}
    for row in dst_con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        tbl = row[0]
        if tbl not in _SKIP_TABLES:
            dst_tables_cols[tbl] = _table_columns(dst_con, tbl)

    for table in _data_tables(src_con):
        if table not in dst_tables_cols:
            # Destination schema doesn't have this table at all — skip.
            continue

        src_cols_all = _table_columns(src_con, table)
        dst_cols = dst_tables_cols[table]

        # Drop autoincrement PK columns (e.g. events.id): the destination
        # assigns fresh IDs so rows from different source slots don't collide
        # on their source-local integer ids.
        skip_src_col = _AUTOINCREMENT_PK_COLS.get(table)
        src_cols = [c for c in src_cols_all if c != skip_src_col]

        # Read all rows from the source (only the columns we'll copy).
        src_rows = src_con.execute(f"SELECT {', '.join(src_cols)} FROM {table}").fetchall()
        rows_read[table] = len(src_rows)

        if not src_rows:
            continue

        # Build the INSERT column list from the source columns, then augment
        # with strategy_id / account if the destination has them but the source
        # doesn't (i.e. we're copying from a pre-unification DB).
        insert_cols = list(src_cols)
        extra_vals: dict[str, str | None] = {}

        # Determine whether the destination expects strategy_id / account.
        has_strategy_id_dst = "strategy_id" in dst_cols
        has_account_dst = "account" in dst_cols
        has_strategy_id_src = "strategy_id" in src_cols
        has_account_src = "account" in src_cols

        if has_strategy_id_dst and not has_strategy_id_src:
            insert_cols.append("strategy_id")
            extra_vals["strategy_id"] = alias
        if has_account_dst and not has_account_src:
            insert_cols.append("account")
            extra_vals["account"] = account

        placeholders = ", ".join("?" for _ in insert_cols)
        sql = f"INSERT OR IGNORE INTO {table} ({', '.join(insert_cols)}) VALUES ({placeholders})"

        # Build a row list: original values + overrides / extras.
        out_rows: list[tuple] = []
        for row in src_rows:
            row_dict = dict(zip(src_cols, row))

            # Override strategy_id / account in-row if source already has them
            # (pre-unification rows are backfilled to '' by migration 0006).
            if has_strategy_id_dst and has_strategy_id_src:
                row_dict["strategy_id"] = alias
            if has_account_dst and has_account_src:
                row_dict["account"] = account

            # Assemble final tuple in insert_cols order.
            values: list = [row_dict[c] for c in src_cols]
            for c in extra_vals:
                values.append(extra_vals[c])
            out_rows.append(tuple(values))

        dst_con.executemany(sql, out_rows)

    dst_con.commit()
    return rows_read


def merge(
    *,
    src_root: Path,
    out_path: Path,
    alias_to_account: dict[str, str],
    aliases: list[str] | None,
) -> int:
    """Main merge logic. Returns 0 on success, 1 if any source DB failed to open."""
    # Lazy import to keep the script dependency-light; StateDAL only used to
    # create the schema in the destination, not to copy rows.
    from hlanalysis.engine.state import StateDAL  # noqa: PLC0415

    sources = _discover_sources(src_root, aliases)
    if not sources:
        print(
            f"ERROR: no source DBs found under {src_root}" + (f" (aliases={aliases})" if aliases else ""),
            file=sys.stderr,
        )
        return 1

    # Create / open destination and bring it to the Alembic head.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    StateDAL(out_path).run_migrations()

    dst_con = sqlite3.connect(out_path)
    dst_con.execute("PRAGMA journal_mode=WAL")

    # Create the internal merge-log table (idempotent).
    dst_con.execute(_CREATE_MERGE_LOG)
    dst_con.commit()

    # Per-alias and totals report.
    report: dict[str, dict[str, int]] = {}  # alias -> table -> rows_read
    failed: list[str] = []

    for alias, db_path in sources:
        src_key = str(db_path)
        # Idempotency check: skip source if already merged (keyed by alias + path).
        already = dst_con.execute(
            f"SELECT 1 FROM {_MERGE_LOG_TABLE} WHERE alias=? AND src_path=?",
            (alias, src_key),
        ).fetchone()
        if already:
            print(f"[{alias}] skipped (already merged)")
            continue

        account = alias_to_account.get(alias, alias)
        try:
            src_con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.OperationalError as exc:
            print(f"ERROR: cannot open {db_path}: {exc}", file=sys.stderr)
            failed.append(alias)
            continue

        try:
            rows_read = _merge_one(
                src_con=src_con,
                dst_con=dst_con,
                alias=alias,
                account=account,
            )
        finally:
            src_con.close()

        # Record this source in the merge log so re-runs skip it.
        dst_con.execute(
            f"INSERT OR IGNORE INTO {_MERGE_LOG_TABLE} (alias, src_path, merged_at) VALUES (?, ?, ?)",
            (alias, src_key, datetime.datetime.now(datetime.UTC).isoformat()),
        )
        dst_con.commit()

        report[alias] = rows_read
        print(f"[{alias}] -> account={account}")
        for table, n in sorted(rows_read.items()):
            print(f"  {table:<20} read={n}")

    dst_con.close()

    # Print totals.
    if report:
        all_tables = sorted({t for rows in report.values() for t in rows})
        dst_totals = _dest_counts(out_path, all_tables)

        print("\n=== totals ===")
        for table in all_tables:
            total_read = sum(r.get(table, 0) for r in report.values())
            dest_count = dst_totals.get(table, 0)
            print(f"  {table:<20} total_read={total_read}  dest_rows={dest_count}")

    if failed:
        print(f"\nERROR: {len(failed)} source DB(s) failed to open: {failed}", file=sys.stderr)
        return 1

    print("\nmerge complete.")
    return 0


def _dest_counts(path: Path, tables: list[str]) -> dict[str, int]:
    """Row counts for *tables* in the destination DB (post-merge)."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    counts: dict[str, int] = {}
    for table in tables:
        try:
            row = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            counts[table] = row[0] if row else 0
        except sqlite3.OperationalError:
            counts[table] = 0
    con.close()
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--src",
        required=True,
        help="Engine root directory (parent of per-slot subdirs, e.g. /opt/hl-recorder/data/engine)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Path for the unified output DB (created / migrated to 0006 head before copy)",
    )
    ap.add_argument(
        "--aliases",
        default=None,
        help="Comma-separated list of aliases to include (default: all discovered)",
    )
    ap.add_argument(
        "--map",
        default=None,
        help="Comma-separated alias=account pairs (default: account == alias)",
    )
    args = ap.parse_args(argv)

    aliases: list[str] | None = None
    if args.aliases:
        aliases = [a.strip() for a in args.aliases.split(",") if a.strip()]

    alias_to_account: dict[str, str] = {}
    if args.map:
        for item in args.map.split(","):
            item = item.strip()
            if "=" not in item:
                ap.error(f"--map item {item!r} must be alias=account")
            k, v = item.split("=", 1)
            alias_to_account[k.strip()] = v.strip()

    return merge(
        src_root=Path(args.src),
        out_path=Path(args.out),
        alias_to_account=alias_to_account,
        aliases=aliases,
    )


if __name__ == "__main__":
    raise SystemExit(main())
