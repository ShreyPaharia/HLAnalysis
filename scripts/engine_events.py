#!/usr/bin/env python3
"""Print the event trace for one question_idx across every per-slot engine
state DB. Invoked by `make engine-events Q=<question_idx>` over SSM.

Why this exists as a script rather than an inline `sqlite3` call: the live box
has NO `sqlite3` CLI, so the previous Makefile one-liner silently failed. The
engine venv always has Python's stdlib `sqlite3`, so we query through that
instead. Read-only; safe to run against the live DBs.

Unified-DB awareness: if `<root>/state.db` is the shared multi-slot DB
(detected by the presence of a `strategy_id` column on the `events` table),
it is queried once with an optional `--strategy <id>` filter. Otherwise the
legacy multi-DB layout (`<root>/*/state.db` + flat `<root>/state.db`) is used.

Usage:
    python scripts/engine_events.py <question_idx> [--data-root DIR]
        [--strategy STRATEGY_ID]
"""

from __future__ import annotations

import argparse
import glob
import sqlite3
from pathlib import Path

_DEFAULT_ROOT = "/opt/hl-recorder/data/engine"


def _has_strategy_id_column(db_path: str) -> bool:
    """Return True if the `events` table has a `strategy_id` column.

    Used to detect a unified (multi-slot) DB vs a legacy per-slot DB.
    """
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = con.execute("PRAGMA table_info(events)").fetchall()
        con.close()
    except sqlite3.Error:
        return False
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == "strategy_id" for row in rows)


def _query_unified_db(db_path: str, question_idx: int, strategy_id: str | None) -> bool:
    """Query the unified DB for events matching question_idx (+ optional strategy).

    Returns True if any events were found.
    """
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        if strategy_id is not None:
            rows = con.execute(
                "SELECT ts_ns, alias, strategy_id, kind, reason, payload_json FROM events "
                "WHERE question_idx=? AND strategy_id=? ORDER BY ts_ns ASC",
                (question_idx, strategy_id),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT ts_ns, alias, strategy_id, kind, reason, payload_json FROM events "
                "WHERE question_idx=? ORDER BY ts_ns ASC",
                (question_idx,),
            ).fetchall()
        con.close()
    except sqlite3.Error as e:
        print(f"== unified == ERROR: {e}")
        return False

    if not rows:
        return False

    label = f"unified (strategy={strategy_id})" if strategy_id else "unified (all slots)"
    print(f"== {label} ({len(rows)} events) ==")
    for ts_ns, ev_alias, ev_strategy_id, kind, reason, payload in rows:
        slot_tag = ev_strategy_id or ev_alias or ""
        print(f"  {ts_ns}  {slot_tag}  {kind}  {reason or ''}  {payload or ''}")
    return True


def _query_legacy_dbs(dbs: list[str], question_idx: int) -> bool:
    """Query legacy per-slot DBs for events matching question_idx.

    Returns True if any events were found.
    """
    found = False
    for db in dbs:
        alias = Path(db).parent.name
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            rows = con.execute(
                "SELECT ts_ns, alias, kind, reason, payload_json FROM events WHERE question_idx=? ORDER BY ts_ns ASC",
                (question_idx,),
            ).fetchall()
            con.close()
        except sqlite3.Error as e:
            print(f"== {alias} == ERROR: {e}")
            continue
        if not rows:
            continue
        found = True
        print(f"== {alias} ({len(rows)} events) ==")
        for ts_ns, ev_alias, kind, reason, payload in rows:
            print(f"  {ts_ns}  {ev_alias}  {kind}  {reason or ''}  {payload or ''}")
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question_idx", type=int)
    ap.add_argument("--data-root", default=_DEFAULT_ROOT, help=f"engine state dir (default: {_DEFAULT_ROOT})")
    ap.add_argument(
        "--strategy", default=None, help="filter by strategy_id (unified DB only; ignored for legacy layout)"
    )
    args = ap.parse_args(argv)

    root = Path(args.data_root)
    flat_db = root / "state.db"

    # -- unified-DB path: single <root>/state.db with strategy_id column ------
    if flat_db.exists() and _has_strategy_id_column(str(flat_db)):
        found = _query_unified_db(str(flat_db), args.question_idx, args.strategy)
        if not found:
            filter_note = f" strategy={args.strategy}" if args.strategy else ""
            print(f"no events for question_idx={args.question_idx}{filter_note}")
        return 0

    # -- legacy path: per-slot DBs (<root>/<alias>/state.db) + flat fallback --
    dbs = sorted(glob.glob(str(root / "*" / "state.db")))
    if flat_db.exists():
        dbs.append(str(flat_db))

    if not dbs:
        print(f"no state DBs under {args.data_root}")
        return 0

    found = _query_legacy_dbs(dbs, args.question_idx)
    if not found:
        print(f"no events for question_idx={args.question_idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
