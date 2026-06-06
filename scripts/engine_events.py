#!/usr/bin/env python3
"""Print the event trace for one question_idx across every per-slot engine
state DB. Invoked by `make engine-events Q=<question_idx>` over SSM.

Why this exists as a script rather than an inline `sqlite3` call: the live box
has NO `sqlite3` CLI, so the previous Makefile one-liner silently failed. The
engine venv always has Python's stdlib `sqlite3`, so we query through that
instead. Read-only; safe to run against the live DBs.

Usage:
    python scripts/engine_events.py <question_idx> [--data-root DIR]
"""
from __future__ import annotations

import argparse
import glob
import sqlite3
from pathlib import Path

_DEFAULT_ROOT = "/opt/hl-recorder/data/engine"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question_idx", type=int)
    ap.add_argument("--data-root", default=_DEFAULT_ROOT,
                    help=f"engine state dir (default: {_DEFAULT_ROOT})")
    args = ap.parse_args(argv)

    # Per-slot DBs (<root>/<alias>/state.db) plus the legacy flat single-account
    # path (<root>/state.db), matching deploy_cfg.state_db_path_for().
    dbs = sorted(glob.glob(str(Path(args.data_root) / "*" / "state.db")))
    flat = Path(args.data_root) / "state.db"
    if flat.exists():
        dbs.append(str(flat))

    if not dbs:
        print(f"no state DBs under {args.data_root}")
        return 0

    found = False
    for db in dbs:
        alias = Path(db).parent.name
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            rows = con.execute(
                "SELECT ts_ns, alias, kind, reason, payload_json FROM events "
                "WHERE question_idx=? ORDER BY ts_ns ASC",
                (args.question_idx,),
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

    if not found:
        print(f"no events for question_idx={args.question_idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
