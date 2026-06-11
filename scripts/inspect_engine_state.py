#!/usr/bin/env python3
"""Open a pulled engine ``state.db.gz`` snapshot and print fills / positions /
settlements for a slot+date — so the local live-vs-sim analysis loop needs ZERO
SSM roundtrips. Counterpart to ``scripts/pull-engine.sh``.

Snapshots land at ``<root>/date=YYYY-MM-DD/<alias>/state.db.gz`` (see
``scripts/sync-engine-to-s3.sh``). This decompresses the gzip to a temp file,
opens it read-only, and summarises the durable tables the engine writes.

    scripts/inspect_engine_state.py ./data/engine --date 2026-06-10 --alias v1
    scripts/inspect_engine_state.py ./data/engine --date 2026-06-10 --alias v1 --json

Importable: ``resolve_snapshot``, ``open_snapshot`` (a context manager yielding a
read-only ``sqlite3.Connection``), and ``summarize`` are unit-tested.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Iterator

# Tables the engine persists (hlanalysis/engine/state.py). Missing tables are
# tolerated (a fresh slot may not have settled yet).
_SUMMARY_TABLES = ("fill", "position", "settlement")


def resolve_snapshot(root: str | Path, date: str, alias: str) -> Path:
    """Return the path to ``<root>/date=<date>/<alias>/state.db.gz``.

    Raises FileNotFoundError with the expected location if it is not present, so
    the operator immediately sees whether the pull ran / the slot/date exist.
    """
    path = Path(root) / f"date={date}" / alias / "state.db.gz"
    if not path.is_file():
        raise FileNotFoundError(f"no engine snapshot at {path}")
    return path


@contextlib.contextmanager
def open_snapshot(gz_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Decompress a ``state.db.gz`` to a temp file and yield a read-only handle.

    The temp file is cleaned up on exit. Read-only (``mode=ro``) so we can never
    mutate an analysis copy.
    """
    gz_path = Path(gz_path)
    tmpdir = tempfile.mkdtemp(prefix="engine-snap-")
    db_path = Path(tmpdir) / "state.db"
    try:
        with gzip.open(gz_path, "rb") as fin, open(db_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def summarize(conn: sqlite3.Connection) -> dict:
    """Return a small summary dict: per-table row counts + headline figures.

    Realized PnL is summed from the durable ledgers the engine writes
    (``fill.closed_pnl`` for trade PnL, ``settlement.realized_pnl`` for HIP-4
    settlement payouts) — the same columns the daily-loss gate reads.
    """
    out: dict = {"tables": {}}
    for table in _SUMMARY_TABLES:
        out["tables"][table] = (
            conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if _table_exists(conn, table)
            else None
        )
    if _table_exists(conn, "fill"):
        out["fill_closed_pnl"] = (
            conn.execute("SELECT COALESCE(SUM(closed_pnl), 0.0) FROM fill").fetchone()[0]
        )
    if _table_exists(conn, "settlement"):
        out["settlement_realized_pnl"] = (
            conn.execute(
                "SELECT COALESCE(SUM(realized_pnl), 0.0) FROM settlement"
            ).fetchone()[0]
        )
    return out


def _print_human(conn: sqlite3.Connection, summary: dict) -> None:
    print("== summary ==")
    for table, count in summary["tables"].items():
        print(f"  {table:<12} rows={count if count is not None else '(absent)'}")
    if "fill_closed_pnl" in summary:
        print(f"  fill closed_pnl total     = {summary['fill_closed_pnl']:+.4f}")
    if "settlement_realized_pnl" in summary:
        print(f"  settlement realized_pnl   = {summary['settlement_realized_pnl']:+.4f}")

    if _table_exists(conn, "position"):
        rows = conn.execute(
            "SELECT question_idx, symbol, qty, avg_entry, realized_pnl "
            "FROM position WHERE qty != 0 ORDER BY question_idx"
        ).fetchall()
        if rows:
            print("\n== open positions ==")
            for r in rows:
                print(
                    f"  q{r['question_idx']:<6} {r['symbol']:<10} "
                    f"qty={r['qty']:+.4f} entry={r['avg_entry']:.4f} "
                    f"realized={r['realized_pnl']:+.4f}"
                )

    if _table_exists(conn, "fill"):
        rows = conn.execute(
            "SELECT ts_ns, symbol, side, price, size, closed_pnl, source "
            "FROM fill ORDER BY ts_ns DESC LIMIT 20"
        ).fetchall()
        if rows:
            print("\n== last 20 fills ==")
            for r in rows:
                print(
                    f"  {r['ts_ns']} {r['symbol']:<10} {r['side']:<4} "
                    f"px={r['price']:.4f} sz={r['size']:.4f} "
                    f"pnl={r['closed_pnl']:+.4f} [{r['source']}]"
                )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="engine snapshot root (e.g. ./data/engine)")
    ap.add_argument("--date", required=True, help="snapshot date, YYYY-MM-DD")
    ap.add_argument("--alias", required=True, help="slot alias, e.g. v1 / v31_pm")
    ap.add_argument("--json", action="store_true", help="emit the summary as JSON")
    args = ap.parse_args(argv)

    try:
        gz_path = resolve_snapshot(args.root, args.date, args.alias)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    with open_snapshot(gz_path) as conn:
        summary = summarize(conn)
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"snapshot: {gz_path}")
            _print_human(conn, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
