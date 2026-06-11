"""Unit tests for scripts/inspect_engine_state.py — the local SSM-free reader for
pulled engine snapshots. Covers snapshot resolution, the gzip→read-only open, and
the PnL/row-count summary against a synthetic state.db matching the engine schema.
"""
from __future__ import annotations

import gzip
import shutil
import sqlite3
from pathlib import Path

import pytest

from scripts.inspect_engine_state import (
    open_snapshot,
    resolve_snapshot,
    summarize,
)


def _make_state_db(path: Path) -> None:
    """A minimal state.db with the fill/position/settlement columns the engine
    writes (hlanalysis/engine/state.py)."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE fill (fill_id TEXT PRIMARY KEY, cloid TEXT, question_idx INT,
            symbol TEXT, side TEXT, price REAL, size REAL, fee REAL, ts_ns INT,
            closed_pnl REAL, source TEXT);
        CREATE TABLE position (question_idx INT PRIMARY KEY, symbol TEXT, qty REAL,
            avg_entry REAL, realized_pnl REAL, last_update_ts_ns INT,
            stop_loss_price REAL, closed_qty REAL);
        CREATE TABLE settlement (question_idx INT PRIMARY KEY, symbol TEXT,
            realized_pnl REAL, ts_ns INT);
        INSERT INTO fill VALUES ('f1','c1',1,'BTC','buy',0.40,100,0,1,0.0,'router');
        INSERT INTO fill VALUES ('f2','c1',1,'BTC','sell',0.55,100,0,2,15.0,'router');
        INSERT INTO position VALUES (2,'ETH',50,0.30,0.0,3,0.10,0.0);
        INSERT INTO settlement VALUES (1,'BTC',7.5,4);
        """
    )
    conn.commit()
    conn.close()


def _make_snapshot_tree(root: Path, date: str, alias: str) -> Path:
    """Lay out <root>/date=<date>/<alias>/state.db.gz and return its path."""
    slot = root / f"date={date}" / alias
    slot.mkdir(parents=True)
    raw = slot / "state.db"
    _make_state_db(raw)
    gz = slot / "state.db.gz"
    with open(raw, "rb") as fin, gzip.open(gz, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    raw.unlink()
    return gz


def test_resolve_snapshot_found(tmp_path):
    gz = _make_snapshot_tree(tmp_path, "2026-06-10", "v1")
    assert resolve_snapshot(tmp_path, "2026-06-10", "v1") == gz


def test_resolve_snapshot_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_snapshot(tmp_path, "2026-06-10", "nope")


def test_open_snapshot_is_readonly_and_loads(tmp_path):
    gz = _make_snapshot_tree(tmp_path, "2026-06-10", "v1")
    with open_snapshot(gz) as conn:
        assert conn.execute("SELECT COUNT(*) FROM fill").fetchone()[0] == 2
        # read-only handle: a write must be rejected
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO fill (fill_id) VALUES ('x')")


def test_summarize_counts_and_pnl(tmp_path):
    gz = _make_snapshot_tree(tmp_path, "2026-06-10", "v1")
    with open_snapshot(gz) as conn:
        s = summarize(conn)
    assert s["tables"] == {"fill": 2, "position": 1, "settlement": 1}
    assert s["fill_closed_pnl"] == pytest.approx(15.0)
    assert s["settlement_realized_pnl"] == pytest.approx(7.5)


def test_summarize_tolerates_missing_tables(tmp_path):
    # A fresh slot may only have a fill table; absent tables report None, no crash.
    slot = tmp_path / "date=2026-06-10" / "v1"
    slot.mkdir(parents=True)
    raw = slot / "state.db"
    conn = sqlite3.connect(raw)
    conn.execute("CREATE TABLE fill (fill_id TEXT PRIMARY KEY, closed_pnl REAL)")
    conn.commit()
    conn.close()
    gz = slot / "state.db.gz"
    with open(raw, "rb") as fin, gzip.open(gz, "wb") as fout:
        shutil.copyfileobj(fin, fout)

    with open_snapshot(gz) as c:
        s = summarize(c)
    assert s["tables"]["fill"] == 0
    assert s["tables"]["position"] is None
    assert s["tables"]["settlement"] is None
    assert "settlement_realized_pnl" not in s
