"""Daily report — the two data-quality traps it must get right.

1. Live PnL must dedup the fill table's router+venue double-count (SHR-72): summing
   both sources doubles PnL. live_cell filters source='venue'.
2. sim_cell must parse total PnL + trades from an hl-bt report.md.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tools.daily_report import latest_settled_day, live_cell, sim_cell


def _make_state_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.execute(
        """CREATE TABLE fill (fill_id TEXT, cloid TEXT, question_idx INT, symbol TEXT,
           side TEXT, price REAL, size REAL, fee REAL, ts_ns INT, closed_pnl REAL, source TEXT)"""
    )
    # Same economic fill recorded twice (router ack + venue reconcile) — SHR-72.
    rows = [
        ("f1", "c1", 1, "#10", "buy", 0.90, 100.0, 0.0, 1000, 0.0, "router"),
        ("f1v", "c1", 1, "#10", "buy", 0.90, 100.0, 0.0, 1000, 0.0, "venue"),
        ("f2", "c2", 1, "#10", "sell", 0.95, 100.0, 0.0, 2000, 5.0, "router"),
        ("f2v", "c2", 1, "#10", "sell", 0.95, 100.0, 0.0, 2000, 5.0, "venue"),
        # an out-of-window venue fill that must be excluded
        ("f3v", "c3", 1, "#10", "buy", 0.50, 10.0, 0.0, 9999, 0.0, "venue"),
    ]
    con.executemany("INSERT INTO fill VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


def test_live_cell_dedups_router_venue(tmp_path):
    db = tmp_path / "state.db"
    _make_state_db(db)
    n, buy_ntl, pnl = live_cell(db, ("#10",), w0=0, w1=3000)
    # venue-only, in-window: 1 buy (100@0.90) + 1 sell → PnL = +5.0 (NOT doubled to 10)
    assert pnl == 5.0
    assert n == 2
    assert buy_ntl == 90.0  # 0.90 * 100, the venue buy only


def test_sim_cell_parses_report(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "report.md").write_text("# v3 run\n\n## Summary\n- questions: 1\n- trades: 74\n- total PnL: $5.18\n")
    n, _buy, pnl = sim_cell(run)
    assert n == 74
    assert pnl == 5.18


def test_sim_cell_missing_report_is_zero(tmp_path):
    run = tmp_path / "empty"
    run.mkdir()
    assert sim_cell(run) == (0, 0.0, 0.0)


def test_latest_settled_day_needs_consecutive_partitions(tmp_path):
    base = tmp_path / "venue=hyperliquid/product_type=prediction_binary/mechanism=clob/event=bbo/symbol=#1"
    for d in ("2026-06-09", "2026-06-10"):  # consecutive → 06-10 settles cleanly
        (base / f"date={d}" / "hour=all").mkdir(parents=True)
    assert latest_settled_day(tmp_path) == "2026-06-10"
