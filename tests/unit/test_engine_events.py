"""Tests for scripts/engine_events.py — unified-DB and legacy-DB layouts."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# Load the script as a module (it lives under scripts/, not hlanalysis/)
# ---------------------------------------------------------------------------

_SCRIPT = Path(__file__).parents[2] / "scripts" / "engine_events.py"


def _load_engine_events():
    spec = importlib.util.spec_from_file_location("engine_events", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Helpers: build minimal in-memory / on-disk state DBs
# ---------------------------------------------------------------------------


def _make_legacy_db(path: Path, alias: str, events: list[dict]) -> None:
    """Create a legacy per-slot state.db WITHOUT a strategy_id column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE IF NOT EXISTS events ("
        "id INTEGER PRIMARY KEY, ts_ns INTEGER, alias TEXT, kind TEXT,"
        " question_idx INTEGER, reason TEXT, payload_json TEXT)"
    )
    for ev in events:
        con.execute(
            "INSERT INTO events (ts_ns, alias, kind, question_idx, reason, payload_json) VALUES (?,?,?,?,?,?)",
            (
                ev["ts_ns"],
                ev.get("alias", alias),
                ev["kind"],
                ev.get("question_idx"),
                ev.get("reason"),
                ev.get("payload_json"),
            ),
        )
    con.commit()
    con.close()


def _make_unified_db(path: Path, events: list[dict]) -> None:
    """Create a unified state.db WITH a strategy_id column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE IF NOT EXISTS events ("
        "id INTEGER PRIMARY KEY, ts_ns INTEGER, alias TEXT, kind TEXT,"
        " question_idx INTEGER, reason TEXT, payload_json TEXT, strategy_id TEXT)"
    )
    for ev in events:
        con.execute(
            "INSERT INTO events (ts_ns, alias, kind, question_idx, reason, payload_json, strategy_id)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                ev["ts_ns"],
                ev.get("alias"),
                ev["kind"],
                ev.get("question_idx"),
                ev.get("reason"),
                ev.get("payload_json"),
                ev.get("strategy_id"),
            ),
        )
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# _has_strategy_id_column
# ---------------------------------------------------------------------------


def test_has_strategy_id_column_true(tmp_path):
    mod = _load_engine_events()
    db = tmp_path / "state.db"
    _make_unified_db(db, [])
    assert mod._has_strategy_id_column(str(db)) is True


def test_has_strategy_id_column_false_legacy(tmp_path):
    mod = _load_engine_events()
    db = tmp_path / "v1" / "state.db"
    _make_legacy_db(db, "v1", [])
    assert mod._has_strategy_id_column(str(db)) is False


def test_has_strategy_id_column_missing_file(tmp_path):
    mod = _load_engine_events()
    assert mod._has_strategy_id_column(str(tmp_path / "nonexistent.db")) is False


# ---------------------------------------------------------------------------
# main() — unified-DB path
# ---------------------------------------------------------------------------


def test_unified_db_shows_events_for_question(tmp_path, capsys):
    mod = _load_engine_events()
    db = tmp_path / "state.db"
    _make_unified_db(
        db,
        [
            {
                "ts_ns": 1000,
                "kind": "entry",
                "question_idx": 42,
                "strategy_id": "v1",
                "alias": "v1",
                "reason": None,
                "payload_json": '{"x":1}',
            },
            {
                "ts_ns": 2000,
                "kind": "exit",
                "question_idx": 42,
                "strategy_id": "v31",
                "alias": "v31",
                "reason": "edge",
                "payload_json": None,
            },
        ],
    )
    rc = mod.main(["42", "--data-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "entry" in out
    assert "exit" in out


def test_unified_db_strategy_filter(tmp_path, capsys):
    """--strategy filters to only that slot."""
    mod = _load_engine_events()
    db = tmp_path / "state.db"
    _make_unified_db(
        db,
        [
            {"ts_ns": 1000, "kind": "entry", "question_idx": 42, "strategy_id": "v1", "alias": "v1"},
            {"ts_ns": 2000, "kind": "exit", "question_idx": 42, "strategy_id": "v31", "alias": "v31"},
        ],
    )
    rc = mod.main(["42", "--data-root", str(tmp_path), "--strategy", "v1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "entry" in out
    assert "exit" not in out  # v31's exit must be excluded


def test_unified_db_no_events_for_question(tmp_path, capsys):
    mod = _load_engine_events()
    db = tmp_path / "state.db"
    _make_unified_db(
        db,
        [
            {"ts_ns": 1000, "kind": "entry", "question_idx": 99, "strategy_id": "v1", "alias": "v1"},
        ],
    )
    rc = mod.main(["42", "--data-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no events" in out


# ---------------------------------------------------------------------------
# main() — legacy-DB path (back-compat)
# ---------------------------------------------------------------------------


def test_legacy_per_slot_db(tmp_path, capsys):
    """Legacy layout: per-slot DBs, no strategy_id column."""
    mod = _load_engine_events()
    # Per-slot DBs (no top-level unified DB)
    v1_db = tmp_path / "v1" / "state.db"
    _make_legacy_db(
        v1_db,
        "v1",
        [
            {"ts_ns": 1000, "kind": "entry", "question_idx": 7, "alias": "v1"},
        ],
    )
    rc = mod.main(["7", "--data-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "entry" in out
    assert "v1" in out


def test_legacy_flat_root_db(tmp_path, capsys):
    """Legacy _root layout: flat <root>/state.db WITHOUT strategy_id column."""
    mod = _load_engine_events()
    flat_db = tmp_path / "state.db"
    _make_legacy_db(
        flat_db,
        "_root",
        [
            {"ts_ns": 5000, "kind": "entry", "question_idx": 5, "alias": "_root"},
        ],
    )
    # Manually verify that detection returns False so we exercise the legacy path
    assert mod._has_strategy_id_column(str(flat_db)) is False
    rc = mod.main(["5", "--data-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "entry" in out


def test_legacy_no_dbs(tmp_path, capsys):
    mod = _load_engine_events()
    rc = mod.main(["1", "--data-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no state DBs" in out


def test_legacy_no_events_for_question(tmp_path, capsys):
    mod = _load_engine_events()
    v1_db = tmp_path / "v1" / "state.db"
    _make_legacy_db(
        v1_db,
        "v1",
        [
            {"ts_ns": 1000, "kind": "entry", "question_idx": 99, "alias": "v1"},
        ],
    )
    rc = mod.main(["42", "--data-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no events" in out
