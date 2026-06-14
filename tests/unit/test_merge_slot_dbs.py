"""Tests for scripts/merge_slot_dbs.py.

Strategy:
* Build 2 source slot DBs (via StateDAL + raw SQL for pre-migration-style rows).
* Run merge() into a fresh unified DB.
* Assert counts, scoping columns, composite-PK coexistence, and idempotency.
* Assert the script exits with non-zero when the source dir has no DBs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hlanalysis.engine.state import StateDAL
from scripts.merge_slot_dbs import _discover_sources, merge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_count(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as con:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _all_rows(db_path: Path, table: str) -> list[dict]:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        return [dict(r) for r in con.execute(f"SELECT * FROM {table}").fetchall()]


def _seed_source_db(db_path: Path, alias: str, question_idxs: list[int]) -> None:
    """Create a migrated source DB and seed it with position + fill + events rows.

    The rows are inserted via raw SQL with strategy_id='' (the value that
    migration 0006 backfills on pre-merge per-slot rows) to simulate a
    real per-slot state.db just after run_migrations() is applied on it.
    """
    dal = StateDAL(db_path)
    dal.run_migrations()

    con = sqlite3.connect(db_path)
    for qidx in question_idxs:
        # Position row — strategy_id='' matches the 0006 backfill default.
        con.execute(
            "INSERT OR IGNORE INTO position "
            "(strategy_id, question_idx, symbol, qty, avg_entry, realized_pnl, "
            "last_update_ts_ns, stop_loss_price, closed_qty, account) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("", qidx, f"@{qidx}", 5.0, 0.90, 0.0, 1_000_000, 0.81, 0.0, None),
        )
        # Fill row — fill_id must be unique across slots, so include alias.
        con.execute(
            "INSERT OR IGNORE INTO fill "
            "(fill_id, cloid, question_idx, symbol, side, price, size, fee, ts_ns, "
            "closed_pnl, source, strategy_id, account) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"f-{alias}-{qidx}",
                f"cloid-{alias}-{qidx}",
                qidx,
                f"@{qidx}",
                "buy",
                0.90,
                5.0,
                0.05,
                2_000_000,
                0.0,
                "router",
                "",  # strategy_id blank, like a post-0006-stamp per-slot row
                None,
            ),
        )
        # Settlement row.
        con.execute(
            "INSERT OR IGNORE INTO settlement "
            "(strategy_id, question_idx, symbol, realized_pnl, ts_ns, account) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("", qidx, f"@{qidx}", 1.23, 3_000_000, None),
        )
        # Event row — alias column matches the slot name; strategy_id='' pre-merge.
        con.execute(
            "INSERT INTO events (ts_ns, alias, kind, question_idx, reason, payload_json, strategy_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (4_000_000, alias, "entry", qidx, None, None, ""),
        )
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def src_root(tmp_path: Path) -> Path:
    root = tmp_path / "engine"
    root.mkdir()
    return root


@pytest.fixture()
def two_slot_src(src_root: Path) -> tuple[Path, Path, Path]:
    """Two seeded slot DBs under src_root. Returns (src_root, db_v1, db_v31)."""
    slot_v1 = src_root / "v1"
    slot_v1.mkdir()
    db_v1 = slot_v1 / "state.db"
    _seed_source_db(db_v1, "v1", question_idxs=[100, 101])

    slot_v31 = src_root / "v31"
    slot_v31.mkdir()
    db_v31 = slot_v31 / "state.db"
    _seed_source_db(db_v31, "v31", question_idxs=[100, 200])  # q100 shared across slots

    return src_root, db_v1, db_v31


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_merge_counts_sum_sources(two_slot_src: tuple, tmp_path: Path) -> None:
    """dest row count per table == sum of source counts."""
    src_root, db_v1, db_v31 = two_slot_src
    out = tmp_path / "unified.db"

    rc = merge(
        src_root=src_root,
        out_path=out,
        alias_to_account={},
        aliases=None,
    )
    assert rc == 0, "merge() must return 0 on success"

    # v1 has q100+q101 (2 rows each table), v31 has q100+q200 (2 rows each).
    # position: composite PK is (strategy_id, question_idx). After merge,
    # strategy_id='v1'/q100, 'v1'/q101, 'v31'/q100, 'v31'/q200 → 4 rows total.
    assert _row_count(out, "position") == 4
    assert _row_count(out, "fill") == 4  # fill_id is unique per alias+qidx
    assert _row_count(out, "settlement") == 4
    assert _row_count(out, "events") == 4


def test_merge_strategy_id_and_account_stamped(two_slot_src: tuple, tmp_path: Path) -> None:
    """Every merged row carries strategy_id == alias and account == alias (default)."""
    src_root, _, _ = two_slot_src
    out = tmp_path / "unified.db"

    merge(src_root=src_root, out_path=out, alias_to_account={}, aliases=None)

    positions = _all_rows(out, "position")
    strategy_ids = {r["strategy_id"] for r in positions}
    assert strategy_ids == {"v1", "v31"}

    # Default account == alias.
    accounts = {r["account"] for r in positions}
    assert accounts == {"v1", "v31"}

    fills = _all_rows(out, "fill")
    assert all(r["strategy_id"] in ("v1", "v31") for r in fills)
    assert all(r["account"] in ("v1", "v31") for r in fills)

    events = _all_rows(out, "events")
    assert all(r["strategy_id"] in ("v1", "v31") for r in events)


def test_merge_custom_account_map(two_slot_src: tuple, tmp_path: Path) -> None:
    """--map overrides the account for a given alias."""
    src_root, _, _ = two_slot_src
    out = tmp_path / "unified.db"

    merge(
        src_root=src_root,
        out_path=out,
        alias_to_account={"v1": "0xABC", "v31": "0xDEF"},
        aliases=None,
    )

    positions = _all_rows(out, "position")
    by_strategy = {r["strategy_id"]: r["account"] for r in positions}
    # Every v1 position should have account=0xABC, v31 → 0xDEF.
    assert by_strategy["v1"] == "0xABC"
    assert by_strategy["v31"] == "0xDEF"


def test_merge_composite_pk_coexists(two_slot_src: tuple, tmp_path: Path) -> None:
    """The same question_idx held by two slots coexists under different strategy_ids."""
    src_root, _, _ = two_slot_src
    out = tmp_path / "unified.db"

    merge(src_root=src_root, out_path=out, alias_to_account={}, aliases=None)

    # question_idx=100 exists in BOTH v1 and v31 — both rows must be present.
    rows = _all_rows(out, "position")
    q100_rows = [r for r in rows if r["question_idx"] == 100]
    assert len(q100_rows) == 2
    strats = {r["strategy_id"] for r in q100_rows}
    assert strats == {"v1", "v31"}


def test_merge_idempotent(two_slot_src: tuple, tmp_path: Path) -> None:
    """Running merge twice leaves row counts unchanged (INSERT OR IGNORE)."""
    src_root, _, _ = two_slot_src
    out = tmp_path / "unified.db"

    merge(src_root=src_root, out_path=out, alias_to_account={}, aliases=None)
    counts_after_first = {tbl: _row_count(out, tbl) for tbl in ("position", "fill", "settlement", "events")}

    merge(src_root=src_root, out_path=out, alias_to_account={}, aliases=None)
    counts_after_second = {tbl: _row_count(out, tbl) for tbl in ("position", "fill", "settlement", "events")}

    assert counts_after_first == counts_after_second


def test_merge_aliases_filter(two_slot_src: tuple, tmp_path: Path) -> None:
    """--aliases restricts the merge to the named slots only."""
    src_root, _, _ = two_slot_src
    out = tmp_path / "unified.db"

    rc = merge(
        src_root=src_root,
        out_path=out,
        alias_to_account={},
        aliases=["v1"],
    )
    assert rc == 0

    # Only v1's 2 rows should be present.
    assert _row_count(out, "position") == 2
    positions = _all_rows(out, "position")
    assert all(r["strategy_id"] == "v1" for r in positions)


def test_merge_empty_src_returns_error(tmp_path: Path) -> None:
    """merge() returns 1 and does not crash when src_root has no DBs."""
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    out = tmp_path / "unified.db"

    rc = merge(
        src_root=empty_root,
        out_path=out,
        alias_to_account={},
        aliases=None,
    )
    assert rc == 1


def test_merge_legacy_flat_db(tmp_path: Path) -> None:
    """Legacy flat <root>/state.db is discovered as alias _root."""
    src_root = tmp_path / "engine"
    src_root.mkdir()

    flat_db = src_root / "state.db"
    _seed_source_db(flat_db, "_root", question_idxs=[999])

    out = tmp_path / "unified.db"
    rc = merge(
        src_root=src_root,
        out_path=out,
        alias_to_account={"_root": "legacy_account"},
        aliases=None,
    )
    assert rc == 0

    positions = _all_rows(out, "position")
    assert len(positions) == 1
    assert positions[0]["strategy_id"] == "_root"
    assert positions[0]["account"] == "legacy_account"


def test_discover_sources_no_dbs(tmp_path: Path) -> None:
    """_discover_sources returns empty list when no state.db files exist."""
    root = tmp_path / "engine"
    root.mkdir()
    # Create a dir without state.db.
    (root / "empty_slot").mkdir()

    sources = _discover_sources(root, None)
    assert sources == []


def test_merge_missing_table_in_source_tolerated(tmp_path: Path) -> None:
    """A source slot whose DB lacks a particular table (e.g. fresh slot with no
    settlements yet) is handled gracefully — the merge continues."""
    src_root = tmp_path / "engine"
    src_root.mkdir()

    slot_dir = src_root / "v1"
    slot_dir.mkdir()
    db_path = slot_dir / "state.db"

    # Build a migrated DB but do NOT insert any settlement rows.
    dal = StateDAL(db_path)
    dal.run_migrations()
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO position "
        "(strategy_id, question_idx, symbol, qty, avg_entry, realized_pnl, "
        "last_update_ts_ns, stop_loss_price, closed_qty, account) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("", 42, "@42", 1.0, 0.5, 0.0, 1, 0.45, 0.0, None),
    )
    con.commit()
    con.close()

    out = tmp_path / "unified.db"
    rc = merge(src_root=src_root, out_path=out, alias_to_account={}, aliases=None)
    assert rc == 0

    assert _row_count(out, "position") == 1
    assert _row_count(out, "settlement") == 0  # no settlement rows seeded
