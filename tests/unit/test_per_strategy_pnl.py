"""Per-strategy realized-PnL isolation tests (Task 6).

Covers:
1. realized_pnl_since is isolated between two StrategyScopedDAL instances.
2. mirror_venue_fills via a scoped DAL stamps strategy_id on venue Fill rows so
   they appear in that strategy's realized_pnl_since but not the other's.
3. record_settlement via a scoped DAL is per-strategy (settlement_pnl_since
   isolation and realized_pnl_since inclusion).

This locks the correctness fix from the Task-6 spec: HL venue fills mirrored
through StrategyScopedDAL.mirror_venue_fills must carry strategy_id so the
daily-loss gate's DAL fallback path (slot.dal.realized_pnl_since) is never
blind to venue/settlement PnL.
"""

from __future__ import annotations

import sqlite3

from hlanalysis.engine.exec_types import UserFillRow
from hlanalysis.engine.scoped_dal import StrategyScopedDAL
from hlanalysis.engine.state import CachedStateDAL, Fill


def _make_shared(tmp_path) -> CachedStateDAL:
    dal = CachedStateDAL(tmp_path / "state.db")
    dal.run_migrations()
    return dal


def _venue_fill(
    fill_id: str, symbol: str = "#10", closed_pnl: float = 0.0, fee: float = 0.0, ts_ns: int = 1
) -> UserFillRow:
    return UserFillRow(
        fill_id=fill_id,
        cloid="",
        symbol=symbol,
        side="sell",
        price=0.9,
        size=10.0,
        fee=fee,
        ts_ns=ts_ns,
        closed_pnl=closed_pnl,
    )


# ---------------------------------------------------------------------------
# 1. realized_pnl_since is per-strategy
# ---------------------------------------------------------------------------


def test_realized_pnl_is_per_strategy(tmp_path):
    """Two StrategyScopedDALs on the same CachedStateDAL must have isolated
    realized_pnl_since: A's fills don't bleed into B's total and vice versa."""
    shared = _make_shared(tmp_path)
    a = StrategyScopedDAL(shared, strategy_id="A", account="0xACC")
    b = StrategyScopedDAL(shared, strategy_id="B", account="0xACC")

    # Strategy A: open fill (closed_pnl=0) + reducing fill (closed_pnl=+5)
    a.append_fill(
        Fill(
            fill_id="a-open",
            cloid="ca1",
            question_idx=1,
            symbol="#1",
            side="buy",
            price=0.80,
            size=10.0,
            fee=0.0,
            ts_ns=100,
            closed_pnl=0.0,
        )
    )
    a.append_fill(
        Fill(
            fill_id="a-reduce",
            cloid="ca2",
            question_idx=1,
            symbol="#1",
            side="sell",
            price=0.90,
            size=5.0,
            fee=0.0,
            ts_ns=200,
            closed_pnl=5.0,
        )
    )

    # Strategy B: losing fill (closed_pnl=-3)
    b.append_fill(
        Fill(
            fill_id="b-reduce",
            cloid="cb1",
            question_idx=2,
            symbol="#2",
            side="sell",
            price=0.70,
            size=5.0,
            fee=0.0,
            ts_ns=150,
            closed_pnl=-3.0,
        )
    )

    assert a.realized_pnl_since(0) == 5.0, "A must see only its own +5 PnL"
    assert b.realized_pnl_since(0) == -3.0, "B must see only its own -3 PnL"


# ---------------------------------------------------------------------------
# 2. mirror_venue_fills via scoped DAL stamps strategy_id
# ---------------------------------------------------------------------------


def test_mirrored_venue_fills_are_attributed(tmp_path):
    """Calling mirror_venue_fills on a StrategyScopedDAL must stamp strategy_id
    on each resulting Fill row, so they appear in that strategy's
    realized_pnl_since but not the other strategy's."""
    shared = _make_shared(tmp_path)
    a = StrategyScopedDAL(shared, strategy_id="A", account="0xACC")
    b = StrategyScopedDAL(shared, strategy_id="B", account="0xACC")

    # Mirror two venue fills through A's scoped DAL
    fills = [
        _venue_fill("tid-1", "#10", closed_pnl=8.0, fee=0.5, ts_ns=10),
        _venue_fill("tid-2", "#11", closed_pnl=3.0, fee=0.0, ts_ns=20),
    ]
    booked = a.mirror_venue_fills(fills)
    assert booked == 2, "Both '#' fills should be inserted"

    # Verify raw DB rows carry strategy_id="A"
    db_path = str(tmp_path / "state.db")
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT fill_id, strategy_id, account FROM fill WHERE fill_id IN ('tid-1','tid-2') ORDER BY fill_id"
        ).fetchall()
    assert len(rows) == 2, "Both rows should be in the DB"
    for fill_id, strategy_id, account in rows:
        assert strategy_id == "A", f"fill {fill_id} has strategy_id={strategy_id!r}, expected 'A'"
        assert account == "0xACC", f"fill {fill_id} has account={account!r}, expected '0xACC'"

    # A sees both fills in realized_pnl_since: (8.0 - 0.5) + (3.0 - 0.0) = 10.5
    assert a.realized_pnl_since(0) == 10.5

    # B sees none of them
    assert b.realized_pnl_since(0) == 0.0


def test_mirrored_venue_fills_idempotent(tmp_path):
    """Re-mirroring the same fills through the scoped DAL is idempotent (dedup by fill_id)."""
    shared = _make_shared(tmp_path)
    a = StrategyScopedDAL(shared, strategy_id="A", account="0xACC")

    fills = [_venue_fill("tid-x", "#10", closed_pnl=5.0)]
    assert a.mirror_venue_fills(fills) == 1  # first time: inserted
    assert a.mirror_venue_fills(fills) == 0  # second time: already exists
    assert a.realized_pnl_since(0) == 5.0  # counted exactly once


# ---------------------------------------------------------------------------
# 3. record_settlement is per-strategy
# ---------------------------------------------------------------------------


def test_settlement_pnl_per_strategy(tmp_path):
    """record_settlement via a scoped DAL is isolated: A's settlement is visible
    to A's settlement_pnl_since and realized_pnl_since but not B's."""
    shared = _make_shared(tmp_path)
    a = StrategyScopedDAL(shared, strategy_id="A", account="0xACC")
    b = StrategyScopedDAL(shared, strategy_id="B", account="0xACC")

    a.record_settlement(question_idx=11, symbol="#110", realized_pnl=25.0, ts_ns=500)
    b.record_settlement(question_idx=12, symbol="#120", realized_pnl=-8.0, ts_ns=600)

    # settlement_pnl_since isolation
    assert a.settlement_pnl_since(0) == 25.0
    assert b.settlement_pnl_since(0) == -8.0

    # realized_pnl_since includes settlement (via the existing SHR-53 path)
    assert a.realized_pnl_since(0) == 25.0
    assert b.realized_pnl_since(0) == -8.0

    # Raw DB check: each row carries the correct strategy_id
    db_path = str(tmp_path / "state.db")
    with sqlite3.connect(db_path) as conn:
        rows = {r[0]: r[1] for r in conn.execute("SELECT question_idx, strategy_id FROM settlement").fetchall()}
    assert rows[11] == "A"
    assert rows[12] == "B"
