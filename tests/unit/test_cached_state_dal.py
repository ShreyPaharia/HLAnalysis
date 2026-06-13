from __future__ import annotations

from pathlib import Path

from hlanalysis.engine.state import (
    CachedStateDAL,
    OpenOrder,
    Position,
    StateDAL,
)


def _mk(tmp_path: Path) -> CachedStateDAL:
    dal = CachedStateDAL(tmp_path / "state.db")
    dal.run_migrations()
    return dal


def _pos(qidx: int, qty: float) -> Position:
    return Position(
        question_idx=qidx,
        symbol=f"#{qidx}",
        qty=qty,
        avg_entry=0.9,
        realized_pnl=0.0,
        last_update_ts_ns=1,
        stop_loss_price=0.8,
    )


def _ord(cloid: str, status: str) -> OpenOrder:
    return OpenOrder(
        cloid=cloid,
        venue_oid=None,
        question_idx=1,
        symbol="#1",
        side="buy",
        price=0.9,
        size=10.0,
        status=status,
        placed_ts_ns=1,
        last_update_ts_ns=1,
        strategy_id="t",
    )


def test_upsert_then_read_from_cache_matches_db(tmp_path):
    dal = _mk(tmp_path)
    dal.upsert_position(_pos(1, 10.0))
    # Cache read
    assert dal.get_position(1).qty == 10.0
    assert [p.question_idx for p in dal.all_positions()] == [1]
    # DB read via a fresh plain DAL on the same file = same truth (write-through)
    fresh = StateDAL(tmp_path / "state.db")
    assert fresh.get_position(1).qty == 10.0


def test_delete_position_evicts_cache_and_db(tmp_path):
    dal = _mk(tmp_path)
    dal.upsert_position(_pos(1, 10.0))
    dal.delete_position(1)
    assert dal.get_position(1) is None
    assert dal.all_positions() == []
    assert StateDAL(tmp_path / "state.db").get_position(1) is None


def test_live_orders_filters_status_from_cache(tmp_path):
    dal = _mk(tmp_path)
    dal.upsert_order(_ord("a", "pending"))
    dal.upsert_order(_ord("b", "filled"))
    dal.update_order_status("a", status="open", venue_oid="x", now_ns=2)
    live = {o.cloid for o in dal.live_orders()}
    assert live == {"a"}  # filled excluded; open included


def test_cache_loads_existing_rows_on_construction(tmp_path):
    seed = StateDAL(tmp_path / "state.db")
    seed.run_migrations()
    seed.upsert_position(_pos(7, 5.0))
    seed.upsert_order(_ord("z", "open"))
    # New cached DAL over the SAME file must surface prior rows (restart recovery)
    dal = CachedStateDAL(tmp_path / "state.db")
    assert dal.get_position(7).qty == 5.0
    assert {o.cloid for o in dal.live_orders()} == {"z"}
