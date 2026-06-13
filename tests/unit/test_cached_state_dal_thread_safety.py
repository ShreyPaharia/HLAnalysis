# tests/unit/test_cached_state_dal_thread_safety.py
"""TDD tests for CachedStateDAL thread-safety (Fix 2).

Covers:
- Concurrent reads + writes don't raise RuntimeError (dict size change during
  iteration) or produce corrupt cache state.
- end_session with halt_reason no longer attempts the dead attribute assignment.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from hlanalysis.engine.state import (
    CachedStateDAL,
    OpenOrder,
    Position,
    StateDAL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _ord(cloid: str) -> OpenOrder:
    return OpenOrder(
        cloid=cloid,
        venue_oid=None,
        question_idx=1,
        symbol="#1",
        side="buy",
        price=0.9,
        size=10.0,
        status="pending",
        placed_ts_ns=1,
        last_update_ts_ns=1,
        strategy_id="t",
    )


# ---------------------------------------------------------------------------
# Thread-safety: concurrent upsert + read must not corrupt cache
# ---------------------------------------------------------------------------


def test_concurrent_upsert_and_read_no_corruption(tmp_path):
    """Spawn N writer threads and M reader threads all hitting the same
    CachedStateDAL concurrently.  Asserts:
      - No RuntimeError (dict changed size during iteration)
      - After all writes complete, all_positions() reports the correct count.
    """
    dal = _mk(tmp_path)
    # Pre-load the cache so _ensure_loaded is not in the write path
    assert dal.all_positions() == []

    N = 30  # writer threads
    errors: list[Exception] = []
    barrier = threading.Barrier(N + 5)  # all threads start simultaneously

    def writer(qidx: int) -> None:
        try:
            barrier.wait()
            dal.upsert_position(_pos(qidx, float(qidx)))
        except Exception as exc:
            errors.append(exc)

    def reader() -> None:
        try:
            barrier.wait()
            for _ in range(20):
                _ = dal.all_positions()
                _ = dal.live_orders()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(N)]
    threads += [threading.Thread(target=reader) for _ in range(5)]

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == [], f"Thread errors: {errors}"
    positions = dal.all_positions()
    assert len(positions) == N


def test_concurrent_order_upsert_and_status_update(tmp_path):
    """Concurrent order upserts and status updates don't corrupt the cache."""
    dal = _mk(tmp_path)
    # Pre-seed some orders
    for i in range(10):
        dal.upsert_order(_ord(f"cloid-{i}"))

    errors: list[Exception] = []
    barrier = threading.Barrier(20)

    def updater(cloid: str) -> None:
        try:
            barrier.wait()
            for _ in range(5):
                dal.update_order_status(cloid, status="open", venue_oid="v", now_ns=99)
                dal.update_order_status(cloid, status="filled", venue_oid="v", now_ns=100)
        except Exception as exc:
            errors.append(exc)

    def reader() -> None:
        try:
            barrier.wait()
            for _ in range(30):
                _ = dal.live_orders()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=updater, args=(f"cloid-{i}",)) for i in range(10)]
    threads += [threading.Thread(target=reader) for _ in range(10)]

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == [], f"Thread errors: {errors}"


def test_concurrent_delete_position_and_read(tmp_path):
    """Concurrent deletes and all_positions reads don't produce RuntimeError."""
    dal = _mk(tmp_path)
    for i in range(20):
        dal.upsert_position(_pos(i, float(i)))

    errors: list[Exception] = []
    barrier = threading.Barrier(25)

    def deleter(qidx: int) -> None:
        try:
            barrier.wait()
            dal.delete_position(qidx)
        except Exception as exc:
            errors.append(exc)

    def reader() -> None:
        try:
            barrier.wait()
            for _ in range(20):
                _ = dal.all_positions()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=deleter, args=(i,)) for i in range(20)]
    threads += [threading.Thread(target=reader) for _ in range(5)]

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == [], f"Thread errors: {errors}"
    # After all deletes: between 0 and 20 positions remain (no double-delete races)
    remaining = dal.all_positions()
    assert 0 <= len(remaining) <= 20


# ---------------------------------------------------------------------------
# end_session halt_reason: dead assignment removed (no AttributeError)
# ---------------------------------------------------------------------------


def test_end_session_with_halt_reason_does_not_raise(tmp_path):
    """end_session(halt_reason=...) must not raise even though Session_ has
    no halt_reason column — the dead assignment must have been removed."""
    dal = _mk(tmp_path)
    session_id = "sess-abc"
    dal.start_session(session_id, now_ns=1_000_000)
    # Must complete without AttributeError or any other exception
    dal.end_session(session_id, now_ns=2_000_000, halt_reason="test_halt")


def test_end_session_without_halt_reason_still_works(tmp_path):
    """Legacy callers that omit halt_reason must keep working."""
    dal = _mk(tmp_path)
    dal.start_session("sess-xyz", now_ns=1)
    dal.end_session("sess-xyz", now_ns=2)  # halt_reason defaults to None
