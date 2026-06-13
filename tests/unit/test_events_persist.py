"""Component 2 — persisted events table + _events_persist_loop.

TDD spec: tests written before implementation, exercising:
- append_event / query methods on StateDAL
- prune_events (age + row-count bounds)
- _events_persist_loop: real EventBus publish → DB row via async task
- publish does not block when the persist consumer is slow (drop path)
"""

from __future__ import annotations

import asyncio
import time

import pytest

from hlanalysis.engine.event_bus import EventBus
from hlanalysis.engine.risk_events import (
    Entry,
    Exit,
    FeedDown,
    FeedRecovered,
    FeedStale,
    OrderRejected,
    RiskVeto,
    StopLossTriggered,
    EngineHeartbeat,
)
from hlanalysis.engine.state import StateDAL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dal(tmp_path):
    db_path = tmp_path / "state.db"
    d = StateDAL(db_path)
    d.run_migrations()
    return d


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def test_events_table_created_by_baseline(dal):
    # The events table (formerly migration 0006) is now part of the Alembic
    # baseline; confirm migrations ran (a head is recorded) and the table exists.
    # applied_versions() reports only the current head revision, which advances
    # with each new migration, so don't pin it to the baseline.
    assert dal.applied_versions()
    import sqlite3

    with sqlite3.connect(dal.db_path) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "events" in names


# ---------------------------------------------------------------------------
# append_event basic round-trip
# ---------------------------------------------------------------------------


def test_append_event_minimal(dal):
    """Minimal insert: only kind + ts_ns required; nullable fields are None."""
    dal.append_event(
        ts_ns=1_000_000,
        alias=None,
        kind="risk_veto",
        question_idx=None,
        reason=None,
        payload_json=None,
    )
    rows = dal.events_since(since_ts_ns=0)
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "risk_veto"
    assert row["ts_ns"] == 1_000_000
    assert row["alias"] is None
    assert row["reason"] is None


def test_append_event_full(dal):
    dal.append_event(
        ts_ns=2_000_000,
        alias="v1",
        kind="order_rejected",
        question_idx=42,
        reason="insufficient balance",
        payload_json='{"cloid":"hla-1"}',
    )
    rows = dal.events_since(since_ts_ns=0)
    assert len(rows) == 1
    r = rows[0]
    assert r["alias"] == "v1"
    assert r["kind"] == "order_rejected"
    assert r["question_idx"] == 42
    assert r["reason"] == "insufficient balance"
    assert r["payload_json"] == '{"cloid":"hla-1"}'


# ---------------------------------------------------------------------------
# events_since (basic window filter)
# ---------------------------------------------------------------------------


def test_events_since_window(dal):
    dal.append_event(ts_ns=100, alias="v1", kind="entry", question_idx=1, reason=None, payload_json=None)
    dal.append_event(ts_ns=200, alias="v1", kind="exit", question_idx=1, reason="exit_edge", payload_json=None)
    dal.append_event(ts_ns=300, alias="v1", kind="risk_veto", question_idx=2, reason="cap", payload_json=None)

    all_rows = dal.events_since(since_ts_ns=0)
    assert len(all_rows) == 3

    late = dal.events_since(since_ts_ns=150)
    assert len(late) == 2
    assert all(r["ts_ns"] >= 150 for r in late)


# ---------------------------------------------------------------------------
# reject_counts_since
# ---------------------------------------------------------------------------


def test_reject_counts_since_groups_by_kind_reason(dal):
    for i in range(3):
        dal.append_event(
            ts_ns=100 + i,
            alias="v1",
            kind="order_rejected",
            question_idx=i,
            reason="bad token",
            payload_json=f'{{"cloid":"c{i}"}}',
        )
    dal.append_event(
        ts_ns=200,
        alias="v1",
        kind="order_rejected",
        question_idx=10,
        reason="min notional",
        payload_json='{"cloid":"cx"}',
    )
    dal.append_event(ts_ns=300, alias="v1", kind="risk_veto", question_idx=11, reason="cap", payload_json=None)

    counts = dal.reject_counts_since(since_ts_ns=0)
    # Two (kind, reason) groups for order_rejected; one for risk_veto
    keys = {(r["kind"], r["reason"]) for r in counts}
    assert ("order_rejected", "bad token") in keys
    assert ("order_rejected", "min notional") in keys
    assert ("risk_veto", "cap") in keys

    by_key = {(r["kind"], r["reason"]): r for r in counts}
    assert by_key[("order_rejected", "bad token")]["count"] == 3
    assert by_key[("order_rejected", "min notional")]["count"] == 1
    # sample_payload should be non-empty for order_rejected rows
    assert by_key[("order_rejected", "bad token")]["sample_payload"] is not None


def test_reject_counts_since_window_cutoff(dal):
    dal.append_event(ts_ns=50, alias="v1", kind="order_rejected", question_idx=1, reason="old", payload_json=None)
    dal.append_event(ts_ns=200, alias="v1", kind="order_rejected", question_idx=2, reason="new", payload_json=None)

    counts = dal.reject_counts_since(since_ts_ns=100)
    reasons = {r["reason"] for r in counts}
    assert "new" in reasons
    assert "old" not in reasons


# ---------------------------------------------------------------------------
# events_for_question
# ---------------------------------------------------------------------------


def test_events_for_question(dal):
    dal.append_event(ts_ns=10, alias="v1", kind="entry", question_idx=42, reason=None, payload_json=None)
    dal.append_event(ts_ns=20, alias="v1", kind="exit", question_idx=42, reason="time_stop", payload_json=None)
    dal.append_event(ts_ns=30, alias="v1", kind="entry", question_idx=99, reason=None, payload_json=None)

    rows = dal.events_for_question(42)
    assert len(rows) == 2
    assert all(r["question_idx"] == 42 for r in rows)
    assert rows[0]["ts_ns"] <= rows[1]["ts_ns"]  # ordered by ts_ns asc


def test_events_for_question_empty(dal):
    assert dal.events_for_question(999) == []


# ---------------------------------------------------------------------------
# last_event_by_kind
# ---------------------------------------------------------------------------


def test_last_event_by_kind(dal):
    dal.append_event(ts_ns=10, alias="v1", kind="risk_veto", question_idx=1, reason="cap", payload_json=None)
    dal.append_event(ts_ns=20, alias="v1", kind="risk_veto", question_idx=2, reason="halt", payload_json=None)
    dal.append_event(ts_ns=30, alias="v1", kind="entry", question_idx=3, reason=None, payload_json=None)

    last = dal.last_event_by_kind("risk_veto", alias="v1")
    assert last is not None
    assert last["ts_ns"] == 20
    assert last["reason"] == "halt"


def test_last_event_by_kind_missing(dal):
    assert dal.last_event_by_kind("entry", alias="v1") is None


def test_last_event_by_kind_no_alias_filter(dal):
    dal.append_event(ts_ns=5, alias="v1", kind="exit", question_idx=1, reason="x", payload_json=None)
    dal.append_event(ts_ns=10, alias="v31", kind="exit", question_idx=2, reason="y", payload_json=None)
    # Without alias filter returns most recent across all aliases
    last = dal.last_event_by_kind("exit")
    assert last["ts_ns"] == 10


# ---------------------------------------------------------------------------
# prune_events — age bound
# ---------------------------------------------------------------------------


def test_prune_age(dal):
    now = time.time_ns()
    old_ns = now - 20 * 24 * 3600 * 1_000_000_000  # 20 days ago
    recent_ns = now - 1 * 24 * 3600 * 1_000_000_000  # 1 day ago

    dal.append_event(ts_ns=old_ns, alias="v1", kind="entry", question_idx=1, reason=None, payload_json=None)
    dal.append_event(ts_ns=recent_ns, alias="v1", kind="exit", question_idx=1, reason="x", payload_json=None)

    max_age_ns = 14 * 24 * 3600 * 1_000_000_000  # 14 days
    dal.prune_events(max_age_ns=max_age_ns, max_rows=1_000_000)

    rows = dal.events_since(since_ts_ns=0)
    assert len(rows) == 1
    assert rows[0]["kind"] == "exit"


def test_prune_row_cap(dal):
    # Insert 10 rows with current-time-based sequential timestamps so the age
    # prune (cutoff = now - max_age) does NOT delete them.
    now = time.time_ns()
    for i in range(10):
        dal.append_event(ts_ns=now + i, alias="v1", kind="entry", question_idx=i, reason=None, payload_json=None)

    # Prune to max 5 rows (max_age far in future so age prune doesn't fire)
    max_age_ns = 9999 * 24 * 3600 * 1_000_000_000
    dal.prune_events(max_age_ns=max_age_ns, max_rows=5)

    rows = dal.events_since(since_ts_ns=0)
    assert len(rows) == 5
    # The OLDEST rows should have been pruned (keep newest 5)
    ts_values = sorted(r["ts_ns"] for r in rows)
    assert ts_values[0] == now + 5  # rows 0..4 deleted, kept 5..9


def test_prune_both_bounds(dal):
    now = time.time_ns()
    old_ns = now - 20 * 24 * 3600 * 1_000_000_000

    # 5 old rows + 5 recent rows; recent rows use now+i so they survive age prune
    for i in range(5):
        dal.append_event(ts_ns=old_ns + i, alias="v1", kind="entry", question_idx=i, reason=None, payload_json=None)
    for i in range(5):
        dal.append_event(ts_ns=now + i, alias="v1", kind="exit", question_idx=i + 10, reason=None, payload_json=None)

    max_age_ns = 14 * 24 * 3600 * 1_000_000_000
    # row cap of 3 means after age-prune (removes 5 old), the 5 recent get
    # pruned further to 3
    dal.prune_events(max_age_ns=max_age_ns, max_rows=3)

    rows = dal.events_since(since_ts_ns=0)
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# _events_persist_loop integration: publish → DB via real EventBus
# ---------------------------------------------------------------------------


def _make_persist_task(bus, dal, tmp_path=None, prune_every_n=1000):
    """Helper: subscribe and start a persist loop task. Returns (task, sub)."""
    from hlanalysis.engine.events_sink import events_persist_loop

    persist_sub = bus.subscribe()
    task = asyncio.create_task(
        events_persist_loop(
            persist_sub,
            [dal],
            max_age_ns=14 * 24 * 3600 * 10**9,
            max_rows=1_000_000,
            prune_every_n=prune_every_n,
        )
    )
    return task, persist_sub


async def _stop_task(task):
    """Cancel a task and wait for it to finish, suppressing CancelledError."""
    task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
    except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
        pass


@pytest.mark.asyncio
async def test_persist_loop_writes_risk_veto(tmp_path):
    """Publish RiskVeto through a real EventBus and assert row lands in DB."""
    db_path = tmp_path / "state.db"
    dal = StateDAL(db_path)
    dal.run_migrations()

    bus = EventBus(maxsize=64)
    task, _ = _make_persist_task(bus, dal)

    ev = RiskVeto(ts_ns=1_234_567, account_alias="v1", reason="daily_cap", question_idx=77)
    await bus.publish(ev)
    # Give the loop one iteration
    await asyncio.sleep(0.05)
    await _stop_task(task)

    rows = dal.events_since(since_ts_ns=0)
    assert len(rows) == 1
    r = rows[0]
    assert r["kind"] == "risk_veto"
    assert r["alias"] == "v1"
    assert r["question_idx"] == 77
    assert r["reason"] == "daily_cap"
    assert r["ts_ns"] == 1_234_567


@pytest.mark.asyncio
async def test_persist_loop_writes_order_rejected(tmp_path):
    db_path = tmp_path / "state.db"
    dal = StateDAL(db_path)
    dal.run_migrations()

    bus = EventBus(maxsize=64)
    task, _ = _make_persist_task(bus, dal)

    ev = OrderRejected(
        ts_ns=5_000,
        account_alias="v31",
        cloid="hla-x",
        question_idx=99,
        symbol="@30",
        side="buy",
        size=10.0,
        price=0.95,
        error="insufficient balance",
    )
    await bus.publish(ev)
    await asyncio.sleep(0.05)
    await _stop_task(task)

    rows = dal.events_since(since_ts_ns=0)
    assert len(rows) == 1
    r = rows[0]
    assert r["kind"] == "order_rejected"
    assert r["alias"] == "v31"
    assert r["question_idx"] == 99
    assert r["reason"] == "insufficient balance"


@pytest.mark.asyncio
async def test_persist_loop_writes_entry_and_exit(tmp_path):
    db_path = tmp_path / "state.db"
    dal = StateDAL(db_path)
    dal.run_migrations()

    bus = EventBus(maxsize=64)
    task, _ = _make_persist_task(bus, dal)

    entry = Entry(
        ts_ns=10, account_alias="v1", cloid="hla-1", question_idx=42, symbol="@30", side="buy", size=10.0, price=0.95
    )
    exit_ev = Exit(
        ts_ns=20, account_alias="v1", question_idx=42, symbol="@30", qty=10.0, realized_pnl=0.5, reason="exit_safety_d"
    )
    await bus.publish(entry)
    await bus.publish(exit_ev)
    await asyncio.sleep(0.05)
    await _stop_task(task)

    rows = dal.events_for_question(42)
    assert len(rows) == 2
    kinds = [r["kind"] for r in rows]
    assert "entry" in kinds
    assert "exit" in kinds
    exit_row = next(r for r in rows if r["kind"] == "exit")
    assert exit_row["reason"] == "exit_safety_d"


@pytest.mark.asyncio
async def test_persist_loop_writes_feed_events(tmp_path):
    db_path = tmp_path / "state.db"
    dal = StateDAL(db_path)
    dal.run_migrations()

    bus = EventBus(maxsize=64)
    task, _ = _make_persist_task(bus, dal)

    await bus.publish(FeedDown(ts_ns=1, account_alias="", consecutive_failures=1))
    await bus.publish(FeedRecovered(ts_ns=2, account_alias=""))
    await asyncio.sleep(0.05)
    await _stop_task(task)

    rows = dal.events_since(since_ts_ns=0)
    kinds = {r["kind"] for r in rows}
    assert "feed_down" in kinds
    assert "feed_recovered" in kinds


@pytest.mark.asyncio
async def test_persist_loop_multiple_event_types(tmp_path):
    """All risk_events types should persist without error."""
    db_path = tmp_path / "state.db"
    dal = StateDAL(db_path)
    dal.run_migrations()

    bus = EventBus(maxsize=128)
    task, _ = _make_persist_task(bus, dal)

    events_to_publish = [
        RiskVeto(ts_ns=1, account_alias="v1", reason="cap"),
        Entry(ts_ns=2, account_alias="v1", cloid="c1", question_idx=1, symbol="@30", side="buy", size=10.0, price=0.95),
        Exit(
            ts_ns=3, account_alias="v1", question_idx=1, symbol="@30", qty=10.0, realized_pnl=0.5, reason="settlement"
        ),
        StopLossTriggered(ts_ns=4, account_alias="v1", question_idx=2, symbol="@30", qty=5.0, trigger_px=0.8),
        OrderRejected(
            ts_ns=5,
            account_alias="v31",
            cloid="c2",
            question_idx=3,
            symbol="@31",
            side="sell",
            size=5.0,
            price=0.05,
            error="bad token",
        ),
        EngineHeartbeat(ts_ns=6, account_alias="", events_ingested=100, d_events=10, n_questions=5),
        FeedStale(ts_ns=7, account_alias="", d_events=0, interval_seconds=30.0),
        FeedDown(ts_ns=8, account_alias="", consecutive_failures=1),
        FeedRecovered(ts_ns=9, account_alias=""),
    ]

    for ev in events_to_publish:
        await bus.publish(ev)

    await asyncio.sleep(0.1)
    await _stop_task(task)

    rows = dal.events_since(since_ts_ns=0)
    assert len(rows) == len(events_to_publish)


# ---------------------------------------------------------------------------
# publish does NOT block when the persist consumer is slow (drop path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_does_not_block_on_slow_persist_consumer():
    """When the persist queue is full, publish drops and returns immediately.

    The bus maxsize=2 ensures the queue fills. We verify publish completes
    without waiting on the slow consumer (no hang or timeout).
    """
    bus = EventBus(maxsize=2, drop_when_full=True)
    _ = bus.subscribe()  # fast consumer — keeps bus honest
    slow = bus.subscribe(maxsize=2)  # persist consumer queue — fills up

    # Fill the slow queue by publishing more events than its capacity
    for i in range(10):
        await asyncio.wait_for(
            bus.publish(RiskVeto(ts_ns=i, account_alias="v1", reason="cap")),
            timeout=0.5,  # must not hang
        )

    # The slow queue should be at capacity, but publish returned every time
    assert slow.qsize() <= 2


# ---------------------------------------------------------------------------
# Persist loop periodic prune is wired (smoke test)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_loop_prune_fires_via_n_counter(tmp_path):
    """With prune_every_n=3 and 5 events, the prune should fire once and
    enforce max_rows=3, leaving only the 3 newest rows."""
    from hlanalysis.engine.events_sink import events_persist_loop

    db_path = tmp_path / "state.db"
    dal = StateDAL(db_path)
    dal.run_migrations()

    bus = EventBus(maxsize=64)
    persist_sub = bus.subscribe()
    task = asyncio.create_task(
        events_persist_loop(
            persist_sub,
            [dal],
            max_age_ns=9999 * 24 * 3600 * 10**9,
            max_rows=3,
            prune_every_n=3,
        )
    )

    for i in range(5):
        await bus.publish(RiskVeto(ts_ns=i + 1, account_alias="v1", reason="x"))
    await asyncio.sleep(0.15)
    await _stop_task(task)

    rows = dal.events_since(since_ts_ns=0)
    assert len(rows) <= 3
