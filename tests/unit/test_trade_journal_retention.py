"""Trade-journal retention + source-veto suppression.

The trade_journal table was historically unbounded (only `events` had a prune),
and the router journaled EVERY decision — including the high-frequency mechanical
gate vetoes (`low_volume`, `stale_data`) that never reach the venue. On a
high-fan-out slot that grew the table to ~360k rows/day / >1GB. Two fixes:

1. ``StateDAL.prune_trade_journal`` (age + row-count bounds) — mirrors
   ``prune_events`` so the persist loop can clean it the same way.
2. ``TradeJournal`` drops decisions vetoed by a configured set of routine
   reasons at the source (no retained row), via ``delete_journal_decision``.
"""

from __future__ import annotations

import time

import pytest

from hlanalysis.engine.state import StateDAL, TradeJournalRow
from hlanalysis.engine.trade_journal import TradeJournal


@pytest.fixture
def dal(tmp_path) -> StateDAL:
    d = StateDAL(tmp_path / "state.db")
    d.run_migrations()
    return d


def _decision(dal: StateDAL, cloid: str, ts_ns: int) -> None:
    dal.add_journal_decision(
        TradeJournalRow(
            cloid=cloid,
            question_idx=1,
            decision_ts_ns=ts_ns,
            action="enter",
        )
    )


# ---------------------------------------------------------------------------
# prune_trade_journal — age + row-count bounds (mirrors prune_events)
# ---------------------------------------------------------------------------


def test_prune_trade_journal_age(dal):
    now = time.time_ns()
    old_ns = now - 20 * 24 * 3600 * 1_000_000_000  # 20 days ago
    recent_ns = now - 1 * 24 * 3600 * 1_000_000_000  # 1 day ago
    _decision(dal, "old", old_ns)
    _decision(dal, "recent", recent_ns)

    max_age_ns = 14 * 24 * 3600 * 1_000_000_000  # 14 days
    dal.prune_trade_journal(max_age_ns=max_age_ns, max_rows=1_000_000)

    assert dal.get_journal_row("old") is None
    assert dal.get_journal_row("recent") is not None


def test_prune_trade_journal_row_cap(dal):
    now = time.time_ns()
    for i in range(10):
        _decision(dal, f"c{i}", now + i)  # all recent, ascending ts
    # keep only newest 3
    dal.prune_trade_journal(max_age_ns=9999 * 24 * 3600 * 10**9, max_rows=3)

    survivors = {f"c{i}" for i in range(10) if dal.get_journal_row(f"c{i}") is not None}
    assert survivors == {"c7", "c8", "c9"}


def test_prune_trade_journal_both_bounds(dal):
    now = time.time_ns()
    old_ns = now - 20 * 24 * 3600 * 1_000_000_000
    for i in range(5):
        _decision(dal, f"old{i}", old_ns + i)
    for i in range(5):
        _decision(dal, f"new{i}", now + i)

    max_age_ns = 14 * 24 * 3600 * 1_000_000_000
    dal.prune_trade_journal(max_age_ns=max_age_ns, max_rows=3)

    survivors = {
        c for c in [f"old{i}" for i in range(5)] + [f"new{i}" for i in range(5)] if dal.get_journal_row(c) is not None
    }
    assert survivors == {"new2", "new3", "new4"}


def test_prune_trade_journal_noop_when_under_bounds(dal):
    now = time.time_ns()
    _decision(dal, "a", now)
    _decision(dal, "b", now + 1)
    dal.prune_trade_journal(max_age_ns=9999 * 24 * 3600 * 10**9, max_rows=1_000_000)
    assert dal.get_journal_row("a") is not None
    assert dal.get_journal_row("b") is not None


# ---------------------------------------------------------------------------
# delete_journal_decision
# ---------------------------------------------------------------------------


def test_delete_journal_decision(dal):
    _decision(dal, "x", time.time_ns())
    assert dal.get_journal_row("x") is not None
    dal.delete_journal_decision("x")
    assert dal.get_journal_row("x") is None


def test_delete_journal_decision_absent_is_noop(dal):
    # Must not raise for a cloid that was never journaled.
    dal.delete_journal_decision("never-existed")


# ---------------------------------------------------------------------------
# TradeJournal source suppression: routine gate vetoes leave no retained row
# ---------------------------------------------------------------------------


def _journal(dal, suppress) -> TradeJournal:
    return TradeJournal(dal, suppress_veto_reasons=frozenset(suppress))


def test_suppressed_reject_removes_the_row(dal):
    j = _journal(dal, {"low_volume", "stale_data"})
    j.record_decision(cloid="c1", question_idx=1, decision_ts_ns=1, action="enter")
    assert dal.get_journal_row("c1") is not None  # inserted at decision time
    j.record_reject(cloid="c1", reject_reason="low_volume")
    assert dal.get_journal_row("c1") is None  # suppressed reason → dropped


def test_non_suppressed_reject_is_retained_with_reason(dal):
    j = _journal(dal, {"low_volume", "stale_data"})
    j.record_decision(cloid="c2", question_idx=1, decision_ts_ns=1, action="enter")
    j.record_reject(cloid="c2", reject_reason="max_total_inventory")
    row = dal.get_journal_row("c2")
    assert row is not None
    assert row.reject_reason == "max_total_inventory"


def test_empty_suppress_set_retains_all(dal):
    j = _journal(dal, set())
    j.record_decision(cloid="c3", question_idx=1, decision_ts_ns=1, action="enter")
    j.record_reject(cloid="c3", reject_reason="low_volume")
    row = dal.get_journal_row("c3")
    assert row is not None
    assert row.reject_reason == "low_volume"


def test_sent_then_filled_row_survives_regardless_of_suppress(dal):
    j = _journal(dal, {"low_volume"})
    j.record_decision(cloid="c4", question_idx=1, decision_ts_ns=1, action="enter")
    j.record_send(cloid="c4", send_ts_ns=2)
    j.record_fill(cloid="c4", fill_ts_ns=3, fill_px=0.5, fill_sz=10.0)
    row = dal.get_journal_row("c4")
    assert row is not None
    assert row.send_ts_ns == 2
    assert row.fill_ts_ns == 3


# ---------------------------------------------------------------------------
# persist-loop integration: trade_journal pruned on the same cadence as events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_loop_prunes_trade_journal(tmp_path):
    import asyncio

    from hlanalysis.engine.event_bus import EventBus
    from hlanalysis.engine.events_sink import events_persist_loop
    from hlanalysis.engine.risk_events import RiskVeto

    d = StateDAL(tmp_path / "state.db")
    d.run_migrations()
    # Seed 6 journal rows; row cap of 2 should leave the 2 newest after a prune.
    now = time.time_ns()
    for i in range(6):
        _decision(d, f"j{i}", now + i)

    bus = EventBus(maxsize=64)
    sub = bus.subscribe()
    task = asyncio.create_task(
        events_persist_loop(
            sub,
            [d],
            max_age_ns=9999 * 24 * 3600 * 10**9,
            max_rows=1_000_000,
            journal_max_age_ns=9999 * 24 * 3600 * 10**9,
            journal_max_rows=2,
            prune_every_n=3,
        )
    )
    for i in range(3):  # 3 events → prune fires once
        await bus.publish(RiskVeto(ts_ns=i + 1, account_alias="v1", reason="x"))
    await asyncio.sleep(0.15)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    survivors = {f"j{i}" for i in range(6) if d.get_journal_row(f"j{i}") is not None}
    assert survivors == {"j4", "j5"}
