from __future__ import annotations

import pytest

from hlanalysis.engine.hl_client import (
    ClearinghouseState,
    OpenOrderRow,
)
from hlanalysis.engine.restart_drift import RestartDriftGate
from hlanalysis.engine.state import OpenOrder, Position, StateDAL


@pytest.fixture
def dal(tmp_path):
    d = StateDAL(tmp_path / "state.db")
    d.run_migrations()
    return d


def test_clean_state_does_not_block(tmp_path, dal):
    block = tmp_path / "restart_blocked"
    gate = RestartDriftGate(dal=dal, block_path=block)
    res = gate.run(
        venue_open=[],
        venue_state=ClearinghouseState(positions=(), account_value_usd=0),
        fills_lookup=lambda _c: [],
        now_ns=2,
    )
    assert res.blocked is False
    assert not block.exists()


def test_local_ghost_blocks_restart(tmp_path, dal):
    dal.upsert_order(
        OpenOrder(
            cloid="hla-ghost",
            venue_oid="v",
            question_idx=42,
            symbol="@30",
            side="buy",
            price=0.95,
            size=10.0,
            status="open",
            placed_ts_ns=1,
            last_update_ts_ns=1,
            strategy_id="x",
        )
    )
    block = tmp_path / "restart_blocked"
    gate = RestartDriftGate(dal=dal, block_path=block)
    res = gate.run(
        venue_open=[],
        venue_state=ClearinghouseState(positions=(), account_value_usd=0),
        fills_lookup=lambda _c: [],
        now_ns=2,
    )
    assert res.blocked is True
    assert block.exists()
    assert "local_ghost" in block.read_text()


def test_existing_block_file_keeps_blocking(tmp_path, dal):
    block = tmp_path / "restart_blocked"
    block.write_text("manual hold")
    gate = RestartDriftGate(dal=dal, block_path=block)
    res = gate.run(
        venue_open=[],
        venue_state=ClearinghouseState(positions=(), account_value_usd=0),
        fills_lookup=lambda _c: [],
        now_ns=2,
    )
    assert res.blocked is True


def test_venue_orphan_blocks(tmp_path, dal):
    block = tmp_path / "restart_blocked"
    gate = RestartDriftGate(dal=dal, block_path=block)
    res = gate.run(
        venue_open=[
            OpenOrderRow(
                cloid="hla-orphan",
                venue_oid="v",
                symbol="@30",
                side="buy",
                price=0.95,
                size=10.0,
                placed_ts_ns=1,
            )
        ],
        venue_state=ClearinghouseState(positions=(), account_value_usd=0),
        fills_lookup=lambda _c: [],
        now_ns=2,
    )
    assert res.blocked is True
    assert "venue_orphan" in block.read_text()


def test_clean_restart_clears_block_file(tmp_path, dal):
    block = tmp_path / "restart_blocked"
    # Block file present but no drift this run, AND auto_clear_on_clean=True
    block.write_text("stale")
    gate = RestartDriftGate(dal=dal, block_path=block, auto_clear_on_clean=True)
    res = gate.run(
        venue_open=[],
        venue_state=ClearinghouseState(positions=(), account_value_usd=0),
        fills_lookup=lambda _c: [],
        now_ns=2,
    )
    assert res.blocked is False
    assert not block.exists()


def test_paper_slot_does_not_vanish_held_position(tmp_path, dal):
    """A paper slot's venue view is synthetic and EMPTY after every restart, so
    the restart gate must NOT treat a held DB position as vanished-from-venue
    and delete it. That silent delete orphaned every eth_ms paper-PM leg and
    drove the de-track / re-entry loop (2026-06-16). Paper slots trust their own
    fill ledger; venue-truth apply at restart is for LIVE slots only.
    """
    dal.upsert_position(
        Position(
            question_idx=42,
            symbol="@30",
            qty=51.81,
            avg_entry=0.9,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=-1.0,
        ),
    )
    block = tmp_path / "restart_blocked"
    gate = RestartDriftGate(dal=dal, block_path=block, paper_mode=True)
    res = gate.run(
        venue_open=[],
        venue_state=ClearinghouseState(positions=(), account_value_usd=0),
        fills_lookup=lambda _c: [],
        now_ns=2,
    )
    # The held position must survive — the empty paper venue is not authoritative.
    assert dal.get_position(42) is not None
    assert res.blocked is False


def test_live_slot_still_vanishes_absent_position(tmp_path, dal):
    """Regression guard: a LIVE slot must keep the restart behaviour — a held
    position absent from the (authoritative) venue is vanished, since restart is
    when venue truth wins and we may have missed the closing fill while down.
    """
    dal.upsert_position(
        Position(
            question_idx=42,
            symbol="@30",
            qty=51.81,
            avg_entry=0.9,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=-1.0,
        ),
    )
    block = tmp_path / "restart_blocked"
    gate = RestartDriftGate(dal=dal, block_path=block, paper_mode=False)
    gate.run(
        venue_open=[],
        venue_state=ClearinghouseState(positions=(), account_value_usd=0),
        fills_lookup=lambda _c: [],
        now_ns=2,
    )
    assert dal.get_position(42) is None
