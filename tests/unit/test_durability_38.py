"""Finding #38: exit_cooldowns.json write must fsync before os.replace, and
prune() must expire elapsed cooldown entries (TTL-based) and persist the pruned map.

Tests:
  - _save_cooldowns fsyncs before os.replace (power-loss durability)
  - prune() removes entries whose cooldown window has elapsed
  - prune() retains entries whose cooldown window has NOT elapsed
  - prune() persists the pruned map to disk
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from hlanalysis.engine.config import AllowlistEntry, GlobalRiskConfig, StrategyConfig
from hlanalysis.engine.event_bus import EventBus
from hlanalysis.engine.router import Router
from hlanalysis.engine.state import StateDAL


def _strategy_cfg(cooldown_s: float = 30.0) -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        max_position_usd=100,
        stop_loss_pct=10,
        tte_min_seconds=60,
        tte_max_seconds=1800,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200,
        vol_max=0.5,
        entry_cooldown_seconds=cooldown_s,
    )
    return StrategyConfig(
        name="late_resolution",
        paper_mode=True,
        allowlist=[entry],
        blocklist_question_idxs=[],
        defaults=entry,
        **{
            "global": GlobalRiskConfig(
                max_total_inventory_usd=500,
                max_concurrent_positions=5,
                daily_loss_cap_usd=200,
                max_strike_distance_pct=10,
                min_recent_volume_usd=1000,
                stale_data_halt_seconds=5,
                reconcile_interval_seconds=60,
            )
        },
    )


def _make_router(tmp_path: Path, cooldown_s: float = 30.0) -> Router:
    db_path = tmp_path / "state.db"
    dal = StateDAL(db_path)
    dal.run_migrations()
    bus = EventBus()
    exec_client = MagicMock()
    gate = MagicMock()
    gate.check_pre_trade.return_value = MagicMock(approved=True, clamped_size=None)
    return Router(
        dal=dal,
        gate=gate,
        bus=bus,
        exec_client=exec_client,
        strategy_cfg=_strategy_cfg(cooldown_s=cooldown_s),
        strategy_id="late_resolution",
        cloid_prefix="hla-v1-",
    )


def test_save_cooldowns_fsyncs_before_replace(tmp_path: Path):
    """_save_cooldowns must flush+fsync the temp file before os.replace.

    This ensures power-loss between write and rename cannot corrupt the persisted
    cooldown map (finding #38).
    """
    router = _make_router(tmp_path)

    now_ns = time.time_ns()
    router._last_exit_ts[42] = now_ns

    fsync_calls: list[int] = []
    replace_calls: list[tuple] = []
    orig_fsync = os.fsync
    orig_replace = os.replace

    def spy_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        orig_fsync(fd)

    def spy_replace(src, dst) -> None:
        # fsync must have been called BEFORE this replace
        assert fsync_calls, (
            "os.replace called before os.fsync — a power-loss between write and "
            "rename would lose the cooldown map (finding #38)"
        )
        replace_calls.append((str(src), str(dst)))
        orig_replace(src, dst)

    import hlanalysis.engine.router as rt_mod

    with (
        patch.object(rt_mod.os, "fsync", side_effect=spy_fsync),
        patch.object(rt_mod.os, "replace", side_effect=spy_replace),
    ):
        router._save_cooldowns()

    assert replace_calls, "_save_cooldowns did not call os.replace"
    assert fsync_calls, "_save_cooldowns did not call os.fsync"

    # Verify the file was actually written and is valid JSON.
    cooldown_path = Path(router.dal.db_path).parent / "exit_cooldowns.json"
    with open(cooldown_path) as f:
        data = json.load(f)
    assert "42" in data
    assert data["42"] == now_ns


def test_prune_removes_expired_cooldown_entries(tmp_path: Path):
    """prune() must drop entries whose cooldown window has already elapsed.

    An elapsed entry is one where (now_ns - last_exit_ts) / 1e9 >= cooldown_s.
    Retaining such entries serves no purpose (the cooldown already expired) and
    causes the JSON file to grow unbounded (finding #38).
    """
    cooldown_s = 10.0
    router = _make_router(tmp_path, cooldown_s=cooldown_s)

    now_ns = time.time_ns()
    # Entry for question 1 whose cooldown has long elapsed (1 hour ago)
    expired_ts = now_ns - int(3600 * 1e9)
    router._last_exit_ts[1] = expired_ts
    # Entry for question 2 that is still within cooldown (1 second ago)
    active_ts = now_ns - int(1 * 1e9)
    router._last_exit_ts[2] = active_ts

    # prune() is called with the active question set; both are "active" to test
    # that TTL (not just inactivity) drives expiry.
    router.prune(active_question_idxs={1, 2}, now_ns=now_ns)

    assert 1 not in router._last_exit_ts, (
        "Expired cooldown entry (question 1) was NOT pruned; it should have been removed because its window elapsed"
    )
    assert 2 in router._last_exit_ts, (
        "Active cooldown entry (question 2) was incorrectly pruned; its window has not elapsed yet"
    )


def test_prune_persists_pruned_map(tmp_path: Path):
    """After prune() removes expired entries it must write the pruned map to
    disk so a restart doesn't re-load the expired entries from the stale JSON."""
    cooldown_s = 10.0
    router = _make_router(tmp_path, cooldown_s=cooldown_s)

    now_ns = time.time_ns()
    expired_ts = now_ns - int(3600 * 1e9)  # long ago
    active_ts = now_ns - int(1 * 1e9)  # still within window

    router._last_exit_ts[99] = expired_ts  # will be pruned
    router._last_exit_ts[100] = active_ts  # will be kept

    router.prune(active_question_idxs={99, 100}, now_ns=now_ns)

    cooldown_path = Path(router.dal.db_path).parent / "exit_cooldowns.json"
    with open(cooldown_path) as f:
        persisted = json.load(f)

    assert "99" not in persisted, "Expired question 99 still in persisted JSON after prune"
    assert "100" in persisted, "Active question 100 should remain in persisted JSON"


def test_prune_retains_unexpired_active_entry(tmp_path: Path):
    """prune() must NOT remove entries whose cooldown has NOT elapsed, even if
    the question is currently active — the cooldown is a time gate, not an
    activity gate."""
    cooldown_s = 60.0
    router = _make_router(tmp_path, cooldown_s=cooldown_s)

    now_ns = time.time_ns()
    # 5 seconds ago — well within the 60s cooldown
    recent_ts = now_ns - int(5 * 1e9)
    router._last_exit_ts[77] = recent_ts

    router.prune(active_question_idxs={77}, now_ns=now_ns)

    assert 77 in router._last_exit_ts, "Active, unexpired cooldown for question 77 was incorrectly removed"
