"""Regression for the P1.4 stop-loss extraction bug.

Moving `positions_db = slot.dal.all_positions()` out of _continuous_checks_loop
and into _enforce_stop_losses left the stale-data-halt block referencing an
undefined `positions_db`. At runtime that NameError is swallowed by the loop's
broad `except Exception`, silently disabling StaleDataHalt alerting for any slot
holding a position — the exact moment the gate matters. This test drives one
iteration of the loop with a held position + a stale book and asserts the
StaleDataHalt is published.
"""

from __future__ import annotations

import asyncio

from hlanalysis.engine.config import (
    AlertsConfig,
    AllowlistEntry,
    DeployConfig,
    GlobalRiskConfig,
    HyperliquidAccount,
    StrategyConfig,
    TelegramConfig,
)
from hlanalysis.engine.risk_events import StaleDataHalt
from hlanalysis.engine.runtime import EngineRuntime
from hlanalysis.engine.state import Position
from hlanalysis.events import BboEvent, Mechanism, ProductType


def _runtime(tmp_path):
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC"},
        max_position_usd=100,
        stop_loss_pct=None,
        tte_min_seconds=0,
        tte_max_seconds=7200,
        price_extreme_threshold=0.85,
        distance_from_strike_usd_min=0,
        vol_max=100,
        vol_lookback_seconds=3600,
        vol_sampling_dt_seconds=60,
    )
    strat = StrategyConfig(
        name="late_resolution",
        account_alias="v1",
        paper_mode=True,
        strategy_type="late_resolution",
        reference_symbol="BTC",
        allowlist=[entry],
        blocklist_question_idxs=[],
        defaults=entry,
        **{
            "global": GlobalRiskConfig(
                max_total_inventory_usd=500,
                max_concurrent_positions=5,
                daily_loss_cap_usd=200,
                max_strike_distance_pct=50,
                min_recent_volume_usd=100,
                stale_data_halt_seconds=30,
                reconcile_interval_seconds=60,
            )
        },
    )
    deploy = DeployConfig(
        env="dev",
        accounts={
            "v1": HyperliquidAccount(
                account_address="0x0",
                api_secret_key="0x0",
                base_url="https://api.hyperliquid.xyz",
            )
        },
        alerts=AlertsConfig(telegram=TelegramConfig(bot_token="x", chat_id="y")),
        state_db_path=str(tmp_path / "state.db"),
        kill_switch_path=str(tmp_path / "halt"),
    )
    rt = EngineRuntime(
        strategies=[strat],
        deploy_cfg=deploy,
        adapter_factory=lambda: None,
        subscriptions=[],
    )
    rt.slots = [rt._build_slot(strat)]
    return rt, rt.slots[0]


def _hold_stale_book(rt, slot):
    """Held position on '#150' with a deep-past book (age >> threshold)."""
    slot.dal.upsert_position(
        Position(
            question_idx=999,
            symbol="#150",
            qty=10.0,
            avg_entry=0.9,
            realized_pnl=0.0,
            last_update_ts_ns=1,
            stop_loss_price=-1.0,
        )
    )
    rt.market_state.apply(
        BboEvent(
            venue="hyperliquid",
            symbol="#150",
            product_type=ProductType.SPOT,
            mechanism=Mechanism.CLOB,
            bid_px=0.9,
            bid_sz=10.0,
            ask_px=0.92,
            ask_sz=10.0,
            exchange_ts=1,
            local_recv_ts=1,
        )
    )


async def _run_one_iteration(rt, slot):
    seen = rt.bus.subscribe()
    task = asyncio.create_task(rt._continuous_checks_loop(slot))
    await asyncio.sleep(0.2)  # let one iteration run
    rt.stop_event.set()
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    events = []
    while not seen.empty():
        events.append(seen.get_nowait())
    return events


async def test_continuous_checks_publishes_stale_halt_when_global_feed_silent(tmp_path):
    """Genuine feed death: a held leg's book is stale AND the whole feed has gone
    silent (no events ingested for >threshold). StaleDataHalt must fire."""
    rt, slot = _runtime(tmp_path)
    _hold_stale_book(rt, slot)
    rt._last_ingest_ns = 1  # last global event in the deep past → silent
    events = await _run_one_iteration(rt, slot)
    assert any(isinstance(e, StaleDataHalt) and e.symbol == "#150" for e in events), (
        f"expected StaleDataHalt for #150, got {[type(e).__name__ for e in events]}"
    )


async def test_continuous_checks_suppresses_stale_halt_when_global_feed_alive(tmp_path):
    """A single held favorite's book going quiet while the global feed is still
    flowing is a quiet market, not stale data — no StaleDataHalt should fire.

    This is the PM-favorite flood: '#150' hasn't printed in minutes but other
    symbols are ticking, so last_ingest_ns is recent."""
    rt, slot = _runtime(tmp_path)
    _hold_stale_book(rt, slot)
    rt._last_ingest_ns = rt._now_ns()  # feed delivered an event just now
    events = await _run_one_iteration(rt, slot)
    assert not any(isinstance(e, StaleDataHalt) for e in events), (
        f"expected no StaleDataHalt while feed alive, got {[type(e).__name__ for e in events]}"
    )
