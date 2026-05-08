from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hlanalysis.adapters.base import VenueAdapter
from hlanalysis.alerts.telegram import TelegramClient
from hlanalysis.config import Subscription
from hlanalysis.engine.config import (
    AlertsConfig, AllowlistEntry, DeployConfig, GlobalRiskConfig, HLConfig,
    StrategyConfig, TelegramConfig,
)
from hlanalysis.engine.hl_client import HLClient
from hlanalysis.engine.runtime import EngineRuntime
from hlanalysis.events import (
    BboEvent, MarkEvent, Mechanism, NormalizedEvent, ProductType,
    QuestionMetaEvent, SettlementEvent,
)


class _FakeAdapter(VenueAdapter):
    venue = "hyperliquid"

    def supports(self, *_a, **_k) -> bool:
        return True

    async def stream(self, _subs):
        # Use real wall-clock timestamps so stale-data check doesn't fire.
        now = time.time_ns()
        # Expiry 10 minutes in the future as YYYYMMDD-HHMM
        expiry_str = datetime.fromtimestamp(
            (now + 10 * 60 * 1_000_000_000) / 1e9, tz=timezone.utc
        ).strftime('%Y%m%d-%H%M')

        yield QuestionMetaEvent(
            venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol="qmeta",
            exchange_ts=now - 60_000_000_000, local_recv_ts=now - 60_000_000_000,
            question_idx=42, named_outcome_idxs=[30, 31],
            keys=["class", "underlying", "period", "expiry", "strike"],
            values=["priceBinary", "BTC", "1h", expiry_str, "80000"],
        )
        for i in range(8):
            yield MarkEvent(
                venue="hyperliquid", product_type=ProductType.PERP,
                mechanism=Mechanism.CLOB, symbol="BTC",
                exchange_ts=now - (8 - i) * 1_000_000,
                local_recv_ts=now - (8 - i) * 1_000_000,
                mark_px=80_300.0 + i * 0.01,
            )
        yield BboEvent(
            venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol="@30",
            exchange_ts=now, local_recv_ts=now,
            bid_px=0.95, bid_sz=10.0, ask_px=0.96, ask_sz=10.0,
        )
        yield BboEvent(
            venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol="@31",
            exchange_ts=now, local_recv_ts=now,
            bid_px=0.04, bid_sz=10.0, ask_px=0.05, ask_sz=10.0,
        )
        # Hold a moment so the scanner ticks
        await asyncio.sleep(1.5)
        # Settlement
        yield SettlementEvent(
            venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol="@30",
            exchange_ts=now + 2_000_000_000, local_recv_ts=now + 2_000_000_000,
            settled_side_idx=30, settle_price=1.0, settle_ts=now + 2_000_000_000,
        )
        await asyncio.sleep(1.5)


class _FakeTelegram:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, text: str, *, markdown: bool = True) -> bool:
        self.messages.append(text)
        return True


@pytest.fixture
def cfgs(tmp_path):
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        max_position_usd=100, stop_loss_pct=10, tte_min_seconds=60,
        tte_max_seconds=1800, price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200, vol_max=0.5,
    )
    strategy = StrategyConfig(
        name="late_resolution", paper_mode=True,
        allowlist=[entry], blocklist_question_idxs=[],
        defaults=entry,
        **{"global": GlobalRiskConfig(
            max_total_inventory_usd=500, max_concurrent_positions=5,
            daily_loss_cap_usd=200, max_strike_distance_pct=10,
            min_recent_volume_usd=0, stale_data_halt_seconds=5,
            reconcile_interval_seconds=60,
        )},
    )
    deploy = DeployConfig(
        env="dev",
        hl=HLConfig(account_address="0x", api_secret_key="0x",
                    base_url="https://api.hyperliquid.xyz"),
        alerts=AlertsConfig(telegram=TelegramConfig(bot_token="x", chat_id="y")),
        state_db_path=str(tmp_path / "state.db"),
        kill_switch_path=str(tmp_path / "halt"),
    )
    return strategy, deploy


@pytest.mark.asyncio
async def test_paper_loop_enters_and_exits_on_settlement(cfgs, tmp_path):
    strategy_cfg, deploy_cfg = cfgs
    fake_tg = _FakeTelegram()

    runtime = EngineRuntime(
        strategy_cfg=strategy_cfg,
        deploy_cfg=deploy_cfg,
        adapter_factory=_FakeAdapter,
        subscriptions=[],
        hl_client_factory=lambda paper: HLClient(
            account_address="0x", api_secret_key="0x",
            base_url="https://api.hyperliquid.xyz", paper_mode=True,
        ),
        telegram_factory=lambda _http: fake_tg,
    )
    runtime_task = asyncio.create_task(runtime.run())
    # Let the loop run, then stop
    await asyncio.sleep(4.0)
    runtime.stop_event.set()
    await asyncio.wait_for(runtime_task, timeout=5.0)

    # Validate alerts
    entries = [m for m in fake_tg.messages if "ENTRY" in m]
    exits = [m for m in fake_tg.messages if "EXIT" in m]
    assert entries, f"expected ENTRY alert, got: {fake_tg.messages}"
    assert exits, f"expected EXIT alert, got: {fake_tg.messages}"
