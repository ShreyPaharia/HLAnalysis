"""Two strategies on two accounts, sharing one MarketState feed.

Verifies the multi-account engine wiring:
  - Each slot writes to its own state.db (per-account isolation)
  - Each slot's router stamps cloids with its account prefix
  - Daily-loss / kill-switch on one account does not halt the other
  - Both strategies observe entries on the same paper question
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hlanalysis.adapters.base import VenueAdapter
from hlanalysis.engine.config import (
    AlertsConfig, AllowlistEntry, DeployConfig, GlobalRiskConfig, HLConfig,
    StrategyConfig, TelegramConfig,
)
from hlanalysis.engine.hl_client import HLClient
from hlanalysis.engine.runtime import EngineRuntime
from hlanalysis.engine.state import StateDAL
from hlanalysis.events import (
    BboEvent, MarkEvent, Mechanism, ProductType, QuestionMetaEvent,
)


class _FakeAdapter(VenueAdapter):
    venue = "hyperliquid"

    def supports(self, *_a, **_k) -> bool:
        return True

    async def stream(self, _subs):
        now = time.time_ns()
        # Expiry 30 minutes out — inside both strategies' TTE window.
        expiry_str = datetime.fromtimestamp(
            (now + 30 * 60 * 1_000_000_000) / 1e9, tz=timezone.utc
        ).strftime("%Y%m%d-%H%M")
        yield QuestionMetaEvent(
            venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol="qmeta",
            exchange_ts=now - 60_000_000_000, local_recv_ts=now - 60_000_000_000,
            question_idx=99, named_outcome_idxs=[3],
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
            mechanism=Mechanism.CLOB, symbol="#30",
            exchange_ts=now, local_recv_ts=now,
            bid_px=0.95, bid_sz=10.0, ask_px=0.96, ask_sz=10.0,
        )
        yield BboEvent(
            venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol="#31",
            exchange_ts=now, local_recv_ts=now,
            bid_px=0.04, bid_sz=10.0, ask_px=0.05, ask_sz=10.0,
        )
        # Hold a moment so both scanners tick at least twice.
        await asyncio.sleep(3.0)


class _FakeTelegram:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, text: str, *, markdown: bool = True) -> bool:
        self.messages.append(text)
        return True


def _v1_strategy() -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        max_position_usd=100, stop_loss_pct=10, tte_min_seconds=60,
        tte_max_seconds=3600, price_extreme_threshold=0.90,
        distance_from_strike_usd_min=200, vol_max=0.5,
    )
    return StrategyConfig(
        name="late_resolution", paper_mode=True,
        account_alias="v1", strategy_type="late_resolution",
        allowlist=[entry], blocklist_question_idxs=[], defaults=entry,
        **{"global": GlobalRiskConfig(
            max_total_inventory_usd=500, max_concurrent_positions=5,
            daily_loss_cap_usd=200, max_strike_distance_pct=10,
            min_recent_volume_usd=0, stale_data_halt_seconds=5,
            reconcile_interval_seconds=60,
        )},
    )


def _v1b_strategy() -> StrategyConfig:
    """Second v1-style slot on a different account. Using a second
    late_resolution instance keeps the test independent of theta_harvester's
    edge-driven entry path (which is more sensitive to seeded vol). The wiring
    we want to prove out — per-slot DAL, cloid prefix, risk isolation — is
    strategy-agnostic."""
    s = _v1_strategy()
    return s.model_copy(update={"account_alias": "v31"})


@pytest.fixture
def deploy_cfg(tmp_path):
    return DeployConfig(
        env="dev",
        hl_accounts={
            "v1": HLConfig(
                account_address="0xv1", api_secret_key="0xv1",
                base_url="https://api.hyperliquid.xyz",
            ),
            "v31": HLConfig(
                account_address="0xv31", api_secret_key="0xv31",
                base_url="https://api.hyperliquid.xyz",
            ),
        },
        alerts=AlertsConfig(telegram=TelegramConfig(bot_token="x", chat_id="y")),
        state_db_path=str(tmp_path / "state.db"),
        kill_switch_path=str(tmp_path / "halt"),
    )


@pytest.mark.asyncio
async def test_two_strategies_isolated_state(deploy_cfg, tmp_path):
    fake_tg = _FakeTelegram()
    # Per-alias paper HL clients so each slot has independent in-memory books.
    hl_clients: dict[str, HLClient] = {}

    def _hl_factory(alias: str, _hl_cfg, paper: bool) -> HLClient:
        c = HLClient(
            account_address=f"0x{alias}", api_secret_key=f"0x{alias}",
            base_url="https://api.hyperliquid.xyz", paper_mode=True,
        )
        hl_clients[alias] = c
        return c

    runtime = EngineRuntime(
        strategies=[_v1_strategy(), _v1b_strategy()],
        deploy_cfg=deploy_cfg,
        adapter_factory=_FakeAdapter,
        subscriptions=[],
        hl_client_factory=_hl_factory,
        telegram_factory=lambda _http: fake_tg,
    )
    runtime_task = asyncio.create_task(runtime.run())
    await asyncio.sleep(4.0)
    runtime.stop_event.set()
    await asyncio.wait_for(runtime_task, timeout=5.0)

    # Per-account state DBs were created and ARE distinct files.
    v1_db = tmp_path / "v1" / "state.db"
    v31_db = tmp_path / "v31" / "state.db"
    assert v1_db.exists(), "v1 state.db should be created under its alias subdir"
    assert v31_db.exists(), "v31 state.db should be created under its alias subdir"
    assert v1_db != v31_db

    # Each DAL only sees ITS account's orders. Inspect the cloid prefixes on
    # any order that was placed during the run — they must be account-tagged.
    v1_dal = StateDAL(v1_db)
    v31_dal = StateDAL(v31_db)
    v1_orders = v1_dal.live_orders()
    v31_orders = v31_dal.live_orders()
    # At least one of the two should have placed an order (paper fills are
    # immediate, so 'live_orders' is empty post-fill; check fills via paper HL).
    v1_paper_fills = hl_clients["v1"]._paper_fills
    v31_paper_fills = hl_clients["v31"]._paper_fills
    # Both slots see the same market data and identical configs, so both
    # should have at least attempted an entry on the favorite leg.
    assert v1_paper_fills, "v1 should have placed at least one paper fill"
    assert v31_paper_fills, "v31 should have placed at least one paper fill"

    # No cross-talk: each paper book's fills carry that account's cloid prefix.
    assert all(f.cloid.startswith("hla-v1-") for f in v1_paper_fills), \
        f"v1 fills must carry 'hla-v1-' prefix, got {[f.cloid for f in v1_paper_fills]}"
    assert all(f.cloid.startswith("hla-v31-") for f in v31_paper_fills), \
        f"v31 fills must carry 'hla-v31-' prefix, got {[f.cloid for f in v31_paper_fills]}"


@pytest.mark.asyncio
async def test_one_slot_halt_does_not_stop_other(deploy_cfg, tmp_path):
    """Pre-set v1's kill switch BEFORE starting the engine. v1 should halt on
    first continuous-checks tick; v31 should keep scanning. The runtime exits
    only when ALL slots are halted, so we stop manually after a short wait."""
    fake_tg = _FakeTelegram()
    # Create the per-alias halt path that the engine will check.
    v1_halt_dir = tmp_path / "v1"
    v1_halt_dir.mkdir(parents=True, exist_ok=True)
    (v1_halt_dir / "halt").touch()

    def _hl_factory(alias: str, _hl_cfg, paper: bool) -> HLClient:
        return HLClient(
            account_address=f"0x{alias}", api_secret_key=f"0x{alias}",
            base_url="https://api.hyperliquid.xyz", paper_mode=True,
        )

    runtime = EngineRuntime(
        strategies=[_v1_strategy(), _v1b_strategy()],
        deploy_cfg=deploy_cfg,
        adapter_factory=_FakeAdapter,
        subscriptions=[],
        hl_client_factory=_hl_factory,
        telegram_factory=lambda _http: fake_tg,
    )
    runtime_task = asyncio.create_task(runtime.run())
    # Wait long enough for the continuous-checks loop on v1 to notice the halt
    # file (~1s tick) and for v31 to complete a few scans.
    await asyncio.sleep(3.0)
    # v1 should be halted; v31 should still be running.
    v1_slot = next(s for s in runtime.slots if s.alias == "v1")
    v31_slot = next(s for s in runtime.slots if s.alias == "v31")
    assert v1_slot.halted, "v1 should latch halted from the pre-set kill switch"
    assert not v31_slot.halted, "v31 should keep scanning while v1 is halted"
    assert v31_slot.scans_completed > 0, "v31 should have completed at least one scan"
    runtime.stop_event.set()
    await asyncio.wait_for(runtime_task, timeout=5.0)
