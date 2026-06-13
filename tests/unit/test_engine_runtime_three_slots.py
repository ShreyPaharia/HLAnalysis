"""3-slot engine build smoke test.

Phase 5 wires up venue-typed account dispatch in `EngineRuntime._build_slot`:
  - HyperliquidAccount → HLClient
  - PolymarketAccount → PMClient

This test constructs a runtime with one HL `v1` slot, one HL `v31` slot,
and one PM `v31_pm` slot (paper_mode=True) and verifies all three
AccountSlots build without network. The PM slot's exec client must be a
PMClient (not an HLClient swapped in by accident).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hlanalysis.engine.config import (
    AlertsConfig,
    AllowlistEntry,
    DeployConfig,
    GlobalRiskConfig,
    HyperliquidAccount,
    PolymarketAccount,
    StrategyConfig,
    TelegramConfig,
    ThetaParams,
)
from hlanalysis.engine.hl_client import HLClient
from hlanalysis.engine.pm_client import PMClient
from hlanalysis.engine.runtime import EngineRuntime


def _late_resolution_strategy(alias: str) -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC"},
        max_position_usd=100,
        stop_loss_pct=10,
        tte_min_seconds=0,
        tte_max_seconds=3600,
        price_extreme_threshold=0.90,
        distance_from_strike_usd_min=0,
        vol_max=100,
    )
    return StrategyConfig(
        name=f"late_resolution_{alias}",
        paper_mode=True,
        account_alias=alias,
        strategy_type="late_resolution",
        allowlist=[entry],
        blocklist_question_idxs=[],
        defaults=entry,
        **{
            "global": GlobalRiskConfig(
                max_total_inventory_usd=500,
                max_concurrent_positions=5,
                daily_loss_cap_usd=200,
                max_strike_distance_pct=50,
                min_recent_volume_usd=0,
                stale_data_halt_seconds=30,
                reconcile_interval_seconds=15,
            )
        },
    )


def _theta_strategy(alias: str) -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC"},
        max_position_usd=200,
        stop_loss_pct=None,
        tte_min_seconds=0,
        tte_max_seconds=86400,
        price_extreme_threshold=0.0,
        price_extreme_max=1.0,
        distance_from_strike_usd_min=0,
        vol_max=100,
    )
    return StrategyConfig(
        name=f"theta_harvester_{alias}",
        paper_mode=True,
        account_alias=alias,
        strategy_type="theta_harvester",
        allowlist=[entry],
        blocklist_question_idxs=[],
        defaults=entry,
        theta=ThetaParams(
            favorite_threshold=0.90,
            edge_buffer=0.03,
            topup_enabled=False,
            min_bid_notional_usd=10.0,
            exit_safety_d=1.0,
        ),
        **{
            "global": GlobalRiskConfig(
                max_total_inventory_usd=1000,
                max_concurrent_positions=5,
                daily_loss_cap_usd=100,
                max_strike_distance_pct=50,
                min_recent_volume_usd=100,
                stale_data_halt_seconds=30,
                reconcile_interval_seconds=15,
            )
        },
    )


@pytest.fixture
def three_slot_deploy(tmp_path):
    return DeployConfig(
        env="dev",
        accounts={
            "v1": HyperliquidAccount(
                account_address="0xv1",
                api_secret_key="0xv1",
                base_url="https://api.hyperliquid.xyz",
            ),
            "v31": HyperliquidAccount(
                account_address="0xv31",
                api_secret_key="0xv31",
                base_url="https://api.hyperliquid.xyz",
            ),
            "v31_pm": PolymarketAccount(
                clob_host="https://clob.polymarket.com",
                chain_id=137,
                private_key="0xstub",
                clob_api_key="stub",
                clob_api_secret="stub",
                clob_api_passphrase="stub",
            ),
        },
        alerts=AlertsConfig(telegram=TelegramConfig(bot_token="t", chat_id="c")),
        state_db_path=str(tmp_path / "state.db"),
        kill_switch_path=str(tmp_path / "halt"),
    )


def test_build_three_slots_dispatches_on_venue(three_slot_deploy):
    """_build_slot must produce an HLClient for HL accounts and a PMClient
    for PM accounts. No network — paper_mode=True everywhere."""
    strategies = [
        _late_resolution_strategy("v1"),
        _theta_strategy("v31"),
        _theta_strategy("v31_pm"),
    ]

    # exec_client_factory swaps HL slots (the constructor would try to derive
    # the SDK wallet from an invalid hex key otherwise). PM slot uses the
    # real PMClient path so we verify the venue dispatch end-to-end.
    def _hl_factory(alias: str, acct, paper: bool):
        if isinstance(acct, HyperliquidAccount):
            return HLClient(
                account_address=acct.account_address,
                api_secret_key=acct.api_secret_key,
                base_url=acct.base_url,
                paper_mode=True,
            )
        # PM slot — fall through to the runtime's built-in dispatch by
        # returning None... but the factory contract requires an instance.
        # Construct PMClient directly to mirror what runtime would do.
        return PMClient(
            paper_mode=True,
            clob_host=acct.clob_host,
            chain_id=acct.chain_id,
            private_key=acct.private_key,
            clob_api_key=acct.clob_api_key,
            clob_api_secret=acct.clob_api_secret,
            clob_api_passphrase=acct.clob_api_passphrase,
        )

    runtime = EngineRuntime(
        strategies=strategies,
        deploy_cfg=three_slot_deploy,
        adapter_factory=lambda: None,  # never invoked since we don't run()
        subscriptions=[],
        exec_client_factory=_hl_factory,
    )

    slots = [runtime._build_slot(s) for s in strategies]
    assert [s.alias for s in slots] == ["v1", "v31", "v31_pm"]
    assert isinstance(slots[0].exec_client, HLClient)
    assert isinstance(slots[1].exec_client, HLClient)
    assert isinstance(slots[2].exec_client, PMClient)
    # PM slot account config carried through verbatim.
    assert isinstance(slots[2].account_cfg, PolymarketAccount)
    assert slots[2].account_cfg.chain_id == 137


def test_build_pm_slot_without_factory_uses_pm_client(three_slot_deploy):
    """Sanity-check the no-factory fallback (the production path):
    runtime._build_slot must construct PMClient itself when given a
    PolymarketAccount."""
    s_pm = _theta_strategy("v31_pm")
    runtime = EngineRuntime(
        strategies=[s_pm],
        deploy_cfg=three_slot_deploy,
        adapter_factory=lambda: None,
        subscriptions=[],
        # No exec_client_factory — exercise the venue dispatch directly.
    )
    slot = runtime._build_slot(s_pm)
    assert isinstance(slot.exec_client, PMClient)
    assert slot.exec_client.paper_mode is True
