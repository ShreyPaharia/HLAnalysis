"""AccountSlot venue cleanup (M2): one `venue` field set at build time, one
`slot.is_pm` predicate, and PM-only mutable state isolated in a
`pm: PmSlotState | None` sub-object instead of five loose fields plus three
ad-hoc "is this PM?" checks.
"""

from __future__ import annotations

import pytest

from hlanalysis.engine.runtime import AccountSlot, EngineRuntime, PmSlotState


def _build_runtime(tmp_path, *, polymarket: bool):
    """Build a real EngineRuntime with a single slot of the requested venue,
    using a stub exec_client_factory so neither venue touches the network."""
    from hlanalysis.engine.config import (
        AlertsConfig,
        AllowlistEntry,
        DeployConfig,
        GlobalRiskConfig,
        HyperliquidAccount,
        PolymarketAccount,
        StrategyConfig,
        TelegramConfig,
    )

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
    alias = "v31_pm" if polymarket else "v1"
    strat = StrategyConfig(
        name="late_resolution",
        account_alias=alias,
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
    if polymarket:
        acct = PolymarketAccount(
            private_key="0x0",
            clob_api_key="k",
            clob_api_secret="s",
            clob_api_passphrase="p",
        )
    else:
        acct = HyperliquidAccount(
            account_address="0x0",
            api_secret_key="0x0",
            base_url="https://api.hyperliquid.xyz",
        )
    deploy = DeployConfig(
        env="dev",
        accounts={alias: acct},
        alerts=AlertsConfig(telegram=TelegramConfig(bot_token="x", chat_id="y")),
        state_db_path=str(tmp_path / "state.db"),
        kill_switch_path=str(tmp_path / "halt"),
    )

    class _StubExec:
        def realized_pnl_since(self, ns):  # noqa: D401
            return 0.0

    rt = EngineRuntime(
        strategies=[strat],
        deploy_cfg=deploy,
        adapter_factory=lambda: None,
        subscriptions=[],
        exec_client_factory=lambda _a, _c, _p: _StubExec(),
    )
    return rt, rt._build_slot(strat)


def test_pm_slot_state_defaults():
    pm = PmSlotState()
    assert pm.alerted_unconfirmed_cloids == set()
    assert pm.settlements == {}
    assert pm.alerted_redemption_qidxs == set()
    assert pm.startup_position_synced is False


def test_hl_slot_venue_and_no_pm_state(tmp_path):
    rt, slot = _build_runtime(tmp_path, polymarket=False)
    assert slot.venue == "hyperliquid"
    assert slot.is_pm is False
    assert slot.pm is None


def test_pm_slot_venue_and_pm_state(tmp_path):
    rt, slot = _build_runtime(tmp_path, polymarket=True)
    assert slot.venue == "polymarket"
    assert slot.is_pm is True
    assert isinstance(slot.pm, PmSlotState)
