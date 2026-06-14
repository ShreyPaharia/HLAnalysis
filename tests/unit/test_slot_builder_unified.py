"""Task 4 — unified shared state DB + StrategyScopedDAL wiring tests.

Two tests per the spec:

1. test_slots_share_one_db_file
   Build two slots from one DeployConfig; assert both DALs resolve to the
   same underlying DB file (state_db_path_shared()), and that position writes
   via slot A's DAL are NOT visible via slot B's DAL (scoped isolation) but
   ARE present in the raw DB file.

2. test_per_strategy_sibling_dirs
   Assert that gate_decisions.jsonl and flag paths differ per strategy and
   both live under slot_dir_for(strategy_id), NOT under the shared DB root.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hlanalysis.engine._slot_builder import build_slot
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
from hlanalysis.engine.event_bus import EventBus
from hlanalysis.engine.market_state import MarketState
from hlanalysis.engine.scoped_dal import StrategyScopedDAL
from hlanalysis.engine.state import CachedStateDAL, Position

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _global_risk() -> GlobalRiskConfig:
    return GlobalRiskConfig(
        max_total_inventory_usd=500,
        max_concurrent_positions=5,
        daily_loss_cap_usd=200,
        max_strike_distance_pct=50,
        min_recent_volume_usd=0,
        stale_data_halt_seconds=30,
        reconcile_interval_seconds=60,
    )


def _lr_entry() -> AllowlistEntry:
    return AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC"},
        max_position_usd=100,
        stop_loss_pct=None,
        tte_min_seconds=0,
        tte_max_seconds=7200,
        price_extreme_threshold=0.90,
        distance_from_strike_usd_min=0,
        vol_max=100,
    )


def _late_resolution_cfg(alias: str) -> StrategyConfig:
    entry = _lr_entry()
    return StrategyConfig(
        name=f"late_resolution_{alias}",
        paper_mode=True,
        account_alias=alias,
        strategy_type="late_resolution",
        allowlist=[entry],
        blocklist_question_idxs=[],
        defaults=entry,
        **{"global": _global_risk()},
    )


def _theta_entry() -> AllowlistEntry:
    return AllowlistEntry(
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


def _theta_cfg(alias: str) -> StrategyConfig:
    entry = _theta_entry()
    return StrategyConfig(
        name=f"theta_{alias}",
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
        **{"global": _global_risk()},
    )


@pytest.fixture
def two_hl_deploy(tmp_path: Path) -> DeployConfig:
    """DeployConfig with two distinct HL accounts (v1 + v31)."""
    return DeployConfig(
        env="test",
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
        },
        alerts=AlertsConfig(telegram=TelegramConfig(bot_token="t", chat_id="c")),
        state_db_path=str(tmp_path / "engine" / "state.db"),
        kill_switch_path=str(tmp_path / "engine" / "halt"),
    )


@pytest.fixture
def mixed_deploy(tmp_path: Path) -> DeployConfig:
    """DeployConfig with one HL account (v1) and one PM account (v31_pm)."""
    return DeployConfig(
        env="test",
        accounts={
            "v1": HyperliquidAccount(
                account_address="0xv1",
                api_secret_key="0xv1",
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
        state_db_path=str(tmp_path / "engine" / "state.db"),
        kill_switch_path=str(tmp_path / "engine" / "halt"),
    )


def _build_two_slots(deploy_cfg: DeployConfig, cfgs: list[StrategyConfig], exec_client_factory=None):
    """Build two AccountSlot objects from a shared deploy config + two strategy configs.

    Uses a single shared CachedStateDAL (mimicking the EngineRuntime.run() path).
    """
    from hlanalysis.engine.hl_client import HLClient
    from hlanalysis.engine.pm_client import PMClient

    def _default_factory(alias: str, acct, paper: bool):
        if isinstance(acct, HyperliquidAccount):
            return HLClient(
                account_address=acct.account_address,
                api_secret_key=acct.api_secret_key,
                base_url=acct.base_url,
                paper_mode=True,
            )
        return PMClient(
            paper_mode=True,
            clob_host=acct.clob_host,
            chain_id=acct.chain_id,
            private_key=acct.private_key,
            clob_api_key=acct.clob_api_key,
            clob_api_secret=acct.clob_api_secret,
            clob_api_passphrase=acct.clob_api_passphrase,
        )

    factory = exec_client_factory or _default_factory

    # ONE shared DAL — same as EngineRuntime.run()
    shared_dal = CachedStateDAL(deploy_cfg.state_db_path_shared())
    shared_dal.run_migrations()

    bus = EventBus()
    ms = MarketState()

    slots = [
        build_slot(
            cfg,
            deploy_cfg=deploy_cfg,
            exec_client_factory=factory,
            bus=bus,
            market_state=ms,
            shared_dal=shared_dal,
        )
        for cfg in cfgs
    ]
    return slots, shared_dal


# ---------------------------------------------------------------------------
# Test 1: both slots share ONE DB file; scoped isolation works
# ---------------------------------------------------------------------------


def test_slots_share_one_db_file(two_hl_deploy: DeployConfig):
    """Both slot DALs must resolve to the SAME underlying DB file.

    Additionally:
    - A position written via slot A's DAL is NOT visible via slot B's DAL
      (StrategyScopedDAL scoping).
    - BUT the row IS present in the raw shared DB file (confirming shared
      physical storage).
    """
    deploy_cfg = two_hl_deploy
    shared_db_path = deploy_cfg.state_db_path_shared()

    cfgs = [_late_resolution_cfg("v1"), _theta_cfg("v31")]
    slots, shared_dal = _build_two_slots(deploy_cfg, cfgs)
    slot_a, slot_b = slots

    # Both DALs are StrategyScopedDAL instances
    assert isinstance(slot_a.dal, StrategyScopedDAL)
    assert isinstance(slot_b.dal, StrategyScopedDAL)

    # Both resolve to the same underlying CachedStateDAL (same object)
    assert slot_a.dal._base is slot_b.dal._base

    # The underlying DB file is the shared state_db_path_shared() path
    assert slot_a.dal._base.db_path == shared_db_path
    assert slot_b.dal._base.db_path == shared_db_path

    # strategy_ids must be distinct and equal to their account_alias
    assert slot_a.dal.strategy_id == "v1"
    assert slot_b.dal.strategy_id == "v31"
    assert slot_a.dal.strategy_id != slot_b.dal.strategy_id

    # --- Scoped isolation ---
    # Write a position via slot A
    pos_a = Position(
        strategy_id="",  # ignored; StrategyScopedDAL stamps its own
        question_idx=42,
        symbol="#42",
        qty=10.0,
        avg_entry=0.90,
        realized_pnl=0.0,
        last_update_ts_ns=1,
        stop_loss_price=0.80,
    )
    slot_a.dal.upsert_position(pos_a)

    # Slot A can read its own position
    assert slot_a.dal.get_position(42) is not None
    assert slot_a.dal.get_position(42).qty == 10.0

    # Slot B cannot see slot A's position (different strategy_id)
    assert slot_b.dal.get_position(42) is None
    assert len(slot_b.dal.all_positions()) == 0

    # The row IS in the raw shared DB (confirming shared physical storage)
    with sqlite3.connect(str(shared_db_path)) as conn:
        row = conn.execute("SELECT strategy_id, account, qty FROM position WHERE question_idx=42").fetchone()
    assert row is not None, "position row must be in the shared DB"
    assert row[0] == "v1", f"strategy_id in DB must be 'v1', got {row[0]!r}"
    assert row[1] == "v1", f"account in DB must be 'v1', got {row[1]!r}"
    assert abs(row[2] - 10.0) < 1e-9


# ---------------------------------------------------------------------------
# Test 2: per-strategy sibling dirs — gate_decisions.jsonl and flag paths
# ---------------------------------------------------------------------------


def test_per_strategy_sibling_dirs(two_hl_deploy: DeployConfig):
    """gate_decisions.jsonl and kill-switch flag paths must be DISTINCT per
    strategy and both live inside slot_dir_for(strategy_id), NOT under the
    shared-DB root.

    Layout expectation:
        <state_db_path.parent>/          ← shared DB root
            state.db                     ← one shared DB
            v1/                          ← slot_dir_for("v1")
                halt                     ← kill switch
                gate_decisions.jsonl
            v31/                         ← slot_dir_for("v31")
                halt
                gate_decisions.jsonl
    """
    deploy_cfg = two_hl_deploy
    shared_db_path = deploy_cfg.state_db_path_shared()

    cfgs = [_late_resolution_cfg("v1"), _theta_cfg("v31")]
    slots, _ = _build_two_slots(deploy_cfg, cfgs)
    slot_v1, slot_v31 = slots

    slot_dir_v1 = deploy_cfg.slot_dir_for("v1")
    slot_dir_v31 = deploy_cfg.slot_dir_for("v31")
    shared_root = shared_db_path.parent

    # Slot dirs must be distinct and must NOT be the shared-DB root
    assert slot_dir_v1 != slot_dir_v31
    assert slot_dir_v1 != shared_root
    assert slot_dir_v31 != shared_root

    # slot dirs must be children of the shared root
    assert slot_dir_v1.parent == shared_root
    assert slot_dir_v31.parent == shared_root

    # Kill-switch paths must be in their respective slot directories
    ks_v1 = slot_v1.kill_switch_path
    ks_v31 = slot_v31.kill_switch_path
    assert ks_v1 != ks_v31
    assert ks_v1.parent == slot_dir_v1, f"v1 kill-switch must be inside slot_dir_for('v1'), got parent {ks_v1.parent}"
    assert ks_v31.parent == slot_dir_v31, (
        f"v31 kill-switch must be inside slot_dir_for('v31'), got parent {ks_v31.parent}"
    )

    # gate_decisions.jsonl paths: resolved via scanner's gate_log_path.
    # The scanner is built with gate_log_path = slot_dir_for(alias) / "gate_decisions.jsonl".
    # We verify this by checking the scanner's gate_log_path attribute.
    gate_v1 = slot_v1.scanner.gate_log_path
    gate_v31 = slot_v31.scanner.gate_log_path
    assert gate_v1 != gate_v31
    assert gate_v1.parent == slot_dir_v1, (
        f"v1 gate_decisions.jsonl must be in slot_dir_for('v1'), got parent {gate_v1.parent}"
    )
    assert gate_v31.parent == slot_dir_v31, (
        f"v31 gate_decisions.jsonl must be in slot_dir_for('v31'), got parent {gate_v31.parent}"
    )

    # Shared DB must NOT live inside either slot dir
    assert shared_db_path.parent == shared_root
    assert shared_db_path.parent != slot_dir_v1
    assert shared_db_path.parent != slot_dir_v31

    # The slot directories were created by build_slot (mkdir parents=True)
    assert slot_dir_v1.exists()
    assert slot_dir_v31.exists()
