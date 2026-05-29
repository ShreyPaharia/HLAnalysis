# tests/unit/test_engine_runtime_cadence.py
"""v3.7 cadence port: the live engine must couple the shared MarketState's
per-symbol mark-bucket period to each slot's vol_sampling_dt_seconds, with no
train/serve skew, the 60s default preserved, and HL/PM independent.
"""
from __future__ import annotations

import pytest

from hlanalysis.engine.config import (
    AllowlistEntry,
    DeployConfig,
    GlobalRiskConfig,
    HyperliquidAccount,
    PolymarketAccount,
    StrategyConfig,
    ThetaParams,
)
from hlanalysis.engine.config import AlertsConfig, TelegramConfig
from hlanalysis.engine.runtime import (
    EngineRuntime,
    reference_sampling_dt_seconds,
    reference_vol_lookback_seconds,
)

_NS = 1_000_000_000


class _FakeExec:
    """Minimal ExecutionClient stand-in — _build_slot only needs an object with
    realized_pnl_since to wire the Scanner's pnl_provider; it is never called
    during cadence registration."""

    def realized_pnl_since(self, _ns: int) -> float:  # pragma: no cover - unused
        return 0.0


def _global() -> GlobalRiskConfig:
    return GlobalRiskConfig(
        max_total_inventory_usd=500, max_concurrent_positions=5,
        daily_loss_cap_usd=200, max_strike_distance_pct=50,
        min_recent_volume_usd=100, stale_data_halt_seconds=30,
        reconcile_interval_seconds=60,
    )


def _theta(dt: int) -> ThetaParams:
    return ThetaParams(vol_lookback_seconds=3600, vol_sampling_dt_seconds=dt)


def _theta_cfg(*, alias: str, reference_symbol: str, dt: int) -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC"},
        max_position_usd=100, stop_loss_pct=None, tte_min_seconds=0,
        tte_max_seconds=43200, price_extreme_threshold=0.0,
        distance_from_strike_usd_min=0, vol_max=100,
    )
    return StrategyConfig(
        name="theta_harvester", account_alias=alias, paper_mode=True,
        strategy_type="theta_harvester", reference_symbol=reference_symbol,
        allowlist=[entry], blocklist_question_idxs=[], defaults=entry,
        theta=_theta(dt), **{"global": _global()},
    )


def _late_cfg(*, alias: str, reference_symbol: str) -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC"},
        max_position_usd=100, stop_loss_pct=None, tte_min_seconds=0,
        tte_max_seconds=7200, price_extreme_threshold=0.85,
        distance_from_strike_usd_min=0, vol_max=100, vol_lookback_seconds=3600,
    )
    return StrategyConfig(
        name="late_resolution", account_alias=alias, paper_mode=True,
        strategy_type="late_resolution", reference_symbol=reference_symbol,
        allowlist=[entry], blocklist_question_idxs=[], defaults=entry,
        **{"global": _global()},
    )


def _runtime(strategies, tmp_path) -> EngineRuntime:
    accounts = {}
    for s in strategies:
        if s.reference_symbol == "BTCUSDT":
            accounts[s.account_alias] = PolymarketAccount(
                private_key="0x0", clob_api_key="k", clob_api_secret="s",
                clob_api_passphrase="p",
            )
        else:
            accounts[s.account_alias] = HyperliquidAccount(
                account_address="0x0", api_secret_key="0x0",
                base_url="https://api.hyperliquid.xyz",
            )
    deploy = DeployConfig(
        env="dev", accounts=accounts,
        alerts=AlertsConfig(telegram=TelegramConfig(bot_token="x", chat_id="y")),
        state_db_path=str(tmp_path / "state.db"),
        kill_switch_path=str(tmp_path / "halt"),
    )
    rt = EngineRuntime(
        strategies=strategies, deploy_cfg=deploy,
        adapter_factory=lambda: None, subscriptions=[],
        exec_client_factory=lambda _a, _c, _p: _FakeExec(),
    )
    rt.slots = [rt._build_slot(s) for s in strategies]
    return rt


# ---- helpers ---------------------------------------------------------------

def test_reference_sampling_dt_defaults_to_60_for_late_resolution():
    cfg = _late_cfg(alias="v1", reference_symbol="BTC")
    assert reference_sampling_dt_seconds(cfg) == 60


def test_reference_sampling_dt_reads_theta_block():
    assert reference_sampling_dt_seconds(
        _theta_cfg(alias="v31", reference_symbol="BTC", dt=5)
    ) == 5
    assert reference_vol_lookback_seconds(
        _theta_cfg(alias="v31", reference_symbol="BTC", dt=5)
    ) == 3600


# ---- live wiring: no skew, default preserved, HL/PM independent ------------

def test_registration_default_60s_preserves_behavior(tmp_path):
    """Production-shaped slots all at dt=60 → every reference symbol still
    buckets at 60s (existing deployments unchanged)."""
    rt = _runtime(
        [
            _late_cfg(alias="v1", reference_symbol="BTC"),
            _theta_cfg(alias="v31", reference_symbol="BTC", dt=60),
            _theta_cfg(alias="v31_pm", reference_symbol="BTCUSDT", dt=60),
        ],
        tmp_path,
    )
    rt._register_reference_cadences(rt.slots)
    assert rt.market_state.mark_bucket_ns_for("BTC") == 60 * _NS
    assert rt.market_state.mark_bucket_ns_for("BTCUSDT") == 60 * _NS


def test_no_train_serve_skew_period_equals_strategy_assumption(tmp_path):
    """Acceptance: the period MarketState buckets at == the period the
    strategy's σ formula assumes, for every slot's reference symbol."""
    strategies = [
        _theta_cfg(alias="v31", reference_symbol="BTC", dt=5),
        _theta_cfg(alias="v31_pm", reference_symbol="BTCUSDT", dt=60),
    ]
    rt = _runtime(strategies, tmp_path)
    rt._register_reference_cadences(rt.slots)
    for cfg in strategies:
        assert rt.market_state.mark_bucket_ns_for(cfg.reference_symbol) == (
            reference_sampling_dt_seconds(cfg) * _NS
        )


def test_hl_pm_independent(tmp_path):
    """Flipping the HL slot to dt=5 does NOT affect the PM reference symbol."""
    rt = _runtime(
        [
            _theta_cfg(alias="v31", reference_symbol="BTC", dt=5),
            _theta_cfg(alias="v31_pm", reference_symbol="BTCUSDT", dt=60),
        ],
        tmp_path,
    )
    rt._register_reference_cadences(rt.slots)
    assert rt.market_state.mark_bucket_ns_for("BTC") == 5 * _NS
    assert rt.market_state.mark_bucket_ns_for("BTCUSDT") == 60 * _NS


def test_conflicting_cadence_same_symbol_raises(tmp_path):
    """Two slots reading the SAME reference symbol with different dt is an
    unsatisfiable request (one shared mark history) — must fail fast at startup
    rather than silently skew one of them."""
    rt = _runtime(
        [
            _theta_cfg(alias="v31", reference_symbol="BTC", dt=5),
            _theta_cfg(alias="v31b", reference_symbol="BTC", dt=60),
        ],
        tmp_path,
    )
    with pytest.raises(ValueError, match="conflicting mark-bucket cadence"):
        rt._register_reference_cadences(rt.slots)
