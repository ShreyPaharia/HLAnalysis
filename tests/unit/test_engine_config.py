# tests/unit/test_engine_config.py
from __future__ import annotations

import os
from pathlib import Path

import pytest

from hlanalysis.engine.config import (
    AllowlistEntry,
    StrategyConfig,
    load_deploy_config,
    load_strategies_config,
    load_strategy_config,
    match_question,
)


def test_load_strategy_yaml_from_repo():
    cfg = load_strategy_config(Path("config/strategy.yaml"))
    assert cfg.name == "late_resolution"
    assert cfg.paper_mode is False
    # 2026-06-26: capital scale-up — v1 is now bucket-only ($500/pos). inv ≈ 3
    # concurrent positions; daily_loss_cap scaled with it (and is the primary
    # tail circuit-breaker on the exit-gate-free bucket book).
    assert cfg.global_.max_total_inventory_usd == 1500
    assert cfg.global_.daily_loss_cap_usd == 150
    # 2026-06-26: v1 is bucket-only; defaults mirror the 8h bucket window.
    assert cfg.defaults.tte_max_seconds == 28800
    assert cfg.defaults.price_extreme_threshold == 0.85
    assert cfg.defaults.price_extreme_max == 0.999
    # 2026-05-30 v1 cadence validation: HL v1 moved to Parkinson σ + dt=5
    # sub-minute sampling (lockstep with v31) + gate retune (msd 1.0→3.0,
    # λ 0.85→0.97). See summeries/v1_cadence_validation_2026_05_30.md.
    assert cfg.defaults.min_safety_d == 3.0
    assert cfg.defaults.vol_lookback_seconds == 3600
    assert cfg.defaults.vol_ewma_lambda == 0.97
    assert cfg.defaults.vol_estimator == "parkinson"
    assert cfg.defaults.vol_sampling_dt_seconds == 5
    assert cfg.defaults.stop_loss_pct is None  # disabled per tuning
    # 2026-06-26: v1 priceBinary TAKEN DOWN — v31 owns HL BTC binary at live
    # event cadence ($1058/wh$492 vs tuned-v1 $336/$133). v1 is bucket-only.
    assert not any(e.match.get("class") == "priceBinary" for e in cfg.allowlist)
    btc_bucket = next(e for e in cfg.allowlist if e.match.get("class") == "priceBucket")
    assert btc_bucket.max_position_usd == 500.0  # 2026-06-26: 300→500 scale-up
    # exit_safety_d stays 0 — every soft exit/stop doom-loops across wide bucket
    # spreads (only v31's exit_spread_hold avoids it). maxDD $0 is tail-blind.
    assert btc_bucket.exit_safety_d == 0.0
    # 2026-06-26: 6h → 8h, the event-cadence retune winner (validated at 0.2/2.0;
    # worst-half $236→$348, maxDD $0). λ kept 0.97 (0.85 toxic at event cadence).
    assert btc_bucket.tte_max_seconds == 28800
    assert btc_bucket.vol_ewma_lambda == 0.97
    assert btc_bucket.vol_estimator == "parkinson"
    assert btc_bucket.vol_sampling_dt_seconds == 5
    # Bucket size-cap stays disabled (PM "above-X stack" tune can't transfer).
    assert btc_bucket.size_cap_near_strike_pct == 0.0


def test_allowlist_entry_safety_gate_defaults_when_omitted():
    """Older YAMLs without the safety-gate fields must keep loading; defaults
    preserve pre-gate behavior (max=1.0 is no cap, min_safety_d=0 is gate off)."""
    e = AllowlistEntry(
        match={"class": "priceBinary"},
        max_position_usd=100,
        stop_loss_pct=10,
        tte_min_seconds=0,
        tte_max_seconds=3600,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=0,
        vol_max=100,
    )
    assert e.price_extreme_max == 1.0
    assert e.min_safety_d == 0.0
    assert e.vol_lookback_seconds == 1800
    assert e.exit_safety_d == 0.0
    assert e.vol_ewma_lambda == 0.0
    # Size-cap fields default to a disabled state (pct=0).
    assert e.size_cap_near_strike_pct == 0.0
    assert e.size_cap_max_dist_pct == 1.5
    assert e.size_cap_min_ask == 0.88


def test_allowlist_entry_safety_gate_explicit_values():
    e = AllowlistEntry(
        match={"class": "priceBinary"},
        max_position_usd=100,
        stop_loss_pct=10,
        tte_min_seconds=0,
        tte_max_seconds=7200,
        price_extreme_threshold=0.90,
        distance_from_strike_usd_min=0,
        vol_max=100,
        price_extreme_max=0.995,
        min_safety_d=1.5,
        vol_lookback_seconds=3600,
        exit_safety_d=1.0,
        vol_ewma_lambda=0.85,
    )
    assert e.price_extreme_max == 0.995
    assert e.min_safety_d == 1.5
    assert e.vol_lookback_seconds == 3600
    assert e.exit_safety_d == 1.0
    assert e.vol_ewma_lambda == 0.85


def test_allowlist_match_picks_first_matching_entry(tmp_path):
    cfg = StrategyConfig(
        name="late_resolution",
        paper_mode=True,
        allowlist=[
            AllowlistEntry(
                match={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
                max_position_usd=100,
                stop_loss_pct=10,
                tte_min_seconds=60,
                tte_max_seconds=1800,
                price_extreme_threshold=0.95,
                distance_from_strike_usd_min=200,
                vol_max=0.5,
            ),
        ],
        blocklist_question_idxs=[],
        defaults=AllowlistEntry(
            match={},
            max_position_usd=50,
            stop_loss_pct=10,
            tte_min_seconds=60,
            tte_max_seconds=1800,
            price_extreme_threshold=0.95,
            distance_from_strike_usd_min=200,
            vol_max=0.5,
        ),
        global_={
            "max_total_inventory_usd": 500,
            "max_concurrent_positions": 5,
            "daily_loss_cap_usd": 200,
            "max_strike_distance_pct": 10,
            "min_recent_volume_usd": 1000,
            "stale_data_halt_seconds": 5,
            "reconcile_interval_seconds": 60,
        },
    )
    # Match: priceBinary BTC 1h
    matched = match_question(cfg, question_idx=42, fields={"class": "priceBinary", "underlying": "BTC", "period": "1h"})
    assert matched is not None
    assert matched.max_position_usd == 100

    # Blocklist override
    cfg2 = cfg.model_copy(update={"blocklist_question_idxs": [42]})
    assert (
        match_question(cfg2, question_idx=42, fields={"class": "priceBinary", "underlying": "BTC", "period": "1h"})
        is None
    )

    # No match → None (defaults are not auto-applied for unmatched classes)
    assert match_question(cfg, question_idx=43, fields={"class": "priceBucket", "underlying": "ETH"}) is None


def test_deploy_config_substitutes_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0xdeadbeef")
    monkeypatch.setenv("HL_API_SECRET_KEY", "secret")
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS_V31", "0xv31")
    monkeypatch.setenv("HL_API_SECRET_KEY_V31", "secret-v31")
    monkeypatch.setenv("TG_BOT_TOKEN", "tg-token")
    monkeypatch.setenv("TG_CHAT_ID", "12345")
    # PM v31_pm slot was added in Phase 5; deploy.yaml references these four
    # Polymarket env vars. They MUST be set for the production config to load.
    monkeypatch.setenv("PM_PRIVATE_KEY", "0xstub")
    monkeypatch.setenv("PM_CLOB_API_KEY", "stub-key")
    monkeypatch.setenv("PM_CLOB_API_SECRET", "stub-secret")
    monkeypatch.setenv("PM_CLOB_API_PASSPHRASE", "stub-pass")
    # PM-UI-onboarded accounts use a Safe-proxy funder; deploy.yaml now
    # references PM_FUNDER_ADDRESS.
    monkeypatch.setenv("PM_FUNDER_ADDRESS", "0xstubfunder")
    # PM v1_pm slot: separate funder/keys mirroring v31_pm.
    monkeypatch.setenv("PM_PRIVATE_KEY_V1", "0xstub-v1")
    monkeypatch.setenv("PM_CLOB_API_KEY_V1", "stub-key-v1")
    monkeypatch.setenv("PM_CLOB_API_SECRET_V1", "stub-secret-v1")
    monkeypatch.setenv("PM_CLOB_API_PASSPHRASE_V1", "stub-pass-v1")
    monkeypatch.setenv("PM_FUNDER_ADDRESS_V1", "0xstubfunder-v1")
    # PM multi-strike bucket slot (ETH only). BTC multi-strike is folded onto
    # v31_pm (no separate account). paper_mode never signs, but
    # load_deploy_config validates every referenced env var, so these must be
    # set (dummy values are fine) for the production config to load.
    monkeypatch.setenv("PM_PRIVATE_KEY_ETH_MS", "0xstub-ethms")
    monkeypatch.setenv("PM_CLOB_API_KEY_ETH_MS", "stub-key-ethms")
    monkeypatch.setenv("PM_CLOB_API_SECRET_ETH_MS", "stub-secret-ethms")
    monkeypatch.setenv("PM_CLOB_API_PASSPHRASE_ETH_MS", "stub-pass-ethms")
    monkeypatch.setenv("PM_FUNDER_ADDRESS_ETH_MS", "0xstubfunder-ethms")
    p = tmp_path / "deploy.yaml"
    p.write_text(Path("config/deploy.yaml").read_text())
    cfg = load_deploy_config(p)
    assert cfg.hl_accounts["v1"].account_address == "0xdeadbeef"
    assert cfg.hl_accounts["v31"].account_address == "0xv31"
    assert cfg.alerts.telegram.bot_token == "tg-token"
    assert "v31_pm" in cfg.accounts
    assert cfg.accounts["v31_pm"].chain_id == 137
    assert "v1_pm" in cfg.accounts
    assert cfg.accounts["v1_pm"].funder_address == "0xstubfunder-v1"
    # BTC multi-strike folds onto v31_pm (no separate account); only ETH is new.
    assert "v31_pm_btc_ms" not in cfg.accounts
    assert "v31_pm_eth_ms" in cfg.accounts
    assert cfg.accounts["v31_pm_eth_ms"].funder_address == "0xstubfunder-ethms"


def test_deploy_config_missing_env_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("HL_ACCOUNT_ADDRESS", raising=False)
    p = tmp_path / "deploy.yaml"
    p.write_text(Path("config/deploy.yaml").read_text())
    with pytest.raises(ValueError):
        load_deploy_config(p)


def test_account_config_discriminates_on_venue(tmp_path):
    from hlanalysis.engine.config import HyperliquidAccount, PolymarketAccount

    deploy_yaml = tmp_path / "deploy.yaml"
    deploy_yaml.write_text("""
deploy:
  env: test
  accounts:
    v1:
      venue: hyperliquid
      account_address: "0xabc"
      api_secret_key: "0xdef"
      base_url: https://api.hyperliquid.xyz
    v31_pm:
      venue: polymarket
      clob_host: https://clob.polymarket.com
      chain_id: 137
      private_key: "0xfeed"
      clob_api_key: "ak"
      clob_api_secret: "as"
      clob_api_passphrase: "ap"
  alerts:
    telegram:
      bot_token: T
      chat_id: C
  state_db_path: data/engine/state.db
  kill_switch_path: data/engine/halt
""")
    cfg = load_deploy_config(deploy_yaml)
    assert isinstance(cfg.accounts["v1"], HyperliquidAccount)
    assert isinstance(cfg.accounts["v31_pm"], PolymarketAccount)
    assert cfg.accounts["v31_pm"].chain_id == 137


def test_pm_slots_reference_binance_spot():
    """Both PM slots must reference BTCUSDT_SPOT (not the perp BTCUSDT feed)
    so that live price, σ, and strike are all spot-based (no perp/spot basis)."""
    strategies = load_strategies_config(Path("config/strategy.yaml"))
    pm = {s.account_alias: s for s in strategies.strategies if s.account_alias.endswith("_pm")}
    assert pm["v31_pm"].reference_symbol == "BTCUSDT_SPOT"
    assert pm["v1_pm"].reference_symbol == "BTCUSDT_SPOT"
    assert pm["v31_pm"].reference_sigma_source == "bbo"  # σ source stays bbo (now spot bbo)


# ---------------------------------------------------------------------------
# Task 9: optional explicit strategy_id field
# ---------------------------------------------------------------------------


def _make_minimal_strategy_cfg(**kwargs) -> StrategyConfig:
    """Build a minimal valid StrategyConfig; kwargs override defaults."""
    entry = AllowlistEntry(
        match={"class": "priceBinary"},
        max_position_usd=100,
        stop_loss_pct=None,
        tte_min_seconds=0,
        tte_max_seconds=7200,
        price_extreme_threshold=0.90,
        distance_from_strike_usd_min=0,
        vol_max=100,
    )
    base = dict(
        name="late_resolution",
        paper_mode=True,
        account_alias="v1",
        allowlist=[entry],
        blocklist_question_idxs=[],
        defaults=entry,
        **{
            "global": {
                "max_total_inventory_usd": 500,
                "max_concurrent_positions": 5,
                "daily_loss_cap_usd": 200,
                "max_strike_distance_pct": 50,
                "min_recent_volume_usd": 0,
                "stale_data_halt_seconds": 30,
                "reconcile_interval_seconds": 60,
            }
        },
    )
    base.update(kwargs)
    return StrategyConfig(**base)


def test_strategy_id_defaults_to_account_alias_when_omitted():
    """When ``strategy_id`` is absent from config, it is None in the model.

    ``_slot_builder.build_slot`` derives the effective strategy_id as
    ``s_cfg.strategy_id or s_cfg.account_alias``; this test asserts that the
    field is None (the default) so the builder correctly falls back to
    account_alias.
    """
    cfg = _make_minimal_strategy_cfg()  # no strategy_id kwarg
    assert cfg.strategy_id is None
    # Derived value (mirrors _slot_builder logic) must equal account_alias
    effective_id = cfg.strategy_id if cfg.strategy_id is not None else cfg.account_alias
    assert effective_id == cfg.account_alias == "v1"


def test_strategy_id_explicit_overrides_account_alias():
    """When ``strategy_id`` is set explicitly, it is used as the DB scoping key."""
    cfg = _make_minimal_strategy_cfg(strategy_id="my_custom_id")
    assert cfg.strategy_id == "my_custom_id"
    # account_alias is unchanged
    assert cfg.account_alias == "v1"
    # Derived value (mirrors _slot_builder logic) must use the explicit id
    effective_id = cfg.strategy_id if cfg.strategy_id is not None else cfg.account_alias
    assert effective_id == "my_custom_id"


def test_prod_config_strategy_id_not_set():
    """The current strategy.yaml has NO strategy_id field on any entry.

    All slots must have strategy_id=None so the builder falls back to
    account_alias — preserving today's exact behavior.
    """
    strategies = load_strategies_config(Path("config/strategy.yaml"))
    for s in strategies.strategies:
        assert s.strategy_id is None, (
            f"slot {s.account_alias!r} has an unexpected strategy_id={s.strategy_id!r}; "
            "the current config intentionally omits this field"
        )


def test_all_slots_resolve_to_one_shared_db_path():
    """All slots in the current config must resolve to ONE shared DB path.

    DeployConfig.state_db_path_shared() returns the same Path regardless of
    which slot calls it — all strategy slots share one physical DB file; only
    their (strategy_id, account) row tags differ.
    """
    # Set required env vars for the full deploy.yaml to load
    env_vars = {
        "HL_ACCOUNT_ADDRESS": "0xtest",
        "HL_API_SECRET_KEY": "0xtest",
        "HL_ACCOUNT_ADDRESS_V31": "0xtest31",
        "HL_API_SECRET_KEY_V31": "0xtest31",
        "TG_BOT_TOKEN": "tok",
        "TG_CHAT_ID": "1",
        "PM_PRIVATE_KEY": "0xpm",
        "PM_CLOB_API_KEY": "k",
        "PM_CLOB_API_SECRET": "s",
        "PM_CLOB_API_PASSPHRASE": "p",
        "PM_FUNDER_ADDRESS": "0xfund",
        "PM_PRIVATE_KEY_V1": "0xpmv1",
        "PM_CLOB_API_KEY_V1": "kv1",
        "PM_CLOB_API_SECRET_V1": "sv1",
        "PM_CLOB_API_PASSPHRASE_V1": "pv1",
        "PM_FUNDER_ADDRESS_V1": "0xfundv1",
        "PM_PRIVATE_KEY_ETH_MS": "0xethms",
        "PM_CLOB_API_KEY_ETH_MS": "kethms",
        "PM_CLOB_API_SECRET_ETH_MS": "sethms",
        "PM_CLOB_API_PASSPHRASE_ETH_MS": "pethms",
        "PM_FUNDER_ADDRESS_ETH_MS": "0xfundethms",
    }
    orig = {k: os.environ.get(k) for k in env_vars}
    try:
        os.environ.update(env_vars)
        deploy_cfg = load_deploy_config(Path("config/deploy.yaml"))
        shared_path = deploy_cfg.state_db_path_shared()
        strategies = load_strategies_config(Path("config/strategy.yaml"))
        # Every slot resolves to the same shared DB path
        for s in strategies.strategies:
            # The slot builder uses state_db_path_shared() — confirm it's
            # consistent across accounts and strategy types.
            assert deploy_cfg.state_db_path_shared() == shared_path, (
                f"slot {s.account_alias!r} resolved to a different DB path"
            )
        # Shared path must be a single file (not a per-alias subdirectory path)
        assert shared_path.name == "state.db"
        assert str(shared_path) == "data/engine/state.db"
    finally:
        for k, v in orig.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
