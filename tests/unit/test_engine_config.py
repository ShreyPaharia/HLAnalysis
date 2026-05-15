# tests/unit/test_engine_config.py
from __future__ import annotations

from pathlib import Path

import pytest

from hlanalysis.engine.config import (
    AllowlistEntry,
    DeployConfig,
    StrategyConfig,
    load_deploy_config,
    load_strategy_config,
    match_question,
)


def test_load_strategy_yaml_from_repo():
    cfg = load_strategy_config(Path("config/strategy.yaml"))
    assert cfg.name == "late_resolution"
    assert cfg.paper_mode is False
    assert cfg.global_.max_total_inventory_usd == 500
    assert cfg.global_.daily_loss_cap_usd == 200
    # v1-final values from the 1y PM walk-forward + focused (thr,max,stop) sweep.
    assert cfg.defaults.tte_max_seconds == 7200
    assert cfg.defaults.price_extreme_threshold == 0.85
    assert cfg.defaults.price_extreme_max == 0.97
    assert cfg.defaults.min_safety_d == 1.0
    assert cfg.defaults.vol_lookback_seconds == 3600
    assert cfg.defaults.exit_safety_d == 1.0
    assert cfg.defaults.vol_ewma_lambda == 0.85
    assert cfg.defaults.stop_loss_pct is None  # disabled per tuning
    btc_binary = next(
        e for e in cfg.allowlist if e.match.get("class") == "priceBinary"
    )
    assert btc_binary.price_extreme_max == 0.97
    assert btc_binary.min_safety_d == 1.0
    assert btc_binary.vol_lookback_seconds == 3600
    assert btc_binary.exit_safety_d == 1.0
    assert btc_binary.vol_ewma_lambda == 0.85
    assert btc_binary.stop_loss_pct is None
    btc_bucket = next(
        e for e in cfg.allowlist if e.match.get("class") == "priceBucket"
    )
    assert btc_bucket.exit_safety_d == 1.0
    assert btc_bucket.vol_ewma_lambda == 0.85


def test_allowlist_entry_safety_gate_defaults_when_omitted():
    """Older YAMLs without the safety-gate fields must keep loading; defaults
    preserve pre-gate behavior (max=1.0 is no cap, min_safety_d=0 is gate off)."""
    e = AllowlistEntry(
        match={"class": "priceBinary"},
        max_position_usd=100, stop_loss_pct=10, tte_min_seconds=0,
        tte_max_seconds=3600, price_extreme_threshold=0.95,
        distance_from_strike_usd_min=0, vol_max=100,
    )
    assert e.price_extreme_max == 1.0
    assert e.min_safety_d == 0.0
    assert e.vol_lookback_seconds == 1800
    assert e.exit_safety_d == 0.0
    assert e.vol_ewma_lambda == 0.0


def test_allowlist_entry_safety_gate_explicit_values():
    e = AllowlistEntry(
        match={"class": "priceBinary"},
        max_position_usd=100, stop_loss_pct=10, tte_min_seconds=0,
        tte_max_seconds=7200, price_extreme_threshold=0.90,
        distance_from_strike_usd_min=0, vol_max=100,
        price_extreme_max=0.995, min_safety_d=1.5, vol_lookback_seconds=3600,
        exit_safety_d=1.0, vol_ewma_lambda=0.85,
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
                max_position_usd=100, stop_loss_pct=10, tte_min_seconds=60,
                tte_max_seconds=1800, price_extreme_threshold=0.95,
                distance_from_strike_usd_min=200, vol_max=0.5,
            ),
        ],
        blocklist_question_idxs=[],
        defaults=AllowlistEntry(
            match={}, max_position_usd=50, stop_loss_pct=10, tte_min_seconds=60,
            tte_max_seconds=1800, price_extreme_threshold=0.95,
            distance_from_strike_usd_min=200, vol_max=0.5,
        ),
        global_={"max_total_inventory_usd": 500, "max_concurrent_positions": 5,
                 "daily_loss_cap_usd": 200, "max_strike_distance_pct": 10,
                 "min_recent_volume_usd": 1000, "stale_data_halt_seconds": 5,
                 "reconcile_interval_seconds": 60},
    )
    # Match: priceBinary BTC 1h
    matched = match_question(cfg, question_idx=42, fields={"class": "priceBinary", "underlying": "BTC", "period": "1h"})
    assert matched is not None
    assert matched.max_position_usd == 100

    # Blocklist override
    cfg2 = cfg.model_copy(update={"blocklist_question_idxs": [42]})
    assert match_question(cfg2, question_idx=42, fields={"class": "priceBinary", "underlying": "BTC", "period": "1h"}) is None

    # No match → None (defaults are not auto-applied for unmatched classes)
    assert match_question(cfg, question_idx=43, fields={"class": "priceBucket", "underlying": "ETH"}) is None


def test_deploy_config_substitutes_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0xdeadbeef")
    monkeypatch.setenv("HL_API_SECRET_KEY", "secret")
    monkeypatch.setenv("TG_BOT_TOKEN", "tg-token")
    monkeypatch.setenv("TG_CHAT_ID", "12345")
    p = tmp_path / "deploy.yaml"
    p.write_text(Path("config/deploy.yaml").read_text())
    cfg = load_deploy_config(p)
    assert cfg.hl.account_address == "0xdeadbeef"
    assert cfg.alerts.telegram.bot_token == "tg-token"


def test_deploy_config_missing_env_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("HL_ACCOUNT_ADDRESS", raising=False)
    p = tmp_path / "deploy.yaml"
    p.write_text(Path("config/deploy.yaml").read_text())
    with pytest.raises(ValueError):
        load_deploy_config(p)
