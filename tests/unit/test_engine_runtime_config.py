# tests/unit/test_engine_runtime_config.py
"""Smoke test that the runtime path threads safety-gate fields from a loaded
strategy YAML into the LateResolutionConfig handed to the strategy."""
from __future__ import annotations

from pathlib import Path

from hlanalysis.engine.config import load_strategy_config
from hlanalysis.engine.runtime import build_late_resolution_config


_YAML_WITH_GATES = """
strategy:
  name: late_resolution
  paper_mode: true
  allowlist:
    - match: {class: priceBinary, underlying: BTC}
      max_position_usd: 100
      stop_loss_pct: 10
      tte_min_seconds: 0
      tte_max_seconds: 7200
      price_extreme_threshold: 0.90
      price_extreme_max: 0.995
      min_safety_d: 1.5
      vol_lookback_seconds: 3600
      exit_safety_d: 1.0
      vol_ewma_lambda: 0.85
      vol_estimator: parkinson
      vol_sampling_dt_seconds: 5
      size_cap_near_strike_pct: 0.5
      size_cap_max_dist_pct: 1.0
      size_cap_min_ask: 0.88
      distance_from_strike_usd_min: 0
      vol_max: 100
  blocklist_question_idxs: []
  defaults:
    match: {}
    max_position_usd: 100
    stop_loss_pct: 10
    tte_min_seconds: 0
    tte_max_seconds: 7200
    price_extreme_threshold: 0.90
    price_extreme_max: 0.995
    min_safety_d: 1.5
    vol_lookback_seconds: 3600
    exit_safety_d: 1.0
    vol_ewma_lambda: 0.85
    vol_estimator: parkinson
    vol_sampling_dt_seconds: 5
    size_cap_near_strike_pct: 0.5
    size_cap_max_dist_pct: 1.0
    size_cap_min_ask: 0.88
    distance_from_strike_usd_min: 0
    vol_max: 100
  global:
    max_total_inventory_usd: 500
    max_concurrent_positions: 5
    daily_loss_cap_usd: 200
    max_strike_distance_pct: 50
    min_recent_volume_usd: 1000
    stale_data_halt_seconds: 5
    reconcile_interval_seconds: 60
"""


_YAML_NO_GATES = """
strategy:
  name: late_resolution
  paper_mode: true
  allowlist:
    - match: {class: priceBinary, underlying: BTC}
      max_position_usd: 100
      stop_loss_pct: 10
      tte_min_seconds: 0
      tte_max_seconds: 3600
      price_extreme_threshold: 0.95
      distance_from_strike_usd_min: 0
      vol_max: 100
  blocklist_question_idxs: []
  defaults:
    match: {}
    max_position_usd: 100
    stop_loss_pct: 10
    tte_min_seconds: 0
    tte_max_seconds: 3600
    price_extreme_threshold: 0.95
    distance_from_strike_usd_min: 0
    vol_max: 100
  global:
    max_total_inventory_usd: 500
    max_concurrent_positions: 5
    daily_loss_cap_usd: 200
    max_strike_distance_pct: 50
    min_recent_volume_usd: 1000
    stale_data_halt_seconds: 5
    reconcile_interval_seconds: 60
"""


def test_runtime_threads_safety_gate_fields_into_strategy_config(tmp_path: Path):
    p = tmp_path / "strategy.yaml"
    p.write_text(_YAML_WITH_GATES)
    cfg = load_strategy_config(p)
    rcfg = build_late_resolution_config(cfg)
    assert rcfg.tte_max_seconds == 7200
    assert rcfg.price_extreme_threshold == 0.90
    assert rcfg.price_extreme_max == 0.995
    assert rcfg.min_safety_d == 1.5
    assert rcfg.vol_lookback_seconds == 3600
    assert rcfg.exit_safety_d == 1.0
    assert rcfg.vol_ewma_lambda == 0.85
    assert rcfg.vol_estimator == "parkinson"
    assert rcfg.vol_sampling_dt_seconds == 5
    assert rcfg.size_cap_near_strike_pct == 0.5
    assert rcfg.size_cap_max_dist_pct == 1.0
    assert rcfg.size_cap_min_ask == 0.88


def test_runtime_falls_back_to_defaults_when_yaml_omits_safety_gates(tmp_path: Path):
    p = tmp_path / "strategy.yaml"
    p.write_text(_YAML_NO_GATES)
    cfg = load_strategy_config(p)
    rcfg = build_late_resolution_config(cfg)
    assert rcfg.price_extreme_max == 1.0
    assert rcfg.min_safety_d == 0.0
    assert rcfg.vol_lookback_seconds == 1800
    assert rcfg.exit_safety_d == 0.0
    assert rcfg.vol_ewma_lambda == 0.0
    assert rcfg.vol_estimator == "stdev"
    assert rcfg.vol_sampling_dt_seconds == 60
    assert rcfg.size_cap_near_strike_pct == 0.0
    assert rcfg.size_cap_max_dist_pct == 1.5
    assert rcfg.size_cap_min_ask == 0.88
    # Fee model defaults to flat / 0 — HL slot behaviour is unchanged.
    assert rcfg.fee_model == "flat"
    assert rcfg.fee_rate == 0.0


_YAML_PM_FEES = """
strategy:
  name: late_resolution
  paper_mode: true
  allowlist:
    - match: {class: priceBinary, underlying: BTC}
      max_position_usd: 50
      stop_loss_pct: null
      tte_min_seconds: 0
      tte_max_seconds: 86400
      price_extreme_threshold: 0.85
      distance_from_strike_usd_min: 0
      vol_max: 100
      fee_model: pm_binary
      fee_rate: 0.07
  blocklist_question_idxs: []
  defaults:
    match: {}
    max_position_usd: 50
    stop_loss_pct: null
    tte_min_seconds: 0
    tte_max_seconds: 86400
    price_extreme_threshold: 0.85
    distance_from_strike_usd_min: 0
    vol_max: 100
    fee_model: pm_binary
    fee_rate: 0.07
  global:
    max_total_inventory_usd: 500
    max_concurrent_positions: 5
    daily_loss_cap_usd: 200
    max_strike_distance_pct: 50
    min_recent_volume_usd: 100
    stale_data_halt_seconds: 30
    reconcile_interval_seconds: 60
"""


def test_runtime_threads_pm_binary_fee_model_into_strategy_config(tmp_path: Path):
    """v1 on Polymarket: pm_binary fee declaration must round-trip from YAML
    through AllowlistEntry into LateResolutionConfig so the live slot's fee
    curve is explicit and matches what the backtest used."""
    p = tmp_path / "strategy.yaml"
    p.write_text(_YAML_PM_FEES)
    cfg = load_strategy_config(p)
    rcfg = build_late_resolution_config(cfg)
    assert rcfg.fee_model == "pm_binary"
    assert rcfg.fee_rate == 0.07
