"""SHR-99: `hl-bt run --slot <alias>` sources params from the live config.

The run path must be able to reference a live strategy.yaml slot directly
instead of a hand-written params JSON, so a sim run is config-faithful by
construction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hlanalysis.backtest.cli import _load_run_params, _run_config_from_args
from hlanalysis.backtest.slot_config import backtest_params_from_slot
from hlanalysis.engine.config import load_strategies_config

LIVE = Path("config/strategy.yaml")


def _args(**kw):
    base = dict(
        slot=None,
        slot_config=str(LIVE),
        slot_class=None,
        config=None,
        strategy=None,
        scan_mode=None,
        scan_min_interval_seconds=0.2,
        scan_max_interval_seconds=2.0,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_slot_loads_live_params_and_sets_strategy_id():
    args = _args(slot="v1")
    params = _load_run_params(args)
    cfg = next(c for c in load_strategies_config(LIVE).strategies if c.account_alias == "v1")
    strategy_id, expected = backtest_params_from_slot(cfg)
    assert params == expected
    assert args.strategy == strategy_id


def test_slot_defaults_to_live_event_cadence():
    """A slot run must mirror the live engine's scan cadence (event-driven,
    scan_min/max from the live GlobalRiskConfig) so intraday exits are evaluated
    at the SAME granularity as live — the legacy 60s fixed scan misses
    sub-minute exit_safety_d triggers and badly misvalidates the slot."""
    args = _args(slot="v31")
    _load_run_params(args)
    cfg = next(c for c in load_strategies_config(LIVE).strategies if c.account_alias == "v31")
    assert args.scan_mode == "event"
    assert args.scan_min_interval_seconds == cfg.global_.scan_min_interval_seconds
    assert args.scan_max_interval_seconds == cfg.global_.scan_max_interval_seconds


def test_explicit_scan_mode_overrides_slot_cadence_default():
    args = _args(slot="v31", scan_mode="fixed")
    _load_run_params(args)
    assert args.scan_mode == "fixed"


def test_non_slot_run_keeps_legacy_fixed_cadence():
    """Config-path (non-slot) runs must stay on the legacy fixed cadence."""
    args = _args(config="/tmp/x.json", strategy="v3_theta_harvester")
    Path("/tmp/x.json").write_text("{}")
    _load_run_params(args)
    assert args.scan_mode != "event"


def test_slot_class_selects_per_class_config():
    args = _args(slot="v1", slot_class="priceBucket")
    params = _load_run_params(args)
    cfg = next(c for c in load_strategies_config(LIVE).strategies if c.account_alias == "v1")
    _, expected = backtest_params_from_slot(cfg, klass="priceBucket")
    assert params == expected


def test_slot_and_config_are_mutually_exclusive():
    args = _args(slot="v1", config="/tmp/params.json")
    with pytest.raises(SystemExit):
        _load_run_params(args)


def test_unknown_slot_errors_with_available_aliases():
    args = _args(slot="does_not_exist")
    with pytest.raises(SystemExit):
        _load_run_params(args)


def _runcfg_args(**kw):
    """A complete Namespace for _run_config_from_args."""
    base = dict(
        scanner_interval_seconds=1.0,
        tick_size=0.001,
        lot_size=1.0,
        slippage_bps=0.0,
        fee_taker=0.0,
        fee_model="flat",
        fee_rate=0.0,
        depth=None,
        order_latency_ms=50.0,
        scan_mode=None,
        scan_min_interval_seconds=0.2,
        scan_max_interval_seconds=2.0,
        min_inter_order_seconds=0.0,
        ioc_fleeting_persistence_seconds=0.0,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_slot_run_stashes_live_inventory_caps_on_args():
    """A slot run must lift the slot's live inventory/daily-loss caps onto args
    so the sim applies the SAME risk envelope the live engine enforces — without
    it, backtests over-enter notional live's RiskGate would have blocked, making
    sim PnL over-optimistic (and untrustworthy for sizing decisions)."""
    args = _args(slot="v1")
    _load_run_params(args)
    cfg = next(c for c in load_strategies_config(LIVE).strategies if c.account_alias == "v1")
    assert args.sim_max_inventory_usd == cfg.global_.max_total_inventory_usd
    assert args.sim_max_concurrent_positions == cfg.global_.max_concurrent_positions
    assert args.sim_daily_loss_cap_usd == cfg.global_.daily_loss_cap_usd
    assert args.sim_daily_window_start_hour_utc == cfg.global_.daily_window_start_hour_utc


def test_run_config_builds_sim_risk_caps_from_args():
    """_run_config_from_args must translate the stashed caps into RunConfig."""
    args = _runcfg_args(
        sim_max_inventory_usd=1000.0,
        sim_max_concurrent_positions=5,
        sim_daily_loss_cap_usd=100.0,
        sim_daily_window_start_hour_utc=6,
    )
    rc = _run_config_from_args(args, None)
    assert rc.sim_risk_caps is not None
    assert rc.sim_risk_caps.max_total_inventory_usd == 1000.0
    assert rc.sim_risk_caps.max_concurrent_positions == 5
    assert rc.sim_risk_caps.daily_loss_cap_usd == 100.0
    assert rc.sim_risk_caps.daily_window_start_hour_utc == 6


def test_run_config_without_caps_leaves_sim_risk_caps_none():
    """Back-compat: a non-slot run with no cap args must keep sim_risk_caps=None
    so existing ad-hoc runs stay bit-identical."""
    rc = _run_config_from_args(_runcfg_args(), None)
    assert rc.sim_risk_caps is None


def test_config_path_still_works(tmp_path):
    cfg_path = tmp_path / "p.json"
    cfg_path.write_text('{"tte_min_seconds": 1}')
    args = _args(config=str(cfg_path), strategy="v1_late_resolution")
    assert _load_run_params(args) == {"tte_min_seconds": 1}
