"""SHR-99: `hl-bt run --slot <alias>` sources params from the live config.

The run path must be able to reference a live strategy.yaml slot directly
instead of a hand-written params JSON, so a sim run is config-faithful by
construction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hlanalysis.backtest.cli import _load_run_params
from hlanalysis.backtest.slot_config import backtest_params_from_slot
from hlanalysis.engine.config import load_strategies_config

LIVE = Path("config/strategy.yaml")


def _args(**kw):
    base = dict(slot=None, slot_config=str(LIVE), slot_class=None, config=None, strategy=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_slot_loads_live_params_and_sets_strategy_id():
    args = _args(slot="v1")
    params = _load_run_params(args)
    cfg = next(c for c in load_strategies_config(LIVE).strategies if c.account_alias == "v1")
    strategy_id, expected = backtest_params_from_slot(cfg)
    assert params == expected
    assert args.strategy == strategy_id


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


def test_config_path_still_works(tmp_path):
    cfg_path = tmp_path / "p.json"
    cfg_path.write_text('{"tte_min_seconds": 1}')
    args = _args(config=str(cfg_path), strategy="v1_late_resolution")
    assert _load_run_params(args) == {"tte_min_seconds": 1}
