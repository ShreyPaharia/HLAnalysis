"""The PM worker factory must rebuild a source matching the parent's PM knobs.

Regression for the spawn-worker divergence: `make_polymarket_source()` is the
zero-arg factory spawn workers call; if it ignores book_source / reference_source
then `--workers>1` silently reverts to synthetic fills and diverges from the
serial path.
"""
from __future__ import annotations

import argparse

import pytest

from hlanalysis.backtest.cli import make_polymarket_source, _set_pm_worker_env


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("HLBT_PM_BOOK_SOURCE", "HLBT_PM_REFERENCE_SOURCE",
              "HLBT_PM_RESAMPLE_SECONDS", "HLBT_PM_BBO_PRODUCT_TYPE",
              "HLBT_PM_FLAVOR", "HLBT_PM_CACHE_ROOT"):
        monkeypatch.delenv(k, raising=False)


def test_defaults_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("HLBT_PM_CACHE_ROOT", str(tmp_path))
    src = make_polymarket_source()
    assert src._book_source == "synthetic"
    assert src._reference_source == "klines"


def test_set_pm_worker_env_round_trips_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("HLBT_PM_CACHE_ROOT", str(tmp_path))
    args = argparse.Namespace(
        data_source="polymarket", pm_flavor="btc_updown",
        pm_book_source="recorded", pm_reference_source="binance_bbo",
        pm_reference_resample_seconds=5, pm_binance_bbo_product_type="spot",
    )
    _set_pm_worker_env(args)
    src = make_polymarket_source()
    assert src._book_source == "recorded"
    assert src._reference_source == "binance_bbo"
    assert src._reference_resample_seconds == 5
    assert src._binance_bbo_product_type == "spot"


def test_set_pm_worker_env_noop_for_non_pm():
    args = argparse.Namespace(data_source="hl_hip4")
    _set_pm_worker_env(args)  # must not raise / touch PM env
