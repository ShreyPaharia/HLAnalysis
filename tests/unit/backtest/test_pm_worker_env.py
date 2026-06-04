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


def test_hl_worker_factory_honors_resample_env(tmp_path, monkeypatch):
    """make_hl_hip4_source must rebuild with the env-propagated resample period
    (coupled to vol_sampling_dt_seconds) — not the hardcoded 60s default."""
    from hlanalysis.backtest.cli import make_hl_hip4_source, _set_hl_worker_env

    monkeypatch.setenv("HLBT_HL_DATA_ROOT", str(tmp_path))
    for k in ("HLBT_HL_RESAMPLE_SECONDS", "HLBT_HL_REF_SOURCE"):
        monkeypatch.delenv(k, raising=False)

    # Default when unset.
    assert make_hl_hip4_source().reference_resample_seconds == 60

    # Propagated from a dt=5 config.
    args = argparse.Namespace(data_source="hl_hip4", ref_source=None)
    _set_hl_worker_env(args, {"vol_sampling_dt_seconds": 5})
    assert make_hl_hip4_source().reference_resample_seconds == 5
