"""TDD tests for SHR-80: derive reference σ-resample cadence from strategy config.

The backtest source for hl_hip4 MUST evaluate σ at the same cadence as the live
engine. ``_source_config_from_args`` must derive ``reference_resample_seconds``
from ``vol_sampling_dt_seconds`` in the loaded config dict, and a mismatch must
raise loudly rather than silently running at 60 s.

Per-class dt note: a single ``hl-bt run`` uses ONE cadence (bucket dt=2 vs
binary dt=5 are separate invocations). The assertion enforces this; mixing two
cadences in one run is unsupported.
"""
from __future__ import annotations

import argparse

import pytest

from hlanalysis.backtest.cli import _source_config_from_args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(data_source: str = "hl_hip4") -> argparse.Namespace:
    """Minimal args namespace that ``_source_config_from_args`` needs."""
    ns = argparse.Namespace()
    ns.data_source = data_source
    ns.cache_root = None
    # hl_hip4-specific args
    ns.ref_source = "hl_perp"
    # polymarket-specific (not exercised here, but keeps the function happy)
    ns.pm_flavor = "btc_updown"
    ns.pm_reference_source = "klines"
    ns.pm_book_source = "synthetic"
    ns.pm_binance_bbo_product_type = "perp"
    ns.pm_liquidity_profile = None
    return ns


# ---------------------------------------------------------------------------
# Test 1: cadence is derived from vol_sampling_dt_seconds for hl_hip4
# ---------------------------------------------------------------------------

def test_hl_hip4_reference_resample_derived_from_config_dt5():
    """_source_config_from_args returns reference_resample_seconds == 5 when
    vol_sampling_dt_seconds=5 is threaded in (not the hardcoded 60)."""
    args = _make_args("hl_hip4")
    cfg = _source_config_from_args(
        args,
        reference_resample_seconds=5,  # threaded from params["vol_sampling_dt_seconds"]
    )
    assert cfg.reference_resample_seconds == 5, (
        f"Expected 5, got {cfg.reference_resample_seconds} — "
        "cadence must be derived from vol_sampling_dt_seconds, not hardcoded 60"
    )


def test_hl_hip4_reference_resample_derived_from_config_dt2():
    """Same check for bucket cadence dt=2 (separate run from binary dt=5)."""
    args = _make_args("hl_hip4")
    cfg = _source_config_from_args(
        args,
        reference_resample_seconds=2,
    )
    assert cfg.reference_resample_seconds == 2


# ---------------------------------------------------------------------------
# Test 2: assert_hl_cadence_match helper
# ---------------------------------------------------------------------------

def test_assert_hl_cadence_match_raises_on_mismatch():
    """assert_hl_cadence_match must raise ValueError when the SourceConfig's
    reference_resample_seconds does not equal the strategy's vol_sampling_dt_seconds."""
    from hlanalysis.backtest.cli import assert_hl_cadence_match
    from hlanalysis.backtest.core.source_config import SourceConfig

    cfg = SourceConfig(kind="hl_hip4", reference_resample_seconds=60)
    with pytest.raises(ValueError, match="vol_sampling_dt_seconds"):
        assert_hl_cadence_match(cfg, {"vol_sampling_dt_seconds": 5})


def test_assert_hl_cadence_match_passes_when_equal():
    """No error when resample matches vol_sampling_dt_seconds."""
    from hlanalysis.backtest.cli import assert_hl_cadence_match
    from hlanalysis.backtest.core.source_config import SourceConfig

    cfg = SourceConfig(kind="hl_hip4", reference_resample_seconds=5)
    assert_hl_cadence_match(cfg, {"vol_sampling_dt_seconds": 5})  # must not raise


def test_assert_hl_cadence_match_skipped_for_polymarket():
    """Polymarket sources do not require vol_sampling_dt_seconds in the config;
    the check must be skipped (no error even without the key)."""
    from hlanalysis.backtest.cli import assert_hl_cadence_match
    from hlanalysis.backtest.core.source_config import SourceConfig

    cfg = SourceConfig(kind="polymarket", reference_resample_seconds=60)
    assert_hl_cadence_match(cfg, {})  # must not raise


def test_assert_hl_cadence_match_skipped_for_pm_nba():
    """pm_nba sources have no vol_sampling_dt_seconds; check must be skipped."""
    from hlanalysis.backtest.cli import assert_hl_cadence_match
    from hlanalysis.backtest.core.source_config import SourceConfig

    cfg = SourceConfig(kind="pm_nba")
    assert_hl_cadence_match(cfg, {})  # must not raise


# ---------------------------------------------------------------------------
# Test 3: end-to-end derivation from params dict (simulating cmd_run flow)
# ---------------------------------------------------------------------------

def test_end_to_end_derivation_from_params_dict():
    """Simulate what cmd_run does: load params, derive dt, build SourceConfig,
    assert match. Covers dt=5 (binary), dt=2 (bucket), and dt=60 (legacy)."""
    from hlanalysis.backtest.cli import assert_hl_cadence_match

    for dt in (5, 2, 60):
        params = {"vol_sampling_dt_seconds": dt, "strategy": "v31"}
        args = _make_args("hl_hip4")
        cfg = _source_config_from_args(
            args,
            reference_resample_seconds=int(params["vol_sampling_dt_seconds"]),
        )
        assert cfg.reference_resample_seconds == dt
        assert_hl_cadence_match(cfg, params)  # must not raise
