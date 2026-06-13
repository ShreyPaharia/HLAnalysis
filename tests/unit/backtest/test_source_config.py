"""SourceConfig replaces the HLBT_* env side-channel.

A spawn worker must rebuild an IDENTICAL data source from the picklable config
carried in the work tuple — NOT reconstruct it from ~20 HLBT_* env vars that
silently revert to defaults. This closes the documented config-drop bug class
(PM ``book_source`` and HL ``reference_resample_seconds`` reverting to
``synthetic`` / ``60`` in subprocess workers).

The pickle round-trip in each test simulates the spawn process boundary: the
config is serialised in the parent and rebuilt in the worker.
"""

from __future__ import annotations

import pickle

from hlanalysis.backtest.core.source_config import SourceConfig


def test_source_config_is_picklable_and_equal():
    cfg = SourceConfig(
        kind="polymarket",
        cache_root="data/sim",
        pm_book_source="recorded",
        pm_reference_source="binance_bbo",
        reference_resample_seconds=5,
        pm_binance_bbo_product_type="spot",
    )
    assert pickle.loads(pickle.dumps(cfg)) == cfg


def test_pm_source_rebuilt_across_process_boundary_keeps_nondefault_knobs(tmp_path):
    cfg = SourceConfig(
        kind="polymarket",
        cache_root=str(tmp_path),
        pm_flavor="btc_updown",
        pm_book_source="recorded",
        pm_reference_source="binance_bbo",
        reference_resample_seconds=5,
        pm_binance_bbo_product_type="spot",
    )
    src = pickle.loads(pickle.dumps(cfg)).build()
    assert src._book_source == "recorded"
    assert src._reference_source == "binance_bbo"
    assert src._reference_resample_seconds == 5
    assert src._binance_bbo_product_type == "spot"


def test_pm_defaults_match_legacy(tmp_path):
    src = SourceConfig(kind="polymarket", cache_root=str(tmp_path)).build()
    assert src._book_source == "synthetic"
    assert src._reference_source == "klines"
    assert src._reference_resample_seconds == 60
    assert src._binance_bbo_product_type == "perp"


def test_hl_source_rebuilt_honors_resample_and_ref_source(tmp_path):
    cfg = SourceConfig(
        kind="hl_hip4",
        cache_root=str(tmp_path),
        hl_ref_source="binance_perp",
        reference_resample_seconds=5,
    )
    src = pickle.loads(pickle.dumps(cfg)).build()
    assert src.reference_resample_seconds == 5
    assert src.ref_source == "binance_perp"


def test_hl_default_resample(tmp_path):
    src = SourceConfig(kind="hl_hip4", cache_root=str(tmp_path)).build()
    assert src.reference_resample_seconds == 60
    assert src.ref_source == "hl_perp"


def test_with_reference_resample_returns_replaced_copy():
    base = SourceConfig(kind="hl_hip4", reference_resample_seconds=60)
    out = base.with_reference_resample(5)
    assert out.reference_resample_seconds == 5
    assert base.reference_resample_seconds == 60  # original unchanged
    assert out.kind == "hl_hip4"


def test_unknown_pm_flavor_raises(tmp_path):
    import pytest

    cfg = SourceConfig(kind="polymarket", cache_root=str(tmp_path), pm_flavor="nope")
    with pytest.raises(SystemExit):
        cfg.build()


def test_tune_cell_overrides_reference_resample_from_cell_params(monkeypatch):
    """A tune worker must build its source with THIS cell's
    vol_sampling_dt_seconds — the dt=5 regression the env side-channel kept
    causing. ``_run_one_cell`` overrides the shipped SourceConfig's cadence per
    cell via ``with_reference_resample``; without it the source reverts to 60s
    while the strategy annualises at dt=5 and every tick gets gated.
    """
    from dataclasses import asdict

    from hlanalysis.backtest import tuning as T
    from hlanalysis.backtest.runner.hftbt_runner import RunConfig

    captured: dict = {}

    class _FakeSource:
        def discover(self, *, start, end):
            return []

    def fake_build(self):
        captured["reference_resample_seconds"] = self.reference_resample_seconds
        return _FakeSource()

    monkeypatch.setattr(SourceConfig, "build", fake_build)
    monkeypatch.setattr(T, "build_strategy", lambda sid, params: object())

    base_cfg = SourceConfig(kind="hl_hip4", reference_resample_seconds=60)
    work = (
        "v1_late_resolution",
        {"vol_sampling_dt_seconds": 5},
        ("q_train",),
        (("q_test", 0.0),),
        asdict(RunConfig()),
        base_cfg,
        None,
        1.0,
    )
    row = T._run_one_cell(work)
    assert captured["reference_resample_seconds"] == 5  # not the base 60
    assert row["n_test"] == 1
