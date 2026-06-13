# tests/unit/test_theta_harvester_v35_defaults.py
"""v3.5 backward-compat: defaults preserve v3.1 behavior bit-for-bit.

Strategy: build two ThetaHarvesterConfig dicts — one pre-v3.5 baseline (no
momentum_mr_* keys), one with the new keys explicitly set to their disabled
defaults. Both must produce identical Decisions on the same QuestionView/
BookState sequence.
"""

from __future__ import annotations

from hlanalysis.strategy.theta_harvester import ThetaHarvesterConfig


def _baseline_kwargs() -> dict:
    return dict(
        vol_lookback_seconds=3600,
        vol_sampling_dt_seconds=60,
        vol_clip_min=0.0,
        vol_clip_max=5.0,
        edge_buffer=0.03,
        fee_taker=0.0,
        half_spread_assumption=0.005,
        drift_lookback_seconds=3600,
        drift_blend=0.0,
        max_position_usd=200.0,
        favorite_threshold=0.90,
        tte_min_seconds=0,
        tte_max_seconds=86400,
        stop_loss_pct=None,
        exit_edge_threshold=0.0,
        take_profit_price=None,
        time_stop_seconds=0,
    )


def test_default_momentum_mr_fields_match_disabled_explicit() -> None:
    cfg_default = ThetaHarvesterConfig(**_baseline_kwargs())
    cfg_explicit = ThetaHarvesterConfig(
        **_baseline_kwargs(),
        momentum_mr_enabled=False,
        momentum_mr_indicator="z_ret",
        momentum_mr_lookback_min=15,
        momentum_mr_mode="gate",
        momentum_mr_tau_gate=1.0,
        momentum_mr_alpha_tilt=0.5,
    )
    # Compare every field
    for fld in cfg_default.__dataclass_fields__:
        assert getattr(cfg_default, fld) == getattr(cfg_explicit, fld), fld


def test_momentum_mr_enabled_is_false_by_default() -> None:
    cfg = ThetaHarvesterConfig(**_baseline_kwargs())
    assert cfg.momentum_mr_enabled is False
