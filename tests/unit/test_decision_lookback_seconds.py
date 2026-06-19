# tests/unit/test_decision_lookback_seconds.py
"""Strategy.decision_lookback_seconds — the σ/drift returns-window the runner
should request per scan tick.

The backtest runner used to pull recent_returns over the RunConfig default
(86_400s ≈ a full day → a 17 280-element array at dt=5, re-tupled EVERY tick).
``decision_lookback_seconds`` lets a strategy report the window it actually
consumes so the runner can bound that array; the runner provisions 2× the
reported value as a verified-safe margin. The hook must:

  * default to None on the base class (legacy: runner uses the full window),
  * for theta, cover the MAX returns consumption across the default cfg AND
    every per-class override (so no class is under-provisioned),
  * fold in the momentum/MR lookback (in samples) when that gate is enabled.

Under-reporting would silently truncate the σ window and change decisions, so
these tests pin the exact derivation.
"""

from __future__ import annotations

from hlanalysis.strategy.base import Strategy
from hlanalysis.strategy.theta_harvester import ThetaHarvesterConfig, ThetaHarvesterStrategy


def _base_cfg(**overrides) -> ThetaHarvesterConfig:
    kwargs: dict = dict(
        vol_lookback_seconds=3600,
        vol_sampling_dt_seconds=5,
        vol_clip_min=0.05,
        vol_clip_max=5.0,
        edge_buffer=0.02,
        fee_taker=0.0,
        half_spread_assumption=0.0,
        drift_lookback_seconds=0,
        drift_blend=0.0,
        max_position_usd=100.0,
        favorite_threshold=0.85,
        tte_min_seconds=0,
        tte_max_seconds=10**9,
        stop_loss_pct=None,
        exit_edge_threshold=-0.01,
        take_profit_price=None,
        time_stop_seconds=0,
        exit_safety_d=0.0,
    )
    kwargs.update(overrides)
    return ThetaHarvesterConfig(**kwargs)


def test_base_strategy_returns_none() -> None:
    # The base contract returns None → runner keeps the legacy full window.
    assert Strategy.decision_lookback_seconds.__doc__ is not None

    class _Dummy(Strategy):
        def evaluate(self, **_kw):  # type: ignore[override]
            raise NotImplementedError

    assert _Dummy().decision_lookback_seconds() is None


def test_theta_default_is_max_of_vol_and_drift() -> None:
    s = ThetaHarvesterStrategy(_base_cfg(vol_lookback_seconds=3600, drift_lookback_seconds=900))
    assert s.decision_lookback_seconds() == 3600

    s2 = ThetaHarvesterStrategy(_base_cfg(vol_lookback_seconds=900, drift_lookback_seconds=2700))
    assert s2.decision_lookback_seconds() == 2700


def test_theta_covers_per_class_overrides() -> None:
    default = _base_cfg(vol_lookback_seconds=900)
    bucket = _base_cfg(vol_lookback_seconds=2700, vol_sampling_dt_seconds=2)
    s = ThetaHarvesterStrategy(default, cfg_by_class={"priceBucket": bucket})
    # Must be the MAX across default + every per-class cfg (else a class whose
    # window is wider than the default would be silently truncated).
    assert s.decision_lookback_seconds() == 2700


def test_theta_folds_in_momentum_mr_when_enabled() -> None:
    # momentum_mr slices recent_returns[-lookback_min:] (count of samples), so it
    # needs lookback_min * vol_sampling_dt_seconds seconds of history.
    cfg = _base_cfg(
        vol_lookback_seconds=600,
        drift_lookback_seconds=0,
        vol_sampling_dt_seconds=5,
        momentum_mr_enabled=True,
        momentum_mr_lookback_min=200,  # 200 samples * 5s = 1000s > 600s
    )
    s = ThetaHarvesterStrategy(cfg)
    assert s.decision_lookback_seconds() == 1000


def test_theta_momentum_ignored_when_disabled() -> None:
    cfg = _base_cfg(
        vol_lookback_seconds=600,
        drift_lookback_seconds=0,
        vol_sampling_dt_seconds=5,
        momentum_mr_enabled=False,
        momentum_mr_lookback_min=200,
    )
    s = ThetaHarvesterStrategy(cfg)
    assert s.decision_lookback_seconds() == 600
