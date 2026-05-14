# tests/unit/test_strategy_model_edge.py
from __future__ import annotations

from hlanalysis.strategy.types import (
    Action, BookState, Position, QuestionView,
)
from hlanalysis.strategy.model_edge import ModelEdgeConfig, ModelEdgeStrategy, _ANNUAL_SECONDS


def _cfg(**overrides) -> ModelEdgeConfig:
    base = dict(
        vol_lookback_seconds=3600,
        vol_sampling_dt_seconds=60,
        vol_clip_min=0.05,
        vol_clip_max=3.0,
        edge_buffer=0.02,
        fee_taker=0.02,
        half_spread_assumption=0.005,
        stop_loss_pct=10.0,
    )
    base.update(overrides)
    return ModelEdgeConfig(**base)


def _q(**overrides) -> QuestionView:
    base = dict(
        question_idx=1,
        yes_symbol="YES",
        no_symbol="NO",
        strike=100_000.0,
        expiry_ns=86_400_000_000_000,
        underlying="BTC",
        klass="priceBinary",
        period="24h",
    )
    base.update(overrides)
    return QuestionView(**base)


def test_holds_when_settled_no_position():
    s = ModelEdgeStrategy(_cfg())
    out = s.evaluate(
        question=_q(settled=True),
        books={},
        reference_price=100_000.0,
        recent_returns=(),
        recent_volume_usd=0.0,
        position=None,
        now_ns=0,
    )
    assert out.action == Action.HOLD


def test_exits_when_settled_with_position():
    s = ModelEdgeStrategy(_cfg())
    pos = Position(question_idx=1, symbol="YES", qty=1.0, avg_entry=0.5,
                   stop_loss_price=0.0, last_update_ts_ns=0)
    out = s.evaluate(
        question=_q(settled=True),
        books={},
        reference_price=100_000.0,
        recent_returns=(),
        recent_volume_usd=0,
        position=pos,
        now_ns=0,
    )
    assert out.action == Action.EXIT


def test_holds_when_recent_returns_insufficient():
    s = ModelEdgeStrategy(_cfg())
    out = s.evaluate(
        question=_q(),
        books={"YES": BookState("YES", 0.4, 100, 0.5, 100, 0, 0),
               "NO":  BookState("NO",  0.4, 100, 0.5, 100, 0, 0)},
        reference_price=100_500.0,
        recent_returns=(0.001,),  # only 1, sample stdev needs ≥ 2
        recent_volume_usd=0,
        position=None,
        now_ns=0,
    )
    assert out.action == Action.HOLD
    diag_msgs = [d.message for d in out.diagnostics]
    assert "vol_insufficient_data" in diag_msgs


def test_enters_yes_when_p_model_far_above_market():
    # BTC well above strike, low vol, market still pricing 0.6 → big YES edge
    s = ModelEdgeStrategy(_cfg(edge_buffer=0.02, vol_clip_min=0.1))
    out = s.evaluate(
        question=_q(strike=100_000.0, expiry_ns=int(0.5 * 86_400 * 1e9)),
        books={"YES": BookState("YES", 0.59, 100, 0.60, 100, 0, 0),
               "NO":  BookState("NO",  0.39, 100, 0.40, 100, 0, 0)},
        reference_price=110_000.0,             # +10% above strike
        recent_returns=tuple([0.0001] * 60),    # very low vol
        recent_volume_usd=10000.0,
        position=None,
        now_ns=0,
    )
    assert out.action == Action.ENTER
    assert out.intents and out.intents[0].symbol == "YES"
    assert out.intents[0].side == "buy"


def test_holds_when_edge_below_buffer():
    s = ModelEdgeStrategy(_cfg(edge_buffer=0.50))   # absurdly high buffer
    out = s.evaluate(
        question=_q(),
        books={"YES": BookState("YES", 0.49, 100, 0.50, 100, 0, 0),
               "NO":  BookState("NO",  0.49, 100, 0.50, 100, 0, 0)},
        reference_price=100_500.0,
        recent_returns=tuple([0.001] * 60),
        recent_volume_usd=10000.0,
        position=None,
        now_ns=0,
    )
    assert out.action == Action.HOLD


def test_enters_no_when_p_model_far_below_market():
    s = ModelEdgeStrategy(_cfg(edge_buffer=0.02, vol_clip_min=0.1))
    out = s.evaluate(
        question=_q(strike=100_000.0, expiry_ns=int(0.5 * 86_400 * 1e9)),
        books={"YES": BookState("YES", 0.59, 100, 0.60, 100, 0, 0),
               "NO":  BookState("NO",  0.39, 100, 0.40, 100, 0, 0)},
        reference_price=90_000.0,              # -10% below strike → NO wins
        recent_returns=tuple([0.0001] * 60),
        recent_volume_usd=10000.0,
        position=None,
        now_ns=0,
    )
    assert out.action == Action.ENTER
    assert out.intents[0].symbol == "NO"


def test_tau_yr_precision_roundtrip_short_tte():
    """tau_yr must survive a parquet string round-trip with relative error < 1e-6
    at short TTE (5 minutes), where :.6f would truncate ~5% of the value.
    ModelEdgeStrategy uses :.12f for tau_yr to achieve sub-ppm precision.
    """
    import math
    from hlanalysis.backtest.runner.result import _parse_edge_fields
    from hlanalysis.strategy.types import BookState, Decision, Diagnostic

    tte_s = 5 * 60  # 300 seconds
    tau_yr_true = tte_s / _ANNUAL_SECONDS  # ≈ 9.506e-6

    # Build a synthetic Decision whose "edge" diagnostic carries tau_yr
    # formatted the way ModelEdgeStrategy now formats it (:.12f).
    diag = Diagnostic("info", "edge", (
        ("p_model", f"{0.55:.4f}"),
        ("edge_yes", f"{0.03:.4f}"),
        ("edge_no", f"{-0.1:.4f}"),
        ("sigma", f"{0.8:.4f}"),
        ("tau_yr", f"{tau_yr_true:.12f}"),
        ("ln_sk", f"{0.001:.4f}"),
    ))

    class _FakeDecision:
        diagnostics = (diag,)

    parsed = _parse_edge_fields(_FakeDecision())
    tau_yr_parsed = parsed["tau_yr"]
    assert tau_yr_parsed is not None
    rel_err = abs(tau_yr_parsed - tau_yr_true) / tau_yr_true
    assert rel_err < 1e-6, (
        f"tau_yr round-trip relative error {rel_err:.2e} >= 1e-6 "
        f"(true={tau_yr_true:.10f}, parsed={tau_yr_parsed:.10f})"
    )


def test_p_model_matches_empirical_on_synthetic_gbm():
    """Generate N GBM paths with known σ and check v2's p_model is close to empirical YES rate."""
    import math
    import numpy as np
    rng = np.random.default_rng(42)
    n_paths = 20_000
    sigma_true = 0.6
    tau_yr = 0.5
    s0 = 100_000.0
    strike = 101_000.0
    # Sample terminal prices under GBM
    z = rng.standard_normal(n_paths)
    s_t = s0 * np.exp(-0.5 * sigma_true ** 2 * tau_yr + sigma_true * np.sqrt(tau_yr) * z)
    empirical_p = float((s_t > strike).mean())

    # Build recent_returns at 60s sampling such that σ_raw × annualization ≈ σ_true.
    sample_dt_s = 60
    n_samples = 240
    sigma_raw = sigma_true / math.sqrt(_ANNUAL_SECONDS / sample_dt_s)
    returns = tuple(rng.normal(loc=-0.5 * sigma_raw**2, scale=sigma_raw, size=n_samples).tolist())

    s = ModelEdgeStrategy(_cfg(
        vol_lookback_seconds=n_samples * sample_dt_s,
        vol_sampling_dt_seconds=sample_dt_s,
        vol_clip_min=0.0,
        vol_clip_max=10.0,
    ))
    out = s.evaluate(
        question=_q(strike=strike, expiry_ns=int(tau_yr * _ANNUAL_SECONDS * 1e9)),
        books={"YES": BookState("YES", 0.0, 1, 1.0, 1, 0, 0),
               "NO":  BookState("NO",  0.0, 1, 1.0, 1, 0, 0)},
        reference_price=s0,
        recent_returns=returns,
        recent_volume_usd=10000.0,
        position=None,
        now_ns=0,
    )
    # Pull p_model out of diagnostics
    diag = next(d for d in out.diagnostics if d.message == "edge")
    fields = dict(diag.fields)
    p_model = float(fields["p_model"])
    assert abs(p_model - empirical_p) < 0.02, f"p_model={p_model} empirical={empirical_p}"
