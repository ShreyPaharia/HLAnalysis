"""Tests for v5 delta-hedged strategy (Phase F)."""

from __future__ import annotations

import pytest

from hlanalysis.strategy.delta_hedged import binary_delta
from hlanalysis.strategy.types import Action, BookState, QuestionView


def test_binary_delta_matches_closed_form() -> None:
    """Δ = φ(d) / (S · σ · √τ) with d = (ln(S/K) + (μ − ½σ²)τ) / (σ √τ).
    Verify against a hand-computed value: S=100, K=100, σ=0.5, τ=0.25, μ=0.
    """
    delta = binary_delta(reference_price=100.0, strike=100.0, sigma=0.5, tau_yr=0.25, mu_eff=0.0)
    # d = (ln(1) + (0 - 0.125)*0.25) / (0.5 * 0.5) = -0.03125/0.25 = -0.125
    # φ(-0.125) ≈ 0.39574; Δ = 0.39574 / (100 * 0.5 * 0.5) = 0.0158
    assert delta == pytest.approx(0.01583, abs=1e-4)


def test_binary_delta_positive_above_strike() -> None:
    """Δ > 0 always for YES (binary cdf is increasing in S)."""
    d = binary_delta(reference_price=110.0, strike=100.0, sigma=0.5, tau_yr=0.1, mu_eff=0.0)
    assert d > 0


def test_emits_hedge_intent_on_binary_entry() -> None:
    """When v3 entry fires, v5 must also emit a hedge intent on the hedge symbol."""
    from hlanalysis.backtest.core.registry import build as build_strategy

    params = dict(
        vol_lookback_seconds=3600,
        vol_sampling_dt_seconds=60,
        edge_buffer=0.02,
        stop_loss_pct=None,
        drift_lookback_seconds=0,
        favorite_threshold=0.0,
        tte_min_seconds=0,
        tte_max_seconds=10**9,
        max_position_usd=100.0,
        exit_edge_threshold=-0.01,
        take_profit_price=None,
        time_stop_seconds=0,
        hedge_symbol="BTC-PERP",
        rebalance_interval_s=0,
        rebalance_threshold=0.0,
    )
    strat = build_strategy("v5_delta_hedged", params)

    # ATM setup: reference_price = strike = 100_000 → p_model ≈ 0.5.
    # YES ask = 0.25 → edge_yes = 0.5 - 0.25 - 0.02 = 0.23 >> edge_buffer → ENTER.
    # ATM also ensures binary_delta is near its maximum (phi(0)/denom), giving a
    # meaningful non-zero hedge size.
    qv = QuestionView(
        question_idx=0,
        yes_symbol="YES",
        no_symbol="NO",
        strike=100_000.0,
        expiry_ns=3600 * 10**9,
        settled=False,
        underlying="BTC",
        klass="priceBinary",
        period="1d",
        kv=(),
    )
    books = {
        "YES": BookState(
            "YES", bid_px=0.24, bid_sz=100.0, ask_px=0.25, ask_sz=100.0, last_trade_ts_ns=0, last_l2_ts_ns=0
        ),
        "NO": BookState(
            "NO", bid_px=0.74, bid_sz=100.0, ask_px=0.75, ask_sz=100.0, last_trade_ts_ns=0, last_l2_ts_ns=0
        ),
        "BTC-PERP": BookState(
            "BTC-PERP", bid_px=99_988.0, bid_sz=1e6, ask_px=100_012.0, ask_sz=1e6, last_trade_ts_ns=0, last_l2_ts_ns=0
        ),
    }
    rets = tuple([0.0001] * 120)

    d = strat.evaluate(
        question=qv,
        books=books,
        reference_price=100_000.0,
        recent_returns=rets,
        recent_volume_usd=1000.0,
        position=None,
        now_ns=0,
    )

    assert d.action == Action.ENTER
    syms = [i.symbol for i in d.intents]
    assert "YES" in syms or "NO" in syms
    assert "BTC-PERP" in syms
    hedge = [i for i in d.intents if i.symbol == "BTC-PERP"][0]
    assert hedge.size > 0
    # YES bought → expect SHORT BTC (sell)
    assert hedge.side == "sell"
