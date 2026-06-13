# tests/unit/test_v35_disabled_bitidentical.py
"""v3.5 disabled = v3.1 bit-identical on a deterministic replay slice."""

from __future__ import annotations

from hlanalysis.backtest.core.registry import build as build_strategy
from hlanalysis.strategy.types import (
    Action,
    BookState,
    QuestionView,
)


def _ticks(n: int):
    """Synthesise n ticks of (qv, books, reference_price, recent_returns, now_ns)."""
    qv = QuestionView(
        question_idx=0,
        klass="priceBinary",
        kv=(),
        yes_symbol="YES",
        no_symbol="NO",
        leg_symbols=(),
        strike=100.0,
        expiry_ns=10**18,
        settled=False,
        underlying="BTC",
        period="1d",
    )
    rets = [0.0001 * ((i % 7) - 3) for i in range(n)]
    for i in range(n):
        books = {
            "YES": BookState(
                symbol="YES", bid_px=0.92, bid_sz=10.0, ask_px=0.94, ask_sz=10.0, last_trade_ts_ns=0, last_l2_ts_ns=0
            ),
            "NO": BookState(
                symbol="NO", bid_px=0.05, bid_sz=10.0, ask_px=0.07, ask_sz=10.0, last_trade_ts_ns=0, last_l2_ts_ns=0
            ),
        }
        yield qv, books, 101.0 + 0.01 * i, tuple(rets[: i + 1]), 10**17 + i * 10**9


def _baseline_params() -> dict:
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


def test_v35_disabled_equals_v31_baseline() -> None:
    v31 = build_strategy("v3_theta_harvester", _baseline_params())
    v35_params = {**_baseline_params(), "momentum_mr_enabled": False}
    v35 = build_strategy("v3_5_momentum_mr", v35_params)
    for qv, books, ref, rets, now_ns in _ticks(150):
        d_a = v31.evaluate(
            question=qv,
            books=books,
            reference_price=ref,
            recent_returns=rets,
            recent_volume_usd=0.0,
            position=None,
            now_ns=now_ns,
        )
        d_b = v35.evaluate(
            question=qv,
            books=books,
            reference_price=ref,
            recent_returns=rets,
            recent_volume_usd=0.0,
            position=None,
            now_ns=now_ns,
        )
        assert d_a.action == d_b.action
        # Diagnostics not required to be byte-identical (no momentum_mr_* diags
        # emitted when disabled), but the trade decision must match.
        assert tuple(i.symbol for i in d_a.intents) == tuple(i.symbol for i in d_b.intents)
        assert tuple(i.size for i in d_a.intents) == tuple(i.size for i in d_b.intents)
