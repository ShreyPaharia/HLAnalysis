"""v3.5 tilt mode: edge_buffer is scaled by (1 - alpha_tilt * score)."""
from __future__ import annotations

from hlanalysis.strategy.theta_harvester import (
    ThetaHarvesterConfig, ThetaHarvesterStrategy,
)
from hlanalysis.strategy.types import (
    Action, BookState, QuestionView,
)

# 4-hour expiry relative to now_ns=10**17. Keeps tau small so p_yes stays
# near 1.0 for reference_price=110 >> strike=100, giving edge_yes ≈ 0.06.
_NOW_NS = 10**17
_EXPIRY_NS = _NOW_NS + 14_400 * 10**9  # 4 hours


def _qv() -> QuestionView:
    return QuestionView(
        question_idx=0,
        yes_symbol="YES",
        no_symbol="NO",
        strike=100.0,
        expiry_ns=_EXPIRY_NS,
        underlying="BTC",
        klass="priceBinary",
        period="1d",
    )


def _cfg(**over) -> ThetaHarvesterConfig:
    base = dict(
        vol_lookback_seconds=3600,
        vol_sampling_dt_seconds=60,
        vol_clip_min=0.0,
        vol_clip_max=5.0,
        edge_buffer=0.05,            # high baseline bar
        fee_taker=0.0,
        half_spread_assumption=0.0,
        drift_lookback_seconds=0,
        drift_blend=0.0,
        max_position_usd=100.0,
        favorite_threshold=0.85,
        tte_min_seconds=0,
        tte_max_seconds=10**9,
        stop_loss_pct=None,
        exit_edge_threshold=0.0,
        take_profit_price=None,
        time_stop_seconds=0,
    )
    base.update(over)
    return ThetaHarvesterConfig(**base)


def _books_yes_fav() -> dict:
    """YES at ~0.95 mid (bid=ask=0.94). Edge vs ask ≈ p_yes - 0.94."""
    return {
        "YES": BookState(symbol="YES", bid_px=0.94, bid_sz=10.0, ask_px=0.94, ask_sz=10.0,
                         last_trade_ts_ns=0, last_l2_ts_ns=0),
        "NO":  BookState(symbol="NO",  bid_px=0.05, bid_sz=10.0, ask_px=0.06, ask_sz=10.0,
                         last_trade_ts_ns=0, last_l2_ts_ns=0),
    }


def test_tilt_loosens_buffer_when_momentum_aligned() -> None:
    # Aligned up-drift → z_ret score=+1.0, regime='momentum'.
    # effective_edge_buffer = edge_buffer * (1 - 1.0 * 1.0) = 0.0.
    # With ref=110 >> strike=100 and small sigma (tiny-variance returns),
    # p_yes ≈ 1.0 → edge_yes ≈ 0.06 > effective_buffer=0.0 → ENTER.
    # Without tilt edge_yes=0.06 already clears 0.05, but the tilt
    # diagnostic must appear to confirm the code path ran.
    rets = tuple([0.0006 if i % 2 == 0 else 0.0007 for i in range(90)])
    cfg = _cfg(
        momentum_mr_enabled=True,
        momentum_mr_indicator="z_ret",
        momentum_mr_lookback_min=60,
        momentum_mr_mode="tilt",
        momentum_mr_alpha_tilt=1.0,
    )
    strat = ThetaHarvesterStrategy(cfg)
    dec = strat.evaluate(
        question=_qv(), books=_books_yes_fav(), reference_price=110.0,
        recent_returns=rets, recent_volume_usd=0.0, position=None,
        now_ns=_NOW_NS,
    )
    assert dec.action == Action.ENTER
    assert any(d.message == "momentum_mr_tilt" for d in dec.diagnostics)


def test_tilt_tightens_buffer_when_mr_against_favorite() -> None:
    # Strong down-drift → z_ret score=-1.0, regime='mr'.
    # effective_edge_buffer = edge_buffer * (1 - 1.0 * (-1.0)) = edge_buffer * 2.0.
    # With edge_buffer=0.04: effective_buffer=0.08.
    # With ref=110 >> strike=100 and tiny sigma, p_yes ≈ 1.0 → edge_yes ≈ 0.06.
    # 0.06 > 0.04 (would enter raw) but 0.06 <= 0.08 (blocked by tilt).
    rets = tuple([-0.0007 if i % 2 == 0 else -0.0006 for i in range(90)])
    cfg = _cfg(
        edge_buffer=0.04,           # raw edge ~0.06 > 0.04, enters without tilt
        momentum_mr_enabled=True,
        momentum_mr_indicator="z_ret",
        momentum_mr_lookback_min=60,
        momentum_mr_mode="tilt",
        momentum_mr_alpha_tilt=1.0,
    )
    strat = ThetaHarvesterStrategy(cfg)
    dec = strat.evaluate(
        question=_qv(), books=_books_yes_fav(), reference_price=110.0,
        recent_returns=rets, recent_volume_usd=0.0, position=None,
        now_ns=_NOW_NS,
    )
    assert dec.action == Action.HOLD
