"""v3.5 gate mode: skip entry when momentum_mr regime is 'mr' and
score < -tau_gate (i.e. strong MR signal against the favorite)."""
from __future__ import annotations

from hlanalysis.strategy.theta_harvester import (
    ThetaHarvesterConfig, ThetaHarvesterStrategy,
)
from hlanalysis.strategy.types import (
    Action, BookState, QuestionView,
)


def _qv(strike: float = 100.0) -> QuestionView:
    return QuestionView(
        question_idx=0,
        yes_symbol="YES",
        no_symbol="NO",
        strike=strike,
        expiry_ns=10**18,  # far future
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
        edge_buffer=0.0,
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


def test_gate_mode_blocks_entry_when_mr_against_favorite() -> None:
    # Strong sustained DOWN move → z_ret against an UP favorite produces
    # regime="mr" and score=-1.0 (|z| >> 2.5, signed against favorite).
    # z_ret is used here because a uniform down drift saturates its MR
    # threshold reliably with ≥30 bars.
    rets = tuple([-0.002] * 90)  # strong down drift
    cfg = _cfg(
        momentum_mr_enabled=True,
        momentum_mr_indicator="z_ret",
        momentum_mr_lookback_min=60,
        momentum_mr_mode="gate",
        momentum_mr_tau_gate=0.3,
    )
    strat = ThetaHarvesterStrategy(cfg)
    # Construct a binary where YES is favorite at 0.95: bid 0.94, ask 0.96.
    books = {
        "YES": BookState(symbol="YES", bid_px=0.94, bid_sz=10.0, ask_px=0.96, ask_sz=10.0,
                         last_trade_ts_ns=0, last_l2_ts_ns=0),
        "NO":  BookState(symbol="NO",  bid_px=0.04, bid_sz=10.0, ask_px=0.06, ask_sz=10.0,
                         last_trade_ts_ns=0, last_l2_ts_ns=0),
    }
    qv = _qv(strike=100.0)
    # reference_price below strike + sustained down drift → MR against YES favorite
    dec = strat.evaluate(
        question=qv, books=books, reference_price=99.0,
        recent_returns=rets, recent_volume_usd=0.0, position=None,
        now_ns=10**17,
    )
    assert dec.action == Action.HOLD
    assert any(d.message == "momentum_mr_gate" for d in dec.diagnostics)


def test_gate_mode_allows_entry_when_momentum_aligned() -> None:
    rets = tuple([0.002] * 90)  # strong UP drift aligned to YES favorite
    cfg = _cfg(
        momentum_mr_enabled=True,
        momentum_mr_indicator="z_ret",
        momentum_mr_lookback_min=60,
        momentum_mr_mode="gate",
        momentum_mr_tau_gate=0.3,
        edge_buffer=-1.0,  # disable edge filter — we only test gate behavior
    )
    strat = ThetaHarvesterStrategy(cfg)
    books = {
        "YES": BookState(symbol="YES", bid_px=0.94, bid_sz=10.0, ask_px=0.96, ask_sz=10.0,
                         last_trade_ts_ns=0, last_l2_ts_ns=0),
        "NO":  BookState(symbol="NO",  bid_px=0.04, bid_sz=10.0, ask_px=0.06, ask_sz=10.0,
                         last_trade_ts_ns=0, last_l2_ts_ns=0),
    }
    qv = _qv(strike=100.0)
    dec = strat.evaluate(
        question=qv, books=books, reference_price=101.0,
        recent_returns=rets, recent_volume_usd=0.0, position=None,
        now_ns=10**17,
    )
    # When edge_buffer is loose and momentum aligns, we expect ENTER (or at
    # least no momentum_mr_gate diagnostic).
    assert not any(d.message == "momentum_mr_gate" for d in dec.diagnostics)


def test_disabled_gate_does_not_fire() -> None:
    rets = tuple([-0.002] * 90)
    cfg = _cfg(momentum_mr_enabled=False)  # the default
    strat = ThetaHarvesterStrategy(cfg)
    books = {
        "YES": BookState(symbol="YES", bid_px=0.94, bid_sz=10.0, ask_px=0.96, ask_sz=10.0,
                         last_trade_ts_ns=0, last_l2_ts_ns=0),
        "NO":  BookState(symbol="NO",  bid_px=0.04, bid_sz=10.0, ask_px=0.06, ask_sz=10.0,
                         last_trade_ts_ns=0, last_l2_ts_ns=0),
    }
    qv = _qv(strike=100.0)
    dec = strat.evaluate(
        question=qv, books=books, reference_price=99.0,
        recent_returns=rets, recent_volume_usd=0.0, position=None,
        now_ns=10**17,
    )
    assert not any(d.message == "momentum_mr_gate" for d in dec.diagnostics)
