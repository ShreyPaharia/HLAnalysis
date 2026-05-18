from __future__ import annotations

import math

from hlanalysis.backtest.core.registry import build as build_strategy
from hlanalysis.strategy.types import Action, BookState, Position, QuestionView


def _qv(*, expiry_ns: int = 10**18, strike: float = 100_000.0) -> QuestionView:
    return QuestionView(
        question_idx=0,
        yes_symbol="YES",
        no_symbol="NO",
        strike=strike,
        expiry_ns=expiry_ns,
        underlying="BTC",
        klass="priceBinary",
        period="1d",
        settled=False,
        kv=(),
    )


def _book(symbol: str, *, bid: float, ask: float) -> BookState:
    return BookState(
        symbol=symbol,
        bid_px=bid,
        bid_sz=100.0,
        ask_px=ask,
        ask_sz=100.0,
        last_trade_ts_ns=0,
        last_l2_ts_ns=0,
    )


def _params(**over) -> dict:
    base = dict(
        vol_lookback_seconds=3600,
        vol_sampling_dt_seconds=60,
        edge_buffer=0.02,
        fee_taker=0.0,
        half_spread_assumption=0.0,
        stop_loss_pct=None,
        max_position_usd=100.0,
        favorite_threshold=0.0,
        tte_min_seconds=0,
        tte_max_seconds=10**9,
        exit_edge_threshold=-0.01,
        take_profit_price=None,
        time_stop_seconds=0,
        drift_lookback_seconds=0,
        drift_blend=0.0,
    )
    base.update(over)
    return base


def test_entry_emits_buy_when_v2_edge_present() -> None:
    """v3 should accept v2-like entries unchanged."""
    strat = build_strategy("v3_theta_harvester", _params())
    # 1 hour to expiry, ref well above strike → p_model ~ 1; YES ask = 0.5 → huge edge
    qv = _qv(expiry_ns=3600 * 10**9)
    books = {"YES": _book("YES", bid=0.49, ask=0.50), "NO": _book("NO", bid=0.49, ask=0.50)}
    rets = tuple([0.0001] * 120)  # 2h of 60s returns, low vol

    decision = strat.evaluate(
        question=qv,
        books=books,
        reference_price=120_000.0,
        recent_returns=rets,
        recent_volume_usd=1000.0,
        position=None,
        now_ns=0,
    )

    assert decision.action == Action.ENTER
    assert len(decision.intents) == 1
    assert decision.intents[0].side == "buy"
    assert decision.intents[0].symbol == "YES"


def _pos(*, symbol: str, qty: float, entry_px: float, stop_pct: float | None = None) -> Position:
    sl_px = 0.0 if stop_pct is None else entry_px * (1.0 - stop_pct)
    return Position(
        question_idx=0, symbol=symbol, qty=qty,
        avg_entry=entry_px, stop_loss_price=sl_px,
        last_update_ts_ns=0,
    )


def test_edge_exit_fires_when_edge_collapses_below_threshold() -> None:
    """After BTC has moved against us so edge_held drops below exit_edge_threshold, EXIT."""
    strat = build_strategy("v3_theta_harvester", _params(exit_edge_threshold=-0.01))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    # Held YES @0.50; current YES ask = 0.45, ref price just barely above strike → p_model ~ 0.50 → edge_held = 0.50 - 0.45 = 0.05 (no exit)
    # Now drop ref price below strike: p_model collapses → edge_held goes negative → exit.
    books = {"YES": _book("YES", bid=0.45, ask=0.46), "NO": _book("NO", bid=0.54, ask=0.55)}
    pos = _pos(symbol="YES", qty=200.0, entry_px=0.50)
    rets = tuple([0.0001] * 120)

    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=99_000.0,   # below strike → p_model well below 0.5
        recent_returns=rets, recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )

    assert decision.action == Action.EXIT
    assert decision.intents[0].side == "sell"
    assert decision.intents[0].reduce_only is True


def test_holds_when_edge_held_still_positive() -> None:
    strat = build_strategy("v3_theta_harvester", _params(exit_edge_threshold=-0.01))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    books = {"YES": _book("YES", bid=0.55, ask=0.56), "NO": _book("NO", bid=0.44, ask=0.45)}
    pos = _pos(symbol="YES", qty=200.0, entry_px=0.50)
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=120_000.0,  # well above strike → p_model ~ 1 → edge_held very positive
        recent_returns=rets, recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    assert decision.action == Action.HOLD


def test_time_stop_fires_when_tau_below_threshold() -> None:
    strat = build_strategy("v3_theta_harvester", _params(time_stop_seconds=600))
    qv = _qv(expiry_ns=500 * 10**9)   # 500s to expiry, < 600s threshold
    books = {"YES": _book("YES", bid=0.95, ask=0.96), "NO": _book("NO", bid=0.04, ask=0.05)}
    pos = _pos(symbol="YES", qty=200.0, entry_px=0.50)
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=120_000.0, recent_returns=rets, recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    assert decision.action == Action.EXIT
    diag_msgs = [d.message for d in decision.diagnostics]
    assert "exit_time_stop" in diag_msgs


def test_take_profit_fires_when_bid_above_threshold() -> None:
    strat = build_strategy("v3_theta_harvester", _params(take_profit_price=0.10))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    books = {"YES": _book("YES", bid=0.62, ask=0.63), "NO": _book("NO", bid=0.37, ask=0.38)}
    pos = _pos(symbol="YES", qty=200.0, entry_px=0.50)  # bid 0.62 >= 0.50 + 0.10
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=120_000.0, recent_returns=rets, recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    assert decision.action == Action.EXIT
    assert "exit_take_profit" in [d.message for d in decision.diagnostics]


def test_stop_loss_fires_when_bid_at_or_below_stop_px() -> None:
    strat = build_strategy("v3_theta_harvester", _params(stop_loss_pct=0.30))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    # entry 0.50, stop_loss_pct 0.30 → stop_px = 0.35. Current bid 0.34 ≤ 0.35.
    books = {"YES": _book("YES", bid=0.34, ask=0.35), "NO": _book("NO", bid=0.65, ask=0.66)}
    pos = _pos(symbol="YES", qty=200.0, entry_px=0.50, stop_pct=0.30)
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=100_000.0, recent_returns=rets, recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    assert decision.action == Action.EXIT
    assert "exit_stop_loss" in [d.message for d in decision.diagnostics]


def test_edge_max_filter_rejects_extreme_edge_entries() -> None:
    """When edge_max is set, entries claiming edge above the cap must HOLD,
    even when the basic edge_buffer would otherwise green-light the trade."""
    # 1h to expiry, ref massively above strike → p_model ~ 1
    # YES ask = 0.50 → edge ≈ 0.50 (huge). edge_max=0.20 must reject.
    strat = build_strategy("v3_theta_harvester", _params(edge_max=0.20))
    qv = _qv(expiry_ns=3600 * 10**9)
    books = {"YES": _book("YES", bid=0.49, ask=0.50), "NO": _book("NO", bid=0.49, ask=0.50)}
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=120_000.0, recent_returns=rets, recent_volume_usd=1000.0,
        position=None, now_ns=0,
    )
    assert decision.action == Action.HOLD
    assert "edge_too_extreme" in [d.message for d in decision.diagnostics]


def test_edge_max_filter_permits_moderate_edge_entries() -> None:
    """When the edge sits in the normal band, edge_max must not interfere."""
    # YES ask 0.85 with p_model ~ 1 → edge ~ 0.15. With edge_max=0.20 this passes.
    strat = build_strategy("v3_theta_harvester", _params(edge_max=0.20))
    qv = _qv(expiry_ns=3600 * 10**9)
    books = {"YES": _book("YES", bid=0.84, ask=0.85), "NO": _book("NO", bid=0.14, ask=0.15)}
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=120_000.0, recent_returns=rets, recent_volume_usd=1000.0,
        position=None, now_ns=0,
    )
    assert decision.action == Action.ENTER
    assert decision.intents[0].symbol == "YES"


def test_edge_max_none_disables_filter_preserving_v3_baseline() -> None:
    """Default behaviour: edge_max=None keeps v3 baseline behavior intact."""
    # Same scenario as test_entry_emits_buy_when_v2_edge_present — huge edge,
    # filter disabled → must still enter.
    strat = build_strategy("v3_theta_harvester", _params())  # edge_max omitted → None
    qv = _qv(expiry_ns=3600 * 10**9)
    books = {"YES": _book("YES", bid=0.49, ask=0.50), "NO": _book("NO", bid=0.49, ask=0.50)}
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=120_000.0, recent_returns=rets, recent_volume_usd=1000.0,
        position=None, now_ns=0,
    )
    assert decision.action == Action.ENTER
