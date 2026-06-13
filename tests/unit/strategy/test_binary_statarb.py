from __future__ import annotations

from hlanalysis.backtest.core.registry import build as build_strategy
from hlanalysis.strategy.types import Action, BookState, Position, QuestionView


def _qv(expiry_ns: int = 10**18) -> QuestionView:
    return QuestionView(
        question_idx=0,
        yes_symbol="YES",
        no_symbol="NO",
        strike=100_000.0,
        expiry_ns=expiry_ns,
        underlying="BTC",
        klass="priceBinary",
        period="1d",
        settled=False,
        kv=(),
    )


def _book(symbol: str, bid: float, ask: float) -> BookState:
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
        lookback_seconds=1800,
        ewma_lambda=0.95,
        z_entry=2.0,
        z_exit=0.5,
        mid_lo=0.2,
        mid_hi=0.8,
        max_position_usd=100.0,
        stop_loss_pct=None,
        time_stop_seconds=600,
        fee_taker=0.0,
        half_spread_assumption=0.005,
        sampling_dt_seconds=60,
    )
    base.update(over)
    return base


def test_holds_until_warmup_done() -> None:
    """During the warmup window (insufficient observations) v4 must HOLD."""
    strat = build_strategy("v4_binary_statarb", _params(lookback_seconds=600, sampling_dt_seconds=60))
    qv = _qv()
    books = {"YES": _book("YES", 0.49, 0.50), "NO": _book("NO", 0.49, 0.50)}
    # Feed only 2 ticks (need >= lookback/sampling_dt = 10 to engage)
    for ts in (0, 60_000_000_000):
        d = strat.evaluate(
            question=qv,
            books=books,
            reference_price=100_000.0,
            recent_returns=(),
            recent_volume_usd=100.0,
            position=None,
            now_ns=ts,
        )
        assert d.action == Action.HOLD


def test_entry_buys_yes_when_mid_crashes_below_trend() -> None:
    """After accumulating a stable mid ~0.50, a sharp drop to 0.40 yields z << -z_entry → BUY YES.

    Warmup note: 12 ticks at lambda=0.95, sampling_dt=60s with lookback=600s requires 10 samples.
    After 12 stable ticks, ewma_mean converges to ~0.50 and ewma_var is non-trivial.
    The 13th tick at 0.40 is well below, yielding a strongly negative z-score.
    """
    strat = build_strategy(
        "v4_binary_statarb",
        _params(
            lookback_seconds=600,
            sampling_dt_seconds=60,
            z_entry=2.0,
        ),
    )
    qv = _qv()
    # Feed 12 stable ticks at mid 0.50
    for i in range(12):
        ts = i * 60_000_000_000
        books = {"YES": _book("YES", 0.495, 0.505), "NO": _book("NO", 0.495, 0.505)}
        strat.evaluate(
            question=qv,
            books=books,
            reference_price=100_000.0,
            recent_returns=(),
            recent_volume_usd=100.0,
            position=None,
            now_ns=ts,
        )
    # 13th tick: mid drops to 0.40 → z very negative
    ts = 12 * 60_000_000_000
    books = {"YES": _book("YES", 0.395, 0.405), "NO": _book("NO", 0.595, 0.605)}
    d = strat.evaluate(
        question=qv,
        books=books,
        reference_price=100_000.0,
        recent_returns=(),
        recent_volume_usd=100.0,
        position=None,
        now_ns=ts,
    )
    assert d.action == Action.ENTER
    assert d.intents[0].symbol == "YES"


def test_exit_when_z_inside_z_exit_band() -> None:
    """After a crash entry, when mid reverts near mean, z falls inside z_exit band → EXIT.

    Uses avg_entry (not entry_price) and last_update_ts_ns=0 per actual Position schema.
    """
    strat = build_strategy(
        "v4_binary_statarb",
        _params(
            lookback_seconds=600,
            sampling_dt_seconds=60,
            z_entry=2.0,
            z_exit=0.5,
        ),
    )
    qv = _qv()
    # Warm up to a mean ~0.50
    for i in range(12):
        ts = i * 60_000_000_000
        books = {"YES": _book("YES", 0.495, 0.505), "NO": _book("NO", 0.495, 0.505)}
        strat.evaluate(
            question=qv,
            books=books,
            reference_price=100_000.0,
            recent_returns=(),
            recent_volume_usd=100.0,
            position=None,
            now_ns=ts,
        )
    # Now ask with a position and mid ~ mean → z ~ 0 → EXIT
    pos = Position(
        question_idx=0,
        symbol="YES",
        qty=200.0,
        avg_entry=0.40,
        stop_loss_price=0.0,
        last_update_ts_ns=0,
    )
    ts = 12 * 60_000_000_000
    books = {"YES": _book("YES", 0.495, 0.505), "NO": _book("NO", 0.495, 0.505)}
    d = strat.evaluate(
        question=qv,
        books=books,
        reference_price=100_000.0,
        recent_returns=(),
        recent_volume_usd=100.0,
        position=pos,
        now_ns=ts,
    )
    assert d.action == Action.EXIT
    assert "exit_reversion" in [diag.message for diag in d.diagnostics]


def test_time_stop_flatten_near_expiry() -> None:
    """When tau_s < time_stop_seconds and a position is held, v4 should EXIT.

    Warmup note: lookback=600, sampling_dt=60 → warmup_needed=10. Feed 12 stable ticks
    starting at ts=60e9 (sampling gate skips ts=0 since no dt has elapsed from last_sample_ns=0).
    Ticks at 60e9..660e9 yield 11 samples, clearing the warmup gate.
    The final evaluate uses now_ns=0 so tau_s=(500e9-0)/1e9=500s < time_stop_seconds=600.
    No new sample is taken (now_ns=0 < last_sample_ns=660e9) but warmup is already satisfied.
    """
    strat = build_strategy(
        "v4_binary_statarb",
        _params(
            lookback_seconds=600,
            sampling_dt_seconds=60,
            time_stop_seconds=600,
        ),
    )
    qv = _qv(expiry_ns=500 * 10**9)
    # Warm up: 12 ticks at ts=0..660e9; sampling gate fires for ticks at 60e9..660e9 (11 samples)
    for i in range(12):
        ts = i * 60_000_000_000
        books = {"YES": _book("YES", 0.495, 0.505), "NO": _book("NO", 0.495, 0.505)}
        strat.evaluate(
            question=qv,
            books=books,
            reference_price=100_000.0,
            recent_returns=(),
            recent_volume_usd=100.0,
            position=None,
            now_ns=ts,
        )
    pos = Position(
        question_idx=0,
        symbol="YES",
        qty=200.0,
        avg_entry=0.40,
        stop_loss_price=0.0,
        last_update_ts_ns=0,
    )
    d = strat.evaluate(
        question=qv,
        books={"YES": _book("YES", 0.495, 0.505), "NO": _book("NO", 0.495, 0.505)},
        reference_price=100_000.0,
        recent_returns=(),
        recent_volume_usd=100.0,
        position=pos,
        now_ns=0,  # tau_s ~ 500 < 600 threshold
    )
    assert d.action == Action.EXIT
    assert "exit_time_stop" in [diag.message for diag in d.diagnostics]
