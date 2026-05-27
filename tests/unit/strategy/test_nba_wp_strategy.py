from hlanalysis.strategy.types import (
    Action, BookState, Position, QuestionView,
)
from hlanalysis.strategy.theta_harvester import ThetaHarvesterConfig
from hlanalysis.strategy.nba_wp import NBAWinProbStrategy


def _qv(yes_sym="HOME", no_sym="AWAY", strike=0.5, expiry_ns=10_000_000_000):
    return QuestionView(
        question_idx=1,
        yes_symbol=yes_sym, no_symbol=no_sym,
        strike=strike, expiry_ns=expiry_ns,
        underlying="NBA", klass="priceBinary", period="game",
        leg_symbols=(yes_sym, no_sym),
    )


def _book(symbol, bid=None, ask=None, size=1000.0):
    return BookState(
        symbol=symbol,
        bid_px=bid, bid_sz=(size if bid is not None else None),
        ask_px=ask, ask_sz=(size if ask is not None else None),
        last_trade_ts_ns=0, last_l2_ts_ns=0,
    )


def _cfg(**overrides):
    base = dict(
        vol_lookback_seconds=300, vol_sampling_dt_seconds=60,
        vol_clip_min=0.05, vol_clip_max=3.0,
        edge_buffer=0.03, fee_taker=0.0, half_spread_assumption=0.0,
        drift_lookback_seconds=0, drift_blend=0.0,
        max_position_usd=100.0, favorite_threshold=0.9,
        tte_min_seconds=0, tte_max_seconds=10**9,
        stop_loss_pct=None, exit_edge_threshold=0.0,
        take_profit_price=None, time_stop_seconds=0,
        fee_model="pm_binary", fee_rate=0.03,  # PM sports = 0.03
    )
    base.update(overrides)
    return ThetaHarvesterConfig(**base)


def test_enters_home_when_wp_exceeds_ask_by_edge_buffer():
    strat = NBAWinProbStrategy(_cfg())
    qv = _qv()
    books = {"HOME": _book("HOME", bid=0.90, ask=0.91), "AWAY": _book("AWAY", bid=0.08, ask=0.09)}
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=0.97,  # WP says home wins w/ p=0.97
        recent_returns=(), recent_volume_usd=0.0, position=None,
        now_ns=1_000_000_000,
    )
    assert decision.action == Action.ENTER
    intent = decision.intents[0]
    assert intent.symbol == "HOME"
    assert intent.side == "buy"


def test_enters_away_when_wp_low():
    strat = NBAWinProbStrategy(_cfg())
    qv = _qv()
    books = {"HOME": _book("HOME", bid=0.06, ask=0.08), "AWAY": _book("AWAY", bid=0.91, ask=0.92)}
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=0.03,  # away strongly favoured
        recent_returns=(), recent_volume_usd=0.0, position=None,
        now_ns=1_000_000_000,
    )
    assert decision.action == Action.ENTER
    assert decision.intents[0].symbol == "AWAY"


def test_holds_when_favorite_below_threshold():
    """v3.1 PM-tuned favorite_threshold=0.9 → both sides at 50/50 = HOLD."""
    strat = NBAWinProbStrategy(_cfg(favorite_threshold=0.9))
    qv = _qv()
    books = {"HOME": _book("HOME", bid=0.49, ask=0.51), "AWAY": _book("AWAY", bid=0.49, ask=0.51)}
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=0.50,
        recent_returns=(), recent_volume_usd=0.0, position=None,
        now_ns=1_000_000_000,
    )
    assert decision.action == Action.HOLD


def test_holds_when_edge_below_buffer():
    """ask is 0.92, p_yes = 0.93 → edge ~ 0.01 (under 0.03 buffer)."""
    strat = NBAWinProbStrategy(_cfg(edge_buffer=0.03))
    qv = _qv()
    books = {"HOME": _book("HOME", bid=0.91, ask=0.92), "AWAY": _book("AWAY", bid=0.07, ask=0.09)}
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=0.93,
        recent_returns=(), recent_volume_usd=0.0, position=None,
        now_ns=1_000_000_000,
    )
    assert decision.action == Action.HOLD


def test_settlement_closes_position():
    strat = NBAWinProbStrategy(_cfg())
    qv = QuestionView(
        question_idx=1, yes_symbol="HOME", no_symbol="AWAY",
        strike=0.5, expiry_ns=10_000_000_000, underlying="NBA",
        klass="priceBinary", period="game", settled=True, settled_side="yes",
        leg_symbols=("HOME", "AWAY"),
    )
    pos = Position(question_idx=1, symbol="HOME", qty=100.0,
                   avg_entry=0.85, stop_loss_price=0.0, last_update_ts_ns=0)
    books = {"HOME": _book("HOME", bid=0.99, ask=1.00), "AWAY": _book("AWAY", bid=0.0, ask=0.01)}
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=0.95, recent_returns=(), recent_volume_usd=0.0,
        position=pos, now_ns=10_000_000_001,
    )
    assert decision.action == Action.EXIT


def test_no_reference_price_holds():
    """When reference_price is 0 (no WP tick yet), HOLD with diagnostic."""
    strat = NBAWinProbStrategy(_cfg())
    qv = _qv()
    books = {"HOME": _book("HOME", bid=0.6, ask=0.7), "AWAY": _book("AWAY", bid=0.3, ask=0.4)}
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=0.0,
        recent_returns=(), recent_volume_usd=0.0, position=None,
        now_ns=1_000_000_000,
    )
    assert decision.action == Action.HOLD
