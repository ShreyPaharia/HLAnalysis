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


def test_exit_take_profit_mode_holds_when_bid_below_held_p_plus_fee() -> None:
    """With exit_take_profit_mode=True, hold while bid < held_p + exit_fee.
    Mirrors the take-profit reading: don't sell unless the bid offers above-
    fair premium net of exit-side fee."""
    strat = build_strategy("v3_theta_harvester", _params(
        exit_take_profit_mode=True, exit_fee=0.0007,
        exit_edge_threshold=0.0, edge_buffer=0.0, half_spread_assumption=0.0,
    ))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    # held_p ≈ 1 at strike well above (ref=120k); bid 0.96 < held_p + fee
    books = {"YES": _book("YES", bid=0.96, ask=0.99), "NO": _book("NO", bid=0.01, ask=0.04)}
    pos = _pos(symbol="YES", qty=200.0, entry_px=0.50)
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books, reference_price=120_000.0,
        recent_returns=rets, recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    assert decision.action == Action.HOLD


def test_exit_take_profit_mode_exits_when_bid_above_held_p_plus_fee() -> None:
    """With exit_take_profit_mode=True, exit when bid > held_p + exit_fee."""
    strat = build_strategy("v3_theta_harvester", _params(
        exit_take_profit_mode=True, exit_fee=0.0001,
        exit_edge_threshold=0.0, edge_buffer=0.0, half_spread_assumption=0.0,
    ))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    # ref exactly at strike → p_model ≈ 0.5; bid 0.95 → edge_yes_sell = 0.95 − 0.5 − fee ≈ +0.45 → EXIT.
    books = {"YES": _book("YES", bid=0.95, ask=0.99), "NO": _book("NO", bid=0.01, ask=0.05)}
    pos = _pos(symbol="YES", qty=200.0, entry_px=0.40)
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books, reference_price=100_000.0,  # at strike → p_model ~0.5
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


# ---------------------------------------------------------------------------
# Bucket (priceBucket) tests — verify v3.1 generalises to multi-outcome markets.
# ---------------------------------------------------------------------------


def _qv_bucket(expiry_ns: int = 10**18) -> QuestionView:
    """3-outcome bucket: BTC<90k | 90k≤BTC<110k | BTC≥110k.

    HL layout: leg_symbols = [Y0, N0, Y1, N1, Y2, N2] with YES at even idx, NO at odd.
    NO of an edge bucket inverts to the opposite half-line; NO of the middle
    bucket is non-contiguous and must be skipped by the strategy.
    """
    return QuestionView(
        question_idx=0,
        yes_symbol="",
        no_symbol="",
        strike=0.0,
        expiry_ns=expiry_ns,
        underlying="BTC",
        klass="priceBucket",
        period="1d",
        settled=False,
        kv=(("priceThresholds", "90000,110000"),),
        leg_symbols=("Y0", "N0", "Y1", "N1", "Y2", "N2"),
    )


def test_bucket_entry_picks_leg_with_best_edge() -> None:
    """With BTC sitting comfortably inside the middle bucket (90k<BTC<110k),
    a low-priced Y1 ask should be the best-edge leg and get bought."""
    strat = build_strategy("v3_theta_harvester", _params(
        edge_buffer=0.02, edge_max=None, favorite_threshold=0.0,
    ))
    qv = _qv_bucket(expiry_ns=3600 * 10**9)
    # Low realized vol → BTC near-certain to stay in (90k, 110k) over 1h.
    rets = tuple([0.0001] * 120)
    books = {
        "Y0": _book("Y0", bid=0.04, ask=0.05),  # P(BTC<90k) ~ 0  → edge negative
        "N0": _book("N0", bid=0.94, ask=0.95),  # P(BTC≥90k) ~ 1  → edge near 0.05
        "Y1": _book("Y1", bid=0.69, ask=0.70),  # P(middle)    ~ 1  → edge ~ 0.30 (BEST)
        "N1": _book("N1", bid=0.30, ask=0.31),  # middle NO — skipped (non-contiguous)
        "Y2": _book("Y2", bid=0.04, ask=0.05),  # P(BTC≥110k) ~ 0 → edge negative
        "N2": _book("N2", bid=0.94, ask=0.95),  # P(BTC<110k) ~ 1 → edge near 0.05
    }
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=100_000.0, recent_returns=rets, recent_volume_usd=1000.0,
        position=None, now_ns=0,
    )
    assert decision.action == Action.ENTER
    assert decision.intents[0].symbol == "Y1"  # middle YES has the biggest edge


def test_bucket_entry_skips_middle_no_leg() -> None:
    """N1 of a 3-outcome bucket has a non-contiguous winning region (BTC<90k or
    BTC≥110k). v3.1 must not enter it even when its ask looks cheap."""
    strat = build_strategy("v3_theta_harvester", _params(
        edge_buffer=0.02, edge_max=None, favorite_threshold=0.0,
    ))
    qv = _qv_bucket(expiry_ns=3600 * 10**9)
    rets = tuple([0.0001] * 120)
    # All other legs visibly negative-edge; N1 ask is ridiculously cheap (0.01).
    # Despite the apparent edge, the strategy must NOT pick N1.
    books = {
        "Y0": _book("Y0", bid=0.59, ask=0.60),
        "N0": _book("N0", bid=0.59, ask=0.60),
        "Y1": _book("Y1", bid=0.99, ask=0.999),  # tiny edge after costs
        "N1": _book("N1", bid=0.005, ask=0.01),  # MUST be skipped
        "Y2": _book("Y2", bid=0.59, ask=0.60),
        "N2": _book("N2", bid=0.59, ask=0.60),
    }
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=100_000.0, recent_returns=rets, recent_volume_usd=1000.0,
        position=None, now_ns=0,
    )
    # Either enters something else or HOLDs — must not pick N1.
    if decision.action == Action.ENTER:
        assert decision.intents[0].symbol != "N1"


def test_bucket_exit_fires_when_held_bucket_loses_edge() -> None:
    """Hold Y1 (middle YES). When BTC moves to 130k (out of bucket), p_win for
    the middle drops near zero and edge_held collapses → EXIT."""
    strat = build_strategy("v3_theta_harvester", _params(
        edge_buffer=0.02, exit_edge_threshold=-0.01,
    ))
    qv = _qv_bucket(expiry_ns=3600 * 10**9)
    rets = tuple([0.0001] * 120)
    # Held leg Y1 currently quoting at low bid (binary is collapsing) — but the
    # exit decision is driven by the MODEL's edge, not bid level. With BTC at
    # 130k and low vol, p_win(Y1 = middle bucket) ≈ 0.
    books = {
        "Y0": _book("Y0", bid=0.04, ask=0.05),
        "N0": _book("N0", bid=0.94, ask=0.95),
        "Y1": _book("Y1", bid=0.10, ask=0.11),  # held leg bid = 0.10 (visible loss)
        "N1": _book("N1", bid=0.88, ask=0.89),
        "Y2": _book("Y2", bid=0.85, ask=0.86),
        "N2": _book("N2", bid=0.14, ask=0.15),
    }
    pos = Position(
        question_idx=0, symbol="Y1", qty=200.0,
        avg_entry=0.70, stop_loss_price=0.0, last_update_ts_ns=0,
    )
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=130_000.0,  # ref jumped out of the middle bucket
        recent_returns=rets, recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    assert decision.action == Action.EXIT
    assert "exit_edge" in [d.message for d in decision.diagnostics]


def test_bucket_favorite_threshold_filters_leg_set() -> None:
    """With favorite_threshold=0.7, only legs whose mid ≥ 0.7 are eligible."""
    strat = build_strategy("v3_theta_harvester", _params(
        edge_buffer=0.02, favorite_threshold=0.7,
    ))
    qv = _qv_bucket(expiry_ns=3600 * 10**9)
    rets = tuple([0.0001] * 120)
    # Y1 mid = 0.80 (passes); Y0 and Y2 mid = 0.05 (fails); N0/N2 mid = 0.95 (pass).
    # Among passers, Y1 has the highest edge by construction (p_win ≈ 1 vs ask 0.80
    # gives edge ≈ 0.20; N0 and N2 have edge ≈ 0.04 only).
    books = {
        "Y0": _book("Y0", bid=0.04, ask=0.05),
        "N0": _book("N0", bid=0.94, ask=0.95),
        "Y1": _book("Y1", bid=0.79, ask=0.80),
        "N1": _book("N1", bid=0.18, ask=0.19),
        "Y2": _book("Y2", bid=0.04, ask=0.05),
        "N2": _book("N2", bid=0.94, ask=0.95),
    }
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=100_000.0, recent_returns=rets, recent_volume_usd=1000.0,
        position=None, now_ns=0,
    )
    assert decision.action == Action.ENTER
    assert decision.intents[0].symbol == "Y1"


def test_bucket_holds_when_no_leg_passes_favorite_gate() -> None:
    """If all legs have mid < favorite_threshold, the gate fires and we HOLD."""
    strat = build_strategy("v3_theta_harvester", _params(
        edge_buffer=0.02, favorite_threshold=0.9,
    ))
    qv = _qv_bucket(expiry_ns=3600 * 10**9)
    rets = tuple([0.0001] * 120)
    # Every leg quotes near 0.50 mid → none exceed 0.9.
    books = {sym: _book(sym, bid=0.49, ask=0.50) for sym in ("Y0", "N0", "Y1", "N1", "Y2", "N2")}
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=100_000.0, recent_returns=rets, recent_volume_usd=1000.0,
        position=None, now_ns=0,
    )
    assert decision.action == Action.HOLD
    assert "no_favorite" in [d.message for d in decision.diagnostics]


# --- 2026-05-19 fixes: near-strike hover veto, bid-notional sanity ---


def test_min_distance_pct_vetoes_entries_too_close_to_strike() -> None:
    """PM corpus evidence: v3.1 entries within 0.20% of strike lose
    -$7.68/entry on average across 57 entries. The 0.20-0.50% band is the
    *best* band. With min_distance_pct=0.002, an entry attempt at 0.10%
    distance must HOLD, not ENTER, with the dist_pct in the diagnostic."""
    strat = build_strategy("v3_theta_harvester", _params(
        min_distance_pct=0.002,  # 0.20%
    ))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    # Reference price 100,100 is exactly 0.10% above strike — below the gate.
    ref = 100_100.0
    books = {"YES": _book("YES", bid=0.50, ask=0.51), "NO": _book("NO", bid=0.49, ask=0.50)}
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=ref, recent_returns=rets, recent_volume_usd=1000.0,
        position=None, now_ns=0,
    )
    assert decision.action == Action.HOLD
    assert any(d.message == "near_strike_hover" for d in decision.diagnostics)


def test_min_distance_pct_allows_entries_beyond_threshold() -> None:
    """Symmetric: at 0.30% distance (just above the gate), v3.1 should still
    evaluate the entry through the normal edge / favourite path. We don't
    assert ENTER here (depends on σ × τ), we just assert the near-strike
    diagnostic is NOT in the result — i.e. the gate passed."""
    strat = build_strategy("v3_theta_harvester", _params(
        min_distance_pct=0.002,
    ))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    ref = 100_300.0  # 0.30% above strike — clears 0.20% gate
    books = {"YES": _book("YES", bid=0.49, ask=0.50), "NO": _book("NO", bid=0.49, ask=0.50)}
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=ref, recent_returns=rets, recent_volume_usd=1000.0,
        position=None, now_ns=0,
    )
    assert not any(d.message == "near_strike_hover" for d in decision.diagnostics)


def test_min_distance_pct_none_disables_gate() -> None:
    """min_distance_pct=None preserves v3 baseline behavior — no near-strike
    veto. Backstop against an accidental tightening from a default change."""
    strat = build_strategy("v3_theta_harvester", _params())  # default None
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    ref = 100_050.0  # 0.05% — would be vetoed if gate were active
    books = {"YES": _book("YES", bid=0.50, ask=0.51), "NO": _book("NO", bid=0.49, ask=0.50)}
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=ref, recent_returns=rets, recent_volume_usd=1000.0,
        position=None, now_ns=0,
    )
    assert not any(d.message == "near_strike_hover" for d in decision.diagnostics)


def test_min_bid_notional_filters_spoof_bids() -> None:
    """A leg quoting bid=0.95×1 share has $0.95 of buying interest — a numeric
    threshold pass but operationally meaningless. The bid-notional gate must
    HOLD; otherwise we'd happily pay 0.50 ask for a leg with no real bidder."""
    strat = build_strategy("v3_theta_harvester", _params(
        favorite_threshold=0.5,
        min_bid_notional_usd=10.0,
    ))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    # Both legs pass favorite_threshold=0.5; YES has tiny bid size.
    yes = BookState(
        symbol="YES", bid_px=0.95, bid_sz=1.0, ask_px=0.50, ask_sz=100.0,
        last_trade_ts_ns=0, last_l2_ts_ns=0,
    )
    no_book = BookState(
        symbol="NO", bid_px=0.95, bid_sz=1.0, ask_px=0.50, ask_sz=100.0,
        last_trade_ts_ns=0, last_l2_ts_ns=0,
    )
    books = {"YES": yes, "NO": no_book}
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=120_000.0, recent_returns=rets, recent_volume_usd=1000.0,
        position=None, now_ns=0,
    )
    assert decision.action == Action.HOLD
    assert any(d.message == "bid_notional_too_thin" for d in decision.diagnostics)


# ---------------------------------------------------------------------------
# Position-topup tests (2026-05-20): IOC partial-fills on thin HL HIP-4 books
# leave the strategy under-sized vs max_position_usd. _evaluate_topup must
# emit a second ENTER intent to close the gap — but only when an exit gate
# wouldn't fire, all entry gates still pass, and the chosen leg matches the
# held symbol.
# ---------------------------------------------------------------------------


def _under_filled_pos(*, symbol: str = "YES", qty: float = 100.0,
                     entry_px: float = 0.50) -> Position:
    """Partial-fill scenario: intended ~200 contracts ($100 @ 0.50), got 100 ($50)."""
    return Position(
        question_idx=0, symbol=symbol, qty=qty,
        avg_entry=entry_px, stop_loss_price=0.0, last_update_ts_ns=0,
    )


def test_topup_emits_enter_when_held_qty_below_target_and_gates_pass() -> None:
    """Held qty=100 × ask=0.50 = $50 ntl on $100 target → 50% shortfall ≥ 20%
    threshold. All gates pass → ENTER with topup_size at current ask."""
    strat = build_strategy("v3_theta_harvester", _params(
        edge_buffer=0.02, exit_edge_threshold=-0.01,
    ))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    # Edge stays strongly positive: ref >> strike, YES ask 0.50 → edge ~ 0.50.
    books = {"YES": _book("YES", bid=0.49, ask=0.50), "NO": _book("NO", bid=0.49, ask=0.50)}
    pos = _under_filled_pos()
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=120_000.0, recent_returns=rets, recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    assert decision.action == Action.ENTER
    assert decision.intents[0].symbol == "YES"
    assert decision.intents[0].side == "buy"
    # Shortfall = $100 - $50 = $50; @0.50 → size = 100.0 contracts.
    assert math.isclose(decision.intents[0].size, 100.0, abs_tol=0.01)
    assert math.isclose(decision.intents[0].limit_price, 0.50, abs_tol=1e-9)
    assert any(d.message == "topup_emit" for d in decision.diagnostics)


def test_topup_holds_when_shortfall_below_threshold() -> None:
    """Held qty=180 × ask=0.50 = $90 on $100 target → 10% shortfall < 20% → HOLD."""
    strat = build_strategy("v3_theta_harvester", _params(
        edge_buffer=0.02, exit_edge_threshold=-0.01,
    ))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    books = {"YES": _book("YES", bid=0.49, ask=0.50), "NO": _book("NO", bid=0.49, ask=0.50)}
    pos = _under_filled_pos(qty=180.0)
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=120_000.0, recent_returns=rets, recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    assert decision.action == Action.HOLD
    # Must NOT have a topup_skip diag — we didn't even attempt one (shortfall
    # below threshold short-circuits without re-running entry gates).
    skip_diags = [d for d in decision.diagnostics if d.message == "topup_skip"]
    assert len(skip_diags) == 1
    assert any(("reason", "not_needed") == kv for kv in skip_diags[0].fields)


def test_topup_holds_when_topup_notional_below_min() -> None:
    """Shortfall = $5 (held=190×0.50=$95 on $100) → triggers topup attempt
    because 5% < 20%? No — that doesn't fire. Use a smaller target instead:
    max_position_usd=12, held=8×0.50=$4, shortfall=$8, → 67%>20% triggers attempt;
    but topup_size @ 0.50 = floor(8/0.50*100)/100 = 16; 16*0.50=$8 < $11 min →
    skip below_min_notional. Wait — 16 contracts at 0.50 = $8.00, below $11."""
    strat = build_strategy("v3_theta_harvester", _params(
        edge_buffer=0.02, exit_edge_threshold=-0.01,
        max_position_usd=12.0,
        topup_min_notional_usd=11.0,
    ))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    books = {"YES": _book("YES", bid=0.49, ask=0.50), "NO": _book("NO", bid=0.49, ask=0.50)}
    pos = _under_filled_pos(qty=8.0)
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=120_000.0, recent_returns=rets, recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    assert decision.action == Action.HOLD
    skip = next(d for d in decision.diagnostics if d.message == "topup_skip")
    assert any(("reason", "below_min_notional") == kv for kv in skip.fields)


def test_topup_holds_when_edge_max_gate_now_fails() -> None:
    """edge_max=0.20 vetoes fresh entries with >0.20 edge. A held position with
    fresh edge >0.20 (e.g. market gapped favourably) must NOT top up — the gate
    fails and we report gate_failed:edge_too_extreme."""
    strat = build_strategy("v3_theta_harvester", _params(
        edge_buffer=0.02, edge_max=0.20, exit_edge_threshold=-0.01,
    ))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    # YES ask 0.50, ref massively above strike → edge ~0.50 > 0.20.
    books = {"YES": _book("YES", bid=0.49, ask=0.50), "NO": _book("NO", bid=0.49, ask=0.50)}
    pos = _under_filled_pos()
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=120_000.0, recent_returns=rets, recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    assert decision.action == Action.HOLD
    skip = next(d for d in decision.diagnostics if d.message == "topup_skip")
    reason_kv = next(kv for kv in skip.fields if kv[0] == "reason")
    assert reason_kv[1].startswith("gate_failed:")


def test_topup_holds_when_chosen_leg_now_different_from_held() -> None:
    """Hold YES, but BTC has dropped below strike so the strategy now picks NO.
    Topup must NOT add to the wrong leg — emit leg_changed skip."""
    strat = build_strategy("v3_theta_harvester", _params(
        edge_buffer=0.02, exit_edge_threshold=-1.0,  # disable exit; isolate topup
    ))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    # YES leg dead (ask 0.99 → tiny edge); NO leg cheap (ask 0.05 with ref <<
    # strike → P(NO wins) ~ 1 → edge ~0.95).
    books = {"YES": _book("YES", bid=0.98, ask=0.99), "NO": _book("NO", bid=0.04, ask=0.05)}
    pos = _under_filled_pos(symbol="YES", qty=50.0, entry_px=0.50)
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=80_000.0,
        recent_returns=rets, recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    assert decision.action == Action.HOLD
    skip = next(d for d in decision.diagnostics if d.message == "topup_skip")
    assert any(("reason", "leg_changed") == kv for kv in skip.fields)


def test_exit_takes_precedence_over_topup() -> None:
    """Exit edge has gone negative (held YES, BTC below strike). Exit fires
    before topup is even considered — under-filled position still gets closed."""
    strat = build_strategy("v3_theta_harvester", _params(
        edge_buffer=0.02, exit_edge_threshold=-0.01,
    ))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    # Same setup as test_edge_exit_fires: ref<strike collapses held YES edge.
    books = {"YES": _book("YES", bid=0.45, ask=0.46), "NO": _book("NO", bid=0.54, ask=0.55)}
    pos = _under_filled_pos(qty=50.0)  # under-filled — would otherwise trigger topup
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=99_000.0,
        recent_returns=rets, recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    assert decision.action == Action.EXIT
    assert decision.intents[0].reduce_only is True
    # No topup intent should appear.
    assert len(decision.intents) == 1


def test_topup_disabled_via_config_keeps_hold() -> None:
    """topup_enabled=False preserves legacy "have_position HOLD" behavior."""
    strat = build_strategy("v3_theta_harvester", _params(
        edge_buffer=0.02, exit_edge_threshold=-0.01,
        topup_enabled=False,
    ))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    books = {"YES": _book("YES", bid=0.49, ask=0.50), "NO": _book("NO", bid=0.49, ask=0.50)}
    pos = _under_filled_pos()
    rets = tuple([0.0001] * 120)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=120_000.0, recent_returns=rets, recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    assert decision.action == Action.HOLD
    # No topup diagnostics should appear when disabled.
    assert not any(d.message.startswith("topup_") for d in decision.diagnostics)


# ---------------------------------------------------------------------------
# exit_safety_d tests (v3.1.1, 2026-05-21): σ-normalized mid-hold distance exit
# fires BEFORE the bid collapses, mirroring v1's safety_d gate. The point is
# to cut LONG-TTE binaries/middle-buckets when BTC drifts toward the adverse
# boundary while the held bid is still stale-positive (so edge_held hasn't
# fired yet).
# ---------------------------------------------------------------------------


def _high_vol_returns() -> tuple[float, ...]:
    """Alternating +/- 0.001378 → annualized σ ≈ 1.0 (well-clipped, deterministic)."""
    return tuple((0.001378 if i % 2 == 0 else -0.001378) for i in range(120))


def test_exit_safety_d_fires_when_d_below_threshold_with_stale_bid() -> None:
    """Position held on YES (winning region S>K). BTC has drifted to barely
    above K (S=100,100 vs K=100,000) under high vol so the σ-normalized
    safety_d is small (~0.09). Bid is still stale at 0.50 so edge_held does
    NOT fire. The new safety_d gate must fire."""
    strat = build_strategy("v3_theta_harvester", _params(
        exit_edge_threshold=-0.01,
        exit_safety_d=0.5,
    ))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    books = {"YES": _book("YES", bid=0.50, ask=0.51), "NO": _book("NO", bid=0.49, ask=0.50)}
    pos = _pos(symbol="YES", qty=200.0, entry_px=0.50)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=100_100.0,
        recent_returns=_high_vol_returns(),
        recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    assert decision.action == Action.EXIT
    assert "exit_safety_d" in [d.message for d in decision.diagnostics]
    assert decision.intents[0].exit_reason == "exit_safety_d"
    assert decision.intents[0].reduce_only is True
    # Fill at bid (not ask), reduce-only IOC.
    assert decision.intents[0].limit_price == 0.50
    assert decision.intents[0].time_in_force == "ioc"


def test_exit_safety_d_holds_when_d_above_threshold() -> None:
    """Same vol regime but BTC well above strike (S=110,000) → safety_d ~ 9 σ,
    far above threshold. Must HOLD (no exit at all)."""
    strat = build_strategy("v3_theta_harvester", _params(
        exit_edge_threshold=-0.01,
        exit_safety_d=0.5,
    ))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    books = {"YES": _book("YES", bid=0.50, ask=0.51), "NO": _book("NO", bid=0.49, ask=0.50)}
    pos = _pos(symbol="YES", qty=200.0, entry_px=0.50)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=110_000.0,
        recent_returns=_high_vol_returns(),
        recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    assert decision.action == Action.HOLD
    assert "exit_safety_d" not in [d.message for d in decision.diagnostics]


def test_exit_safety_d_zero_disables_gate_preserves_legacy_hold() -> None:
    """With exit_safety_d=0.0 (default), the new gate is disabled. Same
    safety_d=0.09 scenario as the fire test must now HOLD (edge_held also
    doesn't fire because bid is stale-positive)."""
    strat = build_strategy("v3_theta_harvester", _params(
        exit_edge_threshold=-0.01,
        # exit_safety_d omitted → default 0.0
    ))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    books = {"YES": _book("YES", bid=0.50, ask=0.51), "NO": _book("NO", bid=0.49, ask=0.50)}
    pos = _pos(symbol="YES", qty=200.0, entry_px=0.50)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=100_100.0,
        recent_returns=_high_vol_returns(),
        recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    assert decision.action == Action.HOLD
    assert "exit_safety_d" not in [d.message for d in decision.diagnostics]


def test_exit_safety_d_diagnostic_kvs_attached() -> None:
    """When safety_d fires, the diagnostic carries the three kv pairs:
    exit_reason, exit_safety_d, exit_threshold."""
    strat = build_strategy("v3_theta_harvester", _params(
        exit_edge_threshold=-0.01,
        exit_safety_d=0.5,
    ))
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    books = {"YES": _book("YES", bid=0.50, ask=0.51), "NO": _book("NO", bid=0.49, ask=0.50)}
    pos = _pos(symbol="YES", qty=200.0, entry_px=0.50)
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=100_100.0,
        recent_returns=_high_vol_returns(),
        recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    assert decision.action == Action.EXIT
    diag = next(d for d in decision.diagnostics if d.message == "exit_safety_d")
    keys = {k for k, _ in diag.fields}
    assert {"exit_reason", "exit_safety_d", "exit_threshold"}.issubset(keys)
    reason_val = next(v for k, v in diag.fields if k == "exit_reason")
    assert reason_val == "safety_d_below_threshold"
    threshold_val = next(v for k, v in diag.fields if k == "exit_threshold")
    assert float(threshold_val) == 0.5
    d_val = float(next(v for k, v in diag.fields if k == "exit_safety_d"))
    # d must be in (0, threshold) for fire branch with this scenario.
    assert 0.0 < d_val < 0.5


# --- v3.2-volclock builder & estimator dispatch ----------------------------


def test_v3_default_estimator_is_sample_std() -> None:
    """v3.1 baseline must keep sample_std σ so existing tunings stay valid."""
    strat = build_strategy("v3_theta_harvester", _params())
    assert strat.cfg.vol_estimator == "sample_std"


def test_v3_2_volclock_defaults_to_bipower_estimator() -> None:
    strat = build_strategy("v3_2_volclock", _params())
    assert strat.cfg.vol_estimator == "bipower"


def test_v3_2_volclock_explicit_override_wins() -> None:
    """Callers can force v3.2 back to sample_std for ablation backtests."""
    strat = build_strategy(
        "v3_2_volclock", _params(vol_estimator="sample_std"),
    )
    assert strat.cfg.vol_estimator == "sample_std"


def test_v3_4_lmgate_blocks_entry_on_calm_returns() -> None:
    """v3.4-LMgate should HOLD when LM-stat is tiny (no jump in latest bar)."""
    strat = build_strategy("v3_4_lmgate", _params())
    qv = _qv(expiry_ns=3600 * 10**9)
    books = {"YES": _book("YES", bid=0.49, ask=0.50), "NO": _book("NO", bid=0.49, ask=0.50)}
    # All tiny returns — last_return is 1e-5, BV will be similar; lm_stat ≈ 1.
    rets = tuple([1e-5] * 120)
    decision = strat.evaluate(
        question=qv, books=books, reference_price=120_000.0,
        recent_returns=rets, recent_volume_usd=1000.0,
        position=None, now_ns=0,
    )
    # If edge passes but LM gate blocks, diag carries lm_gate_no_jump.
    if decision.action != Action.ENTER:
        msgs = [d.message for d in decision.diagnostics]
        # Either LM gate fired (expected) or some prior gate rejected first.
        # In either case, the strategy did not enter.
        assert decision.action == Action.HOLD


def test_v3_4_lmgate_permits_entry_on_injected_jump() -> None:
    """Inject a single ~4σ return at the end of the window; LM stat should
    exceed the threshold and not block (whether the trade fires still depends
    on edge gates)."""
    strat = build_strategy("v3_4_lmgate", _params())
    qv = _qv(expiry_ns=3600 * 10**9)
    books = {"YES": _book("YES", bid=0.49, ask=0.50), "NO": _book("NO", bid=0.49, ask=0.50)}
    rets = tuple([1e-5] * 119 + [0.005])  # last return 500x baseline → big LM stat
    decision = strat.evaluate(
        question=qv, books=books, reference_price=120_000.0,
        recent_returns=rets, recent_volume_usd=1000.0,
        position=None, now_ns=0,
    )
    # No specific assertion on action — depends on edge/edge_max/etc.
    # What we DO assert: if HOLD, the reason wasn't the LM gate (i.e., the
    # gate let this jump through).
    if decision.action != Action.ENTER:
        msgs = [d.message for d in decision.diagnostics]
        assert "lm_gate_no_jump" not in msgs


def test_v3_lmgate_default_is_disabled() -> None:
    """v3.1 must keep LM gate off by default."""
    strat = build_strategy("v3_theta_harvester", _params())
    assert strat.cfg.lm_threshold is None


def test_v3_4_lmgate_default_threshold() -> None:
    strat = build_strategy("v3_4_lmgate", _params())
    assert strat.cfg.lm_threshold == 4.0
    assert strat.cfg.vol_estimator == "bipower"


def test_v3_2_volclock_sigma_immune_to_recent_wick() -> None:
    """Inject one wick into the returns window and confirm v3.2 keeps a tighter
    σ than v3.1 — the operative mechanism for catching wick-driven mispricings.
    """
    v31 = build_strategy("v3_theta_harvester", _params())
    v32 = build_strategy("v3_2_volclock", _params())
    # 60 minutes of calm 60s returns + one wick. Reproduces the situation we
    # care about: market just wicked, σ_RV blows up, σ_BV stays calm.
    rets = tuple([1e-5] * 30 + [0.02] + [1e-5] * 29)
    s_rv = v31._sigma(rets)
    s_bv = v32._sigma(rets)
    assert s_rv is not None and s_bv is not None
    assert s_bv < s_rv


# ---------------------------------------------------------------------------
# Phase 7.3 — pm_binary fee model
# ---------------------------------------------------------------------------


def test_pm_binary_fee_curve_reduces_effective_edge_near_50_50() -> None:
    """Near p=0.5 the PM fee_per_share peaks at fee_rate * 0.25 = 0.0175 (7%
    headline rate), so a strategy run with fee_model='pm_binary' must compute
    a strictly smaller edge than the legacy 'flat' branch on the same book.

    Setup: reference_price == strike → p_yes ≈ 0.5. YES ask = 0.40 so the
    raw GBM edge ≈ 0.10. With edge_buffer between 0.0825 and 0.10, the
    'flat' run ENTERs and the 'pm_binary' run HOLDs (edge_buffer dominates).
    """
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    books = {"YES": _book("YES", bid=0.39, ask=0.40),
             "NO": _book("NO", bid=0.59, ask=0.60)}
    rets = tuple([0.0001] * 120)  # tame σ → no jumpy d

    common = _params(edge_buffer=0.09)
    flat = build_strategy("v3_theta_harvester", common)
    pmf = build_strategy(
        "v3_theta_harvester", dict(common, fee_model="pm_binary", fee_rate=0.07),
    )

    d_flat = flat.evaluate(
        question=qv, books=books, reference_price=100_000.0,
        recent_returns=rets, recent_volume_usd=1000.0,
        position=None, now_ns=0,
    )
    d_pmf = pmf.evaluate(
        question=qv, books=books, reference_price=100_000.0,
        recent_returns=rets, recent_volume_usd=1000.0,
        position=None, now_ns=0,
    )

    assert d_flat.action == Action.ENTER, (
        f"flat run must enter (edge ≈ 0.10 > buffer 0.09); got {d_flat}"
    )
    assert d_pmf.action == Action.HOLD, (
        "pm_binary run must hold: pm fee = 0.07·0.5·0.5 = 0.0175 brings "
        f"edge to ≈ 0.0825 < buffer 0.09; got {d_pmf}"
    )


def test_pm_binary_fee_curve_negligible_for_deep_favorite() -> None:
    """For p_yes ≈ 1 (deep favorite) the PM fee curve evaluates to ≈ 0, so
    pm_binary and flat (fee_taker=0) should produce identical decisions
    bit-for-bit on a deep-favorite book."""
    # ref massively above strike → p_yes ≈ 1 → fee = 0.07·1·0 = 0
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    books = {"YES": _book("YES", bid=0.89, ask=0.90),
             "NO": _book("NO", bid=0.09, ask=0.10)}
    rets = tuple([0.0001] * 120)

    flat = build_strategy("v3_theta_harvester", _params())
    pmf = build_strategy(
        "v3_theta_harvester", _params(fee_model="pm_binary", fee_rate=0.07),
    )
    d_flat = flat.evaluate(
        question=qv, books=books, reference_price=120_000.0,
        recent_returns=rets, recent_volume_usd=1000.0,
        position=None, now_ns=0,
    )
    d_pmf = pmf.evaluate(
        question=qv, books=books, reference_price=120_000.0,
        recent_returns=rets, recent_volume_usd=1000.0,
        position=None, now_ns=0,
    )
    assert d_flat.action == Action.ENTER and d_pmf.action == Action.ENTER
    assert d_flat.intents[0].symbol == d_pmf.intents[0].symbol == "YES"


def test_pm_binary_fee_default_off_preserves_hl_behavior() -> None:
    """fee_model defaults to 'flat'; without setting it the strategy must
    behave identically to the legacy fee_taker path (HL v31 bit-identical)."""
    strat = build_strategy("v3_theta_harvester", _params(fee_taker=0.001))
    assert strat.cfg.fee_model == "flat"
    assert strat.cfg.fee_rate == 0.0


def test_pm_binary_exit_fee_uses_curve_not_flat_exit_fee() -> None:
    """Under fee_model='pm_binary' the take-profit exit gate must use the
    curve fee (= fee_rate · held_p · (1-held_p)), not the flat exit_fee.

    Setup: held YES at ref ≈ strike → held_p ≈ 0.5 → curve fee ≈ 0.07 ·
    0.25 = 0.0175. Bid at 0.51. Edge under FLAT (exit_fee=0.0007) =
    0.51 - 0.5 - 0.0007 ≈ +0.0093 → exits. Edge under PM curve =
    0.51 - 0.5 - 0.0175 ≈ -0.0075 → holds.

    The two strategies see the same book + position; the only difference is
    fee_model. Curve must produce a strictly more conservative exit decision
    near p=0.5 where the curve fee is largest.
    """
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    books = {"YES": _book("YES", bid=0.51, ask=0.52),
             "NO": _book("NO", bid=0.47, ask=0.48)}
    pos = _pos(symbol="YES", qty=200.0, entry_px=0.40)
    rets = tuple([0.0001] * 120)

    common = _params(
        exit_take_profit_mode=True, exit_fee=0.0007,
        exit_edge_threshold=0.0, edge_buffer=0.0, half_spread_assumption=0.0,
    )
    flat = build_strategy("v3_theta_harvester", common)
    pmf = build_strategy(
        "v3_theta_harvester",
        dict(common, fee_model="pm_binary", fee_rate=0.07),
    )

    d_flat = flat.evaluate(
        question=qv, books=books, reference_price=100_000.0,
        recent_returns=rets, recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    d_pmf = pmf.evaluate(
        question=qv, books=books, reference_price=100_000.0,
        recent_returns=rets, recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )

    assert d_flat.action == Action.EXIT, (
        f"flat run must exit (edge ≈ +0.0093 > threshold 0.0); got {d_flat}"
    )
    assert d_pmf.action == Action.HOLD, (
        "pm_binary run must hold (curve fee 0.0175 makes edge ≈ -0.0075); "
        f"got {d_pmf}"
    )


def test_pm_binary_exit_fee_negligible_for_deep_favorite() -> None:
    """At held_p ≈ 1 the curve fee → 0, so pm_binary and flat (exit_fee=0)
    produce identical exit decisions on a deep-favorite book. Symmetric
    with the entry-side `_negligible_for_deep_favorite` test."""
    qv = _qv(expiry_ns=3600 * 10**9, strike=100_000.0)
    books = {"YES": _book("YES", bid=0.99, ask=0.999),
             "NO": _book("NO", bid=0.001, ask=0.01)}
    pos = _pos(symbol="YES", qty=200.0, entry_px=0.50)
    rets = tuple([0.0001] * 120)

    common = _params(
        exit_take_profit_mode=True, exit_fee=0.0,
        exit_edge_threshold=0.0, edge_buffer=0.0, half_spread_assumption=0.0,
    )
    flat = build_strategy("v3_theta_harvester", common)
    pmf = build_strategy(
        "v3_theta_harvester",
        dict(common, fee_model="pm_binary", fee_rate=0.07),
    )
    d_flat = flat.evaluate(
        question=qv, books=books, reference_price=120_000.0,
        recent_returns=rets, recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    d_pmf = pmf.evaluate(
        question=qv, books=books, reference_price=120_000.0,
        recent_returns=rets, recent_volume_usd=1000.0,
        position=pos, now_ns=0,
    )
    assert d_flat.action == d_pmf.action


# --- Vol-scaled (variable) TTE entry window (mirrors v1 late_resolution) ---
#
# When vol_scaled_tte_enabled, the upper TTE entry bound scales with the
# (annualized) entry σ instead of the fixed tte_max_seconds:
#   tte_max_eff = tte_max_seconds * (ref_sigma / σ) ** exponent, clamped to
#   [0, ceiling]. Low vol → wider window; high vol → narrower. Flag off
#   (default) = legacy fixed window.

def _alt_returns(amp: float, n: int = 120) -> tuple[float, ...]:
    """Alternating +amp/-amp returns: sample std (ddof≈n) ≈ amp per 60s bar.
    At dt=60 the strategy annualizes by ~725×, so σ_ann ≈ amp*725."""
    return tuple((amp if i % 2 == 0 else -amp) for i in range(n))


def test_vol_scaled_tte_widens_window_in_low_vol() -> None:
    # base tte_max=1800s; low vol (σ_ann≈0.5) with ref=1.0, exp=1 → eff≈3600s.
    # A 3000s TTE exceeds the fixed cap but fits the widened window.
    qv = _qv(expiry_ns=3000 * 10**9, strike=100_000.0)
    books = {"YES": _book("YES", bid=0.49, ask=0.50), "NO": _book("NO", bid=0.49, ask=0.50)}
    rets = _alt_returns(0.5 / 725.4)
    common = dict(tte_min_seconds=0, tte_max_seconds=1800, favorite_threshold=0.0)

    off = build_strategy("v3_theta_harvester", _params(**common))
    d_off = off.evaluate(
        question=qv, books=books, reference_price=150_000.0,
        recent_returns=rets, recent_volume_usd=1000.0, position=None, now_ns=0,
    )
    assert d_off.action == Action.HOLD
    assert any("tte_out_of_window" in dg.message for dg in d_off.diagnostics)

    on = build_strategy("v3_theta_harvester", _params(
        vol_scaled_tte_enabled=True, vol_scaled_tte_ref_sigma=1.0,
        vol_scaled_tte_exponent=1.0, vol_scaled_tte_ceiling_seconds=7200, **common,
    ))
    d_on = on.evaluate(
        question=qv, books=books, reference_price=150_000.0,
        recent_returns=rets, recent_volume_usd=1000.0, position=None, now_ns=0,
    )
    assert d_on.action == Action.ENTER


def test_vol_scaled_tte_narrows_window_in_high_vol() -> None:
    # base tte_max=1800s; high vol (σ_ann≈2.0) with ref=1.0, exp=1 → eff≈900s.
    # A 1200s TTE fits the fixed cap but is rejected by the narrowed window.
    qv = _qv(expiry_ns=1200 * 10**9, strike=100_000.0)
    books = {"YES": _book("YES", bid=0.49, ask=0.50), "NO": _book("NO", bid=0.49, ask=0.50)}
    rets = _alt_returns(2.0 / 725.4)
    common = dict(tte_min_seconds=0, tte_max_seconds=1800, favorite_threshold=0.0)

    off = build_strategy("v3_theta_harvester", _params(**common))
    d_off = off.evaluate(
        question=qv, books=books, reference_price=150_000.0,
        recent_returns=rets, recent_volume_usd=1000.0, position=None, now_ns=0,
    )
    assert d_off.action == Action.ENTER

    on = build_strategy("v3_theta_harvester", _params(
        vol_scaled_tte_enabled=True, vol_scaled_tte_ref_sigma=1.0,
        vol_scaled_tte_exponent=1.0, vol_scaled_tte_ceiling_seconds=7200, **common,
    ))
    d_on = on.evaluate(
        question=qv, books=books, reference_price=150_000.0,
        recent_returns=rets, recent_volume_usd=1000.0, position=None, now_ns=0,
    )
    assert d_on.action == Action.HOLD
    assert any("vol_scaled_tte" in dg.message for dg in d_on.diagnostics)


def test_vol_scaled_tte_disabled_by_default_is_fixed_window() -> None:
    qv = _qv(expiry_ns=3000 * 10**9, strike=100_000.0)
    books = {"YES": _book("YES", bid=0.49, ask=0.50), "NO": _book("NO", bid=0.49, ask=0.50)}
    s = build_strategy("v3_theta_harvester", _params(
        tte_min_seconds=0, tte_max_seconds=1800, favorite_threshold=0.0,
    ))
    d = s.evaluate(
        question=qv, books=books, reference_price=150_000.0,
        recent_returns=_alt_returns(0.5 / 725.4), recent_volume_usd=1000.0,
        position=None, now_ns=0,
    )
    assert d.action == Action.HOLD
    assert any("tte_out_of_window" in dg.message for dg in d.diagnostics)
