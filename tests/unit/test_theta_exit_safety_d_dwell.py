# tests/unit/test_theta_exit_safety_d_dwell.py
"""Tests for the exit_safety_d_dwell_scans confirmation filter on the
theta_harvester exit_safety_d soft-exit.

The dwell filter requires safety_d to remain below exit_safety_d for
exit_safety_d_dwell_scans CONSECUTIVE evaluate() calls before the soft-exit
fires. dwell=1 (default) is bit-identical to the legacy single-scan behavior.
The counter resets on recovery. Hard stops (stop_loss, time_stop, settlement,
exit_edge) are unaffected regardless of the dwell setting.

Binary YES position (strike=100_000):
  - reference_price=90_000  → safety_d strongly negative  → breach
  - reference_price=110_000 → safety_d strongly positive  → safe / recovered
"""

from __future__ import annotations

from hlanalysis.strategy.theta_harvester import ThetaHarvesterConfig, ThetaHarvesterStrategy
from hlanalysis.strategy.types import Action, BookState, Decision, Position, QuestionView

_NOW_NS = 0
_EXPIRY_NS = 600 * 1_000_000_000  # tau ≈ 600 s


def _question(*, settled: bool = False) -> QuestionView:
    return QuestionView(
        question_idx=7,
        yes_symbol="YES",
        no_symbol="NO",
        strike=100_000.0,
        expiry_ns=_EXPIRY_NS,
        underlying="BTC",
        klass="priceBinary",
        period="1d",
        settled=settled,
        leg_symbols=("YES", "NO"),
        kv=(),
    )


def _position(*, stop_loss_price: float = 0.0) -> Position:
    return Position(
        question_idx=7,
        symbol="YES",
        qty=100.0,
        avg_entry=0.85,
        stop_loss_price=stop_loss_price,
        last_update_ts_ns=0,
    )


def _book(sym: str, *, bid: float, ask: float, sz: float = 100.0) -> BookState:
    return BookState(
        symbol=sym,
        bid_px=bid,
        bid_sz=sz,
        ask_px=ask,
        ask_sz=sz,
        last_trade_ts_ns=0,
        last_l2_ts_ns=0,
    )


# Non-zero alternating returns → σ > 0.
_RETS = tuple([0.0005, -0.0005] * 60)

# YES books for the BREACH scenario (ref=90_000): mid ~0.50.
# exit_edge_threshold=-1.0 won't fire: edge_held ≈ 0-0.50=-0.50 > -1.0.
_BREACH_BOOKS: dict[str, BookState] = {
    "YES": _book("YES", bid=0.50, ask=0.60),
    "NO": _book("NO", bid=0.40, ask=0.50),
}

# YES books for the SAFE scenario (ref=110_000).
_SAFE_BOOKS: dict[str, BookState] = {
    "YES": _book("YES", bid=0.90, ask=0.95),
    "NO": _book("NO", bid=0.05, ask=0.10),
}


def _base_cfg(**overrides) -> ThetaHarvesterConfig:
    kwargs: dict = dict(
        vol_lookback_seconds=3600,
        vol_sampling_dt_seconds=60,
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
        exit_edge_threshold=-1.0,
        take_profit_price=None,
        time_stop_seconds=0,
        exit_safety_d=1.0,
        topup_enabled=False,
    )
    kwargs.update(overrides)
    return ThetaHarvesterConfig(**kwargs)


def _strat(**overrides) -> ThetaHarvesterStrategy:
    return ThetaHarvesterStrategy(_base_cfg(**overrides))


def _eval(
    strat: ThetaHarvesterStrategy, q: QuestionView, books: dict, ref: float, *, position: Position | None = None
) -> Decision:
    return strat.evaluate(
        question=q,
        books=books,
        reference_price=ref,
        recent_returns=_RETS,
        recent_volume_usd=1000.0,
        position=position,
        now_ns=_NOW_NS,
    )


def _msgs(dec: Decision) -> list[str]:
    return [d.message for d in dec.diagnostics]


def _kv(dec: Decision, msg: str) -> dict[str, str]:
    for d in dec.diagnostics:
        if d.message == msg:
            return dict(d.fields)
    return {}


# ---------------------------------------------------------------------------
# 1. default dwell=1 is legacy: exit on single breach
# ---------------------------------------------------------------------------


def test_default_dwell_one_exits_on_single_breach() -> None:
    """Default cfg (no dwell kwarg), one breach scan → action EXIT, exit_safety_d."""
    strat = _strat()
    dec = _eval(strat, _question(), _BREACH_BOOKS, 90_000.0, position=_position())
    assert dec.action == Action.EXIT, f"expected EXIT; got {_msgs(dec)}"
    assert "exit_safety_d" in _msgs(dec)


# ---------------------------------------------------------------------------
# 2. explicit dwell=1 diagnostics are bit-identical to the legacy exit
# ---------------------------------------------------------------------------


def test_default_dwell_one_diag_bit_identical() -> None:
    """Explicit dwell=1 produces exactly one Diagnostic with message 'exit_safety_d'
    and kv keys ('exit_reason', 'exit_safety_d', 'exit_threshold')."""
    strat = _strat(exit_safety_d_dwell_scans=1)
    dec = _eval(strat, _question(), _BREACH_BOOKS, 90_000.0, position=_position())
    assert dec.action == Action.EXIT, f"expected EXIT; got {_msgs(dec)}"
    msgs = _msgs(dec)
    assert msgs.count("exit_safety_d") == 1
    kv = _kv(dec, "exit_safety_d")
    assert set(kv.keys()) >= {"exit_reason", "exit_safety_d", "exit_threshold"}


# ---------------------------------------------------------------------------
# 3. single breach with dwell=2 → HOLD (pending)
# ---------------------------------------------------------------------------


def test_transient_breach_does_not_exit_with_dwell_two() -> None:
    """dwell=2, single breach → HOLD with 'exit_safety_d_dwell_pending' diag."""
    strat = _strat(exit_safety_d_dwell_scans=2)
    dec = _eval(strat, _question(), _BREACH_BOOKS, 90_000.0, position=_position())
    assert dec.action == Action.HOLD, f"expected HOLD; got {_msgs(dec)}"
    assert "exit_safety_d_dwell_pending" in _msgs(dec)


# ---------------------------------------------------------------------------
# 4. two consecutive breaches with dwell=2 → exits on scan 2
# ---------------------------------------------------------------------------


def test_persistent_breach_exits_after_two_scans() -> None:
    """dwell=2: scan1 breach→HOLD, scan2 breach (same instance) → EXIT exit_safety_d."""
    strat = _strat(exit_safety_d_dwell_scans=2)
    q, pos = _question(), _position()
    dec1 = _eval(strat, q, _BREACH_BOOKS, 90_000.0, position=pos)
    assert dec1.action == Action.HOLD, f"scan1 should HOLD; {_msgs(dec1)}"
    dec2 = _eval(strat, q, _BREACH_BOOKS, 90_000.0, position=pos)
    assert dec2.action == Action.EXIT, f"scan2 should EXIT; {_msgs(dec2)}"
    assert "exit_safety_d" in _msgs(dec2)


# ---------------------------------------------------------------------------
# 5. dwell=3 exits on the third consecutive breach
# ---------------------------------------------------------------------------


def test_dwell_three_exits_on_third_consecutive_breach() -> None:
    """dwell=3: scans 1 and 2 → HOLD; scan 3 → EXIT exit_safety_d."""
    strat = _strat(exit_safety_d_dwell_scans=3)
    q, pos = _question(), _position()
    dec1 = _eval(strat, q, _BREACH_BOOKS, 90_000.0, position=pos)
    assert dec1.action == Action.HOLD, f"scan1 should HOLD; {_msgs(dec1)}"
    dec2 = _eval(strat, q, _BREACH_BOOKS, 90_000.0, position=pos)
    assert dec2.action == Action.HOLD, f"scan2 should HOLD; {_msgs(dec2)}"
    dec3 = _eval(strat, q, _BREACH_BOOKS, 90_000.0, position=pos)
    assert dec3.action == Action.EXIT, f"scan3 should EXIT; {_msgs(dec3)}"
    assert "exit_safety_d" in _msgs(dec3)


# ---------------------------------------------------------------------------
# 6. recovery resets the streak; needs a full dwell after recovery
# ---------------------------------------------------------------------------


def test_counter_resets_when_safety_d_recovers() -> None:
    """dwell=3: breach×2 (count=2), safe scan (reset), breach×2→HOLD, breach×1→EXIT."""
    strat = _strat(exit_safety_d_dwell_scans=3)
    q, pos = _question(), _position()
    # Breach ×2 → count=2
    _eval(strat, q, _BREACH_BOOKS, 90_000.0, position=pos)
    _eval(strat, q, _BREACH_BOOKS, 90_000.0, position=pos)
    # Safe → counter resets
    safe = _eval(strat, q, _SAFE_BOOKS, 110_000.0, position=pos)
    assert safe.action == Action.HOLD, f"safe scan should HOLD; {_msgs(safe)}"
    # Post-reset: need 3 fresh breaches
    d1 = _eval(strat, q, _BREACH_BOOKS, 90_000.0, position=pos)
    assert d1.action == Action.HOLD, f"post-reset breach 1 should HOLD; {_msgs(d1)}"
    d2 = _eval(strat, q, _BREACH_BOOKS, 90_000.0, position=pos)
    assert d2.action == Action.HOLD, f"post-reset breach 2 should HOLD; {_msgs(d2)}"
    d3 = _eval(strat, q, _BREACH_BOOKS, 90_000.0, position=pos)
    assert d3.action == Action.EXIT, f"post-reset breach 3 should EXIT; {_msgs(d3)}"
    assert "exit_safety_d" in _msgs(d3)


# ---------------------------------------------------------------------------
# 7. hard stop_loss fires immediately regardless of dwell
# ---------------------------------------------------------------------------


def test_hard_stop_loss_fires_immediately_regardless_of_dwell() -> None:
    """dwell=5, stop_loss fires when bid ≤ stop_loss_price; ref=110_000 (safe)."""
    # bid=0.55 ≤ stop_loss_price=0.60 → stop fires
    pos = _position(stop_loss_price=0.60)
    books = {
        "YES": _book("YES", bid=0.55, ask=0.70),
        "NO": _book("NO", bid=0.30, ask=0.45),
    }
    strat = _strat(exit_safety_d_dwell_scans=5, stop_loss_pct=0.5)
    dec = _eval(strat, _question(), books, 110_000.0, position=pos)
    assert dec.action == Action.EXIT, f"stop_loss should EXIT; {_msgs(dec)}"
    assert "exit_stop_loss" in _msgs(dec)


# ---------------------------------------------------------------------------
# 8. time_stop fires immediately regardless of dwell
# ---------------------------------------------------------------------------


def test_time_stop_fires_immediately_regardless_of_dwell() -> None:
    """dwell=5, time_stop_seconds=100_000 > tau≈600 s → scan1 EXIT exit_time_stop."""
    strat = _strat(exit_safety_d_dwell_scans=5, time_stop_seconds=100_000)
    dec = _eval(strat, _question(), _BREACH_BOOKS, 90_000.0, position=_position())
    assert dec.action == Action.EXIT, f"time_stop should EXIT; {_msgs(dec)}"
    assert "exit_time_stop" in _msgs(dec)


# ---------------------------------------------------------------------------
# 9. settlement exit is unaffected by dwell
# ---------------------------------------------------------------------------


def test_settlement_exit_unaffected_by_dwell() -> None:
    """dwell=5, question.settled=True → scan1 EXIT exit_settlement."""
    strat = _strat(exit_safety_d_dwell_scans=5)
    dec = _eval(strat, _question(settled=True), _BREACH_BOOKS, 90_000.0, position=_position())
    assert dec.action == Action.EXIT, f"settlement should EXIT; {_msgs(dec)}"
    assert "exit_settlement" in _msgs(dec)


# ---------------------------------------------------------------------------
# 10. exit_edge still fires with high dwell when bid is far below model
# ---------------------------------------------------------------------------


def test_exit_edge_still_fires_with_high_dwell() -> None:
    """dwell=5, ref=110_000 (safe, no safety_d breach), exit_edge_threshold=0.95,
    bid=0.10 → edge_held ≈ 0.90 < 0.95 → EXIT exit_edge."""
    books = {
        "YES": _book("YES", bid=0.10, ask=0.20),
        "NO": _book("NO", bid=0.80, ask=0.90),
    }
    strat = _strat(exit_safety_d_dwell_scans=5, exit_edge_threshold=0.95)
    dec = _eval(strat, _question(), books, 110_000.0, position=_position())
    assert dec.action == Action.EXIT, f"exit_edge should EXIT; {_msgs(dec)}"
    assert "exit_edge" in _msgs(dec)


# ---------------------------------------------------------------------------
# 11. exit_spread_hold suppresses safety_d even with dwell=1
# ---------------------------------------------------------------------------


def test_exit_spread_hold_suppresses_safety_d_even_with_dwell_one() -> None:
    """dwell=1, breach, but exit_spread_hold=0.05 and half_spread=0.15>0.05 → HOLD."""
    books = {
        "YES": _book("YES", bid=0.40, ask=0.70),  # half-spread = 0.15
        "NO": _book("NO", bid=0.30, ask=0.60),
    }
    strat = _strat(exit_safety_d_dwell_scans=1, exit_spread_hold=0.05)
    dec = _eval(strat, _question(), books, 90_000.0, position=_position())
    assert dec.action == Action.HOLD, f"spread_hold should suppress exit; {_msgs(dec)}"
    assert "hold_spread_too_wide" in _msgs(dec)


# ---------------------------------------------------------------------------
# 12. flat scan (position=None) prunes the dwell counter
# ---------------------------------------------------------------------------


def test_flat_scan_prunes_dwell_counter() -> None:
    """dwell=2: breach→HOLD (count=1); flat scan (position=None) prunes counter;
    single breach again → HOLD (count=1, not 2, so no EXIT)."""
    strat = _strat(exit_safety_d_dwell_scans=2)
    q, pos = _question(), _position()
    # Breach ×1 → count=1
    dec1 = _eval(strat, q, _BREACH_BOOKS, 90_000.0, position=pos)
    assert dec1.action == Action.HOLD, f"scan1 should HOLD; {_msgs(dec1)}"
    # Flat scan (position=None) — favorite_threshold=0.85; mids=0.50 → no_favorite → HOLD
    flat_books = {
        "YES": _book("YES", bid=0.40, ask=0.60),
        "NO": _book("NO", bid=0.40, ask=0.60),
    }
    _eval(strat, q, flat_books, 90_000.0, position=None)
    # Counter pruned; single breach again → HOLD (count=1, not 2)
    dec3 = _eval(strat, q, _BREACH_BOOKS, 90_000.0, position=pos)
    assert dec3.action == Action.HOLD, f"post-prune single breach should HOLD; {_msgs(dec3)}"
