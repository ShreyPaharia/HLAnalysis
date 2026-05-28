"""v3.6 Jump-Ratio trust weight tests.

Tests:
1. Default off is v3.5 bit-identical.
2. Pure continuous returns → trust scalar near 1.0.
3. Jump-heavy returns → trust ≪ 1 → tilt less aggressive.
"""
from __future__ import annotations

from hlanalysis.strategy.theta_harvester import (
    ThetaHarvesterConfig,
    ThetaHarvesterStrategy,
    _jr_trust_weight,
)
from hlanalysis.strategy.types import (
    Action,
    BookState,
    QuestionView,
)

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


def _books_yes_fav() -> dict:
    """YES at ~0.95 mid (bid=ask=0.94). Edge vs ask ≈ p_yes - 0.94."""
    return {
        "YES": BookState(
            symbol="YES", bid_px=0.94, bid_sz=10.0, ask_px=0.94, ask_sz=10.0,
            last_trade_ts_ns=0, last_l2_ts_ns=0,
        ),
        "NO": BookState(
            symbol="NO", bid_px=0.05, bid_sz=10.0, ask_px=0.06, ask_sz=10.0,
            last_trade_ts_ns=0, last_l2_ts_ns=0,
        ),
    }


def _cfg(**over) -> ThetaHarvesterConfig:
    base = dict(
        vol_lookback_seconds=3600,
        vol_sampling_dt_seconds=60,
        vol_clip_min=0.0,
        vol_clip_max=5.0,
        edge_buffer=0.05,
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


# ---------------------------------------------------------------------------
# Test 1: Default off is v3.5 bit-identical
# ---------------------------------------------------------------------------

def test_jr_default_off_is_v35_bit_identical() -> None:
    """momentum_mr_jr_trust_weight=False (default) must produce identical decisions
    to a config where the field is simply absent (i.e., the dataclass default)."""
    rets = tuple([0.0005 * ((i % 5) - 2) for i in range(90)])

    # Config A: explicitly set jr_trust_weight=False
    cfg_a = _cfg(
        momentum_mr_enabled=True,
        momentum_mr_indicator="z_ret",
        momentum_mr_lookback_min=30,
        momentum_mr_mode="tilt",
        momentum_mr_alpha_tilt=0.5,
        momentum_mr_jr_trust_weight=False,
    )
    # Config B: jr_trust_weight field absent → defaults to False
    cfg_b = _cfg(
        momentum_mr_enabled=True,
        momentum_mr_indicator="z_ret",
        momentum_mr_lookback_min=30,
        momentum_mr_mode="tilt",
        momentum_mr_alpha_tilt=0.5,
        # no momentum_mr_jr_trust_weight
    )
    strat_a = ThetaHarvesterStrategy(cfg_a)
    strat_b = ThetaHarvesterStrategy(cfg_b)

    dec_a = strat_a.evaluate(
        question=_qv(), books=_books_yes_fav(), reference_price=110.0,
        recent_returns=rets, recent_volume_usd=0.0, position=None,
        now_ns=_NOW_NS,
    )
    dec_b = strat_b.evaluate(
        question=_qv(), books=_books_yes_fav(), reference_price=110.0,
        recent_returns=rets, recent_volume_usd=0.0, position=None,
        now_ns=_NOW_NS,
    )

    assert dec_a.action == dec_b.action
    assert tuple(i.symbol for i in dec_a.intents) == tuple(i.symbol for i in dec_b.intents)
    assert tuple(i.size for i in dec_a.intents) == tuple(i.size for i in dec_b.intents)


# ---------------------------------------------------------------------------
# Test 2: Pure continuous returns → trust near 1.0
# ---------------------------------------------------------------------------

def test_jr_with_pure_continuous_returns_full_trust() -> None:
    """Smooth returns (no jumps) → BPV ≈ RV → JR ≈ 0 → trust ≈ 1.0.

    We feed a tilt-mode strategy with smooth returns and jr_trust_weight=True,
    then check the emitted diagnostic contains jr_trust ≥ 0.95.
    """
    # Smooth small positive returns — BPV will track RV closely
    smooth_rets = tuple([0.0001] * 120)

    cfg = _cfg(
        momentum_mr_enabled=True,
        momentum_mr_indicator="z_ret",
        momentum_mr_lookback_min=60,
        momentum_mr_mode="tilt",
        momentum_mr_alpha_tilt=1.0,
        momentum_mr_jr_trust_weight=True,
        edge_buffer=-1.0,  # disable so we reach the tilt block
    )
    strat = ThetaHarvesterStrategy(cfg)
    dec = strat.evaluate(
        question=_qv(), books=_books_yes_fav(), reference_price=110.0,
        recent_returns=smooth_rets, recent_volume_usd=0.0, position=None,
        now_ns=_NOW_NS,
    )

    # Find the tilt diagnostic
    tilt_diags = [d for d in dec.diagnostics if d.message == "momentum_mr_tilt"]
    assert tilt_diags, f"expected momentum_mr_tilt diagnostic, got: {dec.diagnostics}"
    tilt_diag = tilt_diags[0]

    # Extract jr_trust from fields pairs
    fields = dict(tilt_diag.fields)
    assert "jr_trust" in fields, f"jr_trust not in fields: {fields}"
    trust_val = float(fields["jr_trust"])
    assert trust_val >= 0.95, f"Expected trust ≥ 0.95 for smooth returns, got {trust_val:.4f}"

    # Also verify the helper directly
    direct_trust = _jr_trust_weight(smooth_rets, 60)
    assert direct_trust >= 0.95, f"Direct _jr_trust_weight returned {direct_trust:.4f}"


# ---------------------------------------------------------------------------
# Test 3: Jump-heavy returns → trust ≪ 1 → tilt less aggressive
# ---------------------------------------------------------------------------

def test_jr_with_jump_returns_shrinks_score() -> None:
    """One giant spike among zeros → JR near 1 → trust near 0 → tilt suppressed.

    We compare two tilt configs on the same jump-heavy returns:
    - jr_off: no JR weighting  → mm_score unchanged → stronger tilt
    - jr_on: JR weighting      → mm_score shrunk    → weaker tilt

    The tilt LOOSENS the edge_buffer when momentum is aligned (score > 0).
    With JR off: eff_edge_buffer = base * (1 - alpha * score_raw)
    With JR on:  eff_edge_buffer = base * (1 - alpha * score_raw * trust)  ← closer to base

    So with JR on the effective_edge_buffer should be CLOSER to base_edge_buffer
    (less tilt loosening) than with JR off.
    """
    # Mostly-zero returns with one giant spike at position 60 (aligned up with favorite YES)
    jump_rets = (0.0,) * 60 + (0.05,) + (0.0,) * 59

    # Use a modest edge_buffer and alpha_tilt so the difference is visible
    edge_buffer = 0.10
    alpha_tilt = 1.0

    cfg_jr_off = _cfg(
        momentum_mr_enabled=True,
        momentum_mr_indicator="z_ret",
        momentum_mr_lookback_min=60,
        momentum_mr_mode="tilt",
        momentum_mr_alpha_tilt=alpha_tilt,
        momentum_mr_jr_trust_weight=False,
        edge_buffer=edge_buffer,
    )
    cfg_jr_on = _cfg(
        momentum_mr_enabled=True,
        momentum_mr_indicator="z_ret",
        momentum_mr_lookback_min=60,
        momentum_mr_mode="tilt",
        momentum_mr_alpha_tilt=alpha_tilt,
        momentum_mr_jr_trust_weight=True,
        edge_buffer=edge_buffer,
    )

    strat_off = ThetaHarvesterStrategy(cfg_jr_off)
    strat_on = ThetaHarvesterStrategy(cfg_jr_on)

    books = _books_yes_fav()
    dec_off = strat_off.evaluate(
        question=_qv(), books=books, reference_price=110.0,
        recent_returns=jump_rets, recent_volume_usd=0.0, position=None,
        now_ns=_NOW_NS,
    )
    dec_on = strat_on.evaluate(
        question=_qv(), books=books, reference_price=110.0,
        recent_returns=jump_rets, recent_volume_usd=0.0, position=None,
        now_ns=_NOW_NS,
    )

    # Extract eff_edge_buffer from tilt diagnostics (if emitted)
    def _eff_eb(dec) -> float | None:
        for d in dec.diagnostics:
            if d.message == "momentum_mr_tilt":
                fields = dict(d.fields)
                if "eff_edge_buffer" in fields:
                    return float(fields["eff_edge_buffer"])
        return None

    eff_off = _eff_eb(dec_off)
    eff_on = _eff_eb(dec_on)

    # Both paths must reach the tilt diagnostic (not blocked earlier)
    assert eff_off is not None, f"No tilt diag (JR off) in: {dec_off.diagnostics}"
    assert eff_on is not None, f"No tilt diag (JR on) in: {dec_on.diagnostics}"

    # JR on diagnostic must report jr_trust < 0.5 (jump-heavy input)
    tilt_on = [d for d in dec_on.diagnostics if d.message == "momentum_mr_tilt"][0]
    fields_on = dict(tilt_on.fields)
    assert "jr_trust" in fields_on, f"jr_trust not in fields_on: {fields_on}"
    trust_val = float(fields_on["jr_trust"])
    assert trust_val < 0.5, f"Expected low trust for jump returns, got {trust_val:.4f}"

    # With JR on, effective_edge_buffer is closer to base (less tilt) than JR off.
    # Loosening tilt means eff_eb < base. JR on should loosen LESS than JR off.
    # i.e. eff_on should be >= eff_off (closer to base edge_buffer).
    assert eff_on >= eff_off, (
        f"JR on eff_edge_buffer ({eff_on:.5f}) should be >= JR off ({eff_off:.5f}) "
        f"because jump trust shrinks the tilt."
    )

    # Also verify the helper directly returns a low trust for this input
    direct_trust = _jr_trust_weight(jump_rets, 60)
    assert direct_trust < 0.5, f"Direct _jr_trust_weight returned {direct_trust:.4f}"
