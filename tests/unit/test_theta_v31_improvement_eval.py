# tests/unit/test_theta_v31_improvement_eval.py
"""Tests for the v31-improvement-eval flag-gated knobs (2026-06-13 desk study).

Three new behaviors, all OFF BY DEFAULT (bit-identical when disabled):
  (1) favorite_max          — upper-bound the favorite band (Card E FLB).
  (2) vol_regime_sizing      — scale the clip by realized-σ regime (Card F).
  (3) leadlag_veto_k         — veto entry on a sharp adverse reference jump (Card C).

Fixtures mirror tests/unit/test_theta_doom_loop_gate.py.
"""

from __future__ import annotations

from hlanalysis.strategy.theta_harvester import ThetaHarvesterConfig, ThetaHarvesterStrategy
from hlanalysis.strategy.types import Action, BookState, Decision, QuestionView

_ANNUAL_SECONDS = 365.0 * 24.0 * 3600.0


def _binary_question(*, expiry_ns: int = 3600 * 10**9) -> QuestionView:
    return QuestionView(
        question_idx=7,
        yes_symbol="YES",
        no_symbol="NO",
        strike=100_000.0,
        expiry_ns=expiry_ns,
        underlying="BTC",
        klass="priceBinary",
        period="1d",
        settled=False,
        leg_symbols=("YES", "NO"),
        kv=(),
    )


def _bucket_question(*, expiry_ns: int = 3600 * 10**9) -> QuestionView:
    return QuestionView(
        question_idx=42,
        yes_symbol="",
        no_symbol="",
        strike=0.0,
        expiry_ns=expiry_ns,
        underlying="BTC",
        klass="priceBucket",
        period="1d",
        settled=False,
        leg_symbols=("@Y0", "@N0", "@Y1", "@N1", "@Y2", "@N2"),
        kv=(("priceThresholds", "55000,85000"),),
    )


def _book(sym: str, *, bid: float, ask: float, sz: float = 100.0) -> BookState:
    return BookState(symbol=sym, bid_px=bid, bid_sz=sz, ask_px=ask, ask_sz=sz, last_trade_ts_ns=0, last_l2_ts_ns=0)


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
        exit_edge_threshold=-0.01,
        take_profit_price=None,
        time_stop_seconds=0,
        exit_safety_d=0.0,
    )
    kwargs.update(overrides)
    return ThetaHarvesterConfig(**kwargs)


def _strat(**overrides) -> ThetaHarvesterStrategy:
    return ThetaHarvesterStrategy(_base_cfg(**overrides))


# Low-vol returns (60s cadence) → small σ, p_win≈1 for a deep favorite far from strike
_RETS = tuple([0.0001] * 120)


def _eval(strat, q, books, ref, *, rets=_RETS, position=None):
    return strat.evaluate(
        question=q,
        books=books,
        reference_price=ref,
        recent_returns=rets,
        recent_volume_usd=1000.0,
        position=position,
        now_ns=0,
    )


def _msgs(dec: Decision) -> list[str]:
    return [d.message for d in dec.diagnostics]


# A deep-favorite binary book: YES mid ≈ 0.97 (above a 0.95 cap), still cheap vs p_win≈1.
def _deep_fav_binary_books() -> dict[str, BookState]:
    return {"YES": _book("YES", bid=0.965, ask=0.975), "NO": _book("NO", bid=0.025, ask=0.035)}


# A moderate-favorite binary book: YES mid ≈ 0.88 (inside [0.85,0.95]).
def _mod_fav_binary_books() -> dict[str, BookState]:
    return {"YES": _book("YES", bid=0.875, ask=0.885), "NO": _book("NO", bid=0.115, ask=0.125)}


# ---------------------------------------------------------------------------
# (1) favorite_max
# ---------------------------------------------------------------------------


class TestFavoriteMax:
    def test_deep_favorite_enters_without_cap(self) -> None:
        # No cap (default None): a deep favorite (mid≈0.97) still enters.
        strat = _strat(favorite_max=None)
        dec = _eval(strat, _binary_question(), _deep_fav_binary_books(), 130_000.0)
        assert dec.action == Action.ENTER, f"deep favorite should enter without cap; {_msgs(dec)}"

    def test_deep_favorite_vetoed_by_cap(self) -> None:
        # favorite_max=0.95: a deep favorite (mid≈0.97) is excluded → no_favorite.
        strat = _strat(favorite_max=0.95)
        dec = _eval(strat, _binary_question(), _deep_fav_binary_books(), 130_000.0)
        assert dec.action == Action.HOLD
        assert "no_favorite" in _msgs(dec)

    def test_moderate_favorite_still_enters_under_cap(self) -> None:
        # favorite_max=0.95: a 0.88-mid favorite is inside the band → enters.
        strat = _strat(favorite_max=0.95)
        dec = _eval(strat, _binary_question(), _mod_fav_binary_books(), 130_000.0)
        assert dec.action == Action.ENTER, f"in-band favorite should enter; {_msgs(dec)}"

    def test_cap_applies_to_bucket(self) -> None:
        # Bucket @Y2 deep favorite mid≈0.97 with BTC=200k → capped out at 0.95.
        books = {
            "@Y0": _book("@Y0", bid=0.005, ask=0.02),
            "@N0": _book("@N0", bid=0.005, ask=0.02),
            "@Y1": _book("@Y1", bid=0.005, ask=0.02),
            "@N1": _book("@N1", bid=0.005, ask=0.02),
            "@Y2": _book("@Y2", bid=0.965, ask=0.975),
            "@N2": _book("@N2", bid=0.025, ask=0.035),
        }
        capped = _eval(_strat(favorite_max=0.95), _bucket_question(), books, 200_000.0)
        assert capped.action == Action.HOLD and "no_favorite" in _msgs(capped)
        uncapped = _eval(_strat(favorite_max=None), _bucket_question(), books, 200_000.0)
        assert uncapped.action == Action.ENTER

    def test_default_none_bit_identical(self) -> None:
        q, books, ref = _binary_question(), _deep_fav_binary_books(), 130_000.0
        a = _eval(_strat(), q, books, ref)
        b = _eval(_strat(favorite_max=None), q, books, ref)
        assert (a.action, _msgs(a)) == (b.action, _msgs(b))


# ---------------------------------------------------------------------------
# (2) vol_regime_sizing
# ---------------------------------------------------------------------------


def _size_of(dec: Decision) -> float:
    assert dec.action == Action.ENTER
    return float(dec.intents[0].size)


class TestVolRegimeSizing:
    def test_disabled_is_full_clip(self) -> None:
        strat = _strat(vol_regime_sizing=False, max_position_usd=100.0)
        dec = _eval(strat, _binary_question(), _mod_fav_binary_books(), 130_000.0)
        base_size = _size_of(dec)
        # Enabled with both mults 1.0 → identical size.
        strat2 = _strat(
            vol_regime_sizing=True, vol_regime_sigma_threshold=0.01, vol_regime_low_mult=1.0, vol_regime_high_mult=1.0
        )
        dec2 = _eval(strat2, _binary_question(), _mod_fav_binary_books(), 130_000.0)
        assert _size_of(dec2) == base_size

    def test_low_regime_shrinks_clip(self) -> None:
        # σ from _RETS is small; set a high threshold so we are in the LOW regime.
        strat = _strat(
            vol_regime_sizing=True,
            vol_regime_sigma_threshold=5.0,  # σ << 5.0 → low regime
            vol_regime_low_mult=0.5,
            vol_regime_high_mult=1.0,
            max_position_usd=100.0,
        )
        full = _eval(_strat(max_position_usd=100.0), _binary_question(), _mod_fav_binary_books(), 130_000.0)
        scaled = _eval(strat, _binary_question(), _mod_fav_binary_books(), 130_000.0)
        assert _size_of(scaled) < _size_of(full), "low-σ regime should shrink the clip"

    def test_high_regime_full_clip(self) -> None:
        # Threshold below σ → HIGH regime → high_mult (1.0) → full clip.
        strat = _strat(
            vol_regime_sizing=True,
            vol_regime_sigma_threshold=0.0,  # σ >= 0 always → high regime
            vol_regime_low_mult=0.5,
            vol_regime_high_mult=1.0,
            max_position_usd=100.0,
        )
        full = _eval(_strat(max_position_usd=100.0), _binary_question(), _mod_fav_binary_books(), 130_000.0)
        scaled = _eval(strat, _binary_question(), _mod_fav_binary_books(), 130_000.0)
        assert _size_of(scaled) == _size_of(full)


# ---------------------------------------------------------------------------
# (3) leadlag_veto_k
# ---------------------------------------------------------------------------


class TestLeadLagVeto:
    def test_no_veto_when_disabled(self) -> None:
        # A big adverse last return but veto disabled (None) → still enters.
        rets = tuple([0.0001] * 119 + [-0.02])  # sharp down move last bar
        strat = _strat(leadlag_veto_k=None)
        dec = _eval(strat, _binary_question(), _mod_fav_binary_books(), 130_000.0, rets=rets)
        assert dec.action == Action.ENTER

    def test_adverse_jump_vetoes_yes_favorite(self) -> None:
        # YES favorite + sharp DOWN last return → adverse → veto.
        rets = tuple([0.0] * 119 + [-0.02])
        strat = _strat(leadlag_veto_k=3.0)
        dec = _eval(strat, _binary_question(), _mod_fav_binary_books(), 130_000.0, rets=rets)
        assert dec.action == Action.HOLD, f"adverse jump should veto; {_msgs(dec)}"
        assert "leadlag_veto" in _msgs(dec)

    def test_favorable_jump_does_not_veto(self) -> None:
        # YES favorite + sharp UP last return → favorable → NOT adverse → enters.
        rets = tuple([0.0] * 119 + [0.02])
        strat = _strat(leadlag_veto_k=3.0)
        dec = _eval(strat, _binary_question(), _mod_fav_binary_books(), 130_000.0, rets=rets)
        assert dec.action == Action.ENTER, f"favorable jump should not veto; {_msgs(dec)}"

    def test_small_move_does_not_veto(self) -> None:
        # Sub-threshold adverse move → no veto. σ floors at clip_min=0.05 ann →
        # σ_per_sample≈6.9e-5 at dt=60; -1e-4 is ≈1.4σ (< k=3) → enters.
        rets = tuple([0.0] * 119 + [-0.0001])
        strat = _strat(leadlag_veto_k=3.0)
        dec = _eval(strat, _binary_question(), _mod_fav_binary_books(), 130_000.0, rets=rets)
        assert dec.action == Action.ENTER

    def test_bucket_vetoes_on_any_sharp_move(self) -> None:
        # Bucket favorite: any jump-sized move (either sign) vetoes.
        books = {
            "@Y0": _book("@Y0", bid=0.005, ask=0.02),
            "@N0": _book("@N0", bid=0.005, ask=0.02),
            "@Y1": _book("@Y1", bid=0.005, ask=0.02),
            "@N1": _book("@N1", bid=0.005, ask=0.02),
            "@Y2": _book("@Y2", bid=0.875, ask=0.885),
            "@N2": _book("@N2", bid=0.115, ask=0.125),
        }
        rets_up = tuple([0.0] * 119 + [0.02])
        dec = _eval(_strat(leadlag_veto_k=3.0), _bucket_question(), books, 200_000.0, rets=rets_up)
        assert dec.action == Action.HOLD and "leadlag_veto" in _msgs(dec)
