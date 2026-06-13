# tests/unit/test_theta_doom_loop_gate.py
"""Tests for the bucket doom-loop fix (SHR-102).

Two new flag-gated behaviors, both OFF BY DEFAULT:
  (a) entry_spread_gate — skip entry when live half-spread (ask-bid)/2 of the
      chosen leg exceeds the net fair-value edge budget (fair/mid-referenced).
  (b) exit_spread_hold — suppress exit_safety_d + exit_edge liquidations when
      the held book's half-spread exceeds a configurable threshold.
      Stop-loss ALWAYS fires regardless of exit_spread_hold.

All flags default to their disabled values → new code is bit-identical to the
pre-SHR-102 baseline for any caller that doesn't set them.

Bucket fixture design
---------------------
QuestionView with klass="priceBucket" and priceThresholds="55000,85000" (2
thresholds → 3 YES/NO outcome pairs):
  @Y0 / @N0 : YES wins when BTC < 55000
  @Y1 / @N1 : YES wins when 55000 ≤ BTC < 85000
  @Y2 / @N2 : YES wins when BTC ≥ 85000

With reference_price=200,000 BTC, @Y2 is the dominant favorite (p_win ≈ 1.0).
The "wide book" is placed on @Y2 with bid=0.60, ask=0.95 (half-spread=0.175) —
mirroring the real live #1670 / #2280 bucket that drove the doom loop.
"""

from __future__ import annotations

import dataclasses

from hlanalysis.strategy.theta_harvester import ThetaHarvesterConfig, ThetaHarvesterStrategy
from hlanalysis.strategy.types import Action, BookState, Position, QuestionView


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bucket_question(*, expiry_ns: int = 10**18) -> QuestionView:
    """priceBucket with 2 thresholds [55000, 85000] → 3 YES/NO pairs.

    With reference_price=200,000, @Y2 wins (p_win ≈ 1.0).
    With reference_price=10,000,  @Y0 wins (p_win ≈ 1.0).
    """
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


def _binary_question(*, expiry_ns: int = 10**18) -> QuestionView:
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


def _pos(
    sym: str, *, qty: float = 100.0, entry_px: float = 0.95, stop_pct: float | None = None, question_idx: int = 42
) -> Position:
    sl_px = 0.0 if stop_pct is None else entry_px * (1.0 - stop_pct)
    return Position(
        question_idx=question_idx,
        symbol=sym,
        qty=qty,
        avg_entry=entry_px,
        stop_loss_price=sl_px,
        last_update_ts_ns=0,
    )


def _base_cfg(**overrides) -> ThetaHarvesterConfig:
    """Minimal ThetaHarvesterConfig with new doom-loop flags defaulting OFF."""
    kwargs: dict = dict(
        vol_lookback_seconds=3600,
        vol_sampling_dt_seconds=60,
        vol_clip_min=0.05,
        vol_clip_max=5.0,
        edge_buffer=0.02,
        fee_taker=0.0,
        half_spread_assumption=0.005,
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
        exit_safety_d=1.0,
    )
    kwargs.update(overrides)
    return ThetaHarvesterConfig(**kwargs)


def _strat(**overrides) -> ThetaHarvesterStrategy:
    return ThetaHarvesterStrategy(_base_cfg(**overrides))


# Shared realistic returns (low vol, 60s cadence)
_RETS = tuple([0.0001] * 120)

# Wide bucket book: spread = 0.95 - 0.60 = 0.35, half-spread = 0.175
# Mirrors the real #1670 / #2280 book width observed in the live doom loop.
_WIDE_BID = 0.60
_WIDE_ASK = 0.95
_WIDE_HALF_SPREAD = (_WIDE_ASK - _WIDE_BID) / 2.0  # 0.175

# Tight binary book: spread ≈ 0.005, half-spread ≈ 0.0025
_TIGHT_BID = 0.945
_TIGHT_ASK = 0.950


def _wide_bucket_books_entry() -> dict[str, BookState]:
    """Entry scenario books: @Y2 is the winner (BTC=200k > 85k) with wide book."""
    return {
        "@Y0": _book("@Y0", bid=0.01, ask=0.04),
        "@N0": _book("@N0", bid=0.01, ask=0.04),
        "@Y1": _book("@Y1", bid=0.01, ask=0.04),
        "@N1": _book("@N1", bid=0.01, ask=0.04),
        "@Y2": _book("@Y2", bid=_WIDE_BID, ask=_WIDE_ASK),  # wide
        "@N2": _book("@N2", bid=0.01, ask=0.40),
    }


def _tight_bucket_books_entry() -> dict[str, BookState]:
    """Entry scenario books: @Y2 is the winner with a TIGHT book."""
    return {
        "@Y0": _book("@Y0", bid=0.01, ask=0.04),
        "@N0": _book("@N0", bid=0.01, ask=0.04),
        "@Y1": _book("@Y1", bid=0.01, ask=0.04),
        "@N1": _book("@N1", bid=0.01, ask=0.04),
        "@Y2": _book("@Y2", bid=0.94, ask=0.96),  # tight: spread=0.02, half=0.01
        "@N2": _book("@N2", bid=0.01, ask=0.09),
    }


def _wide_held_books() -> dict[str, BookState]:
    """Exit scenario books: holding @Y2 with a wide book."""
    return {"@Y2": _book("@Y2", bid=_WIDE_BID, ask=_WIDE_ASK)}


def _tight_binary_books() -> dict[str, BookState]:
    return {
        "YES": _book("YES", bid=_TIGHT_BID, ask=_TIGHT_ASK),
        "NO": _book("NO", bid=0.05, ask=0.055),
    }


def _dec_action_and_msgs(dec: object) -> tuple[Action, list[str]]:
    """Helper: extract (action, diagnostic_message_list) from a Decision."""
    from hlanalysis.strategy.types import Decision

    assert isinstance(dec, Decision)
    return dec.action, [d.message for d in dec.diagnostics]


# ---------------------------------------------------------------------------
# (a) entry_spread_gate tests
# ---------------------------------------------------------------------------


class TestEntrySpreadGate:
    """entry_spread_gate=True: skip entry when live half-spread exceeds edge budget."""

    def test_gate_off_by_default_wide_book_still_enters(self) -> None:
        """With entry_spread_gate=False (default), a wide book still enters
        (existing behavior unchanged)."""
        strat = _strat(
            entry_spread_gate=False,
            favorite_threshold=0.0,
            edge_buffer=0.0,
            half_spread_assumption=0.0,
        )
        q = _bucket_question(expiry_ns=3600 * 10**9)
        # BTC=200k >> 85k → @Y2 p_win≈1.0; ask=0.95 → raw_edge≈0.05 → enters
        dec = strat.evaluate(
            question=q,
            books=_wide_bucket_books_entry(),
            reference_price=200_000.0,
            recent_returns=_RETS,
            recent_volume_usd=1000.0,
            position=None,
            now_ns=0,
        )
        action, msgs = _dec_action_and_msgs(dec)
        assert action == Action.ENTER, f"expected ENTER without gate; msgs={msgs}"

    def test_gate_fires_on_wide_bucket_book(self) -> None:
        """entry_spread_gate=True + wide bucket book → HOLD when spread > edge budget."""
        # With BTC=200k, @Y2 p_win≈1.0, ask=0.95, mid=0.775, fee=0
        # fair_edge = 1.0 - 0.775 = 0.225
        # edge_budget = fair_edge - edge_buffer = 0.225 - 0.20 = 0.025
        # live_half_spread = 0.175 >> 0.025 → gate fires
        strat = _strat(
            entry_spread_gate=True,
            favorite_threshold=0.0,
            edge_buffer=0.20,  # large buffer so budget < half-spread
            half_spread_assumption=0.0,
            fee_taker=0.0,
        )
        q = _bucket_question(expiry_ns=3600 * 10**9)
        dec = strat.evaluate(
            question=q,
            books=_wide_bucket_books_entry(),
            reference_price=200_000.0,
            recent_returns=_RETS,
            recent_volume_usd=1000.0,
            position=None,
            now_ns=0,
        )
        action, msgs = _dec_action_and_msgs(dec)
        assert action == Action.HOLD, f"expected HOLD with spread gate on wide book; msgs={msgs}"
        assert "entry_spread_too_wide" in msgs, f"expected 'entry_spread_too_wide' diagnostic; got {msgs}"

    def test_gate_does_not_fire_on_tight_binary_book(self) -> None:
        """entry_spread_gate=True + tight binary book → gate does NOT suppress entry."""
        # tight binary: half-spread=0.0025; fair_edge≈1.0-0.9475≈0.0525
        # edge_budget = 0.0525 - 0.02 = 0.0325 >> 0.0025 → gate DOES NOT fire
        strat = _strat(
            entry_spread_gate=True,
            favorite_threshold=0.0,
            edge_buffer=0.02,
            half_spread_assumption=0.0,
            fee_taker=0.0,
        )
        q = _binary_question(expiry_ns=3600 * 10**9)
        dec = strat.evaluate(
            question=q,
            books=_tight_binary_books(),
            reference_price=120_000.0,
            recent_returns=_RETS,
            recent_volume_usd=1000.0,
            position=None,
            now_ns=0,
        )
        action, msgs = _dec_action_and_msgs(dec)
        assert action == Action.ENTER, f"gate should not fire on tight binary; msgs={msgs}"

    def test_gate_inert_when_spread_smaller_than_edge_budget(self) -> None:
        """entry_spread_gate=True is inert when live half-spread < edge budget."""
        # @Y2 tight book: bid=0.94, ask=0.96, half-spread=0.01
        # p_win≈1.0, mid=0.95, fair_edge=1.0-0.95=0.05
        # edge_budget = 0.05 - 0.02 = 0.03 >> 0.01 → gate does NOT fire
        strat = _strat(
            entry_spread_gate=True,
            favorite_threshold=0.0,
            edge_buffer=0.02,
            half_spread_assumption=0.0,
            fee_taker=0.0,
        )
        q = _bucket_question(expiry_ns=3600 * 10**9)
        dec = strat.evaluate(
            question=q,
            books=_tight_bucket_books_entry(),
            reference_price=200_000.0,
            recent_returns=_RETS,
            recent_volume_usd=1000.0,
            position=None,
            now_ns=0,
        )
        action, msgs = _dec_action_and_msgs(dec)
        assert action == Action.ENTER, f"gate should be inert when spread < budget; msgs={msgs}"


# ---------------------------------------------------------------------------
# (b) exit_spread_hold tests
# ---------------------------------------------------------------------------


class TestExitSpreadHold:
    """exit_spread_hold > 0: suppress exit_safety_d + exit_edge on wide held book."""

    def test_exit_spread_hold_off_by_default_safety_d_fires(self) -> None:
        """With exit_spread_hold=0.0 (default), exit_safety_d fires normally."""
        strat = _strat(
            exit_safety_d=1.0,
            exit_spread_hold=0.0,  # disabled
        )
        q = _bucket_question(expiry_ns=3600 * 10**9)
        pos = _pos("@Y2", entry_px=0.95)
        # BTC=10k far below all thresholds → @Y2 safety_d very negative → EXIT
        dec = strat.evaluate(
            question=q,
            books=_wide_held_books(),
            reference_price=10_000.0,
            recent_returns=_RETS,
            recent_volume_usd=1000.0,
            position=pos,
            now_ns=0,
        )
        action, msgs = _dec_action_and_msgs(dec)
        assert action == Action.EXIT, f"exit_safety_d should fire when hold=0.0; msgs={msgs}"

    def test_exit_spread_hold_suppresses_exit_safety_d_on_wide_book(self) -> None:
        """exit_spread_hold active: suppress exit_safety_d (hold to settle)."""
        # half-spread=0.175 > threshold=0.10 → suppress
        strat = _strat(
            exit_safety_d=1.0,
            exit_spread_hold=0.10,
        )
        q = _bucket_question(expiry_ns=3600 * 10**9)
        pos = _pos("@Y2", entry_px=0.95)
        # BTC=10k → safety_d very negative; but book is wide → suppressed
        dec = strat.evaluate(
            question=q,
            books=_wide_held_books(),
            reference_price=10_000.0,
            recent_returns=_RETS,
            recent_volume_usd=1000.0,
            position=pos,
            now_ns=0,
        )
        action, msgs = _dec_action_and_msgs(dec)
        assert action == Action.HOLD, f"exit_spread_hold should suppress exit_safety_d; msgs={msgs}"
        assert "hold_spread_too_wide" in msgs, f"expected 'hold_spread_too_wide' diagnostic; got {msgs}"

    def test_exit_spread_hold_suppresses_exit_edge_on_wide_book(self) -> None:
        """exit_spread_hold active: suppress exit_edge (not just safety_d)."""
        # safety_d disabled; exit_edge fires when edge_held < -0.01
        # BTC=10k → @Y2 p_win≈0; held.bid=0.60 → edge_held = 0-0.60 ≈ -0.60 < -0.01
        # → exit_edge would fire, but wide book (half-spread=0.175 > 0.10) suppresses
        strat = _strat(
            exit_safety_d=0.0,
            exit_spread_hold=0.10,
            exit_edge_threshold=-0.01,
        )
        q = _bucket_question(expiry_ns=3600 * 10**9)
        pos = _pos("@Y2", entry_px=0.95)
        dec = strat.evaluate(
            question=q,
            books=_wide_held_books(),
            reference_price=10_000.0,
            recent_returns=_RETS,
            recent_volume_usd=1000.0,
            position=pos,
            now_ns=0,
        )
        action, msgs = _dec_action_and_msgs(dec)
        assert action == Action.HOLD, f"exit_spread_hold should suppress exit_edge; msgs={msgs}"

    def test_stop_loss_still_fires_when_exit_spread_hold_on(self) -> None:
        """Stop-loss MUST fire even with exit_spread_hold active (invariant)."""
        # entry_px=0.95, stop=0.95*0.90=0.855; bid=0.60 < 0.855 → STOP FIRES
        strat = _strat(
            stop_loss_pct=0.10,
            exit_safety_d=1.0,
            exit_spread_hold=0.10,
        )
        q = _bucket_question(expiry_ns=3600 * 10**9)
        pos = _pos("@Y2", entry_px=0.95, stop_pct=0.10)
        dec = strat.evaluate(
            question=q,
            books=_wide_held_books(),
            reference_price=10_000.0,
            recent_returns=_RETS,
            recent_volume_usd=1000.0,
            position=pos,
            now_ns=0,
        )
        action, msgs = _dec_action_and_msgs(dec)
        assert action == Action.EXIT, f"stop-loss must fire regardless of exit_spread_hold; msgs={msgs}"
        assert "exit_stop_loss" in msgs, f"expected 'exit_stop_loss'; got {msgs}"

    def test_exit_spread_hold_does_not_suppress_when_book_is_tight(self) -> None:
        """exit_spread_hold must NOT suppress exit_safety_d when book is tight."""
        # Tight binary: half-spread=0.0025 < threshold=0.10 → no suppression
        # BTC=50k < strike=100k → safety_d negative → EXIT fires normally
        strat = _strat(
            exit_safety_d=1.0,
            exit_spread_hold=0.10,
        )
        q = _binary_question(expiry_ns=3600 * 10**9)
        pos = _pos("YES", entry_px=0.95, question_idx=7)
        dec = strat.evaluate(
            question=q,
            books=_tight_binary_books(),
            reference_price=50_000.0,
            recent_returns=_RETS,
            recent_volume_usd=1000.0,
            position=pos,
            now_ns=0,
        )
        action, msgs = _dec_action_and_msgs(dec)
        assert action == Action.EXIT, f"exit_safety_d should fire on tight book (spread below threshold); msgs={msgs}"


# ---------------------------------------------------------------------------
# Bit-identical to pre-SHR-102 baseline when flags are at defaults
# ---------------------------------------------------------------------------


class TestDefaultBitIdentical:
    """When new fields are at their disabled defaults, the Decision must be
    identical to a pre-SHR-102 config (same action + same diagnostic messages).

    We compare action + diagnostic messages (not cloid, which is random UUID)
    so the test is stable.
    """

    def _msgs(
        self,
        strat: ThetaHarvesterStrategy,
        *,
        question: QuestionView,
        books: dict[str, BookState],
        ref: float,
        position: Position | None,
    ) -> tuple[Action, list[str]]:
        dec = strat.evaluate(
            question=question,
            books=books,
            reference_price=ref,
            recent_returns=_RETS,
            recent_volume_usd=1000.0,
            position=position,
            now_ns=0,
        )
        return _dec_action_and_msgs(dec)

    def test_entry_binary_bit_identical_with_new_defaults(self) -> None:
        """Binary entry: new fields at disabled defaults → same action + diags."""
        cfg_no_new = _base_cfg(favorite_threshold=0.0, edge_buffer=0.0)
        cfg_with_defaults = _base_cfg(
            favorite_threshold=0.0,
            edge_buffer=0.0,
            entry_spread_gate=False,
            exit_spread_hold=0.0,
        )
        q = _binary_question(expiry_ns=3600 * 10**9)
        books = _tight_binary_books()
        a1, m1 = self._msgs(ThetaHarvesterStrategy(cfg_no_new), question=q, books=books, ref=120_000.0, position=None)
        a2, m2 = self._msgs(
            ThetaHarvesterStrategy(cfg_with_defaults), question=q, books=books, ref=120_000.0, position=None
        )
        assert a1 == a2, f"action mismatch: {a1} vs {a2}"
        assert m1 == m2, f"diags mismatch: {m1} vs {m2}"

    def test_bucket_exit_bit_identical_with_new_defaults(self) -> None:
        """Bucket exit with wide book: new fields at defaults → same action + diags."""
        cfg_no_new = _base_cfg(exit_safety_d=0.0, exit_edge_threshold=-0.01, favorite_threshold=0.0)
        cfg_with_defaults = _base_cfg(
            exit_safety_d=0.0,
            exit_edge_threshold=-0.01,
            favorite_threshold=0.0,
            entry_spread_gate=False,
            exit_spread_hold=0.0,
        )
        q = _bucket_question(expiry_ns=3600 * 10**9)
        pos = _pos("@Y2", entry_px=0.95)
        # BTC=200k → @Y2 p_win≈1.0; bid=0.60; edge_held=1.0-0.60≈0.40 > -0.01 → HOLD
        a1, m1 = self._msgs(
            ThetaHarvesterStrategy(cfg_no_new), question=q, books=_wide_held_books(), ref=200_000.0, position=pos
        )
        a2, m2 = self._msgs(
            ThetaHarvesterStrategy(cfg_with_defaults), question=q, books=_wide_held_books(), ref=200_000.0, position=pos
        )
        assert a1 == a2, f"action mismatch: {a1} vs {a2}"
        assert m1 == m2, f"diags mismatch: {m1} vs {m2}"

    def test_bucket_entry_bit_identical_with_new_defaults(self) -> None:
        """Bucket entry with wide book at defaults → same action + diags as before."""
        cfg_no_new = _base_cfg(favorite_threshold=0.0, edge_buffer=0.0)
        cfg_with_defaults = _base_cfg(
            favorite_threshold=0.0,
            edge_buffer=0.0,
            entry_spread_gate=False,
            exit_spread_hold=0.0,
        )
        q = _bucket_question(expiry_ns=3600 * 10**9)
        a1, m1 = self._msgs(
            ThetaHarvesterStrategy(cfg_no_new),
            question=q,
            books=_wide_bucket_books_entry(),
            ref=200_000.0,
            position=None,
        )
        a2, m2 = self._msgs(
            ThetaHarvesterStrategy(cfg_with_defaults),
            question=q,
            books=_wide_bucket_books_entry(),
            ref=200_000.0,
            position=None,
        )
        assert a1 == a2, f"action mismatch: {a1} vs {a2}"
        assert m1 == m2, f"diags mismatch: {m1} vs {m2}"
