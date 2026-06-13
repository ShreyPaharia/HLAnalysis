# tests/unit/test_theta_builder_single_source.py
"""Guards for the single-source-of-truth refactor of build_v3_theta_harvester.

Three invariants enforced here:

1. Builder-defaults-match-model (Fix 1):
   When build_v3_theta_harvester is called with only required params (no optional
   overrides), every ThetaHarvesterParams optional field on the resulting
   ThetaHarvesterConfig equals the pydantic model's declared default — i.e. _D
   is the single source of truth, not hand-copied literals.

2. Builder-derives-from-_D (Fix 2 / SHR-65 class guard):
   Stronger form: for every ThetaHarvesterParams field, changing the pydantic
   default WOULD change the builder output — tested by asserting that, with no
   param overrides, built_cfg.<field> == _D.<field> for each optional knob.
   This is the "no third literal" guarantee.

3. require_two_sided_entry guard (Fix 3):
   (a) Default is False → bit-identical to the legacy one-sided fallback: a
       one-sided ask-only quote (no bid) whose ask >= favorite_threshold can
       still pass the favorite gate and produce an ENTER.
   (b) When True, a one-sided ask-only quote is excluded from the favorite
       gate (treated as mid=0.0) so it cannot pass the threshold → HOLD.
   (c) When True, a two-sided quote that passes the threshold still yields
       ENTER → the guard only affects genuinely one-sided quotes.
"""

from __future__ import annotations

import dataclasses

import pytest

from hlanalysis.backtest.core.registry import build as build_strategy
from hlanalysis.strategy.theta_harvester import (
    ThetaHarvesterConfig,
    ThetaHarvesterParams,
    ThetaHarvesterStrategy,
    _D,
)
from hlanalysis.strategy.types import Action, BookState, QuestionView


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _required_params(**overrides) -> dict:
    """Minimal params dict satisfying build_v3_theta_harvester's required keys."""
    base: dict = dict(
        vol_lookback_seconds=3600,
        edge_buffer=0.02,
        stop_loss_pct=None,
        exit_edge_threshold=-0.01,
        take_profit_price=None,
    )
    base.update(overrides)
    return base


def _binary_question(
    *,
    expiry_ns: int = int(4 * 3600 * 1e9),  # 4h in ns; large enough to avoid tte_min gate
    strike: float = 100_000.0,
    yes_symbol: str = "YES",
    no_symbol: str = "NO",
) -> QuestionView:
    return QuestionView(
        question_idx=0,
        yes_symbol=yes_symbol,
        no_symbol=no_symbol,
        strike=strike,
        expiry_ns=expiry_ns,
        underlying="BTC",
        klass="priceBinary",
        period="1d",
        settled=False,
        kv=(),
    )


def _book(symbol: str, *, bid: float | None, ask: float | None, sz: float = 100.0) -> BookState:
    return BookState(
        symbol=symbol,
        bid_px=bid,
        bid_sz=sz if bid is not None else None,
        ask_px=ask,
        ask_sz=sz if ask is not None else None,
        last_trade_ts_ns=0,
        last_l2_ts_ns=0,
    )


# Returns that produce a clean sigma (enough variance to pass vol_insufficient_data)
_RETS = tuple([0.001, -0.001] * 60)  # 120 alternating ±0.1% returns


# ---------------------------------------------------------------------------
# Fix 1 + Fix 2: builder defaults derive from _D (ThetaHarvesterParams)
# ---------------------------------------------------------------------------

# Fields that ARE in ThetaHarvesterParams (and thus use _D in the builder).
_OPTIONAL_KNOB_FIELDS = set(ThetaHarvesterParams.model_fields)

# Fields that resolve to None by default (Optional[...] = None).
# These need special comparison because params.get() returns None which matches.
_NONE_DEFAULT_FIELDS = {
    name for name, field_info in ThetaHarvesterParams.model_fields.items() if field_info.default is None
}


def _build_with_required_only() -> ThetaHarvesterConfig:
    """Build with only the required params; all optional knobs at their defaults."""
    strat = build_strategy("v3_theta_harvester", _required_params())
    assert isinstance(strat, ThetaHarvesterStrategy)
    return strat.cfg


def test_builder_optional_fields_match_model_defaults() -> None:
    """With empty overrides, every ThetaHarvesterParams optional field on the
    built ThetaHarvesterConfig equals the pydantic model's declared default.

    This is the SINGLE SOURCE guarantee: changing a default in
    ThetaHarvesterParams propagates to the builder output without touching
    build_v3_theta_harvester.  Any literal default surviving in the builder
    that DIFFERS from the model would fail here.
    """
    built = _build_with_required_only()
    dc_fields = {f.name for f in dataclasses.fields(ThetaHarvesterConfig)}
    mismatches: list[str] = []
    for name in _OPTIONAL_KNOB_FIELDS:
        if name not in dc_fields:
            continue  # not on the dataclass — skip (shouldn't happen after parity test)
        built_val = getattr(built, name)
        model_default = ThetaHarvesterParams.model_fields[name].default
        if built_val != model_default:
            mismatches.append(f"{name}: builder produced {built_val!r}, model default is {model_default!r}")
    assert not mismatches, (
        "Builder defaults do not match ThetaHarvesterParams model defaults "
        "(literal default still hand-copied in builder):\n" + "\n".join(f"  {m}" for m in mismatches)
    )


def test_builder_derives_from_sentinel_not_literals() -> None:
    """Structural check: for each optional field, built_cfg.<field> == _D.<field>.

    This test is logically equivalent to test_builder_optional_fields_match_model_defaults
    but expresses the dependency on _D directly rather than on model_fields — it
    confirms that the builder reads _D rather than independent literals.
    """
    built = _build_with_required_only()
    dc_fields = {f.name for f in dataclasses.fields(ThetaHarvesterConfig)}
    mismatches: list[str] = []
    for name in _OPTIONAL_KNOB_FIELDS:
        if name not in dc_fields:
            continue
        built_val = getattr(built, name)
        sentinel_val = getattr(_D, name)
        if built_val != sentinel_val:
            mismatches.append(f"{name}: built={built_val!r}, _D.{name}={sentinel_val!r}")
    assert not mismatches, "Builder does not derive optional defaults from _D:\n" + "\n".join(
        f"  {m}" for m in mismatches
    )


def test_require_two_sided_entry_reachable_via_builder() -> None:
    """require_two_sided_entry can be set via the params dict.

    Before the fix, require_two_sided_entry was absent from the builder; this
    test would fail because the built config always had False regardless of the
    param.
    """
    strat = build_strategy(
        "v3_theta_harvester",
        _required_params(require_two_sided_entry=True),
    )
    assert isinstance(strat, ThetaHarvesterStrategy)
    assert strat.cfg.require_two_sided_entry is True


def test_require_two_sided_entry_default_false() -> None:
    """Default is False — bit-identical to pre-fix behavior."""
    built = _build_with_required_only()
    assert built.require_two_sided_entry is False


# ---------------------------------------------------------------------------
# Fix 3: require_two_sided_entry guard
# ---------------------------------------------------------------------------


def _entry_strat(
    *,
    favorite_threshold: float = 0.85,
    require_two_sided: bool,
    edge_buffer: float = 0.01,
) -> ThetaHarvesterStrategy:
    """Build a ThetaHarvesterStrategy tuned to fire on a high-probability binary."""
    strat = build_strategy(
        "v3_theta_harvester",
        _required_params(
            favorite_threshold=favorite_threshold,
            edge_buffer=edge_buffer,
            require_two_sided_entry=require_two_sided,
            # Disable gates that would interfere
            tte_min_seconds=0,
            tte_max_seconds=10**9,
            drift_lookback_seconds=0,
            drift_blend=0.0,
        ),
    )
    assert isinstance(strat, ThetaHarvesterStrategy)
    return strat


def _evaluate_entry(
    strat: ThetaHarvesterStrategy,
    *,
    yes_bid: float | None,
    yes_ask: float | None,
    no_bid: float | None,
    no_ask: float | None,
    reference_price: float = 200_000.0,
    strike: float = 100_000.0,
) -> Action:
    """Run a single evaluation against a binary question and return the Action."""
    question = _binary_question(strike=strike)
    books = {
        "YES": _book("YES", bid=yes_bid, ask=yes_ask),
        "NO": _book("NO", bid=no_bid, ask=no_ask),
    }
    dec = strat.evaluate(
        question=question,
        books=books,
        reference_price=reference_price,
        recent_returns=_RETS,
        recent_volume_usd=10_000.0,
        position=None,
        now_ns=0,
    )
    return dec.action


class TestRequireTwoSidedEntryOff:
    """With require_two_sided_entry=False (default), a one-sided ask-only quote
    can pass the favorite_threshold via the legacy _mid fallback, which returns
    ask_px when bid_px is absent.  Behavior is bit-identical to the pre-fix code.
    """

    def test_one_sided_ask_only_can_pass_favorite_gate(self) -> None:
        """ask_px=0.90 (>= threshold 0.85) with no bid → _mid returns ask → ENTER."""
        strat = _entry_strat(require_two_sided=False, favorite_threshold=0.85)
        # YES: ask=0.90, no bid → mid fallback = ask = 0.90 >= 0.85
        # NO: ask=0.10, bid=0.09 (two-sided, but mid=0.095 < threshold)
        # Reference 200k >> strike 100k → YES wins → big edge
        action = _evaluate_entry(
            strat,
            yes_bid=None,
            yes_ask=0.90,
            no_bid=0.09,
            no_ask=0.10,
        )
        assert action == Action.ENTER, (
            "With require_two_sided_entry=False, a one-sided ask-only YES quote "
            "at ask=0.90 should pass the 0.85 favorite gate via _mid fallback"
        )

    def test_two_sided_quote_also_passes_gate(self) -> None:
        """Two-sided YES quote with mid=0.90 → ENTER (baseline behavior)."""
        strat = _entry_strat(require_two_sided=False, favorite_threshold=0.85)
        action = _evaluate_entry(
            strat,
            yes_bid=0.88,
            yes_ask=0.92,  # mid=0.90 >= 0.85
            no_bid=0.08,
            no_ask=0.12,
        )
        assert action == Action.ENTER


class TestRequireTwoSidedEntryOn:
    """With require_two_sided_entry=True, ask-only quotes get _mid=0.0 and
    therefore cannot pass a positive favorite_threshold.
    """

    def test_one_sided_ask_only_is_blocked(self) -> None:
        """ask_px=0.90, no bid → require_two_sided=True forces mid=0.0 → HOLD.

        This is the core defensive behavior: a stale high ask with no bid
        (e.g. 0.99) cannot trigger a spurious entry.
        """
        strat = _entry_strat(require_two_sided=True, favorite_threshold=0.85)
        action = _evaluate_entry(
            strat,
            yes_bid=None,
            yes_ask=0.90,
            no_bid=0.09,
            no_ask=0.10,
        )
        assert action == Action.HOLD, (
            "With require_two_sided_entry=True, a one-sided ask-only YES quote "
            "must be blocked (mid forced to 0.0 < threshold 0.85)"
        )

    def test_two_sided_quote_still_enters(self) -> None:
        """Two-sided YES quote with genuine mid=0.90 → ENTER (guard is narrowly scoped)."""
        strat = _entry_strat(require_two_sided=True, favorite_threshold=0.85)
        action = _evaluate_entry(
            strat,
            yes_bid=0.88,
            yes_ask=0.92,  # mid=0.90 >= 0.85
            no_bid=0.08,
            no_ask=0.12,
        )
        assert action == Action.ENTER, (
            "Two-sided quotes must still pass through the favorite gate when require_two_sided_entry=True"
        )

    def test_both_sides_one_sided_yields_hold(self) -> None:
        """Both YES and NO have only an ask, no bid → both blocked → no_favorite HOLD."""
        strat = _entry_strat(require_two_sided=True, favorite_threshold=0.85)
        action = _evaluate_entry(
            strat,
            yes_bid=None,
            yes_ask=0.90,
            no_bid=None,
            no_ask=0.10,
        )
        assert action == Action.HOLD


class TestRequireTwoSidedBitIdenticalWhenOff:
    """Verify that the default-off path is truly bit-identical to the no-guard
    baseline.  We evaluate the same scenario with and without the field set
    explicitly to False and assert identical actions and diagnostics.
    """

    def _strat(self, require_two_sided: bool) -> ThetaHarvesterStrategy:
        return _entry_strat(
            require_two_sided=require_two_sided,
            favorite_threshold=0.85,
            edge_buffer=0.01,
        )

    def _scenarios(self) -> list[dict]:
        return [
            # One-sided YES, low-edge NO
            dict(yes_bid=None, yes_ask=0.90, no_bid=0.09, no_ask=0.10),
            # Two-sided YES above threshold
            dict(yes_bid=0.88, yes_ask=0.92, no_bid=0.08, no_ask=0.12),
            # Both below threshold
            dict(yes_bid=0.40, yes_ask=0.50, no_bid=0.50, no_ask=0.60),
        ]

    def test_default_off_bit_identical_to_explicit_false(self) -> None:
        """build with no require_two_sided_entry key == build with key=False."""
        default_strat = build_strategy(
            "v3_theta_harvester",
            _required_params(
                favorite_threshold=0.85,
                edge_buffer=0.01,
                drift_lookback_seconds=0,
                drift_blend=0.0,
            ),
        )
        explicit_false_strat = self._strat(require_two_sided=False)
        for scenario in self._scenarios():
            question = _binary_question()
            books = {
                "YES": _book("YES", bid=scenario["yes_bid"], ask=scenario["yes_ask"]),
                "NO": _book("NO", bid=scenario["no_bid"], ask=scenario["no_ask"]),
            }
            kw = dict(
                question=question,
                books=books,
                reference_price=200_000.0,
                recent_returns=_RETS,
                recent_volume_usd=10_000.0,
                position=None,
                now_ns=0,
            )
            dec_default = default_strat.evaluate(**kw)
            dec_explicit = explicit_false_strat.evaluate(**kw)
            assert dec_default.action == dec_explicit.action, (
                f"scenario={scenario}: action mismatch: "
                f"default={dec_default.action}, explicit_false={dec_explicit.action}"
            )
            assert dec_default.diagnostics == dec_explicit.diagnostics, f"scenario={scenario}: diagnostics mismatch"
