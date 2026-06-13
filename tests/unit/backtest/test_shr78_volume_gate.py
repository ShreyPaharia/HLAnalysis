"""SHR-78: real recent_volume_usd in HL HIP-4 backtest runner.

Two tests:

1. Unit: MarketState rolling-window volume accumulator — synthetic trade events
   feed the accumulator; the returned value is the sum of (px*sz) within the
   window and drops trades outside the window.

2. Regression: a volume-gated strategy (min_recent_volume_usd=100) given a
   synthetic question whose trades exceed the gate produces > 0 entries.
   Previously this always produced 0 entries because recent_volume_usd was
   hardcoded 0.0 in run_one_question.
"""

from __future__ import annotations

import pytest

from hlanalysis.backtest.core.events import TradeEvent
from hlanalysis.backtest.runner.market_state import MarketState


# ---------------------------------------------------------------------------
# 1. Unit: MarketState rolling-window volume accumulator
# ---------------------------------------------------------------------------

_ONE_HOUR_NS = 60 * 60 * 1_000_000_000


def _make_trade(ts_ns: int, symbol: str, price: float, size: float) -> TradeEvent:
    return TradeEvent(ts_ns=ts_ns, symbol=symbol, side="buy", price=price, size=size)


class TestMarketStateVolume:
    def test_no_trades_returns_zero(self):
        ms = MarketState()
        assert ms.recent_volume_usd(("sym-yes", "sym-no"), now_ns=1_000_000_000) == 0.0

    def test_single_trade_within_window(self):
        ms = MarketState()
        now_ns = 2 * _ONE_HOUR_NS
        trade_ts = now_ns - 30 * 60 * 1_000_000_000  # 30 min ago — inside 1h window
        ms.apply_trade(_make_trade(trade_ts, "sym-yes", price=0.40, size=50.0))
        vol = ms.recent_volume_usd(("sym-yes",), now_ns=now_ns)
        assert vol == pytest.approx(0.40 * 50.0)

    def test_trade_outside_window_excluded(self):
        ms = MarketState()
        now_ns = 2 * _ONE_HOUR_NS
        old_trade_ts = now_ns - 2 * _ONE_HOUR_NS  # 2 hours ago — outside window
        ms.apply_trade(_make_trade(old_trade_ts, "sym-yes", price=0.40, size=50.0))
        vol = ms.recent_volume_usd(("sym-yes",), now_ns=now_ns)
        assert vol == 0.0

    def test_sum_across_multiple_legs(self):
        ms = MarketState()
        now_ns = 2 * _ONE_HOUR_NS
        recent_ts = now_ns - 10 * 60 * 1_000_000_000  # 10 min ago
        ms.apply_trade(_make_trade(recent_ts, "leg-yes", price=0.60, size=100.0))
        ms.apply_trade(_make_trade(recent_ts, "leg-no", price=0.40, size=80.0))
        vol = ms.recent_volume_usd(("leg-yes", "leg-no"), now_ns=now_ns)
        assert vol == pytest.approx(0.60 * 100.0 + 0.40 * 80.0)

    def test_mixed_window_only_sums_recent(self):
        """Old trade inserted first (chronological order), then recent — only recent counted."""
        ms = MarketState()
        now_ns = 2 * _ONE_HOUR_NS
        old_ts = now_ns - 90 * 60 * 1_000_000_000
        recent_ts = now_ns - 10 * 60 * 1_000_000_000
        # Insert old trade first (chronological order matches real usage).
        ms.apply_trade(_make_trade(old_ts, "sym-yes", price=0.50, size=500.0))
        ms.apply_trade(_make_trade(recent_ts, "sym-yes", price=0.50, size=100.0))
        vol = ms.recent_volume_usd(("sym-yes",), now_ns=now_ns)
        assert vol == pytest.approx(0.50 * 100.0)

    def test_unknown_symbol_returns_zero(self):
        ms = MarketState()
        ms.apply_trade(_make_trade(1_000_000, "some-sym", price=0.5, size=10.0))
        assert ms.recent_volume_usd(("other-sym",), now_ns=2_000_000_000) == 0.0


# ---------------------------------------------------------------------------
# 2. Regression: volume-gated strategy unblocked by real recent_volume_usd
# ---------------------------------------------------------------------------

from collections.abc import Mapping  # noqa: E402

from hlanalysis.backtest.data.synthetic import (  # noqa: E402
    SyntheticDataSource,
    make_default_binary_question,
)
from hlanalysis.backtest.runner.hftbt_runner import RunConfig, run_one_question  # noqa: E402
from hlanalysis.strategy.base import Strategy  # noqa: E402
from hlanalysis.strategy.types import (  # noqa: E402
    Action,
    BookState,
    Decision,
    Diagnostic,
    OrderIntent,
    Position,
)


class _VolumeGatedEnterStrategy(Strategy):
    """Enter YES iff recent_volume_usd >= min_recent_volume_usd.

    Records the last-seen volume so the test can inspect it.
    """

    name = "_vol_gated_enter_yes"

    def __init__(self, min_recent_volume_usd: float = 100.0, size: float = 10.0):
        self._min_vol = min_recent_volume_usd
        self._size = size
        self._fired = False
        self.last_volume_seen: float = -1.0

    def evaluate(
        self,
        *,
        question,
        books: Mapping[str, BookState],
        reference_price: float,
        recent_returns,
        recent_volume_usd: float,
        position: Position | None,
        now_ns: int,
        recent_hl_bars=(),
    ) -> Decision:
        self.last_volume_seen = recent_volume_usd
        diag = (Diagnostic(level="info", message="vol-gated-tick"),)
        if recent_volume_usd < self._min_vol:
            return Decision(action=Action.HOLD, diagnostics=diag)
        if self._fired or position is not None:
            return Decision(action=Action.HOLD, diagnostics=diag)
        book = books.get(question.yes_symbol)
        if book is None or book.ask_px is None:
            return Decision(action=Action.HOLD, diagnostics=diag)
        self._fired = True
        return Decision(
            action=Action.ENTER,
            diagnostics=diag,
            intents=(
                OrderIntent(
                    question_idx=question.question_idx,
                    symbol=question.yes_symbol,
                    side="buy",
                    size=self._size,
                    limit_price=min(1.0, book.ask_px + 0.01),
                    cloid=f"hla-volgated-{now_ns}",
                ),
            ),
        )


def test_volume_gated_strategy_enters_when_trades_present():
    """SHR-78 regression: volume-gated strategy now produces trades (was 0 before fix)."""
    # Synthetic question has trades starting ~1 min in with size=5.0 at ~0.31-0.51.
    # Cumulative notional across the 5 trades ≈ 5 * 5.0 * ~0.40 ≈ 10 USD total;
    # but make_default_binary_question generates trades every other scan tick.
    # We set min_recent_volume_usd low enough that the synthetic corpus clears it.
    sq = make_default_binary_question(
        start_ts_ns=0,
        duration_ns=10 * 60 * 1_000_000_000,  # 10 min
    )
    ds = SyntheticDataSource()
    ds.add_question(sq)

    # Count trades in the synthetic corpus to pick the right min threshold.
    trade_notional = sum(t.price * t.size for t in sq.trades)
    # min_recent_volume_usd just below the total so the gate opens.
    gate_threshold = trade_notional * 0.5

    strat = _VolumeGatedEnterStrategy(min_recent_volume_usd=gate_threshold, size=5.0)
    cfg = RunConfig(
        scanner_interval_seconds=60,
        slippage_bps=0.0,
        fee_taker=0.0,
    )
    res = run_one_question(strat, ds, sq.descriptor, cfg, strike=sq.strike)

    # Before the fix: strat.last_volume_seen == 0.0 (always), 0 non-settle fills.
    # After the fix: strategy sees real volume, enters at least once.
    non_settle_fills = [f for f in res.fills if f.cloid != "settle"]
    assert non_settle_fills, (
        f"SHR-78 regression: volume-gated strategy produced 0 entries. "
        f"last_volume_seen={strat.last_volume_seen:.2f}, "
        f"gate={gate_threshold:.2f}, trade_notional={trade_notional:.2f}"
    )
    # The volume the strategy saw must be positive.
    assert strat.last_volume_seen > 0.0, (
        f"recent_volume_usd was still 0.0 after fix (last seen: {strat.last_volume_seen})"
    )


def test_volume_gated_strategy_blocked_when_no_trades():
    """Strategy with high volume gate produces 0 entries when corpus has no trades."""
    from dataclasses import replace

    sq_with_trades = make_default_binary_question(start_ts_ns=0)
    # Replace trades with an empty tuple so volume stays 0 throughout.
    sq_no_trades = replace(sq_with_trades, trades=())

    ds = SyntheticDataSource()
    ds.add_question(sq_no_trades)

    strat = _VolumeGatedEnterStrategy(min_recent_volume_usd=100.0, size=5.0)
    cfg = RunConfig(scanner_interval_seconds=60, slippage_bps=0.0, fee_taker=0.0)
    res = run_one_question(strat, ds, sq_no_trades.descriptor, cfg, strike=sq_with_trades.strike)

    non_settle_fills = [f for f in res.fills if f.cloid != "settle"]
    assert len(non_settle_fills) == 0, f"Expected 0 entries when no trades, got {len(non_settle_fills)}"
    assert strat.last_volume_seen == pytest.approx(0.0), (
        f"Expected 0 volume with no trades, got {strat.last_volume_seen}"
    )
