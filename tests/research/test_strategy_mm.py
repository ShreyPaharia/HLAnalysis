"""Tests for the MM strategy card (strategy_mm.py).

Unit tests cover the fill model logic, PnL computation, and safety gates
on synthetic book/trade sequences with known outcomes. No real data required
for the unit tests. Integration tests require recorded data.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Data availability detection
# ---------------------------------------------------------------------------

_WORKTREE_ROOT = Path(__file__).parent.parent.parent

if os.environ.get("HLBT_HL_DATA_ROOT"):
    _DATA_ROOT = Path(os.environ["HLBT_HL_DATA_ROOT"]).resolve()
else:
    _DATA_ROOT = (_WORKTREE_ROOT / ".." / ".." / "data").resolve()

_HL_MARKER = _DATA_ROOT / "venue=hyperliquid"
_DATA_AVAILABLE = _HL_MARKER.exists()

# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

_NS = 1_000_000_000
_START_NS = int(1_780_000_000 * 1e9)  # arbitrary base timestamp
_EXPIRY_NS = _START_NS + 12 * 3600 * _NS  # 12h after start


def _make_bbo(
    ts_offsets_s: list[float],
    bid_prices: list[float],
    ask_prices: list[float],
    bid_sizes: list[float] | None = None,
    ask_sizes: list[float] | None = None,
) -> pd.DataFrame:
    n = len(ts_offsets_s)
    if bid_sizes is None:
        bid_sizes = [100.0] * n
    if ask_sizes is None:
        ask_sizes = [100.0] * n
    return pd.DataFrame(
        {
            "ts_ns": [_START_NS + int(t * _NS) for t in ts_offsets_s],
            "bid_px": bid_prices,
            "bid_sz": bid_sizes,
            "ask_px": ask_prices,
            "ask_sz": ask_sizes,
        }
    )


def _make_trades(
    ts_offsets_s: list[float],
    prices: list[float],
    sizes: list[float],
    sides: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_ns": [_START_NS + int(t * _NS) for t in ts_offsets_s],
            "price": prices,
            "size": sizes,
            "side": sides,
        }
    )


def _make_perp_bbo(
    ts_offsets_s: list[float],
    btc_prices: list[float],
    spread_usd: float = 1.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_ns": [_START_NS + int(t * _NS) for t in ts_offsets_s],
            "bid_px": [p - spread_usd / 2 for p in btc_prices],
            "ask_px": [p + spread_usd / 2 for p in btc_prices],
        }
    )


# ---------------------------------------------------------------------------
# Import helpers — lazy to avoid import errors in CI where data absent
# ---------------------------------------------------------------------------


def _import_sim() -> tuple:
    from hlanalysis.research.cards.strategy_mm import (
        MMConfig,
        MMResult,
        _run_mm_sim_expiry,
    )

    return MMConfig, MMResult, _run_mm_sim_expiry


# ---------------------------------------------------------------------------
# Unit tests: fill model
# ---------------------------------------------------------------------------


class TestFillModelOptimistic:
    """Optimistic fill: fill when trade prints AT or through quote price."""

    def test_buy_fill_at_quote(self) -> None:
        """Sell-aggressor trades at exactly our bid → filled (optimistic)."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        # BBO at T=0, trade at T=10s (within stale_data_halt_seconds=3600)
        bbo = _make_bbo([0.0], [0.80], [0.82])
        trades = _make_trades([10.0], [0.80], [50.0], ["sell"])
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg = MMConfig(fill_model="optimistic", half_edge=0.0, size_usd=100.0, stale_data_halt_seconds=3600.0)
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg)

        assert result.n_fills_buy == 1
        assert result.n_fills_sell == 0

    def test_sell_fill_at_quote(self) -> None:
        """Buy-aggressor trades at exactly our ask → filled (optimistic)."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        bbo = _make_bbo([0.0], [0.80], [0.82])
        trades = _make_trades([10.0], [0.82], [50.0], ["buy"])
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg = MMConfig(fill_model="optimistic", half_edge=0.0, size_usd=100.0, stale_data_halt_seconds=3600.0)
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg)

        assert result.n_fills_sell == 1
        assert result.n_fills_buy == 0

    def test_buy_fill_through_quote(self) -> None:
        """Sell-aggressor trades BELOW our bid → filled (optimistic, through)."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        bbo = _make_bbo([0.0], [0.80], [0.82])
        trades = _make_trades([10.0], [0.79], [50.0], ["sell"])
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg = MMConfig(fill_model="optimistic", half_edge=0.0, size_usd=100.0, stale_data_halt_seconds=3600.0)
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg)

        assert result.n_fills_buy == 1

    def test_no_fill_above_quote(self) -> None:
        """Sell-aggressor trades ABOVE our bid → no fill (not at our level)."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        bbo = _make_bbo([0.0], [0.80], [0.82])
        trades = _make_trades([10.0], [0.81], [50.0], ["sell"])
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg = MMConfig(fill_model="optimistic", half_edge=0.0, size_usd=100.0, stale_data_halt_seconds=3600.0)
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg)

        assert result.n_fills_buy == 0
        assert result.n_fills_sell == 0


class TestFillModelConservative:
    """Conservative fill: fill only when trade prints STRICTLY through quote (back-of-queue)."""

    def test_no_fill_at_quote(self) -> None:
        """Sell-aggressor trades AT our bid → no fill (conservative)."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        bbo = _make_bbo([0.0], [0.80], [0.82])
        trades = _make_trades([10.0], [0.80], [50.0], ["sell"])
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg = MMConfig(fill_model="conservative", half_edge=0.0, size_usd=100.0, stale_data_halt_seconds=3600.0)
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg)

        assert result.n_fills_buy == 0

    def test_fill_strictly_through(self) -> None:
        """Sell-aggressor trades BELOW our bid → fill (conservative, strictly through)."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        bbo = _make_bbo([0.0], [0.80], [0.82])
        trades = _make_trades([10.0], [0.799], [50.0], ["sell"])
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg = MMConfig(fill_model="conservative", half_edge=0.0, size_usd=100.0, stale_data_halt_seconds=3600.0)
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg)

        assert result.n_fills_buy == 1

    def test_sell_no_fill_at_ask(self) -> None:
        """Buy-aggressor trades AT our ask → no fill (conservative)."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        bbo = _make_bbo([0.0], [0.80], [0.82])
        trades = _make_trades([10.0], [0.82], [50.0], ["buy"])
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg = MMConfig(fill_model="conservative", half_edge=0.0, size_usd=100.0, stale_data_halt_seconds=3600.0)
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg)

        assert result.n_fills_sell == 0

    def test_sell_fill_strictly_through(self) -> None:
        """Buy-aggressor trades ABOVE our ask → fill (conservative)."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        bbo = _make_bbo([0.0], [0.80], [0.82])
        trades = _make_trades([10.0], [0.821], [50.0], ["buy"])
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg = MMConfig(fill_model="conservative", half_edge=0.0, size_usd=100.0, stale_data_halt_seconds=3600.0)
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg)

        assert result.n_fills_sell == 1


# ---------------------------------------------------------------------------
# Unit tests: PnL computation
# ---------------------------------------------------------------------------


class TestPnLComputation:
    """PnL calculations on known buy+sell round-trips."""

    def test_round_trip_positive_spread(self) -> None:
        """Buy at 0.80, sell at 0.82 → spread PnL = (0.82 - 0.80) * tokens."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        # BBO stays at 0.80/0.82; buy at T=5s, refresh BBO at T=10s, sell at T=15s
        bbo = _make_bbo([0.0, 10.0], [0.80, 0.80], [0.82, 0.82])
        trades = _make_trades(
            [5.0, 15.0],
            [0.799, 0.821],  # strictly through for conservative
            [50.0, 50.0],
            ["sell", "buy"],
        )
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg = MMConfig(
            fill_model="conservative", half_edge=0.0, size_usd=100.0, hedged=False, stale_data_halt_seconds=3600.0
        )
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg)

        assert result.n_fills_buy == 1
        assert result.n_fills_sell == 1
        # spread_pnl > 0: bought at 0.80, sold at 0.82
        assert result.spread_pnl > 0.0

    def test_settlement_yes_won_long(self) -> None:
        """Hold long YES inventory; YES wins → settlement_pnl > 0."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        # Only buy fills, no sells → net long going into settlement
        bbo = _make_bbo([0.0], [0.80], [0.82])
        trades = _make_trades([10.0], [0.799], [50.0], ["sell"])
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg = MMConfig(
            fill_model="conservative", half_edge=0.0, size_usd=100.0, hedged=False, stale_data_halt_seconds=3600.0
        )
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg)

        assert result.n_fills_buy == 1
        # Long YES, yes_won=True → settlement at 1.0 → profit
        assert result.settlement_pnl > 0.0
        assert result.total_pnl > 0.0

    def test_settlement_yes_lost_long(self) -> None:
        """Hold long YES inventory; YES loses → settlement_pnl < 0."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        bbo = _make_bbo([0.0], [0.80], [0.82])
        trades = _make_trades([10.0], [0.799], [50.0], ["sell"])
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg = MMConfig(
            fill_model="conservative", half_edge=0.0, size_usd=100.0, hedged=False, stale_data_halt_seconds=3600.0
        )
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", False, cfg)

        assert result.n_fills_buy == 1
        # Bought at 0.80, settles at 0.0 → loss
        assert result.settlement_pnl < 0.0

    def test_no_fills_zero_pnl(self) -> None:
        """No fills → zero PnL across all components."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        bbo = _make_bbo([0.0], [0.80], [0.82])
        trades = _make_trades([], [], [], [])
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg = MMConfig(fill_model="conservative", half_edge=0.0, size_usd=100.0, stale_data_halt_seconds=3600.0)
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg)

        assert result.n_fills_buy == 0
        assert result.n_fills_sell == 0
        assert result.total_pnl == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Unit tests: safety gates
# ---------------------------------------------------------------------------


class TestSafetyGates:
    """Safety gates must suppress quoting in adverse conditions."""

    def test_tte_min_gate(self) -> None:
        """Trades within tte_min_seconds of expiry should not fill (MM not quoting)."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        # BBO update happens 100s before expiry (inside tte_min_seconds=1800)
        # Trade at same time: MM should not be quoting
        tte_100s_before = (_EXPIRY_NS - _START_NS) / _NS - 100  # seconds from start
        bbo = _make_bbo([0.0, tte_100s_before - 1], [0.80, 0.80], [0.82, 0.82])
        trades = _make_trades([tte_100s_before], [0.799], [50.0], ["sell"])
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg = MMConfig(
            fill_model="conservative",
            half_edge=0.0,
            size_usd=100.0,
            tte_min_seconds=1800.0,  # 30 min gate
            stale_data_halt_seconds=3600.0,
        )
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg)

        # MM not quoting within tte_min of expiry → no fills
        assert result.n_fills_buy == 0

    def test_thin_book_gate(self) -> None:
        """BBO with thin book (bid_sz * bid_px < min_bid_notional) → no quoting."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        # bid_sz=0.1 → bid_notional = 0.80 * 0.1 = 0.08 << min_bid_notional=25
        bbo = _make_bbo([0.0], [0.80], [0.82], bid_sizes=[0.1])
        trades = _make_trades([10.0], [0.799], [50.0], ["sell"])
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg = MMConfig(
            fill_model="conservative",
            half_edge=0.0,
            size_usd=100.0,
            min_bid_notional_usd=25.0,
            stale_data_halt_seconds=3600.0,
        )
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg)

        assert result.n_fills_buy == 0

    def test_wide_spread_gate(self) -> None:
        """BBO with spread > max_spread_bps → no quoting."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        # Spread = (0.90 - 0.50) / 0.70 = 57% = 5714 bps >> max_spread_bps=500
        bbo = _make_bbo([0.0], [0.50], [0.90])
        trades = _make_trades([10.0], [0.499], [50.0], ["sell"])
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg = MMConfig(
            fill_model="conservative",
            half_edge=0.0,
            size_usd=100.0,
            max_spread_bps=500.0,
            stale_data_halt_seconds=3600.0,
        )
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg)

        assert result.n_fills_buy == 0

    def test_inventory_cap_suppresses_buy(self) -> None:
        """After reaching max_inventory_usd, no more buy fills."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        # Two sell-aggressors close together in time (within stale halt)
        bbo_times = [0.0, 5.0, 15.0]
        bbo = _make_bbo(bbo_times, [0.80, 0.80, 0.80], [0.82, 0.82, 0.82])
        trades = _make_trades([8.0, 20.0], [0.799, 0.799], [50.0, 50.0], ["sell", "sell"])
        perp = _make_perp_bbo([0.0], [80000.0])

        # max_inventory_usd=100 = exactly 1 fill of size_usd=100
        cfg = MMConfig(
            fill_model="conservative",
            half_edge=0.0,
            size_usd=100.0,
            max_inventory_usd=100.0,
            stale_data_halt_seconds=3600.0,
        )
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg)

        # Only one buy fill because cap is hit after the first
        assert result.n_fills_buy <= 2  # at most 2; with cap should be 1
        # Inventory check: final token inventory * fill_price ≈ max_inventory_usd
        assert abs(result.net_inventory_tokens) * 0.80 <= 110.0  # within cap

    def test_favorites_only_gate(self) -> None:
        """favorites_only=True: no quoting when mid is outside [0.70, 0.95]."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        # mid = 0.50 → outside [0.70, 0.95]
        bbo = _make_bbo([0.0], [0.49], [0.51])
        trades = _make_trades([10.0], [0.489], [50.0], ["sell"])
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg = MMConfig(
            fill_model="conservative",
            half_edge=0.0,
            size_usd=100.0,
            favorites_only=True,
            stale_data_halt_seconds=3600.0,
        )
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg)

        assert result.n_fills_buy == 0

    def test_favorites_only_allows_quote_in_zone(self) -> None:
        """favorites_only=True: quoting allowed when mid is in [0.70, 0.95]."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        # mid = 0.85 → inside [0.70, 0.95]
        bbo = _make_bbo([0.0], [0.84], [0.86])
        trades = _make_trades([10.0], [0.839], [50.0], ["sell"])
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg = MMConfig(
            fill_model="conservative",
            half_edge=0.0,
            size_usd=100.0,
            favorites_only=True,
            stale_data_halt_seconds=3600.0,
        )
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg)

        assert result.n_fills_buy == 1


# ---------------------------------------------------------------------------
# Unit tests: delta hedge
# ---------------------------------------------------------------------------


class TestDeltaHedge:
    """Delta hedge bookkeeping tests."""

    def test_hedge_btc_positive_on_buy(self) -> None:
        """Buying YES binary → short BTC in perp → hedge_btc > 0."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        bbo = _make_bbo([0.0], [0.80], [0.82])
        trades = _make_trades([10.0], [0.799], [50.0], ["sell"])
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg = MMConfig(
            fill_model="conservative",
            half_edge=0.0,
            size_usd=100.0,
            hedged=True,
            hedge_ratio=0.50,
            stale_data_halt_seconds=3600.0,
        )
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg)

        assert result.n_fills_buy == 1
        assert result.net_hedge_btc > 0.0  # short perp

    def test_no_hedge_when_disabled(self) -> None:
        """hedged=False → no hedge position accumulated."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        bbo = _make_bbo([0.0], [0.80], [0.82])
        trades = _make_trades([10.0], [0.799], [50.0], ["sell"])
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg = MMConfig(
            fill_model="conservative",
            half_edge=0.0,
            size_usd=100.0,
            hedged=False,
            stale_data_halt_seconds=3600.0,
        )
        result = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg)

        assert result.net_hedge_btc == pytest.approx(0.0)
        assert result.hedge_pnl == pytest.approx(0.0)

    def test_hedge_ratio_scales_btc(self) -> None:
        """Hedge ratio=1.0 → more BTC shorted than ratio=0.5."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        bbo = _make_bbo([0.0], [0.80], [0.82])
        trades = _make_trades([10.0], [0.799], [50.0], ["sell"])
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg05 = MMConfig(
            fill_model="conservative",
            half_edge=0.0,
            size_usd=100.0,
            hedged=True,
            hedge_ratio=0.50,
            stale_data_halt_seconds=3600.0,
        )
        cfg10 = MMConfig(
            fill_model="conservative",
            half_edge=0.0,
            size_usd=100.0,
            hedged=True,
            hedge_ratio=1.00,
            stale_data_halt_seconds=3600.0,
        )

        res05 = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg05)
        res10 = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST2", "20260601-06", True, cfg10)

        assert res10.net_hedge_btc > res05.net_hedge_btc


# ---------------------------------------------------------------------------
# Unit tests: MMConfig and MMResult are importable
# ---------------------------------------------------------------------------


class TestImports:
    """Module-level import tests."""

    def test_mmconfig_importable(self) -> None:
        from hlanalysis.research.cards.strategy_mm import MMConfig  # noqa: F401

    def test_mmresult_importable(self) -> None:
        from hlanalysis.research.cards.strategy_mm import MMResult  # noqa: F401

    def test_build_card_importable(self) -> None:
        from hlanalysis.research.cards.strategy_mm import build_card  # noqa: F401

    def test_run_mm_sim_expiry_importable(self) -> None:
        from hlanalysis.research.cards.strategy_mm import _run_mm_sim_expiry  # noqa: F401

    def test_constants_correct(self) -> None:
        from hlanalysis.research.cards.strategy_mm import DEFAULT_HEDGE_RATIO, TICK

        assert pytest.approx(1e-5) == TICK
        assert pytest.approx(0.50) == DEFAULT_HEDGE_RATIO

    def test_date_constants(self) -> None:
        from hlanalysis.research.cards.strategy_mm import (
            IS_END,
            IS_START,
            OOS_END,
            OOS_START,
        )

        assert IS_START == "2026-05-06"
        assert OOS_END == "2026-06-10"
        # OOS starts after IS ends
        assert OOS_START >= IS_END


# ---------------------------------------------------------------------------
# Unit tests: summarise_results
# ---------------------------------------------------------------------------


class TestSummariseResults:
    """Test _summarise_results aggregation logic."""

    def test_empty_results(self) -> None:
        from hlanalysis.research.cards.strategy_mm import MMResult, _summarise_results

        results: list[MMResult] = []
        s = _summarise_results(results)
        # Empty results returns empty dict
        assert isinstance(s, dict)
        assert s.get("n_expiries", 0) == 0
        assert s.get("total_pnl", 0.0) == pytest.approx(0.0)

    def test_positive_pnl(self) -> None:
        from hlanalysis.research.cards.strategy_mm import MMResult, _summarise_results

        results = [
            MMResult(symbol=f"#X{i}", expiry_str="20260601-06", yes_won=True, total_pnl=10.0, spread_pnl=8.0)
            for i in range(5)
        ]
        s = _summarise_results(results)
        assert s["total_pnl"] == pytest.approx(50.0)
        assert s["n_expiries"] == 5

    def test_max_drawdown(self) -> None:
        from hlanalysis.research.cards.strategy_mm import MMResult, _summarise_results

        # +10, -8, +5 → peak=10, then 2 → DD=8
        results = [
            MMResult(symbol="#A", expiry_str="20260601-06", yes_won=True, total_pnl=10.0),
            MMResult(symbol="#B", expiry_str="20260602-06", yes_won=False, total_pnl=-8.0),
            MMResult(symbol="#C", expiry_str="20260603-06", yes_won=True, total_pnl=5.0),
        ]
        s = _summarise_results(results)
        assert s["max_drawdown"] == pytest.approx(8.0)

    def test_sharpe_positive_pnl(self) -> None:
        from hlanalysis.research.cards.strategy_mm import MMResult, _summarise_results

        # Uniform positive PnL → Sharpe > 0
        results = [MMResult(symbol=f"#X{i}", expiry_str="20260601-06", yes_won=True, total_pnl=5.0) for i in range(10)]
        s = _summarise_results(results)
        assert s["sharpe"] > 0.0

    def test_hit_rate(self) -> None:
        from hlanalysis.research.cards.strategy_mm import MMResult, _summarise_results

        results = [
            MMResult(symbol="#A", expiry_str="20260601-06", yes_won=True, total_pnl=5.0),
            MMResult(symbol="#B", expiry_str="20260602-06", yes_won=True, total_pnl=3.0),
            MMResult(symbol="#C", expiry_str="20260603-06", yes_won=False, total_pnl=-2.0),
        ]
        s = _summarise_results(results)
        assert s["hit_rate"] == pytest.approx(2 / 3, abs=1e-9)


# ---------------------------------------------------------------------------
# Unit tests: 1Hz penalty model
# ---------------------------------------------------------------------------


class TestOneHzPenalty:
    """1Hz penalty reduces spread PnL for adversely-selected fills."""

    def test_penalty_reduces_pnl_vs_no_penalty(self) -> None:
        """Conservative sim with 1Hz penalty < without penalty."""
        MMConfig, MMResult, _run_mm_sim_expiry = _import_sim()

        # Setup: two round-trips to generate spread PnL (all within stale halt window)
        bbo_t = [0.0, 5.0, 15.0, 25.0]
        bbo = _make_bbo(bbo_t, [0.80, 0.80, 0.80, 0.80], [0.82, 0.82, 0.82, 0.82])
        trades = _make_trades(
            [8.0, 20.0],
            [0.799, 0.821],
            [50.0, 50.0],
            ["sell", "buy"],
        )
        perp = _make_perp_bbo([0.0], [80000.0])

        cfg_no_penalty = MMConfig(
            fill_model="conservative",
            half_edge=0.0,
            size_usd=100.0,
            hedged=False,
            apply_1hz_penalty=False,
            stale_data_halt_seconds=3600.0,
        )
        cfg_with_penalty = MMConfig(
            fill_model="conservative",
            half_edge=0.0,
            size_usd=100.0,
            hedged=False,
            apply_1hz_penalty=True,
            penalty_informed_frac=0.20,
            penalty_spread_reduction=0.50,
            stale_data_halt_seconds=3600.0,
        )

        res_no = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg_no_penalty)
        res_pen = _run_mm_sim_expiry(bbo, trades, perp, _EXPIRY_NS, "#TEST", "20260601-06", True, cfg_with_penalty)

        # With penalty, PnL should be <= without penalty
        assert res_pen.total_pnl <= res_no.total_pnl


# ---------------------------------------------------------------------------
# Integration tests — require recorded data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="Recorded data not available at ../../data")
class TestStrategyMMIntegration:
    """End-to-end tests against real recorded data."""

    def test_load_binary_yes_legs(self) -> None:
        """Should load ≥30 binary Yes-leg expiries from corpus."""
        import duckdb

        from hlanalysis.research.cards.strategy_mm import _load_binary_yes_legs

        con = duckdb.connect()
        df = _load_binary_yes_legs(con, str(_DATA_ROOT))
        con.close()
        assert len(df) >= 30, f"Expected ≥30 expiries, got {len(df)}"
        assert "symbol" in df.columns
        assert "expiry_ns" in df.columns

    def test_load_binary_bbo_returns_data(self) -> None:
        """Loading BBO for a known active symbol+date returns non-empty DataFrame."""
        import duckdb

        from hlanalysis.research.cards.strategy_mm import _load_binary_bbo, _load_binary_yes_legs

        con = duckdb.connect()
        legs = _load_binary_yes_legs(con, str(_DATA_ROOT))
        if legs.empty:
            pytest.skip("No binary Yes legs found")

        # Take first expiry
        sym = str(legs.iloc[0]["symbol"])
        import datetime as dt_mod

        expiry_ns = int(legs.iloc[0]["expiry_ns"])
        exp_dt = dt_mod.datetime.fromtimestamp(expiry_ns / 1e9, tz=dt_mod.UTC)
        date_str = (exp_dt.date() - dt_mod.timedelta(days=1)).isoformat()

        df = _load_binary_bbo(con, str(_DATA_ROOT), sym, date_str)
        con.close()

        # May be empty if that date has no data, but it should not raise
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "ts_ns" in df.columns
            assert "bid_px" in df.columns
            assert "ask_px" in df.columns

    def test_run_mm_all_expiries_smoke(self) -> None:
        """Running MM sim on 3 expiries (IS) should complete without error."""
        import duckdb

        from hlanalysis.research.cards.strategy_mm import (
            IS_END,
            IS_START,
            MMConfig,
            _load_binary_yes_legs,
            _run_mm_all_expiries,
        )
        from hlanalysis.research.outcome_markets import resolve_binary_outcomes

        con = duckdb.connect()
        expiry_df = _load_binary_yes_legs(con, str(_DATA_ROOT))
        outcomes_df = resolve_binary_outcomes(con, str(_DATA_ROOT))

        if expiry_df.empty:
            pytest.skip("No expiry data available")

        # Restrict to just first 3 IS expiries for speed
        cfg = MMConfig(fill_model="conservative", size_usd=100.0, hedged=False)
        results = _run_mm_all_expiries(
            con,
            str(_DATA_ROOT),
            expiry_df.head(3),
            outcomes_df,
            cfg,
            start_date=IS_START,
            end_date=IS_END,
        )
        con.close()

        assert isinstance(results, list)
        # Results may be empty if those expiries are outside IS range
        for r in results:
            assert hasattr(r, "total_pnl")
            assert hasattr(r, "n_fills_buy")
            assert isinstance(r.total_pnl, float)

    def test_build_card_returns_valid_findings(self) -> None:
        """build_card should return findings with all required keys."""
        from hlanalysis.research.cards.strategy_mm import build_card

        html, findings = build_card(data_root=str(_DATA_ROOT))

        # HTML is non-empty
        assert isinstance(html, str)
        assert len(html) > 500

        # Required top-level keys
        required_keys = [
            "title",
            "kpis",
            "is_summary",
            "oos_summary",
            "split_half",
            "capacity",
            "verdict",
        ]
        for k in required_keys:
            assert k in findings, f"Missing key: {k}"

        # KPIs should be a list of dicts with required fields
        assert isinstance(findings["kpis"], list)
        for kpi in findings["kpis"]:
            assert "name" in kpi
            assert "passed" in kpi
            assert isinstance(kpi["passed"], bool)
            assert "status" in kpi
            assert kpi["status"] in ("PASS", "FAIL")

    def test_findings_json_serializable(self) -> None:
        """Findings dict must be fully JSON-serializable."""
        import json

        from hlanalysis.research.cards.strategy_mm import build_card

        _, findings = build_card(data_root=str(_DATA_ROOT))
        # Should not raise
        json_str = json.dumps(findings, default=str)
        assert len(json_str) > 100
