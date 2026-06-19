"""Tests for Layer 2: fill episode grouping and reconciliation."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from hlanalysis.research.reconcile.book import book_parity_pct
from hlanalysis.research.reconcile.reconcile import (
    _group_episodes,
    reconcile_fills,
)

_T0 = 1_718_000_000_000_000_000
_1S_NS = 1_000_000_000


def _fill(ts_offset_s: int, side: str, price: float, size: float, symbol: str = "#4010") -> dict:
    return {
        "ts_ns": _T0 + ts_offset_s * _1S_NS,
        "side": side,
        "price": price,
        "size": size,
        "symbol": symbol,
        "fee": 0.001,
        "closed_pnl": 0.0,
    }


class TestEpisodeGrouping:
    def test_episode_grouping_buy_buy_sell(self) -> None:
        """BUY, BUY, SELL -> 2 episodes (one BUY, one SELL)."""
        fills = pd.DataFrame(
            [
                _fill(0, "BUY", 0.80, 10),
                _fill(5, "BUY", 0.82, 10),
                _fill(60, "SELL", 0.85, 20),
            ]
        )
        eps = _group_episodes(fills)
        assert len(eps) == 2
        assert eps[0].side == "BUY"
        assert eps[0].n_fills == 2
        assert eps[1].side == "SELL"
        assert eps[1].n_fills == 1

    def test_episode_grouping_alternating(self) -> None:
        """BUY, SELL, BUY -> 3 separate episodes."""
        fills = pd.DataFrame(
            [
                _fill(0, "BUY", 0.80, 10),
                _fill(10, "SELL", 0.85, 10),
                _fill(20, "BUY", 0.78, 10),
            ]
        )
        eps = _group_episodes(fills)
        assert len(eps) == 3
        assert eps[0].side == "BUY"
        assert eps[1].side == "SELL"
        assert eps[2].side == "BUY"

    def test_episode_grouping_empty(self) -> None:
        """Empty fills -> empty episode list."""
        eps = _group_episodes(pd.DataFrame())
        assert eps == []

    def test_episode_grouping_single_fill(self) -> None:
        """Single fill -> single episode with n_fills=1."""
        fills = pd.DataFrame([_fill(0, "BUY", 0.80, 10)])
        eps = _group_episodes(fills)
        assert len(eps) == 1
        assert eps[0].n_fills == 1
        assert eps[0].total_size == 10.0

    def test_episode_grouping_all_same_side(self) -> None:
        """All BUY fills -> single episode."""
        fills = pd.DataFrame(
            [
                _fill(0, "BUY", 0.80, 10),
                _fill(5, "BUY", 0.81, 10),
                _fill(10, "BUY", 0.82, 10),
            ]
        )
        eps = _group_episodes(fills)
        assert len(eps) == 1
        assert eps[0].n_fills == 3
        assert eps[0].total_size == 30.0


class TestEpisodeVWAP:
    def test_episode_vwap_two_fills(self) -> None:
        """2 fills at 0.80 (sz=10) and 0.82 (sz=20) -> VWAP = 0.8133..."""
        fills = pd.DataFrame(
            [
                _fill(0, "BUY", 0.80, 10),
                _fill(5, "BUY", 0.82, 20),
            ]
        )
        eps = _group_episodes(fills)
        assert len(eps) == 1
        expected_vwap = (0.80 * 10 + 0.82 * 20) / 30
        assert abs(eps[0].vwap - expected_vwap) < 1e-9

    def test_episode_vwap_equal_sizes(self) -> None:
        """Equal sizes -> VWAP = simple average."""
        fills = pd.DataFrame(
            [
                _fill(0, "BUY", 0.80, 10),
                _fill(5, "BUY", 0.90, 10),
            ]
        )
        eps = _group_episodes(fills)
        assert abs(eps[0].vwap - 0.85) < 1e-9

    def test_episode_timestamps(self) -> None:
        """Episode start_ns and end_ns match first and last fill timestamps."""
        fills = pd.DataFrame(
            [
                _fill(0, "BUY", 0.80, 10),
                _fill(5, "BUY", 0.82, 10),
                _fill(10, "BUY", 0.84, 10),
            ]
        )
        eps = _group_episodes(fills)
        assert eps[0].start_ns == _T0
        assert eps[0].end_ns == _T0 + 10 * _1S_NS


class TestBookParity:
    def _make_reader(self, ask_px: float, bid_px: float):
        """Return a fake book reader that always returns the given prices."""

        def reader(leg_symbol: str, ts_ns: int, data_root: Path) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "exchange_ts": ts_ns,
                        "bid_px": bid_px,
                        "bid_sz": 100.0,
                        "ask_px": ask_px,
                        "ask_sz": 100.0,
                    }
                ]
            )

        return reader

    def test_book_parity_injected_all_match(self) -> None:
        """All fills at ask_px or below -> 100% parity."""
        fills = pd.DataFrame(
            [
                _fill(0, "BUY", 0.80, 10),
                _fill(5, "BUY", 0.80, 10),
            ]
        )
        reader = self._make_reader(ask_px=0.85, bid_px=0.75)
        pct = book_parity_pct(fills, Path("."), reader=reader)
        assert pct == 1.0

    def test_book_parity_injected_none_match(self) -> None:
        """BUY at price above ask + tol -> 0% parity."""
        fills = pd.DataFrame([_fill(0, "BUY", 0.99, 10)])
        reader = self._make_reader(ask_px=0.80, bid_px=0.75)
        pct = book_parity_pct(fills, Path("."), reader=reader)
        assert pct == 0.0

    def test_book_parity_injected_partial_match(self) -> None:
        """One fill matches, one doesn't -> 50% parity."""
        fills = pd.DataFrame(
            [
                _fill(0, "BUY", 0.80, 10),  # <= ask 0.82 + tol -> match
                _fill(5, "BUY", 0.99, 10),  # > ask 0.82 + tol -> no match
            ]
        )
        reader = self._make_reader(ask_px=0.82, bid_px=0.75)
        pct = book_parity_pct(fills, Path("."), reader=reader)
        assert pct == 0.5

    def test_book_parity_empty_fills(self) -> None:
        """Empty fills -> nan."""
        pct = book_parity_pct(pd.DataFrame(), Path("."), reader=None)
        assert math.isnan(pct)

    def test_book_parity_sell_side(self) -> None:
        """SELL at bid_px or above -> match."""
        fills = pd.DataFrame([_fill(0, "SELL", 0.80, 10)])
        reader = self._make_reader(ask_px=0.85, bid_px=0.78)
        pct = book_parity_pct(fills, Path("."), reader=reader)
        assert pct == 1.0


class TestFillsReconcile:
    def test_fills_reconcile_match(self) -> None:
        """Matching live+sim episodes -> 'match' classification."""
        fills = pd.DataFrame(
            [
                _fill(0, "BUY", 0.80, 10),
                _fill(60, "SELL", 0.85, 10),
            ]
        )
        result = reconcile_fills(live_fills=fills, sim_fills=fills.copy())
        assert result.gap_classification == "match"
        assert result.n_live_fills == 2
        assert result.n_sim_fills == 2
        assert len(result.live_episodes) == 2
        assert len(result.sim_episodes) == 2

    def test_fills_missed_episode(self) -> None:
        """Live has 2 episodes, sim has 1 -> 'missed_episode'."""
        live = pd.DataFrame(
            [
                _fill(0, "BUY", 0.80, 10),
                _fill(60, "SELL", 0.85, 10),
            ]
        )
        sim = pd.DataFrame(
            [
                _fill(0, "BUY", 0.80, 10),
            ]
        )
        result = reconcile_fills(live_fills=live, sim_fills=sim)
        assert result.gap_classification == "missed_episode"
        assert len(result.live_episodes) == 2
        assert len(result.sim_episodes) == 1

    def test_fills_extra_episode(self) -> None:
        """Sim has more episodes than live -> 'extra_episode'."""
        live = pd.DataFrame([_fill(0, "BUY", 0.80, 10)])
        sim = pd.DataFrame(
            [
                _fill(0, "BUY", 0.80, 10),
                _fill(60, "SELL", 0.85, 10),
            ]
        )
        result = reconcile_fills(live_fills=live, sim_fills=sim)
        assert result.gap_classification == "extra_episode"

    def test_fills_vwap_diff_in_table(self) -> None:
        """Episode table shows VWAP diff between live and sim."""
        live = pd.DataFrame([_fill(0, "BUY", 0.80, 10)])
        sim = pd.DataFrame([_fill(0, "BUY", 0.82, 10)])  # slightly different price
        result = reconcile_fills(live_fills=live, sim_fills=sim)
        table = result.episode_table
        assert not table.empty
        assert abs(table.iloc[0]["vwap_diff"] - 0.02) < 1e-9

    def test_fills_both_empty(self) -> None:
        """Both fills empty -> match with 0 episodes."""
        result = reconcile_fills(
            live_fills=pd.DataFrame(),
            sim_fills=pd.DataFrame(),
        )
        assert result.gap_classification == "match"
        assert result.n_live_fills == 0
        assert result.n_sim_fills == 0

    def test_fills_book_parity_with_reader(self) -> None:
        """Book parity computed when reader is injected."""

        def reader(leg_symbol: str, ts_ns: int, data_root: Path) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "exchange_ts": ts_ns,
                        "bid_px": 0.75,
                        "bid_sz": 100.0,
                        "ask_px": 0.85,
                        "ask_sz": 100.0,
                    }
                ]
            )

        live = pd.DataFrame([_fill(0, "BUY", 0.80, 10)])
        result = reconcile_fills(
            live_fills=live,
            sim_fills=live.copy(),
            book_reader=reader,
        )
        assert result.book_parity_pct is not None
        assert result.book_parity_pct == 1.0
