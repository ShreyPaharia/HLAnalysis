"""SHR-147: opportunity-cost bucket for unmatched legs marked at settlement.

The unmatched round-trips (legs one venue traded and the other did not) were
valued only on their own fills, so a leg HELD to settlement (a lone buy with no
closing sell) contributed nothing. Perold's opportunity cost marks that
un-replicated open position at the resolved SETTLEMENT price (1.0 if the held leg
won, 0.0 if it lost).
"""

from __future__ import annotations

import pandas as pd

from hlanalysis.research.reconcile.reconcile import (
    FillEpisode,
    _leg_settlement_price,
    _open_position_at_settlement,
    reconcile_pnl,
)

_T0 = 1_718_000_000_000_000_000
_1S_NS = 1_000_000_000


def _fill(ts_offset_s: int, side: str, price: float, size: float, symbol: str = "#4650", closed_pnl: float = 0.0):
    return {
        "ts_ns": _T0 + ts_offset_s * _1S_NS,
        "side": side,
        "price": price,
        "size": size,
        "symbol": symbol,
        "fee": 0.0,
        "closed_pnl": closed_pnl,
    }


def _ep(side: str, vwap: float, size: float, start_s: int = 0) -> FillEpisode:
    return FillEpisode(
        side=side,
        start_ns=_T0 + start_s * _1S_NS,
        end_ns=_T0 + start_s * _1S_NS,
        total_size=size,
        vwap=vwap,
        n_fills=1,
    )


class TestLegSettlementPrice:
    def test_yes_leg_wins(self) -> None:
        # #4650 -> side_idx 0 -> yes leg; winner yes -> settles at 1.0
        assert _leg_settlement_price("#4650", "yes") == 1.0

    def test_yes_leg_loses(self) -> None:
        assert _leg_settlement_price("#4650", "no") == 0.0

    def test_no_leg_wins(self) -> None:
        # #4651 -> side_idx 1 -> no leg; winner no -> settles at 1.0
        assert _leg_settlement_price("#4651", "no") == 1.0

    def test_unknown_winner(self) -> None:
        assert _leg_settlement_price("#4650", None) is None
        assert _leg_settlement_price("#4650", "unknown") is None


class TestOpenPositionAtSettlement:
    def test_lone_open_buy_marked_at_settlement(self) -> None:
        """A lone buy with no closing sell is valued at the settlement price."""
        eps = [_ep("BUY", vwap=0.70, size=50)]
        assert abs(_open_position_at_settlement(eps, 1.0) - (1.0 - 0.70) * 50) < 1e-9

    def test_closed_roundtrip_has_no_open_value(self) -> None:
        """A balanced buy+sell leaves no open position -> 0 settlement value."""
        eps = [_ep("BUY", vwap=0.70, size=50, start_s=0), _ep("SELL", vwap=0.90, size=50, start_s=10)]
        assert abs(_open_position_at_settlement(eps, 1.0)) < 1e-9

    def test_partial_close_open_remainder(self) -> None:
        eps = [_ep("BUY", vwap=0.70, size=100, start_s=0), _ep("SELL", vwap=0.90, size=40, start_s=10)]
        # 60 shares left open at avg cost 0.70, settled at 1.0
        assert abs(_open_position_at_settlement(eps, 1.0) - (1.0 - 0.70) * 60) < 1e-9


class TestOpportunityCostInWaterfall:
    def test_live_only_held_leg_valued_at_settlement(self) -> None:
        """An unmatched live leg held to settlement shows up in opportunity_cost."""
        # Matched round-trip shared with sim + a lone unmatched live buy held to
        # settlement (winning yes leg).
        live = pd.DataFrame(
            [
                _fill(0, "buy", 0.80, 100, closed_pnl=0.0),
                _fill(100, "sell", 0.85, 100, closed_pnl=5.0),
                _fill(200, "buy", 0.70, 50, closed_pnl=0.0),  # lone, held to settle
            ]
        )
        sim = pd.DataFrame([_fill(0, "buy", 0.80, 100), _fill(100, "sell", 0.85, 100)])
        res = reconcile_pnl(
            live_fills=live,
            sim_fills=sim,
            live_settlement={},
            sim_resolved={"winner_side": "yes"},
        )
        wf = res.waterfall
        assert "opportunity_cost" in wf
        # lone buy 50 @ 0.70 settles at 1.0 -> (1.0 - 0.70) * 50 = 15.0
        assert abs(wf["opportunity_cost"] - 15.0) < 1e-6, wf
        assert abs(sum(wf.values()) - res.pnl_diff) < 1e-4

    def test_no_settlement_zero_opportunity_cost(self) -> None:
        """No resolved winner -> opportunity_cost is 0 (cannot mark to settlement)."""
        live = pd.DataFrame(
            [
                _fill(0, "buy", 0.80, 100),
                _fill(100, "sell", 0.85, 100),
                _fill(200, "buy", 0.70, 50),
            ]
        )
        sim = pd.DataFrame([_fill(0, "buy", 0.80, 100), _fill(100, "sell", 0.85, 100)])
        res = reconcile_pnl(live_fills=live, sim_fills=sim, live_settlement={}, sim_resolved={})
        assert abs(res.waterfall["opportunity_cost"]) < 1e-9
        assert abs(sum(res.waterfall.values()) - res.pnl_diff) < 1e-4

    def test_sim_only_held_leg_valued_at_settlement(self) -> None:
        """A sim-only held leg contributes opportunity_cost with the opposite sign."""
        live = pd.DataFrame([_fill(0, "buy", 0.80, 100), _fill(100, "sell", 0.85, 100)])
        sim = pd.DataFrame(
            [
                _fill(0, "buy", 0.80, 100),
                _fill(100, "sell", 0.85, 100),
                _fill(200, "buy", 0.70, 50),  # sim-only lone buy held to settle
            ]
        )
        res = reconcile_pnl(
            live_fills=live,
            sim_fills=sim,
            live_settlement={},
            sim_resolved={"winner_side": "yes"},
        )
        # sim captured (1.0 - 0.70)*50 = 15 that live did not -> -15 to (live - sim)
        assert abs(res.waterfall["opportunity_cost"] - (-15.0)) < 1e-6, res.waterfall
        assert abs(sum(res.waterfall.values()) - res.pnl_diff) < 1e-4
