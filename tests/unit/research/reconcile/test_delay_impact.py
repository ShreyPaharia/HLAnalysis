"""SHR-146: split matched entry/exit VWAP into DELAY (timing) vs IMPACT (spread).

The matched-leg VWAP gap conflates two causes: sim entered/exited at a DIFFERENT
TIME into a moving market (delay) vs sim filled a DIFFERENT-QUALITY price at the
SAME instant (impact). We separate them with a common arrival benchmark: the
reference price (HL perp ``mark``) sampled at BOTH the live and sim episode times.

    delay  = (ref@sim_time − ref@live_time) × size
    impact = (vwap gap) − delay
"""

from __future__ import annotations

import pandas as pd

from hlanalysis.research.reconcile.reconcile import reconcile_pnl

_T0 = 1_718_000_000_000_000_000
_1S_NS = 1_000_000_000


def _fill(ts_offset_s: int, side: str, price: float, size: float, symbol: str = "#4650") -> dict:
    return {
        "ts_ns": _T0 + ts_offset_s * _1S_NS,
        "side": side,
        "price": price,
        "size": size,
        "symbol": symbol,
        "fee": 0.0,
        "closed_pnl": 0.0,
    }


def _linear_ref(slope_per_s: float, base: float) -> object:
    """A reference-price reader where price moves linearly with time."""

    def reader(ref_symbol: str, ts_ns: int, data_root) -> float:
        return base + slope_per_s * (ts_ns - _T0) / _1S_NS

    return reader


class TestDelayImpactKeys:
    def test_waterfall_has_delay_and_impact_keys(self) -> None:
        """The matched VWAP buckets are now split into delay + impact."""
        live = pd.DataFrame([_fill(0, "buy", 0.80, 100), _fill(100, "sell", 0.90, 100)])
        sim = pd.DataFrame([_fill(0, "buy", 0.81, 100), _fill(100, "sell", 0.89, 100)])
        res = reconcile_pnl(live_fills=live, sim_fills=sim, live_settlement={}, sim_resolved={})
        wf = res.waterfall
        assert "matched_entry_delay" in wf
        assert "matched_entry_impact" in wf
        assert "matched_exit_delay" in wf
        assert "matched_exit_impact" in wf
        # The old conflated keys are gone.
        assert "matched_entry_vwap" not in wf
        assert "matched_exit_vwap" not in wf


class TestSplitSumsBackToOldTotal:
    def test_delay_plus_impact_equals_old_matched_total(self) -> None:
        """entry_delay + entry_impact reconstructs the old matched_entry_vwap."""
        live = pd.DataFrame([_fill(0, "buy", 0.80, 100), _fill(100, "sell", 0.90, 100)])
        sim = pd.DataFrame([_fill(20, "buy", 0.81, 100), _fill(120, "sell", 0.89, 100)])
        reader = _linear_ref(slope_per_s=0.0001, base=0.80)
        res = reconcile_pnl(
            live_fills=live,
            sim_fills=sim,
            live_settlement={},
            sim_resolved={},
            ref_price_reader=reader,
        )
        wf = res.waterfall
        # Old matched_entry_vwap = (sim_buy_vwap − live_buy_vwap) × size
        old_entry = (0.81 - 0.80) * 100
        old_exit = (0.90 - 0.89) * 100
        assert abs((wf["matched_entry_delay"] + wf["matched_entry_impact"]) - old_entry) < 1e-6
        assert abs((wf["matched_exit_delay"] + wf["matched_exit_impact"]) - old_exit) < 1e-6

    def test_full_waterfall_sums_to_pnl_diff(self) -> None:
        live = pd.DataFrame([_fill(0, "buy", 0.80, 100), _fill(100, "sell", 0.90, 100)])
        sim = pd.DataFrame([_fill(20, "buy", 0.81, 100), _fill(120, "sell", 0.89, 100)])
        reader = _linear_ref(slope_per_s=0.0001, base=0.80)
        res = reconcile_pnl(
            live_fills=live,
            sim_fills=sim,
            live_settlement={},
            sim_resolved={},
            ref_price_reader=reader,
        )
        assert abs(sum(res.waterfall.values()) - res.pnl_diff) < 1e-4


class TestPureTimingShift:
    def test_all_delay_zero_impact(self) -> None:
        """Both sides fill exactly at the reference, only at different times.

        The whole matched VWAP gap must be attributed to delay; impact ≈ 0.
        """
        reader = _linear_ref(slope_per_s=0.0002, base=0.80)
        # live and sim fill AT the reference price at their respective times.
        rl_buy = 0.80 + 0.0002 * 0  # ref@0s
        rl_sell = 0.80 + 0.0002 * 100  # ref@100s
        rs_buy = 0.80 + 0.0002 * 50  # ref@50s
        rs_sell = 0.80 + 0.0002 * 150  # ref@150s
        live = pd.DataFrame([_fill(0, "buy", rl_buy, 100), _fill(100, "sell", rl_sell, 100)])
        sim = pd.DataFrame([_fill(50, "buy", rs_buy, 100), _fill(150, "sell", rs_sell, 100)])
        res = reconcile_pnl(
            live_fills=live,
            sim_fills=sim,
            live_settlement={},
            sim_resolved={},
            ref_price_reader=reader,
        )
        wf = res.waterfall
        assert abs(wf["matched_entry_impact"]) < 1e-6, wf
        assert abs(wf["matched_exit_impact"]) < 1e-6, wf
        assert abs(wf["matched_entry_delay"]) > 1e-6
        assert abs(wf["matched_exit_delay"]) > 1e-6


class TestSameTimePriceQuality:
    def test_all_impact_zero_delay(self) -> None:
        """Same instant, different fill quality -> all impact, zero delay."""
        reader = _linear_ref(slope_per_s=0.0002, base=0.80)
        # live and sim fill at the SAME timestamps but different prices.
        live = pd.DataFrame([_fill(0, "buy", 0.80, 100), _fill(100, "sell", 0.90, 100)])
        sim = pd.DataFrame([_fill(0, "buy", 0.82, 100), _fill(100, "sell", 0.88, 100)])
        res = reconcile_pnl(
            live_fills=live,
            sim_fills=sim,
            live_settlement={},
            sim_resolved={},
            ref_price_reader=reader,
        )
        wf = res.waterfall
        assert abs(wf["matched_entry_delay"]) < 1e-6, wf
        assert abs(wf["matched_exit_delay"]) < 1e-6, wf
        assert abs(wf["matched_entry_impact"]) > 1e-6
        assert abs(wf["matched_exit_impact"]) > 1e-6


class TestNoReaderBackwardCompatible:
    def test_no_reader_puts_all_in_impact(self) -> None:
        """Without a reference reader, delay is 0 and the whole gap is impact."""
        live = pd.DataFrame([_fill(0, "buy", 0.80, 100), _fill(100, "sell", 0.90, 100)])
        sim = pd.DataFrame([_fill(20, "buy", 0.81, 100), _fill(120, "sell", 0.89, 100)])
        res = reconcile_pnl(live_fills=live, sim_fills=sim, live_settlement={}, sim_resolved={})
        wf = res.waterfall
        assert abs(wf["matched_entry_delay"]) < 1e-9
        assert abs(wf["matched_exit_delay"]) < 1e-9
        assert abs(wf["matched_entry_impact"] - (0.81 - 0.80) * 100) < 1e-6
        assert abs(wf["matched_exit_impact"] - (0.90 - 0.89) * 100) < 1e-6
        assert abs(sum(wf.values()) - res.pnl_diff) < 1e-4
