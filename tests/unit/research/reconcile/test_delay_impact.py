"""SHR-146: split matched entry/exit VWAP into DELAY (timing) vs IMPACT (spread).

The matched-leg VWAP gap conflates two causes: sim entered/exited at a DIFFERENT
TIME into a moving market (delay) vs sim filled a DIFFERENT-QUALITY price at the
SAME instant (impact). We separate them with a common arrival benchmark sampled at
BOTH the live and sim episode times. The DEFAULT benchmark is the traded leg's own
recorded book mid (its delta-scaled fair value, in option price space); a caller
may inject ``ref_price_reader`` to override it with an explicit benchmark series.

    delay  = (benchmark@sim_time − benchmark@live_time) × size
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


def _book_reader(mids: dict[int, float], spread: float = 0.002) -> object:
    """A recorded-book reader returning a bid/ask bracketing the mid at each ts.

    ``mids`` maps ts_offset_s → option mid; the reader LOCFs to the closest prior key.
    """

    def reader(leg_symbol: str, ts_ns: int, data_root) -> pd.DataFrame:
        off = (ts_ns - _T0) / _1S_NS
        key = max((k for k in mids if k <= off), default=min(mids))
        mid = mids[key]
        return pd.DataFrame([{"exchange_ts": ts_ns, "bid_px": mid - spread / 2, "ask_px": mid + spread / 2}])

    return reader


class TestDeltaScaledOptionMidBenchmark:
    """SHR-146 fix: the default benchmark is the leg's own book mid (delta-scaled).

    The raw perp ``mark`` (~$60k) × option size blew the split to ±$35,970 on
    #1000465. Sampling the option's own mid keeps delay/impact in option-price
    space, so the components are option-scale and still sum to the true gap.
    """

    def test_default_benchmark_is_option_mid_not_blown_up(self) -> None:
        # Held leg like #1000465: live enters later/higher than sim; mid drifts up.
        live = pd.DataFrame([_fill(1000, "buy", 0.9435, 529), _fill(5000, "sell", 0.9806, 529)])
        sim = pd.DataFrame([_fill(0, "buy", 0.9346, 529), _fill(5000, "sell", 0.9740, 529)])
        # Option mid drifts from 0.93 (sim entry) to 0.94 (live entry), ~flat at exit.
        mids = {0: 0.9300, 1000: 0.9400, 5000: 0.9800}
        res = reconcile_pnl(
            live_fills=live,
            sim_fills=sim,
            live_settlement={},
            sim_resolved={},
            book_reader=_book_reader(mids),
        )
        wf = res.waterfall
        # Entry delay = (mid@sim_entry − mid@live_entry) × size = (0.93 − 0.94)×529.
        assert abs(wf["matched_entry_delay"] - (0.9300 - 0.9400) * 529) < 1e-6, wf
        # Everything stays option-scale (single digits), NOT ~$35,970.
        for k in ("matched_entry_delay", "matched_entry_impact", "matched_exit_delay", "matched_exit_impact"):
            assert abs(wf[k]) < 50.0, (k, wf[k])
        # Split is still exact: delay+impact == the true matched VWAP gaps.
        assert abs((wf["matched_entry_delay"] + wf["matched_entry_impact"]) - (0.9346 - 0.9435) * 529) < 1e-6
        assert abs((wf["matched_exit_delay"] + wf["matched_exit_impact"]) - (0.9806 - 0.9740) * 529) < 1e-6
        assert abs(sum(wf.values()) - res.pnl_diff) < 1e-4


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
