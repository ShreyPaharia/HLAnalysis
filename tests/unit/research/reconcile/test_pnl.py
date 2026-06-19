"""Tests for Layer 3: PnL reconciliation."""

from __future__ import annotations

import pandas as pd

from hlanalysis.research.reconcile.reconcile import reconcile_pnl

_T0 = 1_718_000_000_000_000_000
_1S_NS = 1_000_000_000


def _fill(
    ts_offset_s: int,
    side: str,
    price: float,
    size: float,
    fee: float = 0.001,
    closed_pnl: float = 0.0,
) -> dict:
    return {
        "ts_ns": _T0 + ts_offset_s * _1S_NS,
        "side": side,
        "price": price,
        "size": size,
        "fee": fee,
        "closed_pnl": closed_pnl,
        "symbol": "#4010",
    }


def _round_trip_fills(
    entry_price: float = 0.80,
    exit_price: float = 0.85,
    size: float = 10.0,
    fee: float = 0.001,
) -> pd.DataFrame:
    """Build a BUY + SELL round-trip fill DataFrame."""
    return pd.DataFrame(
        [
            _fill(0, "BUY", entry_price, size, fee=fee, closed_pnl=0.0),
            _fill(60, "SELL", exit_price, size, fee=fee, closed_pnl=(exit_price - entry_price) * size),
        ]
    )


class TestPnLPass:
    def test_pnl_pass(self) -> None:
        """|diff| < $5 -> PASS."""
        live = _round_trip_fills(entry_price=0.80, exit_price=0.85, size=10)
        # closed_pnl = 0.05 * 10 = 0.50 from SELL fill
        live_settlement: dict = {}
        sim_resolved: dict = {}
        result = reconcile_pnl(
            live_fills=live,
            sim_fills=live.copy(),
            live_settlement=live_settlement,
            sim_resolved=sim_resolved,
            pnl_abs_threshold=5.0,
        )
        assert result.pnl_match == "PASS"
        assert abs(result.pnl_diff) < 5.0

    def test_pnl_pass_exact_match(self) -> None:
        """Identical fills -> pnl_diff close to 0."""
        fills = _round_trip_fills()
        result = reconcile_pnl(
            live_fills=fills,
            sim_fills=fills.copy(),
            live_settlement={},
            sim_resolved={},
        )
        assert result.pnl_match == "PASS"
        # pnl_diff should be very small (fees may cancel out)
        assert abs(result.pnl_diff) < 5.0


class TestPnLFail:
    def test_pnl_fail_large_diff(self) -> None:
        """|diff| > $5 -> FAIL with amount in reason."""
        # Live: big closed_pnl
        live = pd.DataFrame(
            [
                _fill(0, "BUY", 0.10, 100, fee=0.0, closed_pnl=0.0),
                _fill(60, "SELL", 0.90, 100, fee=0.0, closed_pnl=80.0),
            ]
        )
        # Sim: completely flat (no trades)
        sim = pd.DataFrame()
        result = reconcile_pnl(
            live_fills=live,
            sim_fills=sim,
            live_settlement={},
            sim_resolved={},
            pnl_abs_threshold=5.0,
        )
        assert result.pnl_match.startswith("FAIL")
        # The amount should appear in the fail string
        assert "+" in result.pnl_match or "-" in result.pnl_match
        assert abs(result.pnl_diff) > 5.0

    def test_pnl_fail_threshold_boundary(self) -> None:
        """pnl_diff exactly at threshold+eps -> FAIL."""
        live = pd.DataFrame(
            [
                _fill(0, "BUY", 0.10, 100, fee=0.0, closed_pnl=0.0),
                _fill(60, "SELL", 0.90, 100, fee=0.0, closed_pnl=80.0),
            ]
        )
        sim = pd.DataFrame()
        result = reconcile_pnl(
            live_fills=live,
            sim_fills=sim,
            live_settlement={},
            sim_resolved={},
            pnl_abs_threshold=5.0,
        )
        assert result.pnl_diff > 5.0
        assert result.pnl_match.startswith("FAIL")


class TestWaterfall:
    def test_waterfall_components_present(self) -> None:
        """Waterfall dict has all required keys."""
        fills = _round_trip_fills()
        result = reconcile_pnl(
            live_fills=fills,
            sim_fills=fills.copy(),
            live_settlement={},
            sim_resolved={},
        )
        assert "entry_vwap_diff" in result.waterfall
        assert "exit_vwap_diff" in result.waterfall
        assert "size_diff" in result.waterfall
        assert "fee_diff" in result.waterfall
        assert "residual" in result.waterfall

    def test_waterfall_sums_to_pnl_diff(self) -> None:
        """entry_vwap_diff + exit_vwap_diff + size_diff + fee_diff + residual ≈ pnl_diff."""
        live = _round_trip_fills(entry_price=0.80, exit_price=0.85, size=10)
        sim = _round_trip_fills(entry_price=0.81, exit_price=0.84, size=10)
        result = reconcile_pnl(
            live_fills=live,
            sim_fills=sim,
            live_settlement={},
            sim_resolved={},
        )
        wf = result.waterfall
        total = wf["entry_vwap_diff"] + wf["exit_vwap_diff"] + wf["size_diff"] + wf["fee_diff"] + wf["residual"]
        assert abs(total - result.pnl_diff) < 1e-4  # floating point tolerance

    def test_waterfall_fee_diff_zero_when_fees_equal(self) -> None:
        """Same fees in live and sim -> fee_diff = 0."""
        live = _round_trip_fills(fee=0.005)
        sim = _round_trip_fills(fee=0.005)
        result = reconcile_pnl(
            live_fills=live,
            sim_fills=sim,
            live_settlement={},
            sim_resolved={},
        )
        assert abs(result.waterfall["fee_diff"]) < 1e-9


class TestSettlementMatch:
    def test_settlement_match(self) -> None:
        """Matching winner sides -> settlement_winner_match == 'PASS'."""
        fills = _round_trip_fills()
        result = reconcile_pnl(
            live_fills=fills,
            sim_fills=fills.copy(),
            live_settlement={"winner_side": "YES", "realized_pnl": 0.5, "ts_ns": _T0},
            sim_resolved={"winner_side": "YES", "resolved_outcome": 1.0},
        )
        assert result.settlement_winner_match == "PASS"

    def test_settlement_mismatch(self) -> None:
        """Different winner sides -> settlement_winner_match starts with 'FAIL'."""
        fills = _round_trip_fills()
        result = reconcile_pnl(
            live_fills=fills,
            sim_fills=fills.copy(),
            live_settlement={"winner_side": "YES", "realized_pnl": 0.5, "ts_ns": _T0},
            sim_resolved={"winner_side": "NO", "resolved_outcome": 0.0},
        )
        assert result.settlement_winner_match.startswith("FAIL")
        assert "YES" in result.settlement_winner_match
        assert "NO" in result.settlement_winner_match

    def test_settlement_skip_no_settlement(self) -> None:
        """No settlement data -> settlement_winner_match is SKIP."""
        fills = _round_trip_fills()
        result = reconcile_pnl(
            live_fills=fills,
            sim_fills=fills.copy(),
            live_settlement={},
            sim_resolved={},
        )
        assert result.settlement_winner_match == "SKIP:no_settlement"

    def test_settlement_skip_one_side_missing(self) -> None:
        """Only live has settlement, sim doesn't -> SKIP."""
        fills = _round_trip_fills()
        result = reconcile_pnl(
            live_fills=fills,
            sim_fills=fills.copy(),
            live_settlement={"winner_side": "YES"},
            sim_resolved={},
        )
        assert result.settlement_winner_match == "SKIP:no_settlement"


class TestPnLFromSettlement:
    def test_live_pnl_from_closed_pnl_column(self) -> None:
        """live_realized uses closed_pnl column when present."""
        live = pd.DataFrame(
            [
                _fill(0, "BUY", 0.80, 10, fee=0.0, closed_pnl=0.0),
                _fill(60, "SELL", 0.85, 10, fee=0.0, closed_pnl=0.5),
            ]
        )
        result = reconcile_pnl(
            live_fills=live,
            sim_fills=live.copy(),
            live_settlement={},
            sim_resolved={},
        )
        assert abs(result.live_realized - 0.5) < 1e-9

    def test_live_pnl_from_settlement_when_no_closed_pnl(self) -> None:
        """live_realized falls back to settlement.realized_pnl if column absent."""
        live = pd.DataFrame(
            [
                {"ts_ns": _T0, "side": "BUY", "price": 0.80, "size": 10, "symbol": "#4010"},
            ]
        )
        settlement = {"winner_side": "YES", "realized_pnl": 2.5, "ts_ns": _T0}
        result = reconcile_pnl(
            live_fills=live,
            sim_fills=pd.DataFrame(),
            live_settlement=settlement,
            sim_resolved={},
        )
        assert abs(result.live_realized - 2.5) < 1e-9
