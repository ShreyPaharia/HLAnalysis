"""SHR-151: per-component invariant tolerances replace the single $5 PnL gate.

Each accounting component reconciles within its own tolerance ε (fees <1%,
turnover within N units, slippage within a band, settlement winner exact, realized
PnL ≤ $X). The verdict names the broken invariant instead of a vague total — and
must still FAIL real divergence (no silent loosening).
"""

from __future__ import annotations

import pandas as pd

from hlanalysis.research.reconcile.reconcile import (
    DecisionResult,
    FillsResult,
    InvariantTolerances,
    PnLResult,
    PreconditionResult,
    check_invariants,
    reconcile_pnl,
    verdict,
)

_T0 = 1_718_000_000_000_000_000
_1S_NS = 1_000_000_000


def _pnl(
    pnl_diff: float = 0.0,
    settlement: str = "SKIP:no_settlement",
    waterfall: dict | None = None,
    live_fees: float = 0.0,
    sim_fees: float = 0.0,
    live_size: float = 0.0,
    sim_size: float = 0.0,
) -> PnLResult:
    return PnLResult(
        live_realized=0.0,
        sim_realized=0.0,
        pnl_diff=pnl_diff,
        settlement_winner_match=settlement,
        waterfall=waterfall or {},
        pnl_match="PASS" if abs(pnl_diff) <= 5.0 else f"FAIL:{pnl_diff:+.2f}",
        live_fees=live_fees,
        sim_fees=sim_fees,
        live_size=live_size,
        sim_size=sim_size,
    )


def _named(invs, name: str):
    return next(i for i in invs if i.name == name)


class TestCheckInvariants:
    def test_all_within_tolerance_pass(self) -> None:
        invs = check_invariants(_pnl(), InvariantTolerances())
        assert all(i.passed for i in invs)

    def test_fee_invariant_breaks(self) -> None:
        # 5% fee divergence > 1% tolerance.
        p = _pnl(live_fees=1.00, sim_fees=1.05)
        invs = check_invariants(p, InvariantTolerances(fee_rel=0.01))
        assert not _named(invs, "fees").passed
        assert _named(invs, "realized_pnl").passed

    def test_fee_invariant_within_one_percent_passes(self) -> None:
        p = _pnl(live_fees=1.00, sim_fees=1.005)
        assert _named(check_invariants(p, InvariantTolerances(fee_rel=0.01)), "fees").passed

    def test_turnover_invariant_breaks_absolute(self) -> None:
        # Pure absolute mode (turnover_rel=0): 20-unit diff > 5-unit floor.
        p = _pnl(live_size=500.0, sim_size=520.0)
        invs = check_invariants(p, InvariantTolerances(turnover_units=5.0, turnover_rel=0.0))
        assert not _named(invs, "turnover").passed

    def test_turnover_relative_tolerance_passes_small_pct(self) -> None:
        # 45.7-unit diff on ~2100-unit turnover (~2%) clears the default 5% band.
        p = _pnl(live_size=2144.0, sim_size=2098.3)
        assert _named(check_invariants(p, InvariantTolerances()), "turnover").passed

    def test_turnover_relative_still_fails_real_divergence(self) -> None:
        # Sim traded 603 units, live traded nothing — must still FAIL.
        p = _pnl(live_size=0.0, sim_size=603.0)
        assert not _named(check_invariants(p, InvariantTolerances()), "turnover").passed

    def test_turnover_floor_for_tiny_markets(self) -> None:
        # 0.5-unit diff on a tiny 5-unit market is within the absolute floor.
        p = _pnl(live_size=5.0, sim_size=5.5)
        assert _named(check_invariants(p, InvariantTolerances()), "turnover").passed

    def test_settlement_winner_exact(self) -> None:
        p = _pnl(settlement="FAIL:live=yes sim=no")
        assert not _named(check_invariants(p, InvariantTolerances()), "settlement_winner").passed
        # SKIP and PASS both clear.
        assert _named(check_invariants(_pnl(settlement="PASS"), InvariantTolerances()), "settlement_winner").passed
        assert _named(check_invariants(_pnl(), InvariantTolerances()), "settlement_winner").passed

    def test_slippage_band(self) -> None:
        wf = {"matched_entry_impact": 4.0, "matched_exit_impact": 4.0}  # |8| > band 5
        p = _pnl(waterfall=wf)
        assert not _named(check_invariants(p, InvariantTolerances(slippage_abs=5.0)), "slippage").passed

    def test_slippage_robust_to_binary_delay_impact_artifact(self) -> None:
        """The delay/impact split blows up for binary markets (ref ~$63k × size).

        On #1000465 the entry split was delay=-35972 / impact=+35967 — each huge,
        cancelling to a true entry VWAP gap of ~-$4.68. The slippage invariant must
        measure the NET matched VWAP gap per leg, not the corrupted impact-only
        component, so it reports real slippage (~$8) rather than ~$35,970.
        """
        wf = {
            "matched_entry_delay": -35972.0,
            "matched_entry_impact": 35967.32,  # net entry gap ≈ -4.68
            "matched_exit_delay": 0.0,
            "matched_exit_impact": 3.49,  # net exit gap ≈ +3.49
        }
        p = _pnl(waterfall=wf)
        inv = _named(check_invariants(p, InvariantTolerances(slippage_abs=5.0)), "slippage")
        # Observed ≈ |−4.68| + |3.49| ≈ 8.17, NOT ~35,970.
        assert inv.observed < 100.0, inv.observed
        assert abs(inv.observed - 8.17) < 0.5, inv.observed
        assert not inv.passed  # 8.17 > 5 band → real (data-caused) slippage
        # A wider band clears it.
        assert _named(check_invariants(p, InvariantTolerances(slippage_abs=10.0)), "slippage").passed

    def test_realized_pnl_gate_unchanged(self) -> None:
        p = _pnl(pnl_diff=10.0)
        assert not _named(check_invariants(p, InvariantTolerances(realized_pnl_abs=5.0)), "realized_pnl").passed


class TestVerdictNamesInvariant:
    def _layers(self, l3: PnLResult):
        l0 = PreconditionResult("PASS", "PASS", "PASS", "PASS")
        l1 = DecisionResult(1.0, None, pd.DataFrame(), "match", 10, 10, 10, 0, 0)
        l2 = FillsResult([], [], pd.DataFrame(), None, "match", 0, 0)
        return l0, l1, l2

    def test_fee_break_named_in_reasons(self) -> None:
        l0, l1, l2 = self._layers(None)
        l3 = _pnl(live_fees=1.0, sim_fees=1.5)
        v, reasons = verdict(l0, l1, l2, l3)
        assert v == "FAIL"
        assert any("fees" in r for r in reasons)
        # The vague single total is gone; the specific invariant is named.
        assert all("pnl_diff:" not in r for r in reasons)

    def test_settlement_break_named(self) -> None:
        l0, l1, l2 = self._layers(None)
        l3 = _pnl(settlement="FAIL:live=yes sim=no")
        v, reasons = verdict(l0, l1, l2, l3)
        assert v == "FAIL"
        assert any("settlement_winner" in r for r in reasons)

    def test_clean_result_passes(self) -> None:
        l0, l1, l2 = self._layers(None)
        v, reasons = verdict(l0, l1, l2, _pnl())
        assert v == "PASS", reasons

    def test_realized_pnl_break_still_fails(self) -> None:
        l0, l1, l2 = self._layers(None)
        v, reasons = verdict(l0, l1, l2, _pnl(pnl_diff=12.0))
        assert v == "FAIL"
        assert any("realized_pnl" in r for r in reasons)


class TestReconcilePnlPopulatesComponents:
    def test_fees_and_sizes_exposed(self) -> None:
        live = pd.DataFrame(
            [
                {"ts_ns": _T0, "side": "buy", "price": 0.9, "size": 100, "fee": 0.10, "symbol": "#4650"},
                {
                    "ts_ns": _T0 + 60 * _1S_NS,
                    "side": "sell",
                    "price": 0.95,
                    "size": 100,
                    "fee": 0.11,
                    "symbol": "#4650",
                },
            ]
        )
        sim = pd.DataFrame(
            [
                {"ts_ns": _T0, "side": "buy", "price": 0.9, "size": 100, "fee": 0.09, "symbol": "#4650"},
                {
                    "ts_ns": _T0 + 60 * _1S_NS,
                    "side": "sell",
                    "price": 0.95,
                    "size": 100,
                    "fee": 0.10,
                    "symbol": "#4650",
                },
            ]
        )
        res = reconcile_pnl(live_fills=live, sim_fills=sim, live_settlement={}, sim_resolved={})
        assert abs(res.live_fees - 0.21) < 1e-9
        assert abs(res.sim_fees - 0.19) < 1e-9
        assert abs(res.live_size - 200.0) < 1e-9
        assert abs(res.sim_size - 200.0) < 1e-9
