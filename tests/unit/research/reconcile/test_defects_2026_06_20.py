"""Regression tests for the four reconcile defects found on #1000465 (2026-06-20).

A - sim PnL was $0 because episode side compared against uppercase literals while
    real venue fills use lowercase 'buy'/'sell'.
B - episode matcher mis-paired legs: greedy nearest-start with a 300s hard cap and
    no leg-sequence awareness collapsed when execution timing drifts.
C - Layer-1 "100% match" was bucket-first() over a ~hold-only trace, never
    comparing the handful of enter/exit decisions that actually matter.
D - no detector for gaps in the recorded reference feed (the real cause of the
    #1000465 divergence: a ~20 min hole in the recorded perp mark).
"""

from __future__ import annotations

import pandas as pd

from hlanalysis.research.reconcile.reconcile import (
    _realized_from_fills,
    check_reference_coverage,
    reconcile_decisions,
    reconcile_fills,
    reconcile_pnl,
)

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
    }


# ── A: lowercase side must not zero out sim PnL ──────────────────────────────


class TestDefectA_LowercaseSide:
    def test_sim_realized_computed_for_lowercase_sides(self) -> None:
        """Real venue fills use lowercase 'buy'/'sell'; sim PnL must still compute."""
        sim = pd.DataFrame(
            [
                _fill(0, "buy", 0.90, 100),
                _fill(60, "sell", 0.95, 100),
            ]
        )
        res = reconcile_pnl(
            live_fills=sim.copy(),
            sim_fills=sim,
            live_settlement={},
            sim_resolved={},
        )
        # round-trip = (0.95 - 0.90) * 100 = 5.0, no fees
        assert abs(res.sim_realized - 5.0) < 1e-6, res.sim_realized

    def test_mixed_case_equivalent(self) -> None:
        """Upper- and lower-case fills produce the same sim PnL."""
        lower = pd.DataFrame([_fill(0, "buy", 0.90, 100), _fill(60, "sell", 0.95, 100)])
        upper = pd.DataFrame([_fill(0, "BUY", 0.90, 100), _fill(60, "SELL", 0.95, 100)])
        r_lo = reconcile_pnl(live_fills=lower.copy(), sim_fills=lower, live_settlement={}, sim_resolved={})
        r_hi = reconcile_pnl(live_fills=upper.copy(), sim_fills=upper, live_settlement={}, sim_resolved={})
        assert abs(r_lo.sim_realized - r_hi.sim_realized) < 1e-9


# ── B: matcher must pair by leg sequence and surface unmatched round-trips ────


class TestDefectB_EpisodeMatcher:
    def _drifted_fills(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # live: RT1 (buy@0, sell@200) + held leg (buy@1000, sell@5000)
        live = pd.DataFrame(
            [
                _fill(0, "buy", 0.90, 100),
                _fill(200, "sell", 0.92, 100),
                _fill(1000, "buy", 0.94, 100),
                _fill(5000, "sell", 0.98, 100),
            ]
        )
        # sim: held leg (buy@2000, sell@5000) + RT2 (buy@8000, sell@11000)
        sim = pd.DataFrame(
            [
                _fill(2000, "buy", 0.94, 100),
                _fill(5000, "sell", 0.98, 100),
                _fill(8000, "buy", 0.97, 100),
                _fill(11000, "sell", 0.99, 100),
            ]
        )
        return live, sim

    def test_shared_exit_matches_extra_roundtrips_unmatched(self) -> None:
        """The synchronized exit pairs; live RT1 and sim RT2 stay unmatched."""
        live, sim = self._drifted_fills()
        res = reconcile_fills(live_fills=live, sim_fills=sim, max_match_gap_seconds=1800.0)
        et = res.episode_table
        matched = et[et["match_status"] == "matched"]
        live_only = et[et["match_status"] == "live_only"]
        sim_only = et[et["match_status"] == "sim_only"]
        # held buy (1000<->2000, gap 1000s) + shared sell (5000<->5000) match
        assert len(matched) == 2, et
        # live RT1 (buy@0, sell@200) has no sim counterpart within cap
        assert len(live_only) == 2, et
        # sim RT2 (buy@8000, sell@11000) has no live counterpart within cap
        assert len(sim_only) == 2, et

    def test_synchronized_exit_has_near_zero_latency(self) -> None:
        live, sim = self._drifted_fills()
        res = reconcile_fills(live_fills=live, sim_fills=sim, max_match_gap_seconds=1800.0)
        et = res.episode_table
        sell_match = et[(et["match_status"] == "matched") & (et["live_side"] == "SELL")]
        assert len(sell_match) == 1
        assert abs(float(sell_match.iloc[0]["latency_ns"])) < _1S_NS

    def test_no_300s_cliff(self) -> None:
        """A 6-min execution drift must still match (the old 300s cap dropped it)."""
        live = pd.DataFrame([_fill(0, "buy", 0.90, 100), _fill(100, "sell", 0.95, 100)])
        sim = pd.DataFrame([_fill(400, "buy", 0.90, 100), _fill(500, "sell", 0.95, 100)])
        res = reconcile_fills(live_fills=live, sim_fills=sim, max_match_gap_seconds=1800.0)
        matched = res.episode_table[res.episode_table["match_status"] == "matched"]
        assert len(matched) == 2


# ── Waterfall rebuilt on matched structure (was positional, meaningless) ─────


class TestWaterfallMatchedStructure:
    def test_unmatched_roundtrips_not_dumped_in_residual(self) -> None:
        """The #1000465 shape: shared held leg + a live-only RT + a sim-only RT.

        Components must be small and interpretable, NOT four huge numbers that
        cancel by luck with a giant residual.
        """
        # live: RT1 (buy@0 0.918, sell@200 0.928, sz 543) + held (buy@1000 0.9435,
        #       sell@5000 0.9806, sz 529); closed_pnl carries the venue truth.
        live = pd.DataFrame(
            [
                {**_fill(0, "buy", 0.918, 543), "closed_pnl": 0.0},
                {**_fill(200, "sell", 0.928, 543), "closed_pnl": (0.928 - 0.918) * 543},
                {**_fill(1000, "buy", 0.9435, 529), "closed_pnl": 0.0},
                {**_fill(5000, "sell", 0.9806, 529), "closed_pnl": (0.9806 - 0.9435) * 529},
            ]
        )
        # sim: held (buy@2000 0.9346, sell@5000 0.974, sz 534.75) + RT2 (buy@8000
        #      0.972, sell@11000 0.98, sz 514.4)
        sim = pd.DataFrame(
            [
                _fill(2000, "buy", 0.9346, 534.75),
                _fill(5000, "sell", 0.974, 534.75),
                _fill(8000, "buy", 0.972, 514.4),
                _fill(11000, "sell", 0.98, 514.4),
            ]
        )
        res = reconcile_pnl(live_fills=live, sim_fills=sim, live_settlement={}, sim_resolved={})
        wf = res.waterfall
        # Sums to pnl_diff.
        assert abs(sum(wf.values()) - res.pnl_diff) < 1e-4
        # The live-only round-trip (~$5.6) and sim-only (~$4.1) are now explicit,
        # not buried in residual.
        assert abs(wf["live_only_roundtrips"] - (0.928 - 0.918) * 543) < 0.5
        assert abs(wf["sim_only_roundtrips"] - (-(0.98 - 0.972) * 514.4)) < 0.5
        # Residual is small (no giant cancelling components).
        assert abs(wf["residual"]) < 2.0, wf
        # The matched-leg vwap components are modest (single-leg scale, not ~$24).
        # With no reference reader the split puts the whole gap into impact.
        assert abs(wf["matched_entry_delay"] + wf["matched_entry_impact"]) < 10.0
        assert abs(wf["matched_exit_delay"] + wf["matched_exit_impact"]) < 10.0


class TestRealizedAvgCost:
    def test_scale_in_then_partial_close(self) -> None:
        """Two buys then one partial sell: realized via running average cost."""
        fills = pd.DataFrame(
            [
                _fill(0, "buy", 0.80, 100),
                _fill(10, "buy", 0.90, 100),  # avg cost now 0.85
                _fill(20, "sell", 0.95, 100),  # realize (0.95-0.85)*100 = 10
            ]
        )
        assert abs(_realized_from_fills(fills) - 10.0) < 1e-6

    def test_simple_round_trip_matches_naive(self) -> None:
        fills = pd.DataFrame([_fill(0, "buy", 0.80, 10), _fill(60, "sell", 0.85, 10)])
        assert abs(_realized_from_fills(fills) - 0.5) < 1e-9


# ── C: Layer-1 must not let holds mask enter/exit divergence ─────────────────


def _trace_row(ts_offset_s: int, action: str, reason: str = "hold") -> dict:
    return {
        "ts_ns": _T0 + ts_offset_s * _1S_NS,
        "question_idx": 1000465,
        "klass": "priceBinary",
        "action": action,
        "reason": reason,
        "sigma": None,
        "p_model": None,
        "edge": None,
        "config_hash": "abc",
    }


class TestDefectC_DecisionMasking:
    def test_nonhold_action_surfaced_in_bucket(self) -> None:
        """An enter in a minute dominated by holds must not be masked to a match."""
        # Same 1-min bucket: live holds then enters; sim only holds.
        live = pd.DataFrame([_trace_row(0, "hold"), _trace_row(30, "enter", "entry")])
        sim = pd.DataFrame([_trace_row(0, "hold"), _trace_row(30, "hold")])
        res = reconcile_decisions(live_trace=live, sim_trace=sim)
        assert res.match_rate < 1.0, "live 'enter' was masked by bucket-first hold"

    def test_event_counts_reported(self) -> None:
        """Decision result exposes the count of non-hold events per side."""
        live = pd.DataFrame(
            [_trace_row(0, "hold"), _trace_row(30, "enter", "entry"), _trace_row(90, "exit", "exit_edge")]
        )
        sim = pd.DataFrame([_trace_row(0, "hold"), _trace_row(30, "hold"), _trace_row(90, "hold")])
        res = reconcile_decisions(live_trace=live, sim_trace=sim)
        assert res.n_live_events == 2
        assert res.n_sim_events == 0


# ── Audit fixes: H1 (numeric overlay), H2 (p_model), H3 (coverage), M2/M3 ────


def _trace_row_num(ts_offset_s: int, action: str, sigma=None, p_model=None, edge=None) -> dict:
    return {
        "ts_ns": _T0 + ts_offset_s * _1S_NS,
        "question_idx": 1000465,
        "klass": "priceBinary",
        "action": action,
        "reason": "hold",
        "sigma": sigma,
        "p_model": p_model,
        "edge": edge,
        "config_hash": "abc",
    }


class TestAuditH1_NumericOverlay:
    def test_sigma_compared_even_when_action_row_is_null(self) -> None:
        """σ lives on hold/scan rows; selecting the enter row must not null it out."""
        # Same minute: hold(sigma populated) + enter(sigma None). Sim differs in σ.
        live = pd.DataFrame([_trace_row_num(0, "hold", sigma=0.50), _trace_row_num(30, "enter", sigma=None)])
        sim = pd.DataFrame([_trace_row_num(0, "hold", sigma=0.60), _trace_row_num(30, "enter", sigma=None)])
        res = reconcile_decisions(live_trace=live, sim_trace=sim)
        # σ 0.50 vs 0.60 is a >5% relative divergence and must be detected.
        assert res.first_divergence is not None
        assert res.first_divergence.field == "sigma"


class TestAuditH2_PModel:
    def test_p_model_divergence_detected(self) -> None:
        """p_model was never tested in first_divergence (audit H2)."""
        live = pd.DataFrame([_trace_row_num(0, "hold", p_model=0.50)])
        sim = pd.DataFrame([_trace_row_num(0, "hold", p_model=0.90)])
        res = reconcile_decisions(live_trace=live, sim_trace=sim)
        assert res.first_divergence is not None
        assert res.first_divergence.field == "p_model"


class TestAuditH3_Coverage:
    def test_truncated_sim_fails_coverage_gate(self) -> None:
        """A sim trace covering only a sliver of live must not PASS on match rate."""
        from hlanalysis.research.reconcile.reconcile import (
            DecisionResult,
            FillsResult,
            PnLResult,
            PreconditionResult,
            verdict,
        )

        l0 = PreconditionResult("PASS", "PASS", "PASS", "PASS")
        # 100% match but over only 5 of 1400 buckets.
        l1 = DecisionResult(1.0, None, pd.DataFrame(), "match", 1400, 5, 5, 0, 0)
        l2 = FillsResult([], [], pd.DataFrame(), None, "match", 0, 0)
        l3 = PnLResult(0.0, 0.0, 0.0, "SKIP:no_settlement", {}, "PASS")
        v, reasons = verdict(l0, l1, l2, l3)
        assert v == "FAIL"
        assert any("decision_coverage" in r for r in reasons)


class TestAuditM3_WindowOverlap:
    def test_tiny_sim_window_inside_live_fails(self) -> None:
        from hlanalysis.research.reconcile.reconcile import check_preconditions

        # live spans 24h; sim spans 2 minutes fully inside it.
        live = pd.DataFrame(
            [
                {"ts_ns": _T0, "question_idx": 1, "klass": "b", "config_hash": "h"},
                {"ts_ns": _T0 + 86400 * _1S_NS, "question_idx": 1, "klass": "b", "config_hash": "h"},
            ]
        )
        sim = pd.DataFrame(
            [
                {"ts_ns": _T0 + 1000 * _1S_NS, "question_idx": 1, "klass": "b", "config_hash": "h"},
                {"ts_ns": _T0 + 1120 * _1S_NS, "question_idx": 1, "klass": "b", "config_hash": "h"},
            ]
        )
        res = check_preconditions(live_trace=live, sim_trace=sim)
        assert res.window_match.startswith("FAIL")


class TestAuditM2_SettlementGuard:
    def test_deep_itm_trading_fill_not_misread_as_settlement(self) -> None:
        from hlanalysis.research.reconcile.reconcile import _winner_from_settlement_fill

        expiry = _T0 + 10_000 * _1S_NS
        # A legit 0.99 favorite buy BEFORE expiry, no settlement fill after.
        fills = pd.DataFrame([{**_fill(0, "buy", 0.99, 100), "symbol": "#4650"}])
        assert _winner_from_settlement_fill(fills, expiry_ns=expiry) is None

    def test_settlement_fill_at_expiry_read(self) -> None:
        from hlanalysis.research.reconcile.reconcile import _winner_from_settlement_fill

        expiry = _T0 + 10_000 * _1S_NS
        fills = pd.DataFrame(
            [
                {**_fill(0, "buy", 0.95, 100), "symbol": "#4650"},
                {**_fill(10_001, "sell", 1.00, 100), "symbol": "#4650"},  # settlement at expiry
            ]
        )
        # #4650 -> side_idx 0 -> yes leg; price 1.0 -> won -> "yes"
        assert _winner_from_settlement_fill(fills, expiry_ns=expiry) == "yes"


# ── D: reference-feed gap detector ───────────────────────────────────────────


class TestDefectD_ReferenceCoverage:
    def test_detects_gap(self) -> None:
        """A hole in the recorded reference timestamps is reported as a gap."""
        # ticks every 2s, then a 1239s hole, then resume
        ts = [_T0 + i * 2 * _1S_NS for i in range(5)]
        last = ts[-1]
        ts.append(last + 1239 * _1S_NS)
        ts += [ts[-1] + i * 2 * _1S_NS for i in range(1, 5)]

        def reader(symbol: str, start_ns: int, end_ns: int, data_root) -> list[int]:
            return ts

        gaps = check_reference_coverage(
            start_ns=ts[0],
            end_ns=ts[-1],
            ref_symbol="BTC",
            data_root=None,
            gap_threshold_seconds=60.0,
            ts_reader=reader,
        )
        assert len(gaps) == 1
        assert abs(gaps[0].gap_seconds - 1239.0) < 1.0

    def test_no_gap_when_dense(self) -> None:
        ts = [_T0 + i * 2 * _1S_NS for i in range(50)]

        def reader(symbol: str, start_ns: int, end_ns: int, data_root) -> list[int]:
            return ts

        gaps = check_reference_coverage(
            start_ns=ts[0],
            end_ns=ts[-1],
            ref_symbol="BTC",
            data_root=None,
            gap_threshold_seconds=60.0,
            ts_reader=reader,
        )
        assert gaps == []
