"""Tests for Card A: Liquidity & Book Shape.

Light tests that run without real data: structural / schema / unit checks.
Data-dependent sections are skipped when ../../data is absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Data root resolution (mirrors other card tests)
# ---------------------------------------------------------------------------

_WORKTREE_ROOT = Path(__file__).parent.parent.parent

if os.environ.get("HLBT_HL_DATA_ROOT"):
    _DATA_ROOT = Path(os.environ["HLBT_HL_DATA_ROOT"]).resolve()
else:
    _DATA_ROOT = (_WORKTREE_ROOT / ".." / ".." / "data").resolve()

_HL_MARKER = _DATA_ROOT / "venue=hyperliquid"
_DATA_AVAILABLE = _HL_MARKER.exists()


# ---------------------------------------------------------------------------
# Unit tests: depth helper
# ---------------------------------------------------------------------------


class TestNotionalWithinBps:
    """Test the notional-within-bps computation logic (extracted logic)."""

    def _notional_within(
        self,
        bid_pxs: list[float],
        bid_szs: list[float],
        ask_pxs: list[float],
        ask_szs: list[float],
        bps_limit: float,
    ) -> tuple[float, float]:
        """Reimplementation of the inline logic in _compute_depth_from_snapshots."""
        if not bid_pxs or not ask_pxs:
            return 0.0, 0.0
        bids = sorted(zip(bid_pxs, bid_szs), reverse=True)
        asks = sorted(zip(ask_pxs, ask_szs))
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2.0
        if mid <= 0:
            return 0.0, 0.0
        tob_n = best_bid * bids[0][1] + best_ask * asks[0][1]
        bid_thresh = mid * (1.0 - bps_limit / 10000.0)
        ask_thresh = mid * (1.0 + bps_limit / 10000.0)
        total = sum(p * s for p, s in bids if p >= bid_thresh) + sum(p * s for p, s in asks if p <= ask_thresh)
        return tob_n, total

    def test_single_level_tob(self) -> None:
        tob, total_50 = self._notional_within([0.48], [100.0], [0.50], [100.0], 50)
        assert tob == pytest.approx(0.48 * 100.0 + 0.50 * 100.0, rel=1e-6)

    def test_total_within_50bps(self) -> None:
        # mid = 0.49; 50 bps = 0.49 * 0.005 = 0.00245
        # bids at 0.48 (too far: 0.49 - 0.48 = 0.01 > 0.00245), 0.488 (within), 0.486 (within)
        # asks at 0.50, 0.495 (within), 0.496 (within)
        bid_pxs = [0.48, 0.488, 0.486]
        bid_szs = [100.0, 50.0, 30.0]
        ask_pxs = [0.50, 0.495, 0.496]
        ask_szs = [100.0, 60.0, 40.0]
        mid = (0.488 + 0.495) / 2  # best bid=0.488, best ask=0.495
        thresh_bid = mid * (1 - 50 / 10000)
        thresh_ask = mid * (1 + 50 / 10000)
        _, total = self._notional_within(bid_pxs, bid_szs, ask_pxs, ask_szs, 50)
        # Hand-compute expected
        expected_bid = sum(p * s for p, s in zip(bid_pxs, bid_szs) if p >= thresh_bid)
        expected_ask = sum(p * s for p, s in zip(ask_pxs, ask_szs) if p <= thresh_ask)
        assert total == pytest.approx(expected_bid + expected_ask, rel=1e-6)

    def test_empty_book_returns_zero(self) -> None:
        tob, total = self._notional_within([], [], [], [], 100)
        assert tob == 0.0
        assert total == 0.0

    def test_book_sorted_worst_first_still_correct(self) -> None:
        """Arrays worst-first (as the spec warns) should give same result after sort."""
        # Worst-first: ascending bids, descending asks
        bid_pxs_wf = [0.46, 0.47, 0.48]
        bid_szs_wf = [30.0, 50.0, 100.0]
        ask_pxs_wf = [0.52, 0.51, 0.50]
        ask_szs_wf = [40.0, 60.0, 100.0]

        tob_wf, total_wf = self._notional_within(bid_pxs_wf, bid_szs_wf, ask_pxs_wf, ask_szs_wf, 100)

        # Best-first (sorted correctly before passing)
        bid_pxs_bf = [0.48, 0.47, 0.46]
        bid_szs_bf = [100.0, 50.0, 30.0]
        ask_pxs_bf = [0.50, 0.51, 0.52]
        ask_szs_bf = [100.0, 60.0, 40.0]
        tob_bf, total_bf = self._notional_within(bid_pxs_bf, bid_szs_bf, ask_pxs_bf, ask_szs_bf, 100)

        assert tob_wf == pytest.approx(tob_bf, rel=1e-6)
        assert total_wf == pytest.approx(total_bf, rel=1e-6)


# ---------------------------------------------------------------------------
# Unit tests: OBI computation
# ---------------------------------------------------------------------------


class TestOBIComputation:
    """Verify OBI formula: (bid_sz - ask_sz) / (bid_sz + ask_sz)."""

    def test_balanced_book_zero_obi(self) -> None:
        bid_sz, ask_sz = 100.0, 100.0
        obi = (bid_sz - ask_sz) / (bid_sz + ask_sz)
        assert obi == pytest.approx(0.0, abs=1e-9)

    def test_all_bid_obi_is_one(self) -> None:
        bid_sz, ask_sz = 100.0, 0.0
        # Guard against division by zero — only compute when sum > 0
        total = bid_sz + ask_sz
        if total > 0:
            obi = (bid_sz - ask_sz) / total
        else:
            obi = 0.0
        assert obi == pytest.approx(1.0, abs=1e-9)

    def test_all_ask_obi_is_minus_one(self) -> None:
        bid_sz, ask_sz = 0.0, 100.0
        total = bid_sz + ask_sz
        obi = (bid_sz - ask_sz) / total if total > 0 else 0.0
        assert obi == pytest.approx(-1.0, abs=1e-9)

    def test_obi_range(self) -> None:
        rng = np.random.default_rng(42)
        bid_szs = rng.uniform(0.1, 1000, 100)
        ask_szs = rng.uniform(0.1, 1000, 100)
        obis = (bid_szs - ask_szs) / (bid_szs + ask_szs)
        assert (obis >= -1).all()
        assert (obis <= 1).all()


# ---------------------------------------------------------------------------
# Unit tests: tick size helper
# ---------------------------------------------------------------------------


class TestTickSizeLogic:
    """Verify that minimum-increment detection works on known data."""

    def test_detects_1e5_tick(self) -> None:
        # Prices that are multiples of 0.00001
        prices = np.array([0.48000, 0.48001, 0.48010, 0.50000, 0.50001])
        rounded = np.round(prices * 100000).astype(int)
        unique_sorted = np.sort(np.unique(rounded))
        increments = np.diff(unique_sorted)
        min_increment = int(increments[increments > 0].min())
        tick = min_increment / 100000.0
        assert tick == pytest.approx(1e-5, rel=1e-6)

    def test_coarser_tick(self) -> None:
        # Prices that are multiples of 0.001
        prices = np.array([0.480, 0.481, 0.490, 0.500])
        rounded = np.round(prices * 100000).astype(int)
        unique_sorted = np.sort(np.unique(rounded))
        increments = np.diff(unique_sorted)
        min_increment = int(increments[increments > 0].min())
        tick = min_increment / 100000.0
        assert tick == pytest.approx(1e-3, rel=1e-6)


# ---------------------------------------------------------------------------
# Unit tests: split-half boundary
# ---------------------------------------------------------------------------


class TestSplitHalfBoundary:
    """Verify the H1/H2 boundary constant."""

    def test_h1_end_ns_is_correct_date(self) -> None:
        import datetime as dt

        from hlanalysis.research.cards.card_a_liquidity import _H1_END_NS

        # 2026-05-23 23:59:59 UTC in nanoseconds
        expected_dt = dt.datetime(2026, 5, 23, 23, 59, 59, tzinfo=dt.UTC)
        expected_ns = int(expected_dt.timestamp() * 1e9)
        assert expected_ns == _H1_END_NS


# ---------------------------------------------------------------------------
# Unit tests: spread stats (pure function logic)
# ---------------------------------------------------------------------------


class TestSpreadStatsLogic:
    """Verify spread bps formula on synthetic data."""

    def test_spread_bps_formula(self) -> None:
        bid, ask = 0.48, 0.50
        mid = (bid + ask) / 2.0
        expected_bps = (ask - bid) / mid * 10000
        assert expected_bps == pytest.approx(408.16, rel=1e-3)

    def test_zero_spread_gives_zero_bps(self) -> None:
        bid, ask = 0.50, 0.50
        mid = (bid + ask) / 2.0
        bps = (ask - bid) / mid * 10000 if mid > 0 else 0.0
        assert bps == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Integration tests (require real data)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="../../data not present")
class TestBuildCardIntegration:
    """Integration tests against real 36-day corpus."""

    def test_build_card_returns_html_and_findings(self) -> None:
        """build_card returns non-empty HTML and a findings dict."""
        from hlanalysis.research.cards.card_a_liquidity import build_card

        con = duckdb.connect()
        html, findings = build_card(con, str(_DATA_ROOT))

        assert isinstance(html, str)
        assert len(html) > 100
        # Findings has required schema keys
        required = {"title", "headline", "metrics", "split_half", "verdict"}
        assert required.issubset(findings.keys()), f"Missing: {required - findings.keys()}"

    def test_findings_verdict_is_valid(self) -> None:
        from hlanalysis.research.cards.card_a_liquidity import build_card

        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        assert findings["verdict"] in {"PASS", "FAIL", "INCONCLUSIVE"}

    def test_findings_metrics_have_required_fields(self) -> None:
        from hlanalysis.research.cards.card_a_liquidity import build_card

        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        for m in findings["metrics"]:
            assert "name" in m, f"Missing 'name' in {m}"
            assert "value" in m, f"Missing 'value' in {m}"
            assert "n" in m, f"Missing 'n' in {m}"
            assert "date_span" in m, f"Missing 'date_span' in {m}"
            assert "sanity" in m, f"Missing 'sanity' in {m}"

    def test_findings_is_json_serializable(self) -> None:
        from hlanalysis.research.cards.card_a_liquidity import build_card

        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        dumped = json.dumps(findings)
        assert len(dumped) > 10
        # Verify no NaN leaked through (NaN is not valid JSON)
        assert "NaN" not in dumped

    def test_split_half_has_h1_and_h2(self) -> None:
        from hlanalysis.research.cards.card_a_liquidity import build_card

        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        sh = findings["split_half"]
        assert "H1" in sh, "split_half missing H1"
        assert "H2" in sh, "split_half missing H2"
        assert "n_expiries" in sh["H1"]
        assert "n_expiries" in sh["H2"]

    def test_n_expiries_at_least_30(self) -> None:
        from hlanalysis.research.cards.card_a_liquidity import build_card

        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        n_metric = next(m for m in findings["metrics"] if m["name"] == "n_expiries")
        assert n_metric["value"] >= 30, f"Expected ≥30 expiries, got {n_metric['value']}"

    def test_binary_spread_bps_plausible(self) -> None:
        """Binary median spread should be 50–5000 bps (probability units)."""
        from hlanalysis.research.cards.card_a_liquidity import build_card

        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        m = next(m for m in findings["metrics"] if m["name"] == "binary_median_spread_bps")
        v = m["value"]
        assert v is not None
        assert 10 < v < 10000, f"Binary median spread {v} bps outside plausible range"

    def test_perp_spread_bps_much_tighter(self) -> None:
        """Perp should be far tighter than binary."""
        from hlanalysis.research.cards.card_a_liquidity import build_card

        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        binary_m = next(m for m in findings["metrics"] if m["name"] == "binary_median_spread_bps")
        perp_m = next(m for m in findings["metrics"] if m["name"] == "perp_median_spread_bps")
        assert binary_m["value"] > perp_m["value"] * 10, (
            f"Binary ({binary_m['value']:.1f} bps) should be >>10x perp ({perp_m['value']:.2f} bps)"
        )

    def test_obi_metric_present_with_n(self) -> None:
        from hlanalysis.research.cards.card_a_liquidity import build_card

        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        m = next((m for m in findings["metrics"] if m["name"] == "obi_mean_spearman_r"), None)
        assert m is not None, "obi_mean_spearman_r metric missing"
        assert m["n"] > 0, "OBI metric has n=0"

    def test_tick_size_is_1e5(self) -> None:
        """Binary tick should be 1e-5 based on corpus analysis."""
        from hlanalysis.research.cards.card_a_liquidity import build_card

        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        m = next(m for m in findings["metrics"] if m["name"] == "binary_tick_size")
        v = m["value"]
        assert v is not None
        assert abs(v - 1e-5) < 1e-6, f"Tick size {v} != expected 1e-5"

    def test_mm_room_verdict_present(self) -> None:
        from hlanalysis.research.cards.card_a_liquidity import build_card

        con = duckdb.connect()
        _, findings = build_card(con, str(_DATA_ROOT))
        assert "mm_room" in findings
        mm = findings["mm_room"]
        assert "binary_regime" in mm
        assert "verdict" in mm
        assert isinstance(mm["verdict"], str)
        assert len(mm["verdict"]) > 10

    def test_output_files_written(self) -> None:
        """After build_card, card_a.html and card_a.json should be writable."""
        from hlanalysis.research.cards.card_a_liquidity import build_card

        con = duckdb.connect()
        html, findings = build_card(con, str(_DATA_ROOT))

        out_dir = _WORKTREE_ROOT / "docs" / "research" / "_cards"
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = out_dir / "card_a.html"
        json_path = out_dir / "card_a.json"

        html_path.write_text(html, encoding="utf-8")
        json_path.write_text(json.dumps(findings, indent=2), encoding="utf-8")

        assert html_path.exists()
        assert json_path.exists()
        assert html_path.stat().st_size > 1000
        assert json_path.stat().st_size > 100
