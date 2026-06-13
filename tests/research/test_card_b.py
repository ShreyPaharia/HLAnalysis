"""Tests for Card B: Adverse Selection / Trade Markouts.

Light unit tests run without real data (structural / math checks).
Integration tests are skipped when ../../data is absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Data root resolution (mirrors test_card_e.py pattern)
# ---------------------------------------------------------------------------

_WORKTREE_ROOT = Path(__file__).parent.parent.parent

if os.environ.get("HLBT_HL_DATA_ROOT"):
    _DATA_ROOT = Path(os.environ["HLBT_HL_DATA_ROOT"]).resolve()
else:
    _DATA_ROOT = (_WORKTREE_ROOT / ".." / ".." / "data").resolve()

_HL_MARKER = _DATA_ROOT / "venue=hyperliquid"
_DATA_AVAILABLE = _HL_MARKER.exists()


# ---------------------------------------------------------------------------
# Unit tests: markout sign convention
# ---------------------------------------------------------------------------


class TestMarkoutSignConvention:
    """Pure math tests — no data needed."""

    def test_buy_gets_positive_sign(self) -> None:
        """For a buy aggressor, markout sign = +1."""
        # buy: sign=+1, mid moves up by 0.02 → markout = +0.02
        sign = 1.0
        mid_at_trade = 0.85
        mid_at_h = 0.87
        markout = (mid_at_h - mid_at_trade) * sign
        assert markout == pytest.approx(0.02, abs=1e-9)

    def test_sell_gets_negative_sign(self) -> None:
        """For a sell aggressor, markout sign = -1; mid moves down → positive markout."""
        # sell: sign=-1, mid moves down by 0.02 → markout = -1 * (0.83 - 0.85) = +0.02
        sign = -1.0
        mid_at_trade = 0.85
        mid_at_h = 0.83
        markout = (mid_at_h - mid_at_trade) * sign
        assert markout == pytest.approx(0.02, abs=1e-9)

    def test_buy_adverse_selection_is_negative_markout(self) -> None:
        """When mid moves against the buyer (down after buy), markout is negative."""
        sign = 1.0
        mid_at_trade = 0.85
        mid_at_h = 0.83  # moved against the buyer
        markout = (mid_at_h - mid_at_trade) * sign
        assert markout < 0
        assert markout == pytest.approx(-0.02, abs=1e-9)

    def test_sell_adverse_selection_is_negative_markout(self) -> None:
        """When mid moves against the seller (up after sell), markout is negative."""
        sign = -1.0
        mid_at_trade = 0.50
        mid_at_h = 0.52  # moved against the seller
        markout = (mid_at_h - mid_at_trade) * sign
        assert markout < 0
        assert markout == pytest.approx(-0.02, abs=1e-9)

    def test_zero_markout_when_mid_unchanged(self) -> None:
        """Zero markout when mid doesn't move."""
        for sign in (1.0, -1.0):
            mid = 0.75
            markout = (mid - mid) * sign
            assert markout == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Unit tests: effective spread
# ---------------------------------------------------------------------------


class TestEffectiveSpread:
    """eff_spread = 2 * |price - mid|"""

    def test_buy_above_mid(self) -> None:
        """Buyer pays above mid — eff spread > 0."""
        price = 0.88
        mid = 0.86
        eff = 2.0 * abs(price - mid)
        assert eff == pytest.approx(0.04, abs=1e-9)

    def test_sell_below_mid(self) -> None:
        """Seller receives below mid — eff spread > 0."""
        price = 0.84
        mid = 0.86
        eff = 2.0 * abs(price - mid)
        assert eff == pytest.approx(0.04, abs=1e-9)

    def test_trade_at_mid_zero_spread(self) -> None:
        """Trade exactly at mid → zero spread."""
        price = 0.85
        mid = 0.85
        eff = 2.0 * abs(price - mid)
        assert eff == pytest.approx(0.0, abs=1e-9)

    def test_eff_spread_is_always_non_negative(self) -> None:
        """Effective spread is always >= 0."""
        for price, mid in [(0.90, 0.88), (0.82, 0.86), (0.50, 0.50)]:
            eff = 2.0 * abs(price - mid)
            assert eff >= 0.0


# ---------------------------------------------------------------------------
# Unit tests: realized spread
# ---------------------------------------------------------------------------


class TestRealizedSpread:
    """realized_spread = eff_spread - 2 * markout"""

    def test_positive_realized_when_no_adverse_selection(self) -> None:
        """Maker earns the full spread when markout is zero."""
        eff = 0.04  # 4pp eff spread
        markout = 0.0  # no mid movement
        realized = eff - 2.0 * markout
        assert realized == pytest.approx(0.04, abs=1e-9)

    def test_realized_erodes_with_adverse_selection(self) -> None:
        """Adverse selection (positive markout for taker) erodes maker's realized spread."""
        eff = 0.04
        markout = 0.015  # mid moved 1.5pp in aggressor's favor
        realized = eff - 2.0 * markout
        assert realized == pytest.approx(0.04 - 0.03, abs=1e-9)
        assert realized == pytest.approx(0.01, abs=1e-9)

    def test_realized_negative_when_adverse_selection_exceeds_spread(self) -> None:
        """When adverse selection > spread, maker loses money."""
        eff = 0.04
        markout = 0.03  # large adverse selection
        realized = eff - 2.0 * markout
        assert realized < 0
        assert realized == pytest.approx(-0.02, abs=1e-9)

    def test_half_spread_equals_eff_spread_over_two(self) -> None:
        """Half spread sanity check."""
        bid = 0.84
        ask = 0.88
        mid = (bid + ask) / 2.0
        eff = 2.0 * abs(ask - mid)
        half = ask - mid
        assert eff == pytest.approx(2.0 * half, abs=1e-9)
        assert half == pytest.approx(0.02, abs=1e-9)


# ---------------------------------------------------------------------------
# Unit tests: _compute_markouts helper
# ---------------------------------------------------------------------------


class TestComputeMarkoutsUnit:
    """Test the _compute_markouts function with synthetic data."""

    def _make_trades(self) -> pd.DataFrame:
        """Create a minimal synthetic trades DataFrame."""
        return pd.DataFrame(
            {
                "symbol": ["#100", "#100", "#100"],
                "price": [0.88, 0.84, 0.87],
                "size": [100.0, 50.0, 200.0],
                "side": ["buy", "sell", "buy"],
                "local_recv_ts": [1_000_000_000, 2_000_000_000, 3_000_000_000],
                "exchange_ts": [999_000_000, 1_999_000_000, 2_999_000_000],
            }
        )

    def _make_bbo(self) -> pd.DataFrame:
        """Create a minimal synthetic BBO DataFrame."""
        # BBO ticks at 0.5s intervals around the trades
        return pd.DataFrame(
            {
                "symbol": ["#100"] * 8,
                "bid_px": [0.84, 0.84, 0.85, 0.85, 0.86, 0.86, 0.87, 0.87],
                "ask_px": [0.88, 0.88, 0.89, 0.89, 0.90, 0.90, 0.91, 0.91],
                "mid": [0.86, 0.86, 0.87, 0.87, 0.88, 0.88, 0.89, 0.89],
                "local_recv_ts": [
                    500_000_000,
                    1_500_000_000,
                    2_500_000_000,
                    3_500_000_000,
                    4_000_000_000,
                    60_000_000_000,  # 60s after trade 1
                    120_000_000_000,  # 120s
                    1_800_000_000_000,  # 1800s
                ],
            }
        )

    def _make_expiry_map(self) -> dict:
        """Expiry well in the future so tte_s > 0."""
        return {
            "#100": {
                "expiry_ns": 100_000_000_000_000,  # far future
                "yes_won": True,
            }
        }

    def test_returns_dataframe(self) -> None:
        from hlanalysis.research.cards.card_b_adverse_selection import _compute_markouts

        trades = self._make_trades()
        bbo = self._make_bbo()
        expiry_map = self._make_expiry_map()
        result = _compute_markouts(trades, bbo, expiry_map)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self) -> None:
        from hlanalysis.research.cards.card_b_adverse_selection import (
            _HORIZONS_S,
            _compute_markouts,
        )

        trades = self._make_trades()
        bbo = self._make_bbo()
        expiry_map = self._make_expiry_map()
        result = _compute_markouts(trades, bbo, expiry_map)
        if result.empty:
            pytest.skip("Empty result with synthetic data")

        for col in ["symbol", "price", "size", "side", "mid_at_trade", "eff_spread", "tte_s", "sign"]:
            assert col in result.columns, f"Missing column: {col}"
        for h in _HORIZONS_S:
            assert f"markout_{h}s" in result.columns

    def test_buy_sign_is_positive(self) -> None:
        from hlanalysis.research.cards.card_b_adverse_selection import _compute_markouts

        trades = self._make_trades()
        bbo = self._make_bbo()
        expiry_map = self._make_expiry_map()
        result = _compute_markouts(trades, bbo, expiry_map)
        if result.empty:
            pytest.skip("Empty result with synthetic data")
        buy_rows = result[result["side"] == "buy"]
        assert (buy_rows["sign"] == 1.0).all()

    def test_sell_sign_is_negative(self) -> None:
        from hlanalysis.research.cards.card_b_adverse_selection import _compute_markouts

        trades = self._make_trades()
        bbo = self._make_bbo()
        expiry_map = self._make_expiry_map()
        result = _compute_markouts(trades, bbo, expiry_map)
        if result.empty:
            pytest.skip("Empty result with synthetic data")
        sell_rows = result[result["side"] == "sell"]
        assert (sell_rows["sign"] == -1.0).all()

    def test_eff_spread_is_non_negative(self) -> None:
        from hlanalysis.research.cards.card_b_adverse_selection import _compute_markouts

        trades = self._make_trades()
        bbo = self._make_bbo()
        expiry_map = self._make_expiry_map()
        result = _compute_markouts(trades, bbo, expiry_map)
        if result.empty:
            pytest.skip("Empty result with synthetic data")
        valid = result[result["eff_spread"].notna()]
        assert (valid["eff_spread"] >= 0.0).all()

    def test_empty_bbo_returns_empty(self) -> None:
        from hlanalysis.research.cards.card_b_adverse_selection import _compute_markouts

        trades = self._make_trades()
        bbo_empty = pd.DataFrame(columns=["symbol", "bid_px", "ask_px", "mid", "local_recv_ts"])
        expiry_map = self._make_expiry_map()
        result = _compute_markouts(trades, bbo_empty, expiry_map)
        assert isinstance(result, pd.DataFrame)
        # All markouts should be NaN or result is empty
        if not result.empty:
            assert result["mid_at_trade"].isna().all()

    def test_empty_trades_returns_empty(self) -> None:
        from hlanalysis.research.cards.card_b_adverse_selection import _compute_markouts

        trades_empty = pd.DataFrame(columns=["symbol", "price", "size", "side", "local_recv_ts", "exchange_ts"])
        bbo = self._make_bbo()
        expiry_map = self._make_expiry_map()
        result = _compute_markouts(trades_empty, bbo, expiry_map)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Unit tests: _eff_realized_spread helper
# ---------------------------------------------------------------------------


class TestEffRealizedSpread:
    """Test the spread decomposition helper."""

    def _make_df(self, eff: list[float], markout: list[float]) -> pd.DataFrame:
        n = len(eff)
        return pd.DataFrame(
            {
                "eff_spread": eff,
                "markout_300s": markout,
                "side": ["buy"] * n,
                "size": [100.0] * n,
                "mid_at_trade": [0.85] * n,
                "tte_s": [3600.0] * n,
            }
        )

    def test_basic_computation(self) -> None:
        from hlanalysis.research.cards.card_b_adverse_selection import _eff_realized_spread

        df = self._make_df([0.04, 0.04], [0.01, 0.01])
        result = _eff_realized_spread(df, realized_horizon_s=300)
        assert result["mean_eff_spread"] == pytest.approx(0.04, abs=1e-9)
        assert result["mean_markout_at_horizon"] == pytest.approx(0.01, abs=1e-9)
        # realized = 0.04 - 2*0.01 = 0.02
        assert result["mean_realized_spread"] == pytest.approx(0.02, abs=1e-9)

    def test_all_maker_profitable(self) -> None:
        from hlanalysis.research.cards.card_b_adverse_selection import _eff_realized_spread

        # No adverse selection: realized spread = eff spread
        df = self._make_df([0.04] * 10, [0.0] * 10)
        result = _eff_realized_spread(df, realized_horizon_s=300)
        assert result["pct_maker_profitable"] == pytest.approx(1.0, abs=1e-9)

    def test_empty_returns_empty_dict(self) -> None:
        from hlanalysis.research.cards.card_b_adverse_selection import _eff_realized_spread

        df = pd.DataFrame(columns=["eff_spread", "markout_300s"])
        result = _eff_realized_spread(df, realized_horizon_s=300)
        assert result == {}


# ---------------------------------------------------------------------------
# Integration tests (require real data)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="../../data not present")
class TestBuildCardIntegration:
    """Integration tests that exercise the full build_card pipeline with real data."""

    @pytest.fixture(scope="class")
    def card_result(self):
        """Build card once and share across tests in this class."""
        from hlanalysis.research.cards.card_b_adverse_selection import build_card

        con = duckdb.connect()
        return build_card(con, str(_DATA_ROOT))

    def test_build_card_returns_html_and_dict(self, card_result) -> None:
        html, findings = card_result
        assert isinstance(html, str)
        assert len(html) > 100
        assert "<!DOCTYPE html>" in html or "<html" in html
        assert isinstance(findings, dict)

    def test_findings_has_required_keys(self, card_result) -> None:
        _, findings = card_result
        required = {"title", "headline", "metrics", "split_half", "flb_edge_survival", "verdict"}
        assert required.issubset(findings.keys()), f"Missing keys: {required - findings.keys()}"

    def test_findings_is_json_serializable(self, card_result) -> None:
        _, findings = card_result
        dumped = json.dumps(findings)
        assert len(dumped) > 10
        # Round-trip
        reloaded = json.loads(dumped)
        assert reloaded["title"] == findings["title"]

    def test_n_expiries_at_least_30(self, card_result) -> None:
        _, findings = card_result
        n_metric = next((m for m in findings["metrics"] if m["name"] == "n_expiries"), None)
        assert n_metric is not None, "n_expiries metric not found"
        assert n_metric["value"] >= 30, f"Expected >=30 expiries, got {n_metric['value']}"

    def test_split_half_has_h1_and_h2(self, card_result) -> None:
        _, findings = card_result
        sh = findings["split_half"]
        assert "H1" in sh, "H1 missing from split_half"
        assert "H2" in sh, "H2 missing from split_half"
        assert "sign_stable" in sh

    def test_split_half_h1_h2_have_n(self, card_result) -> None:
        _, findings = card_result
        sh = findings["split_half"]
        assert "n" in sh["H1"]
        assert "n" in sh["H2"]
        assert sh["H1"]["n"] > 0
        assert sh["H2"]["n"] > 0

    def test_flb_edge_survival_has_required_fields(self, card_result) -> None:
        _, findings = card_result
        fe = findings["flb_edge_survival"]
        assert "raw_e_edge_pp" in fe
        assert "adverse_selection_30m_pp" in fe
        assert "net_taker_edge_pp" in fe
        assert "n" in fe
        assert fe["raw_e_edge_pp"] > 0, "Raw edge should be positive"

    def test_verdict_is_string_and_non_empty(self, card_result) -> None:
        _, findings = card_result
        verdict = findings["verdict"]
        assert isinstance(verdict, str)
        assert len(verdict) > 0

    def test_metrics_have_required_fields(self, card_result) -> None:
        _, findings = card_result
        for m in findings["metrics"]:
            assert "name" in m, "Missing 'name' in metric"
            assert "value" in m, "Missing 'value' in metric"
            assert "n" in m, "Missing 'n' in metric"
            assert "date_span" in m, "Missing 'date_span' in metric"
            assert "sanity" in m, "Missing 'sanity' in metric"

    def test_buy_markout_curve_present(self, card_result) -> None:
        _, findings = card_result
        bc = findings.get("buy_markout_curve_pp", {})
        assert len(bc) > 0, "buy_markout_curve_pp is empty"
        # Should have markout at all horizons
        for h in [1, 5, 30, 60, 300, 1800]:
            key = f"buy_markout_{h}s_pp"
            assert key in bc, f"Missing {key} in buy_markout_curve_pp"

    def test_output_html_written(self, card_result) -> None:
        """After build_card, card_b.html should exist on disk."""
        out_path = _WORKTREE_ROOT / "docs" / "research" / "_cards" / "card_b.html"
        assert out_path.exists(), f"card_b.html not found at {out_path}"

    def test_output_json_can_be_written(self, card_result, tmp_path) -> None:
        """Findings can be serialized and written to disk."""
        _, findings = card_result
        out = tmp_path / "card_b_test.json"
        out.write_text(json.dumps(findings, indent=2), encoding="utf-8")
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["title"] == findings["title"]

    def test_n_trades_positive(self, card_result) -> None:
        _, findings = card_result
        n_trade_metric = next((m for m in findings["metrics"] if m["name"] == "n_trades_total"), None)
        assert n_trade_metric is not None
        assert n_trade_metric["value"] > 0

    def test_headline_non_empty(self, card_result) -> None:
        _, findings = card_result
        assert len(findings["headline"]) > 10
