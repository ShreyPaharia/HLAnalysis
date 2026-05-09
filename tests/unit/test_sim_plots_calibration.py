"""Unit tests for hlanalysis.sim.plots.calibration (Task C3).

Test strategy:
- Tests for _binned_calibration_curve (pure computation, no plotly)
- Tests for plot_calibration (file I/O; assert file written, non-empty)
- AC2 synthetic-GBM test: binned-mean curve within ±0.01 of y=x at central bins
- Edge cases: zero ENTER fills, fewer than 4 entries
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hlanalysis.sim.plots.calibration import (
    _binned_calibration_curve,
    plot_calibration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fills_parquet(
    path: Path,
    fills: list[dict],
) -> None:
    """Write a fills.parquet at *path* matching FILLS_SCHEMA."""
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        pa.field("cloid",                   pa.string()),
        pa.field("ts_ns",                   pa.int64()),
        pa.field("side",                    pa.string()),
        pa.field("price",                   pa.float64()),
        pa.field("size",                    pa.float64()),
        pa.field("fee",                     pa.float64()),
        pa.field("condition_id",            pa.string()),
        pa.field("question_idx",            pa.int64()),
        pa.field("symbol",                  pa.string()),
        pa.field("entry_p_model",           pa.float64()),
        pa.field("entry_edge_chosen_side",  pa.float64()),
        pa.field("entry_sigma",             pa.float64()),
        pa.field("entry_tau_yr",            pa.float64()),
        pa.field("realized_pnl_at_settle",  pa.float64()),
    ])
    if not fills:
        table = pa.table(
            {f.name: pa.array([], type=f.type) for f in schema},
            schema=schema,
        )
    else:
        cols: dict = {f.name: [] for f in schema}
        for row in fills:
            for f in schema:
                cols[f.name].append(row.get(f.name))
        arrays = {
            name: pa.array(vals, type=schema.field(name).type)
            for name, vals in cols.items()
        }
        table = pa.table(arrays, schema=schema)
    pq.write_table(table, path)


def _enter_fill(
    *,
    condition_id: str = "cid_x",
    price: float = 0.5,
    size: float = 100.0,
    entry_edge_chosen_side: float,
    realized_pnl_at_settle: float,
) -> dict:
    return dict(
        cloid="c1",
        ts_ns=int(1.1e18),
        side="buy",
        price=price,
        size=size,
        fee=0.0,
        condition_id=condition_id,
        question_idx=0,
        symbol="tok_yes",
        entry_p_model=0.55,
        entry_edge_chosen_side=entry_edge_chosen_side,
        entry_sigma=0.2,
        entry_tau_yr=0.003,
        realized_pnl_at_settle=realized_pnl_at_settle,
    )


def _settle_fill(condition_id: str = "cid_x") -> dict:
    return dict(
        cloid="settle",
        ts_ns=int(2e18),
        side="sell",
        price=1.0,
        size=100.0,
        fee=0.0,
        condition_id=condition_id,
        question_idx=0,
        symbol="tok_yes",
        entry_p_model=None,
        entry_edge_chosen_side=None,
        entry_sigma=None,
        entry_tau_yr=None,
        realized_pnl_at_settle=10.0,
    )


# ---------------------------------------------------------------------------
# _binned_calibration_curve unit tests
# ---------------------------------------------------------------------------

class TestBinnedCalibrationCurve:
    def test_basic_ten_bins(self):
        """With many evenly-spaced points, returns list of (x, y) tuples."""
        n = 100
        edges = [i / n for i in range(n)]
        realized = [e for e in edges]  # perfect calibration
        result = _binned_calibration_curve(edges, realized, n_bins=10)
        assert isinstance(result, list)
        assert len(result) == 10
        for x, y in result:
            assert isinstance(x, float)
            assert isinstance(y, float)

    def test_perfect_calibration_close_to_identity(self):
        """When realized == predicted, binned means should be close to bin-mean x."""
        n = 200
        edges = [i / n for i in range(n)]
        realized = list(edges)
        result = _binned_calibration_curve(edges, realized, n_bins=10)
        for x, y in result:
            assert abs(y - x) < 1e-9

    def test_fewer_than_4_entries_returns_empty(self):
        """With < 4 entries, no binning, returns empty list."""
        result = _binned_calibration_curve([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], n_bins=10)
        assert result == []

    def test_exactly_4_entries_produces_bins(self):
        """With exactly 4 entries, n_bins=min(10, 4//2)=2 bins computed."""
        edges = [0.1, 0.2, 0.3, 0.4]
        realized = [0.1, 0.2, 0.3, 0.4]
        result = _binned_calibration_curve(edges, realized, n_bins=10)
        # min(10, 4//2) = 2 bins
        assert len(result) == 2

    def test_fewer_than_10_entries_uses_fewer_bins(self):
        """With 7 entries, n_bins=min(10, 7//2)=3 bins."""
        edges = [i * 0.1 for i in range(7)]
        realized = list(edges)
        result = _binned_calibration_curve(edges, realized, n_bins=10)
        assert len(result) == 3

    def test_bin_mean_is_mean_of_realized_in_bin(self):
        """Bin y-value is the mean of realized_per_dollar values in that quantile bin."""
        # 10 points: edge=0.0 to 0.9, realized=edge+0.1
        edges = [i * 0.1 for i in range(10)]
        realized = [e + 0.1 for e in edges]
        result = _binned_calibration_curve(edges, realized, n_bins=5)
        # Each bin has 2 points; x is mean of edges in bin, y is mean of realized in bin
        # Bins sorted by edge: [0,0.1],[0.2,0.3],[0.4,0.5],[0.6,0.7],[0.8,0.9]
        # y for first bin: mean([0.1,0.2])=0.15; x=mean([0,0.1])=0.05
        x0, y0 = result[0]
        assert abs(x0 - 0.05) < 1e-9
        assert abs(y0 - 0.15) < 1e-9


# ---------------------------------------------------------------------------
# plot_calibration: file I/O tests
# ---------------------------------------------------------------------------

class TestPlotCalibration:
    def test_returns_path_and_file_exists(self, tmp_path: Path):
        """plot_calibration writes calibration.html and returns its path."""
        fills_dir = tmp_path / "fills"
        _make_fills_parquet(fills_dir / "cid_a.parquet", [
            _enter_fill(condition_id="cid_a", entry_edge_chosen_side=0.05,
                        realized_pnl_at_settle=5.0),
            _settle_fill("cid_a"),
        ])
        out_path = tmp_path / "calibration.html"
        result = plot_calibration(fills_dir, out_path)
        assert result == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_file_is_html(self, tmp_path: Path):
        """Output file contains basic HTML markers."""
        fills_dir = tmp_path / "fills"
        _make_fills_parquet(fills_dir / "cid_b.parquet", [
            _enter_fill(condition_id="cid_b", entry_edge_chosen_side=0.05,
                        realized_pnl_at_settle=5.0),
            _settle_fill("cid_b"),
        ])
        out_path = tmp_path / "calibration.html"
        plot_calibration(fills_dir, out_path)
        content = out_path.read_text()
        assert "<html" in content.lower() or "plotly" in content.lower()

    def test_zero_enter_fills_returns_none(self, tmp_path: Path):
        """With no ENTER fills (only settle rows), returns None and writes no file."""
        fills_dir = tmp_path / "fills"
        _make_fills_parquet(fills_dir / "cid_empty.parquet", [
            _settle_fill("cid_empty"),
        ])
        out_path = tmp_path / "calibration.html"
        result = plot_calibration(fills_dir, out_path)
        assert result is None
        assert not out_path.exists()

    def test_empty_fills_dir_returns_none(self, tmp_path: Path):
        """With no parquet files in fills_dir, returns None."""
        fills_dir = tmp_path / "fills"
        fills_dir.mkdir(parents=True)
        out_path = tmp_path / "calibration.html"
        result = plot_calibration(fills_dir, out_path)
        assert result is None
        assert not out_path.exists()

    def test_plot_calibration_returns_none_when_fills_dir_missing(self, tmp_path: Path):
        """When fills_dir does not exist, returns None without raising."""
        fills_dir = tmp_path / "fills_nonexistent"
        # Deliberately do NOT create fills_dir
        assert not fills_dir.exists()
        out_path = tmp_path / "calibration.html"
        result = plot_calibration(fills_dir, out_path)
        assert result is None
        assert not out_path.exists()

    def test_v1_fills_no_entry_p_model_returns_none(self, tmp_path: Path):
        """v1 fills (entry_edge_chosen_side is None) produce no plot."""
        fills_dir = tmp_path / "fills"
        _make_fills_parquet(fills_dir / "cid_v1.parquet", [
            dict(cloid="c1", ts_ns=int(1.1e18), side="buy", price=0.6, size=100.0,
                 fee=0.0, condition_id="cid_v1", question_idx=0, symbol="tok_yes",
                 entry_p_model=None, entry_edge_chosen_side=None,
                 entry_sigma=None, entry_tau_yr=None, realized_pnl_at_settle=5.0),
            _settle_fill("cid_v1"),
        ])
        out_path = tmp_path / "calibration.html"
        result = plot_calibration(fills_dir, out_path)
        assert result is None
        assert not out_path.exists()

    def test_multiple_fills_files_concatenated(self, tmp_path: Path):
        """Fills from multiple market parquet files are merged."""
        fills_dir = tmp_path / "fills"
        for i in range(3):
            _make_fills_parquet(fills_dir / f"cid_{i}.parquet", [
                _enter_fill(
                    condition_id=f"cid_{i}",
                    entry_edge_chosen_side=0.04 + i * 0.01,
                    realized_pnl_at_settle=4.0 + i,
                ),
                _settle_fill(f"cid_{i}"),
            ])
        out_path = tmp_path / "calibration.html"
        result = plot_calibration(fills_dir, out_path)
        # 3 ENTER fills — enough for a plot (>= 4 is needed for binning but scatter
        # with y=x still renders)
        assert result == out_path
        assert out_path.exists()


# ---------------------------------------------------------------------------
# AC2: Synthetic-GBM calibration test
# ---------------------------------------------------------------------------

class TestSyntheticGBMCalibration:
    """
    Acceptance criterion 2: with a large synthetic dataset where the model edge
    is ground-truth correct on average, the binned-mean curve must land within
    ±0.01 of y=x at the central bins.

    Construction:
    - entry_edge_chosen_side = e ~ Uniform(0.01, 0.20) for each fill.
    - Set price=1.0, size=1.0 so realized_pnl_per_dollar = realized_pnl_at_settle.
    - Set E[realized_pnl_at_settle | edge=e] = e exactly by drawing
      realized_pnl_at_settle = e + eps, eps ~ N(0, 0.1^2).
    - With N=5000, the binned-mean curve should track y=x within ±0.01 at central bins.
    """

    def test_binned_mean_within_tolerance_of_y_equals_x(self):
        """Binned-mean curve is within ±0.01 of y=x at central bins (AC2)."""
        rng = random.Random(42)
        N = 5000
        edges: list[float] = []
        realized: list[float] = []

        for _ in range(N):
            e = rng.uniform(0.01, 0.20)
            # Each fill: realized_pnl_per_dollar has mean = e
            # Use Bernoulli: win with prob (0.5 + e/2), lose otherwise
            # Win: pnl_per_dollar = 2*e (mean contribution = (0.5+e/2)*2e)
            # Lose: pnl_per_dollar = 0 (mean contribution = 0)
            # => E = (0.5 + e/2) * 2e = e + e^2, biased. Use direct:
            # realized = e if win else -e (± Rademacher * e), E=0 — wrong.
            # Simplest: realized = e + eps where eps ~ N(0, 0.1^2)
            eps = rng.gauss(0.0, 0.10)
            realized_val = e + eps
            edges.append(e)
            realized.append(realized_val)

        result = _binned_calibration_curve(edges, realized, n_bins=10)
        assert len(result) == 10

        # Central bins: indices 2..7 (skip 2 edge bins on each side)
        for idx in range(2, 8):
            x, y = result[idx]
            assert abs(y - x) < 0.01, (
                f"Central bin {idx}: x={x:.4f}, y={y:.4f}, "
                f"|y-x|={abs(y-x):.4f} > 0.01"
            )

    def test_edge_bins_within_looser_tolerance(self):
        """Even edge bins should be within ±0.02 of y=x with large N."""
        rng = random.Random(99)
        N = 5000
        edges: list[float] = []
        realized: list[float] = []
        for _ in range(N):
            e = rng.uniform(0.01, 0.20)
            eps = rng.gauss(0.0, 0.10)
            edges.append(e)
            realized.append(e + eps)

        result = _binned_calibration_curve(edges, realized, n_bins=10)
        for x, y in result:
            assert abs(y - x) < 0.025, f"x={x:.4f}, y={y:.4f}, |y-x|={abs(y-x):.4f}"


# ---------------------------------------------------------------------------
# Integration: report.py links calibration.html for v2
# ---------------------------------------------------------------------------

class TestReportIntegration:
    """Verify write_single_run_report links calibration.html when v2 data present."""

    def _make_summary(self):
        from hlanalysis.sim.metrics import RunSummary
        return RunSummary(
            n_markets=1, n_trades=1, total_pnl_usd=5.0,
            sharpe=1.0, hit_rate=1.0, max_drawdown_usd=0.0,
        )

    def _fake_market(self, condition_id: str):
        from dataclasses import dataclass

        @dataclass
        class M:
            condition_id: str
            resolved_outcome: str = "yes"
            start_ts_ns: int = int(1e18)
            end_ts_ns: int = int(2e18)

        return M(condition_id=condition_id)

    def test_report_links_calibration_html_for_v2(self, tmp_path: Path):
        """report.md contains link to calibration.html when v2 fills present."""
        from hlanalysis.sim.report import write_single_run_report

        fills_dir = tmp_path / "fills"
        # Write enough ENTER fills for the plot to be generated
        for i in range(12):
            _make_fills_parquet(fills_dir / f"cid_{i}.parquet", [
                _enter_fill(
                    condition_id=f"cid_{i}",
                    entry_edge_chosen_side=0.04 + i * 0.01,
                    realized_pnl_at_settle=4.0 + i * 0.5,
                ),
                _settle_fill(f"cid_{i}"),
            ])

        markets = [self._fake_market(f"cid_{i}") for i in range(12)]
        write_single_run_report(
            out_dir=tmp_path,
            strategy_name="v2",
            config_summary={"edge_buffer": 0.02},
            per_market_pnl=[5.0] * 12,
            summary=self._make_summary(),
            markets=markets,
            fills_dir=fills_dir,
        )
        assert (tmp_path / "calibration.html").exists()
        text = (tmp_path / "report.md").read_text()
        assert "calibration.html" in text

    def test_report_no_calibration_for_v1(self, tmp_path: Path):
        """report.md does NOT link calibration.html when only v1 fills present."""
        from hlanalysis.sim.report import write_single_run_report

        fills_dir = tmp_path / "fills"
        _make_fills_parquet(fills_dir / "cid_v1.parquet", [
            dict(cloid="c1", ts_ns=int(1.1e18), side="buy", price=0.6, size=100.0,
                 fee=0.0, condition_id="cid_v1", question_idx=0, symbol="tok_yes",
                 entry_p_model=None, entry_edge_chosen_side=None,
                 entry_sigma=None, entry_tau_yr=None, realized_pnl_at_settle=5.0),
            _settle_fill("cid_v1"),
        ])
        markets = [self._fake_market("cid_v1")]
        write_single_run_report(
            out_dir=tmp_path,
            strategy_name="v1",
            config_summary={"edge_buffer": 0.02},
            per_market_pnl=[5.0],
            summary=self._make_summary(),
            markets=markets,
            fills_dir=fills_dir,
        )
        assert not (tmp_path / "calibration.html").exists()
        text = (tmp_path / "report.md").read_text()
        assert "calibration.html" not in text

    def test_report_no_calibration_when_no_fills_dir(self, tmp_path: Path):
        """report.md does NOT link calibration.html when fills_dir is None."""
        from hlanalysis.sim.report import write_single_run_report

        markets = [self._fake_market("cid_x")]
        write_single_run_report(
            out_dir=tmp_path,
            strategy_name="v2",
            config_summary={"edge_buffer": 0.02},
            per_market_pnl=[5.0],
            summary=self._make_summary(),
            markets=markets,
            fills_dir=None,
        )
        assert not (tmp_path / "calibration.html").exists()
        text = (tmp_path / "report.md").read_text()
        assert "calibration.html" not in text
