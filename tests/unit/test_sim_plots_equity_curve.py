"""Unit tests for hlanalysis.sim.plots.equity_curve."""
from __future__ import annotations

from pathlib import Path

from hlanalysis.sim.plots.equity_curve import plot_equity_curve


class TestPlotEquityCurve:
    def test_returns_path_and_file_exists(self, tmp_path: Path):
        """plot_equity_curve writes equity_curve.html and returns its path."""
        out_path = tmp_path / "equity_curve.html"
        result = plot_equity_curve([1.0, -0.5, 2.0], "model_edge", out_path)
        assert result == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_file_is_html(self, tmp_path: Path):
        """Output file contains basic HTML/Plotly markers."""
        out_path = tmp_path / "equity_curve.html"
        plot_equity_curve([1.0, 2.0, -1.0], "v2", out_path)
        content = out_path.read_text()
        assert "<html" in content.lower() or "plotly" in content.lower()

    def test_empty_pnl_list(self, tmp_path: Path):
        """Empty per_market_pnl still produces a valid HTML file."""
        out_path = tmp_path / "equity_curve.html"
        result = plot_equity_curve([], "empty_run", out_path)
        assert result == out_path
        assert out_path.exists()

    def test_creates_parent_dirs(self, tmp_path: Path):
        """parent directories are created automatically."""
        out_path = tmp_path / "nested" / "dir" / "equity_curve.html"
        result = plot_equity_curve([1.0], "v1", out_path)
        assert result == out_path
        assert out_path.exists()
