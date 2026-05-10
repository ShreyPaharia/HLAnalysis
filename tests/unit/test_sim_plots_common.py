"""Unit tests for hlanalysis.sim.plots._common.save_fig (Issue 3).

Test strategy:
- save_fig writes HTML via fig.write_html and returns out_path.
- save_fig creates parent directories automatically.
- save_fig returns the exact path object passed in.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hlanalysis.sim.plots._common import save_fig


class _FakeFig:
    """Minimal stub with a write_html method that records the path written."""

    def __init__(self) -> None:
        self.written: list[str] = []

    def write_html(self, path: str) -> None:
        Path(path).write_text("<html>stub</html>")
        self.written.append(path)


class TestSaveFig:
    def test_returns_out_path(self, tmp_path: Path):
        """save_fig returns the exact out_path passed in."""
        fig = _FakeFig()
        out = tmp_path / "plot.html"
        result = save_fig(fig, out)
        assert result is out

    def test_write_html_called_with_str(self, tmp_path: Path):
        """save_fig calls fig.write_html with str(out_path)."""
        fig = _FakeFig()
        out = tmp_path / "plot.html"
        save_fig(fig, out)
        assert fig.written == [str(out)]

    def test_file_exists_after_save(self, tmp_path: Path):
        """save_fig results in the file existing on disk."""
        fig = _FakeFig()
        out = tmp_path / "plot.html"
        save_fig(fig, out)
        assert out.exists()

    def test_creates_parent_dirs(self, tmp_path: Path):
        """save_fig creates intermediate parent directories automatically."""
        fig = _FakeFig()
        out = tmp_path / "nested" / "deep" / "plot.html"
        assert not out.parent.exists()
        save_fig(fig, out)
        assert out.exists()

    def test_returns_path_object(self, tmp_path: Path):
        """save_fig return type is Path."""
        fig = _FakeFig()
        out = tmp_path / "plot.html"
        result = save_fig(fig, out)
        assert isinstance(result, Path)
