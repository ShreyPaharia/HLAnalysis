"""Tests for hlanalysis.research.report."""

from __future__ import annotations

import base64
import re
import tempfile
from pathlib import Path


def _make_simple_fig():
    """Create a minimal matplotlib figure for testing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([1, 2, 3], [1, 4, 9])
    return fig


class TestFigToBase64:
    def test_returns_nonempty_string(self) -> None:
        from hlanalysis.research.report import fig_to_base64

        fig = _make_simple_fig()
        result = fig_to_base64(fig)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_valid_base64(self) -> None:
        from hlanalysis.research.report import fig_to_base64

        fig = _make_simple_fig()
        result = fig_to_base64(fig)
        # Should be decodable base64
        decoded = base64.b64decode(result)
        # PNG magic bytes: \x89PNG\r\n\x1a\n
        assert decoded[:4] == b"\x89PNG", "Expected PNG magic bytes"

    def test_different_figures_produce_different_output(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from hlanalysis.research.report import fig_to_base64

        fig1, ax1 = plt.subplots()
        ax1.plot([1, 2], [1, 2])

        fig2, ax2 = plt.subplots()
        ax2.plot([1, 2], [2, 1])  # different data

        b1 = fig_to_base64(fig1)
        b2 = fig_to_base64(fig2)
        assert b1 != b2

        import matplotlib.pyplot as plt

        plt.close("all")


class TestReport:
    def test_render_creates_file(self) -> None:
        from hlanalysis.research.report import Report

        rpt = Report(title="Test Report")
        rpt.add_card("Card 1", "<p>Hello world</p>")

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_report.html"
            rpt.render(out)
            assert out.exists()

    def test_render_produces_valid_html(self) -> None:
        from hlanalysis.research.report import Report

        rpt = Report(title="Test Report")
        rpt.add_card("Card 1", "<p>Content</p>")

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.html"
            rpt.render(out)
            html = out.read_text(encoding="utf-8")

        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "Test Report" in html
        assert "Card 1" in html
        assert "Content" in html

    def test_render_with_figure_embeds_base64(self) -> None:
        from hlanalysis.research.report import Report

        fig = _make_simple_fig()
        rpt = Report(title="Fig Report")
        rpt.add_card("Chart", "<p>See chart</p>", fig=fig)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "fig_report.html"
            rpt.render(out)
            html = out.read_text(encoding="utf-8")

        # Should contain a base64 data URI for the PNG
        assert "data:image/png;base64," in html
        # Extract and verify it's valid base64
        m = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", html)
        assert m is not None, "base64 img tag not found in rendered HTML"
        decoded = base64.b64decode(m.group(1))
        assert decoded[:4] == b"\x89PNG"

        import matplotlib.pyplot as plt

        plt.close("all")

    def test_render_dark_theme_css(self) -> None:
        from hlanalysis.research.report import Report

        rpt = Report()
        rpt.add_card("Test", "<p>x</p>")

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "dark.html"
            rpt.render(out)
            html = out.read_text(encoding="utf-8")

        # Check dark theme colors
        assert "#0d1117" in html  # background
        assert "#161b22" in html  # card bg
        assert "#58a6ff" in html  # accent

    def test_render_creates_parent_dirs(self) -> None:
        from hlanalysis.research.report import Report

        rpt = Report()
        rpt.add_card("T", "<p>x</p>")

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "sub" / "dir" / "report.html"
            assert not out.parent.exists()
            rpt.render(out)
            assert out.exists()

    def test_multiple_cards(self) -> None:
        from hlanalysis.research.report import Report

        rpt = Report(title="Multi")
        for i in range(5):
            rpt.add_card(f"Card {i}", f"<p>content {i}</p>")

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "multi.html"
            rpt.render(out)
            html = out.read_text(encoding="utf-8")

        for i in range(5):
            assert f"Card {i}" in html
            assert f"content {i}" in html

    def test_notes_rendered(self) -> None:
        from hlanalysis.research.report import Report

        rpt = Report()
        rpt.add_card("T", "<p>x</p>", notes="This is a note.")

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "notes.html"
            rpt.render(out)
            html = out.read_text(encoding="utf-8")

        assert "This is a note." in html

    def test_empty_report(self) -> None:
        from hlanalysis.research.report import Report

        rpt = Report(title="Empty")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "empty.html"
            rpt.render(out)
            html = out.read_text(encoding="utf-8")
        assert "Empty" in html
        assert "<!DOCTYPE html>" in html
