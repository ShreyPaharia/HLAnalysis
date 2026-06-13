"""Self-contained HTML report builder with dark theme.

Usage::

    from hlanalysis.research.report import Report, fig_to_base64

    rpt = Report(title="HL Outcome Market Analysis")
    rpt.add_card("Spread Analysis", html_body="<p>...</p>", fig=some_fig)
    rpt.render("/path/to/output.html")
"""

from __future__ import annotations

import base64
import io
import textwrap
from pathlib import Path

# Lazy matplotlib import so the module is importable without display
try:
    import matplotlib
    import matplotlib.figure

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


def fig_to_base64(fig: matplotlib.figure.Figure) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string.

    Parameters
    ----------
    fig:
        A ``matplotlib.figure.Figure`` instance.

    Returns
    -------
    Base64-encoded PNG as a plain string (no ``data:`` URI prefix).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


class Report:
    """Accumulate analysis cards and render as a standalone dark-theme HTML file.

    Parameters
    ----------
    title:
        Page title and H1 header.
    """

    # Dark GitHub-style colour palette
    _DARK_CSS = textwrap.dedent(
        """
        :root {
            --bg:        #0d1117;
            --card-bg:   #161b22;
            --border:    #30363d;
            --text:      #e6edf3;
            --muted:     #8b949e;
            --accent:    #58a6ff;
            --code-bg:   #1f2428;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            font-size: 14px;
            line-height: 1.6;
            padding: 2rem 1rem;
        }
        h1 {
            color: var(--accent);
            font-size: 1.8rem;
            margin-bottom: 0.4rem;
        }
        .subtitle {
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 1.5rem;
            padding: 1.2rem 1.4rem;
            max-width: 1100px;
        }
        .card h2 {
            color: var(--accent);
            font-size: 1.1rem;
            margin-bottom: 0.8rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.4rem;
        }
        .card img {
            max-width: 100%;
            border-radius: 4px;
            margin-top: 0.8rem;
        }
        .notes {
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 0.8rem;
            padding: 0.5rem 0.8rem;
            border-left: 3px solid var(--border);
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 0.6rem;
        }
        th, td {
            padding: 0.35rem 0.7rem;
            border: 1px solid var(--border);
            text-align: left;
            font-size: 0.85rem;
        }
        th { background: var(--code-bg); color: var(--accent); }
        tr:nth-child(even) td { background: #1a1f26; }
        code, pre {
            background: var(--code-bg);
            border-radius: 4px;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 0.85em;
        }
        code { padding: 0.1em 0.3em; }
        pre { padding: 0.8em; overflow-x: auto; }
        """
    )

    _HTML_TEMPLATE = textwrap.dedent(
        """\
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>{css}</style>
        </head>
        <body>
            <h1>{title}</h1>
            <div class="subtitle">Generated {generated_at}</div>
            {cards}
        </body>
        </html>
        """
    )

    _CARD_TEMPLATE = textwrap.dedent(
        """\
        <div class="card">
            <h2>{title}</h2>
            {body}
            {img_tag}
            {notes_html}
        </div>
        """
    )

    def __init__(self, title: str = "HL Outcome Market Analysis") -> None:
        self.title = title
        self._cards: list[dict] = []

    def add_card(
        self,
        title: str,
        html_body: str,
        fig: matplotlib.figure.Figure | None = None,
        notes: str | None = None,
    ) -> None:
        """Append a card to the report.

        Parameters
        ----------
        title:
            Card heading.
        html_body:
            Inner HTML content (paragraphs, tables, etc.).
        fig:
            Optional matplotlib figure to embed as a base64 PNG.
        notes:
            Optional annotation rendered in a muted callout below the image.
        """
        self._cards.append(
            {
                "title": title,
                "html_body": html_body,
                "fig": fig,
                "notes": notes,
            }
        )

    def render(self, path: str | Path) -> None:
        """Render the report to a standalone HTML file.

        Parameters
        ----------
        path:
            Output file path.  Parent directories are created if needed.
        """
        import datetime as dt

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        card_htmls = []
        for card in self._cards:
            img_tag = ""
            if card["fig"] is not None:
                b64 = fig_to_base64(card["fig"])
                img_tag = f'<img src="data:image/png;base64,{b64}" alt="{card["title"]}">'

            notes_html = ""
            if card["notes"]:
                notes_html = f'<div class="notes">{card["notes"]}</div>'

            card_htmls.append(
                self._CARD_TEMPLATE.format(
                    title=card["title"],
                    body=card["html_body"],
                    img_tag=img_tag,
                    notes_html=notes_html,
                )
            )

        generated_at = dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
        html = self._HTML_TEMPLATE.format(
            title=self.title,
            css=self._DARK_CSS,
            generated_at=generated_at,
            cards="\n".join(card_htmls),
        )

        path.write_text(html, encoding="utf-8")
