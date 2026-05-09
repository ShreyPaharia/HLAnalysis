"""hlanalysis/sim/plots/_common.py

Shared helpers for all plot modules.

Public API
----------
save_fig(fig, out_path) -> Path
    Write *fig* to *out_path* as HTML, creating parent directories as needed.
    Returns *out_path*.
"""
from __future__ import annotations

from pathlib import Path


def save_fig(fig, out_path: Path) -> Path:
    """Write *fig* to *out_path* as HTML and return *out_path*.

    Creates parent directories automatically.  *fig* must have a
    ``write_html(path: str)`` method (any Plotly figure satisfies this).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    return out_path
