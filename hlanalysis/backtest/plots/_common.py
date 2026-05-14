"""Shared helpers for backtest plot modules.

`save_fig(fig, out_path)` writes a Plotly figure to HTML, creating parent
directories as needed. Returns `out_path`.
"""
from __future__ import annotations

from pathlib import Path


def save_fig(fig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    return out_path
