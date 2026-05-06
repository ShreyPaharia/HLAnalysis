"""Helpers used across analysis notebooks.

Goals:
  - one place to compute hive-partition globs so notebooks stay short
  - one DuckDB connection factory with sensible defaults
  - small load_df helper that returns pandas (most familiar) by default
  - a plotting style preset

Notebooks should depend only on this module + duckdb/pandas/matplotlib.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"


def glob_for(
    *,
    venue: str = "*",
    product_type: str = "*",
    mechanism: str = "*",
    event: str = "*",
    symbol: str = "*",
    date: str = "*",
    hour: str = "*",
) -> str:
    """Return a hive-partitioned parquet glob.

    Symbols with `#` (HIP-4) and other special chars must be passed verbatim.
    """
    return str(
        DATA_ROOT
        / f"venue={venue}"
        / f"product_type={product_type}"
        / f"mechanism={mechanism}"
        / f"event={event}"
        / f"symbol={symbol}"
        / f"date={date}"
        / f"hour={hour}"
        / "*.parquet"
    )


def duck() -> duckdb.DuckDBPyConnection:
    """Fresh in-memory DuckDB connection with hive partitioning enabled."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false;")
    return con


def load_df(sql: str, *, params: Iterable | None = None) -> pd.DataFrame:
    """Run a SQL query and return a pandas DataFrame."""
    con = duck()
    if params:
        return con.execute(sql, list(params)).df()
    return con.execute(sql).df()


def fmt_ts(ns: int | float) -> str:
    """ns since epoch -> 'YYYY-MM-DD HH:MM:SS' UTC string."""
    return dt.datetime.fromtimestamp(int(ns) / 1e9, tz=dt.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def set_mpl_defaults() -> None:
    """Apply consistent matplotlib styling across notebooks."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (11, 4.5),
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "lines.linewidth": 1.3,
        }
    )
