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
from collections.abc import Iterable
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"


def asof_locf(
    query_ts: np.ndarray,
    source_ts: np.ndarray,
    source_val: np.ndarray,
) -> np.ndarray:
    """Vectorized last-observation-carried-forward (as-of) join.

    For each timestamp in ``query_ts``, return the value in ``source_val`` whose
    ``source_ts`` is the largest value ``<=`` the query timestamp.  Query
    timestamps with no source row at or before them yield ``NaN``.

    This is the single shared implementation of the LOCF / backward as-of join
    used across the analysis subsystem (trade markouts, resampled mid paths,
    resampled quoted spreads).

    Parameters
    ----------
    query_ts:
        Timestamps to look up, sorted ascending.
    source_ts:
        Source timestamps, sorted ascending.
    source_val:
        Values parallel to ``source_ts``.

    Returns
    -------
    ``np.ndarray`` of dtype ``float64`` and length ``len(query_ts)``.  Positions
    with no preceding source row are ``NaN``.
    """
    query = np.asarray(query_ts)
    src_ts = np.asarray(source_ts)
    src_val = np.asarray(source_val, dtype="float64")

    out = np.full(query.shape[0], np.nan, dtype="float64")
    if src_ts.shape[0] == 0:
        return out

    # searchsorted(..., side='right') - 1 gives the index of the last source_ts
    # element that is <= the query timestamp; -1 means none precedes it.
    idxs = np.searchsorted(src_ts, query, side="right") - 1
    valid = idxs >= 0
    out[valid] = src_val[idxs[valid]]
    return out


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
    return dt.datetime.fromtimestamp(int(ns) / 1e9, tz=dt.UTC).strftime(
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
