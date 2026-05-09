"""Microstructure metric helpers for analysis notebooks.

All DuckDB-backed functions are thin, stateless wrappers over hive-partitioned
parquet written by the recorder.  The caller manages the DuckDB connection
lifecycle (use ``helpers.duck()``).

Timing convention: all ts_ns values are ``local_recv_ts`` (nanoseconds since
epoch, captured by ``time.time_ns()`` on the recording host).  Venue-provided
``exchange_ts`` is NOT used for ordering — see book.py for per-venue caveats.
"""
from __future__ import annotations

import math

import duckdb
import numpy as np
import pandas as pd

from hlanalysis.analysis.book import mid_path
from hlanalysis.analysis.helpers import glob_for

_NS_PER_S = 1_000_000_000
_NS_PER_MS = 1_000_000


def quoted_spread_bps(
    con: duckdb.DuckDBPyConnection,
    *,
    venue: str,
    product_type: str,
    symbol: str,
    start_ns: int,
    end_ns: int,
    resample_s: int = 1,
) -> pd.DataFrame:
    """Return the quoted spread in basis points, resampled on a regular grid.

    For each BBO row, computes::

        spread_bps = (ask_px - bid_px) / mid * 1e4   where mid = (ask_px + bid_px) / 2

    Then resamples onto a ``resample_s``-second grid using last-observation-
    carried-forward (LOCF) so every grid point has a value as long as at least
    one BBO row was seen up to that point.

    Uses ``local_recv_ts`` for timing.

    Parameters
    ----------
    con:
        Live DuckDB connection (caller manages lifecycle).
    venue, product_type, symbol:
        Hive-partition selectors.
    start_ns, end_ns:
        Window in nanoseconds (local_recv_ts scale).
    resample_s:
        Grid spacing in seconds.

    Returns
    -------
    DataFrame with columns:

        ``ts_ns``      (int64)   — grid point in nanoseconds.
        ``spread_bps`` (float64) — LOCF spread at that grid point;
                                   NaN for grid points before the first BBO.
    """
    glob = glob_for(venue=venue, product_type=product_type, event="bbo", symbol=symbol)
    sql = f"""
        SELECT
            local_recv_ts                                AS ts_ns,
            (ask_px - bid_px) / ((ask_px + bid_px) / 2.0) * 1e4 AS spread_bps
        FROM read_parquet('{glob}', hive_partitioning=true)
        WHERE local_recv_ts <= ?
        ORDER BY local_recv_ts ASC
    """
    try:
        raw = con.execute(sql, [end_ns]).df()
    except duckdb.IOException:
        raw = pd.DataFrame(
            {"ts_ns": pd.Series([], dtype="int64"), "spread_bps": pd.Series([], dtype="float64")}
        )

    raw["ts_ns"] = raw["ts_ns"].astype("int64")
    raw["spread_bps"] = raw["spread_bps"].astype("float64")

    step_ns = resample_s * _NS_PER_S
    grid = list(range(start_ns, end_ns + 1, step_ns))

    if raw.empty:
        return pd.DataFrame(
            {
                "ts_ns": pd.array(grid, dtype="int64"),
                "spread_bps": pd.array([float("nan")] * len(grid), dtype="float64"),
            }
        )

    raw_ts = raw["ts_ns"].to_numpy()
    raw_spread = raw["spread_bps"].to_numpy()

    result_ts: list[int] = []
    result_spread: list[float] = []
    for g in grid:
        idx = int(np.searchsorted(raw_ts, g, side="right")) - 1
        if idx < 0:
            result_ts.append(g)
            result_spread.append(float("nan"))
        else:
            result_ts.append(g)
            result_spread.append(float(raw_spread[idx]))

    return pd.DataFrame(
        {
            "ts_ns": pd.array(result_ts, dtype="int64"),
            "spread_bps": pd.array(result_spread, dtype="float64"),
        }
    )


def book_imbalance(
    con: duckdb.DuckDBPyConnection,
    *,
    venue: str,
    product_type: str,
    symbol: str,
    start_ns: int,
    end_ns: int,
    levels: int = 1,
) -> pd.DataFrame:
    """Return book imbalance from ``book_snapshot`` events (no resampling).

    For each snapshot, computes::

        imb = (sum(bid_szs[:levels]) - sum(ask_szs[:levels]))
              / (sum(bid_szs[:levels]) + sum(ask_szs[:levels]))

    Range is [-1, +1]: +1 = all bid pressure, -1 = all ask pressure.
    Rows where both sides sum to 0 produce NaN (avoids divide-by-zero).

    Uses ``local_recv_ts`` for timing.

    Parameters
    ----------
    con:
        Live DuckDB connection (caller manages lifecycle).
    venue, product_type, symbol:
        Hive-partition selectors.
    start_ns, end_ns:
        Window in nanoseconds (local_recv_ts scale).
    levels:
        Number of top-of-book levels to include in the imbalance sum.

    Returns
    -------
    DataFrame with columns:

        ``ts_ns``      (int64)   — local_recv_ts of the snapshot.
        ``imbalance``  (float64) — imbalance in [-1, +1]; NaN if total size = 0.
    """
    glob = glob_for(
        venue=venue, product_type=product_type, event="book_snapshot", symbol=symbol
    )
    sql = f"""
        SELECT local_recv_ts AS ts_ns, bid_sz, ask_sz
        FROM read_parquet('{glob}', hive_partitioning=true)
        WHERE local_recv_ts BETWEEN ? AND ?
        ORDER BY local_recv_ts ASC
    """
    try:
        raw = con.execute(sql, [start_ns, end_ns]).df()
    except duckdb.IOException:
        return pd.DataFrame(
            {
                "ts_ns": pd.Series([], dtype="int64"),
                "imbalance": pd.Series([], dtype="float64"),
            }
        )

    if raw.empty:
        return pd.DataFrame(
            {
                "ts_ns": pd.Series([], dtype="int64"),
                "imbalance": pd.Series([], dtype="float64"),
            }
        )

    ts_list: list[int] = []
    imb_list: list[float] = []

    for _, row in raw.iterrows():
        ts_list.append(int(row["ts_ns"]))
        bid_szs = row["bid_sz"]  # list[float]
        ask_szs = row["ask_sz"]  # list[float]
        bid_sum = sum(float(x) for x in bid_szs[:levels])
        ask_sum = sum(float(x) for x in ask_szs[:levels])
        total = bid_sum + ask_sum
        if total == 0.0:
            imb_list.append(float("nan"))
        else:
            imb_list.append((bid_sum - ask_sum) / total)

    return pd.DataFrame(
        {
            "ts_ns": pd.array(ts_list, dtype="int64"),
            "imbalance": pd.array(imb_list, dtype="float64"),
        }
    )


def tob_churn(
    con: duckdb.DuckDBPyConnection,
    *,
    venue: str,
    product_type: str,
    symbol: str,
    start_ns: int,
    end_ns: int,
    bin_s: int = 60,
) -> pd.DataFrame:
    """Return BBO update counts per time bin — a proxy for MM activity.

    For each ``bin_s``-second window in ``[start_ns, end_ns]``, counts how many
    BBO events landed (regardless of whether the quote actually changed).

    Uses ``local_recv_ts`` for timing.

    Parameters
    ----------
    con:
        Live DuckDB connection (caller manages lifecycle).
    venue, product_type, symbol:
        Hive-partition selectors.
    start_ns, end_ns:
        Window in nanoseconds (local_recv_ts scale).
    bin_s:
        Bin width in seconds.

    Returns
    -------
    DataFrame with columns:

        ``ts_ns``     (int64) — start of bin in nanoseconds.
        ``n_updates`` (int64) — number of BBO events in that bin.

    One row per bin covering ``[start_ns, end_ns]``.  Bins with zero updates
    still appear with ``n_updates = 0``.
    """
    step_ns = bin_s * _NS_PER_S
    glob = glob_for(venue=venue, product_type=product_type, event="bbo", symbol=symbol)

    # Use DuckDB integer division to bucket each event, then group by bucket.
    # Cast to BIGINT explicitly — DuckDB infers float for large integer literals
    # when using / instead of //. Using // (integer division) avoids the issue.
    sql = f"""
        SELECT
            (local_recv_ts::BIGINT - ?::BIGINT) // {step_ns}::BIGINT * {step_ns}::BIGINT + ?::BIGINT AS bin_ns,
            COUNT(*)                                                                                   AS n_updates
        FROM read_parquet('{glob}', hive_partitioning=true)
        WHERE local_recv_ts BETWEEN ? AND ?
        GROUP BY bin_ns
        ORDER BY bin_ns ASC
    """
    try:
        agg = con.execute(sql, [start_ns, start_ns, start_ns, end_ns]).df()
    except duckdb.IOException:
        agg = pd.DataFrame({"bin_ns": pd.Series([], dtype="int64"), "n_updates": pd.Series([], dtype="int64")})

    # Build the full grid so empty bins are present with count 0.
    bins = list(range(start_ns, end_ns, step_ns))  # end exclusive — last bin starts < end_ns

    if agg.empty:
        return pd.DataFrame(
            {
                "ts_ns": pd.array(bins, dtype="int64"),
                "n_updates": pd.array([0] * len(bins), dtype="int64"),
            }
        )

    counts: dict[int, int] = {int(r["bin_ns"]): int(r["n_updates"]) for _, r in agg.iterrows()}
    return pd.DataFrame(
        {
            "ts_ns": pd.array(bins, dtype="int64"),
            "n_updates": pd.array([counts.get(b, 0) for b in bins], dtype="int64"),
        }
    )


def returns_resampled(
    con: duckdb.DuckDBPyConnection,
    *,
    venue: str,
    product_type: str,
    symbol: str,
    start_ns: int,
    end_ns: int,
    dt_ms: int,
) -> pd.DataFrame:
    """Return log-returns of the mid price on a regular grid.

    Builds a resampled mid-price path via ``book.mid_path`` at ``dt_ms``
    cadence, then computes::

        log_return[i] = log(mid[i] / mid[i-1])

    The first row's ``log_return`` is always NaN.

    Uses ``local_recv_ts`` for timing.

    Parameters
    ----------
    con:
        Live DuckDB connection (caller manages lifecycle).
    venue, product_type, symbol:
        Hive-partition selectors.
    start_ns, end_ns:
        Window in nanoseconds (local_recv_ts scale).
    dt_ms:
        Grid spacing in milliseconds.

    Returns
    -------
    DataFrame with columns:

        ``ts_ns``      (int64)   — grid point in nanoseconds.
        ``log_return`` (float64) — log(mid[i] / mid[i-1]); NaN for first row
                                   and any row where mid or prior mid is NaN.
    """
    mids = mid_path(
        con,
        venue=venue,
        product_type=product_type,
        symbol=symbol,
        start_ns=start_ns,
        end_ns=end_ns,
        resample_ms=dt_ms,
    )

    mid_vals = mids["mid"].to_numpy(dtype="float64")
    # Compute log-return: NaN for first row and any NaN mid values.
    log_ret = np.full(len(mid_vals), float("nan"), dtype="float64")
    for i in range(1, len(mid_vals)):
        prev = mid_vals[i - 1]
        curr = mid_vals[i]
        if not (math.isnan(prev) or math.isnan(curr)) and prev > 0.0:
            log_ret[i] = math.log(curr / prev)

    return pd.DataFrame(
        {
            "ts_ns": mids["ts_ns"].copy(),
            "log_return": pd.array(log_ret, dtype="float64"),
        }
    )


def cross_correlation(
    x: pd.Series,
    y: pd.Series,
    max_lag: int,
) -> pd.DataFrame:
    """Return the cross-correlation function (CCF) between two series.

    Computes the Pearson correlation of ``x`` against ``y`` shifted by ``lag``
    for each integer lag in ``[-max_lag, +max_lag]``:

        lag > 0:  corr(x, y.shift(lag))  — y's past predicts x's present.
        lag = 0:  corr(x, y)             — contemporaneous.
        lag < 0:  corr(x, y.shift(lag))  — x's past predicts y's present.

    Both series are aligned on their index before computing correlation.
    NaN-containing rows are dropped pairwise (default pandas behaviour for
    ``pd.Series.corr``).

    Parameters
    ----------
    x, y:
        Two ``pd.Series`` of equal length.  Index does not need to be sorted.
    max_lag:
        Maximum absolute lag.  Must satisfy ``max_lag < len(x)``.

    Returns
    -------
    DataFrame with columns:

        ``lag`` (int64)   — integer lag value.
        ``ccf`` (float64) — Pearson correlation at that lag.

    Raises
    ------
    ValueError
        If ``max_lag >= len(x)``.
    """
    n = len(x)
    if max_lag >= n:
        raise ValueError(
            f"max_lag ({max_lag}) must be less than len(x) ({n}). "
            "Reduce max_lag or provide a longer series."
        )

    x_arr = x.reset_index(drop=True)
    y_arr = y.reset_index(drop=True)

    lags: list[int] = []
    ccfs: list[float] = []

    for lag in range(-max_lag, max_lag + 1):
        lags.append(lag)
        ccfs.append(float(x_arr.corr(y_arr.shift(lag))))

    return pd.DataFrame(
        {
            "lag": pd.array(lags, dtype="int64"),
            "ccf": pd.array(ccfs, dtype="float64"),
        }
    )
