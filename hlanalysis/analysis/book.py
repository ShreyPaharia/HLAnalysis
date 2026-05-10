"""L2 reconstruction helpers for analysis notebooks.

All functions are thin, stateless wrappers over DuckDB queries against
hive-partitioned parquet written by the recorder. The caller is responsible
for managing the DuckDB connection lifecycle (use ``helpers.duck()``).

Per-venue exchange_ts caveats (from docs/reports/00-overview.md §4):
- Binance spot bbo/book: exchange_ts is sentinel 0.  These functions use
  ``local_recv_ts`` for all timing, so no special filter is needed here.
  However, callers must not interpret ``exchange_ts`` from Binance spot rows.
- HL trade exchange_ts = block time, lagged ~5 s from publish ts.  Use
  ``local_recv_ts`` for real-time ordering.
- HL perp transport latency to the recording host is ~225 ms median.
  Cross-venue comparisons should subtract per-venue median transport latency
  before drawing lead-lag conclusions.
"""
from __future__ import annotations

import math
from typing import Literal

import duckdb
import numpy as np
import pandas as pd

from hlanalysis.analysis.helpers import glob_for


def best_quotes_at(
    con: duckdb.DuckDBPyConnection,
    *,
    venue: str,
    product_type: str,
    symbol: str,
    ts_ns: int,
) -> tuple[float, float]:
    """Return (bid_px, ask_px) of the most recent BBO at or before ts_ns.

    Uses ``local_recv_ts`` for timing — venue-provided ``exchange_ts`` is NOT
    used for ordering.  For Binance spot, ``exchange_ts`` is 0 (sentinel); this
    function is unaffected because it orders by ``local_recv_ts``.

    Parameters
    ----------
    con:
        Live DuckDB connection (caller manages lifecycle).
    venue, product_type, symbol:
        Hive-partition selectors.  Passed verbatim to ``glob_for``; symbols
        containing ``#`` (HIP-4) are handled correctly.
    ts_ns:
        Wall-clock nanoseconds since epoch (local_recv_ts scale).

    Returns
    -------
    (bid_px, ask_px) as floats.

    Raises
    ------
    ValueError
        If no BBO row exists at or before ``ts_ns`` in the matching partitions.
    """
    glob = glob_for(venue=venue, product_type=product_type, event="bbo", symbol=symbol)
    sql = f"""
        SELECT bid_px, ask_px
        FROM read_parquet('{glob}', hive_partitioning=true)
        WHERE local_recv_ts <= ?
        ORDER BY local_recv_ts DESC
        LIMIT 1
    """
    try:
        row = con.execute(sql, [ts_ns]).fetchone()
    except duckdb.IOException:
        # No parquet files exist for this partition at all — treat as no data.
        row = None
    if row is None:
        raise ValueError(
            f"No BBO row found at or before ts_ns={ts_ns} for "
            f"venue={venue!r} product_type={product_type!r} symbol={symbol!r}"
        )
    bid_px, ask_px = float(row[0]), float(row[1])
    return bid_px, ask_px


def mid_path(
    con: duckdb.DuckDBPyConnection,
    *,
    venue: str,
    product_type: str,
    symbol: str,
    start_ns: int,
    end_ns: int,
    resample_ms: int,
) -> pd.DataFrame:
    """Return a resampled mid-price path over [start_ns, end_ns].

    Columns:
        ``ts_ns`` (int64)  — grid point in nanoseconds.
        ``mid``   (float64) — (bid_px + ask_px) / 2 at that grid point,
                              using the last BBO row observed before or at the
                              grid timestamp.  Null for grid points before the
                              first observed BBO row.

    The grid is ``range(start_ns, end_ns + 1, resample_ms * 1_000_000)``.
    Uses ``local_recv_ts`` for ordering.  Binance spot ``exchange_ts`` sentinel
    (= 0) has no impact here.

    Parameters
    ----------
    resample_ms:
        Grid spacing in milliseconds (e.g. 100 → one row every 100 ms).
    """
    step_ns = resample_ms * 1_000_000
    glob = glob_for(venue=venue, product_type=product_type, event="bbo", symbol=symbol)

    # Pull all BBO rows in a generous window (slightly before start to seed last-value).
    # We add one step before start_ns to handle the "last known value" seeding.
    sql = f"""
        SELECT local_recv_ts AS ts_ns,
               (bid_px + ask_px) / 2.0 AS mid
        FROM read_parquet('{glob}', hive_partitioning=true)
        WHERE local_recv_ts <= ?
        ORDER BY local_recv_ts ASC
    """
    try:
        raw = con.execute(sql, [end_ns]).df()
    except duckdb.IOException:
        # No parquet files exist for this partition at all.
        raw = pd.DataFrame({"ts_ns": pd.Series([], dtype="int64"), "mid": pd.Series([], dtype="float64")})

    # Build the grid.
    grid = list(range(start_ns, end_ns + 1, step_ns))
    result_ts: list[int] = []
    result_mid: list[float | None] = []

    if raw.empty:
        return pd.DataFrame({"ts_ns": pd.array(grid, dtype="int64"), "mid": [None] * len(grid)})

    raw_ts = raw["ts_ns"].to_numpy()
    raw_mid = raw["mid"].to_numpy()

    for g in grid:
        # Find the index of the last raw row with ts_ns <= g.
        # np.searchsorted returns the insertion point; we want the last element <=g.
        idx = int(np.searchsorted(raw_ts, g, side="right")) - 1
        if idx < 0:
            result_ts.append(g)
            result_mid.append(None)
        else:
            result_ts.append(g)
            result_mid.append(float(raw_mid[idx]))

    return pd.DataFrame(
        {
            "ts_ns": pd.array(result_ts, dtype="int64"),
            "mid": pd.array(result_mid, dtype="float64"),
        }
    )


def trades_in_window(
    con: duckdb.DuckDBPyConnection,
    *,
    venue: str,
    product_type: str,
    symbol: str,
    start_ns: int,
    end_ns: int,
    only_aggressed: bool = True,
) -> pd.DataFrame:
    """Return trades in [start_ns, end_ns] as a DataFrame.

    Columns:
        ``ts_ns``          (int64)  — local_recv_ts of the trade.
        ``price``          (float64).
        ``size``           (float64).
        ``aggressor_side`` (str)    — taker side: ``"buy"``, ``"sell"``, or
                                      ``"unknown"`` / NULL if not known.

    The recorder stores the taker side in ``TradeEvent.side``
    (``"buy"`` | ``"sell"`` | ``"unknown"``).  When ``only_aggressed=True``
    (default), rows where ``side = 'unknown'`` are dropped.  When
    ``only_aggressed=False``, all rows are returned; ``aggressor_side`` is
    ``"unknown"`` for those rows.

    Venue note: Binance uses ``@trade`` (geo-unblocked), which carries taker
    side.  HL trades carry taker side via ``WsTrade.side``.  In practice both
    venues populate side; ``"unknown"`` rows are rare but possible.
    """
    glob = glob_for(venue=venue, product_type=product_type, event="trade", symbol=symbol)
    where_aggressed = "AND side != 'unknown'" if only_aggressed else ""
    sql = f"""
        SELECT
            local_recv_ts          AS ts_ns,
            price                  AS price,
            size                   AS size,
            side                   AS aggressor_side
        FROM read_parquet('{glob}', hive_partitioning=true)
        WHERE local_recv_ts BETWEEN ? AND ?
          {where_aggressed}
        ORDER BY local_recv_ts ASC
    """
    try:
        df = con.execute(sql, [start_ns, end_ns]).df()
    except duckdb.IOException:
        # No parquet files exist for this partition at all.
        return pd.DataFrame(
            {
                "ts_ns": pd.Series([], dtype="int64"),
                "price": pd.Series([], dtype="float64"),
                "size": pd.Series([], dtype="float64"),
                "aggressor_side": pd.Series([], dtype="object"),
            }
        )
    # Enforce dtypes.
    df["ts_ns"] = df["ts_ns"].astype("int64")
    df["price"] = df["price"].astype("float64")
    df["size"] = df["size"].astype("float64")
    return df


def depth_at_offset_bps(
    con: duckdb.DuckDBPyConnection,
    *,
    venue: str,
    product_type: str,
    symbol: str,
    ts_ns: int,
    side: Literal["bid", "ask"],
    offset_bps: float,
) -> float:
    """Return total size within ``offset_bps`` of the best price on ``side``.

    Looks up the most recent ``book_snapshot`` row at or before ``ts_ns`` and
    sums all levels within ``offset_bps`` of the TOB price.

    For ``side="bid"``: sums bid sizes where
        ``bid_px >= best_bid * (1 - offset_bps / 10_000)``.
    For ``side="ask"``: sums ask sizes where
        ``ask_px <= best_ask * (1 + offset_bps / 10_000)``.

    The recorder stores book levels as parallel arrays in
    ``BookSnapshotEvent``: ``bid_px: list[float]``, ``bid_sz: list[float]``,
    ``ask_px: list[float]``, ``ask_sz: list[float]``.

    Returns
    -------
    Total size as float.  Returns ``0.0`` if no book snapshot exists at or
    before ``ts_ns``.

    Venue note: Binance spot ``exchange_ts`` is 0 (sentinel); this function
    uses ``local_recv_ts`` for ordering and is unaffected.
    """
    glob = glob_for(
        venue=venue, product_type=product_type, event="book_snapshot", symbol=symbol
    )

    # Step 1: find the most recent snapshot row.
    find_sql = f"""
        SELECT bid_px, bid_sz, ask_px, ask_sz
        FROM read_parquet('{glob}', hive_partitioning=true)
        WHERE local_recv_ts <= ?
        ORDER BY local_recv_ts DESC
        LIMIT 1
    """
    try:
        row = con.execute(find_sql, [ts_ns]).fetchone()
    except duckdb.IOException:
        # No parquet files exist for this partition at all.
        return 0.0
    if row is None:
        return 0.0

    bid_pxs, bid_szs, ask_pxs, ask_szs = row[0], row[1], row[2], row[3]

    if side == "bid":
        if not bid_pxs:
            return 0.0
        best_bid = float(bid_pxs[0])
        threshold = best_bid * (1.0 - offset_bps / 10_000.0)
        total = sum(
            float(sz)
            for px, sz in zip(bid_pxs, bid_szs)
            if float(px) >= threshold
        )
    else:  # side == "ask"
        if not ask_pxs:
            return 0.0
        best_ask = float(ask_pxs[0])
        threshold = best_ask * (1.0 + offset_bps / 10_000.0)
        total = sum(
            float(sz)
            for px, sz in zip(ask_pxs, ask_szs)
            if float(px) <= threshold
        )

    return float(total)
