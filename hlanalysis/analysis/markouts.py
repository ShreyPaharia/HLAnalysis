"""Trade markout helpers for analysis notebooks.

A markout measures how much the mid-price moves in the direction of a trade
after the trade occurs.  Positive markout = informed flow (the market moved in
the direction of the aggressor).

All functions are thin, stateless wrappers over DuckDB queries against
hive-partitioned parquet written by the recorder.  The caller is responsible
for managing the DuckDB connection lifecycle (use ``helpers.duck()``).
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from hlanalysis.analysis.book import trades_in_window
from hlanalysis.analysis.helpers import glob_for

_NS_PER_S = 1_000_000_000


def _load_bbo_series(
    con: duckdb.DuckDBPyConnection,
    *,
    venue: str,
    product_type: str,
    symbol: str,
    start_ns: int,
    end_ns: int,
) -> pd.DataFrame:
    """Load all BBO rows in the window plus any seed row just before start_ns.

    Returns a DataFrame with columns:
        ``ts_ns`` (int64)   — local_recv_ts
        ``mid``   (float64) — (bid_px + ask_px) / 2
    sorted ascending by ts_ns.
    """
    glob = glob_for(venue=venue, product_type=product_type, event="bbo", symbol=symbol)
    sql = f"""
        SELECT local_recv_ts AS ts_ns,
               (bid_px + ask_px) / 2.0 AS mid
        FROM read_parquet('{glob}', hive_partitioning=true)
        WHERE local_recv_ts <= ?
        ORDER BY local_recv_ts ASC
    """
    try:
        df = con.execute(sql, [end_ns]).df()
    except duckdb.IOException:
        df = pd.DataFrame(
            {
                "ts_ns": pd.Series([], dtype="int64"),
                "mid": pd.Series([], dtype="float64"),
            }
        )
    df["ts_ns"] = df["ts_ns"].astype("int64")
    df["mid"] = df["mid"].astype("float64")
    return df.sort_values("ts_ns").reset_index(drop=True)


def _asof_mid(
    trade_ts: np.ndarray,
    bbo_ts: np.ndarray,
    bbo_mid: np.ndarray,
) -> np.ndarray:
    """Return the last-known mid for each trade timestamp.

    For each element in ``trade_ts``, find the largest value in ``bbo_ts`` that
    is <= the trade timestamp and return the corresponding mid.  If no such BBO
    exists, return NaN.

    Parameters
    ----------
    trade_ts:
        Sorted int64 array of trade local_recv_ts values.
    bbo_ts:
        Sorted int64 array of BBO local_recv_ts values.
    bbo_mid:
        float64 array of mid values parallel to bbo_ts.
    """
    # np.searchsorted(bbo_ts, t, side='right') - 1 gives the index of the last
    # bbo_ts element that is <= t.  If the result is -1, no BBO precedes t.
    idxs = np.searchsorted(bbo_ts, trade_ts, side="right") - 1
    result = np.where(idxs >= 0, bbo_mid[idxs], np.nan)
    return result


def trade_markouts(
    con: duckdb.DuckDBPyConnection,
    *,
    venue: str,
    product_type: str,
    symbol: str,
    start_ns: int,
    end_ns: int,
    horizons_s: tuple[int, ...] = (1, 5, 30, 60),
) -> pd.DataFrame:
    """Return one row per aggressed trade with markouts at each horizon.

    Markout = (mid_at_trade+h - mid_at_trade) * sign(aggressor_side).

    Sign convention:
        * buy aggressor  -> sign = +1
        * sell aggressor -> sign = -1

    So positive markout means the mid moved in the direction of the aggressor
    (informed flow), regardless of whether the aggressor bought or sold.

    Parameters
    ----------
    con:
        Live DuckDB connection (caller manages lifecycle).
    venue, product_type, symbol:
        Hive-partition selectors.
    start_ns, end_ns:
        Window in nanoseconds (local_recv_ts scale).
    horizons_s:
        Tuple of forward horizons in seconds.

    Returns
    -------
    DataFrame with columns:
        ``ts_ns``            (int64)   — trade local_recv_ts
        ``price``            (float64) — trade price
        ``size``             (float64) — trade size
        ``aggressor_side``   (str)     — "buy" or "sell"
        ``mid_at_trade``     (float64) — last-known mid at trade time (NaN if
                                         no BBO exists at or before ts_ns)
        ``mid_at_h_{h}s``    (float64) — last-known mid at ts_ns + h seconds
                                         (NaN if no BBO in window at that offset)
        ``markout_h_{h}s``   (float64) — markout in price units for horizon h
                                         (NaN if either mid is NaN)

    Notes
    -----
    - ``only_aggressed=True`` is passed to ``trades_in_window``; rows with
      ``side='unknown'`` are excluded upstream.
    - A trade at the very start of the window with no prior BBO yields
      ``mid_at_trade = NaN`` and all markout columns as NaN.
    - A horizon h that falls past ``end_ns`` yields ``mid_at_h_{h}s = NaN``.
    """
    trades = trades_in_window(
        con,
        venue=venue,
        product_type=product_type,
        symbol=symbol,
        start_ns=start_ns,
        end_ns=end_ns,
        only_aggressed=True,
    )

    if trades.empty:
        # Build correctly-typed empty DataFrame with all expected columns.
        cols: dict[str, pd.Series] = {
            "ts_ns": pd.Series([], dtype="int64"),
            "price": pd.Series([], dtype="float64"),
            "size": pd.Series([], dtype="float64"),
            "aggressor_side": pd.Series([], dtype="object"),
            "mid_at_trade": pd.Series([], dtype="float64"),
        }
        for h in horizons_s:
            cols[f"mid_at_h_{h}s"] = pd.Series([], dtype="float64")
            cols[f"markout_h_{h}s"] = pd.Series([], dtype="float64")
        return pd.DataFrame(cols)

    bbo = _load_bbo_series(
        con,
        venue=venue,
        product_type=product_type,
        symbol=symbol,
        start_ns=start_ns,
        end_ns=end_ns,
    )

    trade_ts = trades["ts_ns"].to_numpy(dtype="int64")

    if bbo.empty:
        bbo_ts = np.array([], dtype="int64")
        bbo_mid = np.array([], dtype="float64")
    else:
        bbo_ts = bbo["ts_ns"].to_numpy(dtype="int64")
        bbo_mid = bbo["mid"].to_numpy(dtype="float64")

    # Compute mid at each trade timestamp.
    mid_at_trade = _asof_mid(trade_ts, bbo_ts, bbo_mid)

    # sign: +1 for buy, -1 for sell.
    sign = np.where(trades["aggressor_side"].to_numpy() == "buy", 1.0, -1.0)

    result = trades.copy()
    result["mid_at_trade"] = mid_at_trade

    for h in horizons_s:
        horizon_ns = int(h) * _NS_PER_S
        offset_ts = trade_ts + horizon_ns

        # Horizon mid: asof-join at offset_ts, but only if offset_ts <= end_ns.
        mid_at_h = _asof_mid(offset_ts, bbo_ts, bbo_mid)

        # Null out mids where the horizon target falls past end_ns.
        # (The asof join would return the last BBO in the window, which is
        # misleading — it's the last available point, not the point at t+h.)
        past_end = offset_ts > end_ns
        mid_at_h = np.where(past_end, np.nan, mid_at_h)

        result[f"mid_at_h_{h}s"] = mid_at_h
        result[f"markout_h_{h}s"] = (mid_at_h - mid_at_trade) * sign

    return result.reset_index(drop=True)
