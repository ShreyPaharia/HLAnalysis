"""Cross-venue research panel builder.

Produces an aligned, resampled DataFrame of:
  - HL perp BTC mid + Parkinson realized vol
  - Per outcome-market leg: mid, bid, ask, depth, trade count
  - Per-leg static (forward-filled): class, TTE, moneyness, settlement label

Parquet-cached, keyed by hash of (symbols, start, end, dt, "v1").
Processes data per-day to bound memory.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from hlanalysis.analysis.helpers import asof_locf, duck
from hlanalysis.research.outcome_markets import load_market_reference, load_settlements

_NS = 1_000_000_000
_CACHE_VERSION = "v1"


def _perp_bbo_glob(data_root: str) -> str:
    return str(
        Path(data_root)
        / "venue=hyperliquid"
        / "product_type=perp"
        / "mechanism=clob"
        / "event=bbo"
        / "symbol=BTC"
        / "date=*"
        / "hour=all"
        / "*.parquet"
    )


def _bin_bbo_glob(data_root: str, symbol: str) -> str:
    # URL-encode '#' for path — but parquet glob uses literal path, not URL
    return str(
        Path(data_root)
        / "venue=hyperliquid"
        / "product_type=prediction_binary"
        / "mechanism=clob"
        / "event=bbo"
        / f"symbol={symbol}"
        / "date=*"
        / "hour=all"
        / "*.parquet"
    )


def _bin_trade_glob(data_root: str, symbol: str) -> str:
    return str(
        Path(data_root)
        / "venue=hyperliquid"
        / "product_type=prediction_binary"
        / "mechanism=clob"
        / "event=trade"
        / f"symbol={symbol}"
        / "date=*"
        / "hour=all"
        / "*.parquet"
    )


def _perp_ohlc_glob(data_root: str) -> str:
    return str(
        Path(data_root)
        / "venue=hyperliquid"
        / "product_type=perp"
        / "mechanism=clob"
        / "event=bbo"
        / "symbol=BTC"
        / "date=*"
        / "hour=all"
        / "*.parquet"
    )


def _cache_key(symbols: list[str], start: str, end: str, dt_seconds: int) -> str:
    payload = json.dumps(
        {"symbols": sorted(symbols), "start": start, "end": end, "dt": dt_seconds, "version": _CACHE_VERSION},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _date_range(start: str, end: str) -> list[str]:
    """Return list of date strings 'YYYY-MM-DD' from start to end inclusive."""
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    dates = []
    cur = s
    while cur <= e:
        dates.append(cur.isoformat())
        cur += dt.timedelta(days=1)
    return dates


def _day_bounds_ns(date_str: str) -> tuple[int, int]:
    """Return (start_ns, end_ns) for a calendar day in UTC nanoseconds."""
    d = dt.date.fromisoformat(date_str)
    start = dt.datetime(d.year, d.month, d.day, tzinfo=dt.UTC)
    # end_ns is last nanosecond of the day (exclusive next day - 1ns in seconds units)
    end_ns = int((start + dt.timedelta(days=1)).timestamp() * _NS) - 1
    return int(start.timestamp() * _NS), end_ns


def _parkinson_vol_rolling(
    con: duckdb.DuckDBPyConnection,
    glob: str,
    start_ns: int,
    end_ns: int,
    grid: np.ndarray,
    lookback_ns: int,
) -> np.ndarray:
    """Compute Parkinson vol at each grid point using a rolling lookback window."""
    import math

    # Load raw BBO for range [start_ns - lookback_ns, end_ns]
    sql = f"""
        SELECT local_recv_ts AS ts_ns,
               (bid_px + ask_px) / 2.0 AS mid
        FROM read_parquet('{glob}', union_by_name=true)
        WHERE local_recv_ts BETWEEN ? AND ?
        ORDER BY local_recv_ts ASC
    """
    try:
        df = con.execute(sql, [max(0, start_ns - lookback_ns), end_ns]).df()
    except duckdb.IOException:
        return np.full(len(grid), np.nan)

    if len(df) < 2:
        return np.full(len(grid), np.nan)

    ts = df["ts_ns"].to_numpy(dtype="int64")
    mid = df["mid"].to_numpy(dtype="float64")

    # Build 60s OHLC bars from raw ticks
    bar_ns = 60 * _NS
    bar_grid = np.arange(ts[0] // bar_ns * bar_ns, end_ns + bar_ns, bar_ns, dtype="int64")

    bar_records = []
    for i in range(len(bar_grid) - 1):
        bs, be = bar_grid[i], bar_grid[i + 1]
        mask = (ts >= bs) & (ts < be)
        if mask.sum() == 0:
            continue
        sub_mid = mid[mask]
        bar_records.append(
            {
                "bar_ts": bs,
                "h": float(np.max(sub_mid)),
                "l": float(np.min(sub_mid)),
            }
        )

    if not bar_records:
        return np.full(len(grid), np.nan)

    bar_df = pd.DataFrame(bar_records)
    bar_ts = bar_df["bar_ts"].to_numpy(dtype="int64")
    h_arr = bar_df["h"].to_numpy(dtype="float64")
    l_arr = bar_df["l"].to_numpy(dtype="float64")

    # For each grid point, compute Parkinson vol over lookback window
    out = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        # bars within [g - lookback_ns, g]
        mask = (bar_ts >= g - lookback_ns) & (bar_ts <= g)
        if mask.sum() < 2:
            continue
        h_w = h_arr[mask]
        l_w = l_arr[mask]
        valid = (h_w > 0) & (l_w > 0)
        if valid.sum() < 2:
            continue
        log_hl = np.log(h_w[valid] / l_w[valid])
        park_var = (1.0 / (4.0 * math.log(2.0))) * np.mean(log_hl**2)
        # Annualise per bar (60s bars)
        out[i] = math.sqrt(park_var * (365.25 * 86400.0) / 60.0)

    return out


def _load_sym_bbo_day(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    symbol: str,
    date_str: str,
    grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Load BBO for a single symbol+day and LOCF onto grid."""
    glob = str(
        Path(data_root)
        / "venue=hyperliquid"
        / "product_type=prediction_binary"
        / "mechanism=clob"
        / "event=bbo"
        / f"symbol={symbol}"
        / f"date={date_str}"
        / "hour=all"
        / "*.parquet"
    )

    sql = f"""
        SELECT local_recv_ts AS ts_ns, bid_px, ask_px
        FROM read_parquet('{glob}', union_by_name=true)
        ORDER BY local_recv_ts ASC
    """
    try:
        df = con.execute(sql).df()
    except duckdb.IOException:
        nan_arr = np.full(len(grid), np.nan)
        return {
            "mid": nan_arr.copy(),
            "bid": nan_arr.copy(),
            "ask": nan_arr.copy(),
        }

    if df.empty:
        nan_arr = np.full(len(grid), np.nan)
        return {
            "mid": nan_arr.copy(),
            "bid": nan_arr.copy(),
            "ask": nan_arr.copy(),
        }

    ts = df["ts_ns"].to_numpy(dtype="int64")
    bid = df["bid_px"].to_numpy(dtype="float64")
    ask = df["ask_px"].to_numpy(dtype="float64")
    mid = (bid + ask) / 2.0

    return {
        "mid": asof_locf(grid, ts, mid),
        "bid": asof_locf(grid, ts, bid),
        "ask": asof_locf(grid, ts, ask),
    }


def _load_sym_trade_count_day(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    symbol: str,
    date_str: str,
    grid: np.ndarray,
    dt_seconds: int,
) -> np.ndarray:
    """Count trades per grid bin for a single symbol+day."""
    glob = str(
        Path(data_root)
        / "venue=hyperliquid"
        / "product_type=prediction_binary"
        / "mechanism=clob"
        / "event=trade"
        / f"symbol={symbol}"
        / f"date={date_str}"
        / "hour=all"
        / "*.parquet"
    )

    sql = f"""
        SELECT local_recv_ts AS ts_ns
        FROM read_parquet('{glob}', union_by_name=true)
        ORDER BY local_recv_ts ASC
    """
    try:
        df = con.execute(sql).df()
    except duckdb.IOException:
        return np.zeros(len(grid), dtype="int64")

    if df.empty:
        return np.zeros(len(grid), dtype="int64")

    ts = df["ts_ns"].to_numpy(dtype="int64")
    counts = np.zeros(len(grid), dtype="int64")

    if len(grid) == 0:
        return counts

    # Assign each trade to a grid bin
    bin_idxs = np.searchsorted(grid, ts, side="right") - 1
    valid = (bin_idxs >= 0) & (bin_idxs < len(grid))
    np.add.at(counts, bin_idxs[valid], 1)
    return counts


def _load_sym_depth_day(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    symbol: str,
    date_str: str,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load top-1 depth (bid_sz, ask_sz) LOCF onto grid for a single symbol+day."""
    glob = str(
        Path(data_root)
        / "venue=hyperliquid"
        / "product_type=prediction_binary"
        / "mechanism=clob"
        / "event=bbo"
        / f"symbol={symbol}"
        / f"date={date_str}"
        / "hour=all"
        / "*.parquet"
    )

    sql = f"""
        SELECT local_recv_ts AS ts_ns, bid_sz, ask_sz
        FROM read_parquet('{glob}', union_by_name=true)
        ORDER BY local_recv_ts ASC
    """
    try:
        df = con.execute(sql).df()
    except duckdb.IOException:
        nan_arr = np.full(len(grid), np.nan)
        return nan_arr.copy(), nan_arr.copy()

    if df.empty:
        nan_arr = np.full(len(grid), np.nan)
        return nan_arr.copy(), nan_arr.copy()

    ts = df["ts_ns"].to_numpy(dtype="int64")
    bid_sz = df["bid_sz"].to_numpy(dtype="float64")
    ask_sz = df["ask_sz"].to_numpy(dtype="float64")

    return asof_locf(grid, ts, bid_sz), asof_locf(grid, ts, ask_sz)


def build_panel(
    symbols: list[str],
    start: str,
    end: str,
    dt_seconds: int,
    data_root: str,
    cache_dir: str | None = None,
    fresh: bool = False,
) -> pd.DataFrame:
    """Build an aligned, resampled research panel.

    Columns
    -------
    timestamp             datetime64[ns, UTC] — grid timestamps
    perp_mid              float64  — HL BTC perp mid price (LOCF)
    perp_realized_vol     float64  — rolling 1h Parkinson vol (annualised)
    {sym}_mid             float64  — outcome-leg mid price (LOCF)
    {sym}_bid             float64  — best bid (LOCF)
    {sym}_ask             float64  — best ask (LOCF)
    {sym}_depth_bid       float64  — TOB bid size (LOCF)
    {sym}_depth_ask       float64  — TOB ask size (LOCF)
    {sym}_trade_count     int64    — trades in this dt bin
    {sym}_class           str      — forward-filled market class
    {sym}_tte_s           float64  — seconds to expiry (static per day)
    {sym}_moneyness       float64  — perp_mid / target_price (None for buckets)
    {sym}_settlement_label str     — 'yes_won', 'no_won', or NaN (filled at settlement)

    Parameters
    ----------
    symbols:
        List of leg symbols (e.g. ['#100', '#101', '#1070']).
    start, end:
        Date strings 'YYYY-MM-DD', inclusive.
    dt_seconds:
        Grid spacing in seconds.
    data_root:
        Absolute path to the data root (e.g. '../../data').
    cache_dir:
        Directory for Parquet cache.  If None, caching is disabled.
    fresh:
        If True, ignore cached data and recompute.
    """
    # --- Cache lookup ---
    if cache_dir and not fresh:
        cache_path = Path(cache_dir) / f"panel_{_cache_key(symbols, start, end, dt_seconds)}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

    con = duck()
    data_root_str = str(Path(data_root).resolve())

    # Load reference data once
    ref_df = load_market_reference(con, data_root_str)
    settlements_df = load_settlements(con, data_root_str)

    # Build per-symbol metadata lookup
    sym_meta: dict[str, dict] = {}
    if not ref_df.empty:
        for _, row in ref_df.iterrows():
            sym_meta[row["symbol"]] = row.to_dict()

    # Settlement lookup: Yes-leg symbol -> won label
    # settlement records are for Yes legs (won=True)
    # For No legs: the Yes-leg partner won, so No leg lost
    settle_lookup: dict[str, str] = {}
    if not settlements_df.empty:
        for _, row in settlements_df.iterrows():
            yes_sym = row["symbol"]
            settle_lookup[yes_sym] = "yes_won"
            # Derive No-leg partner: replace last digit '0' with '1'
            no_sym = yes_sym[:-1] + "1"
            settle_lookup[no_sym] = "no_won"

    dates = _date_range(start, end)
    step_ns = dt_seconds * _NS

    day_frames: list[pd.DataFrame] = []

    for date_str in dates:
        day_start_ns, day_end_ns = _day_bounds_ns(date_str)
        # Build grid for this day
        grid = np.arange(day_start_ns, day_end_ns + 1, step_ns, dtype="int64")
        if len(grid) == 0:
            continue

        row_data: dict[str, np.ndarray] = {}
        row_data["ts_ns"] = grid

        # --- Perp BTC mid ---
        perp_glob = _perp_bbo_glob(data_root_str)
        perp_sql = f"""
            SELECT local_recv_ts AS ts_ns, (bid_px + ask_px) / 2.0 AS mid
            FROM read_parquet('{perp_glob}', union_by_name=true)
            WHERE local_recv_ts BETWEEN ? AND ?
            ORDER BY local_recv_ts ASC
        """
        try:
            perp_df = con.execute(perp_sql, [day_start_ns, day_end_ns]).df()
        except duckdb.IOException:
            perp_df = pd.DataFrame({"ts_ns": pd.Series([], dtype="int64"), "mid": pd.Series([], dtype="float64")})

        if perp_df.empty:
            row_data["perp_mid"] = np.full(len(grid), np.nan)
        else:
            row_data["perp_mid"] = asof_locf(
                grid,
                perp_df["ts_ns"].to_numpy(dtype="int64"),
                perp_df["mid"].to_numpy(dtype="float64"),
            )

        # --- Perp Parkinson vol (1h lookback) ---
        lookback_ns = 3600 * _NS
        perp_glob_all = str(
            Path(data_root_str)
            / "venue=hyperliquid"
            / "product_type=perp"
            / "mechanism=clob"
            / "event=bbo"
            / "symbol=BTC"
            / "date=*"
            / "hour=all"
            / "*.parquet"
        )
        row_data["perp_realized_vol"] = _parkinson_vol_rolling(
            con, perp_glob_all, day_start_ns, day_end_ns, grid, lookback_ns
        )

        # --- Per-symbol columns ---
        for sym in symbols:
            meta = sym_meta.get(sym, {})

            bbo_data = _load_sym_bbo_day(con, data_root_str, sym, date_str, grid)
            row_data[f"{sym}_mid"] = bbo_data["mid"]
            row_data[f"{sym}_bid"] = bbo_data["bid"]
            row_data[f"{sym}_ask"] = bbo_data["ask"]

            depth_bid, depth_ask = _load_sym_depth_day(con, data_root_str, sym, date_str, grid)
            row_data[f"{sym}_depth_bid"] = depth_bid
            row_data[f"{sym}_depth_ask"] = depth_ask

            row_data[f"{sym}_trade_count"] = _load_sym_trade_count_day(
                con, data_root_str, sym, date_str, grid, dt_seconds
            )

            # Static (constant per day, forward-filled later)
            market_class = meta.get("market_class", None)
            row_data[f"{sym}_class"] = np.array([market_class] * len(grid), dtype=object)

            # TTE: seconds from each grid point to expiry
            expiry = meta.get("expiry", None)
            if expiry is not None:
                expiry_ns = int(expiry.timestamp() * _NS)
                tte = (expiry_ns - grid) / float(_NS)
                tte = np.where(tte < 0, 0.0, tte)
            else:
                tte = np.full(len(grid), np.nan)
            row_data[f"{sym}_tte_s"] = tte

            # Moneyness: perp_mid / target_price
            target_price = meta.get("target_price", None)
            if target_price is not None and target_price > 0:
                mono = row_data["perp_mid"] / float(target_price)
            else:
                mono = np.full(len(grid), np.nan)
            row_data[f"{sym}_moneyness"] = mono

            # Settlement label
            label = settle_lookup.get(sym)
            row_data[f"{sym}_settlement_label"] = np.array([label] * len(grid), dtype=object)

        day_frames.append(pd.DataFrame(row_data))

    if not day_frames:
        return pd.DataFrame()

    panel = pd.concat(day_frames, ignore_index=True)

    # Convert ts_ns to UTC datetime
    panel["timestamp"] = pd.to_datetime(panel["ts_ns"], unit="ns", utc=True)
    panel = panel.drop(columns=["ts_ns"])

    # Reorder: timestamp first
    cols = ["timestamp", "perp_mid", "perp_realized_vol"]
    for sym in symbols:
        cols += [
            f"{sym}_mid",
            f"{sym}_bid",
            f"{sym}_ask",
            f"{sym}_depth_bid",
            f"{sym}_depth_ask",
            f"{sym}_trade_count",
            f"{sym}_class",
            f"{sym}_tte_s",
            f"{sym}_moneyness",
            f"{sym}_settlement_label",
        ]
    # Add any extra cols not in our list
    extra = [c for c in panel.columns if c not in cols]
    panel = panel[cols + extra]

    # Cache
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = Path(cache_dir) / f"panel_{_cache_key(symbols, start, end, dt_seconds)}.parquet"
        panel.to_parquet(cache_path, index=False)

    return panel
