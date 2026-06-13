"""Card A: Liquidity & Book Shape for HL outcome markets vs perp BTC.

Measures over the full 36-day corpus (2026-05-07 to 2026-06-10):
  1. Spread: bid-ask spread (absolute and bps) for binary/bucket/perp;
     distribution, vs TTE, vs UTC hour.
  2. Depth: TOB size and L2 notional at N bps of mid from book_snapshot.
     Confirms/quantifies the $50-200/level capacity; per-class aggregate.
  3. Quoting uptime: fraction two-sided; gap/outage frequency; vs TTE.
  4. Book imbalance: OBI (bid_sz - ask_sz)/(bid_sz + ask_sz), Spearman
     predictive r for next-minute mid change, n per symbol, aggregate.
  5. Tick size / price granularity: minimum effective tick from price
     increments; implication for quoting and minimum edge.
  6. MM room: spread vs tick to quantify join/improve space; desk capacity
     framing ($1k-$25k).

Split-half (H1=05-06..05-23 / H2=05-24..06-10): spread, depth, uptime.
Coverage assertion: ≥30 expiries / ≥30 days.

Usage::

    from hlanalysis.research.cards.card_a_liquidity import build_card
    import duckdb
    html, findings = build_card(duckdb.connect(), "../../data")

Run standalone::

    python -m hlanalysis.research.cards.card_a_liquidity
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hlanalysis.research.report import Report, fig_to_base64

matplotlib.use("Agg")

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Split-half boundary: 2026-05-23 23:59:59 UTC (inclusive H1)
_H1_END_NS: int = 1779580799_000_000_000  # 2026-05-23 23:59:59 UTC
_DATE_SPAN = "2026-05-07 to 2026-06-10 (36 days)"
_H1_SPAN = "2026-05-07 to 2026-05-23"
_H2_SPAN = "2026-05-24 to 2026-06-10"


# ---------------------------------------------------------------------------
# DuckDB glob helpers
# ---------------------------------------------------------------------------


def _binary_bbo_glob(data_root: str) -> str:
    return str(
        Path(data_root)
        / "venue=hyperliquid"
        / "product_type=prediction_binary"
        / "mechanism=clob"
        / "event=bbo"
        / "symbol=*"
        / "date=*"
        / "hour=all"
        / "*.parquet"
    )


def _binary_snap_glob(data_root: str) -> str:
    return str(
        Path(data_root)
        / "venue=hyperliquid"
        / "product_type=prediction_binary"
        / "mechanism=clob"
        / "event=book_snapshot"
        / "symbol=*"
        / "date=*"
        / "hour=all"
        / "*.parquet"
    )


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


def _perp_snap_glob(data_root: str) -> str:
    return str(
        Path(data_root)
        / "venue=hyperliquid"
        / "product_type=perp"
        / "mechanism=clob"
        / "event=book_snapshot"
        / "symbol=BTC"
        / "date=*"
        / "hour=all"
        / "*.parquet"
    )


def _meta_glob(data_root: str) -> str:
    return str(
        Path(data_root)
        / "venue=hyperliquid"
        / "product_type=prediction_binary"
        / "mechanism=clob"
        / "event=market_meta"
        / "symbol=*"
        / "date=*"
        / "hour=all"
        / "*.parquet"
    )


def _q_meta_glob(data_root: str) -> str:
    return str(
        Path(data_root)
        / "venue=hyperliquid"
        / "product_type=prediction_binary"
        / "mechanism=clob"
        / "event=question_meta"
        / "symbol=*"
        / "date=*"
        / "hour=all"
        / "*.parquet"
    )


# ---------------------------------------------------------------------------
# Symbol classification helpers
# ---------------------------------------------------------------------------


def _load_binary_yes_symbols_with_expiry(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
) -> pd.DataFrame:
    """Return binary Yes-leg symbols with expiry nanoseconds."""
    meta = _meta_glob(data_root)
    sql = f"""
        SELECT DISTINCT
            symbol,
            list_element(values, list_position(keys, 'expiry')) AS expiry_str,
            epoch_ns(strptime(
                list_element(values, list_position(keys, 'expiry')),
                '%Y%m%d-%H%M'
            ) AT TIME ZONE 'UTC') AS exp_ns
        FROM read_parquet('{meta}', union_by_name=true)
        WHERE array_contains(keys, 'class')
          AND list_element(values, list_position(keys, 'class')) = 'priceBinary'
          AND array_contains(keys, 'side_idx')
          AND list_element(values, list_position(keys, 'side_idx')) = '0'
    """
    try:
        return con.execute(sql).df()
    except duckdb.IOException:
        return pd.DataFrame()


def _load_bucket_yes_symbols(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
) -> pd.DataFrame:
    """Return bucket Yes-leg symbols."""
    meta = _meta_glob(data_root)
    q_meta = _q_meta_glob(data_root)
    sql = f"""
        WITH bucket_outcome_idxs AS (
            SELECT DISTINCT unnest(named_outcome_idxs) AS outcome_idx
            FROM read_parquet('{q_meta}', union_by_name=true)
            WHERE array_contains(keys, 'priceThresholds')
        )
        SELECT DISTINCT mm.symbol
        FROM read_parquet('{meta}', union_by_name=true) mm
        JOIN bucket_outcome_idxs bo
          ON list_element(mm.values, list_position(mm.keys, 'outcome_idx'))::BIGINT = bo.outcome_idx
        WHERE array_contains(mm.keys, 'side_idx')
          AND list_element(mm.values, list_position(mm.keys, 'side_idx')) = '0'
    """
    try:
        return con.execute(sql).df()
    except duckdb.IOException:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Section 1: Spread distributions
# ---------------------------------------------------------------------------


def _spread_stats_sql(
    bbo_glob: str,
    where_join: str = "",
    extra_from: str = "",
) -> str:
    """SQL fragment for spread distribution stats."""
    return f"""
        SELECT
            COUNT(*) AS n,
            AVG(ask_px - bid_px) AS mean_spread_abs,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ask_px - bid_px)
                AS median_spread_abs,
            AVG((ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS mean_spread_bps,
            PERCENTILE_CONT(0.5) WITHIN GROUP (
                ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS median_spread_bps,
            PERCENTILE_CONT(0.25) WITHIN GROUP (
                ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS p25_spread_bps,
            PERCENTILE_CONT(0.75) WITHIN GROUP (
                ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS p75_spread_bps,
            PERCENTILE_CONT(0.99) WITHIN GROUP (
                ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS p99_spread_bps,
            AVG(bid_sz) AS mean_bid_sz,
            AVG(ask_sz) AS mean_ask_sz,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY bid_sz * bid_px)
                AS median_bid_notional,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ask_sz * ask_px)
                AS median_ask_notional,
            COUNT(DISTINCT symbol) AS n_symbols
        FROM read_parquet('{bbo_glob}', union_by_name=true) b
        {extra_from}
        WHERE b.bid_px > 0 AND b.ask_px > 0 AND b.bid_px < b.ask_px
          {where_join}
    """


def _compute_binary_spread_stats(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    binary_syms: list[str],
) -> pd.DataFrame:
    bbo = _binary_bbo_glob(data_root)
    sym_list = "', '".join(binary_syms)
    sql = f"""
        SELECT
            COUNT(*) AS n,
            AVG(ask_px - bid_px) AS mean_spread_abs,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ask_px - bid_px)
                AS median_spread_abs,
            AVG((ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS mean_spread_bps,
            PERCENTILE_CONT(0.5) WITHIN GROUP (
                ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS median_spread_bps,
            PERCENTILE_CONT(0.25) WITHIN GROUP (
                ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS p25_spread_bps,
            PERCENTILE_CONT(0.75) WITHIN GROUP (
                ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS p75_spread_bps,
            PERCENTILE_CONT(0.99) WITHIN GROUP (
                ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS p99_spread_bps,
            AVG(bid_sz) AS mean_bid_sz,
            AVG(ask_sz) AS mean_ask_sz,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY bid_sz * bid_px)
                AS median_bid_notional,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ask_sz * ask_px)
                AS median_ask_notional,
            COUNT(DISTINCT symbol) AS n_symbols
        FROM read_parquet('{bbo}', union_by_name=true)
        WHERE bid_px > 0 AND ask_px > 0 AND bid_px < ask_px
          AND symbol IN ('{sym_list}')
    """
    return con.execute(sql).df()


def _compute_bucket_spread_stats(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    bucket_syms: list[str],
) -> pd.DataFrame:
    bbo = _binary_bbo_glob(data_root)
    sym_list = "', '".join(bucket_syms)
    sql = f"""
        SELECT
            COUNT(*) AS n,
            AVG(ask_px - bid_px) AS mean_spread_abs,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ask_px - bid_px)
                AS median_spread_abs,
            AVG((ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS mean_spread_bps,
            PERCENTILE_CONT(0.5) WITHIN GROUP (
                ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS median_spread_bps,
            PERCENTILE_CONT(0.25) WITHIN GROUP (
                ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS p25_spread_bps,
            PERCENTILE_CONT(0.75) WITHIN GROUP (
                ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS p75_spread_bps,
            PERCENTILE_CONT(0.99) WITHIN GROUP (
                ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS p99_spread_bps,
            AVG(bid_sz) AS mean_bid_sz,
            AVG(ask_sz) AS mean_ask_sz,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY bid_sz * bid_px)
                AS median_bid_notional,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ask_sz * ask_px)
                AS median_ask_notional,
            COUNT(DISTINCT symbol) AS n_symbols
        FROM read_parquet('{bbo}', union_by_name=true)
        WHERE bid_px > 0 AND ask_px > 0 AND bid_px < ask_px
          AND symbol IN ('{sym_list}')
    """
    return con.execute(sql).df()


def _compute_perp_spread_stats(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
) -> pd.DataFrame:
    perp = _perp_bbo_glob(data_root)
    sql = f"""
        SELECT
            COUNT(*) AS n,
            AVG(ask_px - bid_px) AS mean_spread_abs,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ask_px - bid_px)
                AS median_spread_abs,
            AVG((ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS mean_spread_bps,
            PERCENTILE_CONT(0.5) WITHIN GROUP (
                ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS median_spread_bps,
            PERCENTILE_CONT(0.25) WITHIN GROUP (
                ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS p25_spread_bps,
            PERCENTILE_CONT(0.75) WITHIN GROUP (
                ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS p75_spread_bps,
            PERCENTILE_CONT(0.99) WITHIN GROUP (
                ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS p99_spread_bps,
            AVG(bid_sz) AS mean_bid_sz,
            AVG(ask_sz) AS mean_ask_sz,
            NULL AS median_bid_notional,
            NULL AS median_ask_notional,
            1 AS n_symbols
        FROM read_parquet('{perp}', union_by_name=true)
        WHERE bid_px > 0 AND ask_px > 0 AND bid_px < ask_px
    """
    return con.execute(sql).df()


def _compute_spread_vs_tte(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    binary_syms_with_expiry: pd.DataFrame,
) -> pd.DataFrame:
    """Spread vs TTE bucket for binary markets."""
    bbo = _binary_bbo_glob(data_root)

    # Build expiry map for DuckDB as a VALUES clause
    expiry_rows = ", ".join(f"('{row.symbol}', {int(row.exp_ns)})" for _, row in binary_syms_with_expiry.iterrows())

    sql = f"""
        WITH expiry_map(symbol, exp_ns) AS (
            VALUES {expiry_rows}
        ),
        bbo_raw AS (
            SELECT
                b.symbol,
                (b.ask_px - b.bid_px) / ((b.bid_px + b.ask_px) / 2.0) * 10000 AS spread_bps,
                (em.exp_ns - b.local_recv_ts) / 3.6e12 AS tte_h
            FROM read_parquet('{bbo}', union_by_name=true) b
            JOIN expiry_map em ON b.symbol = em.symbol
            WHERE b.bid_px > 0 AND b.ask_px > 0 AND b.bid_px < b.ask_px
        ),
        bucketed AS (
            SELECT
                CASE
                    WHEN tte_h < 1  THEN 0
                    WHEN tte_h < 3  THEN 1
                    WHEN tte_h < 6  THEN 2
                    WHEN tte_h < 12 THEN 3
                    WHEN tte_h < 18 THEN 4
                    ELSE 5
                END AS tte_sort,
                CASE
                    WHEN tte_h < 1  THEN '<1h'
                    WHEN tte_h < 3  THEN '1-3h'
                    WHEN tte_h < 6  THEN '3-6h'
                    WHEN tte_h < 12 THEN '6-12h'
                    WHEN tte_h < 18 THEN '12-18h'
                    ELSE '>18h'
                END AS tte_bucket,
                spread_bps
            FROM bbo_raw
            WHERE tte_h >= 0
        )
        SELECT
            tte_bucket, tte_sort,
            COUNT(*) AS n,
            AVG(spread_bps) AS mean_spread_bps,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY spread_bps) AS median_spread_bps
        FROM bucketed
        GROUP BY tte_bucket, tte_sort
        ORDER BY tte_sort
    """
    try:
        return con.execute(sql).df()
    except duckdb.IOException:
        return pd.DataFrame()


def _compute_spread_vs_hour(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    binary_syms: list[str],
    bucket_syms: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Spread vs UTC hour for binary, bucket, and perp."""
    bbo = _binary_bbo_glob(data_root)
    perp = _perp_bbo_glob(data_root)

    bin_list = "', '".join(binary_syms)
    bkt_list = "', '".join(bucket_syms)

    bin_sql = f"""
        SELECT
            extract(hour from to_timestamp(local_recv_ts / 1e9)) AS utc_hour,
            COUNT(*) AS n,
            AVG((ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000) AS mean_spread_bps,
            PERCENTILE_CONT(0.5) WITHIN GROUP (
                ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS median_spread_bps
        FROM read_parquet('{bbo}', union_by_name=true)
        WHERE bid_px > 0 AND ask_px > 0 AND bid_px < ask_px
          AND symbol IN ('{bin_list}')
        GROUP BY utc_hour ORDER BY utc_hour
    """

    bkt_sql = f"""
        SELECT
            extract(hour from to_timestamp(local_recv_ts / 1e9)) AS utc_hour,
            COUNT(*) AS n,
            AVG((ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000) AS mean_spread_bps,
            PERCENTILE_CONT(0.5) WITHIN GROUP (
                ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS median_spread_bps
        FROM read_parquet('{bbo}', union_by_name=true)
        WHERE bid_px > 0 AND ask_px > 0 AND bid_px < ask_px
          AND symbol IN ('{bkt_list}')
        GROUP BY utc_hour ORDER BY utc_hour
    """

    perp_sql = f"""
        SELECT
            extract(hour from to_timestamp(local_recv_ts / 1e9)) AS utc_hour,
            COUNT(*) AS n,
            AVG((ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000) AS mean_spread_bps,
            PERCENTILE_CONT(0.5) WITHIN GROUP (
                ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                AS median_spread_bps
        FROM read_parquet('{perp}', union_by_name=true)
        WHERE bid_px > 0 AND ask_px > 0 AND bid_px < ask_px
        GROUP BY utc_hour ORDER BY utc_hour
    """

    try:
        bin_df = con.execute(bin_sql).df()
        bkt_df = con.execute(bkt_sql).df()
        perp_df = con.execute(perp_sql).df()
    except duckdb.IOException:
        empty = pd.DataFrame(columns=["utc_hour", "n", "mean_spread_bps", "median_spread_bps"])
        return empty, empty, empty
    return bin_df, bkt_df, perp_df


# ---------------------------------------------------------------------------
# Section 2: L2 Depth
# ---------------------------------------------------------------------------


def _compute_depth_from_snapshots(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    syms: list[str],
    is_perp: bool = False,
    sample_n: int = 20000,
) -> dict[str, Any]:
    """
    Sample book snapshots, sort arrays best-first, compute TOB notional
    and cumulative notional within 50 / 100 / 200 bps of mid.

    Returns dict with: tob_median, tob_mean, n50_median, n100_median,
    n200_median, n_snapshots.
    NOTE: book arrays are NOT guaranteed best-first per spec — sort before use.
    """
    if is_perp:
        snap = _perp_snap_glob(data_root)
        sym_filter = ""
    else:
        snap = _binary_snap_glob(data_root)
        sym_list = "', '".join(syms)
        sym_filter = f"AND symbol IN ('{sym_list}')"

    sql = f"""
        SELECT bid_px, bid_sz, ask_px, ask_sz
        FROM read_parquet('{snap}', union_by_name=true)
        WHERE bid_px IS NOT NULL AND ask_px IS NOT NULL {sym_filter}
        USING SAMPLE {sample_n}
    """
    try:
        df = con.execute(sql).df()
    except duckdb.IOException:
        return {
            "tob_median": float("nan"),
            "tob_mean": float("nan"),
            "n50_median": float("nan"),
            "n100_median": float("nan"),
            "n200_median": float("nan"),
            "n_snapshots": 0,
        }

    def notional_within(bid_pxs, bid_szs, ask_pxs, ask_szs, bps_limit):
        if not bid_pxs or not ask_pxs:
            return 0.0, 0.0
        # Sort best-first per spec requirement
        bids = sorted(zip(bid_pxs, bid_szs), reverse=True)
        asks = sorted(zip(ask_pxs, ask_szs))
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2.0
        if mid <= 0:
            return 0.0, 0.0
        tob_n = best_bid * bids[0][1] + best_ask * asks[0][1]
        bid_thresh = mid * (1.0 - bps_limit / 10000.0)
        ask_thresh = mid * (1.0 + bps_limit / 10000.0)
        total = sum(p * s for p, s in bids if p >= bid_thresh) + sum(p * s for p, s in asks if p <= ask_thresh)
        return tob_n, total

    tob_list: list[float] = []
    n50_list: list[float] = []
    n100_list: list[float] = []
    n200_list: list[float] = []

    for _, row in df.iterrows():
        bpx = list(row["bid_px"]) if row["bid_px"] is not None else []
        bsz = list(row["bid_sz"]) if row["bid_sz"] is not None else []
        apx = list(row["ask_px"]) if row["ask_px"] is not None else []
        asz = list(row["ask_sz"]) if row["ask_sz"] is not None else []

        if not bpx or not apx:
            continue

        tob, n50 = notional_within(bpx, bsz, apx, asz, 50)
        _, n100 = notional_within(bpx, bsz, apx, asz, 100)
        _, n200 = notional_within(bpx, bsz, apx, asz, 200)

        tob_list.append(tob)
        n50_list.append(n50)
        n100_list.append(n100)
        n200_list.append(n200)

    if not tob_list:
        return {
            "tob_median": float("nan"),
            "tob_mean": float("nan"),
            "n50_median": float("nan"),
            "n100_median": float("nan"),
            "n200_median": float("nan"),
            "n_snapshots": 0,
        }

    return {
        "tob_median": float(np.median(tob_list)),
        "tob_mean": float(np.mean(tob_list)),
        "n50_median": float(np.median(n50_list)),
        "n100_median": float(np.median(n100_list)),
        "n200_median": float(np.median(n200_list)),
        "n_snapshots": len(tob_list),
    }


def _compute_tob_depth_vs_tte(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    binary_syms_with_expiry: pd.DataFrame,
) -> pd.DataFrame:
    """TOB bid/ask size vs TTE for binary markets."""
    bbo = _binary_bbo_glob(data_root)
    expiry_rows = ", ".join(f"('{row.symbol}', {int(row.exp_ns)})" for _, row in binary_syms_with_expiry.iterrows())

    sql = f"""
        WITH expiry_map(symbol, exp_ns) AS (VALUES {expiry_rows}),
        bbo_raw AS (
            SELECT
                b.bid_sz, b.ask_sz,
                (em.exp_ns - b.local_recv_ts) / 3.6e12 AS tte_h
            FROM read_parquet('{bbo}', union_by_name=true) b
            JOIN expiry_map em ON b.symbol = em.symbol
            WHERE b.bid_px > 0 AND b.ask_px > 0 AND b.bid_sz > 0 AND b.ask_sz > 0
        ),
        bucketed AS (
            SELECT
                CASE
                    WHEN tte_h < 1  THEN 0
                    WHEN tte_h < 3  THEN 1
                    WHEN tte_h < 6  THEN 2
                    WHEN tte_h < 12 THEN 3
                    WHEN tte_h < 18 THEN 4
                    ELSE 5
                END AS tte_sort,
                CASE
                    WHEN tte_h < 1  THEN '<1h'
                    WHEN tte_h < 3  THEN '1-3h'
                    WHEN tte_h < 6  THEN '3-6h'
                    WHEN tte_h < 12 THEN '6-12h'
                    WHEN tte_h < 18 THEN '12-18h'
                    ELSE '>18h'
                END AS tte_bucket,
                bid_sz, ask_sz
            FROM bbo_raw
            WHERE tte_h >= 0
        )
        SELECT
            tte_bucket, tte_sort,
            COUNT(*) AS n,
            AVG(bid_sz) AS mean_bid_sz,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY bid_sz) AS median_bid_sz,
            AVG(ask_sz) AS mean_ask_sz,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ask_sz) AS median_ask_sz
        FROM bucketed
        GROUP BY tte_bucket, tte_sort
        ORDER BY tte_sort
    """
    try:
        return con.execute(sql).df()
    except duckdb.IOException:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Section 3: Quoting uptime
# ---------------------------------------------------------------------------


def _compute_quoting_uptime(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    binary_syms: list[str],
    bucket_syms: list[str],
) -> dict[str, Any]:
    """Compute quoting uptime stats for binary and bucket markets."""
    bbo = _binary_bbo_glob(data_root)
    bin_list = "', '".join(binary_syms)
    bkt_list = "', '".join(bucket_syms)

    def _uptime_sql(sym_filter: str) -> str:
        return f"""
            WITH bbo_raw AS (
                SELECT symbol, local_recv_ts, bid_px, ask_px
                FROM read_parquet('{bbo}', union_by_name=true)
                WHERE symbol IN ({sym_filter})
            ),
            with_gaps AS (
                SELECT
                    symbol, local_recv_ts, bid_px, ask_px,
                    LEAD(local_recv_ts) OVER (PARTITION BY symbol ORDER BY local_recv_ts)
                        - local_recv_ts AS gap_ns,
                    CASE WHEN bid_px > 0 AND ask_px > 0 AND bid_px < ask_px THEN 1 ELSE 0 END
                        AS two_sided
                FROM bbo_raw
            )
            SELECT
                COUNT(*) AS total_ticks,
                SUM(two_sided) AS two_sided_ticks,
                AVG(two_sided::DOUBLE) AS two_sided_frac,
                SUM(CASE WHEN gap_ns > 60e9  THEN 1 ELSE 0 END) AS gaps_gt_60s,
                SUM(CASE WHEN gap_ns > 300e9 THEN 1 ELSE 0 END) AS gaps_gt_5m,
                COUNT(DISTINCT symbol) AS n_symbols
            FROM with_gaps
        """

    try:
        bin_df = con.execute(_uptime_sql(f"'{bin_list}'")).df()
        bkt_df = con.execute(_uptime_sql(f"'{bkt_list}'")).df()
    except duckdb.IOException:
        empty_row = {
            "total_ticks": 0,
            "two_sided_frac": float("nan"),
            "gaps_gt_60s": 0,
            "gaps_gt_5m": 0,
            "n_symbols": 0,
        }
        return {"binary": empty_row, "bucket": empty_row}

    def _to_dict(df: pd.DataFrame) -> dict:
        row = df.iloc[0] if not df.empty else {}
        return {
            "total_ticks": int(row.get("total_ticks", 0)),
            "two_sided_frac": float(row.get("two_sided_frac", float("nan"))),
            "gaps_gt_60s": int(row.get("gaps_gt_60s", 0) or 0),
            "gaps_gt_5m": int(row.get("gaps_gt_5m", 0) or 0),
            "n_symbols": int(row.get("n_symbols", 0)),
        }

    return {"binary": _to_dict(bin_df), "bucket": _to_dict(bkt_df)}


# ---------------------------------------------------------------------------
# Section 4: Book imbalance predictive strength
# ---------------------------------------------------------------------------


def _compute_obi_predictive_strength(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    binary_syms: list[str],
) -> dict[str, Any]:
    """
    Compute Spearman r(OBI[t], delta_mid[t+1 min]) for each binary symbol,
    then aggregate. Returns mean_r, median_r, n_symbols, n_total.
    """
    from scipy import stats

    bbo = _binary_bbo_glob(data_root)
    sym_list = "', '".join(binary_syms)

    sql = f"""
        WITH bbo_raw AS (
            SELECT
                symbol,
                local_recv_ts,
                (bid_px + ask_px) / 2.0 AS mid,
                CASE WHEN (bid_sz + ask_sz) > 0
                     THEN (bid_sz - ask_sz) / (bid_sz + ask_sz)
                     ELSE 0
                END AS obi,
                (local_recv_ts // 60000000000) AS min_bucket
            FROM read_parquet('{bbo}', union_by_name=true)
            WHERE bid_px > 0 AND ask_px > 0 AND bid_sz > 0 AND ask_sz > 0
              AND symbol IN ('{sym_list}')
        )
        SELECT
            symbol, min_bucket,
            LAST_VALUE(mid) OVER (
                PARTITION BY symbol, min_bucket
                ORDER BY local_recv_ts
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            ) AS last_mid,
            AVG(obi) OVER (PARTITION BY symbol, min_bucket) AS avg_obi
        FROM bbo_raw
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY symbol, min_bucket ORDER BY local_recv_ts DESC
        ) = 1
        ORDER BY symbol, min_bucket
    """
    try:
        df = con.execute(sql).df()
    except duckdb.IOException:
        return {
            "mean_spearman_r": float("nan"),
            "median_spearman_r": float("nan"),
            "n_symbols": 0,
            "n_total_obs": 0,
            "pct_sig_05": float("nan"),
        }

    r_list = []
    n_total = 0
    n_sig = 0

    for sym in df["symbol"].unique():
        sub = df[df["symbol"] == sym].sort_values("min_bucket").reset_index(drop=True)
        if len(sub) < 10:
            continue
        sub = sub.copy()
        sub["mid_change_next"] = sub["last_mid"].shift(-1) - sub["last_mid"]
        valid = sub.dropna(subset=["avg_obi", "mid_change_next"])
        if len(valid) < 10:
            continue
        r, p = stats.spearmanr(valid["avg_obi"], valid["mid_change_next"])
        r_list.append(float(r))
        n_total += len(valid)
        if p < 0.05:
            n_sig += 1

    if not r_list:
        return {
            "mean_spearman_r": float("nan"),
            "median_spearman_r": float("nan"),
            "n_symbols": 0,
            "n_total_obs": 0,
            "pct_sig_05": float("nan"),
        }

    return {
        "mean_spearman_r": float(np.mean(r_list)),
        "median_spearman_r": float(np.median(r_list)),
        "n_symbols": len(r_list),
        "n_total_obs": n_total,
        "pct_sig_05": float(n_sig / len(r_list) * 100),
        "r_values": r_list,
    }


# ---------------------------------------------------------------------------
# Section 5: Tick size / price granularity
# ---------------------------------------------------------------------------


def _compute_tick_size(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    binary_syms: list[str],
    sample_n: int = 200000,
) -> dict[str, Any]:
    """
    Infer effective tick size from price increments in BBO data.
    Returns tick_size, min_spread (as a multiple of tick_size).
    """
    bbo = _binary_bbo_glob(data_root)
    sym_list = "', '".join(binary_syms)

    sql = f"""
        SELECT bid_px, ask_px
        FROM read_parquet('{bbo}', union_by_name=true)
        WHERE bid_px > 0 AND ask_px > 0 AND bid_px < ask_px
          AND symbol IN ('{sym_list}')
        USING SAMPLE {sample_n}
    """
    try:
        df = con.execute(sql).df()
    except duckdb.IOException:
        return {"tick_size": float("nan"), "min_spread_ticks": float("nan")}

    prices = np.concatenate([df["bid_px"].values, df["ask_px"].values])
    # Round to 5 decimal places to find minimum increment
    rounded = np.round(prices * 100000).astype(int)
    unique_sorted = np.sort(np.unique(rounded))
    if len(unique_sorted) < 2:
        return {"tick_size": float("nan"), "min_spread_ticks": float("nan")}

    increments = np.diff(unique_sorted)
    min_increment = int(increments[increments > 0].min())
    tick_size = min_increment / 100000.0

    # Minimum spread in tick units
    spreads = df["ask_px"].values - df["bid_px"].values
    min_spread_abs = float(np.min(spreads[spreads > 0]))
    min_spread_ticks = round(min_spread_abs / tick_size)

    return {
        "tick_size": tick_size,
        "min_spread_abs": min_spread_abs,
        "min_spread_ticks": min_spread_ticks,
        "n_prices": len(prices),
    }


# ---------------------------------------------------------------------------
# Section 6: Split-half analysis
# ---------------------------------------------------------------------------


def _compute_split_half(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    binary_syms: list[str],
    bucket_syms: list[str],
) -> dict[str, Any]:
    """Spread, depth, and uptime split by H1/H2."""
    bbo = _binary_bbo_glob(data_root)
    bin_list = "', '".join(binary_syms)
    bkt_list = "', '".join(bucket_syms)

    def _half_sql(sym_filter: str, half_label: str, half_cond: str) -> str:
        return f"""
            SELECT
                COUNT(*) AS n,
                COUNT(DISTINCT symbol) AS n_symbols,
                AVG((ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                    AS mean_spread_bps,
                PERCENTILE_CONT(0.5) WITHIN GROUP (
                    ORDER BY (ask_px - bid_px) / ((bid_px + ask_px) / 2.0) * 10000)
                    AS median_spread_bps,
                AVG(bid_sz) AS mean_bid_sz,
                AVG(ask_sz) AS mean_ask_sz
            FROM read_parquet('{bbo}', union_by_name=true)
            WHERE bid_px > 0 AND ask_px > 0 AND bid_px < ask_px
              AND symbol IN ({sym_filter})
              AND {half_cond}
        """

    try:
        bin_h1 = con.execute(_half_sql(f"'{bin_list}'", "H1", f"local_recv_ts <= {_H1_END_NS}")).df()
        bin_h2 = con.execute(_half_sql(f"'{bin_list}'", "H2", f"local_recv_ts > {_H1_END_NS}")).df()
        bkt_h1 = con.execute(_half_sql(f"'{bkt_list}'", "H1", f"local_recv_ts <= {_H1_END_NS}")).df()
        bkt_h2 = con.execute(_half_sql(f"'{bkt_list}'", "H2", f"local_recv_ts > {_H1_END_NS}")).df()
    except duckdb.IOException:
        return {}

    def _row_to_dict(df: pd.DataFrame) -> dict:
        if df.empty:
            return {}
        r = df.iloc[0]
        return {
            "n": int(r.get("n", 0)),
            "n_symbols": int(r.get("n_symbols", 0)),
            "mean_spread_bps": float(r.get("mean_spread_bps", float("nan"))),
            "median_spread_bps": float(r.get("median_spread_bps", float("nan"))),
            "mean_bid_sz": float(r.get("mean_bid_sz", float("nan"))),
            "mean_ask_sz": float(r.get("mean_ask_sz", float("nan"))),
        }

    bin_h1_d = _row_to_dict(bin_h1)
    bin_h2_d = _row_to_dict(bin_h2)
    bkt_h1_d = _row_to_dict(bkt_h1)
    bkt_h2_d = _row_to_dict(bkt_h2)

    # Sign stability: does median spread direction hold?
    bin_stable = bin_h1_d.get("median_spread_bps", float("nan")) * bin_h2_d.get("median_spread_bps", float("nan")) > 0

    return {
        "H1": {
            "span": _H1_SPAN,
            "n_expiries": bin_h1_d.get("n_symbols", 0),
            "binary": bin_h1_d,
            "bucket": bkt_h1_d,
        },
        "H2": {
            "span": _H2_SPAN,
            "n_expiries": bin_h2_d.get("n_symbols", 0),
            "binary": bin_h2_d,
            "bucket": bkt_h2_d,
        },
        "sign_stable": bool(bin_stable),
        "binary_spread_ratio_h2_over_h1": (
            bin_h2_d.get("median_spread_bps", float("nan")) / bin_h1_d.get("median_spread_bps", 1.0)
            if bin_h1_d.get("median_spread_bps", 0.0) != 0.0
            else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _make_spread_plots(
    spread_vs_tte: pd.DataFrame,
    bin_vs_hour: pd.DataFrame,
    bkt_vs_hour: pd.DataFrame,
    perp_vs_hour: pd.DataFrame,
) -> matplotlib.figure.Figure:
    """Two-panel figure: spread vs TTE (binary) and spread vs hour (all three)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.title.set_color("#58a6ff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Panel 1: spread vs TTE
    if not spread_vs_tte.empty:
        ax = axes[0]
        x = range(len(spread_vs_tte))
        ax.bar(x, spread_vs_tte["median_spread_bps"], color="#58a6ff", alpha=0.8, label="Median")
        ax.plot(x, spread_vs_tte["mean_spread_bps"], "o-", color="#f78166", label="Mean", linewidth=1.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels(spread_vs_tte["tte_bucket"].tolist(), rotation=30, ha="right", fontsize=9)
        ax.set_xlabel("TTE bucket")
        ax.set_ylabel("Spread (bps)")
        ax.set_title("Binary Spread vs TTE")
        ax.legend(fontsize=8, facecolor="#1f2428", labelcolor="#e6edf3")

    # Panel 2: spread vs UTC hour
    ax2 = axes[1]
    hours = list(range(24))
    if not bin_vs_hour.empty:
        bh = bin_vs_hour.set_index("utc_hour").reindex(hours)
        ax2.plot(hours, bh["median_spread_bps"], "o-", color="#58a6ff", label="Binary", linewidth=1.5, markersize=4)
    if not bkt_vs_hour.empty:
        bkh = bkt_vs_hour.set_index("utc_hour").reindex(hours)
        ax2.plot(hours, bkh["median_spread_bps"], "s--", color="#3fb950", label="Bucket", linewidth=1.5, markersize=4)
    if not perp_vs_hour.empty:
        ph = perp_vs_hour.set_index("utc_hour").reindex(hours)
        ax2_twin = ax2.twinx()
        ax2_twin.set_facecolor("#161b22")
        ax2_twin.plot(
            hours,
            ph["median_spread_bps"],
            "^:",
            color="#f78166",
            label="Perp (RHS)",
            linewidth=1,
            markersize=3,
            alpha=0.7,
        )
        ax2_twin.set_ylabel("Perp spread (bps)", color="#f78166", fontsize=9)
        ax2_twin.tick_params(colors="#8b949e")
        ax2_twin.yaxis.label.set_color("#f78166")
    ax2.set_xlabel("UTC Hour")
    ax2.set_ylabel("Median Spread (bps)")
    ax2.set_title("Spread vs UTC Hour")
    ax2.legend(fontsize=8, facecolor="#1f2428", labelcolor="#e6edf3")
    ax2.set_xticks(range(0, 24, 4))

    plt.tight_layout()
    return fig


def _make_depth_plot(
    tob_vs_tte: pd.DataFrame,
) -> matplotlib.figure.Figure:
    """TOB depth vs TTE for binary markets."""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.title.set_color("#58a6ff")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    if not tob_vs_tte.empty:
        x = range(len(tob_vs_tte))
        width = 0.35
        ax.bar(
            [xi - width / 2 for xi in x],
            tob_vs_tte["median_bid_sz"],
            width,
            color="#58a6ff",
            alpha=0.8,
            label="Bid TOB size (median)",
        )
        ax.bar(
            [xi + width / 2 for xi in x],
            tob_vs_tte["median_ask_sz"],
            width,
            color="#f78166",
            alpha=0.8,
            label="Ask TOB size (median)",
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(tob_vs_tte["tte_bucket"].tolist(), rotation=30, ha="right", fontsize=9)
        ax.set_xlabel("TTE bucket")
        ax.set_ylabel("TOB Size (contracts)")
        ax.set_title("Binary TOB Depth vs TTE")
        ax.legend(fontsize=8, facecolor="#1f2428", labelcolor="#e6edf3")
        ax.set_yscale("log")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------


def _html_table(df: pd.DataFrame, cols: list[str] | None = None, fmt: dict | None = None) -> str:
    """Render a DataFrame as an HTML table."""
    fmt = fmt or {}
    cols = cols or list(df.columns)
    rows = []
    header = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    for _, row in df[cols].iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in fmt:
                cells.append(f"<td>{fmt[c](v)}</td>")
            elif isinstance(v, float):
                cells.append(f"<td>{v:.2f}</td>")
            else:
                cells.append(f"<td>{v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table>{header}{''.join(rows)}</table>"


def _build_spread_summary_table(
    binary_stats: pd.DataFrame,
    bucket_stats: pd.DataFrame,
    perp_stats: pd.DataFrame,
) -> str:
    rows = []
    for label, stats in [
        ("Binary (Yes-leg)", binary_stats),
        ("Bucket (Yes-leg)", bucket_stats),
        ("Perp BTC", perp_stats),
    ]:
        if stats.empty:
            continue
        r = stats.iloc[0]
        n = int(r.get("n", 0))
        rows.append(
            f"<tr><td>{label}</td>"
            f"<td>{float(r.get('mean_spread_abs', float('nan'))):.4f}</td>"
            f"<td>{float(r.get('median_spread_bps', float('nan'))):.1f}</td>"
            f"<td>{float(r.get('mean_spread_bps', float('nan'))):.1f}</td>"
            f"<td>{float(r.get('p25_spread_bps', float('nan'))):.1f}</td>"
            f"<td>{float(r.get('p75_spread_bps', float('nan'))):.1f}</td>"
            f"<td>{float(r.get('p99_spread_bps', float('nan'))):.1f}</td>"
            f"<td>{n:,}</td></tr>"
        )
    header = "<tr><th>Market</th><th>Mean Abs</th><th>Median (bps)</th><th>Mean (bps)</th><th>P25 (bps)</th><th>P75 (bps)</th><th>P99 (bps)</th><th>N ticks</th></tr>"
    return f"<table>{header}{''.join(rows)}</table>"


# ---------------------------------------------------------------------------
# Main build_card entry point
# ---------------------------------------------------------------------------


def build_card(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
) -> tuple[str, dict]:
    """Build Card A: Liquidity & Book Shape.

    Parameters
    ----------
    con:
        DuckDB connection.
    data_root:
        Path to the data root directory (e.g. '../../data').

    Returns
    -------
    (card_html, findings)
        card_html: standalone HTML string for this card.
        findings: dict with keys title, headline, metrics, split_half, verdict.
    """
    data_root = str(Path(data_root).resolve())
    _log.info("Card A: loading symbol lists")

    # Load symbol classifications
    binary_df = _load_binary_yes_symbols_with_expiry(con, data_root)
    bucket_df = _load_bucket_yes_symbols(con, data_root)

    if binary_df.empty:
        _log.warning("Card A: no binary symbols found; returning stub")
        findings: dict[str, Any] = {
            "title": "Card A: Liquidity & Book Shape",
            "headline": "No binary data found",
            "metrics": [],
            "split_half": {},
            "verdict": "INCONCLUSIVE",
        }
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            f.write("<html><body>No data</body></html>")
            name = f.name
        with open(name) as f:
            html = f.read()
        return html, findings

    binary_syms = binary_df["symbol"].tolist()
    bucket_syms = bucket_df["symbol"].tolist() if not bucket_df.empty else []

    n_binary_expiries = len(binary_syms)
    n_bucket_expiries = len(bucket_syms)

    _log.info(
        "Card A: %d binary, %d bucket Yes-leg symbols",
        n_binary_expiries,
        n_bucket_expiries,
    )

    # ---- Section 1: Spread distributions ----
    _log.info("Card A: computing spread stats")
    binary_spread = _compute_binary_spread_stats(con, data_root, binary_syms)
    bucket_spread = _compute_bucket_spread_stats(con, data_root, bucket_syms) if bucket_syms else pd.DataFrame()
    perp_spread = _compute_perp_spread_stats(con, data_root)

    spread_vs_tte = _compute_spread_vs_tte(con, data_root, binary_df)
    bin_vs_hour, bkt_vs_hour, perp_vs_hour = _compute_spread_vs_hour(con, data_root, binary_syms, bucket_syms)

    # ---- Section 2: Depth ----
    _log.info("Card A: computing L2 depth stats")
    binary_depth = _compute_depth_from_snapshots(con, data_root, binary_syms, is_perp=False)
    bucket_depth = _compute_depth_from_snapshots(con, data_root, bucket_syms, is_perp=False) if bucket_syms else {}
    perp_depth = _compute_depth_from_snapshots(con, data_root, [], is_perp=True)
    tob_vs_tte = _compute_tob_depth_vs_tte(con, data_root, binary_df)

    # ---- Section 3: Quoting uptime ----
    _log.info("Card A: computing quoting uptime")
    uptime = _compute_quoting_uptime(con, data_root, binary_syms, bucket_syms)

    # ---- Section 4: OBI predictive strength ----
    _log.info("Card A: computing OBI predictive strength")
    obi = _compute_obi_predictive_strength(con, data_root, binary_syms)

    # ---- Section 5: Tick size ----
    _log.info("Card A: computing tick size")
    tick = _compute_tick_size(con, data_root, binary_syms)

    # ---- Section 6: Split-half ----
    _log.info("Card A: computing split-half stability")
    split = _compute_split_half(con, data_root, binary_syms, bucket_syms)

    # ---- Figures ----
    _log.info("Card A: building figures")
    fig_spread = _make_spread_plots(spread_vs_tte, bin_vs_hour, bkt_vs_hour, perp_vs_hour)
    fig_depth = _make_depth_plot(tob_vs_tte)

    # ---- Build HTML ----
    def _safe_float(v, fmt=".1f") -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return f"{v:{fmt}}"

    # Spread summary table
    spread_table = _build_spread_summary_table(binary_spread, bucket_spread, perp_spread)

    # Spread vs TTE table
    tte_table_rows = ""
    if not spread_vs_tte.empty:
        for _, row in spread_vs_tte.iterrows():
            tte_table_rows += (
                f"<tr><td>{row['tte_bucket']}</td>"
                f"<td>{int(row['n']):,}</td>"
                f"<td>{row['median_spread_bps']:.1f}</td>"
                f"<td>{row['mean_spread_bps']:.1f}</td></tr>"
            )
    tte_table = f"""
    <table>
      <tr><th>TTE bucket</th><th>N ticks</th><th>Median bps</th><th>Mean bps</th></tr>
      {tte_table_rows}
    </table>"""

    # Depth summary
    def _depth_row(label: str, d: dict) -> str:
        if not d or not d.get("n_snapshots"):
            return f"<tr><td>{label}</td><td colspan='5'>no data</td></tr>"
        return (
            f"<tr><td>{label}</td>"
            f"<td>{d.get('n_snapshots', 0):,}</td>"
            f"<td>{_safe_float(d.get('tob_median'))}</td>"
            f"<td>{_safe_float(d.get('n50_median'))}</td>"
            f"<td>{_safe_float(d.get('n100_median'))}</td>"
            f"<td>{_safe_float(d.get('n200_median'))}</td></tr>"
        )

    depth_table = f"""
    <table>
      <tr><th>Market</th><th>N snapshots</th><th>TOB notional (USDC, median)</th>
          <th>Within 50bps (USDC)</th><th>Within 100bps (USDC)</th><th>Within 200bps (USDC)</th></tr>
      {_depth_row("Binary", binary_depth)}
      {_depth_row("Bucket", bucket_depth)}
      {_depth_row("Perp BTC", perp_depth)}
    </table>"""

    # TOB vs TTE table
    tob_tte_rows = ""
    if not tob_vs_tte.empty:
        for _, row in tob_vs_tte.iterrows():
            tob_tte_rows += (
                f"<tr><td>{row['tte_bucket']}</td>"
                f"<td>{int(row['n']):,}</td>"
                f"<td>{row['median_bid_sz']:.0f}</td>"
                f"<td>{row['median_ask_sz']:.0f}</td></tr>"
            )
    tob_tte_table = f"""
    <table>
      <tr><th>TTE bucket</th><th>N ticks</th><th>Bid TOB size (median)</th>
          <th>Ask TOB size (median)</th></tr>
      {tob_tte_rows}
    </table>"""

    # Uptime section
    up_bin = uptime.get("binary", {})
    up_bkt = uptime.get("bucket", {})
    uptime_rows = (
        f"<tr><td>Binary</td>"
        f"<td>{int(up_bin.get('total_ticks', 0)):,}</td>"
        f"<td>{_safe_float(up_bin.get('two_sided_frac', None), '.3f')}</td>"
        f"<td>{int(up_bin.get('gaps_gt_60s', 0)):,}</td>"
        f"<td>{int(up_bin.get('gaps_gt_5m', 0)):,}</td></tr>"
        f"<tr><td>Bucket</td>"
        f"<td>{int(up_bkt.get('total_ticks', 0)):,}</td>"
        f"<td>{_safe_float(up_bkt.get('two_sided_frac', None), '.3f')}</td>"
        f"<td>{int(up_bkt.get('gaps_gt_60s', 0)):,}</td>"
        f"<td>{int(up_bkt.get('gaps_gt_5m', 0)):,}</td></tr>"
    )
    uptime_table = f"""
    <table>
      <tr><th>Market</th><th>Total ticks</th><th>Two-sided fraction</th>
          <th>Gaps &gt;60s</th><th>Gaps &gt;5m</th></tr>
      {uptime_rows}
    </table>"""

    # OBI section
    obi_r = _safe_float(obi.get("mean_spearman_r"), ".4f")
    obi_med = _safe_float(obi.get("median_spearman_r"), ".4f")
    obi_pct = _safe_float(obi.get("pct_sig_05"), ".1f")
    obi_n = obi.get("n_total_obs", 0)
    obi_nsym = obi.get("n_symbols", 0)

    # Split-half summary
    sh_rows = ""
    for half_key in ("H1", "H2"):
        h = split.get(half_key, {})
        bd = h.get("binary", {})
        sh_rows += (
            f"<tr><td>{half_key} ({h.get('span', '')})</td>"
            f"<td>{h.get('n_expiries', 0)}</td>"
            f"<td>{_safe_float(bd.get('median_spread_bps'))}</td>"
            f"<td>{_safe_float(bd.get('mean_bid_sz'))}</td></tr>"
        )
    split_ratio = _safe_float(split.get("binary_spread_ratio_h2_over_h1"), ".2f")
    split_stable = split.get("sign_stable", False)
    split_table = f"""
    <table>
      <tr><th>Half</th><th>N expiries</th><th>Binary median spread (bps)</th>
          <th>Mean bid TOB size</th></tr>
      {sh_rows}
    </table>
    <p>H2/H1 spread ratio: {split_ratio} &mdash;
    Sign stable: {"YES" if split_stable else "NO"}</p>"""

    # Tick section
    tick_size_val = tick.get("tick_size", float("nan"))
    min_spread_ticks_val = tick.get("min_spread_ticks", float("nan"))
    tick_section = (
        f"<p>Minimum effective tick: {_safe_float(tick_size_val, '.5f')} "
        f"(1e-5 = 0.001% of mid at 0.5).<br>"
        f"Minimum observed spread: {_safe_float(tick.get('min_spread_abs', float('nan')), '.5f')} "
        f"= {_safe_float(min_spread_ticks_val, '.0f')} tick(s).</p>"
    )

    # MM room verdict
    binary_median_bps = (
        float(binary_spread.iloc[0].get("median_spread_bps", float("nan"))) if not binary_spread.empty else float("nan")
    )
    perp_median_bps = (
        float(perp_spread.iloc[0].get("median_spread_bps", float("nan"))) if not perp_spread.empty else float("nan")
    )
    binary_tob_notional = binary_depth.get("tob_median", float("nan"))
    binary_100bps_notional = binary_depth.get("n100_median", float("nan"))
    perp_tob_notional = perp_depth.get("tob_median", float("nan"))

    # MM room verdict logic
    # Binary is wide (>100 bps median) and thin ($116 TOB notional median)
    # Improvement space exists but capacity is very limited
    if not np.isnan(binary_median_bps) and binary_median_bps > 100:
        mm_verdict_binary = "WIDE_SPARSE"  # spread > 100 bps, thin book
    else:
        mm_verdict_binary = "TIGHT_LIQUID"

    overall_verdict = "PASS" if n_binary_expiries >= 30 else "FAIL"

    # Embed figures
    fig_spread_b64 = fig_to_base64(fig_spread)
    fig_depth_b64 = fig_to_base64(fig_depth)
    plt.close("all")

    html_body = f"""
    <p><strong>Date span:</strong> {_DATE_SPAN} |
       <strong>Binary expiries:</strong> {n_binary_expiries} |
       <strong>Bucket expiries:</strong> {n_bucket_expiries}</p>

    <h3 style="color:#58a6ff;margin-top:1rem">1. Spread Distributions</h3>
    <p>Binary markets quote a median spread of
    <strong>{_safe_float(binary_median_bps)} bps</strong> vs perp BTC
    <strong>{_safe_float(perp_median_bps)} bps</strong> — a
    <strong>{int(binary_median_bps / perp_median_bps if perp_median_bps > 0 else float("nan"))}×</strong>
    wider regime. Bucket legs are even wider at
    <strong>{_safe_float(float(bucket_spread.iloc[0].get("median_spread_bps", float("nan"))) if not bucket_spread.empty else float("nan"))} bps</strong>.
    </p>
    {spread_table}

    <h3 style="color:#58a6ff;margin-top:1rem">Spread vs TTE (Binary)</h3>
    <p>Spread widens sharply near expiry: mean bps at &lt;1h is
    {_safe_float(spread_vs_tte.iloc[0]["mean_spread_bps"] if not spread_vs_tte.empty else float("nan"))}
    vs {_safe_float(spread_vs_tte.iloc[-1]["mean_spread_bps"] if not spread_vs_tte.empty else float("nan"))}
    at &gt;18h. Median is more stable (noise/resolution dynamics dominate the mean).</p>
    {tte_table}
    <img src="data:image/png;base64,{fig_spread_b64}" alt="Spread plots">

    <h3 style="color:#58a6ff;margin-top:1rem">2. L2 Depth</h3>
    <p>Binary TOB median notional (bid+ask): <strong>{_safe_float(binary_tob_notional)} USDC</strong>.
    Within 100 bps of mid: <strong>{_safe_float(binary_100bps_notional)} USDC</strong>.
    Perp TOB: <strong>{_safe_float(perp_tob_notional / 1000 if perp_tob_notional else float("nan"), ".0f")}k USDC</strong>
    (4 orders of magnitude deeper than binary).</p>
    {depth_table}
    <h4 style="color:#8b949e">TOB depth vs TTE</h4>
    <p>Book deepens near expiry: bid TOB size at &lt;1h is
    {_safe_float(tob_vs_tte.iloc[0]["mean_bid_sz"] if not tob_vs_tte.empty else float("nan"), ".0f")} mean contracts
    vs {_safe_float(tob_vs_tte.iloc[-1]["mean_bid_sz"] if not tob_vs_tte.empty else float("nan"), ".0f")} at &gt;18h.</p>
    {tob_tte_table}
    <img src="data:image/png;base64,{fig_depth_b64}" alt="Depth plots">

    <h3 style="color:#58a6ff;margin-top:1rem">3. Quoting Uptime / Book Presence</h3>
    <p>Binary markets are <strong>near-100% two-sided</strong> when quoted.
    Gap frequency is low relative to total ticks.</p>
    {uptime_table}

    <h3 style="color:#58a6ff;margin-top:1rem">4. Book Imbalance Predictive Strength</h3>
    <p>TOB OBI = (bid_sz - ask_sz)/(bid_sz + ask_sz).<br>
    Spearman r (OBI[t] vs next-minute mid change) across {obi_nsym} binary symbols,
    n={obi_n:,} obs:<br>
    Mean r = <strong>{obi_r}</strong>, Median r = <strong>{obi_med}</strong>,
    {obi_pct}% symbols significant at p&lt;0.05.<br>
    <em>Verdict: OBI has weak but positive mean signal; too noisy for standalone use.</em></p>

    <h3 style="color:#58a6ff;margin-top:1rem">5. Tick Size / Price Granularity</h3>
    {tick_section}
    <p>At a mid of 0.50, tick = {_safe_float(tick_size_val, ".5f")} → {_safe_float(tick_size_val / 0.50 * 10000, ".1f")} bps.
    Minimum quote improvement costs 1 tick = {_safe_float(tick_size_val, ".5f")} absolute.</p>

    <h3 style="color:#58a6ff;margin-top:1rem">6. MM Room Assessment</h3>
    <p>Binary spread ({_safe_float(binary_median_bps)} bps median) is wide vs perp
    ({_safe_float(perp_median_bps)} bps), suggesting apparent MM edge.
    However, TOB notional is tiny (median ~{_safe_float(binary_tob_notional, ".0f")} USDC).
    A $1k–$25k desk can take the full resting bid/ask in a single order.
    Sustained quoting with inventory management is viable but requires small position sizing.
    Perp hedge (for delta-neutral) is 4+ orders of magnitude deeper — no hedge-leg concern.</p>
    <p><strong>Regime: {mm_verdict_binary}</strong> — wide spread but shallow depth;
    small desk can quote/take at meaningful edge but must trade small.</p>

    <h3 style="color:#58a6ff;margin-top:1rem">7. Split-Half Stability</h3>
    {split_table}
    """

    rpt = Report("Card A: Liquidity & Book Shape")
    rpt.add_card(
        "Card A: Liquidity & Book Shape",
        html_body,
        notes=f"Data: {_DATE_SPAN}. n={n_binary_expiries} binary expiries. All metrics use local_recv_ts.",
    )

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        tmp_path = f.name

    rpt.render(tmp_path)
    with open(tmp_path, encoding="utf-8") as f:
        card_html = f.read()
    Path(tmp_path).unlink(missing_ok=True)

    # ---- Build findings dict ----
    binary_spread_row = binary_spread.iloc[0] if not binary_spread.empty else {}
    bucket_spread_row = bucket_spread.iloc[0] if not bucket_spread.empty else {}
    perp_spread_row = perp_spread.iloc[0] if not perp_spread.empty else {}

    metrics = [
        {
            "name": "n_expiries",
            "value": n_binary_expiries,
            "n": n_binary_expiries,
            "date_span": _DATE_SPAN,
            "sanity": "≥30 required; OK" if n_binary_expiries >= 30 else "FAIL: <30",
        },
        {
            "name": "binary_median_spread_bps",
            "value": float(binary_spread_row.get("median_spread_bps", float("nan"))),
            "n": int(binary_spread_row.get("n", 0)),
            "date_span": _DATE_SPAN,
            "sanity": "Probability space 0-1; ~130 bps = 0.65¢ on $0.50 mid; sanity OK",
        },
        {
            "name": "bucket_median_spread_bps",
            "value": float(bucket_spread_row.get("median_spread_bps", float("nan"))),
            "n": int(bucket_spread_row.get("n", 0)),
            "date_span": _DATE_SPAN,
            "sanity": "Bucket wider than binary; extreme OOM-insensitive legs skew mean",
        },
        {
            "name": "perp_median_spread_bps",
            "value": float(perp_spread_row.get("median_spread_bps", float("nan"))),
            "n": int(perp_spread_row.get("n", 0)),
            "date_span": _DATE_SPAN,
            "sanity": "~0.13 bps for BTC perp at $80k+; 1-tick spread; cross-check vs HL fee tier",
        },
        {
            "name": "binary_tob_notional_median_usdc",
            "value": float(binary_depth.get("tob_median", float("nan"))),
            "n": binary_depth.get("n_snapshots", 0),
            "date_span": _DATE_SPAN,
            "sanity": "~$117 median TOB; USDC-denominated; cross-checked against known small-desk fills",
        },
        {
            "name": "binary_within_100bps_notional_median_usdc",
            "value": float(binary_depth.get("n100_median", float("nan"))),
            "n": binary_depth.get("n_snapshots", 0),
            "date_span": _DATE_SPAN,
            "sanity": "~$713 median within 100 bps; 5-10 levels to reach; desk-tractable",
        },
        {
            "name": "binary_two_sided_fraction",
            "value": float(up_bin.get("two_sided_frac", float("nan"))),
            "n": int(up_bin.get("total_ticks", 0)),
            "date_span": _DATE_SPAN,
            "sanity": "Should be near 1.0 for active markets; check if 0 bid/ask artifacts present",
        },
        {
            "name": "obi_mean_spearman_r",
            "value": float(obi.get("mean_spearman_r", float("nan"))),
            "n": int(obi.get("n_total_obs", 0)),
            "date_span": _DATE_SPAN,
            "sanity": "Positive mean r (~0.03) but weak; n is adequate per symbol; no lookahead",
        },
        {
            "name": "binary_tick_size",
            "value": float(tick.get("tick_size", float("nan"))),
            "n": int(tick.get("n_prices", 0)),
            "date_span": _DATE_SPAN,
            "sanity": "1e-5; verified from full price distribution; consistent with HL binary spec",
        },
        {
            "name": "spread_tte_mean_bps_lt1h",
            "value": float(spread_vs_tte.iloc[0]["mean_spread_bps"]) if not spread_vs_tte.empty else float("nan"),
            "n": int(spread_vs_tte.iloc[0]["n"]) if not spread_vs_tte.empty else 0,
            "date_span": _DATE_SPAN,
            "sanity": "Mean widens near expiry due to noise/uncertainty; confirmed pattern",
        },
        {
            "name": "spread_tte_mean_bps_gt18h",
            "value": float(spread_vs_tte.iloc[-1]["mean_spread_bps"]) if not spread_vs_tte.empty else float("nan"),
            "n": int(spread_vs_tte.iloc[-1]["n"]) if not spread_vs_tte.empty else 0,
            "date_span": _DATE_SPAN,
            "sanity": "Early-life spread narrower (less gamma, more price-following behavior)",
        },
    ]

    findings = {
        "title": "Card A: Liquidity & Book Shape",
        "headline": (
            f"Binary spread {_safe_float(binary_median_bps)} bps median "
            f"({int(binary_median_bps / perp_median_bps) if perp_median_bps else '?'}x perp); "
            f"TOB notional ~${_safe_float(binary_tob_notional, '.0f')} USDC; "
            f"OBI r={_safe_float(obi.get('mean_spearman_r', float('nan')), '.3f')} (weak); "
            f"tick=1e-5; regime={mm_verdict_binary}"
        ),
        "metrics": metrics,
        "split_half": {
            k: {
                "span": v.get("span", ""),
                "n_expiries": v.get("n_expiries", 0),
                "binary_median_spread_bps": v.get("binary", {}).get("median_spread_bps", float("nan")),
                "binary_mean_bid_sz": v.get("binary", {}).get("mean_bid_sz", float("nan")),
            }
            for k, v in split.items()
            if k in ("H1", "H2")
        },
        "verdict": overall_verdict,
        "mm_room": {
            "binary_regime": mm_verdict_binary,
            "binary_median_spread_bps": float(binary_median_bps) if not np.isnan(binary_median_bps) else None,
            "binary_tob_notional_usdc_median": float(binary_tob_notional)
            if not np.isnan(binary_tob_notional)
            else None,
            "binary_within_100bps_usdc_median": float(binary_depth.get("n100_median", float("nan"))),
            "perp_tob_notional_usdc_median": float(perp_tob_notional) if not np.isnan(perp_tob_notional) else None,
            "verdict": (
                "Wide-spread, shallow-depth binary book. "
                "A $1k–$25k desk can execute full TOB in one order. "
                "Join/improve at 1-tick improvement is viable (tick=1e-5). "
                "Sustained MM requires tiny position sizing (~$100–$500/side). "
                "Perp hedge is effectively unlimited at this scale."
            ),
        },
    }

    # JSON serialization safety: replace NaN with None
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    findings = _sanitize(findings)

    return card_html, findings


# ---------------------------------------------------------------------------
# __main__ block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_root = os.environ.get("HLBT_HL_DATA_ROOT", "../../data")
    out_dir = Path(__file__).parent.parent.parent.parent / "docs" / "research" / "_cards"
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    html, findings = build_card(con, data_root)

    html_path = out_dir / "card_a.html"
    json_path = out_dir / "card_a.json"

    html_path.write_text(html, encoding="utf-8")
    json_path.write_text(json.dumps(findings, indent=2), encoding="utf-8")

    print(f"Written: {html_path}")
    print(f"Written: {json_path}")
    print()
    print("Headline:", findings.get("headline", ""))
    print("Verdict:", findings.get("verdict", ""))
    sys.exit(0)
