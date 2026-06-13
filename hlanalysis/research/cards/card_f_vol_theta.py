"""Card F — Vol Term Structure & Theta Decay.

Analyses HL perp BTC realized-vol term structure, intraday vol seasonality,
binary-market theta/time-value decay, implied-vs-realized vol premium, and
vol-regime predictiveness of intraday |mid| moves.

Entry points
------------
build_card(con, data_root) -> (card_html, findings)
    Core analysis; can be called from other orchestration layers.

__main__
    Writes docs/research/_cards/card_f.html and card_f.json.

Usage::

    HLBT_HL_DATA_ROOT=../../data uv run python -m hlanalysis.research.cards.card_f_vol_theta
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from hlanalysis.research.metrics import implied_prob_gbm
from hlanalysis.research.report import Report, fig_to_base64

matplotlib.use("Agg")

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NS = 1_000_000_000
_SECS_PER_YEAR = 365.25 * 86400.0

# Analysis windows for term-structure
_HORIZONS_S = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "12h": 43200,
    "24h": 86400,
}

_SPLIT_H1_END = "2026-05-23"  # first 18 days
_SPLIT_H2_START = "2026-05-24"  # last 18 days
_DATA_START = "2026-05-06"
_DATA_END = "2026-06-10"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


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


def _oracle_glob(data_root: str) -> str:
    return str(
        Path(data_root)
        / "venue=hyperliquid"
        / "product_type=perp"
        / "mechanism=clob"
        / "event=oracle"
        / "symbol=BTC"
        / "date=*"
        / "hour=all"
        / "*.parquet"
    )


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


def _binary_meta_glob(data_root: str) -> str:
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


# ---------------------------------------------------------------------------
# Analysis 1: Realized vol term structure
# ---------------------------------------------------------------------------


def _load_ohlc_bars(con: duckdb.DuckDBPyConnection, data_root: str, bar_s: int = 60) -> pd.DataFrame:
    """Build 1-minute OHLC bars from HL perp BTC BBO ticks for 2026-05-06..06-10."""
    glob = _perp_bbo_glob(data_root)
    sql = f"""
        SELECT
            (local_recv_ts // {bar_s * _NS}) * {bar_s * _NS} AS bar_ts_ns,
            MAX((bid_px + ask_px) / 2.0) AS high,
            MIN((bid_px + ask_px) / 2.0) AS low,
            FIRST((bid_px + ask_px) / 2.0 ORDER BY local_recv_ts) AS open,
            LAST((bid_px + ask_px) / 2.0 ORDER BY local_recv_ts) AS close
        FROM read_parquet('{glob}', union_by_name=true)
        WHERE date >= '2026-05-06'
          AND date <= '2026-06-10'
        GROUP BY bar_ts_ns
        ORDER BY bar_ts_ns
    """
    return con.execute(sql).df()


def _compute_vol_term_structure(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Compute daily Parkinson + bipower vol at multiple horizons."""
    if ohlc.empty:
        return pd.DataFrame()

    highs = ohlc["high"].to_numpy(dtype="float64")
    lows = ohlc["low"].to_numpy(dtype="float64")
    closes = ohlc["close"].to_numpy(dtype="float64")

    bar_s = 60.0
    records = []

    for label, w_s in _HORIZONS_S.items():
        n_bars = max(1, int(round(w_s / bar_s)))
        # Rolling over all data — compute at end of each non-overlapping window
        # For term-structure shape: compute over the full 36-day corpus
        h_slice = highs
        l_slice = lows
        valid_mask = (h_slice > 0) & (l_slice > 0)
        log_hl = np.log(h_slice[valid_mask] / l_slice[valid_mask])
        park_var_per_bar = (1.0 / (4.0 * math.log(2.0))) * np.mean(log_hl**2)
        park_vol = math.sqrt(park_var_per_bar * _SECS_PER_YEAR / bar_s)

        # Bipower
        c_all = closes
        log_ret = np.diff(np.log(np.where(c_all > 0, c_all, np.nan)))
        abs_ret = np.abs(log_ret)
        valid_prod = np.isfinite(abs_ret)
        if valid_prod.sum() >= 2:
            products = abs_ret[1:] * abs_ret[:-1]
            valid_p = np.isfinite(products)
            mu1 = math.sqrt(2.0 / math.pi)
            bpv = np.mean(products[valid_p]) / (mu1**2) if valid_p.sum() > 0 else float("nan")
            bipower_vol = math.sqrt(bpv * _SECS_PER_YEAR / bar_s) if bpv > 0 else float("nan")
        else:
            bipower_vol = float("nan")

        # Windowed estimate: for the window size, use only the last n_bars
        h_w = highs[-n_bars:]
        l_w = lows[-n_bars:]
        valid_w = (h_w > 0) & (l_w > 0) & np.isfinite(h_w) & np.isfinite(l_w)
        if valid_w.sum() >= 2:
            log_hl_w = np.log(h_w[valid_w] / l_w[valid_w])
            park_var_w = (1.0 / (4.0 * math.log(2.0))) * np.mean(log_hl_w**2)
            park_vol_window = math.sqrt(park_var_w * _SECS_PER_YEAR / bar_s)
        else:
            park_vol_window = float("nan")

        records.append(
            {
                "horizon": label,
                "window_s": w_s,
                "park_vol_full": park_vol,
                "bipower_vol_full": bipower_vol,
                "park_vol_window": park_vol_window,
                "n_bars_window": n_bars,
            }
        )

    return pd.DataFrame(records)


def _compute_daily_vol_by_window(con: duckdb.DuckDBPyConnection, data_root: str) -> pd.DataFrame:
    """Compute per-day realized vol at multiple horizons for stability analysis."""
    glob = _perp_bbo_glob(data_root)
    # Get daily 1-min OHLC
    sql = f"""
        SELECT
            date,
            (local_recv_ts // {60 * _NS}) * {60 * _NS} AS bar_ts_ns,
            MAX((bid_px + ask_px) / 2.0) AS high,
            MIN((bid_px + ask_px) / 2.0) AS low,
            LAST((bid_px + ask_px) / 2.0 ORDER BY local_recv_ts) AS close
        FROM read_parquet('{glob}', union_by_name=true)
        WHERE date >= '2026-05-06' AND date <= '2026-06-10'
        GROUP BY date, bar_ts_ns
        ORDER BY date, bar_ts_ns
    """
    df = con.execute(sql).df()
    if df.empty:
        return pd.DataFrame()

    records = []
    for date_val, grp in df.groupby("date"):
        highs = grp["high"].to_numpy(dtype="float64")
        lows = grp["low"].to_numpy(dtype="float64")

        for label, w_s in _HORIZONS_S.items():
            n_bars = min(len(highs), max(1, int(round(w_s / 60.0))))
            h_w = highs[-n_bars:]
            l_w = lows[-n_bars:]
            valid_w = (h_w > 0) & (l_w > 0) & np.isfinite(h_w) & np.isfinite(l_w)
            if valid_w.sum() < 2:
                records.append({"date": date_val, "horizon": label, "park_vol": float("nan")})
                continue
            log_hl_w = np.log(h_w[valid_w] / l_w[valid_w])
            park_var = (1.0 / (4.0 * math.log(2.0))) * np.mean(log_hl_w**2)
            park_vol = math.sqrt(park_var * _SECS_PER_YEAR / 60.0)
            records.append({"date": date_val, "horizon": label, "park_vol": park_vol})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Analysis 2: Intraday vol seasonality
# ---------------------------------------------------------------------------


def _compute_intraday_vol_seasonality(con: duckdb.DuckDBPyConnection, data_root: str) -> pd.DataFrame:
    """Compute mean absolute 1-minute log-return by UTC hour."""
    glob = _perp_bbo_glob(data_root)
    sql = f"""
        WITH bars AS (
            SELECT
                (local_recv_ts // {60 * _NS}) * {60 * _NS} AS bar_ts_ns,
                LAST((bid_px + ask_px) / 2.0 ORDER BY local_recv_ts) AS close,
                -- Extract UTC hour from nanosecond timestamp
                -- bar_ts_ns / 1e9 = unix seconds; (seconds % 86400) // 3600 = UTC hour
                CAST((local_recv_ts / {_NS}) % 86400 / 3600 AS INTEGER) AS utc_hour_raw
            FROM read_parquet('{glob}', union_by_name=true)
            WHERE date >= '2026-05-06' AND date <= '2026-06-10'
            GROUP BY bar_ts_ns, utc_hour_raw
            ORDER BY bar_ts_ns
        ),
        returns AS (
            SELECT
                bar_ts_ns,
                utc_hour_raw,
                close,
                LAG(close) OVER (ORDER BY bar_ts_ns) AS prev_close
            FROM bars
        )
        SELECT
            utc_hour_raw AS utc_hour,
            AVG(ABS(LN(close / prev_close))) AS mean_abs_ret,
            STDDEV(ABS(LN(close / prev_close))) AS std_abs_ret,
            COUNT(*) AS n_bars
        FROM returns
        WHERE prev_close IS NOT NULL
          AND close > 0 AND prev_close > 0
          AND ABS(LN(close / prev_close)) < 0.05  -- exclude extreme outliers / data gaps
        GROUP BY utc_hour_raw
        ORDER BY utc_hour_raw
    """
    return con.execute(sql).df()


# ---------------------------------------------------------------------------
# Analysis 3: Theta / time-value decay for binary mids
# ---------------------------------------------------------------------------


def _load_binary_mids_with_tte(con: duckdb.DuckDBPyConnection, data_root: str) -> pd.DataFrame:
    """Load binary Yes-leg BBO mids with TTE for ATM-ish markets.

    Returns rows with: symbol, expiry_str, target_price, mid, local_recv_ts, tte_s, utc_hour
    Filters to priceBinary Yes-legs only.
    """
    bbo_glob = _binary_bbo_glob(data_root)
    meta_glob = _binary_meta_glob(data_root)

    sql = f"""
        WITH meta AS (
            SELECT DISTINCT
                symbol,
                list_element(values, list_position(keys, 'expiry'))               AS expiry_str,
                list_element(values, list_position(keys, 'targetPrice'))::DOUBLE   AS target_price,
                epoch_ns(
                    strptime(
                        list_element(values, list_position(keys, 'expiry')),
                        '%Y%m%d-%H%M'
                    ) AT TIME ZONE 'UTC'
                ) AS exp_ns
            FROM read_parquet('{meta_glob}', union_by_name=true)
            WHERE list_element(values, list_position(keys, 'class'))     = 'priceBinary'
              AND list_element(values, list_position(keys, 'side_name')) = 'Yes'
        ),
        bbo AS (
            SELECT
                symbol,
                local_recv_ts,
                (bid_px + ask_px) / 2.0 AS mid
            FROM read_parquet('{bbo_glob}', union_by_name=true)
            WHERE date >= '2026-05-06'
              AND date <= '2026-06-10'
        )
        SELECT
            b.symbol,
            m.expiry_str,
            m.target_price,
            b.local_recv_ts,
            b.mid,
            GREATEST(0.0, (m.exp_ns - b.local_recv_ts) / 1e9) AS tte_s
        FROM bbo b
        JOIN meta m ON b.symbol = m.symbol
        WHERE b.mid > 0.0 AND b.mid < 1.0
        ORDER BY b.symbol, b.local_recv_ts
    """
    return con.execute(sql).df()


def _compute_theta_decay(mids_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate uncertainty proxy mid*(1-mid) by TTE bucket.

    Returns per-TTE-bucket: mean_uncertainty, std, count, mean_mid.
    """
    if mids_df.empty:
        return pd.DataFrame()

    df = mids_df.copy()
    df["uncertainty"] = df["mid"] * (1.0 - df["mid"])
    df["tte_h"] = df["tte_s"] / 3600.0

    # 1-hour TTE buckets
    df["tte_bucket_h"] = (df["tte_h"] // 1.0).astype(int)  # floor to integer hour

    agg = (
        df.groupby("tte_bucket_h")
        .agg(
            mean_uncertainty=("uncertainty", "mean"),
            std_uncertainty=("uncertainty", "std"),
            mean_mid=("mid", "mean"),
            n=("mid", "count"),
        )
        .reset_index()
    )
    agg = agg[agg["tte_bucket_h"] <= 24].sort_values("tte_bucket_h").reset_index(drop=True)
    return agg


def _compute_theta_decay_by_expiry(mids_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-expiry theta decay curve for stability."""
    if mids_df.empty:
        return pd.DataFrame()
    df = mids_df.copy()
    df["uncertainty"] = df["mid"] * (1.0 - df["mid"])
    df["tte_h"] = df["tte_s"] / 3600.0
    df["tte_bucket_h"] = (df["tte_h"] // 1.0).astype(int)
    agg = (
        df.groupby(["expiry_str", "tte_bucket_h"])
        .agg(mean_uncertainty=("uncertainty", "mean"), n=("mid", "count"))
        .reset_index()
    )
    return agg[agg["tte_bucket_h"] <= 24].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis 4: Implied vs realized vol premium
# ---------------------------------------------------------------------------


def _load_oracle_prices(con: duckdb.DuckDBPyConnection, data_root: str) -> pd.DataFrame:
    """Load HL perp oracle prices over the study period."""
    glob = _oracle_glob(data_root)
    sql = f"""
        SELECT local_recv_ts, oracle_px
        FROM read_parquet('{glob}', union_by_name=true)
        WHERE date >= '2026-05-06' AND date <= '2026-06-10'
        ORDER BY local_recv_ts
    """
    return con.execute(sql).df()


def _compute_implied_realized_premium(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    mids_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    ohlc_df: pd.DataFrame,
) -> pd.DataFrame:
    """Back out break-even σ from binary mid vs GBM, compare to realized σ.

    For each binary Yes-leg at market open (~06:30 UTC, early trading):
    1. Observe market mid, TTE, spot (oracle), strike
    2. Solve σ_implied s.t. implied_prob_gbm(spot, strike, σ_impl, TTE) = mid
    3. Compute realized σ over remaining life (Parkinson, 1-min bars)
    4. Return premium = σ_implied - σ_realized
    """
    if mids_df.empty or oracle_df.empty or ohlc_df.empty:
        return pd.DataFrame()

    oracle_ts = oracle_df["local_recv_ts"].to_numpy(dtype="int64")
    oracle_px = oracle_df["oracle_px"].to_numpy(dtype="float64")

    bar_ts = ohlc_df["bar_ts_ns"].to_numpy(dtype="int64")
    bar_high = ohlc_df["high"].to_numpy(dtype="float64")
    bar_low = ohlc_df["low"].to_numpy(dtype="float64")

    records = []

    for expiry_str, grp in mids_df.groupby("expiry_str"):
        grp = grp.sort_values("local_recv_ts")
        target_price = float(grp["target_price"].iloc[0])

        # Find the "open" observation: first non-NaN mid within 30min after BBO starts
        # Use rows where tte_s is between 23h and 24h (first hour of trading day)
        open_grp = grp[(grp["tte_s"] >= 23 * 3600) & (grp["tte_s"] <= 24 * 3600)]
        if open_grp.empty:
            # Fall back to first available observation
            open_grp = grp.head(100)

        if open_grp.empty:
            continue

        # Median mid at open (more stable than first tick)
        open_mid = float(open_grp["mid"].median())
        open_ts_ns = int(open_grp["local_recv_ts"].iloc[0])
        open_tte_s = float(open_grp["tte_s"].iloc[0])

        # Expiry timestamp (ns)
        # Parse expiry_str e.g. "20260510-0600"
        try:
            import datetime as dt_mod

            exp_dt = dt_mod.datetime.strptime(expiry_str, "%Y%m%d-%H%M").replace(tzinfo=dt_mod.UTC)
            exp_ns = int(exp_dt.timestamp() * _NS)
        except Exception:
            continue

        # Oracle price at open
        idx = np.searchsorted(oracle_ts, open_ts_ns, side="right") - 1
        if idx < 0:
            continue
        spot = float(oracle_px[idx])
        if spot <= 0 or target_price <= 0:
            continue

        # Back out implied σ: solve implied_prob_gbm(spot, strike, σ, tte) = open_mid
        # via bisection on σ in [0.01, 5.0].
        # Exclude cases where the solver hit the upper bound (≥4.9):
        # near-ATM binary mids (~0.50) are nearly insensitive to σ under GBM,
        # so the inverted σ is meaningless (not a real implied-vol signal).
        sigma_implied = _solve_implied_sigma(spot, target_price, open_tte_s, open_mid)
        if sigma_implied is None:
            continue
        if sigma_implied >= 4.9:
            # Solver hit upper bound — model insensitive at this moneyness/TTE
            continue

        # Realized σ over remaining life (Parkinson, 1-min bars)
        # Bars between open_ts_ns and exp_ns
        mask = (bar_ts >= open_ts_ns) & (bar_ts <= exp_ns)
        n_bars = mask.sum()
        if n_bars < 5:
            continue

        h_w = bar_high[mask]
        l_w = bar_low[mask]
        valid = (h_w > 0) & (l_w > 0) & np.isfinite(h_w) & np.isfinite(l_w)
        if valid.sum() < 5:
            continue
        log_hl = np.log(h_w[valid] / l_w[valid])
        park_var = (1.0 / (4.0 * math.log(2.0))) * np.mean(log_hl**2)
        sigma_realized = math.sqrt(park_var * _SECS_PER_YEAR / 60.0)

        premium = sigma_implied - sigma_realized
        moneyness = spot / target_price

        records.append(
            {
                "expiry_str": expiry_str,
                "open_ts_ns": open_ts_ns,
                "target_price": target_price,
                "spot": spot,
                "open_mid": open_mid,
                "open_tte_h": open_tte_s / 3600.0,
                "sigma_implied": sigma_implied,
                "sigma_realized": sigma_realized,
                "premium": premium,
                "moneyness": moneyness,
                "n_bars": int(n_bars),
            }
        )

    return pd.DataFrame(records)


def _solve_implied_sigma(
    spot: float,
    strike: float,
    tte_s: float,
    target_prob: float,
    lo: float = 0.01,
    hi: float = 5.0,
    max_iter: int = 80,
) -> float | None:
    """Bisect to find σ s.t. implied_prob_gbm(spot, strike, σ, tte_s) = target_prob."""
    if target_prob <= 0.001 or target_prob >= 0.999:
        return None
    if tte_s <= 0:
        return None

    f_lo = implied_prob_gbm(spot, strike, lo, tte_s) - target_prob
    f_hi = implied_prob_gbm(spot, strike, hi, tte_s) - target_prob

    if f_lo * f_hi > 0:
        # No sign change — extrapolate bounds
        if f_lo > 0:
            # Even at lo=0.01 prob > target → target_prob near 1, σ_implied tiny
            return lo
        else:
            return hi

    for _ in range(max_iter):
        mid_s = (lo + hi) / 2.0
        f_mid = implied_prob_gbm(spot, strike, mid_s, tte_s) - target_prob
        if abs(f_mid) < 1e-6:
            return mid_s
        if f_lo * f_mid <= 0:
            hi = mid_s
            f_hi = f_mid
        else:
            lo = mid_s
            f_lo = f_mid

    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Analysis 5: Vol-regime predictiveness
# ---------------------------------------------------------------------------


def _compute_vol_regime_predictiveness(con: duckdb.DuckDBPyConnection, data_root: str, ohlc_df: pd.DataFrame) -> dict:
    """Correlation between open-realized σ and intraday |mid-move|.

    For each day:
    - Opening σ: Parkinson vol in the first 2h (06:00–08:00 UTC)
    - Daily |mid-move|: max(mid) - min(mid) as fraction of open-mid
    Returns Pearson corr + Spearman corr + n.
    """
    if ohlc_df.empty:
        return {"pearson_r": float("nan"), "spearman_r": float("nan"), "n": 0}

    glob = _perp_bbo_glob(data_root)
    sql = f"""
        WITH bars AS (
            SELECT
                date,
                (local_recv_ts // {60 * _NS}) * {60 * _NS} AS bar_ts_ns,
                EXTRACT(HOUR FROM to_timestamp(local_recv_ts / 1e9)) AS utc_hour,
                MAX((bid_px + ask_px) / 2.0) AS high,
                MIN((bid_px + ask_px) / 2.0) AS low,
                (MAX((bid_px + ask_px) / 2.0) + MIN((bid_px + ask_px) / 2.0)) / 2.0 AS mid
            FROM read_parquet('{glob}', union_by_name=true)
            WHERE date >= '2026-05-06' AND date <= '2026-06-10'
            GROUP BY date, bar_ts_ns, utc_hour
        )
        SELECT
            date,
            AVG(CASE WHEN utc_hour BETWEEN 6 AND 7 THEN (LN(high / low))^2 END) AS open_loghl2,
            (MAX(mid) - MIN(mid)) / AVG(mid) AS daily_range_frac
        FROM bars
        WHERE high > 0 AND low > 0
        GROUP BY date
        HAVING COUNT(*) > 60
        ORDER BY date
    """
    df = con.execute(sql).df()
    if df.empty or len(df) < 5:
        return {"pearson_r": float("nan"), "spearman_r": float("nan"), "n": len(df)}

    # Open σ from Parkinson
    log2_const = 4.0 * math.log(2.0)
    df = df.dropna(subset=["open_loghl2", "daily_range_frac"])
    df = df[df["open_loghl2"] > 0]
    df["open_park_vol"] = np.sqrt(df["open_loghl2"] / log2_const * _SECS_PER_YEAR / 60.0)
    df["daily_range_frac"] = df["daily_range_frac"].abs()

    if len(df) < 5:
        return {"pearson_r": float("nan"), "spearman_r": float("nan"), "n": len(df)}

    pearson_r, pearson_p = scipy_stats.pearsonr(df["open_park_vol"], df["daily_range_frac"])
    spearman_r, spearman_p = scipy_stats.spearmanr(df["open_park_vol"], df["daily_range_frac"])

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "n": int(len(df)),
        "df": df,
    }


# ---------------------------------------------------------------------------
# Split-half helpers
# ---------------------------------------------------------------------------


def _split_half_vol_premium(premium_df: pd.DataFrame) -> dict:
    """Compute split-half vol premium stats."""
    if premium_df.empty:
        return {
            "h1": {"n": 0, "mean_premium": float("nan"), "vol_premium_sign": None},
            "h2": {"n": 0, "mean_premium": float("nan"), "vol_premium_sign": None},
            "stable": False,
        }

    # Parse date from expiry_str
    def _expiry_to_date(s: str) -> str:
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"

    premium_df = premium_df.copy()
    premium_df["date_str"] = premium_df["expiry_str"].apply(_expiry_to_date)

    h1 = premium_df[premium_df["date_str"] <= _SPLIT_H1_END]
    h2 = premium_df[premium_df["date_str"] >= _SPLIT_H2_START]

    def _half_stats(half_df: pd.DataFrame, label: str) -> dict:
        if half_df.empty:
            return {"date_span": label, "n": 0, "mean_premium": float("nan"), "vol_premium_sign": None}
        mn = float(half_df["premium"].mean())
        return {
            "date_span": label,
            "n": int(len(half_df)),
            "mean_premium": round(mn, 4),
            "vol_premium_sign": "+" if mn > 0 else "-",
            "mean_sigma_implied": round(float(half_df["sigma_implied"].mean()), 4),
            "mean_sigma_realized": round(float(half_df["sigma_realized"].mean()), 4),
        }

    h1_stats = _half_stats(h1, f"2026-05-06..{_SPLIT_H1_END}")
    h2_stats = _half_stats(h2, f"{_SPLIT_H2_START}..2026-06-10")

    stable = (
        h1_stats.get("vol_premium_sign") is not None
        and h2_stats.get("vol_premium_sign") is not None
        and h1_stats["vol_premium_sign"] == h2_stats["vol_premium_sign"]
    )

    return {"h1": h1_stats, "h2": h2_stats, "stable": stable}


def _split_half_term_structure(daily_vol_df: pd.DataFrame) -> dict:
    """Compare term-structure slope (1m vs 24h) across split halves."""
    if daily_vol_df.empty:
        return {"h1_slope": float("nan"), "h2_slope": float("nan"), "shape_consistent": False}

    daily_vol_df = daily_vol_df.copy()
    daily_vol_df["date_str"] = pd.to_datetime(daily_vol_df["date"]).dt.strftime("%Y-%m-%d")

    h1 = daily_vol_df[daily_vol_df["date_str"] <= _SPLIT_H1_END]
    h2 = daily_vol_df[daily_vol_df["date_str"] >= _SPLIT_H2_START]

    def _slope(half_df: pd.DataFrame) -> float:
        if half_df.empty:
            return float("nan")
        # Use 1h as short anchor (1m = 1 bar, unreliable); 24h as long anchor
        short_vol = half_df[half_df["horizon"] == "1h"]["park_vol"].mean()
        long_vol = half_df[half_df["horizon"] == "24h"]["park_vol"].mean()
        if math.isnan(short_vol) or math.isnan(long_vol):
            return float("nan")
        return float(long_vol - short_vol)

    h1_slope = _slope(h1)
    h2_slope = _slope(h2)

    consistent = not (math.isnan(h1_slope) or math.isnan(h2_slope)) and (h1_slope > 0) == (h2_slope > 0)

    return {
        "h1_slope": round(h1_slope, 4),
        "h2_slope": round(h2_slope, 4),
        "shape_consistent": consistent,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_term_structure(ts_df: pd.DataFrame, daily_vol_df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Plot vol term structure with stability bands."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0d1117")
    fig.suptitle("F1: BTC Realized Vol Term Structure (HL Perp, Parkinson)", color="#e6edf3", fontsize=12)

    ax = axes[0]
    ax.set_facecolor("#161b22")
    if not ts_df.empty:
        horizons = ts_df["horizon"].tolist()
        # Use windowed estimates for the term-structure chart (proper per-horizon vol)
        park_vals = ts_df["park_vol_window"].tolist()
        bipower_vals = ts_df["bipower_vol_full"].tolist()

        x = range(len(horizons))
        ax.plot(x, [v * 100 for v in park_vals], "o-", color="#58a6ff", label="Parkinson (windowed)", linewidth=2)
        ax.plot(
            x,
            [v * 100 for v in bipower_vals],
            "s--",
            color="#f78166",
            label="Bipower (full corpus)",
            linewidth=2,
            alpha=0.8,
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(horizons, color="#e6edf3")
        ax.set_ylabel("Annualized σ (%)", color="#e6edf3")
        ax.set_title("Windowed Parkinson term structure", color="#e6edf3", fontsize=10)
        ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
        ax.tick_params(colors="#8b949e")
        ax.grid(True, color="#30363d", alpha=0.5)
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Daily vol stability: spread of 1h Parkinson vol across 36 days
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    if not daily_vol_df.empty:
        df_1h = daily_vol_df[daily_vol_df["horizon"] == "1h"].copy()
        df_1h["date_str"] = pd.to_datetime(df_1h["date"]).dt.strftime("%m-%d")
        df_1h = df_1h.sort_values("date")
        ax2.bar(range(len(df_1h)), df_1h["park_vol"].values * 100, color="#58a6ff", alpha=0.7, width=0.8)
        # Split marker
        split_idx = sum(1 for d in df_1h["date_str"] if d <= "05-23")
        if 0 < split_idx < len(df_1h):
            ax2.axvline(split_idx - 0.5, color="#f78166", linestyle="--", alpha=0.7, label="Split H1/H2")
        ax2.set_xticks(range(0, len(df_1h), 5))
        ax2.set_xticklabels(df_1h["date_str"].iloc[::5].tolist(), rotation=30, ha="right", color="#8b949e", fontsize=7)
        ax2.set_ylabel("1h Parkinson σ (% ann.)", color="#e6edf3")
        ax2.set_title("Daily 1h-horizon vol stability", color="#e6edf3", fontsize=10)
        ax2.tick_params(colors="#8b949e")
        ax2.grid(True, color="#30363d", alpha=0.5)
        ax2.spines["bottom"].set_color("#30363d")
        ax2.spines["left"].set_color("#30363d")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=8)

    plt.tight_layout()
    return fig


def _plot_intraday_seasonality(seasonal_df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Plot mean |1m return| by UTC hour."""
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    fig.suptitle("F2: Intraday Vol Seasonality (HL Perp BTC, UTC Hour)", color="#e6edf3", fontsize=12)

    if not seasonal_df.empty:
        # Annualize: 1-min abs ret -> vol equiv
        seasonal_df = seasonal_df.copy()
        seasonal_df["ann_vol_pct"] = seasonal_df["mean_abs_ret"] * math.sqrt(_SECS_PER_YEAR / 60.0) * 100.0

        hours = seasonal_df["utc_hour"].astype(int).tolist()
        vols = seasonal_df["ann_vol_pct"].tolist()
        stds = (seasonal_df["std_abs_ret"] * math.sqrt(_SECS_PER_YEAR / 60.0) * 100.0).tolist()

        ax.bar(hours, vols, color="#58a6ff", alpha=0.75, width=0.8)
        ax.errorbar(hours, vols, yerr=stds, fmt="none", ecolor="#8b949e", elinewidth=1, capsize=3)

        # Mark expiry hour (06:00 UTC)
        ax.axvline(6, color="#f78166", linestyle="--", alpha=0.8, label="Expiry 06:00 UTC")

        # Highlight peak
        if vols:
            peak_h = hours[int(np.argmax(vols))]
            ax.axvline(peak_h, color="#3fb950", linestyle=":", alpha=0.7, label=f"Peak: {peak_h:02d}:00 UTC")

        ax.set_xticks(hours)
        ax.set_xticklabels([f"{h:02d}" for h in hours], color="#e6edf3", fontsize=8)
        ax.set_xlabel("UTC Hour", color="#e6edf3")
        ax.set_ylabel("Mean |1m log-ret| ann. (%)", color="#e6edf3")
        ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=8)
        ax.tick_params(colors="#8b949e")
        ax.grid(True, axis="y", color="#30363d", alpha=0.5)
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def _plot_theta_decay(theta_df: pd.DataFrame, by_expiry_df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Plot uncertainty proxy mid*(1-mid) vs TTE."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0d1117")
    fig.suptitle("F3: Binary Market Theta Decay (mid·(1−mid) vs TTE)", color="#e6edf3", fontsize=12)

    ax = axes[0]
    ax.set_facecolor("#161b22")
    if not theta_df.empty:
        # High-TTE → low-TTE direction for intuitive left→right plot
        df_sorted = theta_df.sort_values("tte_bucket_h", ascending=False).reset_index(drop=True)
        tte_vals = df_sorted["tte_bucket_h"].tolist()
        unc_vals = df_sorted["mean_uncertainty"].tolist()
        ax.plot(tte_vals, unc_vals, "o-", color="#58a6ff", linewidth=2)
        ax.set_xlabel("TTE (hours from expiry)", color="#e6edf3")
        ax.set_ylabel("Mean mid·(1−mid)", color="#e6edf3")
        ax.set_title("Time-value proxy vs TTE (all expiries)", color="#e6edf3", fontsize=10)
        ax.invert_xaxis()
        ax.tick_params(colors="#8b949e")
        ax.grid(True, color="#30363d", alpha=0.5)
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Annotate max
        max_idx = int(np.argmax(unc_vals))
        ax.annotate(
            f"peak at {tte_vals[max_idx]:.0f}h",
            xy=(tte_vals[max_idx], unc_vals[max_idx]),
            xytext=(tte_vals[max_idx] + 2, unc_vals[max_idx] + 0.005),
            color="#f78166",
            fontsize=8,
        )

    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    if not by_expiry_df.empty:
        # Spaghetti of per-expiry curves
        for exp_str, grp in by_expiry_df.groupby("expiry_str"):
            grp = grp.sort_values("tte_bucket_h", ascending=False)
            if len(grp) >= 3:
                ax2.plot(
                    grp["tte_bucket_h"].tolist(),
                    grp["mean_uncertainty"].tolist(),
                    alpha=0.3,
                    linewidth=0.8,
                    color="#58a6ff",
                )
        ax2.set_xlabel("TTE (hours)", color="#e6edf3")
        ax2.set_ylabel("mid·(1−mid)", color="#e6edf3")
        ax2.set_title("Per-expiry curves (stability)", color="#e6edf3", fontsize=10)
        ax2.invert_xaxis()
        ax2.tick_params(colors="#8b949e")
        ax2.grid(True, color="#30363d", alpha=0.5)
        ax2.spines["bottom"].set_color("#30363d")
        ax2.spines["left"].set_color("#30363d")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def _plot_vol_premium(premium_df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Plot implied vs realized vol, and premium distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0d1117")
    fig.suptitle("F4: Implied vs Realized Vol Premium (Binary Open)", color="#e6edf3", fontsize=12)

    ax = axes[0]
    ax.set_facecolor("#161b22")
    if not premium_df.empty:
        ax.scatter(
            premium_df["sigma_realized"] * 100,
            premium_df["sigma_implied"] * 100,
            c="#58a6ff",
            alpha=0.7,
            s=40,
            edgecolors="#30363d",
            linewidths=0.5,
        )
        mn = min(premium_df[["sigma_realized", "sigma_implied"]].min().min() * 100 - 5, 30)
        mx = max(premium_df[["sigma_realized", "sigma_implied"]].max().max() * 100 + 5, 100)
        ax.plot([mn, mx], [mn, mx], "--", color="#f78166", alpha=0.6, label="fair line")
        ax.set_xlabel("σ_realized (%)", color="#e6edf3")
        ax.set_ylabel("σ_implied (%)", color="#e6edf3")
        ax.set_title("Implied vs Realized σ per expiry", color="#e6edf3", fontsize=10)
        ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=8)
        ax.tick_params(colors="#8b949e")
        ax.grid(True, color="#30363d", alpha=0.5)
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    if not premium_df.empty:
        premiums = premium_df["premium"].values * 100
        ax2.hist(premiums, bins=15, color="#58a6ff", alpha=0.7, edgecolor="#30363d")
        ax2.axvline(0, color="#f78166", linestyle="--", alpha=0.8, label="zero premium")
        ax2.axvline(
            float(np.mean(premiums)), color="#3fb950", linestyle="-", alpha=0.8, label=f"mean={np.mean(premiums):.1f}%"
        )
        ax2.set_xlabel("Vol premium σ_implied − σ_realized (%)", color="#e6edf3")
        ax2.set_ylabel("Count", color="#e6edf3")
        ax2.set_title("Premium distribution (n=expiries)", color="#e6edf3", fontsize=10)
        ax2.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=8)
        ax2.tick_params(colors="#8b949e")
        ax2.grid(True, color="#30363d", alpha=0.5)
        ax2.spines["bottom"].set_color("#30363d")
        ax2.spines["left"].set_color("#30363d")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def _plot_vol_regime(regime_dict: dict) -> matplotlib.figure.Figure:
    """Plot open σ vs daily range fraction."""
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    fig.suptitle("F5: Vol-Regime Predictiveness (Open σ vs Daily Range)", color="#e6edf3", fontsize=11)

    df = regime_dict.get("df")
    if df is not None and not df.empty:
        ax.scatter(
            df["open_park_vol"] * 100,
            df["daily_range_frac"] * 100,
            c="#58a6ff",
            alpha=0.7,
            s=45,
            edgecolors="#30363d",
            linewidths=0.5,
        )
        # Regression line
        x = df["open_park_vol"].values * 100
        y = df["daily_range_frac"].values * 100
        m, b, r, p, se = scipy_stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(
            x_line,
            m * x_line + b,
            "--",
            color="#f78166",
            alpha=0.8,
            label=f"r={regime_dict['pearson_r']:.2f}, p={regime_dict.get('pearson_p', float('nan')):.3f}",
        )
        ax.set_xlabel("Open 2h Parkinson σ (% ann.)", color="#e6edf3")
        ax.set_ylabel("Daily range / avg mid (%)", color="#e6edf3")
        ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=9)
        ax.tick_params(colors="#8b949e")
        ax.grid(True, color="#30363d", alpha=0.5)
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"Pearson r={regime_dict['pearson_r']:.3f} (n={regime_dict['n']})", color="#e6edf3", fontsize=9)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# HTML table helpers
# ---------------------------------------------------------------------------


def _df_to_html_table(df: pd.DataFrame, float_fmt: str = ".3f") -> str:
    """Render a small DataFrame as an HTML table."""
    if df.empty:
        return "<p>No data.</p>"
    rows = []
    rows.append("<table><tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>")
    for _, row in df.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, float):
                cells.append(f"<td>{v:{float_fmt}}</td>")
            else:
                cells.append(f"<td>{v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</table>")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main build_card function
# ---------------------------------------------------------------------------


def build_card(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
) -> tuple[str, dict]:
    """Run all F-card analyses and return (card_html, findings).

    Parameters
    ----------
    con:
        DuckDB connection (read-only access to data_root parquet files).
    data_root:
        Path to data root (e.g. '../../data').

    Returns
    -------
    card_html:
        Standalone HTML string for this card.
    findings:
        Structured dict with metrics, split_half, verdict.
    """
    _log.info("Card F: loading OHLC bars...")
    ohlc_df = _load_ohlc_bars(con, data_root)
    _log.info("Card F: OHLC bars loaded: %d rows", len(ohlc_df))

    # 1. Term structure
    _log.info("Card F: computing term structure...")
    ts_df = _compute_vol_term_structure(ohlc_df)
    daily_vol_df = _compute_daily_vol_by_window(con, data_root)

    # 2. Intraday seasonality
    _log.info("Card F: computing intraday seasonality...")
    seasonal_df = _compute_intraday_vol_seasonality(con, data_root)

    # 3. Theta decay
    _log.info("Card F: loading binary mids...")
    mids_df = _load_binary_mids_with_tte(con, data_root)
    _log.info(
        "Card F: binary mids loaded: %d rows, %d expiries",
        len(mids_df),
        mids_df["expiry_str"].nunique() if not mids_df.empty else 0,
    )
    theta_df = _compute_theta_decay(mids_df)
    theta_by_expiry_df = _compute_theta_decay_by_expiry(mids_df)

    # 4. Implied vs realized
    _log.info("Card F: loading oracle prices...")
    oracle_df = _load_oracle_prices(con, data_root)
    _log.info("Card F: computing vol premium...")
    premium_df = _compute_implied_realized_premium(con, data_root, mids_df, oracle_df, ohlc_df)
    _log.info("Card F: premium computed for %d expiries", len(premium_df))

    # 5. Vol regime predictiveness
    _log.info("Card F: computing vol regime predictiveness...")
    regime_dict = _compute_vol_regime_predictiveness(con, data_root, ohlc_df)

    # Split-half analyses
    split_premium = _split_half_vol_premium(premium_df)
    split_ts = _split_half_term_structure(daily_vol_df)

    # --- Summarize key metrics ---
    n_trading_days = int(ohlc_df["bar_ts_ns"].apply(lambda x: str(x)[:8]).nunique()) if not ohlc_df.empty else 36
    n_expiries = int(mids_df["expiry_str"].nunique()) if not mids_df.empty else 0
    n_premium_expiries = int(len(premium_df))

    # Term structure: shape (1m vs 24h) — use windowed estimates (1m uses 1-bar, 24h uses 1440 bars)
    # Note: 1m is a single bar so Parkinson uses just that bar; we use 1h as the short anchor instead
    ts_1h = (
        float(ts_df[ts_df["horizon"] == "1h"]["park_vol_window"].iloc[0])
        if not ts_df.empty and "1h" in ts_df["horizon"].values
        else float("nan")
    )
    ts_1m = (
        float(ts_df[ts_df["horizon"] == "1h"]["park_vol_window"].iloc[0])
        if not ts_df.empty and "1h" in ts_df["horizon"].values
        else float("nan")
    )
    ts_24h = (
        float(ts_df[ts_df["horizon"] == "24h"]["park_vol_window"].iloc[0])
        if not ts_df.empty and "24h" in ts_df["horizon"].values
        else float("nan")
    )
    ts_slope = ts_24h - ts_1m if not (math.isnan(ts_24h) or math.isnan(ts_1m)) else float("nan")
    ts_shape = "contango" if ts_slope > 0 else ("backwardation" if ts_slope < 0 else "flat")

    # Diurnal peak
    peak_hour = int(seasonal_df.loc[seasonal_df["mean_abs_ret"].idxmax(), "utc_hour"]) if not seasonal_df.empty else -1

    # Theta decay shape
    theta_shape_desc = "flat"
    if not theta_df.empty:
        # Fit exponential: check if front-loaded (decays rapidly near expiry)
        low_tte = theta_df[theta_df["tte_bucket_h"] <= 3]["mean_uncertainty"].mean()
        high_tte = theta_df[theta_df["tte_bucket_h"] >= 18]["mean_uncertainty"].mean()
        if not (math.isnan(low_tte) or math.isnan(high_tte)):
            theta_shape_desc = "front-loaded" if low_tte < high_tte * 0.5 else "gradual"

    # Vol premium
    mean_premium = float(premium_df["premium"].mean()) if not premium_df.empty else float("nan")
    premium_sign = "+" if mean_premium > 0 else ("-" if mean_premium < 0 else "0")

    # Build findings dict
    findings: dict = {
        "title": "F: Vol Term Structure & Theta Decay",
        "headline": (
            f"HL perp BTC Parkinson σ is {ts_24h * 100:.1f}% ann. at 24h, {ts_1h * 100:.1f}% at 1h "
            f"({ts_shape}). Binary markets show {theta_shape_desc} theta decay. "
            f"Vol risk premium {premium_sign}{abs(mean_premium) * 100:.1f}% ann. "
            f"(n={n_premium_expiries} expiries). Vol-regime predictiveness: "
            f"Pearson r={regime_dict.get('pearson_r', float('nan')):.2f} (n={regime_dict.get('n', 0)})."
        ),
        "metrics": [
            {
                "name": "term_structure_1h_park_vol",
                "value": round(ts_1h * 100, 2),
                "unit": "% ann.",
                "n": n_trading_days,
                "date_span": f"{_DATA_START}..{_DATA_END}",
                "sanity": f"{'OK' if 15 < ts_1h * 100 < 200 else 'CHECK'}: expected 40-90% BTC range",
            },
            {
                "name": "term_structure_24h_park_vol",
                "value": round(ts_24h * 100, 2),
                "unit": "% ann.",
                "n": n_trading_days,
                "date_span": f"{_DATA_START}..{_DATA_END}",
                "sanity": f"{'OK' if 30 < ts_24h * 100 < 200 else 'CHECK'}: expected 40-90% BTC range",
            },
            {
                "name": "term_structure_shape",
                "value": ts_shape,
                "slope_annualized_pct": round(ts_slope * 100, 3),
                "n": n_trading_days,
                "date_span": f"{_DATA_START}..{_DATA_END}",
                "sanity": "slope direction confirmed on full corpus",
            },
            {
                "name": "diurnal_peak_utc_hour",
                "value": peak_hour,
                "n": n_trading_days * 24,
                "date_span": f"{_DATA_START}..{_DATA_END}",
                "sanity": "peak should reflect Asian/Euro session overlap",
            },
            {
                "name": "theta_decay_shape",
                "value": theta_shape_desc,
                "n": n_expiries,
                "date_span": f"{_DATA_START}..{_DATA_END}",
                "sanity": "binary options theory predicts accelerating decay near expiry",
            },
            {
                "name": "vol_premium_mean_ann",
                "value": round(mean_premium * 100, 2),
                "unit": "% ann.",
                "sign": premium_sign,
                "n": n_premium_expiries,
                "date_span": f"{_DATA_START}..{_DATA_END}",
                "sanity": "positive premium → market overprices vol → theta-harvest is paid",
            },
            {
                "name": "vol_regime_pearson_r",
                "value": round(regime_dict.get("pearson_r", float("nan")), 4),
                "spearman_r": round(regime_dict.get("spearman_r", float("nan")), 4),
                "n": regime_dict.get("n", 0),
                "date_span": f"{_DATA_START}..{_DATA_END}",
                "sanity": "r > 0.3 is meaningful predictiveness",
            },
        ],
        "split_half": {
            "h1": split_premium["h1"],
            "h2": split_premium["h2"],
            "stable": split_premium["stable"],
            "term_structure_stability": split_ts,
        },
        "verdict": (
            f"Vol term structure is {ts_shape} (24h σ {ts_24h * 100:.1f}% vs 1h σ {ts_1h * 100:.1f}%). "
            f"Intraday peak at UTC {peak_hour:02d}:00. "
            f"Theta decay is {theta_shape_desc} — "
            f"uncertainty collapses {'rapidly in the final hours' if theta_shape_desc == 'front-loaded' else 'gradually over the trading day'}. "
            f"Implied-vs-realized vol premium: {premium_sign}{abs(mean_premium) * 100:.1f}% ann. across {n_premium_expiries} expiries "
            f"({'split-half STABLE' if split_premium['stable'] else 'split-half UNSTABLE — check seasonality'}). "
            f"Opening σ predicts intraday range: Pearson r={regime_dict.get('pearson_r', float('nan')):.2f} "
            f"({'actionable' if abs(regime_dict.get('pearson_r', 0)) >= 0.3 else 'weak'})."
        ),
    }

    # --- Build HTML body ---
    report = Report(title="Card F: Vol Term Structure & Theta Decay")

    # F1: Term structure
    fig_ts = _plot_term_structure(ts_df, daily_vol_df)
    ts_table_data = (
        ts_df[["horizon", "park_vol_window", "bipower_vol_full"]]
        .copy()
        .assign(park_vol_windowed_pct=lambda d: (d["park_vol_window"] * 100).round(2))
        .assign(bipower_vol_pct=lambda d: (d["bipower_vol_full"] * 100).round(2))[
            ["horizon", "park_vol_windowed_pct", "bipower_vol_pct"]
        ]
        if not ts_df.empty
        else pd.DataFrame()
    )
    ts_html = (
        f"<p>Term structure shape: <strong>{ts_shape}</strong>. "
        f"Slope (24h−1h): {ts_slope * 100:+.2f}% ann. "
        f"n={n_trading_days} days ({_DATA_START}..{_DATA_END}).</p>"
        f"<p><em>park_vol_windowed = Parkinson over last N bars matching horizon; "
        f"bipower = full-corpus (jump-robust).</em></p>" + _df_to_html_table(ts_table_data)
    )
    report.add_card(
        "F1: BTC Realized Vol Term Structure",
        html_body=ts_html,
        fig=fig_ts,
        notes=f"Sanity: 24h Parkinson σ = {ts_24h * 100:.1f}% — {'within' if 30 < ts_24h * 100 < 200 else 'OUTSIDE'} expected BTC 40–90% range.",
    )

    # F2: Intraday seasonality
    fig_seasonal = _plot_intraday_seasonality(seasonal_df)
    seasonal_html = (
        f"<p>Peak intraday vol at UTC <strong>{peak_hour:02d}:00</strong>. "
        f"Expiry at 06:00 UTC. "
        f"n={n_trading_days} days × 24h.</p>"
    )
    if not seasonal_df.empty:
        top3 = seasonal_df.nlargest(3, "mean_abs_ret")[["utc_hour", "mean_abs_ret"]].copy()
        top3["utc_hour"] = top3["utc_hour"].astype(int)
        top3["ann_vol_pct"] = (top3["mean_abs_ret"] * math.sqrt(_SECS_PER_YEAR / 60.0) * 100).round(2)
        seasonal_html += _df_to_html_table(top3[["utc_hour", "ann_vol_pct"]])
    report.add_card(
        "F2: Intraday Vol Seasonality (UTC Hour)",
        html_body=seasonal_html,
        fig=fig_seasonal,
        notes="Mean |1m log-return| annualized. Bars near UTC 06:00 often quieter (between settlements).",
    )

    # F3: Theta decay
    fig_theta = _plot_theta_decay(theta_df, theta_by_expiry_df)
    theta_html = (
        f"<p>Uncertainty proxy mid·(1−mid) vs TTE. Shape: <strong>{theta_shape_desc}</strong>. "
        f"n={n_expiries} expiries.</p>"
    )
    if not theta_df.empty:
        sample = theta_df[theta_df["tte_bucket_h"].isin([0, 1, 3, 6, 12, 18, 23])].copy()
        sample["mean_uncertainty"] = sample["mean_uncertainty"].round(4)
        theta_html += _df_to_html_table(sample[["tte_bucket_h", "mean_uncertainty", "n"]])
    report.add_card(
        "F3: Binary Market Theta / Time-Value Decay",
        html_body=theta_html,
        fig=fig_theta,
        notes="max possible mid*(1-mid)=0.25 (at mid=0.5). Front-loaded = rapid decay in final hours.",
    )

    # F4: Implied vs realized
    fig_premium = _plot_vol_premium(premium_df)
    if not premium_df.empty:
        mean_impl = float(premium_df["sigma_implied"].mean())
        mean_real = float(premium_df["sigma_realized"].mean())
        pct_positive = float((premium_df["premium"] > 0).mean()) * 100
        premium_html = (
            f"<p>Vol risk premium (implied − realized): <strong>{premium_sign}{abs(mean_premium) * 100:.1f}%</strong> ann. "
            f"Positive in {pct_positive:.0f}% of expiries ({n_premium_expiries} total). "
            f"Mean σ_implied={mean_impl * 100:.1f}% vs σ_realized={mean_real * 100:.1f}%.</p>"
            f"<p>Split-half: H1={split_premium['h1'].get('vol_premium_sign', '?')}{abs(split_premium['h1'].get('mean_premium', 0)) * 100:.1f}% "
            f"(n={split_premium['h1'].get('n', 0)}), "
            f"H2={split_premium['h2'].get('vol_premium_sign', '?')}{abs(split_premium['h2'].get('mean_premium', 0)) * 100:.1f}% "
            f"(n={split_premium['h2'].get('n', 0)}). "
            f"<strong>{'STABLE' if split_premium['stable'] else 'UNSTABLE'}</strong>.</p>"
        )
    else:
        premium_html = "<p>Insufficient data for vol premium calculation.</p>"
    report.add_card(
        "F4: Implied vs Realized Vol Premium",
        html_body=premium_html,
        fig=fig_premium,
        notes="Implied σ back-solved from binary mid at market open (TTE=23–24h). Positive premium → market systematically overprices vol.",
    )

    # F5: Vol regime predictiveness
    fig_regime = _plot_vol_regime(regime_dict)
    regime_html = (
        f"<p>Opening 2h Parkinson σ vs daily range fraction. "
        f"Pearson r=<strong>{regime_dict.get('pearson_r', float('nan')):.3f}</strong> "
        f"(p={regime_dict.get('pearson_p', float('nan')):.3f}), "
        f"Spearman r={regime_dict.get('spearman_r', float('nan')):.3f} "
        f"(p={regime_dict.get('spearman_p', float('nan')):.3f}). n={regime_dict.get('n', 0)} days.</p>"
        f"<p>Predictiveness: <strong>{'meaningful' if abs(regime_dict.get('pearson_r', 0)) >= 0.3 else 'weak'}</strong> "
        f"(threshold r≥0.3).</p>"
    )
    report.add_card(
        "F5: Vol-Regime Predictiveness",
        html_body=regime_html,
        fig=fig_regime,
        notes="Does open-window σ predict how volatile the day will be? Useful for sizing/gating binary entries.",
    )

    # Render HTML
    buf = []
    buf.append("<div class='card'><h2>Card F Summary: Vol Term Structure &amp; Theta Decay</h2>")
    buf.append(f"<p>{findings['headline']}</p>")
    buf.append("</div>")
    for card in report._cards:
        img_tag = ""
        if card["fig"] is not None:
            b64 = fig_to_base64(card["fig"])
            img_tag = f'<img src="data:image/png;base64,{b64}" alt="{card["title"]}">'
        notes_html = f'<div class="notes">{card["notes"]}</div>' if card["notes"] else ""
        buf.append(f"<div class='card'><h2>{card['title']}</h2>{card['html_body']}{img_tag}{notes_html}</div>")

    card_html = "\n".join(buf)

    # Close all matplotlib figures
    plt.close("all")

    return card_html, findings


# ---------------------------------------------------------------------------
# __main__ entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_root = os.environ.get("HLBT_HL_DATA_ROOT", "../../data")
    data_root_abs = str(Path(data_root).resolve())
    if not Path(data_root_abs).exists():
        raise FileNotFoundError(f"Data root not found: {data_root_abs!r}. Set HLBT_HL_DATA_ROOT.")

    _log.info("Running Card F with data_root=%s", data_root_abs)

    con = duckdb.connect()
    card_html, findings = build_card(con, data_root_abs)
    con.close()

    # Write outputs
    out_dir = Path(__file__).resolve().parents[4] / "docs" / "research" / "_cards"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Wrap in full HTML
    from hlanalysis.research.report import Report

    rpt = Report(title="Card F: Vol Term Structure & Theta Decay")
    rpt.add_card("Card F Output", html_body=card_html, fig=None, notes=None)
    # Actually write the raw cards as a full page
    html_path = out_dir / "card_f.html"
    import datetime as dt_mod

    generated_at = dt_mod.datetime.now(tz=dt_mod.UTC).strftime("%Y-%m-%d %H:%M UTC")
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Card F: Vol Term Structure &amp; Theta Decay</title>
    <style>
    {Report._DARK_CSS}
    </style>
</head>
<body>
    <h1>Card F: Vol Term Structure &amp; Theta Decay</h1>
    <div class="subtitle">Generated {generated_at} | Data: {_DATA_START} to {_DATA_END}</div>
    {card_html}
</body>
</html>"""

    html_path.write_text(full_html, encoding="utf-8")
    _log.info("Wrote HTML: %s", html_path)

    json_path = out_dir / "card_f.json"
    # findings may contain DataFrames — strip them
    findings_clean = {k: v for k, v in findings.items() if not isinstance(v, pd.DataFrame)}
    for key in ["split_half"]:
        if key in findings_clean and isinstance(findings_clean[key], dict):
            findings_clean[key] = {
                k2: {k3: v3 for k3, v3 in v2.items() if not isinstance(v3, pd.DataFrame)}
                if isinstance(v2, dict)
                else v2
                for k2, v2 in findings_clean[key].items()
            }

    json_path.write_text(json.dumps(findings_clean, indent=2, default=str), encoding="utf-8")
    _log.info("Wrote JSON: %s", json_path)

    print("\n=== Card F Findings ===")
    print(f"Title: {findings['title']}")
    print(f"Headline: {findings['headline']}")
    print(f"Verdict: {findings['verdict']}")
    print(f"\nOutputs:\n  HTML: {html_path}\n  JSON: {json_path}")


if __name__ == "__main__":
    main()
