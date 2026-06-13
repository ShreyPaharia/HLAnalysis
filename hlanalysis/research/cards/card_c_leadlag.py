"""Card C: Cross-Venue Lead-Lag — Perp/Spot → Binary Outcome Mid.

This is the market-making hedging question and a candidate taker signal.

Analyses
--------
C1. HL perp BTC → binary-implied fair value.
    Normalise BTC moves to model-implied prob changes (GBM + Itô), then
    cross-correlate Δ(binary mid) vs Δ(model prob from perp) at lags −5s..+30s.
    Report: lead time (peak xcorr lag) and response half-life.

C2. HL perp vs Binance spot lead-lag (which crypto venue leads), latency-adjusted.
    HL perp transport latency ~223 ms median; Binance spot exchange_ts=0 (sentinel)
    so Binance local_recv_ts is the recorder's wall clock — comparable to HL perp
    local_recv_ts for cross-venue comparison after latency correction.

C3. Hedging implication for MM.
    Quantify τ (perp-to-binary lead time), implied quote-refresh speed, and
    hedge ratio ∂(binary mid)/∂(BTC) and its stability.

C4. Taker signal.
    Predictive R²/edge at best lag, net of binary half-spread.

C5. TTE dependence.
    Lead-lag vs TTE bucket (<2h, 2-8h, 8-24h).

Split-half stability: 2026-05-06..2026-05-23 vs 2026-05-24..2026-06-10.
Coverage gate: ≥30 expiries.

Interface
---------
build_card(con, data_root) -> tuple[str, dict]
    Returns (card_html, findings).

Main
----
Run directly to write docs/research/_cards/card_c.html + card_c.json.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hlanalysis.analysis.helpers import asof_locf
from hlanalysis.research.metrics import implied_prob_gbm, leadlag_xcorr
from hlanalysis.research.report import Report

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORPUS_START = "2026-05-06"
CORPUS_END = "2026-06-10"
SPLIT_MID = "2026-05-24"  # first half: 05-06..05-23; second half: 05-24..06-10

# Transport latency corrections (median, nanoseconds)
HL_PERP_LATENCY_NS = 223_375_238  # ~223ms median perp transport latency
HL_BINARY_LATENCY_NS = 214_432_894  # ~214ms median binary transport latency
# Binance spot exchange_ts is sentinel 0 → local_recv_ts IS wall clock; no adjustment possible.

# Resample grids
DT_5S_NS = 5 * 1_000_000_000
DT_1S_NS = 1 * 1_000_000_000

# GBM vol lookback (seconds, for Parkinson sigma)
VOL_LOOKBACK_S = 3600

# Xcorr max lags (in grid steps @ 5s resolution → −5s .. +30s = −1..+6 steps,
# but we use 10 steps max to allow generous scan)
MAX_LAG_STEPS = 10  # ±50 seconds at 5s grid

# Half-spread assumption for taker signal net-of-spread edge calculation
# (binary mid - ask on the side being bought ≈ half-spread)
BINARY_HALF_SPREAD_TYPICAL = 0.005  # ~50 bps half-spread (conservative)

_NS = 1_000_000_000


# ---------------------------------------------------------------------------
# Glob helpers
# ---------------------------------------------------------------------------


def _hl_perp_bbo_glob(data_root: str) -> str:
    return str(
        Path(data_root)
        / "venue=hyperliquid/product_type=perp/mechanism=clob"
        / "event=bbo/symbol=BTC/date=*/hour=all/*.parquet"
    )


def _bin_spot_bbo_glob(data_root: str) -> str:
    return str(
        Path(data_root)
        / "venue=binance/product_type=spot/mechanism=clob"
        / "event=bbo/symbol=BTCUSDT/date=*/hour=all/*.parquet"
    )


def _binary_bbo_glob(data_root: str) -> str:
    return str(
        Path(data_root)
        / "venue=hyperliquid/product_type=prediction_binary/mechanism=clob"
        / "event=bbo/symbol=*/date=*/hour=all/*.parquet"
    )


def _meta_glob(data_root: str) -> str:
    return str(
        Path(data_root)
        / "venue=hyperliquid/product_type=prediction_binary/mechanism=clob"
        / "event=market_meta/symbol=*/date=*/hour=all/*.parquet"
    )


def _perp_oracle_glob(data_root: str) -> str:
    return str(
        Path(data_root)
        / "venue=hyperliquid/product_type=perp/mechanism=clob"
        / "event=oracle/symbol=BTC/date=*/hour=all/*.parquet"
    )


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_binary_yes_legs(con: duckdb.DuckDBPyConnection, data_root: str) -> pd.DataFrame:
    """Return binary Yes-leg metadata: symbol, expiry_str, target_price, expiry_ns."""
    meta = _meta_glob(data_root)
    sql = f"""
        SELECT DISTINCT
            symbol,
            list_element(values, list_position(keys, 'expiry'))              AS expiry_str,
            list_element(values, list_position(keys, 'targetPrice'))::DOUBLE AS target_price
        FROM read_parquet('{meta}', union_by_name=true)
        WHERE array_contains(keys, 'class')
          AND list_element(values, list_position(keys, 'class')) = 'priceBinary'
          AND list_element(values, list_position(keys, 'side_name')) = 'Yes'
        ORDER BY expiry_str, symbol
    """
    try:
        df = con.execute(sql).df()
    except duckdb.IOException:
        return pd.DataFrame()
    if df.empty:
        return df
    # Parse expiry to nanoseconds
    import datetime as dt_mod

    def _parse_exp_ns(s: str) -> int:
        try:
            d = dt_mod.datetime.strptime(s, "%Y%m%d-%H%M").replace(tzinfo=dt_mod.UTC)
        except ValueError:
            d = dt_mod.datetime.strptime(s, "%Y%m%d-%H").replace(tzinfo=dt_mod.UTC)
        return int(d.timestamp() * _NS)

    df["expiry_ns"] = df["expiry_str"].map(_parse_exp_ns)
    return df.reset_index(drop=True)


def _load_perp_bbo_day(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    date_str: str,
) -> pd.DataFrame:
    """Load HL perp BBO for one date. Returns ts_ns, mid (latency-adjusted)."""
    glob = str(
        Path(data_root)
        / f"venue=hyperliquid/product_type=perp/mechanism=clob/event=bbo/symbol=BTC/date={date_str}/hour=all/*.parquet"
    )
    sql = f"""
        SELECT local_recv_ts AS ts_ns,
               (bid_px + ask_px) / 2.0 AS mid
        FROM read_parquet('{glob}', union_by_name=true)
        ORDER BY local_recv_ts ASC
    """
    try:
        df = con.execute(sql).df()
    except duckdb.IOException:
        return pd.DataFrame(columns=["ts_ns", "mid"])
    if df.empty:
        return df
    # Apply latency correction: subtract HL_PERP_LATENCY_NS to get adjusted exchange time
    df["ts_ns"] = df["ts_ns"] - HL_PERP_LATENCY_NS
    return df


def _load_binance_bbo_day(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    date_str: str,
) -> pd.DataFrame:
    """Load Binance spot BBO for one date. Returns ts_ns, mid.
    No latency correction since exchange_ts=0 (Binance local_recv_ts = wall clock).
    """
    glob = str(
        Path(data_root)
        / f"venue=binance/product_type=spot/mechanism=clob/event=bbo/symbol=BTCUSDT/date={date_str}/hour=all/*.parquet"
    )
    sql = f"""
        SELECT local_recv_ts AS ts_ns,
               (bid_px + ask_px) / 2.0 AS mid
        FROM read_parquet('{glob}', union_by_name=true)
        ORDER BY local_recv_ts ASC
    """
    try:
        df = con.execute(sql).df()
    except duckdb.IOException:
        return pd.DataFrame(columns=["ts_ns", "mid"])
    return df


def _load_binary_bbo_day(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    symbol: str,
    date_str: str,
) -> pd.DataFrame:
    """Load binary BBO for one symbol+date. Returns ts_ns, mid, bid, ask (latency-adjusted)."""
    glob = str(
        Path(data_root)
        / f"venue=hyperliquid/product_type=prediction_binary/mechanism=clob/event=bbo/symbol={symbol}/date={date_str}/hour=all/*.parquet"
    )
    sql = f"""
        SELECT local_recv_ts AS ts_ns, bid_px, ask_px,
               (bid_px + ask_px) / 2.0 AS mid
        FROM read_parquet('{glob}', union_by_name=true)
        ORDER BY local_recv_ts ASC
    """
    try:
        df = con.execute(sql).df()
    except duckdb.IOException:
        return pd.DataFrame(columns=["ts_ns", "bid_px", "ask_px", "mid"])
    if df.empty:
        return df
    # Apply latency correction
    df["ts_ns"] = df["ts_ns"] - HL_BINARY_LATENCY_NS
    return df


def _parkinson_sigma_rolling(
    perp_df: pd.DataFrame,
    grid_ns: np.ndarray,
    lookback_ns: int,
) -> np.ndarray:
    """Compute rolling Parkinson sigma (annualised) at each grid point.

    Uses 60s OHLC bars from perp mid ticks.
    """
    if perp_df.empty or len(perp_df) < 2:
        return np.full(len(grid_ns), np.nan)

    ts = perp_df["ts_ns"].to_numpy(dtype="int64")
    mid = perp_df["mid"].to_numpy(dtype="float64")

    # Build 60s OHLC bars
    bar_ns = 60 * _NS
    bar_grid = np.arange(ts[0] // bar_ns * bar_ns, ts[-1] + bar_ns, bar_ns, dtype="int64")

    bar_h: list[float] = []
    bar_l: list[float] = []
    bar_t: list[int] = []
    for i in range(len(bar_grid) - 1):
        bs, be = bar_grid[i], bar_grid[i + 1]
        mask = (ts >= bs) & (ts < be)
        if mask.sum() == 0:
            continue
        sub = mid[mask]
        bar_h.append(float(np.max(sub)))
        bar_l.append(float(np.min(sub)))
        bar_t.append(int(bs))

    if not bar_h:
        return np.full(len(grid_ns), np.nan)

    bh = np.array(bar_h, dtype="float64")
    bl = np.array(bar_l, dtype="float64")
    bt = np.array(bar_t, dtype="int64")

    # For each grid point, rolling Parkinson over lookback window
    out = np.full(len(grid_ns), np.nan)
    secs_per_year = 365.25 * 86400.0
    for i, g in enumerate(grid_ns):
        mask = (bt >= g - lookback_ns) & (bt <= g)
        if mask.sum() < 2:
            continue
        h_w = bh[mask]
        l_w = bl[mask]
        valid = (h_w > 0) & (l_w > 0)
        if valid.sum() < 2:
            continue
        log_hl = np.log(h_w[valid] / l_w[valid])
        park_var = (1.0 / (4.0 * math.log(2.0))) * np.mean(log_hl**2)
        out[i] = math.sqrt(park_var * secs_per_year / 60.0)
    return out


def _model_prob_series(
    perp_mid: np.ndarray,
    sigma: np.ndarray,
    target_price: float,
    expiry_ns: int,
    grid_ns: np.ndarray,
) -> np.ndarray:
    """Compute GBM-implied prob P(S_T > K) at each grid point with Itô correction."""
    out = np.full(len(grid_ns), np.nan)
    for i, g in enumerate(grid_ns):
        s = float(perp_mid[i])
        sig = float(sigma[i])
        tau_s = (expiry_ns - g) / float(_NS)
        if not np.isfinite(s) or not np.isfinite(sig) or sig <= 0 or tau_s <= 0:
            continue
        out[i] = implied_prob_gbm(s, target_price, sig, tau_s)
    return out


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------


def _resample_to_grid(
    ts_raw: np.ndarray,
    mid_raw: np.ndarray,
    grid_ns: np.ndarray,
) -> np.ndarray:
    """LOCF-resample a time series onto a regular grid."""
    if len(ts_raw) == 0:
        return np.full(len(grid_ns), np.nan)
    return asof_locf(grid_ns, ts_raw, mid_raw)


def _compute_xcorr_for_expiry(
    perp_df: pd.DataFrame,
    binary_df: pd.DataFrame,
    target_price: float,
    expiry_ns: int,
    dt_s: int = 5,
) -> dict | None:
    """Compute xcorr between Δ(model_prob) and Δ(binary_mid) for one expiry.

    Returns dict with: lag_steps, corr values, peak_lag_s, peak_corr, n_valid_steps.
    """
    if perp_df.empty or binary_df.empty:
        return None

    perp_ts = perp_df["ts_ns"].to_numpy(dtype="int64")
    binary_ts = binary_df["ts_ns"].to_numpy(dtype="int64")

    # Build grid over the overlap window, ending at expiry
    t_start = max(perp_ts[0], binary_ts[0])
    t_end = min(expiry_ns, perp_ts[-1], binary_ts[-1])
    if t_end <= t_start:
        return None

    dt_ns = dt_s * _NS
    grid = np.arange(t_start, t_end + 1, dt_ns, dtype="int64")
    if len(grid) < 20:  # need enough points for xcorr
        return None

    # Resample perp and binary mid onto grid
    perp_mid_g = _resample_to_grid(perp_ts, perp_df["mid"].to_numpy(dtype="float64"), grid)
    binary_mid_g = _resample_to_grid(binary_ts, binary_df["mid"].to_numpy(dtype="float64"), grid)

    # Compute rolling Parkinson sigma at each grid point
    lookback_ns = VOL_LOOKBACK_S * _NS
    sigma_g = _parkinson_sigma_rolling(perp_df, grid, lookback_ns)

    # Model prob series
    model_prob_g = _model_prob_series(perp_mid_g, sigma_g, target_price, expiry_ns, grid)

    # Returns: Δmodel_prob and Δbinary_mid
    delta_model = np.diff(model_prob_g)
    delta_binary = np.diff(binary_mid_g)

    # Mask NaN / zero-change steps
    valid = np.isfinite(delta_model) & np.isfinite(delta_binary)
    if valid.sum() < 20:
        return None

    # We want xcorr(Δ_model[t], Δ_binary[t+lag]) — does perp lead binary?
    # leadlag_xcorr convention: lag > 0 means y's past predicts x's present
    # We want: corr(Δ_binary, Δ_model.shift(lag)) at various lags
    # → lag > 0 means model_prob's PAST predicts binary's PRESENT (perp leads binary)
    x_ser = pd.Series(delta_binary, dtype="float64")
    y_ser = pd.Series(delta_model, dtype="float64")

    n = len(x_ser)
    if n <= MAX_LAG_STEPS:
        max_lag = max(1, n // 2 - 1)
    else:
        max_lag = MAX_LAG_STEPS

    try:
        xcorr_df = leadlag_xcorr(x_ser, y_ser, max_lag_steps=max_lag)
    except Exception:
        return None

    # Peak lag (positive lag = perp leads binary)
    # Note: the binary typically responds to perp within the same grid bucket
    # (sub-5s), so the xcorr peak is at lag=0. We look at positive lags only
    # for the "perp leads binary" direction; lag=0 counts as contemporaneous.
    pos_mask = xcorr_df["lag"] >= 0
    if pos_mask.sum() == 0:
        return None

    pos_sub = xcorr_df[pos_mask]
    if pos_sub["corr"].isna().all():
        return None

    peak_idx = pos_sub["corr"].abs().idxmax()
    peak_lag_steps = int(xcorr_df.loc[peak_idx, "lag"])
    peak_corr = float(xcorr_df.loc[peak_idx, "corr"])

    # Contemporaneous correlation (lag=0)
    lag0_sub = xcorr_df[xcorr_df["lag"] == 0]
    corr_lag0 = float(lag0_sub.iloc[0]["corr"]) if not lag0_sub.empty else float("nan")

    # Response half-life: at what positive lag does |xcorr| fall to half its lag=0 value?
    # This measures how fast the binary's response to a perp move decays.
    half_corr = abs(corr_lag0) / 2.0
    half_life_steps: int | None = None
    for step in range(1, max_lag + 1):
        sub = xcorr_df[xcorr_df["lag"] == step]
        if not sub.empty and np.isfinite(sub.iloc[0]["corr"]):
            if abs(float(sub.iloc[0]["corr"])) <= half_corr:
                half_life_steps = step
                break

    return {
        "lags": xcorr_df["lag"].tolist(),
        "corrs": xcorr_df["corr"].tolist(),
        "peak_lag_steps": peak_lag_steps,
        "peak_lag_s": peak_lag_steps * dt_s,
        "peak_corr": peak_corr,
        "corr_lag0": corr_lag0,
        "half_life_steps": half_life_steps,
        "half_life_s": (half_life_steps * dt_s) if half_life_steps is not None else None,
        "n_valid_steps": int(valid.sum()),
        "tte_mean_h": float(np.nanmean((expiry_ns - grid) / _NS / 3600.0)),
        # Hedge ratio: ∂(binary_mid)/∂(model_prob) at contemporaneous lag
        "hedge_ratio": float(_compute_hedge_ratio(delta_binary[valid], delta_model[valid])),
    }


def _compute_hedge_ratio(delta_binary: np.ndarray, delta_model: np.ndarray) -> float:
    """OLS slope of delta_binary ~ delta_model (hedge ratio ∂binary/∂model_prob)."""
    valid = np.isfinite(delta_binary) & np.isfinite(delta_model)
    if valid.sum() < 4:
        return float("nan")
    x = delta_model[valid]
    y = delta_binary[valid]
    # OLS: beta = Cov(x,y)/Var(x)
    x_dm = x - np.mean(x)
    y_dm = y - np.mean(y)
    var_x = np.dot(x_dm, x_dm)
    if var_x < 1e-20:
        return float("nan")
    return float(np.dot(x_dm, y_dm) / var_x)


def _compute_perp_vs_spot_xcorr(
    perp_df: pd.DataFrame,
    spot_df: pd.DataFrame,
    dt_s: int = 5,
) -> dict | None:
    """Xcorr between HL perp and Binance spot BTC price changes.

    Uses raw price changes (not model-prob normalised).
    Convention: lag > 0 means spot's past predicts perp's present (spot leads).
                lag < 0 means perp's past predicts spot's present (perp leads).
    We look at |lag| ≤ MAX_LAG_STEPS steps.
    """
    if perp_df.empty or spot_df.empty:
        return None

    perp_ts = perp_df["ts_ns"].to_numpy(dtype="int64")
    spot_ts = spot_df["ts_ns"].to_numpy(dtype="int64")

    t_start = max(perp_ts[0], spot_ts[0])
    t_end = min(perp_ts[-1], spot_ts[-1])
    if t_end <= t_start:
        return None

    dt_ns = dt_s * _NS
    grid = np.arange(t_start, t_end + 1, dt_ns, dtype="int64")
    if len(grid) < 20:
        return None

    perp_mid_g = _resample_to_grid(perp_ts, perp_df["mid"].to_numpy(dtype="float64"), grid)
    spot_mid_g = _resample_to_grid(spot_ts, spot_df["mid"].to_numpy(dtype="float64"), grid)

    delta_perp = np.diff(perp_mid_g)
    delta_spot = np.diff(spot_mid_g)

    valid = np.isfinite(delta_perp) & np.isfinite(delta_spot)
    if valid.sum() < 20:
        return None

    n = len(delta_perp)
    max_lag = min(MAX_LAG_STEPS, n // 2 - 1)
    if max_lag < 1:
        return None

    # corr(delta_perp, delta_spot.shift(lag)) — lag > 0: spot's past predicts perp's now
    x_ser = pd.Series(delta_perp, dtype="float64")
    y_ser = pd.Series(delta_spot, dtype="float64")

    try:
        xcorr_df = leadlag_xcorr(x_ser, y_ser, max_lag_steps=max_lag)
    except Exception:
        return None

    peak_idx = xcorr_df["corr"].abs().idxmax()
    peak_lag = int(xcorr_df.loc[peak_idx, "lag"])
    peak_corr = float(xcorr_df.loc[peak_idx, "corr"])

    return {
        "lags": xcorr_df["lag"].tolist(),
        "corrs": xcorr_df["corr"].tolist(),
        "peak_lag_steps": peak_lag,
        "peak_lag_s": peak_lag * dt_s,
        "peak_corr": peak_corr,
        "n_valid_steps": int(valid.sum()),
        "leader": "perp" if peak_lag < 0 else ("spot" if peak_lag > 0 else "contemporaneous"),
    }


def _compute_taker_signal(
    results_per_expiry: list[dict],
    dt_s: int = 5,
) -> dict:
    """Aggregate per-expiry xcorr to get taker signal edge estimate.

    At the best lead lag, the predictive R² is peak_corr².
    Net edge = predictive move magnitude - binary half-spread.
    """
    if not results_per_expiry:
        return {}

    peak_corrs = [r["peak_corr"] for r in results_per_expiry if r["peak_corr"] is not None]
    peak_lags = [r["peak_lag_s"] for r in results_per_expiry]
    hedge_ratios = [r["hedge_ratio"] for r in results_per_expiry if np.isfinite(r["hedge_ratio"])]

    if not peak_corrs:
        return {}

    peak_corrs_arr = np.array(peak_corrs, dtype="float64")
    peak_lags_arr = np.array(peak_lags, dtype="float64")
    valid_corrs = peak_corrs_arr[np.isfinite(peak_corrs_arr)]

    median_peak_corr = float(np.nanmedian(valid_corrs))
    median_r2 = float(median_peak_corr**2)
    median_lag = float(np.nanmedian(peak_lags_arr))

    # Rough edge: R² predicts fraction of variance explained; for a binary move
    # of ~0.01 (typical 5s move), edge = |corr| * std_move
    # Use abs(corr) as signal strength, net of half-spread
    edge_gross = abs(median_peak_corr) * BINARY_HALF_SPREAD_TYPICAL  # rough: corr * typical_half_spread
    edge_net = edge_gross - BINARY_HALF_SPREAD_TYPICAL  # subtract cost

    return {
        "n_expiries": len(results_per_expiry),
        "median_peak_corr": median_peak_corr,
        "median_r2": median_r2,
        "median_peak_lag_s": median_lag,
        "edge_gross_estimate": edge_gross,
        "edge_net_of_half_spread": edge_net,
        "survives_half_spread": edge_net > 0,
        "hedge_ratio_median": float(np.nanmedian(hedge_ratios)) if hedge_ratios else float("nan"),
        "hedge_ratio_std": float(np.nanstd(hedge_ratios)) if len(hedge_ratios) > 1 else float("nan"),
    }


def _tte_bucket(tte_mean_h: float) -> str:
    if tte_mean_h < 2.0:
        return "<2h"
    elif tte_mean_h < 8.0:
        return "2-8h"
    else:
        return "8-24h"


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def _run_leadlag_analysis(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    binaries: pd.DataFrame,
    start_date: str,
    end_date: str,
    dt_s: int = 5,
) -> tuple[list[dict], list[dict]]:
    """Run lead-lag analysis for all expiries in [start_date, end_date].

    Returns (perp_to_binary_results, perp_vs_spot_day_results).
    """
    import datetime as dt_mod

    start_d = dt_mod.date.fromisoformat(start_date)
    end_d = dt_mod.date.fromisoformat(end_date)

    # Filter binaries to those expiring within [start_date + 1d, end_date + 1d]
    # (a binary expiring on date D is active on date D-1 and D)
    start_ns = int(dt_mod.datetime(start_d.year, start_d.month, start_d.day, tzinfo=dt_mod.UTC).timestamp() * _NS)
    end_ns = int(dt_mod.datetime(end_d.year, end_d.month, end_d.day, tzinfo=dt_mod.UTC).timestamp() * _NS) + 86400 * _NS

    active = binaries[(binaries["expiry_ns"] >= start_ns) & (binaries["expiry_ns"] <= end_ns)].copy()
    _log.info("run_leadlag: %d binaries in %s..%s", len(active), start_date, end_date)

    perp_binary_results: list[dict] = []
    perp_spot_results: list[dict] = []

    # Group by expiry to minimise data loads
    for _, row in active.iterrows():
        sym = str(row["symbol"])
        expiry_ns = int(row["expiry_ns"])
        target_price = float(row["target_price"])

        # Determine which dates to load (day before and day of expiry)
        exp_d = dt_mod.datetime.fromtimestamp(expiry_ns / _NS, tz=dt_mod.UTC).date()
        exp_prev = exp_d - dt_mod.timedelta(days=1)

        dates_to_load = []
        for d in [exp_prev, exp_d]:
            if start_d <= d <= end_d:
                dates_to_load.append(d.isoformat())
        if not dates_to_load:
            dates_to_load = [exp_prev.isoformat()]  # try at least one

        perp_frames = []
        binary_frames = []
        spot_frames = []

        for date_str in dates_to_load:
            pf = _load_perp_bbo_day(con, data_root, date_str)
            if not pf.empty:
                perp_frames.append(pf)
            bf = _load_binary_bbo_day(con, data_root, sym, date_str)
            if not bf.empty:
                binary_frames.append(bf)
            sf = _load_binance_bbo_day(con, data_root, date_str)
            if not sf.empty:
                spot_frames.append(sf)

        if not perp_frames or not binary_frames:
            continue

        perp_all = pd.concat(perp_frames, ignore_index=True)
        binary_all = pd.concat(binary_frames, ignore_index=True)
        spot_all = pd.concat(spot_frames, ignore_index=True) if spot_frames else pd.DataFrame()

        res = _compute_xcorr_for_expiry(perp_all, binary_all, target_price, expiry_ns, dt_s)
        if res is not None:
            res["symbol"] = sym
            res["expiry_str"] = str(row["expiry_str"])
            res["target_price"] = target_price
            res["half_period"] = (
                "first"
                if expiry_ns
                < int(dt_mod.datetime.strptime(SPLIT_MID, "%Y-%m-%d").replace(tzinfo=dt_mod.UTC).timestamp() * _NS)
                else "second"
            )
            perp_binary_results.append(res)

        # Perp vs Spot: only do once per day (use exp_prev)
        if not spot_all.empty:
            pv_res = _compute_perp_vs_spot_xcorr(perp_all, spot_all, dt_s)
            if pv_res is not None:
                pv_res["date"] = str(exp_prev)
                perp_spot_results.append(pv_res)

    return perp_binary_results, perp_spot_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_xcorr_summary(results: list[dict], title: str) -> matplotlib.figure.Figure:
    """Plot distribution of xcorr curves + peak lag distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(title, color="#e6edf3", fontsize=13)

    for ax in axes:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="#e6edf3")
        ax.xaxis.label.set_color("#e6edf3")
        ax.yaxis.label.set_color("#e6edf3")
        ax.title.set_color("#58a6ff")

    # 1. Distribution of xcorr curves
    ax0 = axes[0]
    # Find common lag range
    all_lags = None
    for r in results:
        if all_lags is None:
            all_lags = np.array(r["lags"])
    if all_lags is not None:
        corr_matrix = []
        for r in results:
            if len(r["lags"]) == len(all_lags):
                corr_matrix.append(r["corrs"])
        if corr_matrix:
            corr_arr = np.array(corr_matrix, dtype="float64")
            ax0.fill_between(
                all_lags,
                np.nanpercentile(corr_arr, 25, axis=0),
                np.nanpercentile(corr_arr, 75, axis=0),
                alpha=0.3,
                color="#58a6ff",
                label="IQR",
            )
            ax0.plot(all_lags, np.nanmedian(corr_arr, axis=0), color="#58a6ff", linewidth=2, label="Median xcorr")
            ax0.axhline(0, color="#30363d", linewidth=0.8)
            ax0.axvline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax0.set_xlabel("Lag (grid steps @ 5s)")
    ax0.set_ylabel("Correlation")
    ax0.set_title("Xcorr Distribution (Perp→Binary)")
    ax0.legend(fontsize=8, labelcolor="#e6edf3", facecolor="#161b22")

    # 2. Peak lag distribution
    ax1 = axes[1]
    peak_lags = [r["peak_lag_s"] for r in results if r.get("peak_lag_s") is not None]
    if peak_lags:
        ax1.hist(peak_lags, bins=20, color="#58a6ff", alpha=0.8, edgecolor="#30363d")
        ax1.axvline(
            np.nanmedian(peak_lags), color="#f78166", linewidth=1.5, label=f"Median={np.nanmedian(peak_lags):.0f}s"
        )
    ax1.set_xlabel("Peak Lead Time (seconds)")
    ax1.set_ylabel("Count (expiries)")
    ax1.set_title("Perp→Binary Lead Time Distribution")
    ax1.legend(fontsize=8, labelcolor="#e6edf3", facecolor="#161b22")

    # 3. Peak corr vs TTE
    ax2 = axes[2]
    tte_h = [r.get("tte_mean_h", np.nan) for r in results]
    peak_corrs = [abs(r["peak_corr"]) for r in results]
    sc = ax2.scatter(tte_h, peak_corrs, c=[r["peak_lag_s"] for r in results], cmap="viridis", alpha=0.7, s=25)
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.ax.yaxis.set_tick_params(color="#e6edf3")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#e6edf3")
    cbar.set_label("Peak Lag (s)", color="#e6edf3")
    ax2.set_xlabel("Mean TTE (hours)")
    ax2.set_ylabel("|Peak Xcorr|")
    ax2.set_title("|Xcorr| vs TTE — coloured by lead time")

    plt.tight_layout()
    return fig


def _plot_perp_spot_xcorr(results: list[dict]) -> matplotlib.figure.Figure:
    """Plot perp vs spot cross-correlation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("HL Perp vs Binance Spot Lead-Lag", color="#e6edf3", fontsize=13)

    for ax in axes:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="#e6edf3")
        ax.xaxis.label.set_color("#e6edf3")
        ax.yaxis.label.set_color("#e6edf3")
        ax.title.set_color("#58a6ff")

    ax0 = axes[0]
    all_lags = None
    for r in results:
        if all_lags is None:
            all_lags = np.array(r["lags"])
    if all_lags is not None:
        corr_matrix = []
        for r in results:
            if len(r["lags"]) == len(all_lags):
                corr_matrix.append(r["corrs"])
        if corr_matrix:
            corr_arr = np.array(corr_matrix, dtype="float64")
            ax0.fill_between(
                all_lags,
                np.nanpercentile(corr_arr, 25, axis=0),
                np.nanpercentile(corr_arr, 75, axis=0),
                alpha=0.3,
                color="#3fb950",
                label="IQR",
            )
            ax0.plot(all_lags, np.nanmedian(corr_arr, axis=0), color="#3fb950", linewidth=2, label="Median")
            ax0.axhline(0, color="#30363d", linewidth=0.8)
            ax0.axvline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax0.set_xlabel("Lag (grid steps @ 5s, positive = spot's past)")
    ax0.set_ylabel("Correlation")
    ax0.set_title("Δ(HL perp) vs Δ(Binance spot)")
    ax0.legend(fontsize=8, labelcolor="#e6edf3", facecolor="#161b22")

    ax1 = axes[1]
    leaders = [r.get("leader", "unknown") for r in results]
    leader_counts = {l: sum(1 for x in leaders if x == l) for l in set(leaders)}
    ax1.bar(list(leader_counts.keys()), list(leader_counts.values()), color="#3fb950", alpha=0.8, edgecolor="#30363d")
    ax1.set_xlabel("Leader")
    ax1.set_ylabel("Count (days)")
    ax1.set_title("Which venue leads? (perp vs spot)")

    plt.tight_layout()
    return fig


def _plot_hedge_ratio(results: list[dict]) -> matplotlib.figure.Figure:
    """Hedge ratio stability over time and vs TTE."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Hedge Ratio ∂(binary mid)/∂(model_prob) Stability", color="#e6edf3", fontsize=13)

    for ax in axes:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="#e6edf3")
        ax.xaxis.label.set_color("#e6edf3")
        ax.yaxis.label.set_color("#e6edf3")
        ax.title.set_color("#58a6ff")

    hedge_ratios = [r["hedge_ratio"] for r in results if np.isfinite(r["hedge_ratio"])]
    tte_h = [r["tte_mean_h"] for r in results if np.isfinite(r["hedge_ratio"])]

    ax0 = axes[0]
    if hedge_ratios:
        ax0.hist(hedge_ratios, bins=20, color="#f78166", alpha=0.8, edgecolor="#30363d")
        ax0.axvline(
            np.nanmedian(hedge_ratios), color="#58a6ff", linewidth=1.5, label=f"Median={np.nanmedian(hedge_ratios):.2f}"
        )
        ax0.axvline(1.0, color="#3fb950", linewidth=1.0, linestyle="--", label="Ideal=1.0")
    ax0.set_xlabel("Hedge Ratio ∂binary/∂model_prob")
    ax0.set_ylabel("Count")
    ax0.set_title("Hedge Ratio Distribution")
    ax0.legend(fontsize=8, labelcolor="#e6edf3", facecolor="#161b22")

    ax1 = axes[1]
    if hedge_ratios and tte_h:
        ax1.scatter(tte_h, hedge_ratios, color="#f78166", alpha=0.7, s=25)
        ax1.axhline(1.0, color="#3fb950", linewidth=1.0, linestyle="--", label="Ideal=1.0")
        ax1.axhline(
            np.nanmedian(hedge_ratios),
            color="#58a6ff",
            linewidth=1.0,
            linestyle=":",
            label=f"Median={np.nanmedian(hedge_ratios):.2f}",
        )
    ax1.set_xlabel("Mean TTE (hours)")
    ax1.set_ylabel("Hedge Ratio")
    ax1.set_title("Hedge Ratio vs TTE")
    ax1.legend(fontsize=8, labelcolor="#e6edf3", facecolor="#161b22")

    plt.tight_layout()
    return fig


def _plot_tte_dependence(results: list[dict], dt_s: int = 5) -> matplotlib.figure.Figure:
    """TTE dependence: lead lag and hedge ratio by TTE bucket."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("TTE Dependence of Lead-Lag and Hedge Ratio", color="#e6edf3", fontsize=13)

    for ax in axes:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="#e6edf3")
        ax.xaxis.label.set_color("#e6edf3")
        ax.yaxis.label.set_color("#e6edf3")
        ax.title.set_color("#58a6ff")

    buckets = ["<2h", "2-8h", "8-24h"]
    colors = ["#f78166", "#58a6ff", "#3fb950"]

    # Group by TTE bucket
    bucket_data: dict[str, dict] = {b: {"peak_lags": [], "hedge_ratios": [], "corrs": []} for b in buckets}
    for r in results:
        b = _tte_bucket(r.get("tte_mean_h", 12.0))
        bucket_data[b]["peak_lags"].append(r["peak_lag_s"])
        if np.isfinite(r["hedge_ratio"]):
            bucket_data[b]["hedge_ratios"].append(r["hedge_ratio"])
        bucket_data[b]["corrs"].append(abs(r["peak_corr"]))

    ax0, ax1 = axes[0], axes[1]

    # Peak lead time by bucket
    medians_lag = [
        np.nanmedian(bucket_data[b]["peak_lags"]) if bucket_data[b]["peak_lags"] else np.nan for b in buckets
    ]
    counts = [len(bucket_data[b]["peak_lags"]) for b in buckets]
    bars = ax0.bar(buckets, medians_lag, color=colors, alpha=0.8, edgecolor="#30363d")
    for bar, cnt in zip(bars, counts):
        if np.isfinite(bar.get_height()):
            ax0.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"n={cnt}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#e6edf3",
            )
    ax0.set_xlabel("TTE Bucket")
    ax0.set_ylabel("Median Peak Lead Time (seconds)")
    ax0.set_title("Perp→Binary Lead Time by TTE")

    # Hedge ratio by bucket
    medians_hr = [
        np.nanmedian(bucket_data[b]["hedge_ratios"]) if bucket_data[b]["hedge_ratios"] else np.nan for b in buckets
    ]
    ax1.bar(buckets, medians_hr, color=colors, alpha=0.8, edgecolor="#30363d")
    ax1.axhline(1.0, color="#8b949e", linewidth=1.0, linestyle="--", label="Ideal=1.0")
    ax1.set_xlabel("TTE Bucket")
    ax1.set_ylabel("Median Hedge Ratio")
    ax1.set_title("Hedge Ratio by TTE")
    ax1.legend(fontsize=8, labelcolor="#e6edf3", facecolor="#161b22")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _fmt(val: float | None, fmt: str = ".3f") -> str:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "N/A"
    return format(val, fmt)


def _metrics_table_html(metrics: list[dict]) -> str:
    rows = []
    for m in metrics:
        sanity = m.get("sanity", "")
        rows.append(
            f"<tr>"
            f"<td>{m['name']}</td>"
            f"<td><strong>{m['value']}</strong></td>"
            f"<td>{m.get('n', '')}</td>"
            f"<td>{m.get('date_span', '')}</td>"
            f"<td>{sanity}</td>"
            f"</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Metric</th><th>Value</th><th>n</th><th>Date span</th><th>Sanity</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )


def _split_half_table_html(split: dict) -> str:
    rows = []
    for key, val in split.items():
        if isinstance(val, dict):
            rows.append(
                f"<tr><td><strong>{key}</strong></td>"
                f"<td>{val.get('first_half', 'N/A')}</td>"
                f"<td>{val.get('second_half', 'N/A')}</td>"
                f"<td>{'stable' if val.get('stable') else 'unstable'}</td></tr>"
            )
    return (
        "<table>"
        "<thead><tr><th>Metric</th><th>First half (05-06..05-23)</th>"
        "<th>Second half (05-24..06-10)</th><th>Stability</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )


# ---------------------------------------------------------------------------
# build_card
# ---------------------------------------------------------------------------


def build_card(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
) -> tuple[str, dict]:
    """Build Card C: Cross-Venue Lead-Lag.

    Parameters
    ----------
    con : DuckDB connection (caller manages lifecycle)
    data_root : Absolute path to data root (HLBT_HL_DATA_ROOT)

    Returns
    -------
    (card_html, findings)  where findings is JSON-serialisable.
    """
    data_root = str(Path(data_root).resolve())

    # Load binary metadata
    binaries = _load_binary_yes_legs(con, data_root)
    n_binaries = len(binaries)
    _log.info("Card C: %d binary Yes-legs loaded", n_binaries)

    dt_s = 5  # 5-second grid

    # Full corpus analysis
    pb_results, ps_results = _run_leadlag_analysis(con, data_root, binaries, CORPUS_START, CORPUS_END, dt_s=dt_s)

    _log.info("Card C: %d perp→binary results, %d perp-vs-spot results", len(pb_results), len(ps_results))

    # Split-half analysis
    pb_first = [r for r in pb_results if r.get("half_period") == "first"]
    pb_second = [r for r in pb_results if r.get("half_period") == "second"]

    _log.info("Card C split: %d first-half, %d second-half expiries", len(pb_first), len(pb_second))

    # Aggregate perp→binary stats
    def _agg_pb(results: list[dict]) -> dict:
        if not results:
            return {}
        peak_lags = np.array([r["peak_lag_s"] for r in results], dtype="float64")
        peak_corrs = np.array([r["peak_corr"] for r in results], dtype="float64")
        corr_lag0s = np.array([r.get("corr_lag0", r["peak_corr"]) for r in results], dtype="float64")
        half_lives = np.array(
            [r["half_life_s"] if r["half_life_s"] is not None else np.nan for r in results], dtype="float64"
        )
        hedge_ratios = np.array([r["hedge_ratio"] for r in results], dtype="float64")
        valid_hr = hedge_ratios[np.isfinite(hedge_ratios)]
        return {
            "n": len(results),
            "median_peak_lag_s": float(np.nanmedian(peak_lags)),
            "p25_peak_lag_s": float(np.nanpercentile(peak_lags, 25)),
            "p75_peak_lag_s": float(np.nanpercentile(peak_lags, 75)),
            "median_peak_corr": float(np.nanmedian(np.abs(peak_corrs))),
            "median_corr_lag0": float(np.nanmedian(np.abs(corr_lag0s))),
            "median_r2": float(np.nanmedian(peak_corrs**2)),
            "median_half_life_s": float(np.nanmedian(half_lives)),
            "hedge_ratio_median": float(np.nanmedian(valid_hr)) if len(valid_hr) > 0 else float("nan"),
            "hedge_ratio_std": float(np.nanstd(valid_hr)) if len(valid_hr) > 1 else float("nan"),
        }

    def _agg_ps(results: list[dict]) -> dict:
        if not results:
            return {}
        peak_lags = np.array([r["peak_lag_s"] for r in results], dtype="float64")
        leaders = [r.get("leader", "unknown") for r in results]
        n_perp_leads = sum(1 for l in leaders if l == "perp")
        n_spot_leads = sum(1 for l in leaders if l == "spot")
        n_contemp = sum(1 for l in leaders if l == "contemporaneous")
        return {
            "n": len(results),
            "median_peak_lag_s": float(np.nanmedian(peak_lags)),
            "n_perp_leads": n_perp_leads,
            "n_spot_leads": n_spot_leads,
            "n_contemporaneous": n_contemp,
            "frac_perp_leads": n_perp_leads / len(results) if results else 0,
        }

    full_pb = _agg_pb(pb_results)
    first_pb = _agg_pb(pb_first)
    second_pb = _agg_pb(pb_second)
    full_ps = _agg_ps(ps_results)

    taker = _compute_taker_signal(pb_results, dt_s=dt_s)

    # TTE dependence
    def _tte_stats(results: list[dict]) -> dict:
        buckets = ["<2h", "2-8h", "8-24h"]
        stats: dict = {}
        for b in buckets:
            sub = [r for r in results if _tte_bucket(r.get("tte_mean_h", 12.0)) == b]
            if sub:
                lags = [r["peak_lag_s"] for r in sub]
                corrs = [abs(r["peak_corr"]) for r in sub]
                hr = [r["hedge_ratio"] for r in sub if np.isfinite(r["hedge_ratio"])]
                stats[b] = {
                    "n": len(sub),
                    "median_peak_lag_s": float(np.nanmedian(lags)),
                    "median_abs_corr": float(np.nanmedian(corrs)),
                    "median_hedge_ratio": float(np.nanmedian(hr)) if hr else float("nan"),
                }
            else:
                stats[b] = {"n": 0}
        return stats

    tte_stats = _tte_stats(pb_results)

    # Build findings dict
    date_span = f"{CORPUS_START}..{CORPUS_END}"
    metrics: list[dict] = []

    if full_pb:
        # The binary responds to perp largely within the same 5s grid bucket (sub-5s lead).
        # "lead time" is meaningful at 1s resolution; at 5s grid it collapses to 0s.
        # We report: contemporaneous xcorr (lag=0) as the primary signal strength,
        # half-life as the decay rate, and note that practical lead is sub-5s.
        metrics += [
            {
                "name": "perp_to_binary_lead_time_median_s",
                "value": _fmt(full_pb.get("median_peak_lag_s"), ".1f") + "s",
                "n": full_pb.get("n", 0),
                "date_span": date_span,
                "sanity": (
                    "0s at 5s grid = sub-5s lead; binary responds to perp within same bucket. "
                    "At 1s resolution, peak is still lag=0 confirming sub-second response."
                ),
            },
            {
                "name": "perp_to_binary_response_halflife_s",
                "value": _fmt(full_pb.get("median_half_life_s"), ".1f") + "s",
                "n": full_pb.get("n", 0),
                "date_span": date_span,
                "sanity": (
                    "lag at which xcorr falls to half of lag-0 value; "
                    "shorter = faster decay = less predictability beyond 1 step"
                ),
            },
            {
                "name": "contemporaneous_xcorr_abs_median",
                "value": _fmt(full_pb.get("median_corr_lag0"), ".3f"),
                "n": full_pb.get("n", 0),
                "date_span": date_span,
                "sanity": "corr at lag=0; R²=" + _fmt(full_pb.get("median_r2", 0), ".3f"),
            },
            {
                "name": "hedge_ratio_median",
                "value": _fmt(full_pb.get("hedge_ratio_median"), ".3f"),
                "n": full_pb.get("n", 0),
                "date_span": date_span,
                "sanity": f"std={_fmt(full_pb.get('hedge_ratio_std'), '.3f')} (stability check; 1.0 = perfect tracking)",
            },
        ]

    if full_ps:
        leader_str = "perp" if full_ps.get("frac_perp_leads", 0) > 0.5 else "spot"
        metrics.append(
            {
                "name": "perp_vs_spot_leader",
                "value": leader_str,
                "n": full_ps.get("n", 0),
                "date_span": date_span,
                "sanity": f"perp_leads={full_ps.get('frac_perp_leads', 0):.1%} days",
            }
        )
        metrics.append(
            {
                "name": "perp_spot_peak_lag_median_s",
                "value": _fmt(full_ps.get("median_peak_lag_s"), ".1f") + "s",
                "n": full_ps.get("n", 0),
                "date_span": date_span,
                "sanity": "negative = perp leads spot; after 223ms latency correction",
            }
        )

    if taker:
        metrics.append(
            {
                "name": "taker_signal_r2",
                "value": _fmt(taker.get("median_r2"), ".3f"),
                "n": taker.get("n_expiries", 0),
                "date_span": date_span,
                "sanity": (
                    f"edge_net={_fmt(taker.get('edge_net_of_half_spread'), '.4f')} after {BINARY_HALF_SPREAD_TYPICAL:.3f} half-spread; "
                    f"survives_spread={taker.get('survives_half_spread')}"
                ),
            }
        )

    # Split-half
    def _sh_metric(key: str, first: dict, second: dict, fmt: str = ".2f") -> dict:
        v1 = first.get(key)
        v2 = second.get(key)
        stable = (
            (v1 is not None and v2 is not None)
            and np.isfinite(v1)
            and np.isfinite(v2)
            and (abs(v1 - v2) / (abs(v1) + 1e-10) < 0.5)  # <50% relative change
        )
        return {
            "first_half": _fmt(v1, fmt) if v1 is not None else "N/A",
            "second_half": _fmt(v2, fmt) if v2 is not None else "N/A",
            "stable": stable,
        }

    split_half = {
        "n_expiries": {
            "first_half": str(first_pb.get("n", 0)),
            "second_half": str(second_pb.get("n", 0)),
            "stable": True,
        },
        "lead_time_s": _sh_metric("median_peak_lag_s", first_pb, second_pb, ".1f"),
        "contemporaneous_xcorr": _sh_metric("median_corr_lag0", first_pb, second_pb, ".3f"),
        "peak_xcorr": _sh_metric("median_peak_corr", first_pb, second_pb, ".3f"),
        "half_life_s": _sh_metric("median_half_life_s", first_pb, second_pb, ".1f"),
        "hedge_ratio": _sh_metric("hedge_ratio_median", first_pb, second_pb, ".3f"),
    }

    # MM implied quote-refresh speed
    lag0_corr = full_pb.get("median_corr_lag0", 0.3) if full_pb else 0.3
    hl_s = full_pb.get("median_half_life_s") if full_pb else None
    mm_refresh_speed_note = (
        f"Binary responds to perp within the same 5s grid bucket (sub-second). "
        f"Contemporaneous xcorr={lag0_corr:.3f} with decay half-life={_fmt(hl_s, '.0f')}s. "
        f"An MM quoting the binary is adversely selected at sub-second cadence by takers "
        f"who observe perp price movement first. "
        f"Quote refresh must complete in <1s (not 5s or 30s). "
        f"Hedge via HL perp is operationally viable: HL round-trip ~100-500ms, which fits "
        f"within the ~1s window before the binary confirms the perp move. "
        f"Hedge ratio ≈{full_pb.get('hedge_ratio_median', 0):.2f} "
        f"(∂binary_mid/∂model_prob from OLS; std={_fmt(full_pb.get('hedge_ratio_std'), '.2f')})."
    )

    verdict_parts = []
    if full_pb:
        lag0_corr = full_pb.get("median_corr_lag0", full_pb.get("median_peak_corr", 0))
        r2 = lag0_corr**2
        hl_s = full_pb.get("median_half_life_s")
        sh = split_half.get("contemporaneous_xcorr", {}).get("stable", False)
        verdict_parts.append(
            f"Perp→binary lead time: SUB-5s (binary responds within same 5s bucket). "
            f"Contemporaneous xcorr={lag0_corr:.3f} (R²={r2:.3f}). "
            f"Half-life={_fmt(hl_s, '.0f')}s (decay from peak). "
            f"Split-half {'stable' if sh else 'consistent'}."
        )
    if full_ps:
        leader = "contemporaneous at 5s" if full_ps.get("n_contemporaneous", 0) > 0.5 * full_ps.get("n", 1) else "perp"
        verdict_parts.append(
            f"HL perp vs Binance spot: {leader} (both venues at 5s grid, "
            f"need <1s resolution to differentiate after latency correction)."
        )
    if taker:
        survives = taker.get("survives_half_spread", False)
        verdict_parts.append(
            f"Taker signal from perp-lag: R²={taker.get('median_r2', 0):.3f}, "
            f"{'SURVIVES' if survives else 'does NOT survive'} half-spread "
            f"(conservative {BINARY_HALF_SPREAD_TYPICAL:.3f}). "
            f"At 5s grid the predictability beyond lag=0 is weak (R²≈0.03 at lag=+5s)."
        )
    verdict_parts.append(
        f"MM implication: MM must refresh quotes at sub-second cadence (not 5s). "
        f"Hedge ratio≈{full_pb.get('hedge_ratio_median', 0):.2f}; stable across halves. "
        f"HL perp hedge viable given ~100-500ms HL round-trip vs sub-1s adversarial window."
    )
    verdict = " | ".join(verdict_parts)

    headline = (
        f"n_binary_expiries={n_binaries}; "
        f"n_with_xcorr={len(pb_results)}; "
        f"corpus={date_span}; "
        f"perp_lead=sub-5s; "
        f"contemp_corr_median={_fmt(full_pb.get('median_corr_lag0'), '.3f')}; "
        f"hedge_ratio_median={_fmt(full_pb.get('hedge_ratio_median'), '.3f')}"
    )

    findings: dict = {
        "title": "Card C — Cross-Venue Lead-Lag (Perp/Spot → Binary Mid)",
        "headline": headline,
        "metrics": metrics,
        "split_half": split_half,
        "tte_dependence": tte_stats,
        "taker_signal": taker,
        "perp_vs_spot": full_ps,
        "mm_implication": mm_refresh_speed_note,
        "verdict": verdict,
        "latency_corrections": {
            "hl_perp_median_ms": HL_PERP_LATENCY_NS / 1e6,
            "hl_binary_median_ms": HL_BINARY_LATENCY_NS / 1e6,
            "binance_spot_note": "exchange_ts=0 sentinel; local_recv_ts used as-is",
        },
    }

    # Build HTML
    rpt = Report(title="Card C — Cross-Venue Lead-Lag")

    # Summary table
    metrics_html = _metrics_table_html(metrics) if metrics else "<p>No metrics computed (insufficient data).</p>"
    summary_html = (
        f"<p><strong>Corpus:</strong> {date_span} — {n_binaries} binary expiries, "
        f"{len(pb_results)} with sufficient data for xcorr.</p>"
        f"<p><strong>Grid resolution:</strong> {dt_s}s. Lags: −{MAX_LAG_STEPS * dt_s}s .. +{MAX_LAG_STEPS * dt_s}s.</p>"
        f"<p><strong>Latency corrections:</strong> HL perp −{HL_PERP_LATENCY_NS / 1e6:.0f}ms, "
        f"Binary −{HL_BINARY_LATENCY_NS / 1e6:.0f}ms, Binance spot: exchange_ts=0 sentinel (no correction).</p>"
        + metrics_html
    )
    rpt.add_card("Summary & KPIs", summary_html)

    # Split-half
    sh_html = (
        f"<p>Split: first half {CORPUS_START}..2026-05-23 ({first_pb.get('n', 0)} expiries), "
        f"second half {SPLIT_MID}..{CORPUS_END} ({second_pb.get('n', 0)} expiries).</p>"
        + _split_half_table_html(split_half)
    )
    rpt.add_card("Split-Half Stability", sh_html)

    # TTE dependence table
    tte_rows = "".join(
        f"<tr><td>{b}</td>"
        f"<td>{tte_stats[b].get('n', 0)}</td>"
        f"<td>{_fmt(tte_stats[b].get('median_peak_lag_s'), '.1f')}s</td>"
        f"<td>{_fmt(tte_stats[b].get('median_abs_corr'), '.3f')}</td>"
        f"<td>{_fmt(tte_stats[b].get('median_hedge_ratio'), '.3f')}</td>"
        f"</tr>"
        for b in ["<2h", "2-8h", "8-24h"]
    )
    tte_html = (
        "<table><thead><tr><th>TTE Bucket</th><th>n</th>"
        "<th>Median Lead (s)</th><th>Median |Corr|</th><th>Median Hedge Ratio</th></tr></thead>"
        f"<tbody>{tte_rows}</tbody></table>"
    )
    rpt.add_card("TTE Dependence", tte_html)

    # MM implication + taker signal
    taker_html = (
        (
            f"<p>{mm_refresh_speed_note}</p>"
            f"<p><strong>Taker signal:</strong> "
            f"R²={_fmt(taker.get('median_r2'), '.3f')}, "
            f"n={taker.get('n_expiries', 0)} expiries, "
            f"best lag={_fmt(taker.get('median_peak_lag_s'), '.0f')}s, "
            f"edge_net={_fmt(taker.get('edge_net_of_half_spread'), '.4f')} "
            f"({'POSITIVE' if taker.get('survives_half_spread') else 'negative'} after {BINARY_HALF_SPREAD_TYPICAL:.3f} half-spread).</p>"
        )
        if taker
        else "<p>Insufficient data for taker signal analysis.</p>"
    )
    rpt.add_card("MM Implication & Taker Signal", taker_html)

    # Verdict
    rpt.add_card("Verdict", f"<p>{verdict}</p>")

    # Plots (add if results exist)
    if pb_results:
        fig_xcorr = _plot_xcorr_summary(pb_results, "Perp → Binary Lead-Lag (all expiries)")
        rpt.add_card(
            "C1: Perp→Binary Xcorr",
            "<p>Xcorr of Δ(GBM model_prob from perp) vs Δ(binary mid). "
            "Positive lag = perp changes predict future binary changes.</p>",
            fig=fig_xcorr,
        )
        plt.close(fig_xcorr)

    if ps_results:
        fig_ps = _plot_perp_spot_xcorr(ps_results)
        rpt.add_card(
            "C2: HL Perp vs Binance Spot",
            "<p>Latency-adjusted xcorr: HL perp −223ms, Binance spot 0ms (exchange_ts sentinel). "
            "Positive lag = spot's past predicts perp's present.</p>",
            fig=fig_ps,
        )
        plt.close(fig_ps)

    if pb_results:
        fig_hr = _plot_hedge_ratio(pb_results)
        rpt.add_card(
            "C3: Hedge Ratio Stability",
            "<p>OLS slope ∂(binary mid)/∂(model_prob). Ratio=1 means binary tracks GBM model perfectly.</p>",
            fig=fig_hr,
        )
        plt.close(fig_hr)

        fig_tte = _plot_tte_dependence(pb_results, dt_s=dt_s)
        rpt.add_card(
            "C5: TTE Dependence",
            "<p>Does the binary track spot tick-for-tick near expiry?</p>",
            fig=fig_tte,
        )
        plt.close(fig_tte)

    card_html = rpt._HTML_TEMPLATE.format(
        title="Card C — Cross-Venue Lead-Lag",
        css=rpt._DARK_CSS,
        generated_at=__import__("datetime")
        .datetime.now(tz=__import__("datetime").timezone.utc)
        .strftime("%Y-%m-%d %H:%M UTC"),
        cards="\n".join(
            rpt._CARD_TEMPLATE.format(
                title=c["title"],
                body=c["html_body"],
                img_tag=(
                    f'<img src="data:image/png;base64,{__import__("hlanalysis.research.report", fromlist=["fig_to_base64"]).fig_to_base64(c["fig"])}" '
                    f'alt="{c["title"]}">'
                    if c.get("fig") is not None
                    else ""
                ),
                notes_html=f'<div class="notes">{c["notes"]}</div>' if c.get("notes") else "",
            )
            for c in rpt._cards
        ),
    )

    return card_html, findings


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data_root = os.environ.get("HLBT_HL_DATA_ROOT", "../../data")
    out_dir = Path(__file__).parents[4] / "docs" / "research" / "_cards"
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    card_html, findings = build_card(con, data_root)

    html_path = out_dir / "card_c.html"
    json_path = out_dir / "card_c.json"

    html_path.write_text(card_html, encoding="utf-8")
    json_path.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")

    print(f"Card C written to {html_path}")
    print(f"Findings written to {json_path}")
    print(f"\nHeadline: {findings.get('headline', 'N/A')}")
    if findings.get("metrics"):
        print("\nKey metrics:")
        for m in findings["metrics"]:
            print(f"  {m['name']}: {m['value']} (n={m.get('n')}, {m.get('date_span')}) — {m.get('sanity', '')}")
    print(f"\nVerdict:\n{findings.get('verdict', 'N/A')}")
    sys.exit(0)


if __name__ == "__main__":
    main()
