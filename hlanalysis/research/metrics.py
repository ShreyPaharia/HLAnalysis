"""Pure metric functions for HL outcome-market analysis.

All functions are side-effect-free and operate on pandas/numpy primitives.
No DuckDB or file I/O here — that belongs in dataset.py.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from hlanalysis.analysis.microstructure import cross_correlation


def spread_bps(
    bid: float | np.ndarray,
    ask: float | np.ndarray,
) -> float | np.ndarray:
    """Return quoted spread in basis points.

    spread_bps = (ask - bid) / mid * 10_000
    where mid = (ask + bid) / 2.

    Works element-wise on scalars or numpy arrays.

    Parameters
    ----------
    bid, ask:
        Best bid and ask prices.  Scalars or same-length arrays.

    Returns
    -------
    Spread in basis points (float or ndarray matching input shape).
    """
    bid_a = np.asarray(bid, dtype="float64")
    ask_a = np.asarray(ask, dtype="float64")
    mid = (bid_a + ask_a) / 2.0
    result = (ask_a - bid_a) / mid * 10_000.0
    return float(result) if np.ndim(bid) == 0 and np.ndim(ask) == 0 else result


def depth_at_n(levels: list[tuple[float, float]], n: int) -> float:
    """Return total quantity in the top ``n`` levels.

    Parameters
    ----------
    levels:
        Ordered list of (price, qty) tuples, best-first (highest bid or lowest
        ask first).
    n:
        Number of levels to sum.

    Returns
    -------
    Total quantity as float.
    """
    return float(sum(qty for _, qty in levels[:n]))


def trade_markout_curve(
    trades_df: pd.DataFrame,
    bbo_df: pd.DataFrame,
    horizons_s: list[int],
) -> pd.DataFrame:
    """Compute trade markout curve using pre-loaded trade and BBO DataFrames.

    Wraps the logic from ``hlanalysis.analysis.markouts.trade_markouts`` but
    operates on pre-loaded DataFrames rather than issuing DuckDB queries.

    Parameters
    ----------
    trades_df:
        DataFrame with columns: ts_ns (int64), price (float64), size (float64),
        aggressor_side (str: 'buy'|'sell').
    bbo_df:
        DataFrame with columns: ts_ns (int64), bid_px (float64), ask_px (float64).
    horizons_s:
        Forward horizons in seconds.

    Returns
    -------
    DataFrame with one row per trade and columns:
        ts_ns, price, size, aggressor_side, mid_at_trade,
        mid_at_h_{h}s, markout_h_{h}s  (for each horizon h)

    Notes
    -----
    Uses LOCF as-of join on bbo_df to compute mids.
    """
    from hlanalysis.analysis.helpers import asof_locf

    _NS = 1_000_000_000

    if trades_df.empty:
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

    # Build mid series from bbo_df
    if bbo_df.empty:
        bbo_ts = np.array([], dtype="int64")
        bbo_mid = np.array([], dtype="float64")
    else:
        bbo_ts = bbo_df["ts_ns"].to_numpy(dtype="int64")
        bbo_mid = ((bbo_df["bid_px"] + bbo_df["ask_px"]) / 2.0).to_numpy(dtype="float64")
        # Sort ascending
        order = np.argsort(bbo_ts)
        bbo_ts = bbo_ts[order]
        bbo_mid = bbo_mid[order]

    trade_ts = trades_df["ts_ns"].to_numpy(dtype="int64")
    sign = np.where(trades_df["aggressor_side"].to_numpy() == "buy", 1.0, -1.0)

    mid_at_trade = asof_locf(trade_ts, bbo_ts, bbo_mid)

    end_ns = int(bbo_ts[-1]) if len(bbo_ts) > 0 else int(trade_ts.max())

    result = trades_df.copy().reset_index(drop=True)
    result["mid_at_trade"] = mid_at_trade

    for h in horizons_s:
        offset_ts = trade_ts + int(h) * _NS
        mid_at_h = asof_locf(offset_ts, bbo_ts, bbo_mid)
        past_end = offset_ts > end_ns
        mid_at_h = np.where(past_end, np.nan, mid_at_h)
        result[f"mid_at_h_{h}s"] = mid_at_h
        result[f"markout_h_{h}s"] = (mid_at_h - mid_at_trade) * sign

    return result


def leadlag_xcorr(
    x: pd.Series,
    y: pd.Series,
    max_lag_steps: int = 20,
) -> pd.DataFrame:
    """Return cross-correlation function between two series across lags.

    Thin wrapper over ``hlanalysis.analysis.microstructure.cross_correlation``.

    Convention (matching cross_correlation):
        lag > 0:  corr(x, y.shift(lag))  — y's past predicts x's present.
        lag = 0:  contemporaneous.
        lag < 0:  x's past predicts y's present.

    Parameters
    ----------
    x, y:
        Two pandas Series of equal length.
    max_lag_steps:
        Maximum absolute lag in steps.

    Returns
    -------
    DataFrame with columns:
        lag (int64)   — integer lag value in [-max_lag_steps, +max_lag_steps]
        corr (float64) — Pearson correlation at that lag
    """
    ccf_df = cross_correlation(x, y, max_lag=max_lag_steps)
    # Rename 'ccf' -> 'corr' for cleaner API
    ccf_df = ccf_df.rename(columns={"ccf": "corr"})
    return ccf_df


def yes_no_overround(yes_ask: float, no_ask: float) -> float:
    """Return the overround (vig) implied by the Yes and No ask prices.

    overround = yes_ask + no_ask - 1

    A perfectly competitive binary market has overround = 0 (yes_ask + no_ask = 1
    if asks equal fair value).  Positive overround = dealer profit margin.

    Parameters
    ----------
    yes_ask:
        Best ask price for the Yes leg (probability units, 0..1).
    no_ask:
        Best ask price for the No leg.

    Returns
    -------
    Overround as a float (e.g. 0.07 = 7 bps on a 0..1 scale).
    """
    return float(yes_ask + no_ask - 1.0)


def implied_prob_gbm(
    spot: float,
    strike: float,
    sigma: float,
    tau_s: float,
) -> float:
    """Return P(S_T > K) under GBM with Itô drift correction.

    Model: S_T = S_0 * exp((μ - ½σ²)τ + σ√τ Z), Z ~ N(0,1)
    Under risk-neutral measure (μ=0 for martingale):
        P(S_T > K) = Φ(d₋)
    where:
        d₋ = (ln(S_0/K) - ½σ²·τ_yr) / (σ·√τ_yr)
        τ_yr = tau_s / (365.25 × 86400)

    The −½σ²τ term is the Itô correction (as noted in project memory:
    omitting it biases p_model by +σ²τ/2).

    Parameters
    ----------
    spot:
        Current spot price S_0.
    strike:
        Target price K.
    sigma:
        Annualised volatility (e.g. 0.20 = 20% per year).
    tau_s:
        Time to expiry in **seconds**.  Internally converted to years.

    Returns
    -------
    Probability in [0, 1].
    """
    if sigma <= 0.0 or tau_s <= 0.0:
        # Degenerate: if spot > strike probability is ~1, else ~0
        return 1.0 if spot > strike else 0.0

    _secs_per_year = 365.25 * 86400.0
    tau_yr = tau_s / _secs_per_year

    sqrt_tau = math.sqrt(tau_yr)
    log_ratio = math.log(spot / strike) if spot > 0 and strike > 0 else 0.0
    d_minus = (log_ratio - 0.5 * sigma * sigma * tau_yr) / (sigma * sqrt_tau)
    # Φ(d₋) = P(Z < d₋) via math.erfc (no scipy import needed)
    # Φ(x) = 0.5 * erfc(-x / sqrt(2))
    from math import erfc, sqrt

    prob = 0.5 * erfc(-d_minus / sqrt(2.0))
    return float(prob)


def theta_decay_curve(panel: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Compute mid-price vs TTE buckets for a given leg symbol.

    Useful for visualising how the market price evolves toward 0 or 1 as the
    market expires.

    Parameters
    ----------
    panel:
        Output of ``build_panel`` (must contain columns ``{sym}_mid`` and
        ``{sym}_tte_s``).
    sym:
        Leg symbol (e.g. '#100').

    Returns
    -------
    DataFrame with columns:
        tte_bucket_h (float64) — TTE bucket centre in hours
        mean_mid     (float64) — mean mid price in bucket
        std_mid      (float64) — std dev of mid in bucket
        count        (int64)   — number of observations
    """
    mid_col = f"{sym}_mid"
    tte_col = f"{sym}_tte_s"

    if mid_col not in panel.columns or tte_col not in panel.columns:
        return pd.DataFrame(columns=["tte_bucket_h", "mean_mid", "std_mid", "count"])

    sub = panel[[tte_col, mid_col]].dropna()
    if sub.empty:
        return pd.DataFrame(columns=["tte_bucket_h", "mean_mid", "std_mid", "count"])

    # Bucket TTE into 1-hour bins
    sub = sub.copy()
    sub["tte_h"] = sub[tte_col] / 3600.0
    # Edges: 0..24h in 1h steps
    max_tte_h = min(sub["tte_h"].max(), 24.0)
    edges = np.arange(0.0, max_tte_h + 1.0, 1.0)
    labels = (edges[:-1] + edges[1:]) / 2.0
    sub["tte_bucket_h"] = pd.cut(sub["tte_h"], bins=edges, labels=labels, right=False)

    agg = sub.groupby("tte_bucket_h", observed=True)[mid_col].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["tte_bucket_h", "mean_mid", "std_mid", "count"]
    agg["tte_bucket_h"] = agg["tte_bucket_h"].astype("float64")
    return agg.sort_values("tte_bucket_h").reset_index(drop=True)


def settlement_convergence_curve(panel: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Compute mean mid vs TTE, split by settlement outcome.

    Parameters
    ----------
    panel:
        Output of ``build_panel`` (must contain ``{sym}_mid``, ``{sym}_tte_s``,
        ``{sym}_settlement_label``).
    sym:
        Leg symbol.

    Returns
    -------
    DataFrame with columns:
        tte_bucket_h      (float64) — TTE bucket centre in hours
        settlement_label  (str)     — 'yes_won', 'no_won', or 'unknown'
        mean_mid          (float64)
        count             (int64)
    """
    mid_col = f"{sym}_mid"
    tte_col = f"{sym}_tte_s"
    label_col = f"{sym}_settlement_label"

    required = [mid_col, tte_col, label_col]
    if any(c not in panel.columns for c in required):
        return pd.DataFrame(columns=["tte_bucket_h", "settlement_label", "mean_mid", "count"])

    sub = panel[required].dropna(subset=[mid_col, tte_col]).copy()
    if sub.empty:
        return pd.DataFrame(columns=["tte_bucket_h", "settlement_label", "mean_mid", "count"])

    sub["tte_h"] = sub[tte_col] / 3600.0
    max_tte_h = min(sub["tte_h"].max(), 24.0)
    edges = np.arange(0.0, max_tte_h + 1.0, 1.0)
    labels = (edges[:-1] + edges[1:]) / 2.0
    sub["tte_bucket_h"] = pd.cut(sub["tte_h"], bins=edges, labels=labels, right=False)

    sub[label_col] = sub[label_col].fillna("unknown")

    agg = sub.groupby(["tte_bucket_h", label_col], observed=True)[mid_col].agg(["mean", "count"]).reset_index()
    agg.columns = ["tte_bucket_h", "settlement_label", "mean_mid", "count"]
    agg["tte_bucket_h"] = agg["tte_bucket_h"].astype("float64")
    return agg.sort_values(["tte_bucket_h", "settlement_label"]).reset_index(drop=True)


def realized_vol_termstructure(
    ohlc_df: pd.DataFrame,
    windows_s: list[int],
) -> pd.DataFrame:
    """Compute Parkinson and bipower realized vol at multiple windows.

    Parameters
    ----------
    ohlc_df:
        DataFrame with columns: ts_ns (int64), open (float64), high (float64),
        low (float64), close (float64).  Rows sorted ascending by ts_ns.
    windows_s:
        List of rolling window sizes in seconds.  The bar width is inferred
        from the median ts_ns gap.

    Returns
    -------
    DataFrame with columns:
        window_s       (int64)   — window size in seconds
        parkinson_vol  (float64) — annualised Parkinson vol estimate
        bipower_vol    (float64) — annualised bipower variation vol estimate
        n_bars         (int64)   — number of bars in the window
    """
    if ohlc_df.empty:
        return pd.DataFrame(columns=["window_s", "parkinson_vol", "bipower_vol", "n_bars"])

    ts = ohlc_df["ts_ns"].to_numpy(dtype="int64")
    highs = ohlc_df["high"].to_numpy(dtype="float64")
    lows = ohlc_df["low"].to_numpy(dtype="float64")
    closes = ohlc_df["close"].to_numpy(dtype="float64")

    # Infer bar width in seconds from median gap
    if len(ts) > 1:
        bar_s = float(np.median(np.diff(ts)) / 1e9)
    else:
        bar_s = 60.0  # fallback: 1 minute

    # Annualisation factor: 365.25 * 86400 seconds per year
    secs_per_year = 365.25 * 86400.0

    records = []
    for w_s in windows_s:
        n_bars = max(1, int(round(w_s / bar_s)))
        if n_bars > len(highs):
            n_bars = len(highs)

        # Parkinson estimator: sigma^2 = (1/4ln2) * mean((ln(H/L))^2)
        # Uses last n_bars rows
        h_slice = highs[-n_bars:]
        l_slice = lows[-n_bars:]
        valid_mask = (h_slice > 0) & (l_slice > 0)
        if valid_mask.sum() < 2:
            records.append(
                {"window_s": w_s, "parkinson_vol": float("nan"), "bipower_vol": float("nan"), "n_bars": n_bars}
            )
            continue

        log_hl = np.log(h_slice[valid_mask] / l_slice[valid_mask])
        parkinson_var_per_bar = (1.0 / (4.0 * math.log(2.0))) * np.mean(log_hl**2)
        # Annualise: var_annual = var_per_bar * (secs_per_year / bar_s)
        parkinson_vol = math.sqrt(parkinson_var_per_bar * secs_per_year / bar_s)

        # Bipower variation: mean(|r_i| * |r_{i-1}|) / (E[|Z|]^2)
        c_slice = closes[-n_bars:]
        valid_c = c_slice > 0
        log_ret = np.full(len(c_slice), np.nan)
        if np.sum(valid_c) > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_ret[1:] = np.diff(np.log(np.where(valid_c, c_slice, np.nan)))
        abs_ret = np.abs(log_ret)
        if np.sum(np.isfinite(abs_ret)) < 2:
            records.append(
                {"window_s": w_s, "parkinson_vol": parkinson_vol, "bipower_vol": float("nan"), "n_bars": n_bars}
            )
            continue
        # Bipower: (pi/2) * mean(|r_i| * |r_{i-1}|) per bar -> annualise
        products = abs_ret[1:] * abs_ret[:-1]
        valid_prod = np.isfinite(products)
        if valid_prod.sum() < 1:
            bipower_vol = float("nan")
        else:
            mu1 = math.sqrt(2.0 / math.pi)  # E[|Z|] for standard normal
            bpv_per_bar = np.mean(products[valid_prod]) / (mu1**2)
            bipower_vol = math.sqrt(bpv_per_bar * secs_per_year / bar_s)

        records.append(
            {
                "window_s": w_s,
                "parkinson_vol": parkinson_vol,
                "bipower_vol": bipower_vol,
                "n_bars": n_bars,
            }
        )

    return pd.DataFrame(records)
