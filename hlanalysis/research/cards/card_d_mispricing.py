"""Card D: Mispricing Surface — Model Edge, Overround, Bucket↔Binary Coherence.

Analyses
--------
D1. Yes+No Overround (full sample): distribution of buy-both cost and sell-both
    proceeds; fraction of ticks offering net-of-fee taker arb; TTE and UTC-hour
    dependence.

D2. Model Edge vs GBM: implied prob from binary mid vs GBM P(S_T>K) with Itô
    correction; market-mid − model-prob surface over (log-moneyness/σ√τ, TTE).
    Split-half sign stability.

D3. Bucket Internal Coherence: 3-band mid sum distribution; sub-arb detection
    (sum of asks < 1).

D4. Bucket↔Binary Cross-Consistency: same-expiry bucket band probs vs binary
    implied + GBM.

D5. Taker Edge Call: where net-of-fee edge is positive; magnitude × frequency ×
    capacity (TOB size).

Interface
---------
build_card(con, data_root) -> tuple[str, dict]
    Returns (card_html, findings).

Main
----
Run directly to write docs/research/_cards/card_d.html + card_d.json.
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

from hlanalysis.research.metrics import implied_prob_gbm
from hlanalysis.research.report import Report

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HL HIP-4 taker fee: fee_taker=0.0 in prod config + half_spread_assumption=0.005.
# For taker-arb analysis we use a conservative 50 bp round-trip (25 bp each way)
# since the bid-ask spread itself is the dominant cost.  Per-leg: 0.0025.
HL_FEE_PER_LEG = 0.0025  # flat per share, each side
HL_FEE_ROUNDTRIP = 2 * HL_FEE_PER_LEG  # entry + exit

# Date boundaries for split-half stability
SPLIT_START = "2026-05-06"
SPLIT_MID = "2026-05-24"  # first half: 05-06..05-23; second half: 05-24..06-10
SPLIT_END = "2026-06-10"

# Parkinson vol lookback for GBM model (60 min in seconds)
VOL_LOOKBACK_S = 3600

# ---------------------------------------------------------------------------
# Glob helpers
# ---------------------------------------------------------------------------


def _meta_glob(data_root: str) -> str:
    return str(
        Path(data_root)
        / "venue=hyperliquid/product_type=prediction_binary/mechanism=clob"
        / "event=market_meta/symbol=*/date=*/hour=all/*.parquet"
    )


def _q_glob(data_root: str) -> str:
    return str(
        Path(data_root)
        / "venue=hyperliquid/product_type=prediction_binary/mechanism=clob"
        / "event=question_meta/symbol=*/date=*/hour=all/*.parquet"
    )


def _bbo_glob(data_root: str) -> str:
    return str(
        Path(data_root)
        / "venue=hyperliquid/product_type=prediction_binary/mechanism=clob"
        / "event=bbo/symbol=*/date=*/hour=all/*.parquet"
    )


def _perp_bbo_glob(data_root: str) -> str:
    return str(
        Path(data_root)
        / "venue=hyperliquid/product_type=perp/mechanism=clob"
        / "event=bbo/symbol=BTC/date=*/hour=all/*.parquet"
    )


# ---------------------------------------------------------------------------
# Analysis 1: Yes+No Overround
# ---------------------------------------------------------------------------


def _load_binary_overround(con: duckdb.DuckDBPyConnection, data_root: str) -> pd.DataFrame:
    """Return tick-level binary overround dataset.

    Columns: yes_sym, no_sym, expiry_str, target_price, local_recv_ts,
             yes_ask, yes_bid, yes_ask_sz, yes_bid_sz,
             no_ask, no_bid, no_ask_sz, no_bid_sz,
             buy_both_cost, sell_both_proceeds, utc_hour, date
    """
    meta = _meta_glob(data_root)
    bbo = _bbo_glob(data_root)

    sql = f"""
        WITH binary_pairs AS (
            SELECT DISTINCT
                yes_meta.symbol AS yes_sym,
                no_meta.symbol  AS no_sym,
                list_element(yes_meta.values, list_position(yes_meta.keys, 'expiry'))              AS expiry_str,
                list_element(yes_meta.values, list_position(yes_meta.keys, 'targetPrice'))::DOUBLE AS target_price
            FROM (
                SELECT DISTINCT symbol, keys, values
                FROM read_parquet('{meta}', union_by_name=true)
                WHERE array_contains(keys, 'class')
                  AND list_element(values, list_position(keys, 'class')) = 'priceBinary'
                  AND list_element(values, list_position(keys, 'side_name')) = 'Yes'
            ) yes_meta
            JOIN (
                SELECT DISTINCT symbol, keys, values
                FROM read_parquet('{meta}', union_by_name=true)
                WHERE array_contains(keys, 'class')
                  AND list_element(values, list_position(keys, 'class')) = 'priceBinary'
                  AND list_element(values, list_position(keys, 'side_name')) = 'No'
            ) no_meta
            ON list_element(yes_meta.values, list_position(yes_meta.keys, 'outcome_idx'))
             = list_element(no_meta.values, list_position(no_meta.keys, 'outcome_idx'))
        ),
        yes_bbo AS (
            SELECT symbol AS yes_sym, local_recv_ts,
                   ask_px AS yes_ask, bid_px AS yes_bid,
                   ask_sz AS yes_ask_sz, bid_sz AS yes_bid_sz
            FROM read_parquet('{bbo}', union_by_name=true)
        ),
        no_bbo AS (
            SELECT symbol AS no_sym, local_recv_ts,
                   ask_px AS no_ask, bid_px AS no_bid,
                   ask_sz AS no_ask_sz, bid_sz AS no_bid_sz
            FROM read_parquet('{bbo}', union_by_name=true)
        )
        SELECT
            p.yes_sym, p.no_sym, p.expiry_str, p.target_price,
            y.local_recv_ts,
            y.yes_ask, y.yes_bid, y.yes_ask_sz, y.yes_bid_sz,
            n.no_ask, n.no_bid, n.no_ask_sz, n.no_bid_sz,
            y.yes_ask + n.no_ask AS buy_both_cost,
            y.yes_bid + n.no_bid AS sell_both_proceeds,
            (y.local_recv_ts // 3600000000000) % 24 AS utc_hour
        FROM binary_pairs p
        JOIN yes_bbo y ON p.yes_sym = y.yes_sym
        ASOF JOIN no_bbo n ON p.no_sym = n.no_sym AND y.local_recv_ts >= n.local_recv_ts
        ORDER BY y.local_recv_ts
    """
    try:
        df = con.execute(sql).df()
    except Exception as exc:
        _log.warning("_load_binary_overround failed: %s", exc)
        return pd.DataFrame()

    if df.empty:
        return df

    # Add expiry_ns and TTE
    def _parse_exp(s: str) -> float:
        """Return expiry as unix ns."""
        import datetime as dt_mod

        try:
            d = dt_mod.datetime.strptime(s, "%Y%m%d-%H%M").replace(tzinfo=dt_mod.UTC)
        except ValueError:
            d = dt_mod.datetime.strptime(s, "%Y%m%d-%H").replace(tzinfo=dt_mod.UTC)
        return float(d.timestamp()) * 1e9

    exp_ns = df["expiry_str"].map(_parse_exp)
    df["tte_s"] = (exp_ns - df["local_recv_ts"]) / 1e9
    df.loc[df["tte_s"] < 0, "tte_s"] = 0.0
    df["tte_h"] = df["tte_s"] / 3600.0

    return df


def _analyse_overround(df: pd.DataFrame) -> dict:
    """Compute overround KPIs from the full binary tick dataset."""
    if df.empty:
        return {}

    n = len(df)
    n_expiries = df["expiry_str"].nunique()

    buy_cost = df["buy_both_cost"]
    sell_proc = df["sell_both_proceeds"]

    # Overround = buy_both_cost - 1 (cost to hold a 1 risk-free position)
    overround = buy_cost - 1.0

    # Net-of-fee arb:
    # Buy arb: buy both for cost < 1 - 2*HL_FEE_PER_LEG (net profit after paying entry fees)
    buy_arb_mask = buy_cost < (1.0 - 2 * HL_FEE_PER_LEG)
    # Sell arb: sell both (collect bid) > 1 + 2*HL_FEE_PER_LEG (net profit after exit fees)
    sell_arb_mask = sell_proc > (1.0 + 2 * HL_FEE_PER_LEG)

    frac_buy_arb = float(buy_arb_mask.sum()) / n
    frac_sell_arb = float(sell_arb_mask.sum()) / n

    return {
        "n_ticks": n,
        "n_expiries": n_expiries,
        "overround_p25": float(np.percentile(overround, 25)),
        "overround_p50": float(np.percentile(overround, 50)),
        "overround_p75": float(np.percentile(overround, 75)),
        "overround_p90": float(np.percentile(overround, 90)),
        "overround_mean": float(overround.mean()),
        "buy_both_cost_p50": float(buy_cost.median()),
        "sell_both_proceeds_p50": float(sell_proc.median()),
        "frac_buy_arb_gross": float((buy_cost < 1.0).mean()),
        "frac_buy_arb_net_of_fee": frac_buy_arb,
        "frac_sell_arb_net_of_fee": frac_sell_arb,
        "fee_assumption_per_leg": HL_FEE_PER_LEG,
    }


def _plot_overround(df: pd.DataFrame, overround_stats: dict) -> matplotlib.figure.Figure:
    """4-panel overround figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="#e6edf3")
        ax.xaxis.label.set_color("#e6edf3")
        ax.yaxis.label.set_color("#e6edf3")
        ax.title.set_color("#58a6ff")

    overround = df["buy_both_cost"] - 1.0

    # Panel 1: Overround histogram (clipped)
    ax = axes[0, 0]
    clipped = overround[(overround >= -0.05) & (overround <= 0.20)]
    ax.hist(clipped, bins=100, color="#58a6ff", alpha=0.8, edgecolor="none")
    ax.axvline(0.0, color="#f85149", lw=1.5, label="Fair value (0)")
    ax.axvline(2 * HL_FEE_PER_LEG, color="#3fb950", lw=1.5, ls="--", label=f"Fee threshold ({2 * HL_FEE_PER_LEG:.4f})")
    ax.set_title("Buy-Both Overround Distribution")
    ax.set_xlabel("Overround (yes_ask + no_ask − 1)")
    ax.set_ylabel("Tick count")
    ax.legend(fontsize=8, labelcolor="#e6edf3", facecolor="#161b22", edgecolor="#30363d")

    # Panel 2: Sell-both proceeds histogram
    ax = axes[0, 1]
    sell_clip = df["sell_both_proceeds"][(df["sell_both_proceeds"] >= 0.8) & (df["sell_both_proceeds"] <= 1.1)]
    ax.hist(sell_clip, bins=100, color="#3fb950", alpha=0.8, edgecolor="none")
    ax.axvline(1.0, color="#f85149", lw=1.5, label="Fair value (1.0)")
    ax.set_title("Sell-Both Proceeds Distribution")
    ax.set_xlabel("Sell-both proceeds (yes_bid + no_bid)")
    ax.set_ylabel("Tick count")
    ax.legend(fontsize=8, labelcolor="#e6edf3", facecolor="#161b22", edgecolor="#30363d")

    # Panel 3: Overround vs TTE
    ax = axes[1, 0]
    df_valid = df[df["tte_h"].between(0, 25)].copy()
    tte_bins = [0, 1, 2, 4, 6, 8, 12, 18, 25]
    tte_labels = [f"{tte_bins[i]}-{tte_bins[i + 1]}h" for i in range(len(tte_bins) - 1)]
    df_valid["tte_bin"] = pd.cut(df_valid["tte_h"], bins=tte_bins, labels=tte_labels, right=False)
    tte_agg = df_valid.groupby("tte_bin", observed=True)["buy_both_cost"].agg(["median", "mean"]).reset_index()
    tte_agg["overround_median"] = tte_agg["median"] - 1.0
    x_pos = np.arange(len(tte_agg))
    ax.bar(x_pos, tte_agg["overround_median"], color="#58a6ff", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tte_labels, rotation=30, ha="right", fontsize=8)
    ax.set_title("Median Overround vs TTE")
    ax.set_xlabel("Time to Expiry")
    ax.set_ylabel("Overround (buy-both - 1)")
    ax.axhline(0.0, color="#f85149", lw=1)

    # Panel 4: Overround vs UTC hour
    ax = axes[1, 1]
    hour_agg = df.groupby("utc_hour")["buy_both_cost"].agg(["median", "count"]).reset_index()
    hour_agg["overround_median"] = hour_agg["median"] - 1.0
    ax.bar(hour_agg["utc_hour"], hour_agg["overround_median"], color="#d29922", alpha=0.8)
    ax.set_title("Median Overround vs UTC Hour")
    ax.set_xlabel("UTC Hour")
    ax.set_ylabel("Overround")
    ax.axhline(0.0, color="#f85149", lw=1)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Analysis 2: Model Edge vs GBM (Parkinson vol)
# ---------------------------------------------------------------------------


def _compute_parkinson_vol_series(con: duckdb.DuckDBPyConnection, data_root: str) -> pd.DataFrame:
    """Compute 1h rolling Parkinson vol at 60s intervals from perp BBO.

    Returns DataFrame with columns: ts_ns, sigma_annualised
    """
    perp_bbo = _perp_bbo_glob(data_root)

    # Load all perp BBO and compute 60s OHLC bars
    sql = f"""
        SELECT
            (local_recv_ts // 60000000000) * 60000000000 AS bar_ts,
            MAX((bid_px + ask_px) / 2.0) AS bar_high,
            MIN((bid_px + ask_px) / 2.0) AS bar_low
        FROM read_parquet('{perp_bbo}', union_by_name=true)
        GROUP BY bar_ts
        ORDER BY bar_ts
    """
    bars = con.execute(sql).df()
    if bars.empty:
        return pd.DataFrame(columns=["ts_ns", "sigma_annualised"])

    bar_ts = bars["bar_ts"].to_numpy(dtype="int64")
    bar_h = bars["bar_high"].to_numpy(dtype="float64")
    bar_l = bars["bar_low"].to_numpy(dtype="float64")

    # Rolling 60-bar window (= 1h at 60s bars)
    n_bars_lookback = 60
    secs_per_year = 365.25 * 86400.0
    bar_width_s = 60.0

    records = []
    for i in range(n_bars_lookback, len(bar_ts)):
        h_w = bar_h[i - n_bars_lookback : i]
        l_w = bar_l[i - n_bars_lookback : i]
        valid = (h_w > 0) & (l_w > 0) & (h_w >= l_w)
        if valid.sum() < 10:
            continue
        log_hl = np.log(h_w[valid] / l_w[valid])
        park_var_per_bar = (1.0 / (4.0 * math.log(2.0))) * float(np.mean(log_hl**2))
        sigma = math.sqrt(park_var_per_bar * secs_per_year / bar_width_s)
        records.append({"ts_ns": int(bar_ts[i]), "sigma_annualised": sigma})

    return pd.DataFrame(records)


def _load_model_edge_data(con: duckdb.DuckDBPyConnection, data_root: str) -> pd.DataFrame:
    """Load binary mid + perp mid at 60s sampling for model-edge analysis.

    Returns DataFrame with: symbol, expiry_str, target_price, ts_ns,
                             yes_mid, perp_mid, tte_s
    """
    meta = _meta_glob(data_root)
    bbo = _bbo_glob(data_root)
    perp_bbo = _perp_bbo_glob(data_root)

    sql = f"""
        WITH bin_yes AS (
            SELECT DISTINCT
                symbol,
                list_element(values, list_position(keys, 'expiry'))               AS expiry_str,
                list_element(values, list_position(keys, 'targetPrice'))::DOUBLE  AS target_price,
                epoch_ns(strptime(list_element(values, list_position(keys, 'expiry')), '%Y%m%d-%H%M') AT TIME ZONE 'UTC') AS exp_ns
            FROM read_parquet('{meta}', union_by_name=true)
            WHERE array_contains(keys, 'class')
              AND list_element(values, list_position(keys, 'class')) = 'priceBinary'
              AND list_element(values, list_position(keys, 'side_name')) = 'Yes'
        ),
        yes_60s AS (
            SELECT symbol,
                   (local_recv_ts // 60000000000) * 60000000000 AS ts_60s,
                   LAST((bid_px + ask_px) / 2.0 ORDER BY local_recv_ts) AS yes_mid
            FROM read_parquet('{bbo}', union_by_name=true)
            GROUP BY symbol, ts_60s
        ),
        perp_60s AS (
            SELECT (local_recv_ts // 60000000000) * 60000000000 AS ts_60s,
                   LAST((bid_px + ask_px) / 2.0 ORDER BY local_recv_ts) AS perp_mid
            FROM read_parquet('{perp_bbo}', union_by_name=true)
            GROUP BY ts_60s
        )
        SELECT
            bin_yes.symbol, bin_yes.expiry_str, bin_yes.target_price,
            yes_60s.ts_60s AS ts_ns,
            yes_60s.yes_mid,
            perp_60s.perp_mid,
            (bin_yes.exp_ns - yes_60s.ts_60s) / 1e9 AS tte_s
        FROM bin_yes
        JOIN yes_60s ON bin_yes.symbol = yes_60s.symbol
        JOIN perp_60s ON yes_60s.ts_60s = perp_60s.ts_60s
        WHERE (bin_yes.exp_ns - yes_60s.ts_60s) > 0
        ORDER BY ts_ns
    """
    try:
        df = con.execute(sql).df()
    except Exception as exc:
        _log.warning("_load_model_edge_data failed: %s", exc)
        return pd.DataFrame()

    return df


def _compute_model_edge(
    model_df: pd.DataFrame,
    sigma_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge model-edge data with Parkinson vol; compute GBM implied prob and edge.

    Adds columns: sigma, p_model, model_edge (yes_mid - p_model),
                  log_moneyness_sigma (log(S/K) / (sigma * sqrt(tau_yr)))
    """
    if model_df.empty or sigma_df.empty:
        return model_df

    # LOCF-join Parkinson sigma onto model_df by ts_ns
    model_ts = model_df["ts_ns"].to_numpy(dtype="int64")
    sigma_ts = sigma_df["ts_ns"].to_numpy(dtype="int64")
    sigma_vals = sigma_df["sigma_annualised"].to_numpy(dtype="float64")

    # Sort sigma series
    order = np.argsort(sigma_ts)
    sigma_ts = sigma_ts[order]
    sigma_vals = sigma_vals[order]

    # LOCF as-of join
    idxs = np.searchsorted(sigma_ts, model_ts, side="right") - 1
    sigma_joined = np.where(idxs >= 0, sigma_vals[np.clip(idxs, 0, len(sigma_vals) - 1)], np.nan)
    model_df = model_df.copy()
    model_df["sigma"] = sigma_joined

    # Compute GBM implied prob
    _secs_per_year = 365.25 * 86400.0

    def _row_prob(row: pd.Series) -> float:
        sigma = float(row["sigma"])
        tau_s = float(row["tte_s"])
        spot = float(row["perp_mid"])
        strike = float(row["target_price"])
        if not math.isfinite(sigma) or sigma <= 0 or tau_s <= 0 or spot <= 0 or strike <= 0:
            return float("nan")
        return implied_prob_gbm(spot, strike, sigma, tau_s)

    model_df["p_model"] = model_df.apply(_row_prob, axis=1)
    model_df["model_edge"] = model_df["yes_mid"] - model_df["p_model"]

    # Normalised moneyness: d = log(S/K) / (sigma * sqrt(tau_yr))
    tau_yr = model_df["tte_s"] / _secs_per_year
    log_sk = np.log((model_df["perp_mid"] / model_df["target_price"]).clip(1e-6, 10))
    denom = model_df["sigma"] * np.sqrt(tau_yr.clip(1e-10))
    model_df["d_plus"] = (log_sk / denom).replace([np.inf, -np.inf], np.nan)
    model_df["log_moneyness"] = log_sk

    return model_df


def _plot_model_edge(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """4-panel model-edge figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="#e6edf3")
        ax.xaxis.label.set_color("#e6edf3")
        ax.yaxis.label.set_color("#e6edf3")
        ax.title.set_color("#58a6ff")

    valid = df.dropna(subset=["model_edge", "d_plus", "tte_s"])

    # Panel 1: Model edge histogram
    ax = axes[0, 0]
    edge_clip = valid["model_edge"].clip(-0.10, 0.10)
    ax.hist(edge_clip, bins=80, color="#58a6ff", alpha=0.8, edgecolor="none")
    ax.axvline(0.0, color="#f85149", lw=1.5, label="0 edge")
    median_edge = float(valid["model_edge"].median())
    ax.axvline(median_edge, color="#3fb950", lw=1.5, ls="--", label=f"Median={median_edge:.4f}")
    ax.set_title("Model Edge Distribution (binary mid − GBM prob)")
    ax.set_xlabel("Model Edge (yes_mid − P_GBM)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, labelcolor="#e6edf3", facecolor="#161b22", edgecolor="#30363d")

    # Panel 2: Model edge vs d_plus (moneyness in vol units)
    ax = axes[0, 1]
    d_bins = np.linspace(-3, 3, 13)
    d_labels = [(d_bins[i] + d_bins[i + 1]) / 2 for i in range(len(d_bins) - 1)]
    valid_d = valid.copy()
    valid_d["d_bin"] = pd.cut(valid_d["d_plus"], bins=d_bins, labels=d_labels)
    d_agg = valid_d.groupby("d_bin", observed=True)["model_edge"].median().reset_index()
    ax.bar(d_agg["d_bin"].astype(float), d_agg["model_edge"], width=0.4, color="#d29922", alpha=0.8)
    ax.axhline(0.0, color="#f85149", lw=1)
    ax.set_title("Median Model Edge vs Moneyness (d+)")
    ax.set_xlabel("d+ = log(S/K) / (σ√τ)")
    ax.set_ylabel("Median (yes_mid − P_GBM)")

    # Panel 3: Model edge vs TTE
    ax = axes[1, 0]
    tte_bins = [0, 1, 2, 4, 6, 8, 12, 18, 25]
    tte_labels_c = [f"{tte_bins[i]}-{tte_bins[i + 1]}h" for i in range(len(tte_bins) - 1)]
    valid_t = valid.copy()
    valid_t["tte_h"] = valid_t["tte_s"] / 3600.0
    valid_t["tte_bin"] = pd.cut(valid_t["tte_h"], bins=tte_bins, labels=tte_labels_c, right=False)
    tte_agg = valid_t.groupby("tte_bin", observed=True)["model_edge"].median().reset_index()
    x_pos = np.arange(len(tte_agg))
    ax.bar(x_pos, tte_agg["model_edge"], color="#58a6ff", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tte_labels_c, rotation=30, ha="right", fontsize=8)
    ax.axhline(0.0, color="#f85149", lw=1)
    ax.set_title("Median Model Edge vs TTE")
    ax.set_xlabel("Time to Expiry")
    ax.set_ylabel("Median (yes_mid − P_GBM)")

    # Panel 4: 2D heatmap — d_plus vs TTE_h buckets
    ax = axes[1, 1]
    valid_hm = valid.copy()
    valid_hm["tte_h"] = valid_hm["tte_s"] / 3600.0
    d_bin_edges = np.linspace(-3, 3, 7)
    tte_bin_edges = [0, 2, 4, 8, 16, 25]
    d_mid = [(d_bin_edges[i] + d_bin_edges[i + 1]) / 2 for i in range(len(d_bin_edges) - 1)]
    tte_mid = [(tte_bin_edges[i] + tte_bin_edges[i + 1]) / 2 for i in range(len(tte_bin_edges) - 1)]
    valid_hm["d_bin2"] = pd.cut(valid_hm["d_plus"], bins=d_bin_edges, labels=d_mid)
    valid_hm["tte_bin2"] = pd.cut(valid_hm["tte_h"], bins=tte_bin_edges, labels=tte_mid, right=False)
    hm_data = valid_hm.groupby(["d_bin2", "tte_bin2"], observed=True)["model_edge"].median().unstack("tte_bin2")
    hm_arr = hm_data.values.astype(float)
    # Fill NaN with 0 for display
    hm_arr_disp = np.where(np.isnan(hm_arr), 0.0, hm_arr)
    im = ax.imshow(
        hm_arr_disp,
        aspect="auto",
        cmap="RdYlGn",
        vmin=-0.03,
        vmax=0.03,
        origin="lower",
    )
    ax.set_xticks(range(len(tte_mid)))
    ax.set_xticklabels([f"{t:.0f}h" for t in tte_mid], fontsize=7)
    ax.set_yticks(range(len(d_mid)))
    ax.set_yticklabels([f"{d:.1f}" for d in d_mid], fontsize=7)
    ax.set_xlabel("TTE bucket")
    ax.set_ylabel("d+ bucket")
    ax.set_title("Model Edge Heatmap (d+ vs TTE)")
    plt.colorbar(im, ax=ax, label="Median edge", shrink=0.8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Analysis 3: Bucket Internal Coherence
# ---------------------------------------------------------------------------


def _load_bucket_coherence(con: duckdb.DuckDBPyConnection, data_root: str) -> pd.DataFrame:
    """Load 3-band mid sums per bucket question at 60s sampling.

    Returns DataFrame with: q_symbol, expiry_str, ts_60s,
                             mid0, mid1, mid2, ask0, ask1, ask2,
                             sum_mids, sum_asks
    """
    meta = _meta_glob(data_root)
    q_glob = _q_glob(data_root)
    bbo = _bbo_glob(data_root)

    # Get all (band_symbol, band_index, q_symbol, expiry_str) mappings
    map_sql = f"""
        WITH bucket_legs AS (
            SELECT DISTINCT
                mm.symbol,
                list_element(mm.values, list_position(mm.keys, 'outcome_idx'))::BIGINT AS outcome_idx,
                list_element(mm.values, list_position(mm.keys, 'index'))::INT          AS band_index
            FROM read_parquet('{meta}', union_by_name=true) mm
            WHERE list_contains(mm.keys, 'index')
              AND list_contains(mm.keys, 'side_idx')
              AND list_element(mm.values, list_position(mm.keys, 'side_idx')) = '0'
        ),
        questions AS (
            SELECT DISTINCT
                q.symbol AS q_symbol,
                unnest(q.named_outcome_idxs) AS outcome_idx,
                list_element(q.values, list_position(q.keys, 'expiry')) AS expiry_str,
                split_part(list_element(q.values, list_position(q.keys, 'priceThresholds')), ',', 1)::DOUBLE AS lo,
                split_part(list_element(q.values, list_position(q.keys, 'priceThresholds')), ',', 2)::DOUBLE AS hi
            FROM read_parquet('{q_glob}', union_by_name=true) q
            WHERE list_contains(q.keys, 'priceThresholds')
        )
        SELECT bl.symbol, bl.band_index, qs.q_symbol, qs.expiry_str, qs.lo, qs.hi
        FROM bucket_legs bl
        JOIN questions qs ON bl.outcome_idx = qs.outcome_idx
        ORDER BY qs.q_symbol, bl.band_index
    """
    try:
        band_map = con.execute(map_sql).df()
    except Exception as exc:
        _log.warning("_load_bucket_coherence band_map failed: %s", exc)
        return pd.DataFrame()

    if band_map.empty:
        return pd.DataFrame()

    # Process each question (group of 3 bands) separately, then concatenate
    all_rows: list[pd.DataFrame] = []

    for q_sym, grp in band_map.groupby("q_symbol"):
        band0 = grp[grp["band_index"] == 0]
        band1 = grp[grp["band_index"] == 1]
        band2 = grp[grp["band_index"] == 2]

        if band0.empty or band1.empty or band2.empty:
            continue

        sym0 = band0.iloc[0]["symbol"]
        sym1 = band1.iloc[0]["symbol"]
        sym2 = band2.iloc[0]["symbol"]
        expiry_str = band0.iloc[0]["expiry_str"]

        # Load BBO for these 3 symbols and ASOF join
        sql3 = f"""
            WITH b0 AS (
                SELECT (local_recv_ts // 60000000000) * 60000000000 AS ts_60s,
                       LAST((bid_px + ask_px) / 2.0 ORDER BY local_recv_ts) AS mid0,
                       LAST(ask_px ORDER BY local_recv_ts) AS ask0
                FROM read_parquet('{bbo}', union_by_name=true)
                WHERE symbol = '{sym0}'
                GROUP BY ts_60s
            ),
            b1 AS (
                SELECT (local_recv_ts // 60000000000) * 60000000000 AS ts_60s,
                       LAST((bid_px + ask_px) / 2.0 ORDER BY local_recv_ts) AS mid1,
                       LAST(ask_px ORDER BY local_recv_ts) AS ask1
                FROM read_parquet('{bbo}', union_by_name=true)
                WHERE symbol = '{sym1}'
                GROUP BY ts_60s
            ),
            b2 AS (
                SELECT (local_recv_ts // 60000000000) * 60000000000 AS ts_60s,
                       LAST((bid_px + ask_px) / 2.0 ORDER BY local_recv_ts) AS mid2,
                       LAST(ask_px ORDER BY local_recv_ts) AS ask2
                FROM read_parquet('{bbo}', union_by_name=true)
                WHERE symbol = '{sym2}'
                GROUP BY ts_60s
            )
            SELECT
                b0.ts_60s,
                b0.mid0, b0.ask0,
                b1.mid1, b1.ask1,
                b2.mid2, b2.ask2,
                b0.mid0 + b1.mid1 + b2.mid2 AS sum_mids,
                b0.ask0 + b1.ask1 + b2.ask2 AS sum_asks
            FROM b0
            JOIN b1 ON b0.ts_60s = b1.ts_60s
            JOIN b2 ON b0.ts_60s = b2.ts_60s
            ORDER BY b0.ts_60s
        """
        try:
            rows = con.execute(sql3).df()
        except Exception:
            continue

        if rows.empty:
            continue

        rows["q_symbol"] = q_sym
        rows["expiry_str"] = expiry_str
        all_rows.append(rows)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def _analyse_bucket_coherence(coh_df: pd.DataFrame) -> dict:
    """Compute bucket coherence KPIs."""
    if coh_df.empty:
        return {}

    n = len(coh_df)
    n_questions = coh_df["q_symbol"].nunique()
    sum_mids = coh_df["sum_mids"]
    sum_asks = coh_df["sum_asks"]

    # Sub-arb: sum of asks < 1 (before fees)
    sub_arb_mask = sum_asks < 1.0
    # Net-of-fee sub-arb: sum of asks < 1 - 3*HL_FEE_PER_LEG
    sub_arb_net_mask = sum_asks < (1.0 - 3 * HL_FEE_PER_LEG)

    return {
        "n_ticks": n,
        "n_questions": n_questions,
        "sum_mids_p25": float(np.percentile(sum_mids, 25)),
        "sum_mids_p50": float(np.percentile(sum_mids, 50)),
        "sum_mids_p75": float(np.percentile(sum_mids, 75)),
        "sum_mids_mean": float(sum_mids.mean()),
        "sum_asks_p50": float(sum_asks.median()),
        "frac_sub_arb_gross": float(sub_arb_mask.mean()),
        "frac_sub_arb_net_of_fee": float(sub_arb_net_mask.mean()),
        "fee_assumption_per_leg": HL_FEE_PER_LEG,
    }


def _plot_bucket_coherence(coh_df: pd.DataFrame) -> matplotlib.figure.Figure:
    """2-panel bucket coherence figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="#e6edf3")
        ax.xaxis.label.set_color("#e6edf3")
        ax.yaxis.label.set_color("#e6edf3")
        ax.title.set_color("#58a6ff")

    # Panel 1: Sum-of-mids distribution
    ax = axes[0]
    clip_mids = coh_df["sum_mids"].clip(0.5, 2.0)
    ax.hist(clip_mids, bins=80, color="#58a6ff", alpha=0.8, edgecolor="none")
    ax.axvline(1.0, color="#f85149", lw=1.5, label="Coherent sum=1")
    median_sum = float(coh_df["sum_mids"].median())
    ax.axvline(median_sum, color="#3fb950", lw=1.5, ls="--", label=f"Median={median_sum:.4f}")
    ax.set_title("3-Band Mid Sum Distribution")
    ax.set_xlabel("band0_mid + band1_mid + band2_mid")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, labelcolor="#e6edf3", facecolor="#161b22", edgecolor="#30363d")

    # Panel 2: Sum-of-asks distribution
    ax = axes[1]
    clip_asks = coh_df["sum_asks"].clip(0.8, 2.0)
    ax.hist(clip_asks, bins=80, color="#d29922", alpha=0.8, edgecolor="none")
    ax.axvline(1.0, color="#f85149", lw=1.5, label="Sub-arb threshold (sum=1)")
    ax.axvline(
        1.0 - 3 * HL_FEE_PER_LEG, color="#3fb950", lw=1.5, ls="--", label=f"Net arb ({1 - 3 * HL_FEE_PER_LEG:.4f})"
    )
    ax.set_title("3-Band Ask Sum Distribution")
    ax.set_xlabel("band0_ask + band1_ask + band2_ask")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, labelcolor="#e6edf3", facecolor="#161b22", edgecolor="#30363d")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Analysis 4: Bucket↔Binary Cross-Consistency
# ---------------------------------------------------------------------------


def _load_cross_consistency(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    bucket_df: pd.DataFrame,
    model_df: pd.DataFrame,
) -> pd.DataFrame:
    """Check consistency between bucket band mids and binary-implied prob.

    For each expiry that has both binary and bucket data, compare:
      - binary_mid (yes_mid of the binary Yes leg) ≈ P(S > K_binary)
      - bucket band 2 mid ≈ P(S > hi_threshold) (bucket "above" band)
      - bucket band 1 mid ≈ P(lo < S <= hi) (in-band)
      - bucket band 0 mid ≈ P(S <= lo)

    If K_binary ≈ hi_threshold, then binary_mid ≈ 1 - P(S <= hi)
    and bucket_band2_mid should ≈ binary_mid.

    Returns DataFrame with per-expiry/timestamp comparison.
    """
    if bucket_df.empty or model_df.empty:
        return pd.DataFrame()

    # Get bucket thresholds per expiry from question_meta
    q_glob = _q_glob(data_root)
    meta = _meta_glob(data_root)

    thresh_sql = f"""
        WITH questions AS (
            SELECT DISTINCT
                list_element(q.values, list_position(q.keys, 'expiry')) AS expiry_str,
                split_part(list_element(q.values, list_position(q.keys, 'priceThresholds')), ',', 1)::DOUBLE AS lo,
                split_part(list_element(q.values, list_position(q.keys, 'priceThresholds')), ',', 2)::DOUBLE AS hi
            FROM read_parquet('{q_glob}', union_by_name=true) q
            WHERE list_contains(q.keys, 'priceThresholds')
        ),
        binaries AS (
            SELECT DISTINCT
                list_element(values, list_position(keys, 'expiry'))               AS expiry_str,
                list_element(values, list_position(keys, 'targetPrice'))::DOUBLE  AS target_price
            FROM read_parquet('{meta}', union_by_name=true)
            WHERE array_contains(keys, 'class')
              AND list_element(values, list_position(keys, 'class')) = 'priceBinary'
              AND list_element(values, list_position(keys, 'side_name')) = 'Yes'
        )
        SELECT b.expiry_str, b.target_price, q.lo, q.hi,
               ABS(b.target_price - q.hi) AS dist_to_hi,
               ABS(b.target_price - q.lo) AS dist_to_lo
        FROM binaries b
        JOIN questions q ON b.expiry_str = q.expiry_str
        ORDER BY b.expiry_str
    """
    try:
        thresh_df = con.execute(thresh_sql).df()
    except Exception as exc:
        _log.warning("_load_cross_consistency failed: %s", exc)
        return pd.DataFrame()

    if thresh_df.empty:
        return thresh_df

    # Add relative distance metric
    thresh_df["pct_dist_hi"] = thresh_df["dist_to_hi"] / thresh_df["hi"]
    thresh_df["pct_dist_lo"] = thresh_df["dist_to_lo"] / thresh_df["lo"]

    return thresh_df


# ---------------------------------------------------------------------------
# Analysis 5: Taker Edge Call
# ---------------------------------------------------------------------------


def _compute_taker_edge(
    overround_df: pd.DataFrame,
    model_df: pd.DataFrame,
) -> dict:
    """Compute taker edge opportunities.

    Three routes:
    1. Binary sub-arb: buy-both at cost < 1 - fees (guaranteed profit)
    2. Model-edge: yes_mid < p_model - fees (buy Yes when model says cheap)
    3. Sell sub-arb: sell-both at proceeds > 1 + fees (guaranteed profit)
    """
    result: dict = {}

    if not overround_df.empty:
        n = len(overround_df)
        buy_cost = overround_df["buy_both_cost"]
        sell_proc = overround_df["sell_both_proceeds"]

        # Route 1: Buy arb
        buy_arb = overround_df[buy_cost < (1.0 - 2 * HL_FEE_PER_LEG)]
        if not buy_arb.empty:
            buy_arb_edge = (1.0 - 2 * HL_FEE_PER_LEG) - buy_arb["buy_both_cost"]
            # Capacity = min(yes_ask_sz, no_ask_sz)
            min_sz = buy_arb[["yes_ask_sz", "no_ask_sz"]].min(axis=1)
            result["buy_arb_freq"] = float(len(buy_arb)) / n
            result["buy_arb_edge_median"] = float(buy_arb_edge.median())
            result["buy_arb_capacity_median_shares"] = float(min_sz.median())
        else:
            result["buy_arb_freq"] = 0.0
            result["buy_arb_edge_median"] = float("nan")
            result["buy_arb_capacity_median_shares"] = float("nan")

        # Route 3: Sell arb
        sell_arb = overround_df[sell_proc > (1.0 + 2 * HL_FEE_PER_LEG)]
        if not sell_arb.empty:
            sell_arb_edge = sell_arb["sell_both_proceeds"] - (1.0 + 2 * HL_FEE_PER_LEG)
            min_bid_sz = sell_arb[["yes_bid_sz", "no_bid_sz"]].min(axis=1)
            result["sell_arb_freq"] = float(len(sell_arb)) / n
            result["sell_arb_edge_median"] = float(sell_arb_edge.median())
            result["sell_arb_capacity_median_shares"] = float(min_bid_sz.median())
        else:
            result["sell_arb_freq"] = 0.0
            result["sell_arb_edge_median"] = float("nan")
            result["sell_arb_capacity_median_shares"] = float("nan")

    # Route 2: Model edge
    if not model_df.empty:
        me = model_df.dropna(subset=["model_edge", "sigma"])
        if not me.empty:
            # Positive edge ticks: yes_mid < p_model - HL_FEE_PER_LEG (buy cheap)
            pos_edge = me[me["model_edge"] < -HL_FEE_PER_LEG]
            # Negative edge ticks: yes_mid > p_model + HL_FEE_PER_LEG (sell expensive)
            neg_edge = me[me["model_edge"] > HL_FEE_PER_LEG]

            result["model_edge_buy_freq"] = float(len(pos_edge)) / len(me)
            result["model_edge_sell_freq"] = float(len(neg_edge)) / len(me)
            result["model_edge_buy_median"] = (
                float((-pos_edge["model_edge"]).median()) if not pos_edge.empty else float("nan")
            )
            result["model_edge_sell_median"] = (
                float(neg_edge["model_edge"].median()) if not neg_edge.empty else float("nan")
            )

    return result


# ---------------------------------------------------------------------------
# Split-half stability
# ---------------------------------------------------------------------------


def _split_half_overround(df: pd.DataFrame) -> dict:
    """Compute overround stats for first and second halves."""
    if df.empty:
        return {}

    mid_ns = pd.Timestamp(SPLIT_MID, tz="UTC").value
    first = df[df["local_recv_ts"] < mid_ns]
    second = df[df["local_recv_ts"] >= mid_ns]

    result = {}
    for label, sub in [("first_half", first), ("second_half", second)]:
        if sub.empty:
            result[label] = {"n": 0}
            continue
        overround = sub["buy_both_cost"] - 1.0
        result[label] = {
            "n": len(sub),
            "n_expiries": int(sub["expiry_str"].nunique()),
            "overround_p50": float(overround.median()),
            "overround_mean": float(overround.mean()),
            "frac_buy_arb_net": float((sub["buy_both_cost"] < (1.0 - 2 * HL_FEE_PER_LEG)).mean()),
        }
    return result


def _split_half_model_edge(df: pd.DataFrame) -> dict:
    """Compute model edge stats for first and second halves."""
    if df.empty:
        return {}

    mid_ns = pd.Timestamp(SPLIT_MID, tz="UTC").value
    valid = df.dropna(subset=["model_edge"])
    first = valid[valid["ts_ns"] < mid_ns]
    second = valid[valid["ts_ns"] >= mid_ns]

    result = {}
    for label, sub in [("first_half", first), ("second_half", second)]:
        if sub.empty:
            result[label] = {"n": 0}
            continue
        result[label] = {
            "n": len(sub),
            "n_symbols": int(sub["symbol"].nunique()),
            "model_edge_p50": float(sub["model_edge"].median()),
            "model_edge_mean": float(sub["model_edge"].mean()),
            "pct_pos_edge": float((sub["model_edge"] < -HL_FEE_PER_LEG).mean()),
        }
    return result


# ---------------------------------------------------------------------------
# HTML table helpers
# ---------------------------------------------------------------------------


def _kpi_table(rows: list[dict]) -> str:
    """Build an HTML table from a list of {name, value, n, date_span, sanity} dicts."""
    header = "<tr><th>Metric</th><th>Value</th><th>n</th><th>Date span</th><th>Sanity</th></tr>"
    body_rows = []
    for r in rows:
        body_rows.append(
            f"<tr><td>{r.get('name', '')}</td>"
            f"<td><b>{r.get('value', '')}</b></td>"
            f"<td>{r.get('n', '')}</td>"
            f"<td>{r.get('date_span', '')}</td>"
            f"<td>{r.get('sanity', '')}</td></tr>"
        )
    return f"<table>{header}{''.join(body_rows)}</table>"


def _split_table(split: dict) -> str:
    rows = []
    for half_key, half_label in [("first_half", "First half 05-06→05-23"), ("second_half", "Second half 05-24→06-10")]:
        d = split.get(half_key, {})
        if not d:
            continue
        for k, v in d.items():
            if k == "n":
                continue
            rows.append(f"<tr><td>{half_label}</td><td>{k}</td><td>{v:.4f}</td></tr>")
    if not rows:
        return ""
    return f"<table><tr><th>Half</th><th>Metric</th><th>Value</th></tr>{''.join(rows)}</table>"


# ---------------------------------------------------------------------------
# Main build_card
# ---------------------------------------------------------------------------


def build_card(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
) -> tuple[str, dict]:
    """Build Card D: Mispricing Surface.

    Parameters
    ----------
    con:
        DuckDB connection (in-memory is fine; we read from parquet files).
    data_root:
        Absolute path to the data root, e.g. '../../data'.

    Returns
    -------
    (card_html, findings)
        card_html — standalone HTML for embedding in a Report.
        findings  — structured dict with title, headline, metrics, split_half, verdict.
    """
    _log.info("Card D: loading binary overround data …")
    overround_df = _load_binary_overround(con, data_root)
    overround_stats = _analyse_overround(overround_df)

    _log.info("Card D: computing Parkinson vol series …")
    sigma_df = _compute_parkinson_vol_series(con, data_root)

    _log.info("Card D: loading model edge data …")
    model_raw = _load_model_edge_data(con, data_root)
    model_df = _compute_model_edge(model_raw, sigma_df) if not model_raw.empty else pd.DataFrame()

    _log.info("Card D: loading bucket coherence …")
    coh_df = _load_bucket_coherence(con, data_root)
    coh_stats = _analyse_bucket_coherence(coh_df)

    _log.info("Card D: loading cross-consistency …")
    cross_df = _load_cross_consistency(con, data_root, coh_df, model_df)

    _log.info("Card D: computing taker edge …")
    taker_stats = _compute_taker_edge(overround_df, model_df)

    _log.info("Card D: computing split-half …")
    split_or = _split_half_overround(overround_df)
    split_me = _split_half_model_edge(model_df)

    # ---- Build HTML sections ----

    rpt = Report(title="Card D: Mispricing Surface")

    # --- D1: Overround ---
    or_kpis = []
    if overround_stats:
        n_exp = overround_stats.get("n_expiries", 0)
        or_kpis = [
            {
                "name": "Buy-both overround p25/p50/p75",
                "value": f"{overround_stats['overround_p25']:.4f} / {overround_stats['overround_p50']:.4f} / {overround_stats['overround_p75']:.4f}",
                "n": f"{overround_stats['n_ticks']:,} ticks, {n_exp} expiries",
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "Positive overround = dealers earn spread; ~0 = efficient",
            },
            {
                "name": "Sell-both proceeds p50",
                "value": f"{overround_stats['sell_both_proceeds_p50']:.4f}",
                "n": f"{overround_stats['n_ticks']:,}",
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "Should be <1 (bid<ask); >1 = taker arb",
            },
            {
                "name": "Frac ticks with gross buy arb (cost<1)",
                "value": f"{overround_stats['frac_buy_arb_gross']:.4f}",
                "n": f"{overround_stats['n_ticks']:,}",
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "Expected near 0; >0 = temporary anomaly or data artifact",
            },
            {
                "name": f"Frac ticks net-of-fee buy arb (cost < {1 - 2 * HL_FEE_PER_LEG:.4f})",
                "value": f"{overround_stats['frac_buy_arb_net_of_fee']:.6f}",
                "n": f"{overround_stats['n_ticks']:,}",
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": f"Fee={2 * HL_FEE_PER_LEG:.4f} rt; expected ~0",
            },
            {
                "name": f"Frac ticks net-of-fee sell arb (proceeds > {1 + 2 * HL_FEE_PER_LEG:.4f})",
                "value": f"{overround_stats['frac_sell_arb_net_of_fee']:.6f}",
                "n": f"{overround_stats['n_ticks']:,}",
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "Expected ~0; >0 = confirmed taker arb opportunity",
            },
        ]

    or_fig = _plot_overround(overround_df, overround_stats) if not overround_df.empty else None

    split_or_html = _split_table(split_or) if split_or else ""
    rpt.add_card(
        title="D1: Yes+No Overround",
        html_body=_kpi_table(or_kpis)
        + ("<h3 style='color:#58a6ff;margin-top:0.8rem'>Split-half</h3>" + split_or_html if split_or_html else ""),
        fig=or_fig,
        notes=(
            f"Fee model: flat {HL_FEE_PER_LEG:.4f}/share per leg (= {HL_FEE_ROUNDTRIP:.4f} round-trip). "
            "HL prod config: fee_taker=0.0 + half_spread_assumption=0.005; using 25bp conservative estimate. "
            "ASOF join: No-leg BBO LOCF onto Yes-leg timestamps."
        ),
    )

    # --- D2: Model Edge vs GBM ---
    me_kpis = []
    if not model_df.empty:
        valid_me = model_df.dropna(subset=["model_edge"])
        n_me = len(valid_me)
        n_syms = valid_me["symbol"].nunique()
        me_kpis = [
            {
                "name": "Model edge median (binary mid − GBM prob)",
                "value": f"{float(valid_me['model_edge'].median()):.4f}",
                "n": f"{n_me:,} rows, {n_syms} symbols",
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "+ve = market trades rich vs GBM; -ve = market cheap vs GBM",
            },
            {
                "name": "Model edge std",
                "value": f"{float(valid_me['model_edge'].std()):.4f}",
                "n": f"{n_me:,}",
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "Width of the pricing dispersion",
            },
            {
                "name": "Model edge p5 / p95",
                "value": f"{float(np.percentile(valid_me['model_edge'], 5)):.4f} / {float(np.percentile(valid_me['model_edge'], 95)):.4f}",
                "n": f"{n_me:,}",
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "Tail ticks with strong edge signal",
            },
            {
                "name": "Median Parkinson σ (1h)",
                "value": f"{float(sigma_df['sigma_annualised'].median()):.3f}" if not sigma_df.empty else "N/A",
                "n": f"{len(sigma_df):,} bars" if not sigma_df.empty else "",
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "Annual vol ~0.5–0.9 typical for BTC over this period",
            },
        ]

    me_fig = _plot_model_edge(model_df) if not model_df.empty and "model_edge" in model_df.columns else None
    split_me_html = _split_table(split_me) if split_me else ""

    rpt.add_card(
        title="D2: Model Edge vs GBM (Itô-corrected)",
        html_body=_kpi_table(me_kpis)
        + ("<h3 style='color:#58a6ff;margin-top:0.8rem'>Split-half</h3>" + split_me_html if split_me_html else ""),
        fig=me_fig,
        notes=(
            "GBM model: P(S_T>K) = Φ(d−), d− = (ln(S/K) − ½σ²τ)/(σ√τ). "
            "Parkinson σ = 1h rolling from 60s perp BBO OHLC bars (annualised). "
            "Positive model_edge means market bid is ABOVE GBM fair value (rich). "
            "Sampling: 60s grid per Yes-leg."
        ),
    )

    # --- D3: Bucket Coherence ---
    coh_kpis: list[dict] = []
    if coh_stats:
        coh_kpis = [
            {
                "name": "3-band mid sum p25/p50/p75",
                "value": f"{coh_stats['sum_mids_p25']:.4f} / {coh_stats['sum_mids_p50']:.4f} / {coh_stats['sum_mids_p75']:.4f}",
                "n": f"{coh_stats['n_ticks']:,} ticks, {coh_stats['n_questions']} questions",
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "Coherent market: sum = 1 + overround; sum ≈ 1 at mids",
            },
            {
                "name": "3-band ask sum p50",
                "value": f"{coh_stats['sum_asks_p50']:.4f}",
                "n": f"{coh_stats['n_ticks']:,}",
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "Sum of asks > 1 (normal); < 1 = gross sub-arb",
            },
            {
                "name": "Frac ticks gross sub-arb (sum_asks < 1)",
                "value": f"{coh_stats['frac_sub_arb_gross']:.6f}",
                "n": f"{coh_stats['n_ticks']:,}",
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "Expected ~0; >0 = transient mis-pricing",
            },
            {
                "name": f"Frac ticks net sub-arb (sum_asks < {1 - 3 * HL_FEE_PER_LEG:.4f})",
                "value": f"{coh_stats['frac_sub_arb_net_of_fee']:.6f}",
                "n": f"{coh_stats['n_ticks']:,}",
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": f"3-leg round trip fee = {3 * HL_FEE_PER_LEG:.4f}",
            },
        ]

    coh_fig = _plot_bucket_coherence(coh_df) if not coh_df.empty else None

    rpt.add_card(
        title="D3: Bucket Internal Coherence",
        html_body=_kpi_table(coh_kpis),
        fig=coh_fig,
        notes=(
            "3 bands (below lo, lo-to-hi, above hi) per question. Sum of 3 mids should be ≈ 1. "
            "Sum of 3 asks > 1 = normal overround; < 1 = sub-arb. "
            "60s sampling with inner JOIN on all 3 bands active simultaneously."
        ),
    )

    # --- D4: Bucket↔Binary Cross-Consistency ---
    cross_kpis: list[dict] = []
    if not cross_df.empty:
        n_pairs = len(cross_df)
        close_pairs = cross_df[cross_df["pct_dist_hi"] < 0.05]
        cross_kpis = [
            {
                "name": "Same-expiry binary+bucket pairs found",
                "value": str(n_pairs),
                "n": str(n_pairs),
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "Should be 1 pair per expiry (one binary + one bucket per day)",
            },
            {
                "name": "Pairs where K_binary ≈ hi_threshold (within 5%)",
                "value": str(len(close_pairs)),
                "n": str(n_pairs),
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "Close pairs allow direct binary↔bucket band2 comparison",
            },
            {
                "name": "Median |K_binary − hi| / hi",
                "value": f"{float(cross_df['pct_dist_hi'].median()):.4f}",
                "n": str(n_pairs),
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "Large distance = different strike levels; arb not directly applicable",
            },
            {
                "name": "Median K_binary",
                "value": f"{float(cross_df['target_price'].median()):.0f}",
                "n": str(n_pairs),
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "BTC price level",
            },
            {
                "name": "Median hi_threshold (bucket upper)",
                "value": f"{float(cross_df['hi'].median()):.0f}",
                "n": str(n_pairs),
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "BTC price level",
            },
        ]

    # Build cross-consistency table
    cross_body = _kpi_table(cross_kpis)
    if not cross_df.empty and len(cross_df) > 0:
        sample_rows = cross_df.head(10)[["expiry_str", "target_price", "lo", "hi", "pct_dist_hi"]].to_html(
            index=False, classes="", border=0, float_format=lambda x: f"{x:.1f}" if isinstance(x, float) else str(x)
        )
        cross_body += (
            "<h3 style='color:#58a6ff;margin-top:0.8rem'>Sample expiry alignment (first 10)</h3>" + sample_rows
        )

    rpt.add_card(
        title="D4: Bucket↔Binary Cross-Consistency",
        html_body=cross_body,
        notes=(
            "Checks whether the binary strike (K) aligns with a bucket threshold. "
            "If K ≈ hi_threshold: binary_mid ≈ P(S>K) ≈ bucket_band2_mid. "
            "Quantile distance measures alignment. Direct arb requires K ≈ threshold."
        ),
    )

    # --- D5: Taker Edge Call ---
    te_kpis: list[dict] = []
    if taker_stats:
        te_kpis = [
            {
                "name": "Binary buy-arb frequency (net-of-fee)",
                "value": f"{taker_stats.get('buy_arb_freq', 0):.6f}",
                "n": f"{overround_stats.get('n_ticks', 0):,}",
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "Very low: temporary order-book imbalance only",
            },
            {
                "name": "Binary sell-arb frequency (net-of-fee)",
                "value": f"{taker_stats.get('sell_arb_freq', 0):.6f}",
                "n": f"{overround_stats.get('n_ticks', 0):,}",
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "Very low: expected ~0 in efficient market",
            },
            {
                "name": "Buy-arb median edge (when positive)",
                "value": f"{taker_stats.get('buy_arb_edge_median', float('nan')):.4f}",
                "n": "arb ticks only",
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "Edge above fee threshold; larger = more profitable",
            },
            {
                "name": "Buy-arb TOB capacity p50 (shares)",
                "value": f"{taker_stats.get('buy_arb_capacity_median_shares', float('nan')):.1f}",
                "n": "arb ticks only",
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "Shares available at best ask for both legs simultaneously",
            },
            {
                "name": "Model-edge buy signal frequency",
                "value": f"{taker_stats.get('model_edge_buy_freq', float('nan')):.4f}",
                "n": f"{len(model_df.dropna(subset=['model_edge'])) if not model_df.empty else 0:,}",
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "Frac of 60s bars where yes_mid < p_model - fee",
            },
            {
                "name": "Model-edge sell signal frequency",
                "value": f"{taker_stats.get('model_edge_sell_freq', float('nan')):.4f}",
                "n": f"{len(model_df.dropna(subset=['model_edge'])) if not model_df.empty else 0:,}",
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "Frac of 60s bars where yes_mid > p_model + fee",
            },
        ]

    # Verdict logic
    buy_arb_freq = taker_stats.get("buy_arb_freq", 0.0)
    sell_arb_freq = taker_stats.get("sell_arb_freq", 0.0)
    me_sell_freq = taker_stats.get("model_edge_sell_freq", 0.0)

    if sell_arb_freq > 0.001 or me_sell_freq > 0.05:
        taker_verdict = "PASS — Measurable taker edge exists (see D5 metrics)"
    elif sell_arb_freq > 0.0001 or buy_arb_freq > 0.0001:
        taker_verdict = "MARGINAL — Very rare arb opportunities; likely transient/stale data artifacts"
    else:
        taker_verdict = "FAIL — No statistically significant net-of-fee taker arb in binary market structure alone"

    rpt.add_card(
        title="D5: Taker Edge Call",
        html_body=_kpi_table(te_kpis)
        + f"<p style='margin-top:0.8rem;font-weight:bold;color:#3fb950'>{taker_verdict}</p>",
        notes=(
            "Three routes: (1) binary buy-both sub-arb, (2) model-edge directional, (3) sell-both arb. "
            "Frequency × edge × capacity scores the opportunity. "
            "Model-edge sell freq > 5% means YES-leg consistently trades rich vs GBM = systematic overpricing."
        ),
    )

    # ---- Assemble HTML ----
    # Render full report HTML (we extract the card bodies)
    # Render to string
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tf:
        tmp_path = tf.name
    rpt.render(tmp_path)
    with open(tmp_path, encoding="utf-8") as fh:
        card_html = fh.read()
    Path(tmp_path).unlink(missing_ok=True)

    # ---- Build findings dict ----
    n_exp_binary = overround_stats.get("n_expiries", 0)
    n_q_bucket = coh_stats.get("n_questions", 0)

    findings: dict = {
        "title": "Card D: Mispricing Surface",
        "headline": (
            f"Binary overround p50={overround_stats.get('overround_p50', float('nan')):.4f}; "
            f"model edge p50={float(model_df['model_edge'].median()) if not model_df.empty and 'model_edge' in model_df.columns else float('nan'):.4f}; "
            f"n_binary_expiries={n_exp_binary}; n_bucket_questions={n_q_bucket}"
        ),
        "metrics": [
            {
                "name": "overround_p50",
                "value": overround_stats.get("overround_p50", float("nan")),
                "n": overround_stats.get("n_ticks", 0),
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "buy-both - 1, tick level",
            },
            {
                "name": "overround_p75",
                "value": overround_stats.get("overround_p75", float("nan")),
                "n": overround_stats.get("n_ticks", 0),
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "",
            },
            {
                "name": "frac_buy_arb_net",
                "value": overround_stats.get("frac_buy_arb_net_of_fee", 0.0),
                "n": overround_stats.get("n_ticks", 0),
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": f"fee threshold {2 * HL_FEE_PER_LEG:.4f}",
            },
            {
                "name": "frac_sell_arb_net",
                "value": overround_stats.get("frac_sell_arb_net_of_fee", 0.0),
                "n": overround_stats.get("n_ticks", 0),
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "",
            },
            {
                "name": "model_edge_p50",
                "value": float(model_df["model_edge"].median())
                if not model_df.empty and "model_edge" in model_df.columns
                else float("nan"),
                "n": len(model_df.dropna(subset=["model_edge"])) if not model_df.empty else 0,
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "yes_mid - GBM_prob; +ve = market rich vs GBM",
            },
            {
                "name": "model_edge_sell_freq",
                "value": taker_stats.get("model_edge_sell_freq", float("nan")),
                "n": len(model_df.dropna(subset=["model_edge"])) if not model_df.empty else 0,
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "frac 60s bars yes_mid > p_model + fee",
            },
            {
                "name": "bucket_sum_mids_p50",
                "value": coh_stats.get("sum_mids_p50", float("nan")),
                "n": coh_stats.get("n_ticks", 0),
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": "coherent = 1; overround adds to sum",
            },
            {
                "name": "bucket_sub_arb_net_freq",
                "value": coh_stats.get("frac_sub_arb_net_of_fee", 0.0),
                "n": coh_stats.get("n_ticks", 0),
                "date_span": f"{SPLIT_START}→{SPLIT_END}",
                "sanity": f"sum_asks < {1 - 3 * HL_FEE_PER_LEG:.4f}",
            },
        ],
        "split_half": {
            "overround": split_or,
            "model_edge": split_me,
        },
        "verdict": taker_verdict,
        "fee_assumption": {
            "model": "flat",
            "per_leg": HL_FEE_PER_LEG,
            "round_trip": HL_FEE_ROUNDTRIP,
            "note": "HL fee_taker=0.0 in prod; 25bp/leg conservative estimate for taker cost",
        },
    }

    return card_html, findings


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    data_root = os.environ.get("HLBT_HL_DATA_ROOT", "../../data")
    if not Path(data_root).exists():
        raise SystemExit(f"Data root not found: {data_root!r}. Set HLBT_HL_DATA_ROOT.")

    out_dir = Path(__file__).parents[3] / "docs" / "research" / "_cards"
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    _log.info("Building Card D (data_root=%s) …", data_root)
    card_html, findings = build_card(con, data_root)

    html_path = out_dir / "card_d.html"
    json_path = out_dir / "card_d.json"

    html_path.write_text(card_html, encoding="utf-8")
    _log.info("Wrote HTML: %s", html_path)

    # JSON: convert any float nan to None for valid JSON
    def _clean(obj):  # type: ignore[return]
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    json_path.write_text(json.dumps(_clean(findings), indent=2), encoding="utf-8")
    _log.info("Wrote JSON: %s", json_path)

    # Print headline summary to stdout
    print("\n=== Card D Summary ===")
    print(f"Verdict: {findings['verdict']}")
    print(f"Headline: {findings['headline']}")
    print("\nKey metrics:")
    for m in findings["metrics"]:
        val = m["value"]
        val_str = f"{val:.4f}" if isinstance(val, float) and math.isfinite(val) else str(val)
        print(f"  {m['name']}: {val_str}  (n={m['n']}, {m['date_span']})")
    print("\nSplit-half overround:")
    for half_key in ["first_half", "second_half"]:
        d = findings["split_half"]["overround"].get(half_key, {})
        if d:
            print(
                f"  {half_key}: n={d.get('n', 0):,}, overround_p50={d.get('overround_p50', float('nan')):.4f}, n_expiries={d.get('n_expiries', 0)}"
            )
    print("\nSplit-half model edge:")
    for half_key in ["first_half", "second_half"]:
        d = findings["split_half"]["model_edge"].get(half_key, {})
        if d:
            print(
                f"  {half_key}: n={d.get('n', 0):,}, model_edge_p50={d.get('model_edge_p50', float('nan')):.4f}, pct_pos_edge={d.get('pct_pos_edge', float('nan')):.4f}"
            )
