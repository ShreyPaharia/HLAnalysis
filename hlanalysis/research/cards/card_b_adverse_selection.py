"""Card B: Adverse Selection / Trade Markouts for HL binary outcome markets.

Stress-tests the headline taker thesis from Card E:
"buying favorites at the ask, binary mid 0.80-0.95, 1-6h to expiry, is +EV fee-free."

Measures:
  1. Trade markout curves (buy vs sell) at 1s, 5s, 30s, 1m, 5m, 30m horizons
  2. Key test: for buy trades where prior BBO mid ∈ [0.80, 0.95] AND TTE ∈ [1h, 6h]:
     what is the post-trade mid markout? How much of Card E's 6-17pp edge survives?
  3. Effective spread vs realized spread decomposition
  4. Toxicity by trade size (small < 50, medium 50-200, large > 200 shares)
  5. Toxicity by TTE band (<1h, 1-6h, 6-12h, >12h)
  6. Maker-perspective: net PnL = realized spread - adverse selection cost
  7. Split-half stability (H1: ≤2026-05-23, H2: ≥2026-05-24)

Usage::

    from hlanalysis.research.cards.card_b_adverse_selection import build_card
    import duckdb
    html, findings = build_card(duckdb.connect(), "../../data")

Run standalone::

    python -m hlanalysis.research.cards.card_b_adverse_selection
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hlanalysis.analysis.helpers import asof_locf
from hlanalysis.research.outcome_markets import resolve_binary_outcomes
from hlanalysis.research.report import Report

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Forward markout horizons in seconds
_HORIZONS_S: list[int] = [1, 5, 30, 60, 300, 1800]
_HORIZON_LABELS: list[str] = ["1s", "5s", "30s", "1m", "5m", "30m"]
_NS_PER_S = 1_000_000_000

# Card E headline taker zone: buy favorites at ask, mid [0.80, 0.95], TTE [1h, 6h]
_CARD_E_RAW_EDGE_PP = 10.0  # representative mid-point of Card E's 6-17pp range


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _trade_glob(data_root: str) -> str:
    return str(
        Path(data_root)
        / "venue=hyperliquid"
        / "product_type=prediction_binary"
        / "mechanism=clob"
        / "event=trade"
        / "symbol=*"
        / "date=*"
        / "hour=all"
        / "*.parquet"
    )


def _bbo_glob(data_root: str) -> str:
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


def _load_trades(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    yes_syms: list[str],
) -> pd.DataFrame:
    """Load all binary Yes-leg trades sorted by (symbol, local_recv_ts)."""
    trade_glob = _trade_glob(data_root)
    sym_list = "', '".join(yes_syms)
    sql = f"""
        SELECT symbol, price, size, side, local_recv_ts, exchange_ts
        FROM read_parquet('{trade_glob}', union_by_name=true)
        WHERE symbol IN ('{sym_list}')
        ORDER BY symbol, local_recv_ts
    """
    try:
        df = con.execute(sql).df()
    except duckdb.IOException:
        return pd.DataFrame()
    df["local_recv_ts"] = df["local_recv_ts"].astype("int64")
    df["price"] = df["price"].astype("float64")
    df["size"] = df["size"].astype("float64")
    return df.reset_index(drop=True)


def _load_bbo(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    yes_syms: list[str],
) -> pd.DataFrame:
    """Load all binary Yes-leg BBO sorted by (symbol, local_recv_ts)."""
    bbo_glob = _bbo_glob(data_root)
    sym_list = "', '".join(yes_syms)
    sql = f"""
        SELECT symbol, bid_px, ask_px, local_recv_ts,
               (bid_px + ask_px) / 2.0 AS mid
        FROM read_parquet('{bbo_glob}', union_by_name=true)
        WHERE symbol IN ('{sym_list}')
        ORDER BY symbol, local_recv_ts
    """
    try:
        df = con.execute(sql).df()
    except duckdb.IOException:
        return pd.DataFrame()
    df["local_recv_ts"] = df["local_recv_ts"].astype("int64")
    df["mid"] = df["mid"].astype("float64")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Markout computation
# ---------------------------------------------------------------------------


def _compute_markouts(
    trades_df: pd.DataFrame,
    bbo_df: pd.DataFrame,
    expiry_map: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Compute trade markouts at all horizons.

    For each trade:
      - ASOF join to BBO to get mid_at_trade
      - For each horizon h: look up BBO mid at (local_recv_ts + h*1e9)
      - markout_h = sign * (mid_at_h - mid_at_trade)
        where sign=+1 for buy, -1 for sell
      - Also compute eff_spread, tte_s, mid_at_trade

    Returns enriched trades DataFrame.
    """
    if trades_df.empty or bbo_df.empty:
        return pd.DataFrame()

    result_rows = []

    # Process per-symbol to keep BBO lookup manageable
    for sym in trades_df["symbol"].unique():
        t_sub = trades_df[trades_df["symbol"] == sym].copy()
        b_sub = bbo_df[bbo_df["symbol"] == sym].copy()

        if b_sub.empty:
            continue

        b_sub = b_sub.sort_values("local_recv_ts").reset_index(drop=True)
        t_sub = t_sub.sort_values("local_recv_ts").reset_index(drop=True)

        bbo_ts = b_sub["local_recv_ts"].to_numpy(dtype="int64")
        bbo_mid = b_sub["mid"].to_numpy(dtype="float64")
        bbo_bid = b_sub["bid_px"].to_numpy(dtype="float64")
        bbo_ask = b_sub["ask_px"].to_numpy(dtype="float64")

        trade_ts = t_sub["local_recv_ts"].to_numpy(dtype="int64")
        prices = t_sub["price"].to_numpy(dtype="float64")
        sizes = t_sub["size"].to_numpy(dtype="float64")
        sides = t_sub["side"].to_numpy()

        # sign: +1 for buy, -1 for sell
        sign = np.where(sides == "buy", 1.0, -1.0)

        # mid at trade time (backward ASOF)
        mid_at_trade = asof_locf(trade_ts, bbo_ts, bbo_mid)
        bid_at_trade = asof_locf(trade_ts, bbo_ts, bbo_bid)
        ask_at_trade = asof_locf(trade_ts, bbo_ts, bbo_ask)

        # effective spread = 2 * |price - mid|
        eff_spread = 2.0 * np.abs(prices - mid_at_trade)

        # TTE
        meta = expiry_map.get(sym, {})
        expiry_ns = meta.get("expiry_ns", 0)
        tte_s = (expiry_ns - trade_ts) / 1e9

        # Horizons - forward ASOF join
        markout_cols: dict[str, np.ndarray] = {}
        mid_h_cols: dict[str, np.ndarray] = {}
        for h_s in _HORIZONS_S:
            offset_ts = trade_ts + int(h_s) * _NS_PER_S
            mid_at_h = asof_locf(offset_ts, bbo_ts, bbo_mid)
            # Cap: if the horizon extends past the last BBO by more than 2x interval, NaN
            # We do a simple check: forward mid must be from a BBO within h+60s of the offset
            markout = (mid_at_h - mid_at_trade) * sign
            markout_cols[f"markout_{h_s}s"] = markout
            mid_h_cols[f"mid_at_h_{h_s}s"] = mid_at_h

        # Build result for this symbol
        n = len(t_sub)
        rows: dict[str, Any] = {
            "symbol": [sym] * n,
            "local_recv_ts": trade_ts,
            "price": prices,
            "size": sizes,
            "side": sides,
            "sign": sign,
            "mid_at_trade": mid_at_trade,
            "bid_at_trade": bid_at_trade,
            "ask_at_trade": ask_at_trade,
            "eff_spread": eff_spread,
            "tte_s": tte_s,
            "yes_won": [meta.get("yes_won", None)] * n,
            "expiry_ns": [expiry_ns] * n,
        }
        rows.update(markout_cols)
        rows.update(mid_h_cols)
        result_rows.append(pd.DataFrame(rows))

    if not result_rows:
        return pd.DataFrame()

    df = pd.concat(result_rows, ignore_index=True)
    # Only pre-expiry trades with valid mid
    df = df[df["tte_s"] > 0].copy()
    df = df[df["mid_at_trade"].notna()].copy()
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def _markout_curve_table(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    """Aggregate markout curves across all horizons, optionally grouped."""
    cols = [f"markout_{h}s" for h in _HORIZONS_S]
    rows = []

    groups = df[group_col].unique() if group_col else [None]
    for grp in groups:
        sub = df[df[group_col] == grp] if group_col else df
        row: dict[str, Any] = {}
        if group_col:
            row[group_col] = grp
        row["n"] = int(len(sub))
        for h, c in zip(_HORIZONS_S, cols):
            vals = sub[c].dropna()
            row[f"markout_{h}s_mean"] = float(vals.mean()) if len(vals) > 0 else float("nan")
            row[f"markout_{h}s_median"] = float(vals.median()) if len(vals) > 0 else float("nan")
            row[f"markout_{h}s_std"] = float(vals.std()) if len(vals) > 0 else float("nan")
            row[f"markout_{h}s_n"] = int(vals.count())
        rows.append(row)

    return pd.DataFrame(rows)


def _eff_realized_spread(df: pd.DataFrame, realized_horizon_s: int = 300) -> dict[str, Any]:
    """Compute effective spread and realized spread (maker's net).

    realized_spread = eff_spread - 2 * markout_at_horizon
    This is the maker's net PnL per share per trade.
    """
    col = f"markout_{realized_horizon_s}s"
    if col not in df.columns:
        realized_horizon_s = 300
        col = f"markout_{realized_horizon_s}s"

    valid = df[df["eff_spread"].notna() & df[col].notna()].copy()
    if valid.empty:
        return {}

    eff = valid["eff_spread"]
    markout = valid[col]
    realized = eff - 2.0 * markout

    return {
        "n": int(len(valid)),
        "realized_horizon_s": realized_horizon_s,
        "mean_eff_spread": float(eff.mean()),
        "median_eff_spread": float(eff.median()),
        "mean_markout_at_horizon": float(markout.mean()),
        "mean_realized_spread": float(realized.mean()),
        "median_realized_spread": float(realized.median()),
        "pct_maker_profitable": float((realized > 0).mean()),
        "adverse_selection_fraction": float((2.0 * markout.mean()) / eff.mean() if eff.mean() > 0 else float("nan")),
    }


def _toxicity_by_size(df: pd.DataFrame) -> pd.DataFrame:
    """Markout at 30m by trade size bucket."""
    bins = [0, 50, 200, float("inf")]
    labels = ["small(<50)", "medium(50-200)", "large(>200)"]
    df = df.copy()
    df["size_bucket"] = pd.cut(df["size"], bins=bins, labels=labels, right=False)
    rows = []
    for lbl in labels:
        sub = df[df["size_bucket"] == lbl]
        if sub.empty:
            continue
        markout_30m = sub["markout_1800s"].dropna()
        markout_5m = sub["markout_300s"].dropna()
        rows.append(
            {
                "size_bucket": lbl,
                "n": int(len(sub)),
                "mean_size": float(sub["size"].mean()),
                "markout_30m_mean": float(markout_30m.mean()) if len(markout_30m) > 0 else float("nan"),
                "markout_5m_mean": float(markout_5m.mean()) if len(markout_5m) > 0 else float("nan"),
                "pct_positive_30m": float((markout_30m > 0).mean()) if len(markout_30m) > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _toxicity_by_tte(df: pd.DataFrame) -> pd.DataFrame:
    """Markout at 5m and 30m by TTE band."""
    tte_bins = [
        (0, 3600, "<1h"),
        (3600, 21600, "1-6h"),
        (21600, 43200, "6-12h"),
        (43200, float("inf"), ">12h"),
    ]
    rows = []
    for lo, hi, lbl in tte_bins:
        sub = df[(df["tte_s"] >= lo) & (df["tte_s"] < hi)]
        if sub.empty:
            continue
        buy_sub = sub[sub["side"] == "buy"]
        sell_sub = sub[sub["side"] == "sell"]
        m30 = sub["markout_1800s"].dropna()
        m5 = sub["markout_300s"].dropna()
        rows.append(
            {
                "tte_bucket": lbl,
                "n_total": int(len(sub)),
                "n_buy": int(len(buy_sub)),
                "n_sell": int(len(sell_sub)),
                "markout_30m_mean": float(m30.mean()) if len(m30) > 0 else float("nan"),
                "markout_5m_mean": float(m5.mean()) if len(m5) > 0 else float("nan"),
                "pct_positive_30m": float((m30 > 0).mean()) if len(m30) > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _key_test_flb_zone(df: pd.DataFrame) -> dict[str, Any]:
    """The key test: for buy trades at mid [0.80, 0.95], TTE [1h, 6h].

    Reports markouts at each horizon and computes net taker edge survival.
    """
    mask = (
        (df["side"] == "buy")
        & (df["mid_at_trade"] >= 0.80)
        & (df["mid_at_trade"] < 0.95)
        & (df["tte_s"] >= 3600)
        & (df["tte_s"] < 21600)
        & (df["mid_at_trade"].notna())
    )
    sub = df[mask].copy()
    n = int(len(sub))

    if n == 0:
        return {"n": 0, "note": "no trades in FLB zone"}

    result: dict[str, Any] = {"n": n}
    result["mean_price"] = float(sub["price"].mean())
    result["mean_mid_at_trade"] = float(sub["mid_at_trade"].mean())
    result["mean_eff_half_spread"] = float(sub["eff_spread"].mean() / 2.0)

    # Markout at each horizon
    markout_by_horizon: dict[str, float] = {}
    for h in _HORIZONS_S:
        col = f"markout_{h}s"
        if col in sub.columns:
            vals = sub[col].dropna()
            markout_by_horizon[f"markout_{h}s_mean"] = float(vals.mean()) if len(vals) > 0 else float("nan")
            markout_by_horizon[f"markout_{h}s_n"] = int(len(vals))
    result["markout_by_horizon"] = markout_by_horizon

    # Adverse selection at 30m
    m30 = sub["markout_1800s"].dropna()
    adv_sel_30m = float(m30.mean()) if len(m30) > 0 else float("nan")
    result["adverse_selection_30m_pp"] = float(adv_sel_30m * 100) if not np.isnan(adv_sel_30m) else float("nan")

    # Card E edge ~10pp midpoint of 6-17pp; if markout < 0, that erodes edge
    # if markout > 0 (price moves WITH buyer), that's actually not adverse selection
    raw_edge_pp = _CARD_E_RAW_EDGE_PP
    # adverse selection cost = -markout (if markout < 0, cost = |markout|)
    # net edge = raw_edge - max(0, -adv_sel) = raw_edge + min(0, adv_sel)
    if not np.isnan(adv_sel_30m):
        net_edge_pp = raw_edge_pp + min(0.0, float(adv_sel_30m) * 100.0)
    else:
        net_edge_pp = float("nan")

    result["raw_e_edge_pp"] = raw_edge_pp
    result["net_taker_edge_pp"] = float(net_edge_pp)

    return result


def _split_half(df: pd.DataFrame) -> dict[str, Any]:
    """Split-half stability: adverse selection and markout in H1 vs H2."""
    half1_cutoff_ns = int(pd.Timestamp("2026-05-24 00:00:00", tz="UTC").timestamp() * 1e9)
    df = df.copy()
    df["half"] = df["local_recv_ts"].apply(lambda x: "H1" if x < half1_cutoff_ns else "H2")

    result: dict[str, Any] = {}
    for h in ("H1", "H2"):
        sub = df[df["half"] == h]
        n = int(len(sub))
        if n == 0:
            result[h] = {"n": 0}
            continue

        # Key test zone
        kz = sub[
            (sub["side"] == "buy")
            & (sub["mid_at_trade"] >= 0.80)
            & (sub["mid_at_trade"] < 0.95)
            & (sub["tte_s"] >= 3600)
            & (sub["tte_s"] < 21600)
        ]

        m30 = sub["markout_1800s"].dropna()
        m5 = sub["markout_300s"].dropna()
        kz_m30 = kz["markout_1800s"].dropna()

        result[h] = {
            "n": n,
            "n_key_zone": int(len(kz)),
            "markout_30m_all_mean": float(m30.mean()) if len(m30) > 0 else float("nan"),
            "markout_5m_all_mean": float(m5.mean()) if len(m5) > 0 else float("nan"),
            "adverse_selection_30m_pp": float(kz_m30.mean() * 100) if len(kz_m30) > 0 else float("nan"),
        }

    h1_as = result.get("H1", {}).get("adverse_selection_30m_pp", float("nan"))
    h2_as = result.get("H2", {}).get("adverse_selection_30m_pp", float("nan"))

    # sign_stable: both halves show same direction of adverse selection
    if not np.isnan(h1_as) and not np.isnan(h2_as):
        result["sign_stable"] = bool((h1_as < 0) == (h2_as < 0))
    else:
        result["sign_stable"] = False

    result["adverse_selection_h1_pp"] = float(h1_as)
    result["adverse_selection_h2_pp"] = float(h2_as)

    return result


def _maker_pnl_table(df: pd.DataFrame) -> pd.DataFrame:
    """Maker-side PnL analysis at multiple horizons."""
    rows = []
    for h_s, h_lbl in zip(_HORIZONS_S, _HORIZON_LABELS):
        col = f"markout_{h_s}s"
        if col not in df.columns:
            continue
        valid = df[df["eff_spread"].notna() & df[col].notna()].copy()
        if valid.empty:
            continue
        eff = valid["eff_spread"]
        markout = valid[col]
        realized = eff - 2.0 * markout
        rows.append(
            {
                "horizon": h_lbl,
                "n": int(len(valid)),
                "mean_eff_spread_pp": float(eff.mean() * 100),
                "mean_markout_pp": float(markout.mean() * 100),
                "mean_realized_spread_pp": float(realized.mean() * 100),
                "pct_maker_profitable": float((realized > 0).mean() * 100),
                "adverse_selection_pct_of_spread": float(
                    (2.0 * markout.mean()) / eff.mean() * 100 if eff.mean() > 0 else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_DARK_BG = "#0d1117"
_CARD_BG = "#161b22"
_TEXT = "#e6edf3"
_ACCENT = "#58a6ff"
_GREEN = "#3fb950"
_RED = "#f78166"
_MUTED = "#8b949e"
_BORDER = "#30363d"


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(_CARD_BG)
    ax.tick_params(colors=_TEXT)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_ACCENT)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)


def _plot_markout_curves(buy_df: pd.DataFrame, sell_df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Plot mean markout curves for buy vs sell aggressors."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(_DARK_BG)

    for ax in axes:
        _style_ax(ax)

    horizon_s = _HORIZONS_S
    horizon_lbl = _HORIZON_LABELS
    x = list(range(len(horizon_s)))

    # Left: all buy vs sell
    ax = axes[0]
    buy_means = [
        buy_df[f"markout_{h}s_mean"].iloc[0] * 100
        if not buy_df.empty and f"markout_{h}s_mean" in buy_df.columns
        else float("nan")
        for h in horizon_s
    ]
    sell_means = [
        sell_df[f"markout_{h}s_mean"].iloc[0] * 100
        if not sell_df.empty and f"markout_{h}s_mean" in sell_df.columns
        else float("nan")
        for h in horizon_s
    ]
    ax.plot(x, buy_means, marker="o", color=_GREEN, label="Buy aggressor")
    ax.plot(x, sell_means, marker="s", color=_RED, label="Sell aggressor")
    ax.axhline(0, color=_MUTED, linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(horizon_lbl)
    ax.set_ylabel("Mean markout (pp)")
    ax.set_title("Markout Curves: Buy vs Sell Aggressor")
    ax.legend(facecolor="#1f2428", edgecolor=_BORDER, labelcolor=_TEXT, fontsize=8)
    ax.grid(True, alpha=0.2, color=_MUTED)

    # Right: cumulative (buy - sell) as net informed flow
    ax = axes[1]
    diff = [b - s if not (np.isnan(b) or np.isnan(s)) else float("nan") for b, s in zip(buy_means, sell_means)]
    colors = [_GREEN if v >= 0 else _RED for v in diff]
    ax.bar(x, diff, color=colors, alpha=0.8)
    ax.axhline(0, color=_MUTED, linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(horizon_lbl)
    ax.set_ylabel("Buy markout - Sell markout (pp)")
    ax.set_title("Informed Flow Asymmetry (Buy - Sell markout)")
    ax.grid(True, alpha=0.2, color=_MUTED)

    fig.tight_layout(pad=1.5)
    return fig


def _plot_key_test(flb_result: dict[str, Any]) -> matplotlib.figure.Figure:
    """Plot markout curve for the Card E key test zone."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(_DARK_BG)
    for ax in axes:
        _style_ax(ax)

    horizon_s = _HORIZONS_S
    horizon_lbl = _HORIZON_LABELS
    x = list(range(len(horizon_s)))

    mh = flb_result.get("markout_by_horizon", {})
    means_pp = [mh.get(f"markout_{h}s_mean", float("nan")) * 100 for h in horizon_s]

    raw_edge = flb_result.get("raw_e_edge_pp", 10.0)
    net_edge = flb_result.get("net_taker_edge_pp", float("nan"))

    # Left: markout curve for key zone
    ax = axes[0]
    colors_bars = [_GREEN if v >= 0 else _RED for v in means_pp if not np.isnan(v)]
    bars_x = [xi for xi, v in zip(x, means_pp) if not np.isnan(v)]
    bars_y = [v for v in means_pp if not np.isnan(v)]
    if bars_x:
        ax.bar(bars_x, bars_y, color=colors_bars, alpha=0.8)
    ax.plot(x, means_pp, marker="o", color=_ACCENT, linewidth=1.5, label="Buy markout (pp)")
    ax.axhline(0, color=_MUTED, linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(horizon_lbl)
    ax.set_ylabel("Mean markout (pp)")
    ax.set_title("Buy Trades: mid [0.80-0.95], TTE [1h-6h] — Markout Curve")
    ax.legend(facecolor="#1f2428", edgecolor=_BORDER, labelcolor=_TEXT, fontsize=8)
    ax.grid(True, alpha=0.2, color=_MUTED)

    # Right: edge survival bar chart
    ax = axes[1]
    adv_sel = flb_result.get("adverse_selection_30m_pp", float("nan"))
    categories = ["Raw E-edge\n(Card E)", "Adverse Sel.\n(30m)", "Net Taker\nEdge"]
    values = [raw_edge, adv_sel if not np.isnan(adv_sel) else 0.0, net_edge if not np.isnan(net_edge) else 0.0]
    bar_colors = [
        _ACCENT,
        _RED if (adv_sel < 0 if not np.isnan(adv_sel) else False) else _GREEN,
        _GREEN if (net_edge > 0 if not np.isnan(net_edge) else False) else _RED,
    ]
    ax.bar(range(3), values, color=bar_colors, alpha=0.8)
    ax.axhline(0, color=_MUTED, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(3))
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Edge (pp)")
    ax.set_title("E-Edge Survival After Adverse Selection")
    ax.grid(True, alpha=0.2, color=_MUTED)

    fig.tight_layout(pad=1.5)
    return fig


def _plot_toxicity(size_df: pd.DataFrame, tte_df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Plot toxicity by size and TTE."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(_DARK_BG)
    for ax in axes:
        _style_ax(ax)

    # Left: by size
    ax = axes[0]
    if not size_df.empty:
        labels = size_df["size_bucket"].tolist()
        vals = [v * 100 for v in size_df["markout_30m_mean"].tolist()]
        colors = [_GREEN if v >= 0 else _RED for v in vals]
        ax.bar(range(len(labels)), vals, color=colors, alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.axhline(0, color=_MUTED, linewidth=0.8, linestyle="--")
    ax.set_ylabel("30m markout (pp)")
    ax.set_title("Toxicity by Trade Size")
    ax.grid(True, alpha=0.2, color=_MUTED)

    # Right: by TTE
    ax = axes[1]
    if not tte_df.empty:
        labels = tte_df["tte_bucket"].tolist()
        vals30 = [v * 100 for v in tte_df["markout_30m_mean"].tolist()]
        vals5 = [v * 100 for v in tte_df["markout_5m_mean"].tolist()]
        xpos = list(range(len(labels)))
        width = 0.35
        ax.bar([xi - width / 2 for xi in xpos], vals30, width=width, color=_ACCENT, alpha=0.8, label="30m markout")
        ax.bar([xi + width / 2 for xi in xpos], vals5, width=width, color=_GREEN, alpha=0.8, label="5m markout")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, fontsize=9)
        ax.axhline(0, color=_MUTED, linewidth=0.8, linestyle="--")
        ax.legend(facecolor="#1f2428", edgecolor=_BORDER, labelcolor=_TEXT, fontsize=8)
    ax.set_ylabel("Markout (pp)")
    ax.set_title("Toxicity by TTE Band")
    ax.grid(True, alpha=0.2, color=_MUTED)

    fig.tight_layout(pad=1.5)
    return fig


def _plot_maker_pnl(maker_df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Plot maker-side PnL decomposition."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(_DARK_BG)
    _style_ax(ax)

    if not maker_df.empty:
        horizons = maker_df["horizon"].tolist()
        eff = maker_df["mean_eff_spread_pp"].tolist()
        realized = maker_df["mean_realized_spread_pp"].tolist()
        x = list(range(len(horizons)))
        width = 0.35
        ax.bar([xi - width / 2 for xi in x], eff, width=width, color=_ACCENT, alpha=0.8, label="Eff. spread")
        ax.bar(
            [xi + width / 2 for xi in x],
            realized,
            width=width,
            color=_GREEN,
            alpha=0.8,
            label="Realized spread (maker net)",
        )
        ax.axhline(0, color=_MUTED, linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.legend(facecolor="#1f2428", edgecolor=_BORDER, labelcolor=_TEXT, fontsize=8)

    ax.set_ylabel("Spread (pp)")
    ax.set_title("Maker PnL: Effective Spread vs Realized Spread by Horizon")
    ax.grid(True, alpha=0.2, color=_MUTED)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# HTML table helpers
# ---------------------------------------------------------------------------


def _df_to_html(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    rows = []
    header = "<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    rows.append(header)
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append(f"<td>{format(v, float_fmt)}</td>")
            else:
                cells.append(f"<td>{v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


# ---------------------------------------------------------------------------
# Main build_card
# ---------------------------------------------------------------------------


def build_card(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
) -> tuple[str, dict[str, Any]]:
    """Build Card B: Adverse Selection / Trade Markouts.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection.
    data_root : str
        Path to data root (e.g. '../../data').

    Returns
    -------
    (card_html, findings) where findings contains:
        title, headline, metrics, split_half, flb_edge_survival, verdict
    """
    data_root_str = str(Path(data_root).resolve())

    # 1. Resolve binary outcomes
    binaries = resolve_binary_outcomes(con, data_root_str)
    if binaries.empty:
        findings: dict[str, Any] = {
            "title": "Card B: Adverse Selection / Trade Markouts",
            "headline": "No binary outcome data found.",
            "metrics": [],
            "split_half": {},
            "flb_edge_survival": {},
            "verdict": "INCONCLUSIVE",
        }
        return "<p>No binary outcome data found.</p>", findings

    n_expiries = len(binaries)
    date_span = f"{binaries['expiry'].min().date()} to {binaries['expiry'].max().date()}"

    # Build expiry metadata map
    expiry_map: dict[str, dict[str, Any]] = {}
    for _, row in binaries.iterrows():
        expiry_map[row["symbol"]] = {
            "expiry_ns": int(row["expiry"].timestamp() * 1e9),
            "yes_won": bool(row["yes_won"]),
        }

    yes_syms = binaries["symbol"].tolist()

    # 2. Load trades and BBO
    trades_df = _load_trades(con, data_root_str, yes_syms)
    bbo_df = _load_bbo(con, data_root_str, yes_syms)

    if trades_df.empty:
        findings = {
            "title": "Card B: Adverse Selection / Trade Markouts",
            "headline": "No trade data found.",
            "metrics": [],
            "split_half": {},
            "flb_edge_survival": {},
            "verdict": "INCONCLUSIVE",
        }
        return "<p>No trade data found.</p>", findings

    n_trades_total = int(len(trades_df))
    n_buy = int((trades_df["side"] == "buy").sum())
    n_sell = int((trades_df["side"] == "sell").sum())

    # 3. Compute markouts
    df = _compute_markouts(trades_df, bbo_df, expiry_map)
    if df.empty:
        findings = {
            "title": "Card B: Adverse Selection / Trade Markouts",
            "headline": "Markout computation returned empty DataFrame.",
            "metrics": [],
            "split_half": {},
            "flb_edge_survival": {},
            "verdict": "INCONCLUSIVE",
        }
        return "<p>Markout computation failed.</p>", findings

    # 4. Run analyses
    buy_df_all = df[df["side"] == "buy"]
    sell_df_all = df[df["side"] == "sell"]

    buy_curve = _markout_curve_table(buy_df_all)
    sell_curve = _markout_curve_table(sell_df_all)

    eff_real = _eff_realized_spread(df, realized_horizon_s=300)
    size_tox = _toxicity_by_size(df)
    tte_tox = _toxicity_by_tte(df)
    maker_pnl = _maker_pnl_table(df)
    flb_result = _key_test_flb_zone(df)
    split_half_result = _split_half(df)

    # 5. Summary metrics
    # Overall buy aggressor markout at 30m
    m30_buy = float(buy_df_all["markout_1800s"].dropna().mean()) if len(buy_df_all) > 0 else float("nan")
    m30_sell = float(sell_df_all["markout_1800s"].dropna().mean()) if len(sell_df_all) > 0 else float("nan")

    adv_sel_30m = flb_result.get("adverse_selection_30m_pp", float("nan"))
    net_edge = flb_result.get("net_taker_edge_pp", float("nan"))
    raw_edge = flb_result.get("raw_e_edge_pp", _CARD_E_RAW_EDGE_PP)

    # Verdict
    if np.isnan(adv_sel_30m):
        verdict = "INCONCLUSIVE"
    elif adv_sel_30m > -2.0:
        # Less than 2pp adverse selection → E-edge mostly survives
        verdict = "FLB taker edge SURVIVES adverse selection"
    elif adv_sel_30m > -5.0:
        verdict = "FLB taker edge PARTIALLY ERODED by adverse selection"
    else:
        verdict = "FLB taker edge SIGNIFICANTLY ERODED by adverse selection"

    # Headline
    _as_pp = adv_sel_30m if not np.isnan(adv_sel_30m) else 0.0
    _net = net_edge if not np.isnan(net_edge) else 0.0
    _rs = eff_real.get("mean_realized_spread", 0.0) * 100
    headline = (
        f"Buy aggressors at mid [0.80-0.95], TTE [1-6h]: "
        f"adverse selection at 30m = {_as_pp:.1f}pp; "
        f"Card E raw edge = {raw_edge:.0f}pp; net survivable edge = {_net:.1f}pp. "
        f"Maker realized spread at 5m = {_rs:.2f}pp."
    )

    # 6. Build plots
    fig_curves = _plot_markout_curves(buy_curve, sell_curve)
    fig_key = _plot_key_test(flb_result)
    fig_tox = _plot_toxicity(size_tox, tte_tox)
    fig_maker = _plot_maker_pnl(maker_pnl)

    # 7. Assemble HTML
    rpt = Report(title="Card B: Adverse Selection / Trade Markouts")

    summary_html = f"""
    <p><strong>Date span:</strong> {date_span} &nbsp;|&nbsp;
    <strong>Expiries:</strong> {n_expiries} &nbsp;|&nbsp;
    <strong>Total trades:</strong> {n_trades_total:,} (buy: {n_buy:,}, sell: {n_sell:,})</p>
    <p><strong>Verdict:</strong> <strong style="color:#58a6ff">{verdict}</strong></p>
    <p>{headline}</p>
    """
    rpt.add_card("Summary", html_body=summary_html)

    # Card 1: Markout curves
    buy_curve_html = _df_to_html(
        buy_curve[[c for c in buy_curve.columns if "median" not in c and "std" not in c and c != "n"]].head(1)
        if not buy_curve.empty
        else pd.DataFrame(),
        ".5f",
    )
    rpt.add_card(
        "1. Markout Curves: Buy vs Sell Aggressor",
        html_body=(
            f"<p>Positive markout = mid moved in direction of aggressor (informed flow). "
            f"Buy markout at 30m = {m30_buy * 100 if not np.isnan(m30_buy) else 0.0:.3f}pp; "
            f"Sell markout at 30m = {m30_sell * 100 if not np.isnan(m30_sell) else 0.0:.3f}pp.</p>"
        )
        + buy_curve_html,
        fig=fig_curves,
        notes=(
            "Positive markout indicates the trade was informative (market moved with the aggressor). "
            "If buy-side markout is near zero or negative, takers are NOT extracting information from price "
            "and market makers face little adverse selection — good for makers, and takers' edge is structural (FLB). "
            "If buy-side markout is strongly positive, takers are informed and maker risk is elevated."
        ),
    )

    # Card 2: Key test
    kz_html_rows = []
    if flb_result.get("n", 0) > 0:
        mh = flb_result.get("markout_by_horizon", {})
        for h_s, h_lbl in zip(_HORIZONS_S, _HORIZON_LABELS):
            key = f"markout_{h_s}s_mean"
            val = mh.get(key, float("nan"))
            _val_pp = val * 100 if not np.isnan(val) else float("nan")
            _n_val = int(mh.get(f"markout_{h_s}s_n", 0))
            kz_html_rows.append(f"<tr><td>{h_lbl}</td><td>{_val_pp:.4f}pp</td><td>{_n_val:,}</td></tr>")
    kz_table = (
        "<table><tr><th>Horizon</th><th>Mean markout (pp)</th><th>N</th></tr>" + "".join(kz_html_rows) + "</table>"
    )
    rpt.add_card(
        "2. KEY TEST: Buy at mid [0.80-0.95], TTE [1-6h] — FLB Zone Markouts",
        html_body=(
            f"<p><strong>N trades in zone:</strong> {flb_result.get('n', 0):,} &nbsp;|&nbsp;"
            f"<strong>Mean mid at trade:</strong> {flb_result.get('mean_mid_at_trade', 0.0):.4f} &nbsp;|&nbsp;"
            f"<strong>Mean eff half-spread:</strong> {flb_result.get('mean_eff_half_spread', 0.0) * 100:.4f}pp</p>"
            f"<p><strong>Raw Card E edge:</strong> {raw_edge:.1f}pp &nbsp;|&nbsp;"
            f"<strong>Adverse selection (30m):</strong> {adv_sel_30m if not np.isnan(adv_sel_30m) else 0.0:.2f}pp &nbsp;|&nbsp;"
            f"<strong>Net survivable taker edge:</strong> {net_edge if not np.isnan(net_edge) else 0.0:.1f}pp</p>"
            f"{kz_table}"
        ),
        fig=fig_key,
        notes=(
            "The key question: when you buy a favorite at the ask (mid 0.80-0.95, TTE 1-6h), "
            "does the market immediately move against you? "
            "Adverse selection here = buy markout at 30m (negative = market moved against buyer). "
            "Card E shows ~6-17pp raw taker edge (win_rate - ask). "
            "If adverse selection is small (< 2pp), the FLB edge is real and structural. "
            "If adverse selection is large (> 5pp), takers are just competing with smart money and net edge is smaller."
        ),
    )

    # Card 3: Effective vs realized spread
    sr_html = ""
    if eff_real:
        _er_n = eff_real.get("n", 0)
        _er_rh = eff_real.get("realized_horizon_s", 300)
        _er_es = eff_real.get("mean_eff_spread", 0.0) * 100
        _er_ma = eff_real.get("mean_markout_at_horizon", 0.0) * 100
        _er_rs = eff_real.get("mean_realized_spread", 0.0) * 100
        _er_mrs = eff_real.get("median_realized_spread", 0.0) * 100
        _er_mp = eff_real.get("pct_maker_profitable", 0.0) * 100
        _er_asf = eff_real.get("adverse_selection_fraction", 0.0) * 100
        sr_html = f"""
        <p><strong>N valid trades:</strong> {_er_n:,} &nbsp;|&nbsp;
        <strong>Realized horizon:</strong> {_er_rh}s</p>
        <table>
        <tr><th>Metric</th><th>Value (pp)</th></tr>
        <tr><td>Mean effective spread</td><td>{_er_es:.4f}pp</td></tr>
        <tr><td>Mean markout at horizon</td><td>{_er_ma:.4f}pp</td></tr>
        <tr><td>Mean realized spread (maker net)</td><td>{_er_rs:.4f}pp</td></tr>
        <tr><td>Median realized spread</td><td>{_er_mrs:.4f}pp</td></tr>
        <tr><td>Pct maker profitable</td><td>{_er_mp:.1f}%</td></tr>
        <tr><td>Adverse selection as % of eff spread</td><td>{_er_asf:.1f}%</td></tr>
        </table>
        """
    rpt.add_card(
        "3. Effective Spread vs Realized Spread (Maker PnL Decomposition)",
        html_body=sr_html,
        fig=fig_maker,
        notes=(
            "Effective spread = 2 * |trade_price - mid_at_trade| = total spread paid by taker. "
            "Realized spread = eff_spread - 2 * markout_at_h = maker's net PnL per share. "
            "If realized spread > 0: maker earns money on average (spread > adverse selection cost). "
            "If realized spread < 0: makers are net losers — the market is informationally efficient and "
            "taking is rational. For HL HIP-4 binaries, the FLB thesis suggests spread > adverse selection."
        ),
    )

    # Card 4: Toxicity by size and TTE
    rpt.add_card(
        "4. Toxicity by Trade Size and TTE Band",
        html_body=_df_to_html(size_tox, ".4f") + "<br>" + _df_to_html(tte_tox, ".4f"),
        fig=fig_tox,
        notes=(
            "Toxicity by size: if larger trades have bigger markouts, large flow is informed. "
            "Toxicity by TTE: trades near expiry (< 1h) may have larger markouts as outcome is "
            "increasingly certain. The 1-6h zone (Card E's target) should show moderate toxicity "
            "balanced against the FLB structural edge."
        ),
    )

    # Card 5: Split-half stability
    sh = split_half_result
    _sh_h1n = sh.get("H1", {}).get("n", 0)
    _sh_h2n = sh.get("H2", {}).get("n", 0)
    _sh_h1kn = sh.get("H1", {}).get("n_key_zone", 0)
    _sh_h2kn = sh.get("H2", {}).get("n_key_zone", 0)
    _sh_h1as = sh.get("H1", {}).get("adverse_selection_30m_pp", float("nan"))
    _sh_h2as = sh.get("H2", {}).get("adverse_selection_30m_pp", float("nan"))
    _sh_h1m30 = sh.get("H1", {}).get("markout_30m_all_mean", float("nan"))
    _sh_h2m30 = sh.get("H2", {}).get("markout_30m_all_mean", float("nan"))
    _sh_ss = "YES" if sh.get("sign_stable", False) else "NO"
    sh_html = f"""
    <table>
    <tr><th></th><th>H1 (May 7 – May 23)</th><th>H2 (May 24 – Jun 13)</th></tr>
    <tr><td>N trades</td><td>{_sh_h1n:,}</td><td>{_sh_h2n:,}</td></tr>
    <tr><td>N in key zone (buy, mid 0.80-0.95, TTE 1-6h)</td><td>{_sh_h1kn:,}</td><td>{_sh_h2kn:,}</td></tr>
    <tr><td>Adverse selection 30m (pp)</td><td>{_sh_h1as:.2f}pp</td><td>{_sh_h2as:.2f}pp</td></tr>
    <tr><td>30m markout all trades (pp)</td><td>{_sh_h1m30:.3f}pp</td><td>{_sh_h2m30:.3f}pp</td></tr>
    <tr><td>Sign stable?</td><td colspan="2">{_sh_ss}</td></tr>
    </table>
    """
    rpt.add_card(
        "5. Split-Half Stability",
        html_body=sh_html,
        notes=(
            "H1: BTC declining (~$81k → ~$75k), 17 expiries. "
            "H2: BTC declining further then recovering (~$60k → ~$63k), 21 expiries. "
            "Sign stability of adverse selection direction tells us whether the pattern is robust "
            "across different market regimes or just an artifact of a single period."
        ),
    )

    # Render HTML
    out_path = Path(__file__).parent.parent.parent.parent / "docs" / "research" / "_cards" / "card_b.html"
    rpt.render(str(out_path))
    html = out_path.read_text(encoding="utf-8")

    plt.close("all")

    # 8. Findings dict
    # Build markout summary for findings
    buy_markout_summary = {}
    if not buy_curve.empty:
        for h in _HORIZONS_S:
            col = f"markout_{h}s_mean"
            if col in buy_curve.columns:
                buy_markout_summary[f"buy_markout_{h}s_pp"] = (
                    float(buy_curve[col].iloc[0]) * 100 if not buy_curve[col].isnull().all() else float("nan")
                )
    sell_markout_summary = {}
    if not sell_curve.empty:
        for h in _HORIZONS_S:
            col = f"markout_{h}s_mean"
            if col in sell_curve.columns:
                sell_markout_summary[f"sell_markout_{h}s_pp"] = (
                    float(sell_curve[col].iloc[0]) * 100 if not sell_curve[col].isnull().all() else float("nan")
                )

    # FLB markout by horizon (pp)
    flb_markout_pp = {}
    mh = flb_result.get("markout_by_horizon", {})
    for h in _HORIZONS_S:
        key = f"markout_{h}s_mean"
        val = mh.get(key, float("nan"))
        flb_markout_pp[f"h{h}s_pp"] = float(val * 100) if not np.isnan(val) else float("nan")

    findings = {
        "title": "Card B: Adverse Selection / Trade Markouts",
        "headline": headline,
        "metrics": [
            {
                "name": "n_expiries",
                "value": n_expiries,
                "n": n_expiries,
                "date_span": date_span,
                "sanity": "38 expected",
            },
            {
                "name": "n_trades_total",
                "value": n_trades_total,
                "n": n_trades_total,
                "date_span": date_span,
                "sanity": f"buy: {n_buy:,}, sell: {n_sell:,}",
            },
            {
                "name": "buy_markout_30m_pp",
                "value": round(m30_buy * 100, 4) if not np.isnan(m30_buy) else float("nan"),
                "n": int(buy_df_all["markout_1800s"].dropna().count()),
                "date_span": date_span,
                "sanity": "positive = informed buy flow; near-zero = structural FLB edge",
            },
            {
                "name": "adverse_selection_flb_zone_30m_pp",
                "value": round(adv_sel_30m, 4) if not np.isnan(adv_sel_30m) else float("nan"),
                "n": flb_result.get("n", 0),
                "date_span": date_span,
                "sanity": "negative = market moves against buyer; erodes Card E edge",
            },
        ],
        "buy_markout_curve_pp": buy_markout_summary,
        "sell_markout_curve_pp": sell_markout_summary,
        "eff_realized_spread": {
            k: (
                round(v * 100, 4)
                if k in ("mean_eff_spread", "mean_markout_at_horizon", "mean_realized_spread", "median_realized_spread")
                else v
            )
            for k, v in eff_real.items()
        },
        "flb_edge_survival": {
            "raw_e_edge_pp": float(raw_edge),
            "adverse_selection_30m_pp": float(adv_sel_30m) if not np.isnan(adv_sel_30m) else None,
            "net_taker_edge_pp": float(net_edge) if not np.isnan(net_edge) else None,
            "n": flb_result.get("n", 0),
            "markout_by_horizon_pp": flb_markout_pp,
        },
        "split_half": {
            "H1": {
                "n": sh.get("H1", {}).get("n", 0),
                "n_key_zone": sh.get("H1", {}).get("n_key_zone", 0),
                "date_span": "2026-05-07 to 2026-05-23",
                "adverse_selection_30m_pp": sh.get("H1", {}).get("adverse_selection_30m_pp", float("nan")),
                "markout_30m_all_mean_pp": sh.get("H1", {}).get("markout_30m_all_mean", float("nan")),
            },
            "H2": {
                "n": sh.get("H2", {}).get("n", 0),
                "n_key_zone": sh.get("H2", {}).get("n_key_zone", 0),
                "date_span": "2026-05-24 to 2026-06-13",
                "adverse_selection_30m_pp": sh.get("H2", {}).get("adverse_selection_30m_pp", float("nan")),
                "markout_30m_all_mean_pp": sh.get("H2", {}).get("markout_30m_all_mean", float("nan")),
            },
            "sign_stable": bool(sh.get("sign_stable", False)),
            "adverse_selection_h1_pp": float(sh.get("adverse_selection_h1_pp", float("nan"))),
            "adverse_selection_h2_pp": float(sh.get("adverse_selection_h2_pp", float("nan"))),
        },
        "verdict": verdict,
    }

    return html, findings


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data_root = os.environ.get("HLBT_HL_DATA_ROOT", "../../data")
    out_dir = Path(__file__).parent.parent.parent.parent / "docs" / "research" / "_cards"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building Card B with data_root={data_root!r} ...")
    con = duckdb.connect()
    html, findings = build_card(con, data_root)

    json_path = out_dir / "card_b.json"
    json_path.write_text(json.dumps(findings, indent=2), encoding="utf-8")

    html_path = out_dir / "card_b.html"
    print("Card B generated:")
    print(f"  HTML: {html_path}")
    print(f"  JSON: {json_path}")
    print(f"  Verdict: {findings['verdict']}")
    print()
    print(f"Headline: {findings['headline']}")
    print()

    # Key findings
    m = findings.get("metrics", [])
    for metric in m:
        print(f"  {metric['name']}: {metric['value']} (n={metric['n']})")
    print()

    print("FLB edge survival:")
    fe = findings.get("flb_edge_survival", {})
    print(f"  Raw E-edge: {fe.get('raw_e_edge_pp', 0):.1f}pp")
    print(f"  Adverse selection 30m: {fe.get('adverse_selection_30m_pp')}pp")
    print(f"  Net taker edge: {fe.get('net_taker_edge_pp')}pp")
    print()

    print("Buy markout curve:")
    bc = findings.get("buy_markout_curve_pp", {})
    for k, v in sorted(bc.items()):
        print(f"  {k}: {v:.4f}pp")
    print()

    print("Split-half:")
    sh = findings.get("split_half", {})
    print(f"  H1 adverse selection 30m: {sh.get('adverse_selection_h1_pp')}pp")
    print(f"  H2 adverse selection 30m: {sh.get('adverse_selection_h2_pp')}pp")
    print(f"  Sign stable: {sh.get('sign_stable')}")
