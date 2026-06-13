"""Authoritative clean-FLB taker backtest computed directly (no live-strategy gates).

Rule (from Card E/B): for each binary leg, on the FIRST tick where the leg's mid is
in the favorite band and time-to-expiry is in the window, BUY the leg at the ask and
hold to oracle settlement. PnL per $1 = payoff(0/1) - entry_ask.

Labels: oracle price at expiry vs targetPrice (verified recipe). Yes-leg wins if
oracle>target; No-leg wins if oracle<=target.

Run from the worktree with HLBT_HL_DATA_ROOT=../../data (HL recorded venue data).
"""

from __future__ import annotations

import json
import os
import sys

import duckdb

DATA = os.environ.get("HLBT_HL_DATA_ROOT", "../../data")
PB = f"{DATA}/venue=hyperliquid/product_type=prediction_binary/mechanism=clob"
PERP = f"{DATA}/venue=hyperliquid/product_type=perp/mechanism=clob"

OOS_START = "2026-06-04"  # walk-forward holdout boundary
SPLIT = "2026-05-24"  # split-half boundary


def run(price_lo: float, price_hi: float, tte_min_h: float, tte_max_h: float) -> dict:
    con = duckdb.connect()
    con.execute("SET threads=4")
    q = f"""
    WITH mm AS (
      SELECT symbol, generate_subscripts(keys,1) i, keys, values
      FROM read_parquet('{PB}/event=market_meta/**/*.parquet', union_by_name=true)
    ),
    flat AS (SELECT symbol, keys[i] k, values[i] v FROM mm),
    meta AS (
      SELECT symbol,
        max(CASE WHEN k='outcome_name' THEN v END) oname,
        max(CASE WHEN k='side_name' THEN v END) side,
        max(CASE WHEN k='targetPrice' THEN v END)::DOUBLE tp,
        max(CASE WHEN k='expiry' THEN v END) expiry
      FROM flat GROUP BY symbol
    ),
    -- binary legs only (Recurring; buckets are 'Recurring Named Outcome')
    legs AS (
      SELECT symbol, side, tp, expiry,
        epoch_ns(strptime(expiry,'%Y%m%d-%H%M') AT TIME ZONE 'UTC') AS exp_ns
      FROM meta WHERE oname='Recurring' AND tp IS NOT NULL
    ),
    orc AS (SELECT local_recv_ts ts, oracle_px FROM read_parquet('{PERP}/event=oracle/symbol=BTC/**/*.parquet', union_by_name=true)),
    -- oracle at expiry -> per-leg payoff
    labeled AS (
      SELECT l.symbol, l.side, l.expiry, l.exp_ns, l.tp, o.oracle_px,
        CASE WHEN l.side='Yes' THEN (CASE WHEN o.oracle_px>l.tp THEN 1 ELSE 0 END)
             ELSE (CASE WHEN o.oracle_px<=l.tp THEN 1 ELSE 0 END) END AS payoff
      FROM legs l ASOF JOIN orc o ON l.exp_ns >= o.ts
    ),
    bbo AS (
      SELECT symbol, local_recv_ts ts, bid_px, ask_px, ask_sz, (bid_px+ask_px)/2 mid
      FROM read_parquet('{PB}/event=bbo/**/*.parquet', union_by_name=true)
      WHERE bid_px>0 AND ask_px>0
    ),
    -- candidate entry ticks: favorite band + TTE window
    cand AS (
      SELECT b.symbol, b.ts, b.ask_px, b.ask_sz, b.mid, l.payoff, l.exp_ns, l.expiry,
             (l.exp_ns - b.ts)/3.6e12 AS tte_h
      FROM bbo b JOIN labeled l ON b.symbol=l.symbol
      WHERE b.mid >= {price_lo} AND b.mid <= {price_hi}
        AND (l.exp_ns - b.ts)/3.6e12 BETWEEN {tte_min_h} AND {tte_max_h}
    ),
    -- first qualifying tick per leg
    entry AS (
      SELECT * FROM (
        SELECT *, row_number() OVER (PARTITION BY symbol ORDER BY ts) rn FROM cand
      ) WHERE rn=1
    )
    SELECT symbol, expiry, ask_px AS entry_ask, ask_sz, payoff, tte_h,
           (payoff - ask_px) AS pnl_per_dollar,
           strftime(make_timestamp((exp_ns/1000)::BIGINT)::TIMESTAMP, '%Y-%m-%d') AS exp_date
    FROM entry ORDER BY exp_ns
    """
    rows = con.execute(q).fetchall()
    cols = [d[0] for d in con.description]
    trades = [dict(zip(cols, r)) for r in rows]
    return trades


def summarize(trades: list[dict], label: str) -> dict:
    import statistics as st

    if not trades:
        return {"label": label, "n": 0}
    pnl = [t["pnl_per_dollar"] for t in trades]
    wins = [t for t in trades if t["payoff"] == 1]
    n = len(trades)
    total = sum(pnl)
    mean = total / n
    sd = st.pstdev(pnl) if n > 1 else 0.0
    # per-trade Sharpe (unitless, per-trade) and a rough daily-equivalent
    sharpe_trade = mean / sd if sd > 0 else float("nan")
    # max drawdown on cumulative equity (per $1 clips)
    cum = 0.0
    peak = 0.0
    mdd = 0.0
    for p in pnl:
        cum += p
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    return {
        "label": label,
        "n": n,
        "hit_rate": round(len(wins) / n, 4),
        "total_pnl_per_dollar": round(total, 4),
        "mean_edge_pp": round(mean * 100, 3),
        "edge_sd_pp": round(sd * 100, 3),
        "sharpe_per_trade": round(sharpe_trade, 3) if sd > 0 else None,
        "max_dd_per_dollar": round(mdd, 4),
        "mean_entry_ask": round(sum(t["entry_ask"] for t in trades) / n, 4),
        "mean_tob_ask_sz": round(sum((t["ask_sz"] or 0) for t in trades) / n, 1),
    }


def main() -> None:
    band = (0.80, 0.95, 1.0, 6.0)
    trades = run(*band)
    full = summarize(trades, "clean_FLB_full")
    is_t = [t for t in trades if t["exp_date"] < OOS_START]
    oos_t = [t for t in trades if t["exp_date"] >= OOS_START]
    h1 = [t for t in trades if t["exp_date"] < SPLIT]
    h2 = [t for t in trades if t["exp_date"] >= SPLIT]
    out = {
        "rule": {
            "price_band": [band[0], band[1]],
            "tte_window_h": [band[2], band[3]],
            "entry": "buy favorite leg at ask, first qualifying tick, hold to oracle settlement",
            "fee": "HL HIP-4 exchange fee = 0; cost = half-spread (entry at ask)",
        },
        "full": full,
        "in_sample": summarize(is_t, f"IS (<{OOS_START})"),
        "out_of_sample": summarize(oos_t, f"OOS (>={OOS_START})"),
        "split_h1": summarize(h1, f"H1 (<{SPLIT})"),
        "split_h2": summarize(h2, f"H2 (>={SPLIT})"),
    }
    # sweep a few bands for robustness
    sweep = []
    for lo, hi in [(0.80, 0.90), (0.85, 0.95), (0.80, 0.95), (0.90, 0.97), (0.65, 0.80)]:
        for tmin, tmax in [(1.0, 6.0), (0.5, 12.0), (3.0, 6.0)]:
            s = summarize(run(lo, hi, tmin, tmax), f"[{lo},{hi}]@{tmin}-{tmax}h")
            sweep.append(s)
    out["band_sweep"] = sweep
    print(json.dumps(out, indent=2))
    with open("docs/research/_cards/clean_flb_authoritative.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
