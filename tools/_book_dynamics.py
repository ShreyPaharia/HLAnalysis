"""Characterize the TEMPORAL dynamics of HL bucket-leg book illiquidity.

Question: when a bucket book is wide, is it wide for LONG stretches (=> the
market is structurally illiquid, don't trade it) or just brief seconds (=> trade
but wait out / dynamically widen)? Time-weighted (each snapshot's spread holds
until the next), so gaps don't bias the picture.
"""
import glob
import duckdb
import numpy as np
import pandas as pd

DATA = "/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data"
BOOK = (DATA + "/venue=hyperliquid/product_type=prediction_binary/mechanism=clob"
        "/event=book_snapshot")
con = duckdb.connect()

# (label, leg, window start/end ns)  — settle-day market, D-1 07:00 -> D 06:00 settle
NS = 1_000_000_000
def ns(iso):
    from datetime import datetime, timezone
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * NS)

LEGS = [
    ("0606 bucket #1610", "#1610", "2026-06-05T07:00:00Z", "2026-06-06T06:00:00Z"),
    ("0607 bucket #1670", "#1670", "2026-06-06T07:00:00Z", "2026-06-07T06:00:00Z"),
    ("0608 bucket #2230", "#2230", "2026-06-07T07:00:00Z", "2026-06-08T06:00:00Z"),
    ("0609 bucket #2280", "#2280", "2026-06-08T07:00:00Z", "2026-06-09T06:00:00Z"),
    ("0608 BINARY #2200", "#2200", "2026-06-07T07:00:00Z", "2026-06-08T06:00:00Z"),
]

WIDE = 0.10        # spread above this = not cheaply tradeable (round-trip too costly)
MAXGAP = 60 * NS   # cap a single snapshot's held-duration at 60s (feed-gap guard)


def load(sym, s, e):
    g = glob.glob(f"{BOOK}/symbol={sym}/date=*/**/*.parquet", recursive=True)
    df = con.execute(
        f"select local_recv_ts as ts, bid_px, ask_px from read_parquet({g!r}) "
        f"where local_recv_ts between {s} and {e} order by ts"
    ).df()
    df["bb"] = df.bid_px.map(lambda a: float(a[0]) if hasattr(a, "__len__") and len(a) else np.nan)
    df["ba"] = df.ask_px.map(lambda a: float(a[0]) if hasattr(a, "__len__") and len(a) else np.nan)
    df = df[["ts", "bb", "ba"]].dropna().reset_index(drop=True)
    df["spread"] = df.ba - df.bb
    # held-duration of each snapshot (until next), capped
    dt = df.ts.shift(-1) - df.ts
    dt = dt.clip(upper=MAXGAP).fillna(0)
    df["dt"] = dt
    return df


def episodes(df, thr):
    """contiguous runs where spread > thr; return list of durations (seconds)."""
    wide = (df.spread > thr).values
    durs, cur = [], 0.0
    for i in range(len(df)):
        if wide[i]:
            cur += df.dt.iloc[i]
        elif cur > 0:
            durs.append(cur / NS); cur = 0.0
    if cur > 0:
        durs.append(cur / NS)
    return durs


print(f"{'leg':20} {'cover':>6} {'med_spr':>7} {'p90_spr':>7} | "
      f"{'%t<0.05':>7} {'%t≥0.10':>7} {'%t≥0.20':>7} | "
      f"{'#wide':>5} {'medEp':>6} {'p90Ep':>6} {'maxEp':>7} {'%wide>60s':>9}")
print("-" * 120)
for label, sym, si, ei in LEGS:
    df = load(sym, ns(si), ei and ns(ei))
    tot = df.dt.sum()
    if tot <= 0:
        print(f"{label:20}  no coverage"); continue
    cover_h = tot / NS / 3600
    # time-weighted spread quantiles
    order = df.sort_values("spread")
    cw = order.dt.cumsum() / tot
    med = float(order.spread.iloc[(cw >= 0.5).values.argmax()])
    p90 = float(order.spread.iloc[(cw >= 0.9).values.argmax()])
    pct_tight = float(df.loc[df.spread < 0.05, "dt"].sum() / tot * 100)
    pct_w10 = float(df.loc[df.spread >= 0.10, "dt"].sum() / tot * 100)
    pct_w20 = float(df.loc[df.spread >= 0.20, "dt"].sum() / tot * 100)
    eps = episodes(df, WIDE)
    if eps:
        eps_s = sorted(eps)
        medEp = eps_s[len(eps_s) // 2]
        p90Ep = eps_s[min(len(eps_s) - 1, int(len(eps_s) * 0.9))]
        maxEp = eps_s[-1]
        widetime = sum(eps)
        long_frac = sum(d for d in eps if d > 60) / widetime * 100 if widetime else 0
    else:
        medEp = p90Ep = maxEp = long_frac = 0.0
    print(f"{label:20} {cover_h:5.1f}h {med:7.3f} {p90:7.3f} | "
          f"{pct_tight:6.1f}% {pct_w10:6.1f}% {pct_w20:6.1f}% | "
          f"{len(eps):5d} {medEp:5.1f}s {p90Ep:5.1f}s {maxEp:6.0f}s {long_frac:8.1f}%")

print("\nLegend: %t<0.05 = share of WINDOW TIME the book is cheaply tradeable; "
      "%t≥0.10/0.20 = share wide; medEp/p90Ep/maxEp = wide-episode durations; "
      "%wide>60s = share of total wide-time spent in episodes longer than 60s.")

print("\n=== time-weighted MEDIAN spread by hour-into-window (is liquidity a clean window?) ===")
print(f"{'leg':20} " + " ".join(f"h{h:02d}" for h in range(0, 23, 2)))
for label, sym, si, ei in LEGS:
    df = load(sym, ns(si), ns(ei))
    if df.dt.sum() <= 0:
        continue
    start = ns(si)
    df["hr"] = ((df.ts - start) // (3600 * NS)).astype(int)
    cells = []
    for h in range(0, 23, 2):
        seg = df[df.hr == h]
        if seg.dt.sum() <= 0:
            cells.append("  . "); continue
        o = seg.sort_values("spread"); cw = o.dt.cumsum() / o.dt.sum()
        m = float(o.spread.iloc[(cw >= 0.5).values.argmax()])
        cells.append(f"{m:.2f}")
    print(f"{label:20} " + "  ".join(cells))
