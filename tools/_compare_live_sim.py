"""Build the per-(day, slot, kind) live-vs-sim comparison (SHR-98 analysis).

Not a maintained tool — the one-off driver behind
docs/research/2026-06-10-hl-all-days-live-vs-sim-shr98.md.
"""
import glob
from datetime import datetime, timezone

import pandas as pd

LIVE_CSV = "docs/research/2026-06-10-hl-live-fills-v1-v31-window.csv"

# Settlement date -> (binary pair, [bucket pairs]) and window [start, end) ns
# (D-1 07:00 -> D 07:00 UTC). Each leg has a YES (#N) + NO/complement (#N1)
# member; the strategy buys whichever side is the favorite, so include both.
DAYS = {
    "0606": dict(binary=["#1590", "#1591"],
                 buckets=["#1610", "#1611", "#1620", "#1621", "#1630", "#1631"],
                 w=("2026-06-05T07:00:00Z", "2026-06-06T07:00:00Z")),
    "0607": dict(binary=["#1640", "#1641"],
                 buckets=["#1660", "#1661", "#1670", "#1671", "#1680", "#1681"],
                 w=("2026-06-06T07:00:00Z", "2026-06-07T07:00:00Z")),
    "0608": dict(binary=["#2200", "#2201"],
                 buckets=["#2220", "#2221", "#2230", "#2231", "#2240", "#2241"],
                 w=("2026-06-07T07:00:00Z", "2026-06-08T07:00:00Z")),
    "0609": dict(binary=["#2250", "#2251"],
                 buckets=["#2270", "#2271", "#2280", "#2281", "#2290", "#2291"],
                 w=("2026-06-08T07:00:00Z", "2026-06-09T07:00:00Z")),
}


def ns(iso):
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1e9)


# ---- parse live ----
live = []
for ln in open(LIVE_CSV):
    if not ln.startswith("FILL,"):
        continue
    _, slot, ts, coin, dirn, side, px, sz, fee, cpnl, cloid = ln.rstrip("\n").split(",", 10)
    live.append(dict(slot=slot, ts=int(ts), coin=coin, dir=dirn, side=side,
                     px=float(px), sz=float(sz), fee=float(fee), cpnl=float(cpnl)))
L = pd.DataFrame(live)


def live_cell(slot, legs, w):
    s, e = ns(w[0]), ns(w[1])
    df = L[(L.slot == slot) & (L.coin.isin(legs)) & (L.ts >= s) & (L.ts < e)]
    if df.empty:
        return 0, 0.0, 0.0
    orders = df[df["dir"] != "Settlement"]
    n = len(orders)
    buy_ntl = (orders[orders.side == "buy"].px * orders[orders.side == "buy"].sz).sum()
    pnl = (df.cpnl - df.fee).sum()
    return n, float(buy_ntl), float(pnl)


def sim_cell(slot, kind, day, prefix="m_"):
    fp = f"data/sim/runs/{prefix}{slot}_{kind}_{day}/fills.parquet"
    f = glob.glob(fp)
    if not f:
        return 0, 0.0, 0.0, "-"
    df = pd.read_parquet(f[0])
    if df.empty:
        return 0, 0.0, 0.0, "-"
    orders = df[df.cloid != "settle"]
    n = len(orders)
    b = orders[orders.side == "buy"]
    buy_ntl = float((b.price * b["size"]).sum())
    pnl = float(df.groupby("question_id").realized_pnl_at_settle.first().sum())
    legs = "/".join(sorted(df.symbol.unique()))
    return n, buy_ntl, pnl, legs


hdr = (f"{'day':5} {'slot':4} {'kind':7} | {'Lfll':>4} {'Lntl':>7} {'Lpnl':>8} | "
       f"{'fll':>4} {'ntl':>7} {'ON_pnl':>8} | {'fll':>4} {'OFFpnl':>8} | "
       f"{'dON':>7} {'dOFF':>7} | closer")
print(hdr)
print("-" * len(hdr))
rows = []
for day, d in DAYS.items():
    for slot in ("v1", "v31"):
        for kind in ("binary", "bucket"):
            legs = d["binary"] if kind == "binary" else d["buckets"]
            ln_, lntl, lpnl = live_cell(slot, legs, d["w"])
            on_n, on_ntl, on_pnl, slegs = sim_cell(slot, kind, day, "m_")
            of_n, of_ntl, of_pnl, _ = sim_cell(slot, kind, day, "off_")
            d_on = on_pnl - lpnl
            d_off = of_pnl - lpnl
            closer = "=" if abs(d_on) == abs(d_off) else ("ON" if abs(d_on) < abs(d_off) else "OFF")
            rows.append(dict(day=day, slot=slot, kind=kind, ln=ln_, lntl=lntl, lpnl=lpnl,
                             on_n=on_n, on_ntl=on_ntl, on_pnl=on_pnl,
                             of_n=of_n, of_pnl=of_pnl, d_on=d_on, d_off=d_off,
                             closer=closer, slegs=slegs))
            print(f"{day:5} {slot:4} {kind:7} | {ln_:4d} {lntl:7.0f} {lpnl:+8.2f} | "
                  f"{on_n:4d} {on_ntl:7.0f} {on_pnl:+8.2f} | {of_n:4d} {of_pnl:+8.2f} | "
                  f"{d_on:+7.2f} {d_off:+7.2f} | {closer}  {slegs}")

print("\n=== per (day,slot) totals: live | sim_ON | sim_OFF ===")
agg = {}
for r in rows:
    a = agg.setdefault((r["day"], r["slot"]), [0.0, 0.0, 0.0])
    a[0] += r["lpnl"]; a[1] += r["on_pnl"]; a[2] += r["of_pnl"]
for (day, slot), (lp, on, of) in sorted(agg.items()):
    print(f"{day} {slot:4} live {lp:+8.2f} | ON {on:+8.2f} (d {on-lp:+8.2f}) | "
          f"OFF {of:+8.2f} (d {of-lp:+8.2f})")

print("\n=== grand totals ===")
tl = sum(r["lpnl"] for r in rows)
ton = sum(r["on_pnl"] for r in rows)
tof = sum(r["of_pnl"] for r in rows)
print(f"LIVE {tl:+.2f} | SIM_ON {ton:+.2f} (d {ton-tl:+.2f}) | SIM_OFF {tof:+.2f} (d {tof-tl:+.2f})")
con_on = sum(abs(r["d_on"]) for r in rows)
con_off = sum(abs(r["d_off"]) for r in rows)
print(f"Σ|sim−live|  ON {con_on:.2f}   OFF {con_off:.2f}   (lower=closer to live)")
n_on = sum(1 for r in rows if r["closer"] == "ON")
n_off = sum(1 for r in rows if r["closer"] == "OFF")
n_eq = sum(1 for r in rows if r["closer"] == "=")
print(f"cells closer to live: ON={n_on}  OFF={n_off}  tie={n_eq}  (of {len(rows)})")
