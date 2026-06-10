"""Compare the SHR-79/89 IOC re-fire-floor arm (ioc_) vs the no-floor baseline
(base_) against live, reusing the SHR-98 analysis machinery. One-off driver.

Usage: uv run python tools/_compare_ioc.py [ON_prefix] [OFF_prefix]
  defaults: ON=ioc_  OFF=base_
"""
import glob
import sys
from datetime import datetime, timezone

import pandas as pd

LIVE_CSV = "docs/research/2026-06-10-hl-live-fills-v1-v31-window.csv"
ON_PREFIX = sys.argv[1] if len(sys.argv) > 1 else "ioc_"
OFF_PREFIX = sys.argv[2] if len(sys.argv) > 2 else "base_"

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
    pnl = (df.cpnl - df.fee).sum()
    return len(orders), 0.0, float(pnl)


def sim_cell(slot, kind, day, prefix):
    fp = f"data/sim/runs/{prefix}{slot}_{kind}_{day}/fills.parquet"
    f = glob.glob(fp)
    if not f:
        return None, 0, 0.0
    df = pd.read_parquet(f[0])
    if df.empty:
        return 0, 0, 0.0
    orders = df[df.cloid != "settle"]
    pnl = float(df.groupby("question_id").realized_pnl_at_settle.first().sum())
    return len(orders), len(orders), pnl


hdr = (f"{'day':5} {'slot':4} {'kind':7} | {'Lfll':>4} {'Lpnl':>8} | "
       f"{'IOCf':>4} {'IOCpnl':>8} | {'BASf':>4} {'BASpnl':>8} | "
       f"{'dIOC':>8} {'dBASE':>8} | closer")
print(hdr); print("-" * len(hdr))
rows = []
missing = 0
for day, d in DAYS.items():
    for slot in ("v1", "v31"):
        for kind in ("binary", "bucket"):
            legs = d["binary"] if kind == "binary" else d["buckets"]
            lf, _, lpnl = live_cell(slot, legs, d["w"])
            on_f, _, on_pnl = sim_cell(slot, kind, day, ON_PREFIX)
            of_f, _, of_pnl = sim_cell(slot, kind, day, OFF_PREFIX)
            if on_f is None or of_f is None:
                missing += 1
                print(f"{day:5} {slot:4} {kind:7} |  MISSING ({ON_PREFIX if on_f is None else OFF_PREFIX})")
                continue
            d_on, d_off = on_pnl - lpnl, of_pnl - lpnl
            closer = "=" if abs(d_on) == abs(d_off) else ("IOC" if abs(d_on) < abs(d_off) else "BASE")
            rows.append(dict(d_on=d_on, d_off=d_off, closer=closer, on_pnl=on_pnl, of_pnl=of_pnl, lpnl=lpnl))
            print(f"{day:5} {slot:4} {kind:7} | {lf:4d} {lpnl:+8.2f} | "
                  f"{on_f:4d} {on_pnl:+8.2f} | {of_f:4d} {of_pnl:+8.2f} | "
                  f"{d_on:+8.2f} {d_off:+8.2f} | {closer}")

if missing:
    print(f"\n[{missing} cell(s) missing — matrix still running?]")
print("\n=== grand totals ===")
tl = sum(r["lpnl"] for r in rows)
ton = sum(r["on_pnl"] for r in rows)
tof = sum(r["of_pnl"] for r in rows)
print(f"LIVE {tl:+.2f} | IOC {ton:+.2f} (Δ {ton-tl:+.2f}) | BASE {tof:+.2f} (Δ {tof-tl:+.2f})")
con_on = sum(abs(r["d_on"]) for r in rows)
con_off = sum(abs(r["d_off"]) for r in rows)
print(f"Σ|sim−live|  IOC {con_on:.2f}   BASE {con_off:.2f}   (lower=closer to live)")
print(f"   documented analysis baseline: veto-ON 946.07 / veto-OFF 960.21")
n_on = sum(1 for r in rows if r["closer"] == "IOC")
n_off = sum(1 for r in rows if r["closer"] == "BASE")
n_eq = sum(1 for r in rows if r["closer"] == "=")
print(f"cells closer to live: IOC={n_on}  BASE={n_off}  tie={n_eq}  (of {len(rows)})")
