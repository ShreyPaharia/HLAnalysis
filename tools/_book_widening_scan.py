"""Scan: do the big sim entry/exit fills land on TRANSIENT outcome-book widenings?

For every sim fill in the veto-ON matrix, measure the leg's own book spread
(best_ask - best_bid) just-before / at / just-after the fill, plus how far the
fill price sat from the contemporaneous mid. Flag fills that hit a transient
widening (spread spikes at the fill, recovers within ~2 s).
"""
import glob
import duckdb
import pandas as pd

DATA = "/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data"
BOOK = (DATA + "/venue=hyperliquid/product_type=prediction_binary/mechanism=clob"
        "/event=book_snapshot")
con = duckdb.connect()

_book_cache = {}


def leg_book(sym):
    if sym in _book_cache:
        return _book_cache[sym]
    g = glob.glob(f"{BOOK}/symbol={sym}/date=*/**/*.parquet", recursive=True)
    if not g:
        _book_cache[sym] = None
        return None
    df = con.execute(
        f"select local_recv_ts as ts, bid_px, ask_px from read_parquet({g!r}) "
        f"order by ts"
    ).df()
    df["bb"] = df.bid_px.map(lambda a: float(a[0]) if hasattr(a, "__len__") and len(a) else None)
    df["ba"] = df.ask_px.map(lambda a: float(a[0]) if hasattr(a, "__len__") and len(a) else None)
    df = df[["ts", "bb", "ba"]].dropna()
    _book_cache[sym] = df
    return df


def nearest(df, ts):
    """snapshot nearest in time to ts (handles sim fill ts ~0.2s off the snap)."""
    i = (df.ts - ts).abs().values.argmin()
    return df.iloc[i]


def baseline_spread(df, ts, half=30e9):
    """median spread over [ts-30s, ts+30s] = the leg's 'normal' width here."""
    w = df[(df.ts >= ts - half) & (df.ts <= ts + half)]
    if not len(w):
        return None
    return float((w.ba - w.bb).median())


WIDE = 0.10        # spread considered "wide" for an HL favorite leg
SPIKE = 2.5        # spr_at >= SPIKE * baseline => localized widening (vs structural)

rows = []
for run in sorted(glob.glob("data/sim/runs/m_*/fills.parquet")):
    tag = run.split("/")[-2][2:]  # slot_kind_day
    f = pd.read_parquet(run)
    f = f[f.cloid != "settle"]
    for _, r in f.iterrows():
        sym = r["symbol"]
        db = leg_book(sym)
        if db is None or not len(db):
            continue
        ts = int(r["ts_ns"])
        cur = nearest(db, ts)
        spr_at = float(cur.ba - cur.bb)
        mid = float((cur.ba + cur.bb) / 2)
        base = baseline_spread(db, ts)
        dev = float(r["price"]) - mid
        # localized spike: this instant is much wider than the ±30s norm
        spike = base is not None and base > 0 and spr_at >= SPIKE * base and spr_at >= WIDE
        rows.append(dict(tag=tag, sym=sym, side=r["side"], px=float(r["price"]),
                         sz=float(r["size"]), spr_at=spr_at, base=base, mid=mid,
                         dev=dev, spike=bool(spike)))

R = pd.DataFrame(rows)
pd.set_option("display.width", 170)
print(f"total sim fills checked: {len(R)}")
wide = R[R.spr_at >= WIDE]
print(f"fills on a WIDE book (spread>={WIDE}): {len(wide)}  | of those, "
      f"LOCALIZED SPIKE (>= {SPIKE}x the ±30s baseline): {int(wide.spike.sum())}  | "
      f"STRUCTURAL (book just wide here): {int((~wide.spike).sum())}")
print(f"notional on wide-book fills: ${(wide.px*wide.sz).sum():,.0f} of "
      f"${(R.px*R.sz).sum():,.0f}\n")

print("=== per-leg book width (median spread over each run's fills) ===")
for tag, g in R.groupby("tag"):
    db = leg_book(g.sym.iloc[0])
    print(f"  {tag:20} leg {g.sym.iloc[0]:6} median_spread_at_fills={g.spr_at.median():.3f} "
          f"baseline≈{g.base.median():.3f}  wide_fills={int((g.spr_at>=WIDE).sum())}/{len(g)} "
          f"spikes={int(g.spike.sum())}")

print("\n=== wide-book fills, by run (SPIKE = localized; structural = book wide for seconds) ===")
for tag, grp in wide.groupby("tag"):
    print(f"\n-- {tag}  ({len(grp)} wide fills, ${(grp.px*grp.sz).sum():,.0f} ntl) --")
    for _, x in grp.iterrows():
        flag = "SPIKE" if x.spike else "structural"
        print(f"   {x.side:4} {x.sz:7.1f} @{x.px:.3f}  spread_at={x.spr_at:.3f} "
              f"base±30s={x.base:.3f}  mid={x.mid:.3f} dev={x.dev:+.3f}  {flag}")
