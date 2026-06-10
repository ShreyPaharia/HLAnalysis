"""Live-vs-sim FILL MICROSTRUCTURE — why are the bad fills bad, why is live
better, and is there still a cadence gap?

Per (leg, source) compares: fill count, cadence (inter-fill gap), reconstructed
max position, sell-clip sizes (dump magnitude), and the recorded book spread AT
each fill (does one side execute at tighter moments?).
"""
import glob
import duckdb
import numpy as np
import pandas as pd

DATA = "/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data"
BOOK = (DATA + "/venue=hyperliquid/product_type=prediction_binary/mechanism=clob"
        "/event=book_snapshot")
LIVE_CSV = "docs/research/2026-06-10-hl-live-fills-v1-v31-window.csv"
con = duckdb.connect()
NS = 1_000_000_000

# v31 legs where churn matters: (leg, kind, sim_run_day)
LEGS = [
    ("#1591", "binary", "0606"), ("#1610", "bucket", "0606"),
    ("#1640", "binary", "0607"), ("#1670", "bucket", "0607"),
    ("#2200", "binary", "0608"),
    ("#2250", "binary", "0609"), ("#2280", "bucket", "0609"),
]
SLOT = "v31"

_bk = {}
def book(sym):
    if sym in _bk:
        return _bk[sym]
    g = glob.glob(f"{BOOK}/symbol={sym}/date=*/**/*.parquet", recursive=True)
    df = con.execute(f"select local_recv_ts ts, bid_px, ask_px from read_parquet({g!r}) order by ts").df()
    df["bb"] = df.bid_px.map(lambda a: float(a[0]) if hasattr(a,"__len__") and len(a) else np.nan)
    df["ba"] = df.ask_px.map(lambda a: float(a[0]) if hasattr(a,"__len__") and len(a) else np.nan)
    df = df[["ts","bb","ba"]].dropna().reset_index(drop=True)
    _bk[sym] = df
    return df

def spread_at(sym, ts):
    df = book(sym)
    if not len(df):
        return np.nan
    i = (df.ts - ts).abs().values.argmin()
    return float(df.ba.iloc[i] - df.bb.iloc[i])

# ---- live fills ----
live = []
for ln in open(LIVE_CSV):
    if not ln.startswith("FILL,"):
        continue
    _, slot, ts, coin, dirn, side, px, sz, fee, cpnl, cloid = ln.rstrip("\n").split(",", 10)
    if slot != SLOT or dirn == "Settlement":
        continue
    live.append(dict(coin=coin, ts=int(ts), side=side, px=float(px), sz=float(sz), cpnl=float(cpnl)))
L = pd.DataFrame(live)

def metrics(df, sym):
    df = df.sort_values("ts").reset_index(drop=True)
    buys = df[df.side == "buy"]; sells = df[df.side == "sell"]
    # cadence: median gap between consecutive fills
    gaps = df.ts.diff().dropna() / NS
    cad = float(gaps.median()) if len(gaps) else np.nan
    # reconstruct running position (shares) -> max
    pos = 0.0; mx = 0.0
    for _, r in df.iterrows():
        pos += r.sz if r.side == "buy" else -r.sz
        mx = max(mx, pos)
    # spread at fills
    spr_buy = np.median([spread_at(sym, t) for t in buys.ts]) if len(buys) else np.nan
    spr_sell = np.median([spread_at(sym, t) for t in sells.ts]) if len(sells) else np.nan
    return dict(n=len(df), nb=len(buys), ns=len(sells),
                buy_px=float(buys.px.median()) if len(buys) else np.nan,
                sell_px=float(sells.px.median()) if len(sells) else np.nan,
                cad=cad, maxpos=mx,
                sell_med=float(sells.sz.median()) if len(sells) else np.nan,
                sell_max=float(sells.sz.max()) if len(sells) else np.nan,
                spr_buy=spr_buy, spr_sell=spr_sell)

print(f"{'leg':7} {'kind':6} {'src':4} | {'fl':>3} {'b':>3} {'s':>3} | "
      f"{'cad_s':>6} {'maxpos':>6} {'sellMed':>7} {'sellMax':>7} | "
      f"{'buyPx':>5} {'sellPx':>6} | {'sprBuy':>6} {'sprSell':>7}")
print("-" * 110)
for sym, kind, day in LEGS:
    lf = L[L.coin == sym]
    sf_path = glob.glob(f"data/sim/runs/m_{SLOT}_{kind}_{day}/fills.parquet")
    sf = pd.read_parquet(sf_path[0]) if sf_path else pd.DataFrame()
    sf = sf[(sf.symbol == sym) & (sf.cloid != "settle")].rename(columns={"ts_ns":"ts","price":"px","size":"sz"}) if len(sf) else sf
    for src, df in (("live", lf), ("sim", sf)):
        if not len(df):
            print(f"{sym:7} {kind:6} {src:4} |  (no fills)")
            continue
        m = metrics(df[["ts","side","px","sz"]], sym)
        print(f"{sym:7} {kind:6} {src:4} | {m['n']:3d} {m['nb']:3d} {m['ns']:3d} | "
              f"{m['cad']:6.1f} {m['maxpos']:6.0f} {m['sell_med']:7.0f} {m['sell_max']:7.0f} | "
              f"{m['buy_px']:.3f} {m['sell_px']:.3f} | {m['spr_buy']:6.3f} {m['spr_sell']:7.3f}")
    print()

print("cad_s = median seconds between consecutive fills (cadence). maxpos = peak shares held "
      "(inventory). sellMed/Max = sell clip size (dump magnitude). sprBuy/Sell = median recorded "
      "book spread at buy/sell fills.")
