"""Build the analysis notebooks under docs/reports/ from inline cell definitions.

Re-running this overwrites the notebooks. Edit the cell strings below, not the
.ipynb files directly. Each notebook is a list of (cell_type, source) tuples.

Run:
    .venv/bin/python tools/build_notebooks.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "reports"
OUT.mkdir(parents=True, exist_ok=True)


def _build(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python"}
    nb_cells = []
    for kind, src in cells:
        if kind == "md":
            nb_cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))
        else:
            nb_cells.append(nbf.v4.new_code_cell(src.strip("\n")))
    nb.cells = nb_cells
    return nb


# ---------------------------------------------------------------------------
# 01 — P0 Data Quality
# ---------------------------------------------------------------------------
P0 = [
    (
        "md",
        """
# 01 — Data Quality (P0)

**What this notebook answers**

- Are all expected venue × product × event streams flowing?
- What is the per-stream message rate? Any silent gaps?
- How big is the local-vs-exchange clock skew? Does it look stable?
- Schema sanity: NaN/null fields, duplicate sequence numbers, weird prices.

This is the gate-keeper. If anything below looks wrong, **do not** trust higher-phase notebooks until fixed.
""",
    ),
    (
        "code",
        """
from hlanalysis.analysis import duck, glob_for, load_df, set_mpl_defaults, fmt_ts
import pandas as pd, numpy as np, matplotlib.pyplot as plt

set_mpl_defaults()
ALL = glob_for()  # everything
con = duck()
print('parquet root :', glob_for().split('venue=')[0])
""",
    ),
    (
        "md",
        """
## 1. Coverage matrix

Row counts and time-bounds per (venue, product_type, event, symbol). Look for unexpected zeros and short windows.
""",
    ),
    (
        "code",
        """
cov = load_df(f'''
SELECT venue, product_type, event, symbol,
       count(*)                          AS rows,
       min(exchange_ts)                  AS t0_ns,
       max(exchange_ts)                  AS t1_ns,
       (max(exchange_ts)-min(exchange_ts))/1e9/3600.0 AS hours
FROM read_parquet('{ALL}', hive_partitioning=true)
WHERE event NOT IN ('health','market_meta')
GROUP BY 1,2,3,4
ORDER BY 1,2,3,4
''')
cov['t0'] = cov.t0_ns.map(fmt_ts)
cov['t1'] = cov.t1_ns.map(fmt_ts)
cov['rows_per_min'] = cov['rows'] / (cov['hours']*60).clip(lower=1e-9)
cov[['venue','product_type','event','symbol','rows','rows_per_min','hours','t0','t1']]
""",
    ),
    (
        "md",
        """
## 2. Per-minute message rate (top streams)

A flat curve = healthy. Sudden cliffs = WS reconnects, geo-blocks, or exchange-side incidents. Compare visually across venues.
""",
    ),
    (
        "code",
        """
streams = [
    ('hyperliquid','perp','trade','BTC'),
    ('hyperliquid','perp','bbo','BTC'),
    ('binance','perp','trade','BTCUSDT'),
    ('binance','perp','bbo','BTCUSDT'),
    ('binance','spot','trade','BTCUSDT'),
    ('hyperliquid','prediction_binary','bbo','#30'),
    ('hyperliquid','prediction_binary','bbo','#31'),
]

fig, axes = plt.subplots(len(streams), 1, figsize=(11, 1.4*len(streams)), sharex=True)
for ax, (v, p, e, s) in zip(axes, streams):
    g = glob_for(venue=v, product_type=p, event=e, symbol=s)
    df = load_df(f'''
        SELECT date_trunc('minute', to_timestamp(exchange_ts/1e9)) AS minute,
               count(*) AS msgs
        FROM read_parquet('{g}', hive_partitioning=true)
        GROUP BY 1 ORDER BY 1
    ''')
    if df.empty:
        ax.set_title(f'{v}/{p}/{e}/{s}  (no data)', loc='left'); continue
    ax.plot(df.minute, df.msgs)
    ax.set_title(f'{v}/{p}/{e}/{s}  ({df.msgs.sum():,} msgs)', loc='left')
    ax.set_ylabel('msgs/min')
plt.tight_layout(); plt.show()
""",
    ),
    (
        "md",
        """
## 3. Inter-arrival gaps

For each high-frequency stream, the 99.9th percentile inter-arrival should be << 5s. Anything bigger is a stall to investigate.
""",
    ),
    (
        "code",
        """
def gap_stats(venue, product_type, event, symbol):
    g = glob_for(venue=venue, product_type=product_type, event=event, symbol=symbol)
    df = load_df(f'''
      WITH t AS (
        SELECT exchange_ts FROM read_parquet('{g}', hive_partitioning=true)
        ORDER BY exchange_ts
      ),
      d AS (
        SELECT (exchange_ts - lag(exchange_ts) OVER ()) / 1e6 AS gap_ms FROM t
      )
      SELECT
        count(*) AS n,
        approx_quantile(gap_ms, 0.50) AS p50_ms,
        approx_quantile(gap_ms, 0.99) AS p99_ms,
        approx_quantile(gap_ms, 0.999) AS p999_ms,
        max(gap_ms) AS max_ms
      FROM d WHERE gap_ms IS NOT NULL
    ''')
    df.insert(0,'stream', f'{venue}/{product_type}/{event}/{symbol}')
    return df

rows = [gap_stats(*s) for s in streams]
pd.concat(rows, ignore_index=True)
""",
    ),
    (
        "md",
        """
## 4. Clock skew  (local_recv - exchange_ts)

Latency to exchange + our wall-clock drift. Two signals:
- **median**: pure transport latency (ok if ~50–250 ms HL/Binance from your IP).
- **dispersion / drift**: local clock running slow/fast → fix with ntp.
""",
    ),
    (
        "code",
        """
fig, axes = plt.subplots(len(streams), 1, figsize=(11, 1.2*len(streams)), sharex=True)
for ax, (v, p, e, s) in zip(axes, streams):
    g = glob_for(venue=v, product_type=p, event=e, symbol=s)
    df = load_df(f'''
        SELECT to_timestamp(exchange_ts/1e9) AS t,
               (local_recv_ts - exchange_ts)/1e6 AS skew_ms
        FROM read_parquet('{g}', hive_partitioning=true)
        ORDER BY t
    ''')
    if df.empty:
        ax.set_title(f'{v}/{p}/{e}/{s}  (no data)', loc='left'); continue
    df = df.iloc[::max(1, len(df)//4000)]
    ax.plot(df.t, df.skew_ms, alpha=0.6)
    med = df.skew_ms.median()
    ax.axhline(med, color='k', ls='--', lw=0.7)
    ax.set_title(f'{v}/{p}/{e}/{s}  median={med:.0f} ms', loc='left')
    ax.set_ylabel('skew ms')
plt.tight_layout(); plt.show()
""",
    ),
    (
        "md",
        """
## 5. Health log

Subscribed / reconnect / outcome-rolled events. A clean run shows a small set of `subscribed` lines at startup and not much else.
""",
    ),
    (
        "code",
        """
hg = glob_for(event='health')
load_df(f'''
SELECT venue, product_type, kind, detail, to_timestamp(exchange_ts/1e9) AS t
FROM read_parquet('{hg}', hive_partitioning=true)
ORDER BY exchange_ts
''').tail(60)
""",
    ),
    (
        "md",
        """
## 6. Schema sanity — NaNs, dupes, weird values

Sanity-check that no field that should be present is null, and that no obviously-impossible values (e.g. negative size) leaked in.
""",
    ),
    (
        "code",
        """
# Trades: nulls, sign / size / price bounds
sane = load_df(f'''
SELECT venue, symbol,
       count(*) AS n,
       sum(price <= 0)          AS bad_px,
       sum(size <= 0)           AS bad_sz,
       sum(side='unknown')      AS unknown_side,
       count(DISTINCT trade_id) AS distinct_ids
FROM read_parquet('{glob_for(event="trade")}', hive_partitioning=true)
GROUP BY 1,2 ORDER BY 1,2
''')
sane
""",
    ),
    (
        "code",
        """
# BBO: crossed / locked books should be ~0
xb = load_df(f'''
SELECT venue, symbol,
       count(*) AS n,
       sum(bid_px >= ask_px) AS crossed_or_locked,
       avg((ask_px-bid_px)/((ask_px+bid_px)/2)) AS avg_rel_spread
FROM read_parquet('{glob_for(event="bbo")}', hive_partitioning=true)
WHERE bid_px IS NOT NULL AND ask_px IS NOT NULL AND bid_px>0 AND ask_px>0
GROUP BY 1,2 ORDER BY 1,2
''')
xb
""",
    ),
    (
        "md",
        """
## 7. Verdict

Read this off the cells above:
- Coverage row matches expectation? (HL BTC perp, HL UBTC spot if present, HL #30 #31, Binance perp + spot — all should have rows)
- No streams with `rows == 0` for active subscriptions
- p999 inter-arrival under ~5 s on every stream
- Clock skew bounded and not drifting (>>1 s drift over 1 hr ⇒ ntp the host)
- `crossed_or_locked` ≈ 0 on all BBO streams

If any check fails, fix recorder before consuming downstream notebooks.
""",
    ),
]


# ---------------------------------------------------------------------------
# 02 — P1 Cross-venue basis & funding
# ---------------------------------------------------------------------------
P1 = [
    (
        "md",
        """
# 02 — Cross-venue basis & funding (P1)

**What this notebook answers**

- HL perp vs Binance perp basis: how does it move? mean-reverting?
- HL perp vs HL spot (UBTC) and HL perp vs Binance spot basis.
- Funding rate path on both venues; HL is **hourly**, Binance **8h**, so we annualize for parity.
- Funding-implied carry vs realized basis.

These are the inputs for an MM hedge model: if the basis is wide and stable,
inventory cost is the dominant edge. If it's noisy and zero-mean, basis
arbitrage drowns out passive MM PnL.
""",
    ),
    (
        "code",
        """
from hlanalysis.analysis import duck, glob_for, load_df, set_mpl_defaults
import pandas as pd, numpy as np, matplotlib.pyplot as plt
set_mpl_defaults()

def mid_resampled(venue, product_type, symbol, freq='1s'):
    g = glob_for(venue=venue, product_type=product_type, event='bbo', symbol=symbol)
    df = load_df(f'''
        SELECT exchange_ts, (bid_px+ask_px)/2.0 AS mid
        FROM read_parquet('{g}', hive_partitioning=true)
        WHERE bid_px>0 AND ask_px>0
        ORDER BY exchange_ts
    ''')
    if df.empty: return df
    df['t'] = pd.to_datetime(df.exchange_ts, unit='ns', utc=True)
    df = df.set_index('t')[['mid']].resample(freq).last().ffill().dropna()
    return df

hl_perp   = mid_resampled('hyperliquid','perp','BTC').rename(columns={'mid':'hl_perp'})
bn_perp   = mid_resampled('binance','perp','BTCUSDT').rename(columns={'mid':'bn_perp'})
bn_spot   = mid_resampled('binance','spot','BTCUSDT').rename(columns={'mid':'bn_spot'})
joined = hl_perp.join([bn_perp, bn_spot], how='inner')
print(joined.tail()); print('rows:', len(joined))
""",
    ),
    (
        "md",
        """
## 1. Mid prices, all venues, overlaid

Eyeball check: do the curves track? Any structural offset? For BTC at the same
moment they should be visually indistinguishable at this zoom.
""",
    ),
    (
        "code",
        """
fig, ax = plt.subplots(figsize=(11,4))
for c in joined.columns:
    ax.plot(joined.index, joined[c], label=c, lw=1)
ax.set_ylabel('mid (USD)'); ax.legend(loc='upper left'); ax.set_title('BTC mid — HL perp vs Binance perp/spot')
plt.tight_layout(); plt.show()
""",
    ),
    (
        "md",
        """
## 2. Basis time series (HL perp vs Binance perp / spot)

Two flavors:
- **HL_perp − Binance_perp** : carry vs futures-curve term. Should hover near zero, mean-reverting.
- **HL_perp − Binance_spot** : carry vs cash. Should equal funding-implied carry over the holding horizon.
""",
    ),
    (
        "code",
        """
basis = pd.DataFrame({
    'hl_perp_minus_bn_perp_bps': 1e4*(joined.hl_perp - joined.bn_perp)/joined.bn_perp,
    'hl_perp_minus_bn_spot_bps': 1e4*(joined.hl_perp - joined.bn_spot)/joined.bn_spot,
    'bn_perp_minus_bn_spot_bps': 1e4*(joined.bn_perp - joined.bn_spot)/joined.bn_spot,
})

fig, ax = plt.subplots(figsize=(11,4.5))
for c in basis.columns:
    ax.plot(basis.index, basis[c], label=c, lw=1)
ax.axhline(0, color='k', lw=0.6)
ax.set_ylabel('basis (bps of mid)')
ax.legend(loc='upper left')
ax.set_title('Cross-venue basis (bps)')
plt.tight_layout(); plt.show()

print(basis.describe(percentiles=[.01,.05,.5,.95,.99]).T[['mean','std','min','5%','50%','95%','max']])
""",
    ),
    (
        "md",
        """
## 3. Basis distribution & half-life of mean reversion

A symmetric tight distribution and short half-life ⇒ classic stat-arb regime;
asymmetric or fat-tailed ⇒ regime in transition. Half-life from AR(1) on
deviations from mean.
""",
    ),
    (
        "code",
        """
def half_life(s):
    s = s.dropna()
    if len(s) < 50: return np.nan
    s_lag = s.shift(1).dropna()
    s_diff = (s - s_lag).dropna()
    s_lag = s_lag.loc[s_diff.index]
    # OLS  Δs = a + b*s_lag
    import numpy.linalg as la
    X = np.column_stack([np.ones_like(s_lag), s_lag.values])
    beta, *_ = la.lstsq(X, s_diff.values, rcond=None)
    b = beta[1]
    if b >= 0: return np.nan
    return -np.log(2)/np.log(1+b)

fig, axes = plt.subplots(1, 3, figsize=(13,3.5))
for ax, c in zip(axes, basis.columns):
    s = basis[c].dropna()
    ax.hist(s, bins=80)
    hl = half_life(s)
    ax.set_title(f'{c}\\nhalf-life ≈ {hl:.1f} samples' if hl==hl else c)
    ax.set_xlabel('bps')
plt.tight_layout(); plt.show()
""",
    ),
    (
        "md",
        """
## 4. Funding paths

HL funding is hourly, Binance is 8-hourly. We plot both as **annualized %**:
`annual = funding_rate * periods_per_year`. HL: ×24×365, Binance: ×3×365.
""",
    ),
    (
        "code",
        """
def funding_df(venue, symbol, periods_per_day):
    g = glob_for(venue=venue, product_type='perp', event='funding', symbol=symbol)
    df = load_df(f'''
        SELECT exchange_ts, funding_rate, premium
        FROM read_parquet('{g}', hive_partitioning=true)
        ORDER BY exchange_ts
    ''')
    if df.empty: return df
    df['t'] = pd.to_datetime(df.exchange_ts, unit='ns', utc=True)
    df['annual_pct'] = df.funding_rate * periods_per_day * 365 * 100
    return df.set_index('t')

f_hl = funding_df('hyperliquid','BTC', periods_per_day=24)
f_bn = funding_df('binance','BTCUSDT', periods_per_day=3)
print('HL funding rows:', len(f_hl), ' BN funding rows:', len(f_bn))

fig, ax = plt.subplots(figsize=(11,3.5))
if not f_hl.empty: ax.plot(f_hl.index, f_hl.annual_pct, label='HL annualized %')
if not f_bn.empty: ax.plot(f_bn.index, f_bn.annual_pct, label='Binance annualized %')
ax.axhline(0, color='k', lw=0.6); ax.set_ylabel('annualized funding %'); ax.legend()
ax.set_title('Funding paths (annualized)'); plt.tight_layout(); plt.show()
""",
    ),
    (
        "md",
        """
## 5. Funding-implied vs realized basis

Carry over the holding horizon ≈ funding × dt. For an MM holding HL-perp inventory
hedged on Binance, we pay HL funding and earn Binance funding (or vice versa).
The realised basis path should track this carry on average.
""",
    ),
    (
        "code",
        """
# join funding to basis on minute grid
if not f_hl.empty and not f_bn.empty:
    fhl = f_hl.annual_pct.resample('1min').last().ffill().rename('hl_pct')
    fbn = f_bn.annual_pct.resample('1min').last().ffill().rename('bn_pct')
    df = basis.resample('1min').last().join([fhl,fbn], how='inner').dropna()
    df['carry_diff_pct'] = df.hl_pct - df.bn_pct  # HL - Binance, annualized
    fig, ax1 = plt.subplots(figsize=(11,4))
    ax1.plot(df.index, df.hl_perp_minus_bn_perp_bps, label='basis HL-BN (bps)', color='C0')
    ax1.set_ylabel('basis (bps)'); ax1.axhline(0, color='k', lw=0.6)
    ax2 = ax1.twinx()
    ax2.plot(df.index, df.carry_diff_pct, label='carry diff annual %', color='C1', alpha=0.6)
    ax2.set_ylabel('carry diff (annual %)')
    ax1.set_title('Basis vs funding-implied carry')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.95))
    plt.tight_layout(); plt.show()
else:
    print('Not enough funding rows yet to plot.')
""",
    ),
]


# ---------------------------------------------------------------------------
# 03 — P2 Microstructure
# ---------------------------------------------------------------------------
P2 = [
    (
        "md",
        """
# 03 — Microstructure (P2)

**What this notebook answers**

- Quoted spread distribution (bps) by venue.
- Top-of-book depth and depth profile.
- Order-book imbalance: does it predict short-horizon mid moves?
- Top-of-book churn (quote updates per second) — proxy for venue tick speed.

These set the **passive-quote regime** the MM is competing in. A 1-bps spread
on Binance vs 5-bps on HL means very different fill/edge math.
""",
    ),
    (
        "code",
        """
from hlanalysis.analysis import duck, glob_for, load_df, set_mpl_defaults
import pandas as pd, numpy as np, matplotlib.pyplot as plt
set_mpl_defaults()

def bbo_df(venue, product_type, symbol):
    g = glob_for(venue=venue, product_type=product_type, event='bbo', symbol=symbol)
    df = load_df(f'''
        SELECT exchange_ts, bid_px, bid_sz, ask_px, ask_sz
        FROM read_parquet('{g}', hive_partitioning=true)
        WHERE bid_px>0 AND ask_px>0
        ORDER BY exchange_ts
    ''')
    if df.empty: return df
    df['t'] = pd.to_datetime(df.exchange_ts, unit='ns', utc=True)
    df['mid'] = 0.5*(df.bid_px+df.ask_px)
    df['spread_bps'] = 1e4*(df.ask_px-df.bid_px)/df.mid
    df['imbalance'] = (df.bid_sz - df.ask_sz)/(df.bid_sz + df.ask_sz)
    return df.set_index('t')

streams = {
    'HL perp BTC':       ('hyperliquid','perp','BTC'),
    'Binance perp BTC':  ('binance','perp','BTCUSDT'),
    'Binance spot BTC':  ('binance','spot','BTCUSDT'),
    'HL HIP-4 #30':      ('hyperliquid','prediction_binary','#30'),
    'HL HIP-4 #31':      ('hyperliquid','prediction_binary','#31'),
}
data = {k: bbo_df(*v) for k,v in streams.items()}
{k: len(v) for k,v in data.items()}
""",
    ),
    (
        "md",
        """
## 1. Quoted-spread distribution
""",
    ),
    (
        "code",
        """
fig, ax = plt.subplots(figsize=(11,4))
for k, df in data.items():
    if df.empty: continue
    s = df.spread_bps.clip(upper=df.spread_bps.quantile(0.999))
    ax.hist(s, bins=120, histtype='step', label=f'{k}  median={s.median():.2f} bps', lw=1.4, density=True)
ax.set_xscale('log'); ax.set_xlabel('quoted spread (bps, log)'); ax.set_ylabel('density')
ax.set_title('Quoted spread distribution by venue'); ax.legend()
plt.tight_layout(); plt.show()
""",
    ),
    (
        "md",
        """
## 2. Quoted spread time series (1s resampled)
""",
    ),
    (
        "code",
        """
fig, ax = plt.subplots(figsize=(11,4))
for k, df in data.items():
    if df.empty: continue
    ax.plot(df.spread_bps.resample('1s').median(), label=k, lw=0.8)
ax.set_yscale('log'); ax.set_ylabel('median quoted spread (bps)')
ax.set_title('Quoted spread over time'); ax.legend()
plt.tight_layout(); plt.show()
""",
    ),
    (
        "md",
        """
## 3. Top-of-book depth (bid_sz + ask_sz) over time
""",
    ),
    (
        "code",
        """
fig, ax = plt.subplots(figsize=(11,4))
for k, df in data.items():
    if df.empty: continue
    tob = (df.bid_sz + df.ask_sz).resample('5s').median()
    ax.plot(tob, label=k, lw=0.8)
ax.set_yscale('log'); ax.set_ylabel('top-of-book size (base units)')
ax.set_title('Top-of-book depth'); ax.legend()
plt.tight_layout(); plt.show()
""",
    ),
    (
        "md",
        """
## 4. Depth profile from L2 snapshots

For each venue with `book_snapshot` available, plot cumulative bid/ask size as
a function of distance from mid. Use the most recent snapshot.
""",
    ),
    (
        "code",
        """
def latest_book(venue, product_type, symbol):
    g = glob_for(venue=venue, product_type=product_type, event='book_snapshot', symbol=symbol)
    df = load_df(f'''
        SELECT * FROM read_parquet('{g}', hive_partitioning=true)
        ORDER BY exchange_ts DESC LIMIT 1
    ''')
    if df.empty: return None
    r = df.iloc[0]
    return r

books = {
    'HL perp BTC':      latest_book('hyperliquid','perp','BTC'),
    'Binance perp BTC': latest_book('binance','perp','BTCUSDT'),
    'Binance spot BTC': latest_book('binance','spot','BTCUSDT'),
}

fig, ax = plt.subplots(figsize=(11,4.5))
for label, r in books.items():
    if r is None: continue
    bid_px = np.asarray(r['bid_px'], dtype=float); bid_sz = np.asarray(r['bid_sz'], dtype=float)
    ask_px = np.asarray(r['ask_px'], dtype=float); ask_sz = np.asarray(r['ask_sz'], dtype=float)
    if len(bid_px)==0 or len(ask_px)==0: continue
    mid = 0.5*(bid_px[0]+ask_px[0])
    bid_dist = 1e4*(mid - bid_px)/mid
    ask_dist = 1e4*(ask_px - mid)/mid
    ax.plot(-bid_dist, np.cumsum(bid_sz), label=f'{label} bids')
    ax.plot( ask_dist, np.cumsum(ask_sz), label=f'{label} asks')
ax.set_xlabel('distance from mid (bps; negative=bid)'); ax.set_ylabel('cumulative size')
ax.set_xscale('symlog'); ax.set_yscale('log')
ax.set_title('L2 depth profile (latest snapshot)'); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
""",
    ),
    (
        "md",
        """
## 5. Imbalance → next-mid-move

Sign-correlation of book imbalance with the **next** mid return at 1s/5s. If
this is meaningfully positive, simple skew is a free predictor for an MM.
""",
    ),
    (
        "code",
        """
def imb_predict(df, horizons=(1,5,15)):
    if df.empty: return pd.DataFrame()
    s = df[['mid','imbalance']].copy().resample('1s').last().dropna()
    rows = []
    for h in horizons:
        ret = (s['mid'].shift(-h) - s['mid'])/s['mid']
        c = s['imbalance'].corr(ret)
        rows.append({'horizon_s': h, 'corr': c})
    return pd.DataFrame(rows)

out = []
for k, df in data.items():
    if df.empty: continue
    r = imb_predict(df)
    r.insert(0, 'stream', k)
    out.append(r)
pd.concat(out, ignore_index=True) if out else 'no data'
""",
    ),
    (
        "md",
        """
## 6. Top-of-book churn (BBO updates per second)

Higher = more aggressive market: harder to be passive without getting picked off.
""",
    ),
    (
        "code",
        """
churn = {}
for k, df in data.items():
    if df.empty: continue
    churn[k] = df['mid'].resample('1s').count()
fig, ax = plt.subplots(figsize=(11,3.6))
for k, s in churn.items():
    ax.plot(s.rolling(30).median(), label=f'{k}  median={s.median():.1f}/s', lw=0.8)
ax.set_ylabel('BBO updates / s (30 s med)'); ax.set_title('TOB churn'); ax.legend()
plt.tight_layout(); plt.show()
""",
    ),
]


# ---------------------------------------------------------------------------
# 04 — P3 Trade flow & lead-lag
# ---------------------------------------------------------------------------
P3 = [
    (
        "md",
        """
# 04 — Trade flow & lead-lag (P3)

**What this notebook answers**

- Aggressor balance (taker-buy vs taker-sell volume) by venue.
- Trade-size distribution (CCDF, log-log).
- Cross-venue lead-lag of trade-flow imbalance via cross-correlation.
- Cross-venue lead-lag of mid returns.

If Binance reliably leads HL by Δ ms, that's directly actionable for an MM:
widen / skew quotes on HL when Binance prints aggression.
""",
    ),
    (
        "code",
        """
from hlanalysis.analysis import duck, glob_for, load_df, set_mpl_defaults
import pandas as pd, numpy as np, matplotlib.pyplot as plt
set_mpl_defaults()

def trades_df(venue, product_type, symbol):
    g = glob_for(venue=venue, product_type=product_type, event='trade', symbol=symbol)
    df = load_df(f'''
        SELECT exchange_ts, price, size, side
        FROM read_parquet('{g}', hive_partitioning=true)
        ORDER BY exchange_ts
    ''')
    if df.empty: return df
    df['t'] = pd.to_datetime(df.exchange_ts, unit='ns', utc=True)
    df['signed_size'] = np.where(df.side=='buy', df['size'], np.where(df.side=='sell', -df['size'], 0))
    return df.set_index('t')

trades = {
    'HL perp BTC':      trades_df('hyperliquid','perp','BTC'),
    'Binance perp BTC': trades_df('binance','perp','BTCUSDT'),
    'Binance spot BTC': trades_df('binance','spot','BTCUSDT'),
}
{k: len(v) for k,v in trades.items()}
""",
    ),
    (
        "md",
        """
## 1. Aggressor imbalance per minute
""",
    ),
    (
        "code",
        """
fig, ax = plt.subplots(figsize=(11,4))
for k, df in trades.items():
    if df.empty: continue
    m = df['signed_size'].resample('1min').sum() / df['size'].resample('1min').sum()
    ax.plot(m, label=k, lw=0.9)
ax.axhline(0, color='k', lw=0.5); ax.set_ylim(-1,1)
ax.set_ylabel('aggressor imbalance (buy−sell)/total size')
ax.set_title('Aggressor imbalance per minute'); ax.legend()
plt.tight_layout(); plt.show()
""",
    ),
    (
        "md",
        """
## 2. Trade-size CCDF (log-log)

Power-law-ish ⇒ heavy tail ⇒ giant prints drive markouts. Slope is informative.
""",
    ),
    (
        "code",
        """
fig, ax = plt.subplots(figsize=(11,4))
for k, df in trades.items():
    if df.empty: continue
    s = np.sort(df['size'].values)
    ccdf = 1 - np.arange(1, len(s)+1)/len(s)
    ax.plot(s, ccdf, label=k, lw=1.0)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('trade size (base units, log)'); ax.set_ylabel('P(X >= size)')
ax.set_title('Trade-size CCDF'); ax.legend()
plt.tight_layout(); plt.show()
""",
    ),
    (
        "md",
        """
## 3. Cross-venue lead-lag — flow imbalance

Resample signed flow at 100 ms; compute cross-correlation HL vs Binance perp at
±5 s lags. Peak position tells you who leads.
""",
    ),
    (
        "code",
        """
def flow_grid(df, freq='100ms'):
    if df.empty: return pd.Series(dtype=float)
    return df['signed_size'].resample(freq).sum().fillna(0)

a = flow_grid(trades['HL perp BTC'])
b = flow_grid(trades['Binance perp BTC'])
common = a.index.intersection(b.index)
a = a.loc[common]; b = b.loc[common]

def xcorr(a, b, max_lag=50):
    a = (a - a.mean())/(a.std() or 1)
    b = (b - b.mean())/(b.std() or 1)
    n = len(a); lags = np.arange(-max_lag, max_lag+1)
    out = []
    for L in lags:
        if L < 0:   c = np.dot(a[-L:].values, b[:n+L].values)/(n-abs(L))
        elif L > 0: c = np.dot(a[:n-L].values, b[L:].values)/(n-L)
        else:       c = np.dot(a.values, b.values)/n
        out.append(c)
    return lags, np.array(out)

lags, cc = xcorr(a, b, max_lag=50)  # ±5s at 100ms grid
fig, ax = plt.subplots(figsize=(10,3.5))
ax.bar(lags*100, cc, width=80)
ax.axvline(0, color='k', lw=0.6)
ax.set_xlabel('lag (ms)  — positive = HL lags Binance')
ax.set_ylabel('xcorr (signed flow)')
ax.set_title('Lead-lag: HL perp vs Binance perp signed flow')
plt.tight_layout(); plt.show()
peak = lags[int(np.argmax(np.abs(cc)))]*100
print(f'peak |xcorr| at lag = {peak} ms (positive ⇒ HL lags Binance)')
""",
    ),
    (
        "md",
        """
## 4. Cross-venue lead-lag — mid returns
""",
    ),
    (
        "code",
        """
from hlanalysis.analysis import glob_for
def mid_grid(venue, product_type, symbol, freq='100ms'):
    g = glob_for(venue=venue, product_type=product_type, event='bbo', symbol=symbol)
    df = load_df(f'''
        SELECT exchange_ts, (bid_px+ask_px)/2.0 AS mid
        FROM read_parquet('{g}', hive_partitioning=true)
        WHERE bid_px>0 AND ask_px>0 ORDER BY exchange_ts
    ''')
    if df.empty: return pd.Series(dtype=float)
    s = df.set_index(pd.to_datetime(df.exchange_ts, unit='ns', utc=True))['mid']
    s = s.resample(freq).last().ffill()
    return s.pct_change()

ra = mid_grid('hyperliquid','perp','BTC')
rb = mid_grid('binance','perp','BTCUSDT')
common = ra.index.intersection(rb.index)
ra = ra.loc[common].fillna(0); rb = rb.loc[common].fillna(0)
lags, cc = xcorr(ra, rb, max_lag=50)
fig, ax = plt.subplots(figsize=(10,3.5))
ax.bar(lags*100, cc, width=80); ax.axvline(0, color='k', lw=0.6)
ax.set_xlabel('lag (ms)'); ax.set_ylabel('xcorr (returns)')
ax.set_title('Lead-lag: HL perp vs Binance perp mid returns')
plt.tight_layout(); plt.show()
peak = lags[int(np.argmax(np.abs(cc)))]*100
print(f'peak |xcorr| at lag = {peak} ms')
""",
    ),
]


# ---------------------------------------------------------------------------
# 05 — P5 HIP-4 binary fair value
# ---------------------------------------------------------------------------
P5 = [
    (
        "md",
        """
# 05 — HIP-4 binary fair value (P5)

**What this notebook answers**

- HIP-4 binary `BTC > 79980 by May 5 06:00 UTC` is split into two CLOBs (`#30` YES / `#31` NO). The two best-bids should sum to **≤ 1.0** and best-asks to **≥ 1.0** — otherwise free arb.
- Theoretical fair value via Black–Scholes digital: `P_yes = N(d2)` where
  `d2 = (ln(S/K) − σ²(T−t)/2) / (σ√(T−t))`.
- We use HL perp mid as `S`, the strike from market metadata as `K`, and a
  realised vol estimate as `σ`. Compare to traded prices and the implied vol you'd
  back out from market price.
- Cross-book arb spread: `1 − (bid_yes + bid_no)`.
""",
    ),
    (
        "code",
        """
from hlanalysis.analysis import duck, glob_for, load_df, set_mpl_defaults
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.stats import norm
set_mpl_defaults()

# Pull market metadata for #30 and #31
mm = load_df(f'''
SELECT symbol, exchange_ts, keys, values
FROM read_parquet('{glob_for(venue='hyperliquid', product_type='prediction_binary', event='market_meta')}',
                  hive_partitioning=true)
ORDER BY exchange_ts DESC
''')
mm
""",
    ),
    (
        "code",
        """
# Extract strike + expiry per symbol.
# HIP-4 metadata uses keys: outcome_idx, side_idx, side_name, outcome_name,
# class, underlying, expiry (e.g. '20260506-0600'), targetPrice, period.
def kv_lookup(row):
    d = dict(zip(row['keys'], row['values']))
    exp_raw = d.get('expiry','')
    try:
        exp_ts = pd.to_datetime(exp_raw, format='%Y%m%d-%H%M', utc=True) if exp_raw else None
    except Exception:
        exp_ts = None
    try:
        K = float(d.get('targetPrice','nan'))
    except Exception:
        K = float('nan')
    return pd.Series({
        'strike': K,
        'expiry': exp_ts,
        'side_name': d.get('side_name',''),
        'underlying': d.get('underlying',''),
        'period': d.get('period',''),
    })

latest_meta = mm.sort_values('exchange_ts').groupby('symbol').tail(1).reset_index(drop=True)
meta = pd.concat([latest_meta[['symbol']], latest_meta.apply(kv_lookup, axis=1)], axis=1)
meta
""",
    ),
    (
        "md",
        """
## 1. Cross-book arb spread

If `bid_yes + bid_no > 1.0` we could lift both bids and earn the difference at
settlement. If `ask_yes + ask_no < 1.0` we can buy both for less than 1. Either is
free money — should be ~0 net of fees.
""",
    ),
    (
        "code",
        """
def bbo_grid(symbol, freq='5s'):
    g = glob_for(venue='hyperliquid', product_type='prediction_binary', event='bbo', symbol=symbol)
    df = load_df(f'''
        SELECT exchange_ts, bid_px, bid_sz, ask_px, ask_sz
        FROM read_parquet('{g}', hive_partitioning=true)
        WHERE bid_px>0 AND ask_px>0 ORDER BY exchange_ts
    ''')
    if df.empty: return df
    df['t'] = pd.to_datetime(df.exchange_ts, unit='ns', utc=True)
    return df.set_index('t').resample(freq).last().ffill().dropna()

yes = bbo_grid('#30').rename(columns=lambda c: c+'_yes')
no  = bbo_grid('#31').rename(columns=lambda c: c+'_no')
both = yes.join(no, how='inner').dropna()
both['bid_sum'] = both.bid_px_yes + both.bid_px_no
both['ask_sum'] = both.ask_px_yes + both.ask_px_no
both[['bid_sum','ask_sum']].describe()
""",
    ),
    (
        "code",
        """
fig, ax = plt.subplots(figsize=(11,3.8))
ax.plot(both.index, both.bid_sum, label='bid_yes + bid_no')
ax.plot(both.index, both.ask_sum, label='ask_yes + ask_no')
ax.axhline(1.0, color='k', lw=0.7)
ax.set_ylabel('sum'); ax.set_title('HIP-4 cross-book sums (should bracket 1.0)')
ax.legend(); plt.tight_layout(); plt.show()
""",
    ),
    (
        "md",
        """
## 2. Theoretical fair value vs traded mid

We estimate σ from HL perp mid log-returns over the available window (annualized).
Then `P_yes = N(d2)`.
""",
    ),
    (
        "code",
        """
# Volatility from HL perp BTC mid
hl = load_df(f'''
SELECT exchange_ts, (bid_px+ask_px)/2.0 AS mid
FROM read_parquet('{glob_for(venue='hyperliquid', product_type='perp', event='bbo', symbol='BTC')}',
                  hive_partitioning=true)
WHERE bid_px>0 AND ask_px>0 ORDER BY exchange_ts
''')
hl['t'] = pd.to_datetime(hl.exchange_ts, unit='ns', utc=True)
hl = hl.set_index('t')[['mid']].resample('1s').last().ffill().dropna()
ret = np.log(hl['mid']).diff().dropna()
# annualised vol from 1s log-rets
sigma_ann = ret.std() * np.sqrt(365*24*3600)
print(f'Realized 1s vol (annualised): {sigma_ann:.3%}')
""",
    ),
    (
        "code",
        """
# Theoretical YES price using #30 metadata (strike + expiry)
m = meta[meta.symbol=='#30'].iloc[0] if (meta.symbol=='#30').any() else None
if m is None or pd.isna(m.strike) or m.expiry is None:
    print('No #30 metadata yet — print of meta:'); print(meta)
else:
    K = float(m.strike); T_expiry = pd.Timestamp(m.expiry).tz_convert('UTC')
    print(f'#30 strike={K:,.0f}  expiry={T_expiry}')

    grid = both.join(hl[['mid']].rename(columns={'mid':'S'}).resample('5s').last().ffill(), how='inner').dropna()
    tau = (T_expiry - grid.index).total_seconds() / (365*24*3600)
    tau = np.clip(tau, 1e-9, None)
    d2 = (np.log(grid['S']/K) - 0.5 * sigma_ann**2 * tau) / (sigma_ann * np.sqrt(tau))
    grid['theo_yes'] = norm.cdf(d2)
    grid['mid_yes']  = 0.5*(grid.bid_px_yes + grid.ask_px_yes)
    grid['mid_no']   = 0.5*(grid.bid_px_no  + grid.ask_px_no)

    fig, axes = plt.subplots(2,1, figsize=(11,6), sharex=True)
    axes[0].plot(grid.index, grid['mid_yes'], label='mid YES (#30)')
    axes[0].plot(grid.index, 1-grid['mid_no'], label='1 − mid NO (#31)')
    axes[0].plot(grid.index, grid['theo_yes'], label=f'N(d2) σ={sigma_ann:.0%}', ls='--')
    axes[0].set_ylabel('YES price'); axes[0].legend()
    axes[0].set_title(f'YES vs theoretical  (K={K:,.0f})')

    axes[1].plot(grid.index, grid['S'], color='C3'); axes[1].set_ylabel('BTC perp mid (S)')
    plt.tight_layout(); plt.show()

    grid[['S','mid_yes','mid_no','theo_yes']].tail()
""",
    ),
    (
        "md",
        """
## 3. Implied σ time series

Given traded YES mid, invert `N(d2)` for σ at each timestamp. A spike vs
realised σ is a cleaner signal than raw price for measuring market disagreement.
""",
    ),
    (
        "code",
        """
def implied_vol(p_yes, S, K, tau):
    # find σ s.t. N(d2) = p_yes
    if not (0 < p_yes < 1) or tau <= 0 or S <= 0 or K <= 0: return np.nan
    target = norm.ppf(p_yes)  # = d2
    # d2(σ) = (ln(S/K) - 0.5 σ² τ) / (σ √τ)
    # solve quadratic in σ:  -0.5 τ σ² - (target √τ) σ + ln(S/K) = 0
    a = -0.5*tau; b = -target*np.sqrt(tau); c = np.log(S/K)
    disc = b*b - 4*a*c
    if disc < 0: return np.nan
    s1 = (-b + np.sqrt(disc))/(2*a); s2 = (-b - np.sqrt(disc))/(2*a)
    cands = [s for s in (s1, s2) if s and s > 0]
    return min(cands) if cands else np.nan

if 'grid' in globals():
    iv = []
    for ts, row in grid.iterrows():
        tau = (T_expiry - ts).total_seconds()/(365*24*3600)
        iv.append(implied_vol(row.mid_yes, row.S, K, tau))
    grid['impl_vol'] = iv
    fig, ax = plt.subplots(figsize=(11,3.6))
    ax.plot(grid.index, grid['impl_vol'], label='Implied σ from YES mid')
    ax.axhline(sigma_ann, color='k', ls='--', label=f'Realized σ = {sigma_ann:.0%}')
    ax.set_ylabel('annualised σ'); ax.legend()
    ax.set_title('HIP-4 implied vs realised vol')
    plt.tight_layout(); plt.show()
""",
    ),
]


# ---------------------------------------------------------------------------
# 06 — HIP-4 pricing from BTC perp (theo vs actual on local_recv_ts grid)
# ---------------------------------------------------------------------------
P6 = [
    (
        "md",
        """
# 06 — HIP-4 pricing from BTC/HL-perp (theo vs actual)

The HIP-4 binary `BTC > 80,930 by 2026-05-06 06:00 UTC` is a **digital
cash-or-nothing call** on BTC. If we believe BTC log-returns are roughly
Gaussian under the risk-neutral measure, the fair value of the YES leg is

$$P_\\mathrm{YES}(t) \\;=\\; \\Pr\\!\\big(S_T > K \\mid S_t\\big) \\;=\\; \\Phi(d_2),$$

with

$$d_2 \\;=\\; \\frac{\\ln(S_t/K) + (r-\\tfrac12 \\sigma^2)(T-t)}{\\sigma\\sqrt{T-t}}.$$

For an on-chain perp with no real risk-free rate and a holding horizon of a
few hours, the drift term is dominated by funding/carry; we set $r=0$ here
(the carry contribution to a 1-day binary is ~10⁻⁴ — well below quoted
spread). What's left is two inputs:

- **$S_t$**: HL BTC perp mid (closest to the oracle that settles the binary).
- **$\\sigma$**: realized vol of HL perp log-returns over a rolling window.

We then plot the theo $\\Phi(d_2)$ on the same time axis as the actual mid of
`#30` (YES) and `1 − mid(#31)` (the implied YES from the NO book). All series
are aligned on **`local_recv_ts`** so we compare prices we actually had access
to at each instant — not the venue's stamped time.

Read this as a fair-value baseline, not a strategy: any structural offset
between theo and market is the reward the market is paying / charging for the
σ assumption being wrong, plus the binary's bid-ask premium for liquidity.
""",
    ),
    (
        "code",
        """
from hlanalysis.analysis import duck, glob_for, load_df, set_mpl_defaults
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.stats import norm
set_mpl_defaults()
""",
    ),
    (
        "md",
        """
## 1. Strike + expiry from market metadata

`market_meta` is emitted on every wildcard refresh. We take the latest values
per symbol.
""",
    ),
    (
        "code",
        """
mm = load_df(f'''
SELECT symbol, exchange_ts, local_recv_ts, keys, values
FROM read_parquet('{glob_for(venue='hyperliquid', product_type='prediction_binary', event='market_meta')}',
                  hive_partitioning=true)
ORDER BY local_recv_ts
''')

def kv(row):
    d = dict(zip(row['keys'], row['values']))
    try: K = float(d.get('targetPrice','nan'))
    except Exception: K = float('nan')
    try: exp = pd.to_datetime(d.get('expiry',''), format='%Y%m%d-%H%M', utc=True)
    except Exception: exp = None
    return pd.Series({'strike': K, 'expiry': exp,
                      'side': d.get('side_name',''), 'underlying': d.get('underlying','')})

latest = mm.sort_values('local_recv_ts').groupby('symbol').tail(1).reset_index(drop=True)
meta   = pd.concat([latest[['symbol']], latest.apply(kv, axis=1)], axis=1)
meta
""",
    ),
    (
        "code",
        """
yes_meta = meta[meta.symbol=='#30'].iloc[0]
K       = float(yes_meta.strike)
T_exp   = pd.Timestamp(yes_meta.expiry).tz_convert('UTC')
print(f'Binary: BTC > {K:,.0f} by {T_exp}')
""",
    ),
    (
        "md",
        """
## 2. Load all three legs, indexed on `local_recv_ts`

We deliberately use `local_recv_ts` (host arrival ns) — not the venue's
`exchange_ts` — because the host clock is the only common timeline across the
three streams (and the Binance spot adapter doesn't even stamp `exchange_ts`,
see [`adapters/binance.py:171`](../../hlanalysis/adapters/binance.py:171)).
""",
    ),
    (
        "code",
        """
def bbo_local_grid(venue, product_type, symbol, freq='5s'):
    g = glob_for(venue=venue, product_type=product_type, event='bbo', symbol=symbol)
    df = load_df(f'''
        SELECT local_recv_ts, bid_px, ask_px
        FROM read_parquet('{g}', hive_partitioning=true)
        WHERE bid_px>0 AND ask_px>0
        ORDER BY local_recv_ts
    ''')
    if df.empty: return df
    df['t'] = pd.to_datetime(df.local_recv_ts, unit='ns', utc=True)
    df['mid'] = 0.5*(df.bid_px + df.ask_px)
    return (df.set_index('t')[['mid','bid_px','ask_px']]
              .resample(freq).last().ffill().dropna())

S    = bbo_local_grid('hyperliquid','perp','BTC').rename(columns={'mid':'S'})
y    = bbo_local_grid('hyperliquid','prediction_binary','#30').rename(columns={
            'mid':'mid_yes','bid_px':'bid_yes','ask_px':'ask_yes'})
n    = bbo_local_grid('hyperliquid','prediction_binary','#31').rename(columns={
            'mid':'mid_no', 'bid_px':'bid_no', 'ask_px':'ask_no'})

g = S[['S']].join(y[['mid_yes','bid_yes','ask_yes']], how='inner') \\
            .join(n[['mid_no','bid_no','ask_no']],     how='inner').dropna()
print('aligned rows:', len(g),  'window:', g.index.min(), '→', g.index.max())
g.head(3)
""",
    ),
    (
        "md",
        """
## 3. Rolling realized volatility of HL perp

σ for the BS-digital must be **forward-looking until expiry**, but we only
have history. Three pragmatic choices:

- `sigma_rolling`: 30-min rolling window of log-returns, annualized — tracks
  current regime, but noisy when window is short.
- `sigma_full`: realized over the full data window so far — stable but stale.
- `sigma_fixed_25`: a fixed prior, 25% — where BTC realized typically sits;
  acts as a sanity check.

We display all three theo paths so the chart shows the **σ-sensitivity** of
the binary explicitly.
""",
    ),
    (
        "code",
        """
# 1-second log returns from HL perp BBO mid (separate finer grid → more data)
hl_fine = bbo_local_grid('hyperliquid','perp','BTC', freq='1s')['mid']
ret_1s = np.log(hl_fine).diff().dropna()
SEC_PER_YEAR = 365*24*3600

# rolling 30-min annualized sigma (median over 30-min window)
sigma_rolling = (ret_1s.rolling('30min').std() * np.sqrt(SEC_PER_YEAR))
sigma_full    = float(ret_1s.std() * np.sqrt(SEC_PER_YEAR))
print(f'σ (full window) ≈ {sigma_full:.2%}')

fig, ax = plt.subplots(figsize=(11,3))
ax.plot(sigma_rolling, lw=0.9, label='σ (30-min rolling, 1 s rets)')
ax.axhline(sigma_full, color='k', ls='--', lw=0.7, label=f'σ full = {sigma_full:.0%}')
ax.axhline(0.25,        color='C2', ls=':', lw=0.8, label='σ prior = 25%')
ax.set_ylabel('annualised σ'); ax.legend(); ax.set_title('Rolling realised vol — HL perp BTC')
plt.tight_layout(); plt.show()
""",
    ),
    (
        "md",
        """
## 4. Theoretical YES = Φ(d₂)

Compute the theo using each of the three σ choices, on the same 5-second grid
as the market mids.
""",
    ),
    (
        "code",
        """
sigma_roll_5s = sigma_rolling.reindex(g.index, method='ffill').bfill()
tau = np.asarray((T_exp - g.index).total_seconds() / SEC_PER_YEAR, dtype=float)
tau = np.clip(tau, 1e-9, None)
S_arr = g['S'].values

def theo(sig, S_arr, K, tau):
    sig = np.asarray(sig); tau = np.asarray(tau)
    d2 = (np.log(S_arr/K) - 0.5*sig**2*tau) / (sig*np.sqrt(tau))
    return norm.cdf(d2)

g['theo_roll']  = theo(sigma_roll_5s.values, S_arr, K, tau)
g['theo_full']  = theo(np.full_like(S_arr, sigma_full),   S_arr, K, tau)
g['theo_25']    = theo(np.full_like(S_arr, 0.25),         S_arr, K, tau)
g['mkt_yes_via_no'] = 1 - g['mid_no']
g[['S','mid_yes','mkt_yes_via_no','theo_roll','theo_full','theo_25']].tail()
""",
    ),
    (
        "md",
        """
## 5. The chart you asked for — theo vs actual, aligned on receive time
""",
    ),
    (
        "code",
        """
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                         gridspec_kw={'height_ratios':[2.4, 1]})

ax = axes[0]
ax.plot(g.index, g['mid_yes'],         label='market mid YES (#30)', color='C0', lw=1.4)
ax.plot(g.index, g['mkt_yes_via_no'],  label='1 − mid NO  (from #31)', color='C0', lw=0.8, alpha=0.55, ls='--')
ax.fill_between(g.index, g['bid_yes'], g['ask_yes'], color='C0', alpha=0.10, label='YES bid/ask band')
ax.plot(g.index, g['theo_roll'],       label='theo Φ(d₂), σ=30-min rolling', color='C1', lw=1.2)
ax.plot(g.index, g['theo_full'],       label=f'theo Φ(d₂), σ={sigma_full:.0%} (full window)', color='C2', lw=1.0, alpha=0.8)
ax.plot(g.index, g['theo_25'],         label='theo Φ(d₂), σ=25%', color='C3', lw=0.9, alpha=0.7, ls=':')
ax.set_ylabel('P(YES)')
ax.set_ylim(0, 1)
ax.set_title(f'HIP-4 #30 YES: market vs Φ(d₂) — strike {K:,.0f}, expiry {T_exp:%Y-%m-%d %H:%M} UTC')
ax.legend(loc='upper left', fontsize=8, ncol=2)

# bottom panel: BTC perp mid with strike line
axes[1].plot(g.index, g['S'], color='C3', lw=1.0)
axes[1].axhline(K, color='k', ls='--', lw=0.8, label=f'strike {K:,.0f}')
axes[1].set_ylabel('BTC HL-perp mid')
axes[1].legend(loc='upper left', fontsize=8)

plt.tight_layout(); plt.show()
""",
    ),
    (
        "md",
        """
## 6. Residuals — where does the market disagree with the model?

Positive = market YES > theo (market over-pays for the upside);
negative = market YES < theo. A persistent sign tells you σ is mis-specified;
zero-mean noise tells you the model is the right shape.
""",
    ),
    (
        "code",
        """
fig, ax = plt.subplots(figsize=(12, 3.6))
for col, lbl, c in [('theo_roll','market − theo (σ rolling)','C1'),
                    ('theo_full','market − theo (σ full)','C2'),
                    ('theo_25',  'market − theo (σ=25%)',  'C3')]:
    diff = g['mid_yes'] - g[col]
    ax.plot(g.index, diff, label=lbl, lw=0.9, color=c, alpha=0.85)
ax.axhline(0, color='k', lw=0.6)
ax.set_ylabel('YES_market − YES_theo')
ax.set_title('Residuals'); ax.legend(loc='upper left', fontsize=8)
plt.tight_layout(); plt.show()

print('residual stats (price units):')
for col in ['theo_roll','theo_full','theo_25']:
    d = (g['mid_yes'] - g[col]).dropna()
    print(f'  vs {col:9s}  mean={d.mean():+.4f}  std={d.std():.4f}  '
          f'5%={d.quantile(0.05):+.4f}  95%={d.quantile(0.95):+.4f}')
""",
    ),
    (
        "md",
        """
## 7. Implied σ inverted from market YES

For each timestamp, solve `Φ(d₂(σ)) = YES_market` for σ. Cleaner signal than
the residual because it normalises for the position on the S-curve.
""",
    ),
    (
        "code",
        """
def implied_vol_one(p_yes, S, K, tau):
    if not (0 < p_yes < 1) or tau <= 0 or S <= 0 or K <= 0: return np.nan
    target = norm.ppf(p_yes)  # = d2
    a = -0.5*tau; b = -target*np.sqrt(tau); c = np.log(S/K)
    disc = b*b - 4*a*c
    if disc < 0: return np.nan
    s1 = (-b + np.sqrt(disc))/(2*a); s2 = (-b - np.sqrt(disc))/(2*a)
    cands = [s for s in (s1, s2) if s and 0 < s < 5]
    return min(cands) if cands else np.nan

iv = np.array([
    implied_vol_one(p, s, K, t)
    for p, s, t in zip(g['mid_yes'].values, g['S'].values, tau)
])
g['impl_vol'] = iv

fig, ax = plt.subplots(figsize=(12, 3.6))
ax.plot(g.index, g['impl_vol'], lw=1.0, label='implied σ from YES_market')
ax.plot(sigma_rolling.reindex(g.index).index, sigma_rolling.reindex(g.index),
        lw=0.9, alpha=0.7, color='C1', label='realised σ (30-min rolling)')
ax.axhline(sigma_full, color='k', ls='--', lw=0.7, label=f'realised σ full = {sigma_full:.0%}')
ax.set_ylabel('annualised σ'); ax.set_title('Implied vs realised σ')
ax.legend(loc='upper left', fontsize=8)
plt.tight_layout(); plt.show()

print(f'median implied σ = {np.nanmedian(iv):.2%}   median realised (rolling) = {sigma_rolling.median():.2%}')
""",
    ),
    (
        "md",
        """
## 8. Reading the chart

- If **theo (σ rolling) tracks market YES tightly** → BS-digital + perp-mid is a
  decent fair value at this regime; passive MM on the binary can quote
  symmetric around theo and harvest the bid-ask.
- If **market sits persistently above theo** → either the market is using a
  higher σ than realized (priced uncertainty), or there's a settlement-rule
  premium / dust-tail mass we're not modelling (e.g. the binary settles off
  oracle which can deviate from perp mid; depeg / outage risk gets pushed into
  YES).
- **Implied vs realised σ** is the key panel for an MM: if implied is stable
  and persistently higher than realised (typical for short-dated binaries),
  selling YES spreads + delta-hedging on perp is a positive-EV trade modulo
  hedge cost.

For a real trading decision we'd want:
1. ≥1 full settlement cycle of data (06:00 UTC roll) so we see the post-jump
   re-pricing — currently still pre-roll.
2. Settlement source confirmed (the binary uses HL **oracle** at expiry, not
   the perp mid); for pricing the *settlement* we should use oracle, even
   though for *hedging* perp mid is cleaner.
3. A funding-aware drift term once the binary moves to multi-day expiries.
""",
    ),
]


# ---------------------------------------------------------------------------
# Strategy hypotheses footer — appended to every notebook
# ---------------------------------------------------------------------------
HYPOTHESIS_FOOTER = (
    "md",
    """
## Strategy hypotheses

- **Supports:** H?: (fill in after reviewing the chart above)
- **Rules out:** H?: (fill in if evidence rules it out)
- **Next probe:** (what would refine this further)
""",
)


# ---------------------------------------------------------------------------
# 07 — HIP-4 late-YES strategy with stop-loss
# ---------------------------------------------------------------------------
P7 = [
    (
        "md",
        """
# 07 — HIP-4 late-YES strategy with stop-loss

**What this notebook produces**

A ranked set of trading strategies for the daily HL HIP-4 BTC binary, with concrete entry/stop/sizing parameters for the one to start with on $1k of capital. Builds on the fair-value machinery from `05-hip4-binaries.ipynb` and `06-hip4-pricing-from-perp.ipynb`.

**HIP-4 mechanics (from the [contract spec](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/contract-specifications.md#recurring-outcomes))**

- Recurring binary outcome on the **HL BTC mark price**. Daily expiry at **06:00 UTC**.
- Settlement: linear interpolation between the two mark-price updates immediately before and after 06:00 UTC. Resolves YES iff the interpolated mark ≥ `targetPrice`.
- Each outcome has two sides quoted as separate coins. With `outcome_idx` and `side_idx ∈ {0,1}`, the coin name is `#{10*outcome_idx + side_idx}` (e.g. `#50`, `#51` are the two sides of outcome 5). Buying matched 1 unit of each pays exactly 1 USDC at settle.
- Description metadata (parsed below): `class:priceBinary|underlying:BTC|expiry:YYYYMMDD-HHMM|targetPrice:K|period:1d`.

**Why a late-YES strategy**

- In the last 30–60 min of life, time-decay has done most of its work. If `S` is far from `K` in vol units, `P(YES | now)` is close to 1.
- Edge is small per trade (a few ¢) but hit-rate is high. Asymmetric payoff means a single bad print without a stop wipes out many winners.
- A stop converts the left tail into bounded loss. Bonus: because settlement is HL **mark price**, the **mark price itself** is the most informative stop trigger — earlier signal than the binary's own mid.

**Capital frame**

- Bankroll: $1k. Per-trade size band: $100–$200 (10–20% of bankroll).
- One trade per day max while validating. ~30 trades = ~1 month.

**Notebook layout**

1. Setup, load HL BTC perp, parse HIP-4 metadata.
2. `q_model` and the canonical entry signal: distance-to-strike `d` in vol units.
3. `d` and `q_model` over the life of the most recent outcome.
4. Regime guards — vol-up tripwire + perp depth check.
5. Binary BBO + spread guard.
6. Backtest scaffold — single-cycle simulator.
7. Stop-loss sweep.
8. Calibration plot — `d` vs realized YES rate.
9. Strategy options (priority ordering + concrete starter params).
""",
    ),
    (
        "code",
        """
from hlanalysis.analysis import duck, glob_for, load_df, set_mpl_defaults
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.stats import norm
from dataclasses import dataclass, field
import datetime as dt
set_mpl_defaults()
pd.options.display.float_format = '{:,.6g}'.format

UTC = dt.timezone.utc
SETTLE_HOUR_UTC = 6
SEC_PER_YEAR = 365 * 24 * 3600

# --- Strategy parameters (edit these to tune; defaults are the "Strategy 1" starter) ---
STRAT = dict(
    entry_window_min=(60, 5),       # only enter inside [T-60min, T-5min]
    d_min=3.0,                       # require distance >= 3 vol-units
    price_band=(0.93, 0.99),         # binary mid must be within this range
    stop_d=1.5,                      # exit if d drops below this
    stop_price=0.85,                 # exit if binary mid drops below this
    vol_ratio_max=1.5,               # exit/blackout if RV_5m / RV_60m exceeds this
    vol_floor=0.30,                  # σ floor (annualised) so quiet regimes don't lull d to ∞
    binary_spread_max_cents=1.5,     # binary ask-bid <= 1.5¢
    perp_depth_min_frac=0.50,        # require >= 50% of session-median top-10 depth
    trade_size_usd=100,              # starter size (10% of $1k bankroll)
)
STRAT
""",
    ),
    (
        "md",
        """
## 1. Load HL BTC perp + parse HIP-4 metadata

Mark price ticks (the **settlement source**), perp BBO, and book snapshots. Strikes/expiries come from `outcomeMeta` description, written into `market_meta` events.
""",
    ),
    (
        "code",
        """
# Mark = the actual settlement source. Use it for q_model, not perp mid.
mark = load_df(f'''
    SELECT local_recv_ts, mark_px
    FROM read_parquet('{glob_for(venue='hyperliquid', product_type='perp', event='mark', symbol='BTC')}',
                      hive_partitioning=true)
    ORDER BY local_recv_ts
''')
mark['t']  = pd.to_datetime(mark.local_recv_ts, unit='ns', utc=True)
mark = mark.set_index('t')[['mark_px']]
mark_1s = mark['mark_px'].resample('1s').last().ffill().dropna()
print(f'mark ticks: {len(mark):,} | range: {mark.index.min()} -> {mark.index.max()}')

# HIP-4 metadata. Each row is a snapshot; take latest per symbol.
mm = load_df(f'''
    SELECT symbol, exchange_ts, local_recv_ts, keys, values
    FROM read_parquet('{glob_for(venue='hyperliquid', product_type='prediction_binary', event='market_meta')}',
                      hive_partitioning=true)
    ORDER BY local_recv_ts
''')

def parse_meta(row):
    d = dict(zip(row['keys'], row['values']))
    try: K = float(d.get('targetPrice', 'nan'))
    except Exception: K = float('nan')
    try: exp = pd.to_datetime(d.get('expiry', ''), format='%Y%m%d-%H%M', utc=True)
    except Exception: exp = pd.NaT
    return pd.Series({
        'strike': K, 'expiry': exp,
        'side_idx':   int(d.get('side_idx', -1)),
        'side_name':  d.get('side_name', ''),
        'underlying': d.get('underlying', ''),
        'period':     d.get('period', ''),
        'cls':        d.get('class', ''),
    })

if mm.empty:
    meta = pd.DataFrame()
    print('No HIP-4 metadata yet — recorder needs to run through at least one outcome boundary.')
else:
    latest = mm.sort_values('local_recv_ts').groupby('symbol').tail(1).reset_index(drop=True)
    meta   = pd.concat([latest[['symbol']], latest.apply(parse_meta, axis=1)], axis=1)
    meta = meta.sort_values(['expiry', 'side_idx']).reset_index(drop=True)
meta
""",
    ),
    (
        "md",
        """
## 2. The entry signal — distance `d` in vol units

For a digital that pays 1 if `S_T ≥ K`, log-normal returns give:

$$d_2 = \\frac{\\ln(S/K) - \\tfrac{1}{2}\\sigma^2 \\tau}{\\sigma\\sqrt{\\tau}}, \\qquad q_{\\text{model}} = \\Phi(d_2)$$

The **distance to strike in vol units** is:

$$d \\;=\\; \\frac{S - K}{S \\cdot \\sigma \\sqrt{\\tau}}$$

Why `d` instead of just "binary mid ≥ 95¢" as the entry trigger:

- A binary at 96¢ with `d = 1.5` is mispriced *toward* you the wrong way — the market is overconfident relative to vol.
- A binary at 96¢ with `d = 4` is conservative and that's your edge.
- Same screen price, opposite trade. `d` exposes that.

Notebook 06 already validated `Φ(d_2)` against market mid; here we use the same machinery as a gate, not a model fit.
""",
    ),
    (
        "code",
        """
def q_model(S, K, sigma_annual, tau_years):
    \"\"\"P(S_T >= K) under log-normal. Vectorised over S/sigma/tau (K scalar).\"\"\"
    S = np.asarray(S, dtype=float)
    sig = np.maximum(np.asarray(sigma_annual, dtype=float), 1e-9)
    tau = np.maximum(np.asarray(tau_years, dtype=float), 1e-12)
    d2 = (np.log(S / K) - 0.5 * sig**2 * tau) / (sig * np.sqrt(tau))
    return norm.cdf(d2)

def distance_to_strike(S, K, sigma_annual, tau_years):
    \"\"\"Signed distance |S - K| / (S sigma sqrt(tau)). Sign convention:
    positive when YES (S > K) is the comfortable side.\"\"\"
    S = np.asarray(S, dtype=float)
    sig = np.maximum(np.asarray(sigma_annual, dtype=float), 1e-9)
    tau = np.maximum(np.asarray(tau_years, dtype=float), 1e-12)
    return (S - K) / (S * sig * np.sqrt(tau))

# Realised vol estimators. Mark-price ticks are the right thing to differentiate
# (BBO mid is noisier and the binary settles on mark, not mid).
log_ret_1s = np.log(mark_1s).diff().dropna()
def rolling_rv(returns_1s, window_seconds):
    var = (returns_1s ** 2).rolling(window_seconds, min_periods=window_seconds // 2).sum()
    return np.sqrt(var) * np.sqrt(SEC_PER_YEAR / window_seconds)

rv_5m  = rolling_rv(log_ret_1s, 5 * 60)
rv_60m = rolling_rv(log_ret_1s, 60 * 60)

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(rv_5m.index,  rv_5m,  lw=0.9, alpha=0.7, label='RV 5m')
ax.plot(rv_60m.index, rv_60m, lw=1.4,             label='RV 60m')
ax.axhline(STRAT['vol_floor'], ls='--', color='grey', label=f"σ-floor = {STRAT['vol_floor']:.0%}")
ax.set_ylabel('annualised σ'); ax.legend(); ax.set_title('HL BTC mark — rolling realised vol')
plt.tight_layout(); plt.show()
print(f"σ_60m median: {rv_60m.median():.2%} | σ_5m median: {rv_5m.median():.2%}")
""",
    ),
    (
        "md",
        """
## 3. `d` and `q_model` over the life of the most recent outcome

Pick the most-recently-expired (or active) outcome from `meta`, build the time series of `d` and `q_model`. The horizontal lines `d = 3, 1.5, 0` are the strategy's entry / stop / strike-touch thresholds.
""",
    ),
    (
        "code",
        """
def build_d_series(K, expiry, mark_1s, rv_60m, vol_floor):
    \"\"\"Construct a 1s-grid DataFrame of (S, sigma, tau_years, d, q) for the cycle ending at expiry.\"\"\"
    s_grid = mark_1s.loc[:expiry]
    sigma  = rv_60m.reindex(s_grid.index).clip(lower=vol_floor).ffill().bfill()
    tau    = ((expiry - s_grid.index).total_seconds() / SEC_PER_YEAR).to_numpy().clip(min=1e-12)
    S_arr  = s_grid.to_numpy()
    sig_arr = sigma.to_numpy()
    d_arr   = (S_arr - K) / (S_arr * sig_arr * np.sqrt(tau))
    q_arr   = q_model(S_arr, K, sig_arr, tau)
    return pd.DataFrame({'S': S_arr, 'sigma': sig_arr, 'tau_years': tau,
                         'd': d_arr, 'q_yes': q_arr}, index=s_grid.index)

if meta.empty:
    print('No metadata yet — section skipped.')
    cycle_df = pd.DataFrame()
else:
    yes = meta[meta.side_idx == 0].dropna(subset=['strike','expiry']).sort_values('expiry').iloc[-1]
    K, T_exp, sym = float(yes.strike), pd.Timestamp(yes.expiry), yes.symbol
    print(f"Cycle: {sym}  K = {K:,.0f}  expiry = {T_exp:%Y-%m-%d %H:%M} UTC")
    cycle_df = build_d_series(K, T_exp, mark_1s, rv_60m, STRAT['vol_floor'])
    cycle_df = cycle_df.loc[T_exp - pd.Timedelta(days=1):T_exp]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                             gridspec_kw={'height_ratios': [1.4, 1]})
    axes[0].plot(cycle_df.index, cycle_df.S, lw=1.0, color='C0')
    axes[0].axhline(K, ls='--', color='red', label=f'strike K = {K:,.0f}')
    axes[0].set_ylabel('BTC mark'); axes[0].legend()
    axes[1].plot(cycle_df.index, cycle_df.d, lw=1.1, color='C2')
    for level, col, txt in [(STRAT['d_min'], 'green', 'd_min (entry)'),
                            (STRAT['stop_d'], 'orange', 'stop_d (exit)'),
                            (0, 'red', 'strike touch')]:
        axes[1].axhline(level, ls='--', color=col, alpha=0.7, label=f'{txt} = {level}')
    axes[1].set_ylabel('distance d (vol units)'); axes[1].legend(loc='lower left', fontsize=8)
    fig.suptitle(f'{sym} — S vs K, distance d over the day')
    plt.tight_layout(); plt.show()
    cycle_df.tail(3)
""",
    ),
    (
        "md",
        """
## 4. Regime guards — vol-up tripwire + perp depth

A trade is allowed only if **all** the following hold (in addition to the `d` and price filters):

- `RV_5m / RV_60m ≤ 1.5` — no vol-regime change in progress.
- BTC perp top-10 level depth ≥ 50% of session-median — book is healthy enough that a benign trade can exit on stop without massive slippage.
- (Operational) Not within ±60 min of a scheduled macro release (CPI/FOMC/NFP/ECB). External calendar; not enforced in code.

The small outcome market is *not* a manipulation risk source: the maximum profit available to an attacker scales with the binary's open interest, while the cost to push BTC mark through the strike scales with HL perp depth. Until the binary is materially deeper than today's market, manipulation cost dominates by orders of magnitude. The depth check below is a guard against **uncorrelated** big BTC moves, not against directed manipulation.
""",
    ),
    (
        "code",
        """
# Vol-regime ratio
vol_ratio = (rv_5m / rv_60m).clip(0.1, 5)
fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(vol_ratio.index, vol_ratio, lw=0.9)
ax.axhline(STRAT['vol_ratio_max'], ls='--', color='red',
           label=f"blackout above {STRAT['vol_ratio_max']}")
ax.axhline(1.0, ls=':', color='grey')
ax.set_ylabel('RV_5m / RV_60m'); ax.legend(); ax.set_title('Vol-regime ratio (entry blackout above red line)')
plt.tight_layout(); plt.show()

# HL BTC perp top-of-book depth
book = load_df(f'''
    SELECT local_recv_ts, bid_px, bid_sz, ask_px, ask_sz
    FROM read_parquet('{glob_for(venue='hyperliquid', product_type='perp', event='book_snapshot', symbol='BTC')}',
                      hive_partitioning=true)
    ORDER BY local_recv_ts
''')
if book.empty:
    print('No HL perp book snapshots — depth guard cannot be calibrated yet.')
else:
    book['t'] = pd.to_datetime(book.local_recv_ts, unit='ns', utc=True)
    book = book.set_index('t').sort_index()
    def top_n_notional(px_arr, sz_arr, n=5):
        if px_arr is None or len(px_arr) == 0: return np.nan
        k = min(n, len(px_arr))
        return float(np.sum(np.array(px_arr[:k]) * np.array(sz_arr[:k])))
    book['bid_top5_usd'] = [top_n_notional(p, s) for p, s in zip(book.bid_px, book.bid_sz)]
    book['ask_top5_usd'] = [top_n_notional(p, s) for p, s in zip(book.ask_px, book.ask_sz)]
    book['top10_usd']    = book.bid_top5_usd + book.ask_top5_usd

    median_depth = book['top10_usd'].median()
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(book.index, book.top10_usd / 1e6, lw=0.8)
    ax.axhline(median_depth / 1e6, ls='--', color='black',
               label=f'session median = ${median_depth/1e6:.2f}M')
    ax.axhline(STRAT['perp_depth_min_frac'] * median_depth / 1e6, ls='--', color='red',
               label=f"depth-guard floor = {STRAT['perp_depth_min_frac']:.0%} of median")
    ax.set_ylabel('top-10 levels notional ($M)'); ax.legend()
    ax.set_title('HL BTC perp top-of-book depth')
    plt.tight_layout(); plt.show()
    print(f'median top-10 depth: ${median_depth:,.0f}')
""",
    ),
    (
        "md",
        """
## 5. Outcome-side BBO + the binary spread/depth guard

Loads the binary BBO if collected. If not yet present, this cell simply prints a status note — the recorder is already configured to capture `prediction_binary` BBO (see `config/symbols.yaml`); we just need it to accumulate cycles.
""",
    ),
    (
        "code",
        """
def load_binary_bbo(symbol):
    g = glob_for(venue='hyperliquid', product_type='prediction_binary', event='bbo', symbol=symbol)
    try:
        df = load_df(f'''
            SELECT local_recv_ts, bid_px, bid_sz, ask_px, ask_sz
            FROM read_parquet('{g}', hive_partitioning=true)
            WHERE bid_px > 0 AND ask_px > 0
            ORDER BY local_recv_ts
        ''')
    except Exception as e:
        return pd.DataFrame()
    if df.empty: return df
    df['t']      = pd.to_datetime(df.local_recv_ts, unit='ns', utc=True)
    df['mid']    = 0.5 * (df.bid_px + df.ask_px)
    df['spread'] = df.ask_px - df.bid_px
    df['top1_usd'] = df.bid_sz * df.bid_px + df.ask_sz * df.ask_px
    return df.set_index('t')

if meta.empty:
    yes_bbo = no_bbo = pd.DataFrame()
    print('no metadata; skipping binary BBO load.')
else:
    pair = meta.dropna(subset=['expiry']).sort_values('expiry').iloc[-2:]   # last YES/NO pair
    yes_sym = pair[pair.side_idx == 0].symbol.iloc[0] if (pair.side_idx == 0).any() else None
    no_sym  = pair[pair.side_idx == 1].symbol.iloc[0] if (pair.side_idx == 1).any() else None
    yes_bbo = load_binary_bbo(yes_sym) if yes_sym else pd.DataFrame()
    no_bbo  = load_binary_bbo(no_sym)  if no_sym  else pd.DataFrame()
    print(f'YES {yes_sym}: {len(yes_bbo):,} BBO rows | NO {no_sym}: {len(no_bbo):,} BBO rows')

if yes_bbo.empty:
    print('Binary BBO not yet collected for this cycle. Re-run after recorder accumulates ≥ 1 day.')
else:
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True,
                             gridspec_kw={'height_ratios': [1.6, 1]})
    axes[0].plot(yes_bbo.index, yes_bbo.mid, lw=1.0, label=f'{yes_sym} mid')
    axes[0].fill_between(yes_bbo.index, yes_bbo.bid_px, yes_bbo.ask_px, alpha=0.15)
    axes[0].axhspan(*STRAT['price_band'], color='green', alpha=0.10, label='entry price band')
    axes[0].set_ylabel('YES price'); axes[0].set_ylim(0, 1); axes[0].legend(loc='lower right', fontsize=8)
    axes[1].plot(yes_bbo.index, yes_bbo.spread * 100, lw=0.9, color='C3', label='spread (¢)')
    axes[1].axhline(STRAT['binary_spread_max_cents'], ls='--', color='red',
                    label=f"max spread = {STRAT['binary_spread_max_cents']}¢")
    axes[1].set_ylabel('spread (cents)'); axes[1].legend(loc='upper right', fontsize=8)
    fig.suptitle(f'{yes_sym} binary BBO with strategy bands')
    plt.tight_layout(); plt.show()
""",
    ),
    (
        "md",
        """
## 6. Backtest scaffold — single-cycle simulator

Given a cycle (`K`, `expiry`, plus the `mark_1s`, `rv_60m`, `book`, `yes_bbo`, `no_bbo` series), the simulator walks the entry window once per second:

- **Gate check** at each tick `t`: `T_remaining ∈ window`, `d_t ≥ d_min`, `mid_t ∈ price_band`, `vol_ratio_t ≤ 1.5`, `perp_depth_t ≥ 0.5 × median`, binary spread `≤ 1.5¢`.
- On first pass: enter YES at `ask_t` for `trade_size_usd / ask_t` units. Place stop on `mid ≤ stop_price` AND `d ≤ stop_d` (whichever fires first).
- Walk forward; if any stop fires, exit at next bid; else hold to settlement.
- Realized PnL: at settlement `S_settle ≥ K → 1, else → 0`. Per-unit PnL = `(payoff - entry_ask)` if held; `(stop_bid - entry_ask)` if stopped.

The simulator emits one `TradeResult` per cycle (or `None` if no entry condition was met). Stop-loss sweeps and option comparisons reuse this primitive.
""",
    ),
    (
        "code",
        """
@dataclass
class TradeResult:
    cycle_symbol: str
    K: float
    expiry: pd.Timestamp
    entry_t: pd.Timestamp | None
    entry_ask: float | None
    exit_t:  pd.Timestamp | None
    exit_px: float | None
    exit_reason: str
    d_at_entry: float | None
    sigma_at_entry: float | None
    settle_S: float | None
    payoff: float | None
    pnl_per_unit: float | None
    units: float | None
    pnl_usd: float | None

def simulate_cycle(K, expiry, *, sym, mark_1s, rv_60m, book, yes_bbo, strat=STRAT) -> TradeResult | None:
    \"\"\"Single late-YES trade with stop-loss. Returns None if no entry condition was ever met.\"\"\"
    if yes_bbo.empty:
        return None

    window_start = expiry - pd.Timedelta(minutes=strat['entry_window_min'][0])
    window_end   = expiry - pd.Timedelta(minutes=strat['entry_window_min'][1])

    # Pre-compute decision-grid features on yes_bbo's timestamps.
    grid = yes_bbo[['bid_px','ask_px','mid','spread']].copy()
    grid = grid.loc[window_start:expiry]
    if grid.empty: return None

    # Align mark, sigma, depth, vol-ratio, no_mid onto grid via asof-merge (forward-fill).
    grid = grid.join(mark_1s.rename('S').reindex(grid.index, method='ffill'))
    grid['sigma']    = rv_60m.reindex(grid.index, method='ffill').clip(lower=strat['vol_floor'])
    grid['vol_ratio']= (rv_5m / rv_60m).reindex(grid.index, method='ffill')
    if not book.empty and 'top10_usd' in book.columns:
        depth_med = book['top10_usd'].median()
        grid['depth_ok'] = (book['top10_usd'].reindex(grid.index, method='ffill')
                            >= strat['perp_depth_min_frac'] * depth_med)
    else:
        grid['depth_ok'] = True
    grid['tau_yr'] = ((expiry - grid.index).total_seconds() / SEC_PER_YEAR).clip(lower=1e-12)
    grid['d']      = (grid.S - K) / (grid.S * grid.sigma * np.sqrt(grid.tau_yr))

    # Entry gate
    enter_mask = (
        (grid.index <= window_end)
        & (grid['d'] >= strat['d_min'])
        & (grid['mid'].between(*strat['price_band']))
        & (grid['vol_ratio'].fillna(1.0) <= strat['vol_ratio_max'])
        & (grid['spread'] <= strat['binary_spread_max_cents'] / 100.0)
        & (grid['depth_ok'])
    )
    if not enter_mask.any():
        return TradeResult(sym, K, expiry, None, None, None, None, 'no-entry',
                           None, None, None, None, None, None, None)

    entry_t   = grid.index[enter_mask][0]
    entry_row = grid.loc[entry_t]
    entry_ask = float(entry_row['ask_px'])
    units     = strat['trade_size_usd'] / entry_ask

    # Walk forward; check stop on every later tick.
    fwd = grid.loc[entry_t:]
    stop_mask = (fwd['mid'] <= strat['stop_price']) | (fwd['d'] <= strat['stop_d'])
    if stop_mask.any():
        exit_t   = fwd.index[stop_mask][0]
        exit_row = fwd.loc[exit_t]
        exit_px  = float(exit_row['bid_px'])         # exit on bid
        reason   = 'stop_d' if exit_row['d'] <= strat['stop_d'] else 'stop_price'
        # The settlement payoff doesn't matter once stopped; we record it for context.
    else:
        # Held to settlement.
        S_at_settle = float(mark_1s.asof(expiry))
        payoff = 1.0 if S_at_settle >= K else 0.0
        exit_t  = expiry
        exit_px = payoff
        reason  = 'settle_yes' if payoff == 1.0 else 'settle_no'

    pnl_per_unit = (exit_px - entry_ask) if reason.startswith('stop') else (exit_px - entry_ask)
    settle_S = float(mark_1s.asof(expiry)) if expiry in mark_1s.index or len(mark_1s.loc[:expiry])>0 else None
    return TradeResult(
        cycle_symbol=sym, K=K, expiry=expiry,
        entry_t=entry_t, entry_ask=entry_ask,
        exit_t=exit_t, exit_px=exit_px,
        exit_reason=reason,
        d_at_entry=float(entry_row['d']),
        sigma_at_entry=float(entry_row['sigma']),
        settle_S=settle_S,
        payoff=(1.0 if settle_S is not None and settle_S >= K else 0.0) if settle_S else None,
        pnl_per_unit=pnl_per_unit,
        units=units,
        pnl_usd=pnl_per_unit * units,
    )

# Run the simulator on the most-recent cycle (if data is present).
results: list[TradeResult] = []
if not meta.empty and not yes_bbo.empty:
    pair = meta.dropna(subset=['strike','expiry']).sort_values('expiry').iloc[-2:]
    yes  = pair[pair.side_idx == 0].iloc[0]
    res = simulate_cycle(float(yes.strike), pd.Timestamp(yes.expiry),
                         sym=yes.symbol, mark_1s=mark_1s, rv_60m=rv_60m,
                         book=book, yes_bbo=yes_bbo)
    if res is not None: results.append(res)
    print(res)
else:
    print('Simulator ready; need outcome BBO + multiple historical cycles before running.')
""",
    ),
    (
        "md",
        """
## 7. Stop-loss sweep — what's the right `stop_price` and `stop_d`?

Vary one stop knob at a time over many cycles. Plot:

- **EV per trade** — does a tighter stop improve net EV (false stops vs catastrophic saves)?
- **95th-percentile loss** — drawdown control.
- **Hit rate** — fraction of trades that resolve YES (or stop) profitably.

Analytical reminder: a stop improves EV iff

$$\\frac{g}{f} > \\frac{q}{1-q} \\cdot \\frac{1-s}{s}$$

where `q` is true YES probability, `s` is stop price, `f` is false-stop rate (stopped on a YES that would have resolved), `g` is save rate (stopped on a NO that would have resolved). For `q=0.99, s=0.85` the bar is `g/f > 17`. For `q=0.95` it's `g/f > 3.3`. The lower your conviction, the more useful the stop.

In early validation `q` is uncertain, so the stop is correct even if EV-neutral — it bounds drawdown on a $1k bankroll.
""",
    ),
    (
        "code",
        """
def all_cycles_meta(meta_df):
    \"\"\"Group meta by (outcome, expiry) and return one (sym_yes, K, expiry) per cycle.\"\"\"
    if meta_df.empty: return []
    yes = meta_df[(meta_df.side_idx == 0)].dropna(subset=['strike','expiry'])
    return [(r.symbol, float(r.strike), pd.Timestamp(r.expiry)) for r in yes.itertuples()]

def sweep_stop_price(stop_grid, cycles, **base):
    rows = []
    for s in stop_grid:
        strat = {**STRAT, **base, 'stop_price': s}
        trades = []
        for sym, K, exp in cycles:
            yb = load_binary_bbo(sym)
            if yb.empty: continue
            tr = simulate_cycle(K, exp, sym=sym, mark_1s=mark_1s, rv_60m=rv_60m,
                                book=book, yes_bbo=yb, strat=strat)
            if tr is not None and tr.pnl_per_unit is not None:
                trades.append(tr)
        if not trades: continue
        pnls = np.array([t.pnl_per_unit for t in trades])
        rows.append({
            'stop_price': s,
            'n_trades': len(trades),
            'mean_pnl_per_unit': pnls.mean(),
            'p05_pnl': np.quantile(pnls, 0.05),
            'p95_pnl': np.quantile(pnls, 0.95),
            'win_rate': float((pnls > 0).mean()),
        })
    return pd.DataFrame(rows)

cycles = all_cycles_meta(meta)
print(f'cycles available: {len(cycles)}')

if len(cycles) >= 5 and not yes_bbo.empty:
    sweep = sweep_stop_price(np.linspace(0.70, 0.92, 12), cycles)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(sweep.stop_price, sweep.mean_pnl_per_unit, '-o'); axes[0].axhline(0, color='grey', lw=0.5)
    axes[0].set_title('mean PnL per unit'); axes[0].set_xlabel('stop_price')
    axes[1].plot(sweep.stop_price, sweep.p05_pnl, '-o', color='C3'); axes[1].axhline(0, color='grey', lw=0.5)
    axes[1].set_title('5th-pct PnL (left tail)'); axes[1].set_xlabel('stop_price')
    axes[2].plot(sweep.stop_price, sweep.win_rate, '-o', color='C2')
    axes[2].set_title('win rate'); axes[2].set_xlabel('stop_price')
    plt.tight_layout(); plt.show()
    print(sweep)
else:
    print('Need ≥ 5 cycles of binary BBO before stop-sweep is informative.')
    print('Today: cycles=', len(cycles), 'binary_bbo_present=', not yes_bbo.empty)
""",
    ),
    (
        "md",
        """
## 8. Calibration plot — `d` vs realized YES rate

The central reliability check: if our `q_model` says 99%, does the market really resolve YES 99% of the time? Bucket cycles by `d` (measured at T-30min) and plot the empirical YES rate vs the modeled rate.

- A clean monotonic curve hugging the diagonal = `q_model` is calibrated and entry rule is sound.
- Persistent below-diagonal at high `d` = model is too confident; tighten `d_min` or use a higher σ proxy.
- Persistent above-diagonal at low `d` = market is too cautious; you have edge even at lower `d`.

This plot is the single most-important output of the first month of paper trading.
""",
    ),
    (
        "code",
        """
def calibration_table(cycles, eval_offset_min=30):
    \"\"\"For each historical cycle, sample (d, q_model) at T - eval_offset and pair with realized YES.\"\"\"
    rows = []
    for sym, K, exp in cycles:
        eval_t = exp - pd.Timedelta(minutes=eval_offset_min)
        if eval_t not in mark_1s.index and len(mark_1s.loc[:eval_t]) == 0:
            continue
        S      = float(mark_1s.asof(eval_t))
        sigma  = float(rv_60m.asof(eval_t)) if not pd.isna(rv_60m.asof(eval_t)) else STRAT['vol_floor']
        sigma  = max(sigma, STRAT['vol_floor'])
        tau    = max((exp - eval_t).total_seconds() / SEC_PER_YEAR, 1e-12)
        d_val  = (S - K) / (S * sigma * np.sqrt(tau))
        q_pred = float(q_model(np.array([S]), K, np.array([sigma]), np.array([tau]))[0])
        S_settle = float(mark_1s.asof(exp))
        rows.append({'symbol': sym, 'K': K, 'expiry': exp,
                     'S_eval': S, 'sigma_eval': sigma, 'd': d_val, 'q_pred': q_pred,
                     'S_settle': S_settle, 'realized_yes': int(S_settle >= K)})
    return pd.DataFrame(rows)

cal = calibration_table(cycles, eval_offset_min=30)
print(f'calibration sample size: {len(cal)}')

if len(cal) >= 10:
    # Bucket on q_pred (= calibration plot a la weather forecasters)
    cal['bucket'] = pd.cut(cal.q_pred, bins=[0, 0.5, 0.7, 0.85, 0.93, 0.97, 0.99, 1.0])
    grp = cal.groupby('bucket').agg(predicted=('q_pred','mean'),
                                    realized=('realized_yes','mean'),
                                    n=('symbol','size')).dropna()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0,1], [0,1], 'k--', lw=0.7, label='perfect calibration')
    ax.scatter(grp.predicted, grp.realized, s=60 + 3*grp.n, alpha=0.7)
    for _, r in grp.iterrows():
        ax.annotate(f'n={int(r.n)}', (r.predicted, r.realized), xytext=(4,4), textcoords='offset points', fontsize=8)
    ax.set_xlabel('predicted q'); ax.set_ylabel('realized YES rate')
    ax.set_title(f'Reliability diagram @ T-{30}min  (n={len(cal)})')
    ax.legend(); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    plt.tight_layout(); plt.show()
    print(grp)
else:
    print('Need ≥ 10 historical cycles for a meaningful reliability diagram.')
""",
    ),
    (
        "md",
        """
## 9. Strategy options — priority ordering

Four candidate strategies, scored on (i) capital efficiency on a $1k bankroll, (ii) operational complexity, (iii) clarity of testable edge, (iv) tail-risk control. Run them in this order; do not advance to the next one until the current one has 20+ paper or live trades and a reliability diagram that holds.

### Tier 1 — START HERE

**Strategy A: Mark-anchored late YES with `d`-stop and price-stop**

*One-liner:* Manually enter YES in the last 30–15 min when `d ≥ 3`, regime is calm, BTC perp depth is healthy, binary spread is tight, mid is 93–99¢. Exit on `d ≤ 1.5` or mid `≤ 0.85`, whichever first.

- **Why first:** Settlement source (HL mark) is exactly what `q_model` predicts, so calibration is testable on the data we already collect for the perp. Single-leg, a single human can monitor 1 trade/day. Stop is informative without being a hedge — exactly your constraint.
- **Capital:** $100/trade (10% of $1k bankroll). Max loss per trade: ~$10–12 with stop at 0.85.
- **What to validate first:** the reliability diagram in section 8. If `q_pred ≈ 0.95` buckets resolve YES <90%, your σ proxy is wrong; refit before going live.
- **Operational footprint:** open `07` once a day in the last 60 min before 06:00 UTC; check the gates pass; place the trade; set a programmatic stop on HL UI; let it ride.

### Tier 2 — once Tier 1 has 20+ trades and calibration holds

**Strategy B: Cross-side pair (YES + cheap NO insurance)**

*One-liner:* Same entry as Strategy A, but also buy a small position of the *opposite* side (e.g. 1¢–4¢) so total max-loss is bounded by the insurance leg, eliminating the need to monitor a stop in real time.

- **Why second:** Same edge source as A but converts the trade from "asymmetric with stop" to "defined-loss". You give up a little EV (paying the insurance premium) for operational simplicity (no stop monitoring required).
- **When to switch from A→B:** if the stop-loss sweep in section 7 shows the binary-price stop has high false-trigger rate (`f` large), the implicit insurance from a cross-side hedge is mathematically equivalent to a perfect stop.
- **Capital:** $100 YES + ~$3–8 NO. Max loss bounded by `(YES_paid - 0)` minus `(NO_paid * 1)` recovered = predetermined.
- **Constraint:** requires the NO side to have liquidity. If NO is one-sided (no MM), this strategy is not available.

### Tier 3 — once binary microstructure is mapped

**Strategy C: Settlement-window passive provision**

*One-liner:* Post resting bids at 0.94–0.96 in the last 30 min instead of crossing the spread. Get filled only when an aggressor lifts you. Stop based on perp mark moving toward strike.

- **Why third:** Captures spread instead of paying it. Higher gross EV per filled trade. But it's *adversely-selected by construction* — you are most likely to fill exactly when someone has news that justifies the lower price.
- **Prerequisite analysis:** a Phase-P4-style markout study on the binary BBO (when a passive bid is hit, what does the binary mid do over the next 30s?). Until that markout is positive net of fees, do not provide.
- **Capital:** same $100, but expect lower fill rate so daily PnL is more variable.
- **Risk gotcha:** during HL outages a resting order can be filled while you have no working stop. **Do not pursue this until automated stop-monitoring is in place.**

### Tier 4 — speculative, defer until ≥3 months of data

**Strategy D: Settlement-pin / gamma-magnet directional**

*One-liner:* Predict whether HL BTC mark "magnets" toward the strike near 06:00 UTC (a known options gamma-hedging effect) and pre-position accordingly. Pure microstructure / order-flow trade.

- **Why last:** Requires the binary to be liquid enough that delta-hedgers exist; today's HIP-4 binary is too small. Only run this once binary OI is at least 10× current.
- **Inputs needed:** order-flow imbalance in the last 15 min, perp depth asymmetry near strike, end-of-day funding-print history.

---

### Concrete starter parameters for Strategy A

| Parameter | Starting value | How we know it's right |
|---|---|---|
| Entry window | T-30min to T-5min | Slippage + edge-decay analysis from collected BBO |
| `d_min` | 3.0 | Reliability diagram in section 8 |
| Price band | [0.93, 0.99] | Below 93¢ payoff doesn't cover frictions; above 99¢ insufficient liquidity |
| Vol-ratio max | 1.5 | Section 4 (RV_5m / RV_60m) |
| BTC perp depth min | 50% of session-median | Section 4 depth chart |
| Binary spread max | 1.5¢ | Section 5 |
| Position size | $100 (10% of bankroll) | Bump to $200 after 20 paper or 10 live wins that hit calibration target |
| Stop A: `d`-stop | exit if d ≤ 1.5 | Sweep in section 7 |
| Stop B: price-stop | exit if mid ≤ 0.85 | Sweep in section 7 |
| Macro blackout | ±60 min around CPI/FOMC/NFP/ECB | External calendar |

### What this notebook needs to evolve into a live signal

1. **Outcome-side BBO/book/trade data** — recorder is already configured; we just need accumulation. After ~7 cycles section 7's stop sweep starts being informative; after ~20 cycles section 8's reliability diagram is statistically meaningful.
2. **Per-trade journal** — a small parquet at `data/journal/strategy=late-yes/...parquet` capturing `(entry_t, d, S, K, sigma, mid, exit_reason, pnl)` for every paper and live trade.
3. **Daily 06:00 UTC report** — schedule this notebook to run headlessly at 06:01 UTC and emit `docs/reports/late-yes-YYYY-MM-DD.md` with the cycle's chart pack + the reliability diagram refreshed.

### What to **not** do

- Do not enter inside the last 5 minutes — gamma is too violent and slippage compounds.
- Do not enter at price ≥ 0.995. Net edge is below fee floor and slippage on exit is unbounded.
- Do not trade across multiple cycles in one day (until 1 cycle per day produces calibrated results).
- Do not skip the macro-event blackout — vol spikes and depth collapse simultaneously.
""",
    ),
]


def main() -> None:
    notebooks = {
        "01-data-quality.ipynb": P0,
        "02-cross-venue-basis.ipynb": P1,
        "03-microstructure.ipynb": P2,
        "04-trade-flow-leadlag.ipynb": P3,
        "05-hip4-binaries.ipynb": P5,
        "06-hip4-pricing-from-perp.ipynb": P6,
        "07-binary-late-yes-strategy.ipynb": P7,
    }
    for name, cells in notebooks.items():
        nb = _build([*cells, HYPOTHESIS_FOOTER])
        path = OUT / name
        with path.open("w") as f:
            nbf.write(nb, f)
        print(f"wrote {path.relative_to(REPO)}  ({len(cells) + 1} cells, +footer)")


if __name__ == "__main__":
    main()
