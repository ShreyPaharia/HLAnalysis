# HLAnalysis — Market Data Analysis Plan

The recorder is up; this document is the analytical roadmap. The goal is to derive
the inputs needed to design and risk-budget a market-making engine for Hyperliquid
BTC perp + HIP-4 binaries, using Binance as the reference and hedge venue.

The framing throughout: **how an experienced hedge-fund quant approaches a new
dataset** — sanity first, microstructure second, edge attribution third, strategy
calibration last. No analysis on data you haven't validated. No strategy on
microstructure you haven't measured.

---

## 0. Stack and conventions

| Layer | Tool | Why |
|---|---|---|
| Storage | Hive-partitioned parquet (zstd) | Already in place; cheap to query |
| Query | DuckDB direct on parquet | No ETL, columnar, joins fast across event types |
| Transform | Polars (preferred) or Pandas | Polars is ~5–10× faster on parquet |
| Charts | Plotly (interactive), matplotlib (static) | Interactive for exploration, static for reports |
| Notebooks | jupytext-paired markdown | Reviewable diffs, runnable as `.py` |

All time joins use `local_recv_ts` (single clock, monotonic on this host) unless
explicitly comparing against `exchange_ts`. Use UTC throughout; do not convert
to local time until the very last rendering step. Never use floats for prices —
use `Decimal` only at the boundary where rounding matters; `float64` is fine for
analytics provided you are aware of the ~15-significant-digit limit.

Common base path: `data/venue=*/product_type=*/mechanism=*/event=*/symbol=*/date=*/hour=*/*.parquet`.
Useful idiom:

```python
import duckdb
con = duckdb.connect()
con.execute("CREATE VIEW trades AS SELECT * FROM read_parquet('data/**/event=trade/**/*.parquet', hive_partitioning=true)")
con.execute("CREATE VIEW bbo    AS SELECT * FROM read_parquet('data/**/event=bbo/**/*.parquet', hive_partitioning=true)")
con.execute("CREATE VIEW book   AS SELECT * FROM read_parquet('data/**/event=book_snapshot/**/*.parquet', hive_partitioning=true)")
```

(Parametrize `data/` to your data root.)

---

## 1. Priorities and pacing

A reasonable two-week sprint, assuming the recorder accumulates at least 24h of
data before serious work begins:

| Phase | Days | Deliverable |
|---|---|---|
| P0 — Data quality and sanity | 1 | `01-data-quality.ipynb` + a one-page diagnostic |
| P1 — Cross-venue basis | 2–3 | Basis time series, half-life, distribution |
| P2 — Microstructure (spread, depth, churn) | 3–4 | Spread + depth heatmaps, intraday seasonality |
| P3 — Trade flow and lead-lag | 5–6 | Aggressor balance, cross-venue CCF, Granger |
| P4 — Adverse selection / markout | 7–8 | Markout-PnL by venue × size × time-of-day |
| P5 — HIP-4 binaries (digital options) | 9–11 | Implied vs theoretical, greeks, settlement event study |
| P6 — Volatility and funding regimes | 12 | Vol cones, funding–vol relationship |
| P7 — MM strategy calibration | 13–14 | Spread vs fill rate vs adverse curve, hedge cost model |

P0 must finish before P1+. Don't analyse contaminated data.

---

## 2. Phase P0 — Data quality and sanity

### 2.1 Coverage

What an analyst checks on day one of any new dataset: **are the lights on**?

- Rows per (venue, symbol, event, hour). Look for hour-buckets with zero rows
  for events that should be continuous (bbo, book). A dropout of >10 minutes
  is evidence of a connectivity issue or recorder crash worth correlating
  against `health` events.
- Daily roll-ups. Is volume in a reasonable range vs the venue's published
  daily volume figures? An order-of-magnitude mismatch indicates a wrong
  symbol, missing channel, or geo-restricted stream.

```sql
-- Row counts per (venue, symbol, event), 1h buckets
SELECT
  venue,
  product_type,
  symbol,
  event,
  date_trunc('hour', to_timestamp(local_recv_ts/1e9)) AS hour,
  count(*) AS rows
FROM read_parquet('data/**/*.parquet', hive_partitioning=true)
GROUP BY 1,2,3,4,5
ORDER BY 1,2,3,4,5;
```

**Chart 1.** Stacked bar, x = hour, y = rows, colour = (venue, event_type).
One row per (symbol). Visually obvious if any market goes silent.

**Chart 2.** Reconnect timeline: x = time, marker per `health.kind ∈ {connected,
reconnect, outcome-rolled, outcome-discovered}`. Overlay onto a price chart of
the relevant venue. Reveals whether reconnects cluster around volatility events
(usually a sign you're hitting venue rate limits) or are random (hardware/network).

### 2.2 Latency and clock skew

Both `exchange_ts` and `local_recv_ts` are recorded; the difference is
informative even when noisy. Negative p50 latency means the local clock is
behind the venue's clock — an NTP issue, not a network issue.

**Per-venue `exchange_ts` semantics — read this before computing latencies:**

| (venue, event)            | `exchange_ts` source                                  | Notes |
| ------------------------- | ----------------------------------------------------- | ----- |
| binance perp bbo / trade  | `T` field (event/trade time, ms)                      | Real exchange-side ts. ~62ms typical recv latency. |
| binance spot bbo          | **0 (sentinel — not provided)**                       | `@bookTicker` payload has no ts field. Filter `WHERE exchange_ts > 0` before any latency calc. |
| binance spot book         | **0 (sentinel — not provided)**                       | `@depth<N>` payload has no ts field. Same filter. |
| binance spot trade        | `T` field (trade time, ms)                            | Real exchange-side ts. |
| hyperliquid bbo / book    | `time` field (block time, ms)                         | Block-finality time, not ws send time. |
| hyperliquid trade         | `time` field = **block time** (ms; mirrored to `block_ts`) | HL ships trades in batched ws messages several seconds AFTER block finalization. Median `local_recv_ts - exchange_ts` ~ 5s is real publishing lag, not network. Use `block_ts` when you need to be explicit you're talking about block time. |
| hyperliquid activeAssetCtx (mark/funding/oracle) | `recv_ns` (no ts in ctx payload)            | Not a real exchange ts; `local_recv_ts - exchange_ts` ≈ 0. |

```sql
SELECT
  venue, product_type, symbol, event,
  approx_quantile((local_recv_ts - exchange_ts)/1e6, 0.5)  AS p50_ms,
  approx_quantile((local_recv_ts - exchange_ts)/1e6, 0.95) AS p95_ms,
  approx_quantile((local_recv_ts - exchange_ts)/1e6, 0.99) AS p99_ms
FROM read_parquet('data/**/*.parquet', hive_partitioning=true)
WHERE event IN ('trade','bbo','book_snapshot')
  AND symbol != '*'
  AND exchange_ts > 0  -- exclude binance spot bbo/book where exchange_ts is sentinel
GROUP BY 1,2,3,4
ORDER BY 1,2,3,4;
```

**Chart 3.** Box-whisker per (venue, event). Tail watch: p99 > 1s persistently
means the venue's gateway is queueing during bursts. p99 spikes that line up
with reconnects mean we lost data.

**Diagnostic.** `local_recv_ts - exchange_ts` should be:
- Positive (we receive after the event).
- Slowly drifting (not jumping).
- Roughly stable across the day, with a small uplift during exchange peak hours.

A bimodal distribution suggests two paths (e.g., re-routed traffic). A
monotonically increasing distribution suggests a slow leak in your async event
loop or buffer build-up — investigate.

### 2.3 Cross-venue price agreement

A simple sanity check: HL `oraclePx` ≈ Binance `indexPrice` (when we record it)
≈ Binance spot mid ≈ HL mid, all within a few bps. If any of these diverge by
>20 bps for sustained periods, you have a feed bug or you've discovered an arb
(spoiler: you haven't).

```sql
WITH
  hl_oracle AS (
    SELECT date_trunc('second', to_timestamp(local_recv_ts/1e9)) AS t, avg(oracle_px) AS px
    FROM read_parquet('data/**/event=oracle/**/symbol=BTC/**/*.parquet')
    GROUP BY 1
  ),
  bnc_spot_mid AS (
    SELECT date_trunc('second', to_timestamp(local_recv_ts/1e9)) AS t,
           avg((bid_px + ask_px)/2) AS px
    FROM read_parquet('data/**/event=bbo/symbol=BTCUSDT/**/*.parquet')
    WHERE product_type = 'spot'
    GROUP BY 1
  )
SELECT
  hl_oracle.t,
  hl_oracle.px AS hl_oracle,
  bnc_spot_mid.px AS bnc_spot,
  (hl_oracle.px - bnc_spot_mid.px) / bnc_spot_mid.px * 10000 AS bps_diff
FROM hl_oracle JOIN bnc_spot_mid USING (t)
ORDER BY t;
```

**Chart 4.** Time series, two lines + a third on a secondary axis showing
`bps_diff`. The bps line should hover within ±5 bps for BTC.

### 2.4 Schema integrity

- Every event has a non-null exchange_ts and local_recv_ts? (count nulls)
- Are there duplicate events (same `seq` or `trade_id`)?
- Trade prices within (best_bid_at_time, best_ask_at_time)? (Off-book trades
  appear here on some venues — flag them, don't reject them.)

### 2.5 What "good" looks like at the end of P0

- ≥99% hour-buckets non-empty for continuous events.
- p99 within-venue latency < 500 ms.
- Cross-venue oracle/index/spot agreement within ±5 bps for BTC.
- A short markdown report saved under `docs/reports/01-data-quality-YYYY-MM-DD.md`
  with the four charts above and a single sentence verdict at the top.

---

## 3. Phase P1 — Cross-venue basis and carry

The meat of why we record both venues. Three basis series matter:

1. **Perp-perp basis**: `mid(HL perp) − mid(Binance perp)`. The clean cross-venue
   comparison; differences are venue mispricings.
2. **Perp-spot basis**: `mid(HL perp) − mid(Binance spot)`. The cash-and-carry
   basis; theoretical = funding income discounted to expiry. For perpetuals,
   "expiry" is replaced by the funding cadence.
3. **Funding-implied carry**: After normalising both venues to the same time
   horizon (e.g. 1h), compare funding-rate spreads with realised perp-spot basis
   drift.

### 3.1 Perp-perp basis

```sql
WITH
  hl AS (
    SELECT (local_recv_ts/1e9)::int AS sec, last((bid_px + ask_px)/2) AS mid
    FROM read_parquet('data/**/event=bbo/symbol=BTC/**/*.parquet')
    WHERE product_type='perp'
    GROUP BY 1
  ),
  bnc AS (
    SELECT (local_recv_ts/1e9)::int AS sec, last((bid_px + ask_px)/2) AS mid
    FROM read_parquet('data/**/event=bbo/symbol=BTCUSDT/**/*.parquet')
    WHERE product_type='perp'
    GROUP BY 1
  )
SELECT
  sec,
  hl.mid AS hl_mid,
  bnc.mid AS bnc_mid,
  (hl.mid - bnc.mid) AS basis_usd,
  (hl.mid - bnc.mid) / bnc.mid * 10000 AS basis_bps
FROM hl JOIN bnc USING (sec)
ORDER BY sec;
```

### 3.2 Half-life of basis

Fit `basis_t = a + b * basis_{t-1} + e_t` (AR(1)). Half-life = `ln(0.5) / ln(b)`
(in seconds, since we sampled at 1s). Half-life < 30s means the basis is
arbitraged out fast (you cannot trade it without colocation and signing speed).
Half-life > a few minutes means there's room.

### 3.3 Distribution and tails

Plot the histogram of `basis_bps`. Look for:

- **Mean ≠ 0**: persistent venue mispricing. On HL/Binance for BTC this is
  usually small but non-zero (HL trades a few bps rich vs Binance during Asian
  hours, e.g.).
- **Heavy tails**: the basis blows out during news or liquidations. The
  blow-outs are when an MM either makes a fortune or gets stopped. P99 vs
  median tells you the tail-risk-to-mean ratio of the basis trade.
- **Skew**: directional asymmetry. Often present and an artifact of one venue
  being slow to update.

### 3.4 Charts

**Chart 5.** Three-panel time series, sharing x-axis:
- Top: HL perp mid, Binance perp mid (overlaid)
- Middle: basis in bps
- Bottom: rolling 5-min stdev of basis

**Chart 6.** Histogram of basis with marginal kernel density estimate. Annotate
mean, median, p1, p99.

**Chart 7.** Basis vs HL trade volume scatter (1-minute aggregation), one point
per minute. Sometimes basis explodes when one side is illiquid; this plot shows
whether basis is a function of activity.

### 3.5 Funding-implied carry

```sql
-- Normalize both funding rates to 1h-equivalent
WITH funding AS (
  SELECT venue, symbol, exchange_ts, local_recv_ts,
         CASE
           WHEN venue = 'hyperliquid' THEN funding_rate         -- already 1h on HL
           WHEN venue = 'binance'     THEN funding_rate / 8     -- 8h -> 1h
         END AS hourly_rate
  FROM read_parquet('data/**/event=funding/**/*.parquet')
)
SELECT
  date_trunc('minute', to_timestamp(local_recv_ts/1e9)) AS minute,
  max(CASE WHEN venue='hyperliquid' THEN hourly_rate END) AS hl_1h,
  max(CASE WHEN venue='binance'     THEN hourly_rate END) AS bnc_1h
FROM funding GROUP BY 1 ORDER BY 1;
```

**Chart 8.** Funding rate (1h-equivalent) overlaid for both venues. Annotate
the cross-venue spread (`hl_1h − bnc_1h`).

A persistent positive HL-Binance funding spread implies HL longs are paying
more than Binance longs — which should be reflected in HL's perp trading at a
discount to Binance's. Verify alignment.

### 3.6 Verdict at end of P1

Three numbers you should know cold:
1. Median perp-perp basis (in bps).
2. Half-life of basis to 1/e.
3. Annualised basis-trade Sharpe (assuming daily rebalance).

If half-life is sub-30s, the basis is not directly tradeable but is information
(it tells you which venue to fade vs follow during execution).

---

## 4. Phase P2 — Microstructure: spread, depth, churn

Everything the MM engine cares about. Each of the following is computed per
(venue, product, symbol).

### 4.1 Quoted spread

```sql
SELECT
  venue, product_type, symbol,
  date_trunc('minute', to_timestamp(local_recv_ts/1e9)) AS minute,
  approx_quantile(ask_px - bid_px, 0.5) AS spread_p50,
  approx_quantile(ask_px - bid_px, 0.95) AS spread_p95,
  approx_quantile((ask_px - bid_px) / ((ask_px + bid_px)/2) * 10000, 0.5) AS spread_bps_p50
FROM read_parquet('data/**/event=bbo/**/*.parquet')
GROUP BY 1,2,3,4 ORDER BY 1,2,3,4;
```

**Chart 9.** Time series of `spread_bps_p50`, one line per (venue, symbol).
Overlay vertical bands for funding settlement times (e.g. Binance 00:00, 08:00,
16:00 UTC) and for HIP-4 daily settlement (06:00 UTC). Spreads typically widen
in the minutes leading up to settlement.

### 4.2 Intraday seasonality of spread

Group spread by minute-of-day across the entire dataset:

```sql
SELECT
  venue, symbol,
  extract(hour FROM to_timestamp(local_recv_ts/1e9)) AS hour_utc,
  approx_quantile((ask_px - bid_px) / ((ask_px + bid_px)/2) * 10000, 0.5) AS spread_bps
FROM read_parquet('data/**/event=bbo/**/*.parquet')
GROUP BY 1,2,3 ORDER BY 1,2,3;
```

**Chart 10.** Heatmap, x = hour-of-day UTC, y = day, colour = spread_bps. Reveals
the Asia/EU/US session structure. BTC is typically tightest during US session
overlap, widest in the Asia-only window.

### 4.3 Depth profile

For each book snapshot, build the cumulative size at increasing distance from
mid: how much can you fill at 1bp, 5bp, 10bp, 25bp, 100bp away?

```sql
-- Depth at top-N levels per snapshot
WITH unnested AS (
  SELECT
    local_recv_ts, venue, symbol,
    (bid_px[1] + ask_px[1]) / 2 AS mid,
    unnest(bid_px) AS bp, unnest(bid_sz) AS bz,
    unnest(ask_px) AS ap, unnest(ask_sz) AS az
  FROM read_parquet('data/**/event=book_snapshot/**/*.parquet')
)
SELECT
  venue, symbol,
  date_trunc('minute', to_timestamp(local_recv_ts/1e9)) AS minute,
  sum(CASE WHEN (mid - bp) / mid <= 0.0001 THEN bz ELSE 0 END) AS bid_depth_1bp,
  sum(CASE WHEN (mid - bp) / mid <= 0.0005 THEN bz ELSE 0 END) AS bid_depth_5bp,
  sum(CASE WHEN (mid - bp) / mid <= 0.0010 THEN bz ELSE 0 END) AS bid_depth_10bp,
  sum(CASE WHEN (ap - mid) / mid <= 0.0001 THEN az ELSE 0 END) AS ask_depth_1bp,
  sum(CASE WHEN (ap - mid) / mid <= 0.0005 THEN az ELSE 0 END) AS ask_depth_5bp,
  sum(CASE WHEN (ap - mid) / mid <= 0.0010 THEN az ELSE 0 END) AS ask_depth_10bp
FROM unnested
GROUP BY 1,2,3 ORDER BY 1,2,3;
```

**Chart 11.** Two-panel heatmap, one per venue:
- x = time, y = price-level (in bps from mid), colour = log(size at that level).
- Reveals queue compression near the top of book and the typical depth shape.

**Chart 12.** Depth-bin time series at 5bp and 25bp. The ratio of these is a
measure of order book "shape" — a venue with steep walls vs flat depth.

### 4.4 Order book imbalance

Often the simplest and most useful microstructure feature.

```sql
SELECT
  local_recv_ts, venue, symbol,
  (bid_sz[1] - ask_sz[1]) / NULLIF(bid_sz[1] + ask_sz[1], 0) AS top_imbalance,
  (sum(bid_sz[1:5]) - sum(ask_sz[1:5])) / NULLIF(sum(bid_sz[1:5]) + sum(ask_sz[1:5]), 0) AS top5_imbalance
FROM read_parquet('data/**/event=book_snapshot/**/*.parquet');
```

For each imbalance measurement at time `t`, compute forward returns at
horizons 100ms, 1s, 5s, 30s. The classic finding: top imbalance is mildly
predictive of next-tick direction but the predictive power decays rapidly.

**Chart 13.** Bin the imbalance into deciles, plot mean forward return at each
horizon vs imbalance decile. A monotonic curve = imbalance is predictive. A
flat curve = the venue is too efficient for this naive feature.

### 4.5 Top-of-book churn

How often does the best bid or ask change? A venue with high churn has more
queue position turnover (good for passive MM if your latency is competitive,
bad if not).

```sql
WITH bbo_changes AS (
  SELECT
    venue, symbol, local_recv_ts,
    bid_px,
    lag(bid_px) OVER (PARTITION BY venue, symbol ORDER BY local_recv_ts) AS prev_bid_px,
    ask_px,
    lag(ask_px) OVER (PARTITION BY venue, symbol ORDER BY local_recv_ts) AS prev_ask_px
  FROM read_parquet('data/**/event=bbo/**/*.parquet')
)
SELECT
  venue, symbol,
  date_trunc('minute', to_timestamp(local_recv_ts/1e9)) AS minute,
  sum(CASE WHEN bid_px != prev_bid_px THEN 1 ELSE 0 END) AS bid_changes,
  sum(CASE WHEN ask_px != prev_ask_px THEN 1 ELSE 0 END) AS ask_changes
FROM bbo_changes GROUP BY 1,2,3 ORDER BY 1,2,3;
```

**Chart 14.** Churn rate (changes/sec) per venue, time series. Also report
churn-per-trade (changes per public trade) — a high ratio means the book is
moving without trades, often due to layering or HFT activity.

### 4.6 Effective vs quoted spread

Effective spread = `2 * |trade_px − mid_at_trade_time|`. Lower than quoted
spread means trades cluster on the inside (price improvement); higher means
sweeps. The *realised spread* (effective spread minus market-impact decay over
the next N seconds) is what an MM actually keeps.

This is heavy: it requires joining each trade against the BBO immediately
prior. Use an asof join in DuckDB:

```sql
WITH trade AS (
  SELECT venue, symbol, local_recv_ts, price, side
  FROM read_parquet('data/**/event=trade/**/*.parquet')
),
bbo AS (
  SELECT venue, symbol, local_recv_ts, (bid_px + ask_px)/2 AS mid
  FROM read_parquet('data/**/event=bbo/**/*.parquet')
)
SELECT t.venue, t.symbol, t.local_recv_ts,
       t.price, t.side, b.mid,
       2 * abs(t.price - b.mid) AS effective_spread,
       (t.price - b.mid) / b.mid * 10000 * (CASE t.side WHEN 'buy' THEN 1 ELSE -1 END) AS taker_cost_bps
FROM trade t ASOF JOIN bbo b USING (venue, symbol, local_recv_ts);
```

**Chart 15.** Quoted vs effective spread, two lines per venue. The gap is a
proxy for fragmentation / sub-tick activity.

### 4.7 What "good" looks like at end of P2

You should know the spread, depth at three tiers, and churn rate per venue,
plus how each varies through the day. A short note explaining "why is HL spread
~3× wider than Binance perp at all hours? — because HL has lower volume and
fewer pro MMs". Knowing the answer gives you the realistic ceiling on your
quoting tightness.

---

## 5. Phase P3 — Trade flow and lead-lag

### 5.1 Aggressor balance

```sql
SELECT
  venue, symbol,
  date_trunc('minute', to_timestamp(local_recv_ts/1e9)) AS minute,
  sum(CASE WHEN side='buy'  THEN size ELSE 0 END) AS buy_vol,
  sum(CASE WHEN side='sell' THEN size ELSE 0 END) AS sell_vol,
  sum(CASE WHEN side='buy'  THEN size ELSE -size END) AS imbalance
FROM read_parquet('data/**/event=trade/**/*.parquet')
GROUP BY 1,2,3 ORDER BY 1,2,3;
```

**Chart 16.** Cumulative net aggressor volume over the day, per venue. A
strong upward drift = persistent buying pressure, often a leading indicator of
price moves. Compare across venues — does the aggressor imbalance show up
earlier on Binance than HL?

### 5.2 Trade size distribution

**Chart 17.** Complementary CDF of trade sizes (log-log). Heavy tails are
universal in crypto; the *shape* of the tail tells you about the participant
mix. A power-law with α ≈ 1.3 is typical retail-dominated; α closer to 1.0
means whales/HFT dominate.

Bucket trades by size: micro (<$100), small ($100–$1k), medium ($1k–$10k),
large ($10k–$100k), whale (>$100k). Track the share of each bucket over time.
A regime where large+whale share rises is when MM has to be most defensive.

### 5.3 Inter-arrival times and burstiness

Trade arrivals are typically clustered (Hawkes-like). A simple diagnostic:
plot the histogram of `delta_t = trade_t - trade_{t-1}` on log scale. If it's
exponential (straight line on semi-log), the process is Poisson. Anything more
peaked at the origin = self-exciting. Almost every market is self-exciting.

### 5.4 Big-trade event study

For trades in the top 1% by size, plot the average mid-price path in the
30 seconds before and 30 seconds after the trade. This reveals:

- Information leakage: does mid drift in the direction of the impending big
  trade *before* the trade hits? If yes, the venue has front-running by HFT
  watching order arrivals.
- Permanent vs temporary impact: does mid recover post-trade or stick? Sticky
  = informed flow.

**Chart 18.** Average normalised price path (`(p_t - p_event) / p_event`)
indexed at the big-trade time, with confidence bands.

### 5.5 Cross-venue lead-lag

The headline question for cross-venue MM: when prices move, does Binance lead
HL or vice versa? Quantify with cross-correlation of returns at multiple lags.

```python
import polars as pl, numpy as np
hl  = pl.scan_parquet("data/venue=hyperliquid/product_type=perp/**/event=bbo/**/*.parquet")
bnc = pl.scan_parquet("data/venue=binance/product_type=perp/**/event=bbo/**/*.parquet")

# 100ms-resampled mid returns
def to_returns(df, name):
    return (
        df.with_columns(((pl.col("bid_px") + pl.col("ask_px")) / 2).alias("mid"))
          .with_columns(pl.col("local_recv_ts").cast(pl.Datetime("ns")).alias("t"))
          .sort("t")
          .group_by_dynamic("t", every="100ms").agg(pl.col("mid").last())
          .with_columns((pl.col("mid").pct_change()).alias(name))
          .select(["t", name])
    )

hl_r  = to_returns(hl, "hl")
bnc_r = to_returns(bnc, "bnc")

merged = hl_r.join(bnc_r, on="t", how="inner").drop_nulls().collect()
# Now compute CCF over ±2 sec range (±20 lags at 100ms)
import numpy as np
lags = np.arange(-20, 21)
ccf = [
    np.corrcoef(merged["bnc"][20:-20], merged["hl"][20+L:-20+L if -20+L != 0 else None])[0,1]
    for L in lags
]
```

**Chart 19.** CCF as bar chart, x = lag in 100ms increments. Peak at positive
lag means Binance leads HL; peak at negative = HL leads. For BTC perps the
historical answer is "Binance leads by 100–300 ms" — verify on this dataset.

**Chart 20.** Rolling lead-time at peak correlation, daily window. Lead time
varies by regime; widening lead = HL is becoming less efficient (often during
low-volume hours).

### 5.6 Granger causality

Run a Granger causality test of `bnc_r → hl_r` and the reverse. Expected: a
strong significant Granger effect from Binance to HL, weaker the other way.
Use this as a structural confirmation of the CCF visual.

### 5.7 What "good" looks like at end of P3

A single number: the lead-time of Binance over HL in milliseconds, plus its
intraday distribution. This is the latency budget within which an MM strategy
that uses Binance signals to quote on HL must operate.

---

## 6. Phase P4 — Adverse selection and markout

This is what separates an MM that survives from one that doesn't. For every
*public trade*, ask: "if I had been on the other side, how much would I have
lost over the next N seconds?"

### 6.1 Markout calculation

```sql
WITH trade AS (
  SELECT venue, symbol, local_recv_ts, price, size, side, trade_id
  FROM read_parquet('data/**/event=trade/**/*.parquet')
),
bbo AS (
  SELECT venue, symbol, local_recv_ts, (bid_px + ask_px)/2 AS mid
  FROM read_parquet('data/**/event=bbo/**/*.parquet')
),
joined AS (
  SELECT
    t.venue, t.symbol, t.trade_id, t.local_recv_ts AS t_ts,
    t.price, t.size, t.side,
    b1.mid AS mid_at_trade,
    b2.mid AS mid_1s,
    b3.mid AS mid_5s,
    b4.mid AS mid_30s
  FROM trade t
  ASOF JOIN bbo b1 USING (venue, symbol, local_recv_ts)
  -- repeat with offset joins for 1s, 5s, 30s after trade
  ...
)
SELECT *,
  -- Markout for the LIQUIDITY PROVIDER opposite of the taker:
  -- if taker bought, MM sold; MM's PnL is (price - future_mid) per unit.
  (price - mid_1s)  * (CASE side WHEN 'buy' THEN 1 ELSE -1 END) AS markout_1s_per_unit,
  (price - mid_5s)  * (CASE side WHEN 'buy' THEN 1 ELSE -1 END) AS markout_5s_per_unit,
  (price - mid_30s) * (CASE side WHEN 'buy' THEN 1 ELSE -1 END) AS markout_30s_per_unit
FROM joined;
```

A *positive* markout-per-unit is good for the MM (price went the way they
wanted after the fill). A *negative* markout means the MM got picked off. The
mean markout (signed) over many trades = expected PnL per fill from spread
capture, before fees.

### 6.2 Markout by trade size

Aggregate `markout_5s_per_unit` by trade size bucket:

**Chart 21.** Box plot, x = size bucket (micro/small/medium/large/whale),
y = markout_5s_per_unit. The expected pattern: micro trades have positive
markout (retail noise, MM wins), large trades have negative markout (informed
flow, MM loses). The size at which markout flips sign is the *toxic threshold*
— larger orders are statistically informed.

### 6.3 Markout by hour-of-day

**Chart 22.** Heatmap, x = hour-of-day UTC, y = size bucket, colour = mean
markout. Identifies which hour-size cells are toxic. Sometimes US session +
large size = toxic, while Asia session + large size is fine.

### 6.4 Cross-venue: Binance trade as a feature for HL adverse selection

If we condition HL trades on whether a Binance trade happened in the preceding
500ms, do markouts differ? This is a direct measure of "Binance-leakage" toxic
flow on HL.

**Chart 23.** Same as Chart 21 but split into two cohorts: HL trades preceded
by a same-side Binance trade in the last 500ms vs not. The cohort with
preceding Binance trades should show much more negative markout — quantifies
how much edge a cross-venue MM gets by simply *not quoting* when Binance has
just moved.

### 6.5 Top-trader attribution

The HL Info API exposes `trader_stats`. Pull the top 50 traders by 30-day PnL.
If their wallet addresses appear in the `users` field of trades (HL exposes
this in WS trades), tag those trades. Compute their markouts separately.

If the top-50 cohort has dramatically more negative markouts, those are the
counterparties to avoid quoting against — hence "do not quote when wallet X
has been actively trading in the last N seconds".

### 6.6 What "good" looks like at end of P4

A markout curve of `mean signed markout per unit` over horizons (100ms, 1s, 5s,
30s, 5min) per venue. The 5s value is the most-cited single number — it's the
realistic horizon at which an MM has either repriced or been adversely
selected. A markout of −0.3 bps at 5s on a venue with a 5 bps spread means
roughly 6% of spread is paid back to informed flow; the rest is your gross
margin.

---

## 7. Phase P5 — HIP-4 binaries (digital options)

Binary outcome markets are *digital options*. The mathematical framework is
different from linear MM, but our recorder gives us all the inputs.

### 7.1 The fair value model

For a binary "BTC > K at time T" paying 1 USDC if true:

```
Fair value = N(d2) where d2 = (ln(S/K) + (r - q - σ²/2)(T-t)) / (σ √(T-t))
```

For our use case (1-day expiry, BTC, USDC-collateralised, no dividends or
material rates):

```
d2 ≈ (ln(S/K) - σ² (T-t) / 2) / (σ √(T-t))
fair = Φ(d2)        # standard normal CDF
```

with `σ` = annualised vol estimated from BTC perp returns.

### 7.2 Implied probability vs theoretical

For each tick of (#30 mid, BTC perp mid), compute:

- `implied = mid(#30)` (observed market probability)
- `theoretical = Φ(d2)` (with `σ` from a rolling realised vol estimator)
- `vol_premium = implied − theoretical`

If `vol_premium` is consistently positive, the market is paying more for YES
than the realised-vol-implied fair value would suggest — sentiment skew. If
consistently negative, traders are *under*-pricing tail outcomes.

```python
import numpy as np
from scipy.stats import norm

def hip4_fair(S, K, T_remaining_yr, sigma_annual):
    if T_remaining_yr <= 0 or sigma_annual <= 0:
        return 1.0 if S > K else 0.0
    d2 = (np.log(S / K) - 0.5 * sigma_annual**2 * T_remaining_yr) / (sigma_annual * np.sqrt(T_remaining_yr))
    return norm.cdf(d2)
```

The strike `K` and expiry come from the recorded `MarketMetaEvent` for the
relevant `#XX` coin.

**Chart 24.** Three lines on shared x-axis (last 24h before settlement):
- BTC perp mid
- Implied probability (from #30 mid), scaled to right axis
- Theoretical Φ(d2) (computed from BTC mid + rolling 1h vol + remaining time),
  scaled to right axis

**Chart 25.** `implied − theoretical` over the 24h life of the market. Trends
toward zero near expiry (the digital becomes deterministic). Persistent gap =
arbitrage signal.

### 7.3 Greeks

For a digital call:

```
Δ = ∂(price)/∂S = ϕ(d2) / (S σ √(T-t))     # density divided by spot * vol * sqrt-time
Γ = ∂²(price)/∂S² ≈ −d1 ϕ(d2) / (S² σ² (T-t))
Vega = ϕ(d2) √(T-t) / σ          # units: per unit of vol (small for digitals)
Theta = derivative wrt time      # usually large near strike near expiry
```

Two practical observations on these greeks specifically for HIP-4:
- **Delta of a digital approaches a Dirac delta as T→0 near the strike.** This
  is what makes the last 30 minutes before settlement a gamma-hedger's
  nightmare and an MM's most dangerous window. Quantify by plotting Δ as a
  function of (BTC distance from strike, time-to-expiry).
- **Cross-book delta** of (long #30 + long #31) is *zero* — they replicate a
  full unit of payoff, regardless of where BTC settles. So you can hedge
  inventory exposure by acquiring the *other* side, not just by trading BTC.

**Chart 26.** Delta surface: heatmap, x = (BTC mid − strike) in % terms, y =
hours-to-expiry, colour = computed Δ. Reveals the regions where delta is
manageable vs catastrophic.

**Chart 27.** Theta time series: for each minute, compute `theoretical_now −
theoretical_one_minute_ago` holding S fixed. The expected time decay per
minute. Useful for setting time-based inventory bands.

### 7.4 Cross-book arbitrage spread

`bid(#30) + bid(#31)` should sum to slightly less than 1.0 (selling both =
locked-in $1 at expiry minus the spread you crossed). `ask(#30) + ask(#31)`
should sum to slightly more than 1.0. The deviation from 1.0 is the *internal*
spread of the binary — how much the two books disagree.

```sql
WITH
  yes AS (SELECT local_recv_ts, bid_px, ask_px FROM read_parquet('data/**/event=bbo/symbol=#30/**/*.parquet')),
  no  AS (SELECT local_recv_ts, bid_px, ask_px FROM read_parquet('data/**/event=bbo/symbol=#31/**/*.parquet'))
SELECT
  date_trunc('minute', to_timestamp(yes.local_recv_ts/1e9)) AS minute,
  avg(yes.bid_px + no.bid_px) AS bid_sum,
  avg(yes.ask_px + no.ask_px) AS ask_sum,
  avg(yes.ask_px + no.ask_px - yes.bid_px - no.bid_px) AS combined_spread
FROM yes ASOF JOIN no USING (local_recv_ts)
GROUP BY 1 ORDER BY 1;
```

**Chart 28.** `bid_sum`, `ask_sum`, and `1.0` reference line — three lines.
Tightness = MM efficiency.

### 7.5 Volume profile by time-of-day

HIP-4 markets are 1-day cycles ending at 06:00 UTC. Volume is not uniform.

**Chart 29.** Trade volume in 10-minute bins, aggregated across multiple
daily cycles. Expect a bump at the open (price discovery in the first 15
minutes), a steady mid-day, and a final spike in the last 30 minutes as
positions are squared.

### 7.6 Settlement event study

For each daily roll, capture the last 60 minutes of (#30 mid, #31 mid, BTC
mid, settlement price). Compute:

- Implied probability path in the last 60m, 10m, 1m before settle.
- Spread blow-out: spread of #30 in last 5 minutes vs prior hour.
- Final settlement vs final mid: was the market well-calibrated?

**Chart 30.** Multi-panel zoom of the last 60 minutes pre-settlement, one row
per daily cycle:
- BTC mid (with strike line)
- #30 implied prob
- #30 spread in bps

You will probably see #30 mid converge to {0, 1} sharply in the final minutes;
the speed of convergence is a microstructure parameter for designing the
settlement-window quoting strategy (e.g. "stop quoting in last 5 minutes if
|S−K|/K < 0.1%").

### 7.7 What "good" looks like at end of P5

A model that, given (BTC mid, time-to-expiry, current realised vol), predicts
the HIP-4 mid within ±2% across most of the day, and a documented
characterization of the settlement-window microstructure (the part where the
model breaks down).

---

## 8. Phase P6 — Volatility and funding regimes

### 8.1 Realised volatility

Several estimators worth computing in parallel:

- Close-to-close (per-minute) — the simplest.
- Garman-Klass — uses high/low/open/close, more efficient.
- Parkinson — high/low only, very efficient.
- Realised kernel — for high-frequency, accounts for microstructure noise.

For 5-second bars on BTC over 1 hour, you have ~720 observations; that's
enough for a 5-min realised vol with reasonable noise.

**Chart 31.** Three estimators overlaid; expect rough agreement except in
microstructure-heavy regimes.

### 8.2 Vol cones

Using historical realised vol, plot percentile bands (5/25/50/75/95) for vol
across horizons (5m, 15m, 1h, 4h, 1d). Today's realised at each horizon
plotted as a line: where in the historical distribution is current vol?

**Chart 32.** Vol cone with current vol overlay. A line that hugs the 95th
percentile = realising in the upper tail = expect mean reversion.

### 8.3 Funding–vol relationship

Aggregate funding rate to a daily series; aggregate realised vol to a daily
series. Scatter and fit — typically funding spikes during sustained
trends/vol.

**Chart 33.** Scatter, x = daily realised vol, y = absolute mean daily
funding rate, one point per day. A positive correlation is the norm and
quantifies the "vol-funding tax" on directional positions.

### 8.4 Funding mean reversion

Half-life of funding rate residuals around its long-run mean. Useful for
deciding how to scale carry-trade position size.

### 8.5 What "good" looks like at end of P6

A daily pipeline that emits the current vol percentile and current funding
percentile per asset. A simple "regime label" — e.g. (high vol / high
funding / extended) — is worth a single data point per day for later
strategy gating.

---

## 9. Phase P7 — MM strategy calibration

The plan output: a sized strategy specification ready for paper trading.

### 9.1 Spread vs fill-rate vs adverse-selection curve

For a candidate MM that quotes `bid(t) = mid − s/2`, `ask(t) = mid + s/2` at
spread `s`:

1. **Fill rate**: estimate from the empirical depth-of-book profile and the
   realised flow direction. P(fill within Δt at distance s/2 from mid) ≈
   Σ over trades_in_Δt I[trade crosses s/2 against quote].
2. **Edge per fill**: `s/2 − markout_at_horizon`. The horizon should match
   how long you'd hold the position before re-flattening.
3. **Adverse selection**: from Phase P4, you have markout curves. Smaller `s`
   means more fills but worse markout (more aggressive flow crosses you).

The optimisation: maximise `fill_rate(s) × (s/2 − markout(s))` − inventory_cost.
Plot as a function of `s` (varying from 0.5 bp to 50 bp).

**Chart 34.** Three curves on a shared x-axis (`s` in bps):
- Expected fill rate (per minute).
- Expected edge per fill in bps.
- Their product (expected $/min PnL).

The peak of curve 3 is your starting candidate spread.

### 9.2 Inventory dynamics

Simulate: assuming the chosen `s` and the empirical fill arrival process from
Phase P3, simulate a quoting strategy over 1 trading day. Track inventory
(positive = long, negative = short) over time.

**Chart 35.** 100 simulated inventory paths over a trading day, with a band
showing |inventory| stays within Q% of the mean for X% of paths. Use this to
set inventory limits.

### 9.3 Hedge cost model

When inventory exceeds a threshold, the MM must hedge. Two routes:

a. **Cross-book hedge for HIP-4**: long #30 → buy #31 to neutralise. Cost =
   spread of #31 + slippage. Pros: same venue, no funding exposure. Cons:
   doubles capital tied up.
b. **Cross-asset hedge**: long #30 → short BTC perp. Cost = perp spread +
   funding cost over hold + delta-recompute frequency. Pros: capital-efficient.
   Cons: imperfect hedge (delta drifts), funding cost.

Build a simple cost model:

```
hedge_cost_perp = perp_spread + |funding| × hold_time + |Δ_drift| × |inventory|
hedge_cost_book = #31_spread × inventory
```

**Chart 36.** Hedge cost as a function of (inventory size, time to hold). The
crossover point between the two routes tells you the rule for which to use.

### 9.4 Edge waterfall

For the final strategy, attribute expected edge:

```
gross_spread_capture
  − markout_cost_5s
  − inventory_carry_cost
  − hedge_execution_cost
  − exchange_fees_taker_when_hedging
  = net_edge_per_unit_of_volume
```

**Chart 37.** Waterfall chart of gross edge → net edge, with each subtraction
labelled. Is a sanity check that the strategy is actually positive after
realistic costs.

### 9.5 What "good" looks like at end of P7

A quoting spec — `s_bps`, inventory thresholds, hedge route per inventory tier
— that, run through the simulator on the last 7 days of data, produces a
positive net edge curve under realistic markout assumptions. Negative? Iterate
spread/hedge logic. The sim is wrong before the strategy is.

---

## 10. Cadence and reporting

| Cadence | Output |
|---|---|
| Continuous | Recorder running; `scripts/status.sh` checked once a day |
| Daily | Auto-emit a 10-line health note: row counts, latency p99, reconnect count, cross-venue oracle agreement |
| Weekly | Refresh charts 5, 15, 19, 21, 26 — these are the "core" microstructure dashboards |
| Bi-weekly | Re-run P5 (HIP-4 fair-value model) with newer vol estimate |
| Monthly | Re-run P7 (strategy calibration) and update the quoting spec |

A monthly report saved as `docs/reports/YYYY-MM-summary.md` with all 37 charts
embedded keeps a clean trail of the strategy's evolution. The first month's
report will be sparser; by month 3 the strategy spec stabilises.

---

## 11. Deliberately not in scope

Things that experienced analysts would also do but are deferred until post-MVP:

- **Tick-by-tick replay simulator** with realistic queue-position modelling.
  We've recorded enough to build it later; defer until paper-trading reveals
  what edge the simulator must capture.
- **Latency arbitrage modelling**. Requires knowing your own send-side latency
  distribution, which depends on infra we haven't decided.
- **Regime-switching models** (HMM on returns / volume). Worth building only
  after you have ≥3 months of data.
- **Wallet-level attribution** beyond the top-trader cohort. Could be powerful
  but the cost of building a wallet-clustering pipeline is high; revisit if
  P4 reveals strong wallet-driven adverse selection.

---

## 12. Reading list

Worth keeping handy while writing the analyses above. Not a comprehensive list —
just the books and papers that consistently come up in this kind of work.

- Cartea, Jaimungal, Penalva — *Algorithmic and High-Frequency Trading* (the
  reference for MM math; chapter 10 on adverse selection is the one).
- Almgren, Chriss — *Optimal Execution* (canonical impact model).
- Stoikov — papers on optimal market making with inventory; the Avellaneda-Stoikov
  framework is the starting point for any spread/inventory optimisation.
- Easley, López de Prado, O'Hara — VPIN and toxicity papers.
- Aldridge, Krawciw — *Real-Time Risk* (cross-venue and slippage).

For HIP-4 specifically, no canonical text exists yet — the digital-option
math from Hull plus a willingness to read Polymarket order-book studies will
get you 80% of the way.
