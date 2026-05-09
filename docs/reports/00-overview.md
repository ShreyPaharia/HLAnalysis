# Analysis overview — HLAnalysis MM research

**Date:** 2026-05-09  
**Window analyzed:** 2026-05-06 19:02 → 2026-05-09 14:32 UTC (~67.5 h)  
**Recorder pid:** running, see `scripts/status.sh`

This doc is the entry point to the research notebooks. It states what data we
have, what each notebook is asking, and the early read-outs.

---

## 1. Data sources

All data is recorded by `hl-recorder` (defined in [`hlanalysis/recorder/main.py`](../../hlanalysis/recorder/main.py))
into hive-partitioned parquet under [`data/`](../../data/):

```
data/venue=*/product_type=*/mechanism=*/event=*/symbol=*/date=*/hour=*/*.parquet
```

Subscriptions live in [`config/symbols.yaml`](../../config/symbols.yaml):

| venue        | product_type        | symbol     | streams                                          |
|--------------|---------------------|------------|--------------------------------------------------|
| hyperliquid  | perp                | BTC        | trade, bbo, book_snapshot, mark, oracle, funding |
| hyperliquid  | prediction_binary   | `*` → #30, #31 | trade, bbo, book_snapshot, mark             |
| binance      | perp                | BTCUSDT    | trade (`@trade`), bbo, book_snapshot, mark, funding (REST poll) |
| binance      | spot                | BTCUSDT    | trade, bbo, book_snapshot                        |

**Schema:** [`hlanalysis/events.py`](../../hlanalysis/events.py) — discriminated union
on `event_type`. Every event carries `exchange_ts` (venue ns) and `local_recv_ts`
(host ns).

**Adapters:**
- [`hlanalysis/adapters/hyperliquid.py`](../../hlanalysis/adapters/hyperliquid.py) — wildcard expansion + auto-roll for HIP-4
- [`hlanalysis/adapters/binance.py`](../../hlanalysis/adapters/binance.py) — uses `@trade` and REST `premiumIndex` to side-step the geo-blocked `aggTrade`/`markPrice` streams

**Plan:** [`docs/analysis-plan.md`](../analysis-plan.md) — 7-phase, 37-chart roadmap. The notebooks below cover the slices that are tractable on ~3 h of data.

---

## 2. Notebooks

Each notebook is self-contained and re-runnable. Helpers in
[`hlanalysis/analysis/helpers.py`](../../hlanalysis/analysis/helpers.py) keep them short.

| # | Notebook | What it answers |
|---|----------|-----------------|
| 01 | [Data quality (P0)](01-data-quality.ipynb) | Coverage, msg rates, gaps, clock skew, BBO sanity |
| 02 | [Cross-venue basis (P1)](02-cross-venue-basis.ipynb) | HL/Binance perp+spot basis, funding paths, carry |
| 03 | [Microstructure (P2)](03-microstructure.ipynb) | Spreads, depth, book imbalance, TOB churn |
| 04 | [Trade flow & lead-lag (P3)](04-trade-flow-leadlag.ipynb) | Aggressor balance, size CCDF, x-venue CCF on flow + returns |
| 05 | [HIP-4 binaries (P5)](05-hip4-binaries.ipynb) | Cross-book sum check, BS-digital fair value, implied vs realized σ |
| 06 | [HIP-4 pricing from BTC perp](06-hip4-pricing-from-perp.ipynb) | Theo Φ(d₂) vs actual YES on `local_recv_ts` grid, residuals, implied σ — fair-value baseline for the binary using HL perp as $S$ |
| 07 | [HIP-4 late-YES strategy](07-binary-late-yes-strategy.ipynb) | Strategy A (mark-anchored late-YES with `d`-stop), backtest scaffold, stop-loss sweep, calibration reliability diagram |

**Rebuild:** `tools/build_notebooks.py` regenerates all seven from inline cell
strings — edit there, not the `.ipynb`. Re-execute with:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace docs/reports/0*.ipynb
```

---

## 3. Early findings (67.5 h sample)

These are read straight off the executed notebooks. The window now spans ~2.8 days
and includes multiple HIP-4 24h settlement cycles (outcomes #40/#41, #50/#51,
#70/#71, #80/#81, #90/#91 all rolled). Numbers are now statistically meaningful
for spreads and basis; vol estimates are directionally reliable.

### 3.1 Coverage looks healthy

All expected `(venue × product × event × symbol)` partitions have data with flat
per-minute message rates. See notebook 01 §1–§2.

| stream | rows | rows/min |
|--------|-----:|---------:|
| binance/perp/bbo BTCUSDT | 40,212,035 | ~9,929 |
| binance/spot/bbo BTCUSDT | 15,918,007 | ~3,930 |
| hyperliquid/perp/bbo BTC | 701,053 | ~173 |
| hyperliquid/prediction_binary/bbo #50 | 46,908 | ~58 |
| hyperliquid/prediction_binary/bbo #51 | 52,340 | ~65 |
| hyperliquid/perp/trade BTC | 1,432,707 | ~354 |
| binance/perp/trade BTCUSDT | 5,009,376 | ~1,237 |

**Note:** Binance perp BBO rate dropped from ~19,720/min in the initial 2.77 h
sample to ~9,929/min over the full 67.5 h window — this reflects the initial
burst settling to steady-state, not a data loss. HL perp BBO dropped from
~803 to ~173/min for the same reason. The Binance/HL churn ratio remains ~57×.
That ratio matters for the MM model: passive quotes on HL face far less
top-of-book competition than on Binance.

### 3.2 Spreads — HL perp is *tight*, HIP-4 binaries vary widely by cycle

Median quoted spread over the full 67.5 h window:

| venue/product | symbol | median (bps) | p95 (bps) |
|---------------|--------|------------:|---------:|
| binance/perp | BTCUSDT | 0.01 | 0.01 |
| **hyperliquid/perp** | **BTC** | **0.12** | **1.26** |
| hyperliquid/prediction_binary | #50 (YES) | 66.9 | 297.9 |
| hyperliquid/prediction_binary | #51 (NO) | 58.5 | 208.9 |
| hyperliquid/prediction_binary | #40 | 103.8 | 1,684.7 |
| hyperliquid/prediction_binary | #41 | 157.4 | 577.6 |

**Reasoning:**
- Binance remains tick-bound. HL perp at 0.12 bps is ~10× wider — same conclusion
  as the initial sample.
- HIP-4 spreads are materially wider than the initial #30/#31 sample (20–40 bps).
  #50/#51 are tighter (~60–67 bps median), while #40/#41 are much fatter (103–157 bps
  median). This likely reflects proximity to expiry at the time those symbols were
  active — fresher-cycle binaries tend to have tighter markets. The p95 spread
  of >1,000 bps on #40 confirms thin-book risk near settlement.

### 3.3 HL/Binance perp basis is tight, mean-positive, and small-σ

`HL_perp − BN_perp` over 67.5 h, 1-second resampled mids (90,406 joint samples):

| metric | value |
|--------|------:|
| mean   | +0.97 bps |
| median | +0.91 bps |
| std    |  1.31 bps |
| 5%–95% | -1.02 to +3.28 bps |

**Note vs initial sample:** Mean basis rose from +0.66 → +0.97 bps (+47%) and std
widened from 0.96 → 1.31 bps (+36%). The HL premium is persistent and slightly
larger over the multi-day window, though still well below HL's own quoted spread.
The σ of ~1.3 bps remains far too small for short-horizon basis trading to
compete with passive MM.

### 3.4 HIP-4 cross-book sums are well-policed

Across 46,883 ASOF-joined snapshots of `(#50 YES, #51 NO)` — the most liquid
active pair:

| sum | median | p1 | p99 |
|-----|-------:|---:|---:|
| bid_yes + bid_no | 0.9971 | 0.9853 | 1.0000 |
| ask_yes + ask_no | 1.0038 | 1.0000 | 1.0264 |

**Reasoning:** No bid_sum ever exceeded 1.0 (no free-money arb). The structure
matches the prior #30/#31 findings exactly: bid sum drifts just below 1 and ask
sum just above, with the gap representing the round-trip arb cost. The ~1.0264
p99 on ask_sum shows occasional widening consistent with stale quotes near
binary-price inflection points, not a structural inefficiency.

### 3.5 Realized BTC vol (log-returns, annualized) ≈ 17%

From HL perp mid log-returns over 67.5 h (614,759 consecutive-tick pairs,
mean inter-tick spacing ~0.145 s), annualized as `σ_tick × √(SEC_PER_YEAR / avg_dt)`:

**σ_realized ≈ 16.8%**

**Note vs initial sample:** The initial 2.77 h estimate was ~14%. The longer
window gives 16.8% — an increase of ~20%. This is consistent with short windows
undersampling infrequent larger moves. 16.8% is still on the low side of the
historical BTC range (25–80%) and suggests a relatively quiet regime. Implied σ
on the binary should track this; any persistent large divergence (>50% of
realized) is a calibration flag or regime-change signal.

### 3.6 Lead-lag

Notebook 04 computes the cross-correlation of signed flow and of mid returns
between HL perp and Binance perp at ±5 s, 100 ms grid. With 67.5 h of data
(~1.4M HL trades, ~5M Binance perp trades) the CCF estimator is considerably
more stable than the initial 2.77 h sample. Read the peak lag direct off the
chart — the HL transport latency floor is ~225 ms (§4.3 below), so any peak
beyond that represents genuine information lead by Binance.

---

## 4. Known data-quality issues to fix

Surfaced while running the notebooks. Each is real, not cosmetic.

1. **Binance spot bookTicker has no exchange timestamp.**
   The spot `@bookTicker` payload omits an event-time field, so the adapter
   currently sets `exchange_ts = local_recv_ts` (see
   [`hlanalysis/adapters/binance.py:171`](../../hlanalysis/adapters/binance.py:171)
   and `:195`). This corrupts any cross-venue latency or lead-lag involving
   spot BBO. **Fix:** parse `E` from the spot `@trade` and `@depth20` streams
   that *do* carry it, and stamp BBOs with the most-recent observed `E` from
   the same connection (or accept that spot BBO has no venue ts and document).

2. **Binance funding rate field stores `estimatedSettlePrice`, not a rate.**
   `FundingEvent.premium` for Binance currently holds a USD price, not a
   premium ratio. Notebook 02 only uses `funding_rate` (which is correct), but
   any consumer touching `premium` will get garbage units. Already noted in
   prior summary; **fix:** derive `premium = (mark − index) / index` and add
   `OracleEvent` for `indexPrice` on Binance.

3. **HL perp clock skew runs ~225 ms median**, vs Binance perp at 85 ms.
   That's transport latency to api.hyperliquid.xyz from this IP, not a clock
   issue. It does mean cross-venue lead-lag results have a structural ~140 ms
   floor in HL's favor (HL events look "later" than Binance's). Worth
   subtracting the per-venue median before reporting lead-lag numbers.

4. **No NTP discipline on the host.**
   `local_recv_ts` p99 spans of ~800 ms suggest occasional kernel jitter or
   network bursts. Run `sntp` once and add a daily drift check.

5. **Binance liquidations not recorded** (`forceOrder` is geo-blocked from
   this IP). Either drop from analysis-plan §P3 big-trade event study or
   proxy via large-trade clusters.

---

## 5. What's gated on more data

With 67.5 h of continuous data and multiple HIP-4 settlement cycles now observed,
most of the blockers from the initial 2.77 h sample have been cleared:

- **HIP-4 settlement event study (P5.3)** — now tractable. Outcomes #40/#41,
  #50/#51, #70/#71, #80/#81, #90/#91 have all settled. Notebook
  `08-hip4-markouts` (next sprint) can run a settlement-window analysis.
- **Markout / adverse-selection profile (P4)** — 1.4M HL perp trades and 5M
  Binance perp trades are now in the window. Markout analysis at 1s/5s/30s/60s
  horizons is now statistically meaningful.
- **Late-YES strategy calibration (notebook 07)** — requires ≥7 cycles for
  stop sweep, ≥20 for reliability diagram. We have ~5 observable cycles from
  the current window; 2 more days of recording unlocks the stop sweep.

Still gated on more data:

- **Vol cones / funding regimes (P6)** — 7+ days minimum. Current 2.8-day
  window is borderline; re-run this section after 1 week of continuous data.
- **Top-trader attribution (P4)** — requires HL S3 backfill of `node_fills`;
  recorder-only data does not include user IDs.

---

## 6. Reproducing this report

```bash
# 1. one-time deps
.venv/bin/python -m pip install -e '.[analysis]'

# 2. recorder must be running (or backfill loaded)
scripts/status.sh

# 3. rebuild + execute notebooks
.venv/bin/python tools/build_notebooks.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace docs/reports/0*.ipynb

# 4. open in JupyterLab
.venv/bin/jupyter lab docs/reports/
```
