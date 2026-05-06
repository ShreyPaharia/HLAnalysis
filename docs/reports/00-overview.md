# Analysis overview — HLAnalysis MM research

**Date:** 2026-05-05  
**Window analyzed:** 2026-05-05 07:57 → 10:35 UTC (~2.77 h)  
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

**Rebuild:** `tools/build_notebooks.py` regenerates all five from inline cell
strings — edit there, not the `.ipynb`. Re-execute with:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace docs/reports/0*.ipynb
```

---

## 3. Early findings (2.77 h sample)

These are read straight off the executed notebooks. They are **directionally
informative** but not statistically robust — wait for ≥ 24 h (and at least one
HIP-4 settlement at 06:00 UTC) before any of these become decision inputs.

### 3.1 Coverage looks healthy

All 17 expected `(venue × product × event × symbol)` partitions have data with
matching wall-clock windows. Per-minute message rates over the run are flat
(no reconnect cliffs). See notebook 01 §1–§2.

| stream | rows | rows/min |
|--------|-----:|---------:|
| binance/perp/bbo BTCUSDT | 3,277,667 | ~19,720 |
| binance/spot/bbo BTCUSDT | 1,148,717 | ~6,910 |
| hyperliquid/perp/bbo BTC | 133,395 | ~803 |
| hyperliquid/prediction_binary/bbo #30 | 8,544 | ~51 |
| hyperliquid/prediction_binary/bbo #31 | 7,061 | ~42 |
| hyperliquid/perp/trade BTC | 84,983 | ~511 |
| binance/perp/trade BTCUSDT | 486,997 | ~2,930 |

**Reasoning:** Binance perp BBO churns ~25× faster than HL perp BBO, which
is consistent with HL having coarser tick + fewer market makers. That ratio
matters for the MM model — passive quotes on HL face less aggressive
top-of-book competition than on Binance.

### 3.2 Spreads — HL perp is *tight*, HIP-4 binaries are not

Median quoted spread:

| venue/product | median (bps) | p95 (bps) |
|---------------|------------:|---------:|
| binance/spot BTCUSDT | 0.00 | 0.00 |
| binance/perp BTCUSDT | 0.01 | 0.01 |
| **hyperliquid/perp BTC** | **0.12** | **0.12** |
| hyperliquid/prediction_binary #31 (NO) | 20.21 | 183.55 |
| hyperliquid/prediction_binary #30 (YES) | 41.95 | 238.94 |

**Reasoning:**
- Binance is essentially tick-bound at the BTC price level (1 USD on ~80k mid).
- HL perp at 0.12 bps is ~10× wider than Binance — there is room for an MM to
  step inside on HL provided the hedge cost (Binance round-trip) stays under
  the captured edge.
- HIP-4 spreads are fat (~20–40 bps) and asymmetric between the YES/NO books,
  consistent with a thin retail-driven book; this is where passive MM has the
  largest absolute edge per fill *if* inventory can be neutralized.

### 3.3 HL/Binance perp basis is tight, mean-positive, and small-σ

`HL_perp − BN_perp` over 2.77 h, 1-second resampled mids:

| metric | value |
|--------|------:|
| mean   | +0.66 bps |
| median | +0.58 bps |
| std    |  0.96 bps |
| 5%–95% | -0.74 to +2.49 bps |

**Reasoning:** HL trades a sub-bp persistent premium. That's compatible with
funding-driven carry (HL annualised funding has been mildly positive on BTC).
The σ of ~1 bps is much smaller than even HL's own quoted spread (12 bps), so
short-horizon basis trading is **not** a passive-MM strategy at these widths;
it would be a latency-arb game. Notebook 02 plots the path and computes
half-life.

### 3.4 HIP-4 cross-book sums are well-policed

Across 1,974 5-second snapshots of `(#30 YES, #31 NO)`:

| sum | median | p1 | p99 |
|-----|-------:|---:|---:|
| bid_yes + bid_no | 0.9996 | 0.984 | 1.0000 |
| ask_yes + ask_no | 1.0014 | 1.0000 | 1.020 |

**Reasoning:** No samples violated `bid_sum > 1` (i.e. no free-money arb across
the two CLOBs). The bid sum sits just **below** 1 and ask sum just **above** 1
by a few bps — the gap is the round-trip spread an arber would have to pay.
That confirms the schema decomposition is real and the books are arbed by *someone*.
This also gives us a lower bound on the realized arb cost: ~10–15 bps round-trip.

### 3.5 Realized BTC vol (1s log-returns, annualized) ≈ 14%

From HL perp mid over 2.77 h. Compared with the YES-price-implied σ in
notebook 05, this is the input to the BS-digital fair-value baseline.

**Reasoning:** A 14% BTC realized vol is on the low side of the regime range
(historically 25–80%). Implied σ on the binary should track this — large
divergence is a calibration issue or a real "the market knows something"
moment. The window is too short to over-interpret.

### 3.6 Lead-lag

Notebook 04 computes the cross-correlation of signed flow and of mid returns
between HL perp and Binance perp at ±5 s, 100 ms grid. Read the peak lag
direct off the chart — anything beyond ~300 ms is a strong signal worth
trading; sub-100 ms is noise on this sample size and clock-skew quality.

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

The plan's later phases need a longer window than 2.77 h:

- **Vol cones / funding regimes (P6)** — 7+ days minimum.
- **HIP-4 settlement event study (P5.3)** — at least one settlement boundary
  (next: 2026-05-06 06:00 UTC).
- **Markout / adverse-selection profile (P4)** — needs at least 100k tagged
  trades per venue with mid resampling at multiple horizons.
- **Top-trader attribution (P4)** — requires HL S3 backfill of `node_fills`;
  recorder-only data does not include user IDs.

Once we have 24 h continuous (and the HIP-4 roll has been observed) the
notebooks above should be re-run as the first pass of the daily report; new
notebooks 06/07 (vol cones, MM strategy calibration) get added then.

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
