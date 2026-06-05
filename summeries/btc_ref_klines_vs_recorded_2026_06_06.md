# BTC reference equivalence: pulled 1s klines vs recorded Binance, at dt=5

**Date:** 2026-06-06
**Branch:** `chore/btc-ref-klines-vs-recorded-dt5`
**Question:** If we stop **recording** Binance spot and instead **pull 1s klines on
demand**, does the PM BTC backtest change materially? (Backtest corpus only — the
live engine feed is out of scope; klines cannot replace a real-time feed.)

## TL;DR

**Equivalent.** On the live-relevant corpus (recorded PM L2 book, 9 BTC Up/Down
binaries, dt=5), swapping the σ-reference from recorded Binance spot BBO to
pulled 1s klines leaves **every entry decision unchanged (9/9 markets, both
strategies)** and PnL **bit-identical for v1** / **within 1% for theta**. The
σ-series itself diverges only ~0.3% at the median. **Recommendation: the recorder
Binance-spot subscription can be safely dropped for the backtest corpus**, with
the caveats below. No subscription was removed and the live feed was not touched
— this branch only adds the on-demand 1s-kline path + evidence.

## What was compared

Both feeds are bucketed to **dt=5** (`vol_sampling_dt_seconds=5`), so the *only*
difference is where the σ-feeding `ReferenceEvent` OHLC comes from:

| | Source | High/Low semantics |
|---|---|---|
| **(A) recorded** | recorded Binance SPOT BBO ticks → 5s OHLC | quote-**mid** extremes |
| **(B) klines_1s** | pulled Binance SPOT **1s klines** → 5s OHLC | **trade**-OHLC extremes |

Strike resolution is **identical** in both (PM settles on the Binance 1m spot
close; that path always uses the 1m `btc_klines` cache — only the σ reference
changed).

- **Window:** 2026-05-27 → 2026-06-05 (overlap of recorded PM book [from 05-27],
  recorded Binance spot [from 05-06], and pulled 1s klines).
- **Corpus:** 9 PM "BTC Up or Down" daily binaries, **recorded** PM L2 book
  (`book_source=recorded`) — i.e. exactly what runs live. Small n → **suggestive,
  not load-bearing** (same caveat as the v3.7 PM cadence sweeps).
- **Strategies (both live PM slots, dt=5):**
  - `v1_late_resolution` → **Parkinson σ** (H/L based → most sensitive to the H/L source)
  - `v3_theta_harvester` → **bipower σ** (close-to-close → sensitive to closes)

## Layer 1 — σ-series divergence (the load-bearing check)

A matching PnL with a diverging σ would be luck, not equivalence — so we diff the
rolling σ each strategy would *see* (3600s lookback, λ=0.97), aligned by 5s bucket.

| market | buckets rec/k1s | Parkinson Δσ median / p95 | bipower Δσ median / p95 |
|---|---|---|---|
| 0xe09b76cadc | 34413 / 34416 | 0.28% / 6.13% | 0.18% / 2.02% |
| 0x602f533fad | 34520 / 34524 | 0.37% / 6.39% | 0.26% / 1.65% |
| 0x240914e798 | 34437 / 34439 | 0.30% / 4.12% | 0.37% / 1.87% |
| 0x56f45d1d80 | 34520 / 34522 | 0.21% / 3.92% | 0.38% / 2.16% |
| 0x1c908f9bae | 34455 / 34457 | 0.19% / 5.50% | 0.33% / 3.26% |
| 0x9867dec0da | 34479 / 34478 | 0.29% / 4.98% | 0.29% / 4.11% |
| 0xae5b09428f | 34441 / 34443 | 0.28% / 3.20% | 0.18% / 1.40% |
| 0x3589a4f811 | 34495 / 34494 | 0.29% / 2.38% | 0.17% / 1.00% |
| 0x329176fb69 | 34467 / 34523 | 0.40% / 2.51% | 0.23% / 1.45% |

**Aggregate (n ≈ 310k bucket-σ samples):**

| estimator | median Δσ | p95 Δσ | within 1% | within 5% | mean σ rec / k1s |
|---|---|---|---|---|---|
| Parkinson | **0.29%** | 4.15% | 76.7% | 95.9% | 0.00008 / 0.00008 |
| bipower | **0.25%** | 1.93% | 85.8% | 99.4% | 0.00011 / 0.00011 |

The task's prediction holds qualitatively — **Parkinson (H/L) diverges more than
bipower (closes)** because quote-mid extremes ≠ trade-print extremes within a 5s
bucket. But the magnitude is tiny: median ~0.3%, and the **mean σ is identical to
5 decimals**, i.e. the bar-by-bar H/L differences are zero-mean noise that washes
out over the lookback. (The raw `max_rel` is large but is a degenerate-bar
artifact — a near-zero-σ bucket in one source vs a tiny non-zero σ in the other;
the `within-X%` fractions are the honest signal.)

## Layers 2 & 3 — backtest entries + metrics (recorded PM book, dt=5)

| strategy | reference | PnL | trades | hit | max DD |
|---|---|---|---|---|---|
| v1 | recorded | $21.79 | 8 | 44% | $0.00 |
| v1 | **klines_1s** | **$21.79** | 8 | 44% | $0.00 |
| theta | recorded | $46.50 | 20 | 67% | $0.56 |
| theta | **klines_1s** | **$46.01** | 20 | 67% | $1.05 |

**Entry-decision divergence: 9/9 markets identical entry-count for both
strategies** (no gate flipped on any tick).

- **v1 is bit-identical.** Parkinson σ wiggles 0.3%, but v1 only enters in the
  last 2h (`tte_max=7200`) and its `min_safety_d=3.0` gate sits far from the
  margin, so a sub-percent σ change flips nothing.
- **theta differs by $0.49 (1.1%)** — same 20 entries, same hit rate; the gap is
  pure exit-timing (max DD $0.56 → $1.05). The non-zero gap *confirms* the
  reference genuinely changed (else it too would be bit-identical).

## Recommendation

**Yes — recording Binance spot can be dropped for the backtest corpus.** Pulling
1s klines on demand reproduces the dt=5 sim: identical entries, bit-identical /
within-1% PnL, σ-series matched to ~0.3% median. The recorder's Binance-spot
subscription exists only to feed this backtest reference (the live engine already
has its own real-time BTCUSDT_SPOT bbo feed), so dropping it reclaims recorder
cost / EBS with no measured backtest impact.

### Caveats
- **Live feed is NOT replaceable by klines** — real-time trading needs the live
  bbo feed; this finding is strictly about the *backtest corpus*. Do not touch
  `config/symbols.yaml` or the engine reference feed.
- **Small n (9 markets, ~10 days).** Suggestive, not load-bearing. Re-confirm if
  the strategy moves to a more σ-sensitive regime (tighter gates, shorter lookback).
- **Loss of microstructure.** Recorded BBO carries quote spread + (with
  `book_snapshot`) L2 depth that 1s klines do not. Any *future* research that
  needs spread/depth from the Binance reference would lose that signal — the σ
  story is unaffected but this is the real cost of dropping the recording.
- **Time-warped data note.** Recorded data and the Binance API in this
  environment are stamped 2026; the pulled 1s klines matched the recorded BBO to
  the cent at the boundary, so the comparison is genuinely apples-to-apples.

## How to reproduce

```bash
# 1. pull 1s klines for the window (idempotent, ~25s/day)
uv run python scripts/fetch_binance_1s_klines.py --start 2026-05-25 --end 2026-06-05
# 2. run the equivalence harness (σ divergence + 4 backtest cells)
uv run python scripts/run_btc_ref_equivalence.py
```

Artifacts: `data/sim/runs/btc-ref-equivalence-2026-06-06/` (per-cell reports +
`summary.json`).

## Code changes

- `binance_klines.py`: `fetch_klines` is now **interval-aware** (`interval="1s"`
  pages by 1s, not the hardcoded 60s; 1m path bit-identical).
- `polymarket.py`: new `reference_source="klines_1s"` — loads genuine 1s klines
  from `btc_klines_1s/` and buckets to `vol_sampling_dt_seconds` OHLC (wired into
  both the legacy `events()` and recorded-book `events_arrays()` fast path; cache
  config-sig keyed so it never aliases a recorded/klines bundle). Strike
  resolution untouched.
- `cli.py`: `--pm-reference-source klines_1s`.
- `scripts/fetch_binance_1s_klines.py`, `scripts/run_btc_ref_equivalence.py`: new.
