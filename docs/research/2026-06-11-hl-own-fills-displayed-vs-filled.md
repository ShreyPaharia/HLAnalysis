# Ground truth: our own HL fills, displayed-vs-filled per order (SHR-104)

**Date:** 2026-06-11
**Branch:** `feat/shr-104-own-fills-displayed-vs-filled`
**Scope:** ANALYSIS-ONLY. No engine/strategy/fill-model/config change, no deploy.
Decisive first step of the **SHR-103** trade-tape fill-model epic; sizes the effect
from our own ground truth before anything is built. Extends
[2026-06-11-ioc-refire-floor-hl-fill-model.md](2026-06-11-ioc-refire-floor-hl-fill-model.md)
and [2026-06-10-hl-all-days-live-vs-sim-shr98.md](2026-06-10-hl-all-days-live-vs-sim-shr98.md).

---

## TL;DR

1. **Our wallets are in the recorded trade tape, and the join works.** The recorder
   writes `buyer`/`seller` on every HL print. v31 prints on 7 of its window legs;
   v1 on 5 (and is absent on #1670, matching live). All 186 of our prints are
   **taker** (aggressor) — consistent with the engine's IOC-at-touch — zero maker
   prints. A spot-check confirms the join is sound: a v31 buy of 150 @0.8509 reads a
   pre-trade book of exactly 150 resting at the 0.8509 touch (66 ms earlier) and the
   next snapshot shows that level consumed.
2. **Per-order, the displayed top-of-book is broadly hittable — median
   filled/displayed = 1.00** (n=123 orders), i.e. a typical IOC clears ≈ one
   top-level's worth and the remainder cancels at the touch. The distribution has
   **fat tails both ways** (p10 0.40, p90 3.41): sometimes we take a fraction of the
   touch, sometimes the book refills and we catch several touches' worth. Measured
   against the **full marketable ladder** (`filled / displayed-at-limit`) the median
   is also **1.00** (p10 0.38, p90 2.00) and holds across tight, mid, and wide books.
   **The over-fill in the sim is therefore *not* an aggregate-depth error** — when the
   sim dumps the displayed ladder, the depth really was hittable in aggregate.
3. **The real gap is per-CLIP granularity, and it is concentrated on wide bucket
   books.** A marketable IOC catches the book in **many small prints** that re-fire as
   the book refills; the sim crosses the **whole displayed ladder in one instantaneous
   clip**. On the #1670 doom-loop bucket our live **sell** clips have **median 14 sh**
   (max 397) — exactly the SHR-103 anchor — while the sim does a single 516-share
   dump. On deep, tight binary books the distinction nearly vanishes: our own clips
   there run **median 141 sh, up to 542** — a big single dump *is* realistic when the
   touch is genuinely that thick.
4. **Lever for SHR-105: cap the per-IOC *clip* size, do not haircut per-order depth.**
   The market clip-size distribution (per print, **excluding our own trades** to avoid
   circularity) is **median 35 sh / p90 166 / max 18,093 on binary legs** and
   **median 31 / p90 142 / max 1,352 on bucket legs**. That per-clip cap — not a
   fractional depth multiplier — is what reshapes the sim's single 516-dump into the
   small re-fired clips we and the market actually trade. Handed to SHR-105 as
   `2026-06-11-hl-market-clips-ex-own.csv`.

---

## Method

- **Window / legs.** The 06-06→06-09 live-eval window (corpus ends 06-09). The
  canonical traded legs are taken from the committed `user_fills` CSV
  (`2026-06-10-hl-live-fills-v1-v31-window.csv`) `coin` field — **this is load-bearing**:
  HL records every *binary* trade on **both** the YES and the NO (complement) book
  with complementary prices (p ↔ 1−p), flipped side, identical size, and **distinct
  `trade_id`s**. Scanning both legs would double-count. Keying on the legs the CSV
  says we actually traded (e.g. #1591, not its mirror #1590) selects one book per
  market cleanly. The same CSV supplies our per-slot wallet addresses (`DIAG` rows)
  and the leg→kind map (`KLASS` rows).
- **Our prints.** `side` is the aggressor side. A print is *ours-as-taker* iff
  `(side=buy ∧ buyer=us)` or `(side=sell ∧ seller=us)`. We confirmed **zero** maker
  prints for either wallet — the engine never rests, as expected for IOC.
- **Orders.** Consecutive same-side prints within **0.5 s** are one marketable IOC.
  Intra-order sub-fills share an `exchange_ts` (gap 0); distinct re-fired orders are
  **≥ 0.73 s** apart (the measured live re-fire floor). 0.5 s sits in that empty band
  — every observed non-zero inter-print gap was ≥ 0.73 s, so no two distinct orders
  merge and no single order splits, except the rare case of two cloids landing in the
  same venue block (quantified in the cross-check below).
- **Decision book.** The L2 snapshot with `exchange_ts` **strictly before** the
  order's first print — the resting depth the order saw, not the post-trade book. The
  engine fires an IOC limit **at the touch** (`hl_client.py`), so "displayed size at
  our limit" = displayed size at the best level.
- **Ratios.** `ratio_top = filled / displayed-top-of-book`;
  `ratio_at_limit = filled / Σ(depth at-or-better than our worst fill price)`.
- **Tool:** `tools/hl_own_fills_displayed_vs_filled.py` (pure analysis core unit-tested;
  thin DuckDB I/O over the recorded parquet). Outputs the per-order table and the
  market clip CSV. Re-run:

  ```
  python tools/hl_own_fills_displayed_vs_filled.py \
    --data-root <corpus> \
    --fills-csv docs/research/2026-06-10-hl-live-fills-v1-v31-window.csv \
    --out-orders docs/research/2026-06-11-hl-own-fills-displayed-vs-filled.csv \
    --out-market docs/research/2026-06-11-hl-market-clips-ex-own-summary.csv \
    --out-market-raw /tmp/hl-market-clips-raw.csv   # full per-print dump (large; not committed)
  ```

---

## Displayed-vs-filled per order (n=123)

`ratio_top = filled / displayed top-of-book at decision`.

| group | n | p10 | median | p90 |
| :-- | --: | --: | --: | --: |
| **ALL** | 123 | 0.40 | **1.00** | 3.41 |
| v1 / binary | 2 | 11.41 | 12.75 | 14.09 |
| v1 / bucket | 4 | 0.10 | 0.62 | 1.68 |
| v31 / binary | 70 | 0.31 | **1.00** | 3.47 |
| v31 / bucket | 47 | 0.71 | **1.00** | 2.85 |

By book width (`best_ask − best_bid`):

| width | n | p10 | median | p90 |
| :-- | --: | --: | --: | --: |
| tight (<0.02) | 72 | 0.32 | 1.00 | 3.82 |
| mid (0.02–0.10) | 25 | 0.96 | 1.00 | 2.60 |
| wide (≥0.10) | 21 | 0.31 | 1.00 | 2.00 |

`ratio_at_limit = filled / full marketable ladder` (drops 3 orders whose limit gapped
below the strictly-before snapshot's best ask → 0 marketable depth):

| group | n | p10 | median | p90 |
| :-- | --: | --: | --: | --: |
| ALL | 120 | 0.38 | **1.00** | 2.00 |
| binary | 71 | 0.20 | 1.00 | 1.00 |
| bucket | 49 | 0.69 | 1.00 | 2.00 |

**Shape of the distribution (ratio_top):** 54 orders fill **exactly** the displayed
top (IOC clears the touch, remainder cancels — displayed == hittable), 29 fill less,
40 fill more (walk deeper / book refilled). 68 % fill ≤ the displayed top.

**Why the median is 1.0, not ≪ 1.** The original SHR-103 hypothesis ("displayed depth
≫ hittable depth") is **not** what the per-order data shows: in aggregate the displayed
ladder *was* hittable (median 1.0 at both top-of-book and full-ladder, across every
width band). Two confounders to read the tails correctly:
- `ratio < 1` does **not** always mean "couldn't fill" — it often means **we didn't
  ask for more** (our order was smaller than the resting depth). The extreme
  v1 #2280 buy filled 552 against a 66,577-share touch (ratio 0.008): a small desired
  size, not a liquidity wall.
- `ratio > 1` (40 orders) is the book **refilling within the order window** or the IOC
  **walking past the touch** — e.g. a v31 #2200 sell of 446 against a 25-share touch
  (ratio_top 17.8) but `ratio_at_limit` ≈ 1.0 (the deeper ladder was there).

So per-order depth is roughly faithful. The fidelity gap lives one level down — in how
that depth is consumed.

---

## The real gap: per-clip granularity on wide books

The decisive contrast is the **per-print clip size**, not the per-order total.

**#1670 (the doom-loop bucket), v31:**

| series | n | median | max |
| :-- | --: | --: | --: |
| our live **sell** clips | 39 | **14 sh** | 397 |
| our live **buy** clips | 6 | 236 sh | 413 |
| market prints (ex-own) | 117 | 58 sh | 947 |
| **sim** (per SHR-103) | — | 70 sh | **516 (single dump)** |

Our exit (sell) execution on this persistently-wide book fragments into **~tens of
shares per print, re-fired over seconds** while the price walks *down* the spread
(0.71 → 0.667 → 0.656 → 0.643 → 0.624 → 0.600). The sim instead sells the whole
position — up to 516 sh — in **one instantaneous clip** at one snapshot. The depth it
crosses genuinely exists in aggregate (`ratio_at_limit` ≈ 0.95 on the matching 706-share
clustered exit), but **no single real clip is that large**, and crossing it all at once
books the full wide spread immediately instead of bleeding it gradually as the real
re-fired clips do.

**Our own clip sizes overall** (one marketable IOC catches this much at once):

| group | n | p10 | median | p90 | max |
| :-- | --: | --: | --: | --: | --: |
| ALL | 186 | 13 | 100 | 422 | 542 |
| binary | 104 | 25 | **141** | 501 | 542 |
| bucket | 82 | 12 | **46** | 355 | 515 |

On **tight binary** books a single IOC routinely catches **100–540 sh** — there a large
single dump *is* realistic, which is why the sim tracks live on binary legs. The clip
collapses to **tens of shares only on the wide bucket books** — exactly where the sim
diverges.

---

## Tape vs `user_fills` CSV cross-check

| slot/leg | kind | tape prints | tape orders | csv fills | csv cloids |
| :-- | :-- | --: | --: | --: | --: |
| v1/#1610 | bucket | 1 | 1 | 1 | 1 |
| v1/#2200 | binary | 2 | 1 | 2 | 1 |
| v1/#2230 | bucket | 1 | 1 | 1 | 1 |
| v1/#2250 | binary | 2 | 1 | 2 | 1 |
| v1/#2280 | bucket | 4 | 2 | 2 | 2 |
| v31/#1591 | binary | 39 | 31 | 39 | 31 |
| v31/#1610 | bucket | 11 | 10 | 11 | 10 |
| v31/#1640 | binary | 21 | 12 | 21 | 12 |
| v31/#1670 | bucket | 45 | 32 | 48 | 41 |
| v31/#2200 | binary | 23 | 17 | 23 | 17 |
| v31/#2250 | binary | 17 | 10 | 17 | 10 |
| v31/#2280 | bucket | 20 | 5 | 8 | 5 |

- **The tape and CSV agree on print count for 9 of 12 legs.** Two diverge with the
  **tape richer** (v1/#2280: 4 vs 2; v31/#2280: 20 vs 8) — on-venue prints the
  `user_fills` REST view did not return. One is **tape-poorer** (v31/#1670: 45 vs 48) —
  3 CSV fills with no matching tape print (likely sub-fills the recorder coalesced).
  Net, the tape is the richer source, as SHR-103 expected.
- **Time-clustering slightly under-counts orders vs cloids** where it differs (v31
  #1670: 32 clusters vs 41 cloids) because distinct cloids occasionally settle in the
  **same venue block** (gap 0) and merge. This does not bias the ratio — same-block
  orders share one decision book and are economically one marketable event — but the
  cloid is the authority on "distinct order"; the cluster count is a lower bound.

---

## Handoff to SHR-105 — market clip-size distribution (per print, ex-own)

Excludes every trade either of our wallets touched (on either side) to avoid
circularity. Committed as the compact per-leg/per-kind percentile summary
`2026-06-11-hl-market-clips-ex-own-summary.csv`; the full 112k-row per-print
distribution is reproducible via `--out-market-raw`.

| leg | kind | n | p10 | median | p90 | max |
| :-- | :-- | --: | --: | --: | --: | --: |
| #1591 | binary | 33,499 | 15 | 39 | 200 | 16,013 |
| #1640 | binary | 36,840 | 16 | 29 | 125 | 10,507 |
| #2200 | binary | 19,973 | 18 | 39 | 200 | 13,623 |
| #2250 | binary | 20,396 | 17 | 50 | 169 | 18,093 |
| #1610 | bucket | 1,210 | 13 | 27 | 124 | 1,352 |
| #1670 | bucket | 117 | 13 | 58 | 226 | 947 |
| #2230 | bucket | 124 | 22 | 58 | 250 | 740 |
| #2280 | bucket | 83 | 5 | 25 | 238 | 307 |
| **binary** | | 110,708 | 16 | **35** | 166 | 18,093 |
| **bucket** | | 1,534 | 13 | **31** | 142 | 1,352 |

**Recommended SHR-105 calibration:** model a marketable IOC as filling a **clip drawn
from this per-print distribution** (cap, not a fractional depth haircut), then re-fire
the remainder under the existing 0.73 s floor — rather than crossing the full displayed
ladder in one clip. A simple, conservative first cut is a **per-clip cap at the market
p90 (≈140 sh bucket / ≈170 sh binary)**; a richer model samples the distribution. This
preserves the (faithful) aggregate depth while reshaping the single 516-dump into the
small re-fired clips reality trades — directly targeting the wide-bucket divergence the
06-10/06-11 docs traced to execution granularity.

---

## Caveats

- **`ratio < 1` ≠ unfillable.** Per-order fills also reflect *desired* size; a small
  order against a deep touch reads as a low ratio without any liquidity constraint. The
  clean, desired-size-independent signal is the **per-clip size distribution**, which is
  why SHR-105 should calibrate on that, not on the per-order ratio.
- **IOC limit is inferred.** The submitted limit is unrecoverable from fills; we use the
  worst realized price as a lower bound, so `displayed-at-limit` is the depth the order
  *demonstrably reached*. 3 orders whose limit gapped below the strictly-before snapshot
  (book moved between snapshot and fill) have 0 marketable depth and drop from
  `ratio_at_limit`.
- **Decision-book staleness.** The strictly-before snapshot can trail the fill by tens
  of ms to seconds on quiet books; mid-window refills are the main source of the
  `ratio > 1` tail. This is inherent to event-sampled book data, not a join bug
  (verified on the 66 ms #1591 example).
- **Mirror legs.** Binary YES/NO books mirror each other in the tape; we measure only
  the canonical leg the CSV says we traded. Bucket legs do not mirror in our trade set.
- **Corpus ends 06-09.** 06-10/06-11 are not recorded.

## Artifacts

- Extractor: `tools/hl_own_fills_displayed_vs_filled.py` (unit tests:
  `tests/unit/tools/test_hl_own_fills_displayed_vs_filled.py`).
- Per-order table: `docs/research/2026-06-11-hl-own-fills-displayed-vs-filled.csv`.
- Market clip distribution summary (ex-own, for SHR-105):
  `docs/research/2026-06-11-hl-market-clips-ex-own-summary.csv`.
- Ground-truth live fills (input): `docs/research/2026-06-10-hl-live-fills-v1-v31-window.csv`.
