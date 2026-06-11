# HL per-IOC clip-cap calibration (SHR-105)

**Date:** 2026-06-11
**Branch:** `feat/shr-105-clip-cap-calibration`
**Scope:** ANALYSIS / CALIBRATION-ONLY. No engine/strategy/live change, no
deploy. Calibrates the per-IOC clip-size cap for the HL fill model from the
recorded market clip distribution (SHR-104 output). Feeds the parameter spec
consumed by SHR-107.

---

## TL;DR

1. **The lever is a per-clip cap, not a fractional depth haircut.** SHR-104
   showed that per-order displayed depth is broadly hittable (median
   filled/displayed = 1.00). The residual sim↔live gap is that the sim fills
   the *whole displayed book in a single clip* while the market (and we,
   live) fill in **many small clips** of tens-of-shares each, re-fired over
   seconds. Aggregate volume is not the constraint; per-clip granularity is.

2. **Calibrated caps from the market ex-own distribution (112,242 clips):**

   | kind | books | p50 | p90 | p99 | max | proposed cap |
   | :-- | :-- | --: | --: | --: | --: | --: |
   | binary | tight (0.001–0.009) | 35 sh | **166 sh** | 922 sh | 18,093 sh | **166 sh** |
   | bucket | wide (0.13–0.40) | 31 sh | **142 sh** | 765 sh | 1,352 sh | **142 sh** |

   Both caps are at **market p90**, which (a) covers 90% of real clips by
   count, (b) is below the sim's 516-sh doom-loop single-dump (i.e. will
   actually reshape the overshoot), and (c) is above the per-clip median for
   both our own binary clips (141 sh) and bucket clips (46 sh) from SHR-104.

3. **Regime-keyed is better than uniform** because binary and bucket p90s are
   close (166 vs 142), but the per-leg variation is large: #1670 bucket p90 is
   226 sh while #2280 bucket p90 is only 238 sh (dominated by rare wide
   prints). A uniform cap of 166 sh (binary default) is a safe single-scalar
   first cut.

4. **Parameter spec (for SHR-107):** Two new `RunConfig` scalars —
   `ioc_clip_cap_binary_shares` (default **166**, A/B off = `None`) and
   `ioc_clip_cap_bucket_shares` (default **142**, A/B off = `None`) — with
   corresponding `--ioc-clip-cap-binary-shares` / `--ioc-clip-cap-bucket-shares`
   CLI flags. A single `ioc_clip_cap_shares` scalar is acceptable for a
   minimal first cut. See [Parameter spec](#parameter-spec) below.

---

## Context

### The problem

Across the 16-cell live-vs-sim matrix (SHR-106), the shipped IOC model (re-fire
floor 0.75 s) leaves **84% of filled sim size phantom** — displayed depth that
no real taker ever swept. The re-fire floor reduces this by only −1.2%
(SHR-106, SHR-89). The binding constraint is **not** cadence but per-clip
granularity: on the #1670 doom-loop bucket the sim dumps 516 sh in one clip
while all 132 real market prints (and our own live exit clips) average in the
tens-of-shares range.

### Why SHR-104 changed the framing

SHR-104 measured per-order displayed-vs-filled from our own live fills. Key
finding: **median filled/displayed = 1.00 per order** (i.e. the displayed
ladder was genuinely hittable in aggregate). The original SHR-103 hypothesis
("displayed depth ≫ hittable depth") was wrong at the per-order level. The gap
is one level lower: the depth is hittable in aggregate, but it is consumed in
**many small clips** across multiple re-fires, not in a single instantaneous
dump. Per-clip sizes from our own fills (SHR-104):

| group | n | p10 | median | p90 | max |
| :-- | --: | --: | --: | --: | --: |
| ALL own clips | 186 | 13 | 100 | 422 | 542 |
| binary own clips | 104 | 25 | **141** | 501 | 542 |
| bucket own clips | 82 | 12 | **46** | 355 | 515 |

The decisive contrast: on #1670 (the doom-loop bucket) our live **sell** clips
have median **14 sh** while the sim does a single **516-sh** dump. On tight
binary books the distinction nearly vanishes (our clips median 141 sh, sim
fills to similar sizes), consistent with the sim being accurate on binary legs.

---

## Per-regime clip-size distribution (market ex-own)

All 112,242 clips exclude every trade either of our wallets participated in on
either side (to avoid circularity; produced by SHR-104's
`--out-market-raw` flag from the 06-06→06-09 corpus). Source:
`docs/research/2026-06-11-hl-market-clips-ex-own-summary.csv`.

### Per-leg detail (kind × width regime)

All binary legs in the window are **tight** (typical spread 0.001–0.009); all
bucket legs are **wide** (spread 0.13–0.40). No mid-width legs were traded.

| leg | kind | width regime | n | p10 | p50 | p90 | p99 | max |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: |
| #1591 | binary | tight | 33,499 | 15 | 39 | 200 | 1,200 | 16,013 |
| #1640 | binary | tight | 36,840 | 16 | 29 | 125 | 743 | 10,507 |
| #2200 | binary | tight | 19,973 | 18 | 39 | 200 | 1,096 | 13,623 |
| #2250 | binary | tight | 20,396 | 17 | 50 | 169 | 931 | 18,093 |
| #1610 | bucket | wide | 1,210 | 13 | 27 | 124 | 881 | 1,352 |
| #1670 | bucket | wide | 117 | 13 | 58 | 226 | 765 | 947 |
| #2230 | bucket | wide | 124 | 22 | 58 | 250 | 682 | 740 |
| #2280 | bucket | wide | 83 | 5 | 25 | 238 | 307 | 307 |

### Kind-aggregate (used for cap derivation)

| kind | n | p10 | p25 | p50 | p75 | p90 | p99 | max | mean |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| **binary** | 110,708 | 16 | 21 | 35 | 100 | **166** | 922 | 18,093 | 94 |
| **bucket** | 1,534 | 13 | 17 | 31 | 68 | **142** | 765 | 1,352 | 70 |
| ALL | 112,242 | 16 | 21 | 35 | 100 | 166 | 919 | 18,093 | 94 |

**Binary p90 = 166 sh; bucket p90 = 142 sh.** Both kinds have near-identical
medians (35 vs 31 sh); the separation is in the p75+ tails. Note that p99 and
max are driven by rare mega-clips that are genuine large institutional sweeps
(binary #1250 has a 18,093-sh single print); those outliers do not represent
typical hittable-per-IOC depth for our position sizes ($300–$1,000 notional).

---

## Cap derivation

### Choosing p90 as the cap level

A cap at the **market p90** is a reasonable conservative first cut because:

- It covers **90% of real on-venue clips by count**, so the simulated fills are
  limited to sizes that actually happen in the market 90% of the time.
- It is **well below the sim's single-dump outlier**: the #1670 doom-loop single
  dump is 516 sh; the binary p90 (166 sh) and bucket p90 (142 sh) are both
  below this, so the cap *will* reshape the dominant overshoot.
- A cap at p99 (922 sh binary / 765 sh bucket) would be more permissive but
  would leave the majority of phantom exposure intact (p99 is still realistic;
  the phantom is concentrated in the multi-hundred to multi-thousand share
  single-dump range, which is the tail *above* p99).

A cap at p50 (35 sh both kinds) would be overly conservative — it would cap
below our own live per-clip medians (141 sh binary, 46 sh bucket) and would
badly understate achievable entry/exit fill sizes on tight binary books.

### Per-leg variation

Within each kind the p90 varies significantly across legs:

- Binary: range 125–200 sh per leg (p90). The 166 sh aggregate is a clean
  pooled estimate.
- Bucket: range 124–250 sh per leg (p90). #1670 (the doom-loop) has the
  highest bucket p90 at 226 sh; #1610 (the highest-volume bucket leg, n=1,210)
  has p90 of 124 sh. A single pooled 142 sh is correct on average but will
  over-cap the doom-loop bucket (where clips run larger because the desperate
  wide-spread book occasionally prints a big clip) and under-cap #1610 slightly.

For the first implementation (SHR-107) a single scalar per kind is the right
trade-off. A per-leg lookup table is not warranted with only 4 legs per kind.

---

## Reconciliation with SHR-104 own-fills

The cap applies **per-clip** (per on-venue print within a marketable IOC), not
per-order. The SHR-104 per-order CSV has `filled` = total fill across all
sub-clips in one clustered IOC. The per-clip values from the SHR-104 research
note are:

| kind | per-clip median | per-clip p90 | per-clip max |
| :-- | --: | --: | --: |
| binary | 141 sh | 501 sh | 542 sh |
| bucket | 46 sh | 355 sh | 515 sh |

**Binary:** the proposed 166-sh cap is above the per-clip median (141 sh) but
below the per-clip p90 (501 sh). This means a typical single clip on a tight
binary book (median 141 sh real, but our own clips median 141 sh = exactly the
market median) is uncapped; only the larger clips (>166 sh) are trimmed. That
is the right place to cut: clips >166 sh exist in the market but are rare
(~10% of all binary clips), and our own live IOC never achieved the sim's
typical uncapped 302–516 sh dump in a single clip.

Importantly, per-order total fills on binary can legitimately reach 200+ sh
(SHR-104 binary per-order median = 200 sh) because an order spans **multiple
clips** re-fired over seconds. The cap limits per-clip, so a multi-clip order
can still accumulate 200+ sh total across several 100–166-sh clips. This is
consistent with the SHR-104 finding that per-order depth is broadly hittable
(median ratio_top = 1.00).

**Bucket:** the proposed 142-sh cap is well above the per-clip median (46 sh)
— most real bucket clips are far below the cap. The cap primarily trims the
#1670-style outlier where a wide-book print occasionally lands at 200–947 sh.
The sim's 516-sh single-dump is above the bucket p90 (142 sh), so the cap
will directly reduce the doom-loop phantom.

**Cross-check: the cap does not contradict p90 ratio_top = 3.41 (SHR-104).**
The p90 ratio_top of 3.41 means some orders fill 3× the displayed top level —
that is the book *refilling across multiple re-fires within one clustered
order*, not a single 3× clip. Per-clip, each refill is a new clip bounded by
the cap. The cap + re-fire floor together correctly model this: one clip fills
≤ cap, then the re-fire floor (0.75 s) enforces a delay before the next clip,
matching the live cadence.

---

## What the cap does NOT fix

- **Structural wide-bucket doom-loop (SHR-102):** the strategy still enters
  positions it cannot exit profitably on very wide buckets. The cap reduces
  per-clip fill size (making each re-fire smaller), but the strategy will still
  re-fire repeatedly until the loss cap triggers. This is a strategy-level
  defect, not a fill-model defect.
- **Shared inventory cap (SHR-91):** the sim enters multiple positions
  simultaneously where live serializes entries. The cap only affects how large
  each individual clip is.
- **Open-burst v1 phantom (SHR-106):** v1 binary cells show 100% phantom
  (`hitSz = 0`) — the sim fills full-size at open against depth with zero
  contemporaneous prints. The clip cap will reduce each clip from ~302 to ≤166
  sh, but the core issue is the strategy firing at open when no real taker has
  moved; this may require a marketability re-check fix (SHR-98) rather than
  just a clip cap.

---

## Parameter spec

### Proposed `RunConfig` additions

Mirrors the existing `min_inter_order_seconds` pattern: a documented default
calibrated from the live data, plus an explicit `None` / `0.0` A/B-disable arm.

```python
# hlanalysis/backtest/runner/hftbt_runner.py → class RunConfig:

# SHR-107: per-IOC clip-size cap (shares). After the depth-limited fill
# (mechanism 1) computes the raw fill quantity from the displayed book, this
# cap further limits the per-clip fill to the observed market p90. Real takers
# catch only tens-to-~150 shares per IOC on wide bucket books and ≤166 shares
# on tight binary books (median 35 sh across 112k ex-own market clips,
# 06-06→06-09 corpus, SHR-104/SHR-105). Without a cap the sim can cross
# 500+ shares in a single clip — larger than essentially every real print.
#
# Regime-keyed: binary legs use ioc_clip_cap_binary_shares (tight books,
# calibrated from p90 = 166 sh); bucket legs use ioc_clip_cap_bucket_shares
# (wide books, calibrated from p90 = 142 sh). ``None`` disables the cap for
# that kind (A/B arm = legacy uncapped behaviour).
#
# Measured calibration values (docs/research/2026-06-11-hl-clip-cap-calibration-shr105.md):
#   binary p90 = 166 sh (n=110,708 ex-own clips across 4 binary legs)
#   bucket p90 = 142 sh (n=1,534  ex-own clips across 4 bucket legs)
#
# Default None (disabled) preserves bit-identical legacy behaviour.
# Set to 166 (binary) / 142 (bucket) for the calibrated HL arm.
ioc_clip_cap_binary_shares: float | None = None
ioc_clip_cap_bucket_shares: float | None = None
```

### CLI flags

```
--ioc-clip-cap-binary-shares FLOAT
    (SHR-107) Per-IOC clip-size cap for binary legs (shares). A single
    marketable IOC on a binary leg fills at most this many shares; larger
    displayed depth is partially filled, with the remainder cancelling and
    re-firing under the existing inter-order floor. Calibrated from the
    market p90 of the ex-own print distribution (166 sh, 06-06→06-09
    corpus; docs/research/2026-06-11-hl-clip-cap-calibration-shr105.md).
    Default: None (disabled; legacy uncapped A/B arm).

--ioc-clip-cap-bucket-shares FLOAT
    (SHR-107) Per-IOC clip-size cap for bucket legs (shares). Same
    mechanism as --ioc-clip-cap-binary-shares. Calibrated from market p90
    of bucket prints (142 sh). Default: None (disabled).
```

### Single-scalar simplification

If regime-split is too complex for a first cut, a single
`ioc_clip_cap_shares: float | None = None` scalar applied to both kinds is
acceptable. The calibration value for the combined pool is also 166 sh (the
binary-dominated aggregate). This slightly over-restricts bucket legs (142 vs
166 sh cap) and is the more conservative choice.

### A/B disable arm

`ioc_clip_cap_binary_shares=None` (or `0.0` by convention, same as
`min_inter_order_seconds`) preserves the pre-SHR-107 behaviour exactly. The
disable arm is the baseline in the re-run matrix (SHR-107).

### Integration with the existing IOC model

The cap inserts between mechanism 1 (depth-limited fill) and the fill
recording:

```
raw_fill_qty = depth_limited_fill(book, limit, lot_size)   # existing
if ioc_clip_cap_shares is not None:
    raw_fill_qty = min(raw_fill_qty, ioc_clip_cap_shares)  # NEW (SHR-107)
record_fill(raw_fill_qty, ...)
# remainder (if any) cancels → re-fires after min_inter_order_seconds
```

The cap does **not** replace mechanism 1; it is an additional upper bound.
Mechanism 1 (walking the ladder) remains the primary fill-size governor; the
cap trims the tail where the displayed ladder is thick but real takers never
actually sweep it all at once.

---

## Caveats

- **Four legs, one week.** The calibration rests on the 06-06→06-09 corpus
  (4 binary legs, 4 bucket legs). The market clip distribution is likely stable
  across weeks for established HIP-4 markets, but the per-leg p90 range is
  wide (125–200 sh binary, 124–250 sh bucket). A longer corpus would tighten
  this. Use p90 not p99 precisely because p99 is driven by rare institutional
  sweeps that are corpus-dependent.
- **Binary mega-clips.** Binary max clips of 10,000–18,000 sh do exist in the
  tape. These are real (not recorder artefacts) and represent rare large market
  sweeps that deep tight binary books occasionally support. The p90 cap (166 sh)
  correctly ignores these in the 90th percentile sense. Our strategy size
  ($300–$1,000 notional ÷ price ≈ 300–1,000 sh) would only encounter a
  mega-clip if we ran a very large position into one — not the current regime.
- **Bucket sample is thin.** The four bucket legs total only 1,534 market
  clips (vs 110,708 binary). The bucket p90 (142 sh) is robust to the
  per-leg variation seen in the corpus but should be re-validated when more
  bucket data accumulates.
- **No mid-width regime.** All binary legs are tight, all bucket legs are wide
  in this corpus. A mid-width regime (spread 0.02–0.10) was not observed. The
  SHR-104 per-order breakdown by width shows n=25 mid-width orders at
  median ratio_top = 1.00 (same as tight/wide) — if mid-width legs appear live,
  use the aggregate 166-sh cap as a starting point until more data is available.
- **cap ≠ hittable fraction.** A fractional multiplier on displayed depth (the
  original SHR-103 framing) was rejected by SHR-104: per-order depth is
  broadly hittable (median 1.00). The cap is an absolute share count, not a
  fraction; do not confuse the two in the SHR-107 implementation.

---

## Artifacts

- Tool: `tools/hl_clip_cap_calibration.py` (pure analysis + CLI; reads the
  SHR-104 summary CSV and optionally the per-order CSV for reconciliation).
- Tests: `tests/unit/tools/test_hl_clip_cap_calibration.py` (32 unit tests).
- Input: `docs/research/2026-06-11-hl-market-clips-ex-own-summary.csv`
  (SHR-104 output, 8 per-leg rows + 3 aggregate rows).
- Input: `docs/research/2026-06-11-hl-own-fills-displayed-vs-filled.csv`
  (SHR-104 per-order table, for reconciliation).

To re-run:

```bash
uv run python tools/hl_clip_cap_calibration.py \
    --summary docs/research/2026-06-11-hl-market-clips-ex-own-summary.csv \
    --own-clips-csv docs/research/2026-06-11-hl-own-fills-displayed-vs-filled.csv \
    --cap-percentile p90
```
