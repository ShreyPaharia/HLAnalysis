# Independent sim-fill validation against the recorded HL trade tape (SHR-106)

**Date:** 2026-06-11
**Branch:** `feat/shr-106-sim-fill-tape-validation`
**Scope:** VALIDATION / TOOLING-ONLY. No engine/strategy/fill-model/config change, no
deploy. A check *on* the fill model (SHR-103 epic), independent of the implementation
ticket SHR-107.
**Tool:** `tools/sim_fill_tape_validation.py` (sits alongside `tools/_compare_ioc.py`).

---

## TL;DR

1. **The standing comparison (`_compare_ioc.py`) joins sim fills to the live
   `user_fills` CSV / settlement PnL.** That scores *aggregate PnL*, but cannot tell
   you whether an individual sim fill was even *possible*. This harness uses the
   recorded **trade tape** — every on-venue print with price, size, aggressor side —
   as an independent ground truth, and flags **phantom-liquidity fills**: sim fills
   the displayed book absorbed but that no taker ever actually swept.

2. **Across the 16-cell matrix the shipped IOC model (re-fire floor 0.75 s) is
   84% phantom by filled size** — 99 of 127 sim fills (78%), **15,379 of 18,301
   filled shares**, ≈ **$13,419** of phantom notional. The result is *robust to the
   match window*: ±0.5 s → 89.5%, ±1 s → 84%, ±2 s → 81%, ±5 s → 77%. Counting
   either aggressor side (looser) still leaves 77.7%.

3. **The re-fire floor barely moves phantom exposure** — `base` (no floor) is
   85.0% / $13,588 vs `ioc` 84.0% / $13,419 (**−1.2%**). This *independently
   corroborates* `2026-06-11-ioc-refire-floor-hl-fill-model.md`: the floor is a small
   cadence lever; the binding constraint is **per-IOC clip size** (the sim crosses the
   whole displayed book at once while real takers catch ~tens of shares per IOC), i.e.
   **SHR-107**, not re-fire churn.

4. **Two phantom regimes show up distinctly.** (a) *Open-burst entries*: many v1
   binary/bucket cells are **100% phantom with `hitSz = 0`** — the sim fills its full
   clip at market open against displayed depth, but **zero** trade prints at-or-through
   that level land in the window (on #1670 the first real print is ~51 min after the
   sim's first fill). (b) *Wide-bucket doom-loop*: #1670 (0607 v31 bucket) fills 4,718
   sim shares of which only **1,113** were backed by contemporaneous flow → 4,176
   phantom. Both are real depth-vs-hittable gaps, not recorder gaps (`no_tape = 0`
   everywhere).

---

## Method

For each sim IOC fill (the non-settle, non-hedge rows of each cell's `fills.parquet`):

* **Window.** Take the recorded prints for the fill's symbol within `[t ± w]`
  (`w = 1.0 s` default; the fill timestamp already includes the SHR-89 arrival latency).
* **At-or-through.** Keep prints at `price ≤ limit` for a BUY (we lift asks at/below
  our limit) or `price ≥ limit` for a SELL. By default also require the print's
  **aggressor side to match** the sim fill's direction — a sim BUY is only "covered" by
  a real taker who *also* bought (consumed ask depth).
* **Greedy volume ledger.** Within a symbol, fills are processed earliest-first and each
  consumes real print volume; consumed shares are **not** available to a later fill. So
  when the strategy re-fires across two snapshots of the same fleeting level, the first
  fill is credited and the second is correctly flagged. This reproduces the documented
  **#2230** case (2×100 sim vs 100 traded → **100** phantom; a naïve per-fill match
  would double-credit the 100 and report **0**).
* **Phantom.** `phantom_excess = size − tape_filled`; a fill is phantom when
  `tape_filled < size`. `hitSz` in the table is the *raw windowed sum* (double-countable,
  diagnostic); `phSz` is the *ledger* (non-double-counted) phantom that drives the verdict.

**Honest limitations.** (1) The ledger consumes within each fill's window in time order;
total credit is bounded by what really traded *and* by each fill's window, so it is a
fair, deterministic estimate — never crediting more than the tape shows. (2) A symbol with
no recorded prints at all is reported as `no_tape`, never silently counted as phantom (it
was 0 in every cell here). (3) The tape is the *independent* truth precisely because it is
not the order book — a fill can be faithful to the displayed book (SHR-79/89 verified) and
still phantom against the tape; that gap is the whole point.

---

## Results — 16-cell matrix

Shipped model (`ioc`, re-fire floor 0.75 s), `window = ±1.0 s`, aggressor-matched:

```
day   slot kind    | fills phantom no_tape |     simSz     hitSz      phSz |       ph$ | ph%fill   ph%sz
--------------------------------------------------------------------------------------------------------
0606  v1   binary  |     1       1       0 |       302         0       302 |    299.99 |  100.0%  100.0%
0606  v1   bucket  |    15      14       0 |       871        20       851 |    698.08 |   93.3%   97.7%
0606  v31  binary  |    28      18       0 |      3407      2292      2739 |   2410.13 |   64.3%   80.4%
0606  v31  bucket  |     3       2       0 |      1050        67      1032 |    769.86 |   66.7%   98.3%
0607  v1   binary  |     2       2       0 |       311         0       311 |    300.42 |  100.0%  100.0%
0607  v1   bucket  |     2       2       0 |       611         0       611 |    504.06 |  100.0%  100.0%
0607  v31  binary  |     4       4       0 |       556         0       556 |    474.27 |  100.0%  100.0%
0607  v31  bucket  |    41      32       0 |      4718      1113      4176 |   3541.92 |   78.0%   88.5%
0608  v1   binary  |     1       1       0 |       302         0       302 |    300.00 |  100.0%  100.0%
0608  v1   bucket  |     1       1       0 |       305         0       305 |    299.99 |  100.0%  100.0%
0608  v31  binary  |     4       2       0 |       575       588       282 |    245.53 |   50.0%   49.1%
0608  v31  bucket  |     2       1       0 |       500       100       400 |    388.00 |   50.0%   80.0%
0609  v1   binary  |     2       2       0 |       307         0       307 |    300.00 |  100.0%  100.0%
0609  v1   bucket  |     3       3       0 |       901         0       901 |    891.29 |  100.0%  100.0%
0609  v31  binary  |    13       9       0 |      2589      3867      1308 |   1064.17 |   69.2%   50.5%
0609  v31  bucket  |     5       5       0 |       996         0       996 |    931.26 |  100.0%  100.0%
--------------------------------------------------------------------------------------------------------
TOTAL              |   127      99       0 |     18301      8047     15379 |  13418.96 |   78.0%   84.0%
```

**Arm A/B against the tape** (the reason this is a *reusable* check — score any future
fill-model change against the tape, not just the CSV):

| arm | fills | phantom fills | phantom size | phantom $ | ph % size |
| --- | ----: | ----: | ----: | ----: | ----: |
| **`base`** (no floor) | 114 | 92 | 15,577 | $13,587.68 | **85.0%** |
| **`ioc`** (floor 0.75 s) | 127 | 99 | 15,379 | $13,418.96 | **84.0%** |

The floor nets **−$169 (−1.2%)** of phantom exposure — a small, correct nudge, exactly
as the re-fire-floor analysis predicted. The headline gap is **not** cadence.

**Window / aggressor sensitivity** (`ioc` arm totals):

| variant | ph % fills | ph % size | phantom $ |
| --- | ----: | ----: | ----: |
| window ±0.5 s | 85.8% | 89.5% | 14,317.55 |
| **window ±1.0 s (default)** | **78.0%** | **84.0%** | **13,418.96** |
| window ±2.0 s | 70.9% | 81.1% | 12,963.33 |
| window ±5.0 s | 61.4% | 77.1% | 12,441.08 |
| ±1.0 s, either aggressor | 74.8% | 77.7% | 12,364.06 |

Even at a generous ±5 s window and counting either side, **≥77% of filled size is
phantom** — the displayed depth genuinely did not trade.

---

## Interpretation → ticket map

| signal | reading | ticket |
| --- | --- | --- |
| 84% of filled size phantom, robust to window | the sim crosses displayed depth that no taker swept; **displayed ≠ hittable** | **SHR-103 / SHR-107** (trade-flow-limited IOC / per-IOC hittable haircut) |
| floor moves phantom only −1.2% | re-fire cadence is not the binding constraint | confirms `2026-06-11-ioc-refire-floor…` |
| v1 cells 100% phantom, `hitSz = 0` (open-burst) | full clip filled at open against depth with **zero** contemporaneous prints | **SHR-107** clip cap; partly **SHR-91** (over-entry) |
| #1670 doom-loop 4,176/4,718 phantom | structural wide-book over-cross | **SHR-102** (strategy spread gate) + **SHR-91** (shared inventory cap) |
| `no_tape = 0` everywhere | no recorder coverage gaps — the phantom is real, not missing data | — |

**Bottom line.** The trade tape independently confirms the SHR-103 thesis with a hard
number: **~84% of the HL backtest's filled size has no contemporaneous on-venue print to
back it.** This is the quantity SHR-107's trade-flow-limited fill must shrink, and this
harness is the standing scorecard to measure it — re-run it after the SHR-107 change and
the phantom-size total should collapse.

---

## How to run

```bash
# 1. (re)generate the 16-cell sim matrix (both arms) — gitignored, reproducible
bash tools/_run_ioc_matrix.sh

# 2. score the shipped arm against the tape (vs the no-floor base arm), and
#    write the per-fill verdicts parquet for drill-down
HLBT_HL_DATA_ROOT=$PWD/data uv run python tools/sim_fill_tape_validation.py \
    --prefix ioc_ --baseline-prefix base_ \
    --out data/sim/runs/_tape_verdicts.parquet
```

`--report <path>` dumps the raw table to a scratch markdown file (this curated
page is hand-written around those numbers, not generated).

Knobs: `--window-seconds` (match window), `--no-match-aggressor` (count either side),
`--price-tol`. The matching / aggregation core is pure and unit-tested in
`tests/unit/tools/test_sim_fill_tape_validation.py`.

## Artifacts

- Tool: `tools/sim_fill_tape_validation.py` (pure core + IO + CLI).
- Tests: `tests/unit/tools/test_sim_fill_tape_validation.py` (19 tests).
- Per-fill verdicts (gitignored): `data/sim/runs/_tape_verdicts.parquet`
  (`arm, cloid, symbol, side, price, size, ts_ns, n_prints, hittable_size,
  tape_filled, tape_covered, is_phantom, phantom_excess, phantom_notional`).
- Sim runs (gitignored, reproducible): `data/sim/runs/{ioc,base}_*`.
- Trade tape: `data/venue=hyperliquid/product_type=prediction_binary/.../event=trade/`.
