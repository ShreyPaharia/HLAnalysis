# HL v31 binary: `sample_std` vs `bipower` — full-corpus estimator sweep

**Date:** 2026-06-12
**Question:** Live HL v31 binary runs `vol_estimator=sample_std`. `bipower` is
jump-robust and flips knife-edge entry timing on vol-burst days (prior finding:
06-08 one market −$4.72 vs +$58). Should HL v31 binary move to `bipower`?
**Verdict: NO — keep `sample_std`.**

## Method

`tools/_estimator_sweep.py`. For EVERY recorded HL settlement day (35 days,
2026-05-07 → 06-10), run the v31 priceBinary cell twice with ALL params held
equal and only `vol_estimator` varied (`sample_std` → `bipower`). Live-faithful
flags: `--data-source hl_hip4 --ref-source hl_perp --ref-event mark
--reference-ticks raw --scan-mode event --no-cache`, `fee_taker=0` (HL HIP-4 is
fee-free). Caps off on both arms (held equal → the delta isolates the estimator).
Base params dumped from the live `v31`/`priceBinary` slot via
`backtest_params_from_slot`.

## Result

| metric | `sample_std` (live) | `bipower` | Δ |
| --- | ---: | ---: | ---: |
| total PnL (35 days) | **$1105.77** | $1084.02 | **−$21.75** |
| fills (churn) | 984 | 623 | −361 (−37%) |
| days better / worse / flat | — | 13 / 12 / 10 | mixed |

**bipower is flat-to-slightly-worse in aggregate and higher-variance.** The
aggregate is dominated by a single tail day:

- **2026-05-19: $83.30 → −$125.96 (−$209).** RCA: same question (Q1000060,
  settled "yes"), both arms trade the final ~40 min before settlement;
  `bipower`'s σ leads it to lose $126 where `sample_std` makes $83. A **real
  strategy difference, not a data artifact** (identical inputs, same outcome) —
  the knife-edge-σ-at-the-entry-boundary failure, here near settlement.

Excluding 05-19, bipower would be +$187 ahead, but you don't get to exclude a
real loss day. bipower also has the bigger wins (06-06 +$94, 05-12 +$73, 06-08
+$63) — it is simply **higher-variance both directions**.

## Why keep `sample_std`

1. **Aggregate is a wash / slightly worse** (−2% over 35 days).
2. **Variance matters more as sizes grow.** `bipower`'s −$126 tail day is exactly
   the outcome to avoid when increasing capital; `sample_std` is more consistent.
3. **The churn reduction does not pay.** HL HIP-4 is fee-free, so −37% fills
   yields no direct fee saving (it only helps via spread-crossing, which the
   fill model under-models — a sim-fidelity gap, not a reason to flip live).
4. **The recent-regime edge isn't robust.** bipower wins 06-04→06-10 decisively
   (+$243) but loses across much of May; that's regime luck, not a durable signal.

**No live change on binary.** `sample_std` stays for the binary cell.

## Bucket cell — OPPOSITE result

Same method, `--kind bucket`, live `v31`/`priceBucket` params (which include the
deployed `exit_spread_hold=0.04` and also default to `sample_std`).

| metric | `sample_std` (live) | `bipower` | Δ |
| --- | ---: | ---: | ---: |
| total PnL (35 days) | $921.76 | **$1017.12** | **+$95.36 (+10.3%)** |
| fills (churn) | 941 | 482 | **−459 (−49%)** |
| days better / worse / flat | — | 19 / 9 / 7 | bipower wins |

On buckets bipower is **clearly better and halves churn.** Structural reason:
buckets have persistently WIDE books (the doom-loop habitat) where churn =
spread-crossing bleed. bipower's jump-robust σ cuts spurious entries → fewer
fills → attacks the **entry** side of the doom-loop, complementing the
`exit_spread_hold` exit-side fix. On binary (tight, fee-free books) the churn
cut is a no-op; on buckets it is the whole game — and the fill model
*understates* its live value (deterministic fills don't fully price the wide
spread bipower avoids).

**The +$95 hides a winner-abandonment left tail (decisive).** Drilling into the
loss days: on **5 of 6** bucket "WORSE" days, bipower flips the day's hit rate
from **100% → 0%** — it ABANDONS a favorite that eventually WON, via a premature
σ-driven exit during an intraday dip, locking in a real loss right before
settlement (05-27 Q20 *yes*: sample_std holds to 05:16 → +$28/hit 100%; bipower
exits 04:51 → −$82/hit 0%/maxDD $82. 05-14 Q6 *yes*: ss +$57/100% vs bp
−$39/0%). These are REAL losses, not thinner profit, and they persist DESPITE the
deployed `exit_spread_hold=0.04` (so the abandonment is via `exit_safety_d`/stop,
not the spread gate). The PnL edge is also recent-regime tilted (net negative
early May). The −49% churn cut is real, but on the bad days it co-occurs with
abandoning the winner — "less churn" ≠ "better" uniformly.

## Recommendation (concrete)

**Keep `sample_std` on BOTH cells. Do not flip live.**

- **Binary:** bipower flat-to-worse (−$22), higher variance, one −$126 tail.
- **Bucket:** +$95 mean is seductive but bought with a **winner-abandonment fat
  left tail** (hit 100%→0% on ~5 days, −$82/−$39 losses). For a desk whose goal
  is to *increase sizes*, that left tail is disqualifying — at 5× it is
  −$410/−$195 days. Positive-mean, wrong-shaped distribution.
- **Pursue the churn win differently.** The −49% churn is worth capturing, but
  via the **entry-side spread gate** (real `(ask−bid)/2` vs the static
  `half_spread_assumption=0.005`) to SKIP all-day-wide bucket books — that cuts
  churn as a *no-entry*, not a *premature-exit*, so it never abandons a winner.
  That is the SHR-102-family `entry_spread_gate` lever (off by default), the
  clean next step. Swapping the vol estimator is the wrong tool.

Revisit bipower only if a σ-near-settlement / hold-to-settlement guard is added
that removes the winner-abandonment, or after the entry-gate churn fix lands.
