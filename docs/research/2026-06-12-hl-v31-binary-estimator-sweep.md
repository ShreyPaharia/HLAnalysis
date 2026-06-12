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

**No live change.** `sample_std` stays. The standing open question is now closed
with hard numbers; revisit only if the bucket/binary fill-model fidelity work
(SHR-89) materially changes how churn is priced, or if a σ-near-settlement guard
is added to tame bipower's tail.
