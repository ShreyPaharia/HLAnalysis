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

**Caveat — regime-tilted PnL.** The +$95 is concentrated in the recent (June)
regime; bipower is net *negative* in early May (05-14 −96, 05-18 −52, 05-27
−110). So the PnL edge is regime-dependent, but the **−49% churn reduction is
regime-independent and structurally protective** for the doom-loop.

**Recommendation:** binary → keep `sample_std`; bucket → `bipower` is a genuine
improvement (entry-side doom-loop complement, halves churn, +10% recent-regime
PnL). It is a single per-class `theta_override` (`vol_estimator: bipower`), easily
reverted. Operator call on whether to flip live now vs shadow-validate over a few
settlements given the regime tilt + early-May tail days.
