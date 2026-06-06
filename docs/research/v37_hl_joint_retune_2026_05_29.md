# v3.7 JOINT RETUNE — HL HIP-4 cadence × lookback × alpha

**Branch:** `feat/hl-joint-retune-cadence-lookback-alpha`
**Date:** 2026-05-29
**Strategy:** v3.6 universal (`jr-tilt-ma_sigma-lb30-a100`), pure config sweep — no strategy/engine code touched.
**Corpus:** HL HIP-4 BTC binaries, 2026-05-09 → 2026-05-27, 18 questions.
**Grid:** `vol_sampling_dt_seconds ∈ {3,5,7,10}` × `momentum_mr_lookback_min ∈ {5,10,15,30}` × `momentum_mr_alpha_tilt ∈ {0.75,1.0,1.5}` = 48 cells + 3 anchor baselines.
**Decision:** **NO STRICT PARETO IMPROVEMENT — keep the T13 winner as the shippable config.** A robust directional finding (shorter lookback at fine cadence) warrants a low-risk follow-up tweak; see Recommendation.

---

## TL;DR

- The strict ship gate (PnL > AND Sharpe > AND **maxDD <** T13 best) yields **zero** ships — but only because **maxDD is a hard structural floor at $26.28 across 46/48 cells**. It is the single position-capped loss on Q1000060 (−$26.28), identical in every cell. It is not a tunable quantity on this grid, so `maxDD <` can never fire. The gate is degenerate here, not informative.
- Relaxing `maxDD <` → `maxDD ≤` (justified by the floor), **many cells beat T13 on both PnL and Sharpe** at equal DD. The headline cell is `dt=3/lb=5/a=1.5` → **$344.10 / Sharpe 21.27** (+$27.84 / +2.67 vs T13).
- **But the edge is fragile.** The +$25–28 lead over T13 is ~90% one question (**Q1000095**) that flips from −$10.46 to +$11.83 **only at dt=3**. At dt=5 the same short lookback keeps Q1000095 negative. Per the spec's own warning ("single-cell wins on 18 markets are sampling noise"), this fails the *spirit* of the robustness check even though it is mechanically top-5 at dt∈{3,5}.
- **Robust, mechanism-aligned signal that survives the noise:** shorter lookback (lb=5–15) uniformly beats lb=30 across the entire dt∈{3,5} plateau, and there is a sharp **PnL cliff between dt=5 and dt=7** (the whole cadence benefit lives at dt≤5).

---

## Anchor baselines (all dt=60 except T13)

| Cell           | dt | lb | α   | PnL     | Trades | Hit   | Sharpe | maxDD   |
|----------------|----|----|-----|---------|--------|-------|--------|---------|
| base_v31_dt60  | 60 | —  | off | $263.57 | 62     | 77.8% | 15.67  | $36.36  |
| base_v36_dt60  | 60 | 30 | 1.0 | $267.01 | 72     | 83.3% | 16.56  | $32.74  |
| **t13_best_dt5** | **5** | **30** | **1.0** | **$316.26** | **60** | **88.9%** | **18.60** | **$26.28** |

`base_v36_dt60` reproduces the documented v3.6 HL number ($267.01 / 72 trades / Sharpe 16.56 / maxDD $32.74) exactly, and `t13_best_dt5` (= dt5/lb30/a1.0 in the grid) reproduces T13 bit-for-bit ($316.26 / Sharpe 18.60). Sanity confirmed.

---

## Heatmap — PnL ($) by dt × lookback, one table per α

**α = 0.75**

| dt \ lb | 5       | 10      | 15      | 30      |
|---------|---------|---------|---------|---------|
| **3**   | 341.24  | 326.13  | 323.96  | 319.89  |
| **5**   | 324.07  | 319.13  | 324.74  | 323.09  |
| **7**   | 273.26  | 287.83  | 277.49  | 278.81  |
| **10**  | 275.99  | 266.79  | 272.39  | 274.30  |

**α = 1.0**

| dt \ lb | 5       | 10      | 15      | 30      |
|---------|---------|---------|---------|---------|
| **3**   | 341.24  | 327.26  | 332.48  | 325.47  |
| **5**   | 324.49  | 316.49  | 327.96  | 316.26* |
| **7**   | 273.38  | 281.11  | 271.71  | 280.82  |
| **10**  | 268.81  | 276.66  | 276.62  | 276.50  |

`*` dt5/lb30/a1.0 = the T13 anchor ($316.26).

**α = 1.5**

| dt \ lb | 5        | 10      | 15      | 30      |
|---------|----------|---------|---------|---------|
| **3**   | **344.10** | 331.93  | 332.84  | 328.22  |
| **5**   | 317.23   | 314.17  | 320.44  | 313.78  |
| **7**   | 273.02   | 273.23  | 273.81  | 283.45  |
| **10**  | 278.05   | 275.73  | 275.31  | 275.45  |

Sharpe and maxDD track PnL tightly: every dt∈{3,5} cell has Sharpe 18.3–21.3 and maxDD $26.28; every dt∈{7,10} cell has Sharpe 14.0–15.8 and maxDD $26.28 (two dt=7/α=1.5 cells leak to $27.35/$28.70). Full metrics in `data/sim/runs/v3-7-hl-joint-retune-2026-05-29/summary.json`.

### The dt=5 → dt=7 cliff (most important structural finding)

```
best PnL per dt:   dt=3 → $344    dt=5 → $328    dt=7 → $288    dt=10 → $278
```

The *worst* dt=5 cell ($313.78) still beats the *best* dt=7 cell ($287.83). There is a discontinuous drop of ~$30–40 between dt=5 and dt=7 — the entire cadence advantage from the v3.7 spike lives at **dt ≤ 5**. T13 only sampled dt∈{60,10,5,1} and so saw dt=5 as the sweet spot; this grid shows (a) dt=3 is modestly better still, and (b) dt=7 already throws nearly all of it away. dt=10 here (~$276) matches T13's dt=10 ($276.50) — independent cross-check that the grid is consistent with the prior run.

---

## Strict ship gate (per spec)

Require strict Pareto improvement over T13 best on **(PnL >, Sharpe >, maxDD <)**.

| Criterion        | T13 best | Best grid cell achievable | Pass? |
|------------------|----------|---------------------------|-------|
| PnL >            | $316.26  | $344.10 (dt3/lb5/a1.5)    | ✅    |
| Sharpe >         | 18.60    | 21.27 (same cell)         | ✅    |
| maxDD <          | $26.28   | $26.28 (floor, 46/48 cells)| ❌   |

**Strict Pareto improvements found: 0.** The `maxDD <` leg is unsatisfiable: $26.28 is the position-capped single loss on Q1000060 and is identical in every cell, so no parameter setting on this grid reduces it. Per the literal gate, the conclusion is **keep the T13 winner**.

This is the right call given the corpus size, but for the right reason it needs the fragility analysis below, not the degenerate maxDD leg.

---

## Robustness (top-5 by PnL at two consecutive dt)

| (lb, α)      | top-5 at dt | in high-PnL plateau? |
|--------------|-------------|----------------------|
| (5, 0.75)    | 3, 5        | ✅ yes (dt3 $341 / dt5 $324) |
| (5, 1.0)     | 3, 5        | ✅ yes (dt3 $341 / dt5 $324) |
| (15, 1.0)    | 3, 5        | ✅ yes (dt3 $332 / dt5 $328) |
| (30, 0.75)   | 5, 7        | spans the cliff      |
| (10, 1.0)    | 7, 10       | low-PnL region       |
| (30, 1.0)    | 7, 10       | low-PnL region       |

The headline cell **(lb=5, α=1.5)** is top-5 **only at dt=3** — it is *not* robust. The robust short-lookback combos in the high-PnL plateau are **(5, 0.75)**, **(5, 1.0)** and **(15, 1.0)**.

---

## Per-question: candidate winner vs T13

`dt3/lb5/a1.0` ($341.24) vs T13 `dt5/lb30/a1.0` ($316.26). Δ = +$24.98.

| qid       | outc | T13 PnL | new PnL | Δ        |
|-----------|------|---------|---------|----------|
| Q1000095  | no   | −10.46  | 11.83   | **+22.29** |
| Q1000040  | yes  | 15.28   | 17.60   | +2.32    |
| Q1000050  | no   | 33.01   | 35.02   | +2.01    |
| Q1000080  | no   | 8.77    | 10.75   | +1.98    |
| Q1000105  | no   | 23.71   | 25.22   | +1.51    |
| Q1000070  | yes  | 35.14   | 36.55   | +1.41    |
| Q1000085  | yes  | 35.77   | 36.64   | +0.87    |
| Q1000025  | yes  | 16.84   | 15.98   | −0.86    |
| Q1000060  | yes  | −26.28  | −26.28  | 0 (DD floor) |
| Q1000055  | no   | 2.43    | 0.00    | −2.43    |
| Q1000020  | yes  | 3.25    | −0.87   | −4.12    |
| *(8 others)* | | | | 0.00 |

**The win is one question.** Q1000095 alone is +$22.29 of the +$24.98 total; the other 17 net +$2.69. And Q1000095 only flips at dt=3:

| Q1000095 PnL | dt3/lb5 | dt3/lb30 | dt5/lb5 | dt5/lb15 | dt7/lb10 | T13 |
|--------------|---------|----------|---------|----------|----------|-----|
|              | +11.83  | +1.18    | −9.61   | −10.46   | −10.04   | −10.46 |

It requires **dt=3 AND lb≤10** to go positive. At dt=5 (the robustness-confirming cadence) the same short lookback leaves it negative — so the dt=3 PnL lead does **not** carry its source question to dt=5. Worse, Q1000020 moves the *opposite* way (prefers coarser cadence: dt3 −0.87 → dt5 +12.09 → dt7 +20.15), so the two big swing questions partially cancel and trade places as you change cadence. This is textbook 18-market sampling noise.

---

## Hypothesis test

**H1 — optimal lookback DECREASES at finer cadence: SUPPORTED (directionally).**
At dt=3, lb=5 is the clear best across all three α (341–344), and lb=30 is the worst within dt=3 (320–328). The 30-min default genuinely over-smooths at dt=3 — exactly the predicted mechanism (a 30-min window holds 600 samples at dt=3 vs 30 at dt=60; the moving-average / σ-spread washes out real moves). The effect is strongest and cleanest at dt=3; at dt=5 lb=5 and lb=15 are within $4, and at dt≥7 lookback ordering is noise on top of the cliff.

**H2 — optimal alpha INCREASES at finer cadence: WEAKLY SUPPORTED / mostly inert.**
At dt=3 the best α is 1.5 (344 vs 341 for 0.75/1.0); at dt=5 lower α wins (lb15: a1.0 328 > a1.5 320). So the α optimum does sit highest at the finest cadence, consistent with the JR-trust-is-sharper-with-more-samples story. **But the magnitude is tiny** — at dt=3/lb=5 the three α values span just $341.24→$341.24→$344.10 ($2.86), and a0.75 and a1.0 are byte-identical there. JR-trust is clamping the tilt so hard that α is nearly a no-op at fine cadence. Treat H2 as "directionally right, practically negligible."

**H3 — dt=7 matches/beats dt=5 if lookback reduces: REJECTED.**
dt=7 is uniformly and dramatically worse than dt=5 at every lookback (best dt=7 lb=10 = $288 vs worst dt=5 = $314). Reducing lookback at dt=7 does help *within* dt=7 (lb=10 $288 > lb=5 $273) but cannot close the ~$30 cliff. The cadence benefit is a cliff at dt≤5, not a smooth slope.

---

## Decision & Recommendation

**Ship gate (strict, per spec): no Pareto improvement found → keep T13 winner (`dt=5 / lb=30 / α=1.0`).**

Rationale, in order of weight:
1. `maxDD <` is unsatisfiable (structural $26.28 floor) — strict gate cannot fire.
2. The only cell that would win on a relaxed (`maxDD ≤`) gate, `dt=3/lb=5`, derives ~90% of its edge from a single question (Q1000095) that does not replicate at dt=5. That is precisely the sampling-noise failure mode the robustness rule exists to catch.

**Actionable, low-risk follow-up (does NOT require a cadence change beyond what's already planned):**
The one robust, mechanism-aligned, *not*-single-question result is that **shorter lookback beats lb=30 across the whole dt∈{3,5} plateau.** At the cadence the engine port is already targeting (dt=5), dropping **lb 30 → 15 at α=1.0** gives `dt5/lb15/a1.0` = **$327.96 / Sharpe 19.71** vs T13's $316.26 / 18.60 (+$11.70 / +1.11, equal DD), is top-5 at both dt=3 and dt=5, and is **not** Q1000095-driven (Q1000095 stays −$10.46 there; its edge comes from Q1000020 and spread). If we want to bank a robust win without betting on dt=3, that is the cell — bundle the `lb=30→15` change with the dt=60→5 engine port rather than shipping dt=3 chasing a fragile +$13.

**Do not ship dt=3.** Higher headline PnL, but the marginal +$13 over dt=5 is one dt=3-only question. Revisit dt=3 and the lb=5 region when the HIP-4 corpus reaches 3+ months (memory `[[v36_ouz_sdr_jr_2026_05_28]]` / `[[v31_final_state_2026_05_23]]` flag the same 18-market power limit).

---

## Caveats

- **18 questions.** Two swing questions (Q1000095, Q1000020, each ±$20–40) dominate every cross-cell comparison and move in opposite cadence directions. No conclusion about absolute magnitude is load-bearing; only the directional findings (H1, the dt≤5 cliff) are.
- **maxDD is not a discriminator on this corpus** — it is a single position-capped loss (Q1000060), invariant to the swept params. Future sweeps should report per-question loss distribution, not just headline maxDD, when the corpus is this small.
- **No engine code touched.** `engine/market_state.py` `_mark_bucket_ns` is still hardcoded to 60s; the live-engine cadence port (and any lookback change) remains a separate follow-up. Backtest-only impact here.
- **HL-only.** Per `[[v31_final_state_2026_05_23]]`, these params do not transfer to PM. No PM run (needs the kline-cache extension; separate task).

## Artifacts

- Runner: `scripts/run_v37_hl_joint_retune.py`
- Results: `data/sim/runs/v3-7-hl-joint-retune-2026-05-29/{base_*,t13_best_dt5,dt<N>-lb<M>-a<P>}/report.md` + `summary.json`
- Builds on `[[v37_hl_1s_sampling_2026_05_28]]` (T13) and `[[v36_ouz_sdr_jr_2026_05_28]]` (indicator choice).
