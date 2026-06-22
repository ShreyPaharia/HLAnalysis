# v31/theta HL binary — safety_d hysteresis band retune

**Date:** 2026-06-22 · **Scope:** HL HIP-4 `priceBinary` v31/theta band
(`min_safety_d` entry floor / `exit_safety_d` mid-hold exit). **Decision:** raise
the band `min_safety_d 2.0 → 2.5`, `exit_safety_d 1.0 → 1.5`. Config-only.

## Motivation

The question that started this: should `safety_d` be conditioned on recent (1-day)
volatility? **No** — `safety_d` is already a z-score,
`d = (ln(S/K) + (μ−½σ²)τ) / (σ√τ)` (`hlanalysis/strategy/_theta_math.py`), so it is
intrinsically vol-normalized; conditioning the *threshold* on σ double-counts
volatility. The real open question is whether the constant threshold *pair* is
near-optimal. It is not.

## Method

- **Corpus:** HL binary, recorded data 2026-05-10 → 2026-06-21, n=42 settled
  questions. Runner `experiments/scripts/run_v31_hl_safetyd_sweep.py` (anchored on
  the current live config: `vlb=900, dt=5`, msd2 revert).
- **Sweep:** `min_safety_d ∈ {0,1,1.5,2,2.5,3}` × `exit_safety_d ∈ {0,0.5,1,1.5,2}`,
  all other knobs frozen at live. Walk-forward early/late split, ranked by
  **worst-half PnL**.
- **Tail stress:** `experiments/scripts/run_v31_hl_safetyd_tail_stress.py` —
  reproduces the 2026-06-18 loss-injection model (each entered favorite held to
  settlement as a win flips to a loss at its own implied rate `1−entry_price`;
  flip costs the full stake under buy-and-hold, the config's empirical adverse-exit
  loss when the soft exit is on). Validated by reproducing the buy-and-hold collapse.
- **Fill-cost stress:** round-trip fee bump (`fee_taker 0→0.0005`,
  `exit_fee 0.0007→0.0025`). NOTE: `--slippage-bps` is a **no-op** on HL recorded-book
  fills (bit-identical PnL at 0/10/25 bps) — use the fee knobs.

## Result

The current live `(msd2.0/esd1.0)` is one of the **worst** cells: worst-half −$27,
the highest maxDD in the grid ($343). The literal grid winner is the
`exit_safety_d=0` column (buy-and-hold) — **rejected** as the known tail-blind trap
(it reproduces the 2026-06-18 collapse: stressed EV $169 / 5th-pct −$1054 /
P(loss) 30%). Among tail-safe (exits-on) configs, a higher+tighter band dominates:

| config | full PnL | worst-half | maxDD | Sharpe | stressed EV | stressed P(loss) | trades |
|---|---|---|---|---|---|---|---|
| **live** 2.0/1.0 | $635 | −$27 | $343 | 4.3 | $415 | 1.5% | 431 |
| 2.5/1.0 (entry-only) | $662 | +$127 | $206 | 6.3 | $464 | 0.8% | 307 |
| **2.5/1.5 (adopted)** | **$721** | **+$178** | **$199** | **7.2** | **$503** | **0.6%** | **329** |
| 2.5/2.0 (upside) | $827 | +$247 | $42 | 13.5 | $711 | 0.0% | 468 |

Fill-cost stress: all three candidates retain ~99% of PnL and the edge over live is
unchanged — the advantage is **not** a churn/fill artifact (the configs hold to
settlement and do occasional protective exits; they are not thrashing).

## Why `2.5/1.5`

Dominates live on every axis (realized worst-half, maxDD, Sharpe, tail-stress EV /
5th-pct / P(loss)) while trading **fewer** round-trips than live (329 vs 431) → least
fill-model exposure. The mechanism is the hysteresis band working as designed: the
higher entry floor (2.5) suppresses the re-entry churn that the more aggressive exit
(1.5) would otherwise cause — which is why this **overturns** the prior
`strategy.yaml` note that "esd=1.0 is the production ceiling / esd=1.5 → re-entry
cascade" (that cascade was measured under the low entry floor). `esd=2.0` is genuine
further upside but earns it through more aggressive exit/re-entry → gate behind a
paper soak. Ultra-conservative fallback = raise the entry floor only (`2.5/1.0`).

## Caveats

- **Single corpus, n=42, tail-deficient** (favorites mostly resolved ITM). The
  *relative* ranking is robust (agrees across realized worst-half + independent tail
  stress + fee stress); absolute $ are inflated.
- Backtest ran at the live binary cadence (`dt=5`), so cadence-mismatch risk is low,
  but a real-engine paper soak is the textbook gate and was compressed here.
- Entry floor < ~1.0 never binds (`favorite_threshold=0.85` already implies ≥1σ).
- PM track not swept (`min_safety_d=0` there) — separate follow-up.
