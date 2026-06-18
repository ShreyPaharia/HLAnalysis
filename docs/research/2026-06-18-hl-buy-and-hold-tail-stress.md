# HL v31 buy-and-hold is tail-blind — revert binary + bucket to `msd2`

**Date:** 2026-06-18 · **Scope:** HL v31 `priceBinary` (live on buy-and-hold since
2026-06-17) and `priceBucket` (live on C3). **Decision: revert binary to `msd2`,
move bucket to `msd2`** (re-enable the σ-based mid-hold exit; keep the
`min_safety_d=2` entry gate). Driven by a loss-injection stress, not realized PnL.

## Configs

| | exit_safety_d | exit_edge_threshold | min_safety_d | meaning |
|--|--|--|--|--|
| baseline (old prod) | 1.0 | 0.0 | 0.0 | exits on, no entry gate |
| **msd2** | 1.0 | 0.0 | **2.0** | exits on + entry gate ← **adopted** |
| hold_msd2 (was deployed, binary) | **0.0** | **1.0** | 2.0 | buy-and-hold (exits off) |
| holdstop15/30/50 | 0.0 | 1.0 | 2.0 | buy-and-hold + hard bid stop @15/30/50% |

## Binary — realized vs stressed (full corpus, 40 q, live event cadence)

Realized PnL ranks `hold_msd2 > baseline > msd2`. But the headline that sold the
buy-and-hold — **maxDD $0** — is **tail-blind**: the 40-question corpus never
sampled a losing favorite. A **loss-injection stress** (each entered favorite
flips to a loss at its *own* implied rate, mean ~10%, ~3.5 flips/run; a flipped
favorite under buy-and-hold loses the **full stake** with no exit):

| config | realized | stressed **EV** | 5th-pct | **P(net loss)** |
|--------|---------:|----------------:|--------:|----------------:|
| baseline (exits cap each flip ~$50-141) | $1318 | **+$904** | +$485 | **0.1%** |
| **msd2** (exits cap each flip ~$50-141) | $968 | **+$608** | +$194 | **0.8%** |
| **hold_msd2** (no exit → full-stake loss) | $1924 | **−$3** | **−$1,760** | **47.9%** |

Buy-and-hold's realized +$1924 collapses to **EV ≈ $0**, a **~48% chance of a net
loss**, and a 5th-percentile of **−$1,760**. The σ-based mid-hold exit (which
`msd2` keeps) caps each flip at ~$50-141 instead of the full ~$528 — *that* is the
load-bearing protection, and it makes `msd2` the risk-adjusted winner.

## A hard bid stop is the *worst* of both (re-confirms the deploy memo)

Buy-and-hold + a hard `stop_loss_pct` (sell when the bid drops X% from entry) was
tested as a "best of both" (cap the tail without the soft-exit leakage):

| config | realized | stressed EV | P(loss) |
|--------|---------:|------------:|--------:|
| holdstop 15% | **−$2621** | −$2748 | 100% |
| holdstop 30% | −$423 | −$684 | 100% |
| holdstop 50% | $135 | −$292 | 79% |

**Catastrophic.** Binary favorites are volatile — a 0.88 favorite routinely dips
to 0.75 and recovers within the day — so a bid stop trips on that noise, dumps the
position, re-buys high after the cooldown, and repeats: a buy-high-sell-low churn
machine (−$2621 at 15%). The σ-based `exit_safety_d` is *smarter* — it fires on
genuine drift toward the adverse boundary (a model estimate), not on bid noise, so
it protects the tail **without** the churn. Protection hierarchy on binary:
**σ-soft-exit (good) ≫ buy-and-hold (tail-blind) ≫ hard bid stop (destructive)**.

## Buckets — buy-and-hold is even worse; `msd2` is the improvement

The bucket 1s 3-config sweep (42 q) already showed buy-and-hold's tail is **not**
hypothetical there — the corpus *contains* losing favorites:

| bucket config | PnL | Sharpe | maxDD |
|--|--:|--:|--:|
| baseline (C3) | $657 | 6.09 | $284 |
| **msd2** | $651 | **6.33** | **$273** |
| hold_msd2 | $668 | 2.74 | **$480** (two −$460 full-stake blowups) |

`msd2` ≈ baseline PnL with better Sharpe / lower maxDD, and on the churniest 5
markets `msd2` made **more** ($103 vs $81) while cutting entries 67%. So buckets
adopt `msd2` (add the entry gate, keep exits); buy-and-hold is rejected outright.

## Decision (config/strategy.yaml v31)

- **Binary** base theta: `exit_safety_d 0.0→1.0`, `exit_edge_threshold 1.0→0.0`,
  keep `min_safety_d 2.0` (revert the 2026-06-17 buy-and-hold to `msd2`).
- **Bucket** override: `min_safety_d 0.0→2.0`, keep `exit_safety_d 1.0` (C3 + entry gate).
- v31_pm / v1 untouched. Config-pinning tests updated. Not yet deployed (SSM step).

## Caveat on the stress

The loss-injection model assumes a flipped favorite loses (a) the full stake under
buy-and-hold and (b) ~30% of stake under the soft exit (the empirical worst exit
loss ≤$141 on a ~$500 stake). Real recovery varies. But the qualitative result —
buy-and-hold trades a certain ~$468 leakage saving for a probabilistic full-stake
tail that erases the edge ~48% of the time — is robust, and it is the *same*
tail-blindness that produced the −$460 bucket blowups in a corpus that did sample
the tail.
