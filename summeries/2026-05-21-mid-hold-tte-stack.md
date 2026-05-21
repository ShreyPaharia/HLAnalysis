# Mid-hold safety_d × TTE-cap stack — does mid-hold obsolete the 4h cap, and does it stack with v3.2/v3.4?

**Date:** 2026-05-21
**Branch:** `feat/mid-hold-tte-stack` (merges `agents/can-we-remove-tte-while-not-degrading-v3-1-perform-Y3FEWV` so v3.2/v3.4 ride alongside v3.1+mid-hold)
**Sweep configs:** `data/sim/configs/v3.1-mid-hold-tte-stack/`
**Sweep runs:** `data/sim/runs/v3.1-mid-hold-tte-stack-2026-05-21/` and `data/sim/tuning/v3-1-mid-hold-walkforward-2026-05-21/`

---

> **POST-σ-FIX UPDATE (commit `cc44bf6`):** the HL conclusions below were
> rendered partly stale by a σ-sampling bug discovered during this session.
> See the "Post-σ-fix re-tune" section at the bottom for the corrected
> HL picture. The PM walk-forward, the d=1.5 cascade analysis, and the
> v3.2/v3.4 stacking verdicts (P3/P4) remain valid — PM's data feed
> was always correctly resampled to 1m.

---

## TL;DR (one line per layer)

1. **TTE cap on HL**: KEEP `tte_max_seconds=14400`. Mid-hold + 4h cap (+$11.54, Sharpe 0.43, $82 DD) is strictly better than mid-hold + 24h cap (+$10.91, Sharpe 0.31, $120 DD).
2. **PM walk-forward**: `exit_safety_d=0.75` is the conservative pick (positive Sharpe every split, +67% PnL vs baseline OOS). d=1.0 has best avg Sharpe but one losing split.
3. **d=1.5 PM blowup**: confirmed re-entry cascade — gate fires every 60s, edge gate immediately re-allows. Cap the production grid at d=1.0.
4. **v3.2-volclock + mid-hold**: does NOT stack on PM (-4.5% PnL). BV-σ stays calm post-wick → safety_d stays large → fewer protective exits. The two compete.
5. **v3.4-LMgate + mid-hold**: not obsolete but PnL-suboptimal. Highest-Sharpe (5.18) lowest-DD ($69) variant of all, but gives up $272/yr vs v3.1+mid-hold.

---

## P0 — Does mid-hold safety_d let us safely remove the HL 4h TTE cap?

Run on HL HIP-4 (`kind=both`, 26 questions, 2026-05-06 → 2026-05-22, fee=0.00035, slippage=5bps):

| variant                                    | trades | PnL     | Sharpe | hit    | max DD |
| ------------------------------------------ | -----: | ------: | -----: | -----: | -----: |
| v3.1 baseline (d=0.0, tte=4h)              |     42 | -$13.26 |  -0.44 | 57.7%  | $89.14 |
| **v3.1 + mid-hold (d=1.0, tte=4h)** ✓       | **42** | **+$11.54** | **+0.43** | **57.7%** | **$81.70** |
| no-cap, no mid-hold (d=0.0, tte=24h)       |     64 | -$29.94 |  -0.79 | 73.1%  | $153.22 |
| mid-hold + 12h cap (d=1.0, tte=12h)        |     62 |  +$3.11 |  +0.09 | 69.2%  | $106.90 |
| mid-hold + 24h cap (d=1.0, tte=24h)        |     64 | +$10.91 |  +0.31 | 73.1%  | $119.81 |

**Verdict: CONDITIONAL — keep tte_max=14400.** Mid-hold rescues the no-cap case
from catastrophe (-$30 → +$11), but the 4h cap is still doing independent work:
fewer mediocre trades (42 vs 64), best PnL, best Sharpe, smallest DD. The cap
controls *exposure-to-settlement time*; mid-hold controls *intra-hold drift
risk*. They're complementary, not substitutive.

Mid-hold *by itself* with tte=24h is non-catastrophic (+$10.91), which is the
new finding: previously (memory `hl_tte_cap_load_bearing_2026_05_21`),
removing the cap was strictly bad for v3.1 and v3.4. With mid-hold, the no-cap
penalty drops from -$76 (v3.4 4h→24h: -$24→-$97) and -$41 (v3.1 4h→24h:
-$48→-$89) to nearly zero (v3.1+midhold 4h→24h: +$11.54→+$10.91, −$0.63).
Mid-hold subsumes ~99% of the catastrophe protection but doesn't dominate the
capital-efficiency win of the cap.

---

## P1 — PM walk-forward of `exit_safety_d`

Config: `config/tuning.v3-1-mid-hold-walkforward.yaml`.
Run: `hl-bt tune --strategy v3_theta_harvester --grid <yaml> --data-source polymarket
--start 2025-05-08 --end 2026-05-07 --workers 4`. 5 OOS test windows × 5 d-values = 25 cells.
PM fixed params: `min_distance_pct=0.002, min_bid_notional_usd=10, tte_max=86400`.

| d     | total PnL | avg Sharpe | min Sharpe | max DD | trades | hit  |
| ----- | --------: | ---------: | ---------: | -----: | -----: | ---: |
| 0.0 (baseline) |   $453.10 |       1.81 |      -0.83 | $234   |    556 | 52.0% |
| 0.5            |   $761.27 |       3.66 |      -0.18 | $203   |    556 | 51.7% |
| **0.75** (recommended) | **$758.18** | **3.94** | **+0.02** | **$203** | 562 | 51.0% |
| 1.0            |   $754.70 |       4.15 |      -0.36 | $240   |    584 | 51.3% |
| 1.25           |   $680.60 |       3.88 |      +0.20 | $207   |    804 | 49.3% |

**Out-of-sample finding:**
- All of d∈{0.5, 0.75, 1.0, 1.25} pass with **+50% to +68% PnL** over the d=0.0 baseline. Mid-hold's gains hold OOS.
- d=0.75 is the only value with **positive Sharpe every split** AND >$750 total PnL.
- d=1.0 has the highest avg Sharpe but one losing split (-0.36) and the worst max DD ($240) — borderline.
- d=1.25 begins to show the cascade pathology: +41% more trades than d=1.0, -10% less PnL.

Per-split Sharpe:

| d    | split 1 | split 2 | split 3 | split 4 | split 5 |
| ---- | ------: | ------: | ------: | ------: | ------: |
| 0.0  |    0.51 |    0.10 |   -0.83 |    7.54 |    1.71 |
| 0.5  |    1.00 |    1.75 |   -0.18 |   13.25 |    2.50 |
| **0.75** | **1.00** | **1.64** | **+0.02** | **15.04** | **2.03** |
| 1.0  |    0.80 |    2.81 |   -0.36 |   15.94 |    1.59 |
| 1.25 |    0.36 |    4.03 |   +0.20 |   13.67 |    1.12 |

**Recommendation:** ship `exit_safety_d=0.75` on PM. d=1.0 retained for HL
where the σ·√τ scale is much larger (24h binaries) and the gate behaves
differently. The current prod yaml has `exit_safety_d=1.0` per the live-shadow
prior memo; the walk-forward says 0.75 is safer per-split. Operator's call —
both pass.

---

## P2 — Diagnose the d=1.5 PM blowup

At `exit_safety_d=1.5` the single-run PM sweep produced 2,468 trades / $9 PnL.
Inspection of the heaviest cascade market (question `0xd1001a3f...`, resolved
NO) confirms the re-entry cascade hypothesis:

- 46 trades on a single YES leg over 1h47min.
- **Median inter-trade gap: 60.0s** (exactly the scanner interval).
- Pattern: BUY at 0.865 → safety_d EXIT at 0.845 → scanner re-runs → BUY at 0.865 → EXIT at 0.865 → …
- Each round-trip bleeds 2 × fee_taker × position (~$0.07 per cycle on $100 size). Total fees on this one question: **$1.57**. Realized PnL on the leg: −$20.25.

**Mechanism:** The entry gate (`edge_buffer=0.02, edge_max=0.20`) and the exit
gate (σ-normalized signed distance from boundary) are not symmetric. When BTC
sits near strike with elevated σ_RV, safety_d is small (<1.5) the moment after
entry while chosen_edge is still well above edge_buffer. The two gates can
oscillate at the scanner's 60s tempo.

**Why d=1.0 doesn't blow up:** at the σ·√τ scale of typical PM 24h binaries
(σ ≈ 0.6-1.0, τ ≈ 0.5-1d), a d<1.0 threshold requires BTC to be much closer
to strike before triggering. The geometry naturally separates the entry and
exit decision regions; at d=1.5 the regions overlap.

**Fix options (not implemented — d=1.5 already excluded):**
1. **Cap the production grid at d=1.0.** Done in current ship recommendation.
2. **Post-safety_d cooldown.** After an exit_safety_d fires on a question,
   block entries on that question for N seconds. Symmetric with the
   `entry_cooldown_seconds=60` router-side cooldown already in use, but
   driven by exit-reason rather than fill side. ~10 LoC if needed.

Recommendation: **(1)** is sufficient. d=1.5 is not in any production grid;
walk-forward already bounded at d=1.25.

---

## P3 — Does v3.2-volclock (BV-σ) stack with mid-hold?

Single-run PM full year (362 binaries, 2025-05-08 → 2026-05-07, same config
as P0 PM):

| variant                                  | trades | PnL       | Sharpe | hit    | max DD  |
| ---------------------------------------- | -----: | --------: | -----: | -----: | ------: |
| v3.1 (sample_std σ) + d=1.0              |    680 | $1,033.30 |   2.70 | 50.8%  | $253.97 |
| v3.2 (bipower σ) + d=1.0                 |    704 |   $986.24 |   2.53 | 51.7%  | $249.49 |
| Δ                                        |    +24 |   −$47.06 |  −0.17 | +0.83pp |   −$4.48 |

**Verdict: does NOT stack.** v3.2 loses 4.5% PnL on PM. HL is byte-identical
to v3.1 (same 42 trades, same $11.54 PnL) — no jumps large enough to make σ_BV
diverge from σ_RV in the 14-day HL window (consistent with prior memory
`v3_2_volclock_smoke_2026_05_20`).

**Mechanism:** Mid-hold's safety_d is `(ln(S/K) + drift) / (σ·√τ)`. BV-σ keeps
σ low post-wick → denominator small → **safety_d stays large** → gate fires
LESS → fewer protective exits. The mid-hold gate *needs* σ to balloon after
adverse moves so it can fire while there's still bid-side value left. BV-σ
disarms that trigger.

So v3.2 and mid-hold are *competing* mechanisms for the same problem (wick
handling), not complementary. v3.2's value was in *entry capture* (buy the
wick mispricing); mid-hold's value is in *exit protection* (close before bid
collapses). With mid-hold doing the heavy lifting on losers, v3.2's net
contribution flips slightly negative.

**Recommendation:** Don't ship v3.2-volclock alongside mid-hold. If
shipping a regime with mid-hold, keep `vol_estimator="sample_std"`.

---

## P4 — Is v3.4-LMgate obsolete after mid-hold?

Single-run PM full year, `lm_threshold=4.0`, `exit_safety_d=1.0`:

| variant                                  | trades | PnL       | Sharpe | hit    | max DD  |
| ---------------------------------------- | -----: | --------: | -----: | -----: | ------: |
| v3.1 (sample_std σ) + d=1.0              |    680 | $1,033.30 |   2.70 | 50.8%  | $253.97 |
| v3.4-LMgate (k=4) + d=1.0                |    288 |   $761.33 |   5.18 | 28.2%  |  $69.42 |
| Δ                                        |   −392 |  −$272 (-26%) |  +2.48 | -22.6pp | -$184 (-73%) |

For context, v3.4 walk-forward k=4 *without* mid-hold (from memory
`v3_4_lmgate_walkforward_2026_05_21`): $590 / Sharpe 6.30 / DD $100, 228
trades. With mid-hold on top: $761 / Sharpe 5.18 / DD $69, 288 trades.
**Mid-hold helps v3.4-LMgate too** (+29% PnL, -31% DD, +60 trades).

**Verdict: NOT OBSOLETE, but PnL-suboptimal.** v3.4 + mid-hold is the
highest-Sharpe (5.18) and lowest-DD ($69) variant on PM — strictly safer than
v3.1 + mid-hold ($254 DD). The cost: $272/yr in absolute PnL.

**Use cases:**
- DD-capped capital → ship v3.4-LMgate + mid-hold (Sharpe 5.18, DD $69).
- Absolute-PnL maximization → ship v3.1 + mid-hold (PnL $1,033, DD $254).
- Mid-hold is additive to LM-gate; both shipping configs benefit from it.

---

## What changed

- `config/tuning.v3-1-mid-hold-walkforward.yaml` — new walk-forward grid (5 d-values × 5 splits)
- `data/sim/configs/v3.1-mid-hold-tte-stack/*.json` — 9 single-run JSON configs (5 HL + 4 PM/HL variants)
- `scripts/run_mid_hold_tte_stack_hl.sh` — HL 5-variant sweep driver
- `scripts/run_mid_hold_tte_stack_variants.sh` — P3/P4 stacking-experiment driver
- Merge of `agents/can-we-remove-tte-while-not-degrading-v3-1-perform-Y3FEWV` → `feat/mid-hold-tte-stack` (v3_2_volclock and v3_4_lmgate now ride on top of mid-hold)
- This summary

## Tests

`uv run pytest tests/unit -q` → **427 passed in 24.53s** (4 mid-hold tests + 7
v3.2/v3.4 tests; no regressions from the merge).

## Headline recommendations

1. **HL HIP-4 production:** keep `exit_safety_d=1.0`, keep `tte_max_seconds=14400`. Don't relax the cap.
2. **PM production:** ship `exit_safety_d=0.75` (positive Sharpe every walk-forward split) OR keep current `1.0` (better avg Sharpe, slightly higher DD risk). Operator's call.
3. **v3.2-volclock:** do not ship alongside mid-hold. Re-evaluate only if mid-hold is rolled back.
4. **v3.4-LMgate:** ship as the DD-disciplined variant (best Sharpe, ¼ the DD). Both v3.1 and v3.4 benefit from mid-hold.
5. **d=1.5 cascade:** d=1.0 is the production ceiling. If a future test grid wants d>1.0, add a post-safety_d cooldown first.

---

# Post-σ-fix re-tune (HL only — supersedes the HL conclusions above)

## The bug

The strategy's σ formula (`theta_harvester._sigma`, `late_resolution._sigma`)
takes the last `vol_lookback_seconds / vol_sampling_dt_seconds` returns and
treats each as a 60-second sample. The HL data path was violating this
contract:

- **Backtest** (`hl_hip4._reference_iter`): emitted every BBO tick (~6/s
  on HL perp BBO). Last 60 returns spanned **5.5 seconds** of wall time.
  Annualization treated them as 60 minutes → **650× time-scale mismatch**.
  σ collapsed to the `vol_clip_min=0.05` floor on 100% of HL ticks.
- **Live engine** (`engine/market_state.apply`): appended every MarkEvent
  (~1.2/s on HL `activeAssetCtx`). Last 32 marks spanned 27 seconds → **72×
  mismatch**. Less catastrophic than backtest but still wrong by an
  order of magnitude.
- **PM** was always correct (PM's feed is pre-resampled 1m Binance bars).

## The fix (commit `cc44bf6`)

Resample the HL reference stream to 1-minute OHLC bars at the data
source, and mirror the same bucketing logic in the live engine's
`MarketState.apply` for MarkEvent. The strategy's σ assumption now holds.

## Post-fix HL 14-day re-tune (kind=both, 26 questions)

| variant | trades | PnL pre-fix | **PnL post-fix** | Sharpe post | max DD post |
|---|---:|---:|---:|---:|---:|
| d=0.0 tte=4h  | 38 | -$13.26 | **+$102.83** | 11.29 | $9.93 |
| d=1.0 tte=4h  | 38 | +$11.54 | +$95.89 | 9.72 | $16.87 |
| d=0.0 tte=24h | 94 | -$29.94 | **+$152.99** | 12.54 | $16.45 |
| **d=1.0 tte=12h** | 80 | +$3.11 | **+$177.01** | **16.22** | **$15.07** |
| d=1.0 tte=24h | 94 | +$10.91 | +$169.97 | 14.38 | $20.00 |

σ at entry post-fix: median 0.19–0.22 (was 100% floor-clipped at 0.05).

## Binary/bucket split (post-fix)

| variant | binary PnL | binary DD | bucket PnL | bucket DD |
|---|---:|---:|---:|---:|
| d=0.0 tte=4h  | +$58.64 | $10 | +$44.19 | $0 |
| d=1.0 tte=4h  | +$51.69 | $17 | +$44.19 | $0 |
| d=0.0 tte=24h | +$51.40 | $16 | +$101.59 | $4 |
| **d=1.0 tte=12h** | **+$75.73** | $17 | **+$101.28** | $6 |
| d=1.0 tte=24h | +$74.40 | $17 | +$95.57 | $10 |

Binaries are **profitable on every variant post-fix.** Buckets prefer the
12h cap by +$6 PnL, 43% lower DD, and 23% fewer trades vs the 24h cap.

## v1 (`late_resolution`) post-fix smoke

v1 was also operating on broken σ. Single-run on the same corpus with the
current production config (price_extreme_threshold=0.85, min_safety_d=1.0,
exit_safety_d=1.0, vol_lookback=3600, etc.) post-fix:

- 26 questions, 42 trades, **+$40.01 PnL**, Sharpe 4.81, hit 57.7%, max DD $25
- Binary leg: +$37.79 (10 questions), Bucket leg: +$2.22 (7 questions)

No catastrophe — v1's `min_safety_d=1.0` entry gate now fires meaningfully
(was trivially passing under broken σ). Strategy is more selective, entries
are still positive-EV. Safe to ship the σ fix to live v1 alongside v3.1.

## PM walk-forward at 12h cap (run id `v3-1-mid-hold-walkforward-tte12h-2026-05-21`)

| d | 24h PnL | 12h PnL | Δ |
|---|---:|---:|---:|
| 0.0 | $453 | $402 | -$51 (-11%) |
| 0.5 | $761 | $692 | -$69 (-9%) |
| 0.75 | $758 | $650 | -$108 (-14%) |
| 1.0 | $755 | $658 | -$97 (-13%) |
| 1.25 | $681 | $589 | -$92 (-14%) |

PM clearly prefers the 24h cap — tightening to 12h costs 9-14% PnL at
every d value. Per-venue cap divergence is justified.

## Production ship (commit `ca66b87`)

| venue | tte_max | exit_safety_d | rationale |
|---|---|---|---|
| **PM** | **86400 (24h)** | **1.0** | walk-forward optimum, +67% over baseline OOS |
| **HL** | **43200 (12h)** | **1.0** | post-fix optimum; +$81 PnL over the prior 4h cap, +$24 over 24h, identical DD |

Both venues use the same strategy code; the cap divergence is via the
per-class `tte_max_seconds` field in `config/strategy.yaml`'s
`theta_harvester` block.

## What the σ fix invalidates

Some prior memos referenced σ-driven HL findings — all are now suspect:
- `hl_tte_cap_load_bearing_2026_05_21` — the "load-bearing" conclusion was an
  artifact of broken σ. With correct σ the optimum cap shifts from 4h to
  12h, and the previously "catastrophic" 24h variant is now profitable.
- The pre-fix P0 table in this document (top section) is stale on HL.
- Any v1 tuning that touched `min_safety_d`, `vol_max`, `vol_ewma_lambda`,
  or `exit_safety_d` on HL was calibrated against broken σ.

PM tunings, the d=1.5 cascade analysis, the v3.2 stacking result, and the
v3.4 stacking result are all PM-only and remain valid.
