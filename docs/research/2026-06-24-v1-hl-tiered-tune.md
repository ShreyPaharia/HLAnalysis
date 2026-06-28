# v1 (late_resolution) HL BTC — tiered tune + event-cadence validation

**Date:** 2026-06-24 (tune) → 2026-06-26 (validation + deploy) · **Scope:**
`v1_late_resolution` on **HL HIP-4 BTC**, **binary** and **bucket** classes, tuned
**independently**. Method: tiered coarse→fine walk-forward tune, then a
**live event-cadence (scan 0.2/2.0) re-run** that head-to-heads v1 against the live
**v31** (`theta_harvester`), then a **loss-injection tail stress** on the
exit-gate-less bucket.

> ⚠️ **Read §1–§3 first.** The tiered grid in §4 ran at the default **fixed 60 s**
> scanner cadence (σ-sampling was live dt=5, but the *decision* cadence was 60 s, not
> the live event-driven 0.2/2.0). Its **recommendations did not survive** the
> event-cadence re-run: the binary "fix" is dominated by v31, and the bucket
> `vol_ewma_lambda 0.97→0.85` rec is **toxic** at event cadence. §4 is retained as the
> method/historical record only. The authoritative outcome is **§1**.

---

## §1 — What shipped (2026-06-26, DEPLOYED LIVE, `config/strategy.yaml`)

| slot / class | before | after | rationale |
|---|---|---|---|
| **v1 priceBinary** | live, worst-half **−$108** | **TAKEN DOWN** (allowlist entry removed) | v31 dominates v1 on binary at event cadence (§2); no reason to run both |
| **v1 priceBucket** | tte 6h, $300 | **tte 8h (28800 s), $500**, `exit_safety_d=0`, `vol_ewma_lambda=0.97` (kept), `vol_lookback=3600` (kept) | bucket edge is real and competitive (§2); 8h captures more of the late-resolution window; **λ0.85 rejected** (toxic at event cadence) |
| **v31 priceBinary + priceBucket** | $500 | **$750** | capital scale-up; v31 now owns HL binary |
| globals (v1 pool) | inv $1000 / loss-cap $100 | **inv $1500 / loss-cap $150** | ≈3 concurrent $500 bucket legs; loss-cap is the **only** tail brake on the exit-gate-less bucket (§3) |
| globals (v31 pool) | — | inv $1650 / loss-cap $150 | scaled with $750 per-position |

HYPE and PM slots untouched. **No mid-hold/stop exit was added to v1 bucket** — §3
shows every exit mechanism doom-loops across the wide bucket spread; `daily_loss_cap`
is the deliberate substitute.

---

## §2 — Event-cadence head-to-head (scan 0.5/2.0, the authoritative comparison)

Re-run via `experiments/scripts/run_v1_v31_event_compare.py`. HL BTC recorded
2026-05-06..2026-06-24. Walk-forward worst-half = min(early, late) over the
median-date split. Winners re-confirmed at the live 0.2/2.0 cadence
(`data/sim/runs/v1-hl-validate-0p2-2026-06-24`).

### Binary (n=45, early=22 / late=23) — sorted by worst-half

| config | full | Sharpe | maxDD | early | late | **worst-half** | trd | hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **v31_live** | **1060** | **11.0** | $106 | 496 | 564 | **$496** | 541 | 78% |
| v1_tuned (vlb5400/esd0) | 223 | 2.3 | $251 | 52 | 171 | $52 | 102 | 56% |
| v1_tuned_cons (esd1) | 283 | 5.9 | $95 | 272 | 12 | $12 | 174 | 49% |
| v1_live | −90 | −0.8 | $300 | −108 | 18 | **−$108** | 91 | 47% |

**v31 dominates v1 on every metric.** The best event-cadence v1 binary
(worst-half $52) is an order of magnitude below v31 ($496) at 5× the Sharpe gap.
→ **v1 binary taken down; v31 owns HL binary.**

### Bucket (n=47, early=23 / late=24) — sorted by worst-half

| config | full | Sharpe | maxDD | early | late | **worst-half** | trd | hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **v1_live** (tte6h) | 698 | **19.6** | **$0** | 462 | 236 | **$236** | 459 | 91% |
| v31_live | 787 | 9.6 | $153 | 595 | 192 | $192 | 439 | 68% |
| v1_tuned (tte8h/λ0.85) | 634 | 6.3 | $247 | 561 | 73 | $73 | 178 | 94% |

The **v1 bucket edge is real** — it beats v31 on worst-half ($236 vs $192) and
Sharpe (19.6 vs 9.6) at maxDD $0, so it stays live. **Note the λ0.85 trap:** the
`v1_tuned` cell carried the 60 s rec's `vol_ewma_lambda=0.85`, which *collapses*
worst-half to $73 and blows maxDD to $247 at event cadence. The shipped bucket
therefore keeps **λ0.97** and changes only `tte_max 6h→8h` (more late-window
capture) + sizing.

---

## §3 — Bucket tail stress (loss-injection Monte-Carlo)

`experiments/scripts/run_v1_bucket_tail_stress.py`, N=20000. v1 bucket has **no
active exit gate** (`stop_loss=null`, `exit_safety_d=0`), so its realized **maxDD $0
is tail-blind** — the corpus simply had no favorite lose at settlement. Each
won-and-held favorite is flipped to a loss at its implied rate (1 − avg entry); with
no exit, a flip loses the **full stake**.

| config | esd | realized | ~flips | EV | 5th-pct | **P(net loss)** |
|---|---:|---:|---:|---:|---:|---:|
| v1 bucket tte8h (shipped) | 0.0 | $906 | 2.89 | **−$4** | −$950 | **51.3%** |
| v1 bucket tte6h (prior live) | 0.0 | $699 | 2.22 | −$11 | −$870 | 39.3% |
| v31 bucket (esd1.0) | 1.0 | $779 | 1.98 | **+$582** | +$337 | **0.0%** |

The exit-gate-less v1 bucket is a **coin-flip on any month with one adverse
settlement** (EV≈0, P(loss)≈51%), whereas v31's soft exit caps the tail (P(loss) 0%).
Adding an exit to v1 bucket was tested and rejected: σ-band and `stop_loss` both
**doom-loop** across the wide HIP-4 bucket spread (re-buying into the same favorite),
and `exit_spread_hold` — the only mechanism that avoids it — *is* v31's design.
Conclusion: **the v1 bucket edge is structurally unhedgeable; `daily_loss_cap_usd=150`
is the only tail brake**, which is why the global cap was scaled with the sizing.

---

## §4 — Tiered tune at 60 s cadence (HISTORICAL — method record, recommendations SUPERSEDED)

> The grid below ran the inner `hl-bt run` cells at the **default fixed 60 s**
> cadence. Its recommendations (binary vlb5400/λ0.85/esd0; bucket tte8h/**λ0.85**)
> are **not** what shipped — see §1/§2. Kept for the tiered method + as the candidate
> generator that fed §2.

**Method.** Corpus HL BTC 2026-05-06..06-24, 45 binary + 47 bucket settled questions.
Walk-forward median-date split, ranked by worst-half PnL. Every (config × question)
cell run through the warm-chunk resumable driver
(`scripts/perf/resumable_run.py` via `experiments/scripts/run_v1_hl_tiered_tune.py`)
— in-process bundle memo shares each question's decode across all configs. Held
fixed: `vol_sampling_dt_seconds=5` (v1+v31 share the BTC feed and move cadence in
lockstep), `vol_estimator=parkinson`.

**Tier-1 coarse OFAT directions (60 s worst-half):**
- *Binary* — `vol_lookback` dominant, sharp interior peak at **5400**; `exit_safety_d=0`,
  `λ=0.85`, `min_safety_d=2.0` secondary; `tte_max`/`price_extreme` don't rescue.
- *Bucket* — only `tte_max` helps (peak **8h**, catastrophic at 9h+); `exit_safety_d`
  must stay **0** (mid-hold exit poison on buckets); `vol_lookback` **inert** (opposite
  of binary).

**Tier-2 fine grid (60 s) headline cells:**
- Binary `vlb5400/esd0/λ0.85/msd3.0`: full $366 / worst-half $113 / Sharpe 12.8 /
  maxDD $0 — *looked* like a fix, but §2 shows it's dominated by v31 at event cadence.
- Bucket `tte8h/λ0.85`: full $838 / worst-half $280 / maxDD $0 at 60 s — but λ0.85
  **inverts to a loss** at event cadence (§2). The robust change was tte 6h→8h alone.

**Why the 60 s grid mis-ranked:** 60 s under-samples the exit gates and the
event-driven entry timing, so it over-credited `exit_safety_d=0` on binary and the
λ0.85 σ-smoothing on bucket. Only the event-cadence re-run (§2) — at the actual live
0.5/2.0, confirmed at 0.2/2.0 — reflects what the engine does.

---

## §5 — Caveats

1. **Single corpus, n=45/47, ~7 weeks.** Worst-half walk-forward mitigates but does
   not eliminate over-fit. Event-cadence validation closed the biggest gap (cadence
   mismatch); fresh-data / paper-soak confirmation still warranted for any *further*
   change.
2. **v1 bucket tail is real and unhedged** (§3). The position is held to settlement;
   the daily-loss cap, not a per-position exit, bounds the tail. Monitor live
   bucket settlements for the first adverse loser.
3. **Driver per-question runs don't enforce cross-market inventory caps.** Buckets can
   hold concurrent legs, so bucket $ may be slightly optimistic; the live
   `max_total_inventory_usd` enforces the real cap.
4. **Class-specific directions confirmed:** binary's lever (vol_lookback) is inert on
   buckets and vice-versa — they stay tuned independently. And v1↔v31 do **not**
   substitute symmetrically: v31 wins binary, v1 wins/ties bucket on worst-half.
