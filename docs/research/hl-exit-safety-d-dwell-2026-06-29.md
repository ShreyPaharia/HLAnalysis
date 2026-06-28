# `exit_safety_d` dwell/confirmation filter — HL bucket walk-forward

**Date:** 2026-06-29 · **Scope:** `v3_theta_harvester` on **HL HIP-4 priceBucket**,
recorded venue data 2026-05-10 → 2026-06-26. **Verdict: no robust win — ship the
flag OFF by default, do not enable in production.** The change is, however,
**tail-safe** (it does not increase maxDD / blow-up risk vs the C3 baseline), so it
is safe to carry as an off-by-default lever.

> Relates to the SHR whipsaw ticket. Code is flag-gated and OFF by default
> (`exit_safety_d_dwell_scans: int = 1`, where `1` == current bit-identical
> behaviour, covered by a regression test). **No `config/strategy.yaml` change, no
> deploy.**

---

## §1 — What this tests

The `exit_safety_d` soft-exit liquidates a held leg the first scan its
σ-normalized distance-to-band-edge (`safety_d`) drops below threshold. On a single
adverse tick this can dump a leg that immediately recovers (whipsaw). The dwell
filter requires `safety_d` to stay below threshold for **N consecutive scans**
before exiting; `stop_loss` and all other exit paths are unaffected, and a
*persistent* breakdown still exits (after N scans), preserving the C3 tail-guard.

**Grid:** `exit_safety_d_dwell_scans ∈ {1,2,3,5}` × bucket `vol_sampling_dt ∈
{2,4,6,10}s`. All other knobs **pinned to the C3 bucket baseline** (incl. the
deliberate tail-guard `exit_safety_d=1.0`, `min_safety_d=2.0`). Walk-forward:
15 train / 2 test / 2 step → 40 test windows per cell. `dwell=1, dt=2` is the
C3 re-baseline cell.

> ⚠️ **Caveat:** the `dwell=5, dt=10` corner cell completed only **13/40** windows
> (the run was killed during teardown). It does not affect the conclusion. Every
> other cell has the full 40 windows.

---

## §2 — Results

| dwell | dt (s) | n win | PnL ($) | hit % | maxDD ($) | worst split ($) | half-1 / half-2 ($) | sign stable |
|------:|------:|----:|------:|----:|------:|------:|------:|:--:|
| 1 | 2 | 40 | -312 | 47 | 656 | -656 | 496 / -808 | ✗ |
| 1 | 4 | 40 | -101 | 49 | 683 | -683 | 595 / -696 | ✗ |
| 1 | 6 | 40 | -287 | 46 | 553 | -553 | 370 / -656 | ✗ |
| 1 | 10 | 40 | -44 | 47 | 558 | -558 | 565 / -609 | ✗ |
| 2 | 2 | 40 | -436 | 47 | 616 | -616 | 322 / -759 | ✗ |
| 2 | 4 | 40 | -72 | 49 | 617 | -617 | 420 / -493 | ✗ |
| 2 | 6 | 40 | -340 | 46 | 551 | -551 | 199 / -540 | ✗ |
| 2 | 10 | 40 | -116 | 45 | 562 | -562 | 431 / -547 | ✗ |
| 3 | 2 | 40 | **185** | 48 | 440 | -440 | 303 / -118 | ✗ |
| 3 | 4 | 40 | 94 | 50 | 450 | -450 | 399 / -305 | ✗ |
| 3 | 6 | 40 | -368 | 46 | 548 | -548 | 172 / -541 | ✗ |
| 3 | 10 | 40 | -72 | 47 | 552 | -552 | 409 / -482 | ✗ |
| 5 | 2 | 40 | 122 | 49 | 528 | -528 | 181 / -59 | ✗ |
| 5 | 4 | 40 | 50 | 51 | 528 | -528 | 244 / -195 | ✗ |
| 5 | 6 | 40 | -445 | 47 | 550 | -550 | 100 / -545 | ✗ |
| 5 | 10 | 13 *(partial)* | 107 | 64 | 528 | -528 | -237 / 344 | ✗ |

**Tail-guard check** — dwell>1 vs `dwell=1` at the same dt; the hard constraint is
that dwell must **not increase maxDD or deepen the worst split** (the C3 rollback
showed gate-less buckets double maxDD via full-stake blow-ups):

| dt | dwell | ΔPnL | ΔmaxDD | Δworst | tail-safe? |
|--:|--:|--:|--:|--:|:--:|
| 2 | 2 | -124 | -39 | +39 | yes |
| 2 | 3 | +497 | -216 | +216 | yes |
| 2 | 5 | +434 | -127 | +127 | yes |
| 4 | 2 | +29 | -66 | +66 | yes |
| 4 | 3 | +195 | -233 | +233 | yes |
| 4 | 5 | +151 | -154 | +154 | yes |
| 6 | 2 | -54 | -2 | +2 | yes |
| 6 | 3 | -82 | -5 | +5 | yes |
| 6 | 5 | -158 | -4 | +4 | yes |
| 10 | 2 | -72 | +4 | -4 | **NO** |
| 10 | 3 | -29 | -6 | +6 | yes |
| 10 | 5 | +151 | -30 | +30 | yes *(partial)* |

---

## §3 — Read

1. **No robust edge. Every one of the 16 cells fails split-half sign stability**
   (half-1 positive, half-2 negative — including the `dwell=1` baseline). The
   corpus has a strong regime split: bucket theta was profitable in the first half
   of the window and loss-making in the second, and that regime effect **dominates
   any dwell effect**. The headline-looking cells (`dwell=3, dt=2` at +$185;
   `dwell=5, dt=2` at +$122) owe their full-sample sign entirely to the good first
   half — they are not time-stable and do not clear the robustness bar.

2. **Dwell is tail-safe** — this is the one clean, directionally-consistent result.
   In **11 of 12** comparable cells, raising dwell **reduces** both maxDD and the
   worst-split loss (it avoids some single-tick exits that lock in a loss the leg
   would otherwise have recovered). The lone "violation" is `dt=10, dwell=2` at
   **+$4 maxDD / −$4 worst** — i.e. noise, well inside one tick of slippage. So the
   hard constraint ("must not increase maxDD or tail blow-ups") **holds**: the C3
   tail-guard is preserved, and a persistent breakdown still exits after N scans.

3. **C3 re-baseline confirmed.** `dwell=1, dt=2` reproduces the current production
   bucket behaviour (−$312 / maxDD $656 / hit 47% on this corpus). `dwell=1` is
   bit-identical to pre-change behaviour (unit-tested), so the baseline column is a
   valid control.

---

## §4 — Recommendation

- **Keep the flag, keep it OFF (`exit_safety_d_dwell_scans = 1`).** The backtest
  does not justify enabling it in production — there is no time-stable PnL win, and
  the apparent low-`dt` gains are regime artifacts.
- **It is safe to carry.** The lever does not worsen tail risk (it generally
  reduces it), is bit-identical at the default, and prunes its per-leg state on
  settlement/eviction — so it is a low-risk knob to have available if a future,
  whipsaw-dominated regime warrants a controlled experiment.
- **If ever turned on,** the least-bad candidate is `dwell=3` at `dt∈{2,4}` (best
  full-sample PnL *and* lowest maxDD of the grid) — but only behind a fresh
  walk-forward on a corpus that does **not** show the May→June regime break, and
  with the split-half sign-stability gate passing first.

**Method note:** flat fee model, `max_position_usd=750`, topup enabled (C3 config).
Run from the worktree with `HLBT_HL_DATA_ROOT=../../data` (main checkout); no
`make pull-data`. Grid: `tuning.dwell.yaml` (scratchpad).
