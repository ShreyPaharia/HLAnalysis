# v31/theta HL HIP-4 **priceBINARY** — independent walk-forward tune

**Date:** 2026-06-05  **Engine:** main @ `74fe66f` (sigma-resample fix; event-array
cache default-ON w/ `_bundle_config_sig` safeguards — bit-exact, coverage-tested)
**Corpus:** 2026-05-06 → 2026-06-04, `--kind binary`, n=25 questions
**Selection:** walk-forward early(05-06..05-20) / late(05-20..06-04); pick by best
**worst-half** PnL (overfit-robust on n=25). DD reported throughout.
**Harness:** `scripts/run_v31_hl_binary_joint.py` (mirrors the bucket-tune scripts).

## 0. Baseline reproduction — PASS
prod v31/theta config, `--kind binary`, full corpus reproduces to the cent:

| metric | got | expect |
|---|---|---|
| questions | 25 | 25 |
| trades | 70 | 70 |
| total PnL | $830.24 | $830.24 |
| Sharpe | 12.345 | 12.345 |
| hit | 80.00% | 80.00% |
| maxDD | $124.57 | $124.57 |

## 1. Headline finding — `vol_sampling_dt` is the dominant axis, and binary wants the OPPOSITE of buckets

`vol_sampling_dt` is a **shared feed-level lockstep** param (v1 + bucket + binary
share the HL BTC reference feed). Buckets *love* dt=2. **Binary is hurt by it on
the walk-forward worst-half**, even though dt=2 raises full-sample PnL:

| config (everything else = prod) | full | early | late | **worst-half** | maxDD | Sharpe |
|---|---|---|---|---|---|---|
| prod (dt=5) | $830.24 | $364.10 | $466.14 | **$364.10** | $124.57 | 12.35 |
| prod knobs **but dt=2** | $872.33 | $250.43 | $621.90 | **$250.43** | $86.20 | 13.39 |

dt=2 trades a fatter late half for a **collapsed early half** → worst-half drops
**−$114**. Full PnL and DD look better, but the DD-disciplined / walk-forward
criterion rejects it. **Every cell in the dt=2 joint grid inherits this damage**
(see §3): no dt=2 cell beats prod's $364 worst-half.

## 2. Binary's native optimum is at **dt=5 + vol_lookback 3600→900**

At dt=5, a single-knob change is a clean three-way (PnL + worst-half + DD) win:

| config @ dt=5 | full | early | late | **worst-half** | maxDD | Sharpe | trades | hit |
|---|---|---|---|---|---|---|---|---|
| prod | $830.24 | $364.10 | $466.14 | $364.10 | $124.57 | 12.35 | 70 | 80% |
| **+ vol_lookback=900** | **$863.51** | $385.52 | $477.99 | **$385.52** | **$111.51** | 12.94 | 126 | 76% |
| + edge_buffer=0.0 | $881.42 | $370.22 | $511.19 | $370.22 | $124.57 | 13.38 | 102 | 88% |
| + vol_lookback=600 | $832.52 | $316.60 | $515.92 | $316.60 | $136.30 | 12.02 | 156 | 80% |

- **vol_lookback=900** is the worst-half optimum *and* lowers maxDD ($111.51 < $124.57).
  900 is interior: 600 is worse (worst $316), 1800/2700/3600 all lower (3600=prod).
- **edge_buffer=0.0** is the full-PnL leader (+$51) but a smaller worst-half gain.
- **They do NOT stack** — the round-2 "two free wins" are mutually destructive:

| config @ dt=5 | full | early | late | **worst-half** | maxDD |
|---|---|---|---|---|---|
| eb=0.0 + vlb=900 (center_dt5) | $796.17 | **$207.02** | $589.14 | **$207.02** | $166.13 |

Combining them craters the early half to $207 and inflates DD to $166. Keep
**eb at prod 0.02** when vlb=900.

## 3. Joint grid (fav × edge_buffer × vol_lookback) at the FORCED dt=2 — for completeness

Per the brief, the joint grid was run at the shared-lockstep dt=2 (tte=12h,
exit_safety_d=1.0 fixed). Best cells by worst-half:

| cell (dt=2) | full | **worst-half** | maxDD | Sharpe | trades | hit |
|---|---|---|---|---|---|---|
| fav0.8 / eb0.01 / vlb900 | $941.99 | $281.27 | $168.78 | 12.29 | 204 | 84% |
| fav0.8 / eb0.01 / vlb600 | $821.98 | $239.00 | $161.76 | 11.91 | 220 | 80% |
| fav0.85 / eb0.0 / vlb1800 | $832.13 | $230.69 | **$94.19** | 12.55 | 102 | 76% |
| fav0.85 / eb0.005 / vlb1800 | $806.59 | $229.63 | $99.36 | 12.15 | 100 | 76% |
| *prod-knobs @ dt2 (fixedknobs_only)* | $872.33 | **$250.43** | $86.20 | 13.39 | 70 | 76% |

Key reads at dt=2:
- The **headline-PnL** cell (fav0.8/eb0.01/vlb900, full $942) is **not DD-disciplined**:
  worst-half $281, maxDD $169 (2× prod's DD).
- **No retune beats prod-knobs-at-dt2 on worst-half + DD jointly** — prod-at-dt2
  is worst $250 / DD $86; the only cells with higher worst-half also carry ~2× DD.
- So *if binary is forced to dt=2, the DD-disciplined move is to keep prod theta*
  (a binary-specific retune buys headline PnL only by spending drawdown).

## 4. Recommendation

The decision hinges on whether binary can run **dt=5** independently of the
buckets/v1 feed (they currently share one HL BTC reference feed → one dt).

- **PRIMARY (binary at dt=5): prod theta + `vol_lookback_seconds: 3600 → 900`.**
  Only divergent knob from prod. full **$863.51** (+$33), worst-half **$385.52**
  (+$21), maxDD **$111.51** (−$13), Sharpe 12.94, 126 trades, hit 76%.
  Keep `edge_buffer=0.02` (eb=0.0 does not stack), `favorite_threshold=0.85`,
  `tte_max=43200` (12h), `exit_safety_d=1.0`, `vol_sampling_dt=5`.

- **FALLBACK (binary forced to dt=2 by the shared feed): keep full prod theta.**
  No dt=2 retune is DD-justified — every worst-half improvement over prod-at-dt2
  costs ~2× drawdown. Accept the dt=2 worst-half hit as the price of the lockstep,
  or **decouple the binary reference cadence to dt=5** to unlock the PRIMARY pick.

**Operator decision required:** can the HL BTC reference feed serve binary at dt=5
while buckets/v1 run dt=2? If yes → PRIMARY. If no → FALLBACK (prod) + flag the
worst-half cost of the lockstep on binary.

## 5. Binary vs prod vs bucket — the divergences

| knob | prod | **binary pick** | bucket pick (memory) | note |
|---|---|---|---|---|
| `vol_sampling_dt` | 5 | **5** | **2** | shared lockstep; binary worst-half −$114 at dt=2, bucket loves dt=2 |
| `vol_lookback_seconds` | 3600 | **900** (@dt5) | 2700 | binary 900 interior @dt5; does **not** stack with eb=0 |
| `edge_buffer` | 0.02 | **0.02** (keep) | 0.005 | eb=0 best on *full* but non-stacking + worst-half-neutral |
| `favorite_threshold` | 0.85 | **0.85** | 0.80 | binary 0.90 craters (full $437); 0.80 weak worst-half |
| `tte_max_seconds` | 43200 | **43200** (12h) | 28800 (8h) | binary 12h interior (14h/16h worse, <12h far worse) |
| `exit_safety_d` | 1.0 | **1.0** | 0.0 | binary esd=0 → early collapses to $143; bucket wants 0 |

Binary and buckets diverge on **5 of 6** load-bearing theta knobs — confirming the
per-class plumbing thesis. Binary is the *conservative, long-TTE, high-favorite,
mid-hold-exit-ON, dt=5* class; buckets are *loose-favorite, short-TTE, no-mid-hold,
dt=2*. The bucket config applied to binary is destructive (e.g. esd=0 → worst $143,
fav=0.80 + dt=2 → worst-half regression).

## 6. Caveats
- **n=25** — worst-half on a 12/13-split is the guard, but single-market swings move
  it; treat ±$30 worst-half differences as noise.
- **dt is plumbing, not a free knob** — the PRIMARY pick assumes binary can hold
  dt=5; verify feed topology before acting (memory: `engine_cadence_port`,
  `v1_cadence_validation`).
- **No live change / no deploy** — analysis + recommendation only.
- Safety gates (min_bid_notional, stop_loss, stale_data_halt) untouched per
  `feedback_keep_safety_gates`.
