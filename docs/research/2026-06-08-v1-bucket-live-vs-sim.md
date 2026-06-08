# v1 (late_resolution) HL priceBucket — live-vs-sim fill-by-fill divergence

**Window:** 2026-05-31 → 2026-06-08 UTC
**Strategy:** `v1_late_resolution`, HL HIP-4 **priceBucket** track
**Author:** analysis run 2026-06-08/09
**Scope:** ANALYSIS ONLY — no strategy or engine behavior was changed. The
backtest config and the live-fill dump script are committed alongside this doc.

---

## TL;DR

- **Live** v1 bucket: **+$51.35**, 26 fills (19 buy / 7 sell), 7 trading days,
  one bucket coin per day, **every position held to the 06:00 UTC settlement and
  won (7/7)**. The 7 "sells" are settlement fills at price `1.0000`, not market
  exits — v1 buckets are pure buy-and-hold-to-resolution.
- **Sim** (gate disabled per SHR-78, default 60 s scan): **+$36.32**, 12 fills
  (6 buy / 6 settle), 6 entries, 75 % hit (4/6 question-days resolved `yes` with
  a position; one resolved `yes` but was never entered; one `unknown`).
- Within the **comparable window (05-31→06-07, excludes the 06-08 live trade
  that isn't in the recorded corpus yet)**: **live +$45.22 vs sim +$36.32**.
- **The single biggest knob is the sim's scanner cadence, not the strategy.**
  Re-running at `--scanner-interval-seconds 5` (vs the default 60) lifts the sim
  to **+$44.18**, essentially matching live's in-window +$45.22. Live evaluates
  entries on every tick; the default-60 s sim misses the first-seconds favorable
  fills that drive v1's edge.
- The residual divergence is **not** one cause. Per leg it splits into: (a) scan
  cadence + single-shot fills (SHR-79) making entry PnL hypersensitive to *when*
  the one fill lands; (b) a genuine `safety_d` gate divergence that keeps the sim
  out of one live winner (#1510) **even at 5 s**; (c) SHR-78's disabled volume
  gate letting the sim trade one day (#1670, 06-07) that live skipped.

---

## Inputs & method

- **Recorded corpus:** canonical `make pull-data` tree at
  `…/HLAnalysis/data/venue=hyperliquid` (topped up through 2026-06-07; the
  06-08 daily partition is not yet sealed to the archive). Reference =
  `--ref-source hl_perp` (HL perp mark; Binance not needed).
- **Sim config:** `config/backtest/v1_hl_bucket_since0531.json` — the current
  **live** v1 priceBucket params, with **`min_recent_volume_usd: 0.0`** to work
  around **SHR-78** (the runner hardcodes `recent_volume_usd=0.0`
  — `hlanalysis/backtest/runner/hftbt_runner.py:814` — so the live gate of 100
  would veto *every* entry → 0 trades). This means the sim evaluates a slightly
  **different entry set** than live; see the 06-07 leg below.
- **Sim command:** `hl-bt run --strategy v1_late_resolution --data-source
  hl_hip4 --kind bucket --ref-source hl_perp --fee-taker 0.0 --slippage-bps 0
  --start 2026-05-31 --end 2026-06-08`. Outputs in
  `data/sim/runs/v1_bucket_since0531/` (+ `…_scan5/` for the 5 s sensitivity).
- **Live fills:** read-only venue `user_fills` via AWS SSM
  (`tools/dump_v1_bucket_fills.py`; SSH to the box is blocked). Classified
  `priceBucket` by the engine's `coin_klass_map`. Fees observed = 0 on every HL
  HIP-4 fill, so **SHR-57 (HL fees) is a non-factor** here.

---

## Per-day net PnL: live vs sim

| Day    | Coin   | Live net | sim@60s | sim@5s | Match? |
| :----- | :----- | -------: | ------: | -----: | :----- |
| 05-31  | #1290  | +6.873   | +5.188  | +10.559 | cadence/partial-fill |
| 06-01  | #1340  | +10.231  | +10.238 | +10.238 | **match** |
| 06-02  | #1380  | +10.388  | +4.568  | +4.568  | gate/σ (persists at 5 s) |
| 06-03  | #1460  | +4.107   | +4.260  | +6.748  | cadence |
| 06-04  | #1510  | +8.281   | **0** (no entry) | **0** (no entry) | **safety_d gate** |
| 06-05  | —      | 0        | 0       | 0       | both skip |
| 06-06  | #1610  | +5.344   | +1.508  | +1.508  | gate/σ late entry |
| 06-07  | #1670  | **0** (no live trade) | +10.559 | +10.559 | **SHR-78 entry set** |
| 06-08  | #2230  | +6.123   | — (outside corpus) | — | corpus coverage |
| **Σ in-window (05-31→06-07)** | | **+45.22** | **+36.32** | **+44.18** | |
| **Σ total** | | **+51.35** | — | — | |

Sim question→coin mapping is exact (sim uses the same `#N` symbols as the
venue): Q24=#1290, Q25=#1340, Q26=#1380, Q27=#1460, Q28=#1510, Q30=#1610,
Q31=#1670.

---

## The dominant driver: scanner cadence (default 60 s)

The diagnostics rows are spaced **exactly 60 s apart** — the `hl-bt run` default
`--scanner-interval-seconds 60`, which the Step-3 command does not override.
The live engine evaluates entries event-driven / at the dt=5 cadence. v1's edge
is a near-resolution favorite that drifts from ~0.96 toward 1.0 over the session,
so **the first seconds after market open are the cheapest entry** — and a 60 s
scanner systematically misses them.

Re-running everything identically at `--scanner-interval-seconds 5`:

| | sim@60s | sim@5s | live (in-window) |
| :-- | --: | --: | --: |
| Total PnL | +$36.32 | **+$44.18** | +$45.22 |
| Entries | 6 | 6 | 6 (+1 on 06-08) |

**~$7.9 of the ~$8.9 in-window gap is pure scan cadence.** This is *not* an edge
difference — at 5 s the sim reproduces live's economics almost exactly. The
correct reading of the headline "sim +$36 vs live +$45" is: **the sim is not
mispricing v1's edge; the default 60 s decision cadence is starving it of the
early ticks live actually trades on.**

---

## Leg-by-leg on the divergent days

### 06-04 #1510 — live +$8.28, sim **never enters** (the biggest single gap)

Live entered at **00:00:03** (2 fills, 166 + 141 @ ~0.97) and held to settle:
+$8.281. The sim **never trades this market at either cadence.** Diagnostics for
Q28 show the in-window samples are wall-to-wall `safety_d_below_min` (with a few
`no_extreme_leg`), e.g.:

```
06-04 00:00:52  safety_d_below_min  yes_bid 0.940 yes_ask 0.977  ref 64292.5
06-04 00:02:52  safety_d_below_min  yes_bid 0.968 yes_ask 0.977  ref 64384.5
06-04 00:12:52  safety_d_below_min  yes_bid 0.970 yes_ask 0.977  ref 64279.5
```

The favorite leg *is* extreme (yes_ask 0.977 ≥ 0.85), but `safety_d =
distance_to_bucket_boundary / (σ·√τ)` stays **below the `min_safety_d=3.0`
floor for the entire window** — the price sits too close to the bucket boundary
relative to the Parkinson σ. **This persists at 5 s**, so it is *not* a cadence
artifact: it is a real gate/σ divergence. The most likely live-vs-sim mechanism
is **σ warm-up at market open** — at 00:00:03 the live engine's vol lookback is
nearly empty → σ underestimated → `safety_d` transiently clears → live enters;
by the sim's first evaluated sample the recorded book + lookback give a σ large
enough to keep `safety_d < 3.0`. (We cannot confirm live's σ at entry without
live diagnostics; flagged as the leading hypothesis.) Net: **the sim is *more*
conservative than live here**, and it costs the comparison $8.28.

### 06-02 #1380 — live +$10.39 @ 0.9665 (01:35), sim +$4.57 @ 0.985 (01:50/01:53)

Both enter mid-session (≈1.5 h after open), but the sim enters **~15–18 min
later, after the favorite has drifted 0.9665 → 0.985**. Earlier in-window samples
were `no_extreme_leg` / `safety_d_below_min`; the gate only clears at 01:50.
**This gap persists at 5 s**, so it is gate/σ timing, not cadence — the recorded
book + Parkinson-σ at dt let the sim through ~15 min later than live did. Cost:
≈$5.8. This is the cleanest candidate for an **SHR-80 σ-cadence** contribution.

### 06-06 #1610 — live +$5.34 @ 0.9825 (04:19), sim +$1.51 @ 0.995 (04:59)

Same shape as #1380, later in the session: the sim's gate clears ~40 min after
live's, at 0.995 (almost no edge left → +$1.51). Persists at 5 s → gate/σ, not
cadence. Cost ≈$3.8.

### 05-31 #1290 & 06-03 #1460 — cadence + single-shot fills (SHR-79)

These flip the *other* way once cadence is fixed:

- **#1290:** Live churned **11 partial buys** from 0.9661 → 0.9815 (sizes 7–131,
  classic top-up scaling) for a blended entry ≈0.977 → +$6.873. The sim has no
  partial/queue model (**SHR-79**, `no_partial_fill_exchange`): it takes **one
  full-size fill at the touch**. At 60 s that one fill lands late at 0.983 →
  +$5.19 (pessimistic); at 5 s it lands on the very first tick at 0.966 →
  +$10.56 (optimistic). The single-fill model makes entry PnL **hypersensitive
  to which tick the lone fill snaps to** — live's smear across the book is the
  faithful behavior, and the sim brackets it depending on cadence.
- **#1460:** sim@5 enters 00:06 @0.978 → +$6.75 vs live 00:13 @0.9865 → +$4.11.
  At 5 s the sim actually enters *earlier/cheaper* than live → optimistic.

### 06-07 #1670 — sim +$10.56, **live never traded** (SHR-78 entry set)

The sim enters #1670 at 00:00:01 @0.966 and wins +$10.56. **Live placed no v1
bucket trade on 06-07.** The one config knob that differs between the two is the
**volume gate**: live runs `min_recent_volume_usd=100`; the sim has it disabled
(SHR-78 workaround). 06-07 is the lone day the sim trades that live didn't, so
the disabled gate is the prime suspect (live's recent-volume read was likely
<100 at the entry moment, vetoing live while every other day cleared). We can't
confirm live's gate value without live diagnostics, but this is exactly the
**different entry set** the SHR-78 workaround warns about — and here it *adds*
$10.56 to the sim that live never realized.

---

## Hold-to-settlement behavior

Live is **pure buy-and-hold**: 19 buys / 7 sells, where all 7 sells are the
06:00 settlement fill at `1.0000`. There are **no intraday market exits** — with
`exit_safety_d=0.0` and `stop_loss_pct=null`, v1 buckets never exit early. **The
sim reproduces this exactly:** every entered question has exactly one `enter`
then a `settle` fill at 1.000; the only `hold` reasons after entry are
`have_position`. So unlike v31 (which churns mid-hold exits), **v1's
hold-to-settlement structure is faithfully reproduced** — the divergence is
entirely on the **entry** side (timing, price, and which legs get taken), not on
exits.

---

## Divergence attribution summary

| Root cause | Mechanism | Legs affected | Direction | ~$ impact (in-window) |
| :--------- | :-------- | :------------ | :-------- | :-------------------- |
| **Scanner cadence (60 s default)** | misses first-seconds cheap fills | #1290, #1460 (+#1380/#1610 partially) | sim pessimistic | **≈ +$7.9 recovered at 5 s** |
| **SHR-79** (no partial/queue fills) | single full-size fill at touch → PnL snaps to one tick | #1290 (live scaled 0.966→0.982) | either way | bounded by tick spread; ±$1–5/leg |
| **safety_d gate × σ warm-up** | sim σ keeps `safety_d<3` at open; live clears | **#1510 (never entered)** | sim conservative | **−$8.28** |
| **SHR-80** (σ cadence) | gate clears later in sim → entry at worse price | #1380 (−$5.8), #1610 (−$3.8) | sim pessimistic | **≈ −$9.6** |
| **SHR-78** (volume gate disabled) | sim takes a leg live's gate vetoed | #1670 (06-07) | sim optimistic | **+$10.56** |
| **Corpus coverage** | 06-08 not yet sealed to archive | #2230 | n/a | live-only +$6.12 |

These largely offset: in-window the sim's conservative misses (#1510, #1380,
#1610 ≈ −$17.9 vs live) are nearly cancelled by its optimistic extras (#1670
+$10.56, and the cadence/partial-fill swings on #1290/#1460). **The aggregate
"+$36 vs +$45" understates how close the two are once cadence is matched
(+$44 vs +$45).**

---

## Caveats (do not over-read)

1. **SHR-78 workaround changes the entry set.** With the volume gate at 0 the sim
   can take legs live would veto (#1670). Aggregate PnL is therefore *not* a
   clean apples-to-apples; read it per-leg.
2. **SHR-79 single-fill model** makes any single-leg entry PnL sensitive to the
   exact fill tick. The honest live behavior is the partial-fill smear; the sim
   brackets it.
3. **SHR-80 / cadence:** the headline number used the default 60 s scan. The 5 s
   sensitivity is the more faithful live comparison and should be the one quoted.
4. **No live diagnostics.** Live veto/σ values at the entry instant are not
   recorded, so the #1510 (σ warm-up) and #1670 (volume gate) mechanisms are the
   leading hypotheses, consistent with the data, not proven from live state.
5. **06-08 (#2230, +$6.12) is outside the recorded corpus** — `--end 2026-06-08`
   is end-exclusive and the 06-08 daily partition isn't archived yet.

---

## Artifacts

- Config: `config/backtest/v1_hl_bucket_since0531.json`
- Live-fill dump: `tools/dump_v1_bucket_fills.py` (read-only, SSM)
- Sim outputs: `data/sim/runs/v1_bucket_since0531/` (60 s),
  `data/sim/runs/v1_bucket_since0531_scan5/` (5 s sensitivity)
- Reproduce: see "Inputs & method" above; set
  `HLBT_HL_DATA_ROOT=…/HLAnalysis/data`.
