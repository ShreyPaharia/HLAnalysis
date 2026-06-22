# v31/theta HL binary — full theta-axes walk-forward tune

**Date:** 2026-06-22 · **Scope:** HL HIP-4 `priceBinary` v31/theta, the *other*
theta axes (`favorite_threshold`, `edge_buffer`, `vol_lookback_seconds`,
`vol_sampling_dt_seconds`, `tte_max_seconds`) — sweeping around the band that was
just deployed live earlier today (`min_safety_d=2.5` / `exit_safety_d=1.5`, see
[2026-06-22-v31-hl-safetyd-band.md](2026-06-22-v31-hl-safetyd-band.md)).
**Decision: a parameter change DOES beat live — recommend
`vol_lookback 900→1800` + `favorite_threshold 0.85→0.80` + `edge_buffer 0.02→0.01`,
gated on a paper-soak / fresh-data re-validation (NOT a deploy-now).** Analysis only.

## Motivation

The safety band was retuned and deployed live today on the n=42 corpus. Before the
next change we ran a *broader* walk-forward sweep on the full recorded corpus to see
whether raising the band moved the optimum on any of the other theta axes. (It is
the recurring "validate on recorded data" loop — and a deliberate check that we are
not serially over-fitting the same 6-week window.)

## Method

- **Corpus:** HL binary, recorded data **2026-05-10 → 2026-06-21, n=42** settled
  questions (the same corpus the band retune used). Walk-forward split via the
  source's own discovery: **early = 21 q** (2026-05-10→05-31), **late = 21 q**
  (2026-05-31→06-21). Ranked by **worst-half PnL** = `min(early, late)`.
- **Anchor (live):** the deployed binary block — `min_safety_d=2.5`,
  `exit_safety_d=1.5`, `vol_lookback=900`, `dt=5`, `favorite_threshold=0.85`,
  `edge_buffer=0.02`, `tte_max=43200`, `max_position_usd=500`, topup on.
  Reproduced to the cent: **full $720.80 / 329 trades / Sharpe 7.20 / hit 71.4% /
  maxDD $198.76** (monolithic `hl-bt run`), matching the band doc.
- **Sweep:** OFAT on all five axes + a `favorite_threshold × edge_buffer ×
  vol_lookback` (3×3×3) joint grid, safety band frozen at live. 43 cells.
- **Engine path / speed:** each (config, question) cell run via
  `scripts/perf/resumable_run.py` (warm-chunk driver, in-process bundle memo) so the
  expensive decode+settlement is shared across the 43 configs. Per-question PnL in
  discovery order reconstructs the **exact** portfolio metrics (`summarise_run`) the
  monolithic report emits — validated: driver prod_ref full = **$720.83** vs
  monolithic **$720.80** (±$0.03), and the offline early/late split reproduces the
  2026-06-22 separate-run halves to the cent ($543.16 / $177.64).
- **Tail stress:** `run_v31_hl_binary_axes_tail_stress.py` — the 2026-06-18
  loss-injection model (each entered-and-held favorite that won flips to a loss at
  its own implied rate `1−entry_price`; flip costs the empirical adverse-exit loss
  with the soft exit on). Validated by reproducing the buy-and-hold collapse.
- **Harness:** `experiments/scripts/run_v31_hl_binary_axes_sweep.py` (config defs),
  `aggregate_v31_hl_binary_axes.py` (walk-forward aggregation),
  `run_v31_hl_binary_axes_tail_stress.py`.

## Result — single-axis (OFAT) worst-half PnL vs live

| axis | values (worst-half $) | live | best |
|---|---|---|---|
| **vol_lookback_seconds** | 600 → **$248**, 900 → **$178** (live), 1800 → **$297**, 3600 → **$247** | $178 | **1800** |
| **favorite_threshold** | 0.80 → **$205**, 0.85 → **$178** (live), 0.90 → **$104** | $178 | **0.80** |
| **edge_buffer** | 0.0 → **$206**, 0.005 → **$208**, 0.01 → **$205**, 0.02 → **$178** (live), 0.03 → **$112** | $178 | **≤0.01** |
| **vol_sampling_dt_seconds** | 2 → **$223**, 5 → **$178** (live), 10 → **$183** | $178 | 2 (marginal) |
| **tte_max_seconds** | 8h → **$90**, 10h → **$165**, 12h → **$178** (live), 14h → **$144**, 16h → **$144** | $178 | **12h (live)** |

Read-out: **`tte_max=43200` (12h) is a clean interior optimum — keep it** (every other
value is strictly worse; consistent with the standing "TTE cap is load-bearing"
finding). **`dt=5` stays** — `dt=2` is marginally higher worst-half but it is the
bucket-lockstep value that binary history repeatedly flags as risky, and the gain is
small. The three real levers are **`vol_lookback`, `favorite_threshold`,
`edge_buffer`**, all of which the live config has placed sub-optimally.

Note `vol_lookback=1800` *alone* trades full PnL ($616 vs live $721) for a much
better worst-half ($297) and maxDD ($177); the full-PnL recovery comes from pairing
it with the lower `favorite_threshold`/`edge_buffer` (more entries).

## Result — top of the joint grid (sorted by worst-half PnL)

| config | full | early | late | **worst-half** | maxDD | Sharpe | trades | hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **LIVE** fav.85/eb.02/vlb900 | $720.8 | $543.2 | $177.7 | **$177.7** | $198.8 | 7.20 | 329 | 71% |
| **cand1 fav.80/eb.01/vlb1800** | **$754.6** | $384.9 | $369.7 | **$369.7** | **$177.2** | **8.46** | **212** | **79%** |
| cand2 fav.80/eb.00/vlb1800 | $736.7 | $370.5 | $366.3 | $366.3 | $177.2 | 8.18 | 241 | 81% |
| cand3 fav.85/eb.01/vlb1800 | $697.0 | $323.2 | $373.7 | $323.2 | $177.2 | 8.45 | 204 | 79% |

The live cell ranks **34th of 43** on worst-half — its $543/$178 halves are badly
*unbalanced* (the late half is weak). cand1 **balances** the halves ($385/$370),
which is exactly the robustness worst-half optimization rewards, and does it with
**fewer trades** (212 vs 329) and **higher hit** (79% vs 71%) — the longer
`vol_lookback` smooths σ, so entries are more selective.

## Tail stress (loss-injection, N=20 000, n=42)

| config | esd | realized | flip-loss | **EV** | **5th-pct** | **P(loss)** |
|---|---:|---:|---:|---:|---:|---:|
| **LIVE** fav.85/eb.02/vlb900 | 1.5 | $721 | $83 | $507 | $205 | 0.4% |
| **cand1 fav.80/eb.01/vlb1800** | 1.5 | $755 | $56 | **$579** | **$338** | **0.0%** |
| cand2 fav.80/eb.00/vlb1800 | 1.5 | $737 | $62 | $551 | $297 | 0.1% |
| cand3 fav.85/eb.01/vlb1800 | 1.5 | $697 | $52 | $554 | **$349** | 0.0% |
| BAH validation (esd0.0) | 0.0 | $1053 | $36 | $219 | **−$996** | **35.3%** |

The buy-and-hold validation cell reproduces the documented collapse (EV $219, 5th-pct
−$996, P(loss) 35.3%), so the candidate numbers are trustworthy. **All three
candidates dominate live on the tail too** — higher EV, higher 5th-pct, ~0% P(loss).
cand1's smaller per-flip loss ($56 vs $83) is the same selectivity mechanism: a
smoother σ ⇒ fewer marginal favorites entered ⇒ smaller adverse exits.

## Verdict

**A parameter change clearly beats live.** Recommended cell:
**`favorite_threshold 0.85→0.80`, `edge_buffer 0.02→0.01`, `vol_lookback 900→1800`**
(keep `tte_max=43200`, `dt=5`, and the live safety band msd2.5/esd1.5). It dominates
the live config on **every** axis measured:

| metric | live | cand1 | Δ |
|---|---:|---:|---:|
| worst-half PnL | $177.7 | $369.7 | **+$192** |
| full PnL | $720.8 | $754.6 | +$34 |
| maxDD | $198.8 | $177.2 | **−$22** |
| Sharpe | 7.20 | 8.46 | +1.26 |
| hit | 71% | 79% | +8pp |
| trades | 329 | 212 | −117 (less fill exposure) |
| stressed EV | $507 | $579 | +$72 |
| stressed 5th-pct | $205 | $338 | +$133 |
| stressed P(loss) | 0.4% | 0.0% | better |

**Highest-confidence single lever:** `vol_lookback 900→1800`. It is OFAT-robust
(every `vlb1800` grid cell tops the table), it lowers maxDD ($199→$177) regardless of
the other two knobs, and it is mechanistically sound (a 30-min σ window is steadier
than 15 min, raising entry selectivity). If only one change is made, make this one.
`favorite_threshold 0.80` and `edge_buffer 0.01` are secondary, consistent
improvements that recover (and exceed) the full-PnL the longer lookback gives up.

**This is a recommendation to *promote after re-validation*, not a deploy-now** — for
the same reason the band doc gated `esd=2.0`: re-confirm on ≥1 week of fresh recorded
data (and/or a real-engine paper soak) before pushing, since this is the **same n=42,
tail-deficient corpus** that today's band change already optimized against.

## Caveats (honest)

- **Single corpus, n=42, tail-deficient.** Favorites mostly resolved ITM, so realized
  maxDD/worst-half understate true tail risk and the absolute $ are inflated. The
  *relative* ranking is what we trust — and it is corroborated three ways (walk-forward
  worst-half, balanced early/late halves, and the independent loss-injection stress).
- **Serial over-fitting risk.** Today's band retune *and* this axes sweep both
  optimize the same 6-week window. The recommended move should be confirmed on
  out-of-corpus data before deploy; that is why the verdict is "promote after
  re-validation."
- **Three params move together.** `vol_lookback=1800` is the robust lever in
  isolation; `fav=0.80`/`eb=0.01` are smaller joint effects. A conservative deploy is
  `vol_lookback` alone first, then the other two after a soak.
- **Fill realism.** Walk-forward runs use the recorded HL book at the live cadence
  (`dt=5`); `--slippage-bps` is a no-op on recorded-book fills. The candidate trades
  **fewer** round-trips than live (212 vs 329), so it is *less* fill-model-exposed,
  not more. The resumable-driver per-question runs do not enforce cross-market
  inventory caps (accepted for HL binary; caps rarely bind on sequential 24h markets,
  and the prod_ref aggregate matched the cap-enforcing monolithic run to ±$0.03).
- **No live change made.** `config/strategy.yaml` untouched; nothing deployed.
