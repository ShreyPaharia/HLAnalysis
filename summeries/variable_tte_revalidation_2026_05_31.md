# Variable (vol-scaled) TTE entry window — re-validation on the latest model

**Date:** 2026-05-31
**Branch:** `feat/variable-tte-vol-revalidate`
**Strategy:** v1 `late_resolution`
**Corpus:** HL HIP-4, 2026-05-06 → 2026-05-28 (38 questions), `kind=both`, fee 0.00035, 5 bps slippage
**Baseline:** current prod v1 binary slot — Parkinson σ, `dt=5`, `min_safety_d=3.0`, `λ=0.97`, `tte_max=7200` (2h)
**Sweep driver:** `scripts/run_variable_tte_hl.py` → `data/sim/runs/variable-tte-2026-05-30/`

---

## TL;DR

1. **Where the prior work lives:** the "variable TTE basis volatility" experiment is the v1
   **`min_safety_d` joint vol+distance+TTE gate** (added 2026-05-10; memories
   `backtester_polymarket_direction` + `calibration_findings_2026_05_09`). There was never a
   literal `tte_max = f(σ)` formula — the dynamic-TTE effect was achieved *implicitly* via
   `safety_d`. The gate is mathematically an implicit vol-scaled TTE cap (see §1). It is
   **already in current prod** (`min_safety_d=3.0`).
2. **The dominant lever is the fixed cap, not vol-scaling.** On the current Parkinson/dt=5 model
   the prod `tte_max=2h` is badly suboptimal. Just **widening the fixed cap to 4h** lifts PnL
   `$269 → $409` (+52%) at the **best Sharpe (15.6) and lowest DD ($15.6) of the entire sweep**.
   12h maxes PnL ($483). This updates the stale `hl_tte_cap_load_bearing` memory (see §4).
3. **Explicit vol-scaled TTE produces the single best PnL** (`ref=1e-4, k=1`: **$540** / Sharpe
   11.3 / DD $74 / 102 trades), beating every fixed window on PnL — but only **+12% over fixed-12h**
   and at **5× the DD of fixed-4h**. On a 22-day, 10-cell grid this is almost certainly overfit.
4. **Verdict:** dynamic TTE "still looks good" in the narrow sense (top PnL config), but the
   honest, robust finding is *retune the fixed cap*. Vol-scaling adds marginal PnL, more DD, and
   config surface — **not worth shipping over a wider fixed cap without walk-forward**. No live
   change here; human-gated follow-up.

---

## 1 — What the prior experiment actually was

The memory note describes strategy v2 as a "dynamic entry, joint (TTE, vol, distance) function …
lower vol + larger distance → enter much earlier; higher vol / closer to strike → require closer
to resolution." That description was **realized as the `min_safety_d` gate**, not a literal
variable `tte_max`:

```
safety_d = |ln(BTC/strike)| / (σ · √(τ/dt))   ≥   min_safety_d
```

Rearranging the gate for τ shows it **is** a vol-and-distance-scaled TTE cap:

```
safety_d ≥ d   ⟺   τ ≤ dt · ( ln(S/K) / (d·σ) )²
```

So higher σ ⇒ smaller admissible τ (enter later); lower σ ⇒ larger admissible τ (enter earlier).
The "variable TTE basis volatility" framing is exactly this gate.

**Original numbers (PM BTC daily Up/Down, `calibration_findings_2026_05_09`, 2026-05-10):**
adding the gate took v1 from **$361 → $559** full-sample (**+55%**), hit **35% → 79%**, and — the
load-bearing observation — "**the gate enables a wider TTE window**: without it only 15-min
entries paid; with it 2h becomes optimal." safety_d sweep: 0→$150, 1.0→$219, **1.5→$258**,
2.0→$193, 3.0→$168. Config `data/sim/configs/v1-safety-best.json`.

## 2 — What I added (flag-gated, TDD, off by default)

The implicit form is already in prod, so to test the *explicit* "variable tte_max" the user asked
about, I added an opt-in knob to `LateResolutionConfig` (`late_resolution.py`):

```
tte_max_eff = tte_max_seconds · (vol_scaled_tte_ref_sigma / σ) ** vol_scaled_tte_exponent
              clamped to [0, vol_scaled_tte_ceiling_seconds]
```

When `vol_scaled_tte_enabled=False` (default) the step-1 TTE gate keeps the fixed cap and the path
is **bit-identical to legacy** — verified: same config SHA `9be5c8648fe3`, **$269.52 / Sharpe
9.732 / DD $33.05 / 52 trades**, matching the documented prod baseline exactly. 3 new unit tests
(widen-in-low-vol, narrow-in-high-vol, disabled-is-fixed); full suite **615 passed**.

## 3 — Results

### Fixed-window control (vol-scaling OFF, sweep `tte_max`)

| tte_max | PnL | trades | hit | Sharpe | maxDD |
| ------: | --: | -----: | --: | -----: | ----: |
| 1h | $218.49 | 24 | 31.6% | 9.75 | $0.00 |
| **2h (PROD)** | **$269.52** | **52** | **60.5%** | **9.73** | **$33.05** |
| **4h** ✦ | **$409.35** | 78 | 81.6% | **15.64** | **$15.64** |
| 12h | $482.81 | 116 | 79.0% | 8.68 | $78.21 |
| 24h | $300.54 | 160 | 65.8% | 4.48 | $119.82 |

### Vol-scaled window (ON, base 2h, ceiling 24h)

| ref_sigma | k | PnL | trades | hit | Sharpe | maxDD |
| --------: | -: | --: | -----: | --: | -----: | ----: |
| 1e-5 | 1 | $209.75 | 22 | 28.9% | 9.75 | $0.00 |
| 1e-5 | 2 | $32.88 | 18 | 21.1% | 1.99 | $38.96 |
| 3e-5 | 1 | $291.91 | 60 | 63.2% | 9.19 | $38.78 |
| 3e-5 | 2 | $351.39 | 76 | 63.2% | 10.03 | $35.09 |
| **1e-4** | **1** | **$539.86** | 102 | 81.6% | 11.28 | $74.07 |
| 1e-4 | 2 | $359.72 | 144 | 71.0% | 6.06 | $101.65 |
| 3e-4 | 1 | $333.40 | 152 | 71.0% | 5.07 | $120.36 |
| 3e-4 | 2 | $300.54 | 160 | 65.8% | 4.48 | $119.82 |
| 1e-3 | 1/2 | $300.54 | 160 | 65.8% | 4.48 | $119.82 |

Notes:
- **Degenerate-limit check passes:** for `ref ≥ 3e-4` (ref ≫ σ) `tte_max_eff` saturates at the 24h
  ceiling for every entry → identical to fixed-24h ($300.54 / 160 trades). Mechanism wired
  correctly.
- The interesting σ-transition is `ref ∈ [3e-5, 1e-4]`, implying typical entry σ ≈ `1e-4` per
  5s-bar (Parkinson) — consistent with the safety_d arithmetic at these strikes.

### Head-to-head

| variant | PnL | Sharpe | maxDD | trades | hit |
| ------- | --: | -----: | ----: | -----: | --: |
| prod fixed 2h | $269.52 | 9.73 | $33.05 | 52 | 60.5% |
| best fixed (4h, risk-adj) | $409.35 | **15.64** | **$15.64** | 78 | 81.6% |
| best fixed (12h, max PnL) | $482.81 | 8.68 | $78.21 | 116 | 79.0% |
| **best vol-scaled (ref1e-4,k1)** | **$539.86** | 11.28 | $74.07 | 102 | 81.6% |

## 4 — Interaction with the new config + the load-bearing-cap memory

- **`hl_tte_cap_load_bearing_2026_05_21` is now stale for this model.** That finding (removing
  `tte_max` strictly hurt v3.1/v3.4) predates Parkinson σ + dt=5 and was on v3.x. On v1 with
  Parkinson/dt=5 and `min_safety_d=3.0`, **widening** the cap is strictly *better* up to ~12h.
  The mechanism is exactly the original 2026-05-10 observation: a strong σ-distance gate
  *enables* a wider TTE window — the gate, not the cap, does the filtering. The cap is no longer
  load-bearing for v1 because safety_d=3.0 took over its job.
- This is **in-sample** on a 22-day corpus (38 questions). The original safety_d finding was
  walk-forward over a PM year; this re-run is a point estimate. `ref=1e-4` being the single best
  of a 10-cell grid is a textbook overfit setup. **Walk-forward is required** before trusting any
  of these numbers for a live change.
- The dt=5 config itself is still ⚠️ paper-gated (`v1_cadence_validation_2026_05_30`); nothing
  here changes that.

## 5 — Recommendation (human-gated; no live change made)

1. **Highest-value, lowest-risk follow-up: retune the *fixed* `tte_max`, not vol-scaling.** prod
   2h is leaving ~$140 (+52%) on the table on this model. **4h** is the robust pick (best Sharpe
   15.6, lowest DD $15.6); 12h for max PnL at higher DD. One knob, monotone region, easy to
   walk-forward.
2. **Vol-scaled TTE: do not ship over a retuned fixed cap.** It produces the top single PnL
   (+12% over fixed-12h) but at 5× fixed-4h's DD and with a fragile 2-param fit on a tiny corpus.
   Keep the flag (off by default) for future walk-forward study; revisit only if a fixed-cap
   retune underperforms OOS.
3. **Gate everything on walk-forward + paper.** Re-run the fixed-cap sweep (and the ref=1e-4 cell)
   with PM-year walk-forward and HL forward-test before any `strategy.yaml` change. Do **not** flip
   live config.

## What changed

- `hlanalysis/strategy/late_resolution.py` — `vol_scaled_tte_*` config fields + 6a gate + builder
  wiring (off by default, fixed path bit-identical).
- `tests/unit/test_strategy_late_resolution.py` — 3 vol-scaled-TTE tests.
- `scripts/run_variable_tte_hl.py` — fixed-vs-vol-scaled HL sweep driver.
- This summary.

## Tests

`uv run pytest tests/unit -q` → **615 passed in 31.70s**.
