# v1 (late_resolution) sub-minute reference-sampling validation — 2026-05-30

**Branch:** `feat/v1-cadence-validation`
**Strategy:** v1 `late_resolution` (near-resolution favorite arb on binaries)
**Corpora:** HL HIP-4 BTC binaries+buckets (2026-05-06 → 2026-05-28, 38 questions)
and PM BTC Up/Down via Binance-perp BBO ticks (2026-05-06 → 2026-05-29, ~18).
**Companion to:** `v37_hl_1s_sampling_2026_05_28.md` (the theta/v31 dt sweep that
picked dt=5) and `engine_cadence_port_2026_05_29.md` (the shared-feed constraint).

## TL;DR — decision: **LOCKSTEP IS VIABLE *if* v1 also switches to Parkinson σ; the estimator swap is the real prize (+$48 at the current dt=60)**

> **This supersedes an earlier "keep v1 at 60, decouple v31" conclusion.** That
> conclusion was reached before sweeping v1's σ-estimator and EWMA-λ, which were
> frozen at the prod values (`stdev`, `λ=0.85`). Those two are uniquely bad on
> 5s bars and were the entire source of the apparent dt=5 damage. Once swept,
> the picture inverts. The history of how we got here is kept below in full.

The robust, **cadence-independent** finding: switching v1's vol estimator from
close-to-close `stdev` to **Parkinson** (range-based, H/L) lifts HL PnL from
**$212 → $260 at the existing dt=60** (best cell `msd1.0/parkinson/λ0.85`,
DD $72 ≈ baseline). 5s close-to-close returns are dominated by bid-ask bounce,
which Parkinson's range estimator shrugs off; this is why the prod `stdev` config
cratered at dt=5 ($88) but Parkinson does not.

On **cadence**, once on Parkinson, the two cadences are a near-tie on PnL with a
risk-adjusted tilt to dt=5:

| best cell per cadence            | PnL     | hit   | Sharpe | maxDD  |
|----------------------------------|---------|-------|--------|--------|
| dt=60  `msd1.0/parkinson/λ0.85`  | $259.74 | 60.5% | 7.58   | $72.15 |
| dt=5   `msd3.0/parkinson/λ0.97`  | $269.52 | 60.5% | 9.73   | $33.05 |
| — prod today `stdev/λ0.85` dt=60 | $212.01 | 57.9% | 6.42   | $75.46 |

dt=5 wins +$10 PnL (noise on 38 questions) but roughly **halves drawdown** and
lifts Sharpe. There is a real cadence×gate interaction: dt=60 prefers a looser
entry gate (`min_safety_d=1.0`), dt=5 a tighter one (`3.0`) — denser data
supports more selectivity. Both optima land ~$260–270, ~25% above today's $212.

**Decision (revised):**
1. **v1 is NOT hurt by dt=5** — so the v31 dt=5 flip **can be lockstep, no feed
   decoupling needed**, *provided v1 is also moved to Parkinson σ* (with `stdev`
   it still craters at dt=5). The decoupling work scoped earlier is unnecessary.
2. **Adopt Parkinson for v1 regardless of the v31 question** — it is a standalone
   +$48 win at the current dt=60 and is the single largest lever found.
3. Both the estimator swap and any cadence flip are **separate, human-gated
   config changes** to validate on a longer, valid-strike corpus first (the
   38-question HL window makes the +$10 dt=5-over-dt=60 edge noise-level; the
   +$48 Parkinson effect is the only robust-magnitude result).

> **Implementation status (2026-05-30):** both follow-ups are now implemented on
> this branch (NOT deployed). #1 — HL v1 → Parkinson. #2 — engine plumbing
> (`AllowlistEntry.vol_sampling_dt_seconds`/`vol_estimator`, `build_late_resolution_config`,
> `reference_sampling_dt_seconds` reads the late_resolution slot) + `config/strategy.yaml`
> HL v1 & v31 BTC slots flipped to `vol_sampling_dt_seconds: 5` in lockstep with
> the v1 gate retune (msd 3.0 / λ 0.97 / parkinson). PM slots untouched (dt=60).
> ⚠️ The dt=5 cutover is gated on ≥1-week paper validation before deploy — the
> YAML carries warning comments and this branch is review-only. 625 tests pass.

---

## Why this task exists (the shared-feed constraint)

`engine_cadence_port_2026_05_29.md` made the live engine's mark-bucketing period
config-driven per reference symbol, with a **conflict guard**: re-registering the
same symbol at a different cadence raises at startup. v1 (`late_resolution`) and
v31 (`theta_harvester`) **both** read `reference_symbol="BTC"` and share one
bucketed history. So flipping only v31 to dt=5 makes the engine refuse to start.
The operator must either (a) move both in lockstep, or (b) decouple the feeds.
This validation answers which, by testing v1 first.

## Structural finding: v1 had no cadence-aware σ (now fixed)

Unlike v31 (whose `model_edge` already annualizes returns assuming each spans
`vol_sampling_dt_seconds`), v1's `LateResolutionConfig` had **no
`vol_sampling_dt_seconds` field**. Its σ math hardcoded a 60s-bar assumption in
two places that the backtest loader's resample period feeds into:

- `n_keep = vol_lookback_seconds // 60` — sample count of the σ window.
- `sigma_window = σ · √(tte_s / 60)` and the drift term `μ · (tte_s / 60)` —
  scaling per-bar σ to the settlement horizon.

The backtest CLI already couples the **loader's** reference-resample period to
`vol_sampling_dt_seconds` (`cli.py` → `hl_reference_resample_seconds`), so at
dt=5 the loader hands v1 5s-spaced returns. But with the math hardcoded at 60s:

- the σ window collapses 12× (60 samples × 5s = **300s**, not 3600s), and
- `sigma_window` is computed over `tte_s/60` bars when the bars are 5s wide, so
  it is **~√12 ≈ 3.5× too small** → `safety_d` ~3.5× too large → the
  `exit_safety_d` mid-hold cut barely fires → losers ride to settlement → the
  $255 drawdown in test #1.

**This PR ports cadence-awareness into v1** (mirrors v31): adds
`vol_sampling_dt_seconds` to `LateResolutionConfig` (default 60) and replaces the
hardcoded `// 60` / `/ 60.0` at all three σ sites (entry gate, σ₁ₕ mid-hold exit,
σ₅ₘ fast exit) with the configured period. At dt=60 every path is bit-identical
(the literal `60` is now an ivar holding `60`); `dt=60` backtest reproduces
**$212.01 / 62 trades / Sharpe 6.42 / DD $75.46** exactly.

---

## HL HIP-4 results (load-bearing corpus — real strikes)

Config = v1 HL prod binary slot (`config/strategy.yaml`): fav=0.85 /
extreme_max=0.99 / min_safety_d=1.0 / exit_safety_d=1.0 / vol_ewma_lambda=0.85 /
size_cap / bid-gate / fee_taker=0.00035 / slippage=5bps. Held fixed; cadence swept.

### Test 1 — naive flip (loader cadence only, 60s-hardcoded σ math)

| dt(s) | PnL      | ΔvsBase  | trades | hit   | Sharpe | maxDD    |
|-------|----------|----------|--------|-------|--------|----------|
| 60    | $212.01  | —        | 62     | 57.9% | 6.42   | $75.46   |
| 5     | $68.41   | −$143.60 | 56     | 60.5% | 0.77   | $255.44  |
| 1     | −$27.16  | −$239.17 | 56     | 57.9% | −0.29  | $255.44  |

The DD blow-up is the mis-scaled exit gate (see structural finding). This is the
state of the world **if the loader/engine cadence were flipped without the v1
code fix** — i.e. the silent train/serve skew the conflict guard exists to block.

### Test 2 — cadence-aware code fix, same prod params

| dt(s) | PnL      | ΔvsBase  | trades | hit   | Sharpe | maxDD    |
|-------|----------|----------|--------|-------|--------|----------|
| 60    | $212.01  | —        | 62     | 57.9% | 6.42   | $75.46   |
| 5     | $87.62   | −$124.39 | 84     | 47.4% | 2.68   | $73.69   |
| 1     | $145.50  | −$66.51  | 76     | 55.3% | 3.76   | $108.63  |

The fix removes the catastrophe (dt=5 DD $255 → $74) but sub-minute still loses
PnL. At dt=5 the entry gate fires **more** (84 vs 62 trades) at **lower** hit
(47.4% vs 57.9%): the 5s realized-vol read trips entries on noisier signals.

### Test 3 — dt=5 re-tune (16-cell grid, cadence-aware code)

Swept `min_safety_d ∈ {1.0,1.5,2.0,3.0}` × `exit_safety_d ∈ {1.0,2.0}` ×
`vol_lookback_seconds ∈ {3600,7200}`:

| cell (dt=5)             | PnL      | trades | hit   | Sharpe | maxDD    |
|-------------------------|----------|--------|-------|--------|----------|
| **msd3.0 / esd1.0**     | **$129.76** | 66  | 47.4% | 4.72   | $38.18   |
| msd1.5 / esd1.0         | $108.99  | 82     | 47.4% | 3.52   | $67.66   |
| msd1.0 / esd1.0 (base)  | $87.62   | 84     | 47.4% | 2.68   | $73.69   |
| msd2.0 / esd1.0         | $79.33   | 78     | 47.4% | 2.59   | $79.58   |
| msd3.0 / esd2.0         | $55.91   | 112    | 42.1% | 2.24   | $43.31   |
| msd2.0 / esd2.0         | −$47.87  | 138    | 44.7% | −1.38  | $152.53  |
| msd1.5 / esd2.0         | −$69.26  | 154    | 42.1% | −1.85  | $167.16  |
| msd1.0 / esd2.0         | −$185.74 | 166    | 42.1% | −3.61  | $258.33  |
| — baseline dt=60 —      | $212.01  | 62     | 57.9% | 6.42   | $75.46   |

(vol_lookback 3600 ≡ 7200 in every row — the loader supplies fewer returns than
`7200//5 = 1440`, so the longer window is capped to the same data; it is inert.)

Three takeaways:
- **`exit_safety_d=2.0` is uniformly destructive** — a tighter mid-hold cut
  churns out and re-enters (138–166 trades), bleeding fees/slippage.
- **Tightening `min_safety_d` helps but plateaus.** msd3.0 cuts to 66 trades
  (≈ baseline's 62) and the lowest DD of any cell ($38), but PnL caps at $130.
- **Hit rate is pinned at 47.4%** for every `esd=1.0` dt=5 cell, independent of
  `min_safety_d`. The entry gate changes *how many* trades, never *which side
  wins*. **⚠️ This was initially read as "structural edge degradation." Test 4
  proves that was wrong** — the 47.4% pin is an artifact of the *frozen*
  `vol_estimator=stdev` + `vol_ewma_lambda=0.85`, not of cadence. Both were held
  constant in tests 1–3.

### Test 4 — dt=5 round-2 grid: the cadence-SENSITIVE frozen params

Tests 1–3 froze three params most likely to interact with sub-minute sampling.
Swept `vol_estimator ∈ {stdev,parkinson}` × `exit_safety_d_5m ∈ {0,1}` ×
`vol_ewma_lambda ∈ {0,0.85,0.97}` × `min_safety_d ∈ {1.0,3.0}` (24 cells, dt=5,
`exit_safety_d=1.0`). Top + reference rows:

| cell (dt=5)                       | PnL     | trd | hit   | Sharpe | maxDD   |
|-----------------------------------|---------|-----|-------|--------|---------|
| **msd3.0 / parkinson / λ0.97**    | **$269.52** | 52 | 60.5% | 9.73 | $33.05 |
| msd1.0 / parkinson / λ0.97        | $255.87 | 56  | 63.2% | 6.28   | $87.50  |
| msd3.0 / parkinson / λ0.85        | $243.60 | 54  | 63.2% | 7.02   | $70.41  |
| msd1.0 / stdev / λ0.0             | $221.41 | 56  | 57.9% | 6.49   | $76.10  |
| msd1.0 / stdev / λ0.85 (prod)     | $87.62  | 84  | 47.4% | 2.68   | $73.69  |
| — baseline dt=60 —                | $212.01 | 62  | 57.9% | 6.42   | $75.46  |

- **`vol_estimator=parkinson` is the dominant lever.** Every top cell is
  Parkinson; `stdev` tops out at $221 (λ=0). 5s close-to-close returns are
  bounce-contaminated; Parkinson's H/L range estimator is robust to it. The
  loader supplies real per-bar H/L (resampled OHLC), so this is a genuine effect,
  not a fallback to stdev.
- **`vol_ewma_lambda`: 0.85 is the poison at dt=5.** λ=0.85 ≈ 35s half-life on 5s
  bars → jumpy σ → the 47.4% hit pin. λ=0 (sample std) or λ=0.97 (smooth) both
  recover hit to 58–63%.
- **`exit_safety_d_5m` is inert** (e5m0.0 ≡ e5m1.0 in nearly every row).
- **`min_safety_d` optimum shifts with cadence**: dt=5 likes 3.0 (denser data →
  more selectivity affordable); dt=60 likes 1.0 (Test 5).

### Test 5 — dt=60 control (is the win cadence or estimator?)

Ran the Test-4 winning configs at dt=60 to disambiguate. Sanity: prod
`msd1.0/stdev/λ0.85` reproduces **$212.01** exactly.

| config                       | dt=60 PnL / DD  | dt=5 PnL / DD   | cadence Δ |
|------------------------------|-----------------|-----------------|-----------|
| msd1.0 / parkinson / λ0.85   | **$259.74** / $72 | $238.96 / $104 | −$21      |
| msd1.0 / parkinson / λ0.97   | $250.17 / $54   | $255.87 / $88   | +$6       |
| msd3.0 / parkinson / λ0.97   | $191.03 / $7    | **$269.52** / $33 | +$78    |
| msd3.0 / parkinson / λ0.85   | $214.90 / $0    | $243.60 / $70   | +$29      |
| msd1.0 / stdev / λ0.0        | $217.75 / $72   | $221.41 / $76   | +$4       |
| msd1.0 / stdev / λ0.85 (prod)| $212.01 / $75   | $87.62 / $74    | −$124     |

**The win is mostly the estimator, not cadence.** Parkinson alone lifts dt=60
from $212 → $260 (+$48, cadence-independent). On top of Parkinson, cadence is a
near-tie: best dt=60 $259.74 vs best dt=5 $269.52 (+$10 PnL, noise on 38q) — but
dt=5's best halves drawdown ($33 vs $72) and lifts Sharpe (9.73 vs 7.58). The
per-cadence gate optimum differs (dt=60→msd1.0, dt=5→msd3.0), a real interaction.

Conclusion: **v1 is not hurt by dt=5** (the round-1 reading was a frozen-param
artifact). With Parkinson, dt=5 is PnL-neutral-to-mildly-better and risk-reducing.

---

## PM results (suggestive only — small, broken-strike corpus)

Sub-minute on PM is only measurable via the `binance_bbo` reference source
(klines are native 1m). Config = v1 PM prod slot (tte_max=86400, fee_model
pm_binary, fee_rate 0.07). Cadence-aware code NOT applied to these runs (they
predate the fix), so treat as the *naive* sweep on PM:

| cell        | dt(s) | ref          | PnL      | trades | hit   | Sharpe | maxDD   |
|-------------|-------|--------------|----------|--------|-------|--------|---------|
| klines_dt60 | 60    | klines (1m)  | $23.78   | 18     | 9.1%  | 1.10   | $60.98  |
| bbo_dt60    | 60    | binance_bbo  | $240.20  | 78     | 50.0% | 8.90   | $65.43  |
| bbo_dt5     | 5     | binance_bbo  | $498.95  | 34     | 63.6% | 19.34  | $30.82  |
| bbo_dt1     | 1     | binance_bbo  | $419.90  | 58     | 68.2% | 13.66  | $44.59  |

PM appears to *love* dt=5 (+$259), the opposite of HL. **Do not weight this:**
- n ≈ 18 markets, and `_binary_strike` is broken past 2026-05-09 (degrades to
  0.0; see `v37_hl_1s_sampling_2026_05_28.md` caveat). With strike≈0,
  `ln(S/K)→∞` so v1's `safety_d` gates are degenerate — the cadence effect here
  flows through paths that won't exist with valid strikes.
- klines_dt60 is a "no-data" baseline (kline cache ends 2026-05-09), not a fair
  reference.

The PM signal is too compromised to lean on. Notably, though, PM also favoured
dt=5 — directionally consistent with the corrected HL read (Test 4/5) that dt=5
is fine for v1 once the estimator/λ are cadence-appropriate. (These PM runs used
the prod `stdev`/λ0.85 yet still liked dt=5 — but the degenerate strikes make
that uninterpretable.) Revisit PM when the kline cache + BBO coverage extend
through a longer window with valid strikes, and re-run with Parkinson.

---

## Recommendation

**(a, conditional) v1 *tolerates* dt=5 — so the v31 flip can be lockstep with NO
feed decoupling, provided v1 is also switched to `vol_estimator=parkinson`. The
estimator swap is a standalone +$48 win worth taking at dt=60 regardless.**

Three things follow, in priority order:

1. **Adopt Parkinson σ for v1 (cadence-independent, do first).** At the existing
   dt=60 it lifts HL PnL $212 → ~$260 (best `msd1.0/parkinson/λ0.85`, DD ≈ flat),
   the single largest and most robust lever found. This is just a `vol_estimator`
   config change on the v1 HL slot — no code, the strategy already supports it.
   Validate on a longer corpus, then a human-gated `config/strategy.yaml` edit.

2. **The v31 dt=5 flip no longer needs decoupling.** Round-1's blocker — "dt=5
   craters v1, so the shared BTC feed must be split" — was a frozen-param
   artifact. Once v1 is on Parkinson, dt=5 is PnL-neutral-to-better for it
   (best dt=5 $269 vs best dt=60 $260, and lower DD). So **flip v1+v31 to dt=5
   in lockstep**; v1 just wants its gate retuned for the cadence
   (`min_safety_d` 1.0 → 3.0, `λ` → 0.97). The per-(symbol,dt) history change is
   **unnecessary** — drop it from the plan.

3. **Gate everything on a longer, valid-strike corpus.** The HL window is 38
   questions; the +$10 dt=5-over-dt=60 edge is noise-level and the per-config
   cadence deltas swing −$21…+$78. Only the +$48 Parkinson effect is
   robust-magnitude. Do NOT flip live cadence on this corpus alone — run the
   ≥1-week paper validation from `engine_cadence_port_2026_05_29.md` first, and
   ideally re-confirm on 2–3 months of HIP-4.

### Sequencing for the live path

- **Now (this PR):** strategy-side cadence-awareness landed (backtest-only).
- **Next, independent:** switch v1 to Parkinson at dt=60 (human-gated config).
- **Then, if pursuing v31 dt=5:** because v1 will be on Parkinson, set v1+v31
  both to `vol_sampling_dt_seconds: 5` in lockstep and retune v1's gate
  (`min_safety_d`, `λ`). The engine-side knob still needed for v1:
  `engine/config.py:build_late_resolution_config` must read
  `vol_sampling_dt_seconds` (and `vol_estimator`) from the v1 YAML slot and
  register that cadence on the shared MarketState — mirror how theta exposes it.
  No `MarketState` decoupling required (the conflict guard is satisfied because
  both slots agree on dt=5).

---

## What this PR changes

1. **`hlanalysis/strategy/late_resolution.py`** — `LateResolutionConfig` gains
   `vol_sampling_dt_seconds: int = 60`; the entry gate, σ₁ₕ mid-hold exit, and
   σ₅ₘ fast exit replace hardcoded `// 60` / `/ 60.0` with the configured period.
   `build_v1_late_resolution` threads the param. Default 60 = bit-identical.
2. **`tests/unit/test_strategy_late_resolution.py`** — `+2`: default-60 marker
   and a cadence-awareness test (same inputs ENTER at dt=60, HOLD at dt=5).
3. **Runners** (backtest-only): `scripts/run_v1_cadence_hl.py`,
   `run_v1_cadence_pm.py`, `run_v1_cadence_hl_fixed.py`,
   `run_v1_cadence_hl_tune_dt5.py`, `run_v1_cadence_hl_tune_dt5_estimator.py`,
   `run_v1_cadence_hl_control_dt60.py`.

### Follow-up implementation landed on this branch (review-only, not deployed)

4. **Engine plumbing** — `AllowlistEntry` gains `vol_estimator` +
   `vol_sampling_dt_seconds`; `_late_resolution_config_from_entry` threads both;
   `reference_sampling_dt_seconds` now reads the late_resolution slot's cadence
   (so v1 participates in the shared-feed cadence registration / conflict guard).
   Tests in `test_engine_runtime_config.py` + `test_engine_runtime_cadence.py`.
5. **Live config flip** (`config/strategy.yaml`, DEPLOY-AFFECTING, paper-gated):
   HL v1 (ref BTC) binary+bucket+defaults → `vol_estimator: parkinson`,
   `vol_sampling_dt_seconds: 5`, `min_safety_d: 3.0`, `vol_ewma_lambda: 0.97`;
   v31 (ref BTC) theta block → `vol_sampling_dt_seconds: 5` (lockstep). PM slots
   (ref BTCUSDT) untouched at dt=60/stdev. Verified coherent: BTC→{5}, BTCUSDT→{60}.

Safety gates untouched. The dt=5 cutover still requires ≥1-week paper validation
before deploy (warning comments in the YAML).

## Artifacts

- Naive HL sweep (test 1):    `data/sim/runs/v1-cadence-hl-2026-05-29/`
- Cadence-aware HL (test 2):  `data/sim/runs/v1-cadence-hl-fixed-2026-05-30/`
- dt=5 gate tune (test 3):    `data/sim/runs/v1-cadence-hl-tune-dt5-2026-05-30/`
- dt=5 estimator grid (test 4): `data/sim/runs/v1-cadence-hl-tune-dt5-est-2026-05-30/`
- dt=60 control (test 5):     `data/sim/runs/v1-cadence-hl-control-dt60-2026-05-30/`
- PM sweep:                   `data/sim/runs/v1-cadence-pm-2026-05-29/`
