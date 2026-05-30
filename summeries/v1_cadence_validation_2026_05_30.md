# v1 (late_resolution) sub-minute reference-sampling validation — 2026-05-30

**Branch:** `feat/v1-cadence-validation`
**Strategy:** v1 `late_resolution` (near-resolution favorite arb on binaries)
**Corpora:** HL HIP-4 BTC binaries+buckets (2026-05-06 → 2026-05-28, 38 questions)
and PM BTC Up/Down via Binance-perp BBO ticks (2026-05-06 → 2026-05-29, ~18).
**Companion to:** `v37_hl_1s_sampling_2026_05_28.md` (the theta/v31 dt sweep that
picked dt=5) and `engine_cadence_port_2026_05_29.md` (the shared-feed constraint).

## TL;DR — decision: **KEEP v1 AT dt=60; DECOUPLE v31's BTC feed for its flip**

Sub-minute reference sampling **hurts v1 on HL HIP-4**, the opposite of what it
does for v31/theta (+$49 at dt=5). This holds across three escalating tests:

1. **Naive flip** (loader cadence only, v1's σ math left hardcoded at 60s):
   catastrophic — dt=5 = $68 / DD $255, dt=1 = −$27 / DD $255 (vs $212 / DD $75).
2. **Cadence-aware code fix** (this PR — v1's σ math now scales by the actual
   bar period): repairs the drawdown blow-up but PnL still trails — dt=5 = $88,
   dt=1 = $146, both < $212 baseline.
3. **dt=5 re-tune** (16-cell grid over the entry/exit gates): best cell is
   $129.76 (Sharpe 4.72, DD $38) — still **−$82 (−39%)** vs the dt=60 baseline.

Because v1 and v31 share one `reference_symbol="BTC"` bucketed mark history and
the engine's conflict guard refuses to start if they disagree on cadence
(`engine_cadence_port_2026_05_29.md`), moving v31 to dt=5 in **lockstep** would
drag v1 from $212 → ~$130 on HL. So v31's dt=5 flip must instead **decouple**
its BTC history from v1's (scoped below). v1 itself stays at dt=60.

> No live/production config changed. `config/strategy.yaml` keeps every slot at
> `vol_sampling_dt_seconds: 60`. The only code change is making v1's σ math
> cadence-aware (bit-identical at the default dt=60; verified by the full suite).

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
  wins*. The 5s σ mis-ranks the "safe favorite" relative to the 60s σ (57.9%
  hit). This is the crux: v1's late-resolution edge is **structurally degraded**
  by denser sampling, not a tuning miss. No dt=5 cell reaches $212.

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

The PM signal is too compromised to override the clean HL result. The HL HIP-4
corpus has real strikes and is the corpus v1's HL gates are tuned for; it
decisively says dt=5 hurts v1. Revisit PM when the kline cache + BBO coverage
extend through a longer window with valid strikes.

---

## Recommendation

**(b) Keep v1 at `vol_sampling_dt_seconds=60`. The v31 dt=5 flip requires
decoupling v31's BTC mark history from v1's.**

Rationale: v1 loses $82–$124 (39–59%) on HL at dt=5 even after the σ-scaling fix
and a full gate re-tune, and the loss is structural (hit-rate degradation, not a
tunable). A lockstep flip would sacrifice that to give v31 its +$49. Net is
negative. So the two strategies must run at different cadences on the shared BTC
feed, which the current single-history-per-symbol engine forbids.

### Scoping the decouple (do NOT implement here — human-gated, separate PR)

The engine keeps **one** bucketed history per `reference_symbol`
(`MarketState._mark_history_by_symbol`, `_mark_bucket_ns_by_symbol`), and
`set_reference_cadence` raises on a same-symbol cadence conflict. Two options to
let v1@60 and v31@5 coexist on `BTC`:

- **Option A — per-(symbol, cadence) histories (preferred).** Key the bucket
  period and history ring by `(symbol, dt)` instead of `symbol`. On each
  inbound BTC `MarkEvent`, fan it into every registered `(BTC, dt)` bucket.
  `Scanner._required_returns_n` and the returns lookup then read the
  `(symbol, dt)` series for their slot. Drop the conflict guard (now satisfiable).
  Cost: a second ~720-deep deque for BTC; one extra bucketing op per BTC tick.
- **Option B — distinct symbol key for v31.** Register v31's slot under a
  synthetic key (e.g. `"BTC@5s"`) fed from the same BTC mark stream. Simpler
  routing but leaks a cadence detail into the symbol namespace and needs the
  adapter/scanner to map it; messier than A.

Either is backtest-neutral (backtest already parameterizes the loader per run).
Gate behind the ≥1-week paper-trade validation from
`engine_cadence_port_2026_05_29.md` before any live cutover.

### On a future v1 dt=5

If a longer, valid-strike corpus ever shows v1 benefiting from dt=5, the
strategy-side plumbing is already done (this PR). The remaining step would be the
engine-side knob: have `engine/config.py:build_late_resolution_config` read
`vol_sampling_dt_seconds` from the v1 YAML slot (mirroring how theta exposes it)
and register that cadence on the shared MarketState. Not needed today.

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
   `run_v1_cadence_hl_tune_dt5.py`.

No engine code, no live config, no safety gates touched.

## Artifacts

- Naive HL sweep:  `data/sim/runs/v1-cadence-hl-2026-05-29/`
- Cadence-aware HL: `data/sim/runs/v1-cadence-hl-fixed-2026-05-30/`
- dt=5 HL tune:    `data/sim/runs/v1-cadence-hl-tune-dt5-2026-05-30/`
- PM sweep:        `data/sim/runs/v1-cadence-pm-2026-05-29/`
