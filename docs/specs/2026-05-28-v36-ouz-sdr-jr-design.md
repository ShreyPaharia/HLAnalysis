# v3.6 — OU-Z + SDR + Jump-Ratio trust weight (PM/HL extension)

**Date:** 2026-05-28
**Baseline:** v3.5 winning configs:
- PM-only: `ma_sigma tilt-lb30-a100` (+$98 PnL on PM, −$11 on HL)
- Universal: `z_ret tilt-lb3-a100` (+$57 PM, +$13 HL)

**Research note:** `docs/research/alternative_momentum_mr_indicators.md` (commit 86be45e)

## Motivation

Two free upgrades + one risk-aware weight, identified from the alternative-indicators research:

1. **OU-Z** plugs into ma_sigma's slot. Fixes ma_sigma's arbitrary-window-mean problem by fitting an AR(1) to BTC log-price, scoring `(price − μ_eq) / σ_eq` in the OU coordinate system. Likely amplifies ma_sigma's +$98 PM gain; may even fix HL.
2. **SDR (Signed Drift Ratio)** plugs into z_ret's slot. Same formula but denominator is `√BPV` (jump-robust) instead of rolling sample-std. Strips jump contamination — z_ret over-fires on news gaps; SDR doesn't. Likely amplifies +$57 z_ret gain.
3. **Jump Ratio trust weight**: multiply the score by `(1 − JR)` where `JR = max(0, (RV − BPV)/RV)` reads the jump fraction over the same lookback. When the market is gapping, momentum estimators are less reliable — JR throttles the tilt. Free add-on (BPV already computed in v3.2 volclock path).

## Design

### Indicators (added to `hlanalysis/strategy/momentum_mr.py`)

| Indicator | Formula | Regime |
|---|---|---|
| `sdr` | `cum_ret_LB / (√BPV_LB · √LB)`, signed to favorite | mirror z_ret thresholds |
| `ou_z` | OLS fit `P_t = c + φ·P_{t-1} + ε_t` on cumsum(returns). `μ_eq = c/(1−φ)`. `σ_eq = std(ε) / √(1−φ²)`. Score `(P_last − μ_eq) / σ_eq`, signed. | `mr` when `|score|>2` AND φ ∈ (0, 1); `momentum` when φ ≥ 1 (random-walk / trending); else `neutral` |

Both clipped to [-1, +1] like existing indicators. Pure NumPy.

Edge cases:
- OU-Z: `|φ| ≥ 1` → not mean-reverting; fall back to score 0, regime "neutral".
- OU-Z: insufficient data (< 30 bars) → neutral.
- SDR: BPV ≤ 0 → fall back to z_ret-style sample std.

### Jump Ratio trust weight (config-level, in `theta_harvester._evaluate_entry`)

Add `momentum_mr_jr_trust_weight: bool = False` to `ThetaHarvesterConfig`. When enabled (and `momentum_mr_enabled=True`), compute:

```
RV = sum(r_LB ** 2)
BPV = (π/2) * sum(|r_i| * |r_{i-1}|)   # over the same LB window
JR = max(0, min(1, (RV - BPV) / RV))   # jump fraction in [0, 1]
score *= (1 - JR)                       # shrink tilt under heavy jumps
```

Applied to both gate and tilt modes. Default False keeps v3.5 bit-identical.

## Sweep methodology

Reuse `scripts/run_v35_extension.py` infrastructure with new indicator names. Two sweep extensions:

### 1. OU-Z + SDR core sweep (PM)

Same grid as v3.5 original: `lb ∈ {5, 15, 30, 60}` × `α ∈ {0.25, 0.5, 1.0}`, tilt-only (gate was inert across the board). 4 × 3 × 2 indicators = 24 cells.

### 2. JR-trust-weight stack (PM)

Apply JR-trust-weight on top of the top 4 v3.5 + v3.6 cells: `[ma_sigma-lb30-a100, z_ret-lb3-a100, ou_z-best, sdr-best]`. 4 cells with JR on, 4 off (baseline reproducibility). 8 cells.

### 3. HL validation

For each of the top 3 PM winners across v3.5 + v3.6, run a single `hl-bt run --kind binary` on HL HIP-4 BTC. 3 runs.

**Total compute:** ~32 cells × 22s = ~12 min PM + ~30s HL × 3 = ~14 min total wall clock.

## Ship criteria

Same as v3.5:
1. PnL ≥ v3.1 baseline ($1295 PM, $266.59 HL)
2. Worst-fold Sharpe > baseline (PM: 0.224)
3. maxDD ≤ baseline ($207 PM, $33.34 HL)
4. Trades ≥ 0.6 × baseline
5. PM only: ≥ 2 robust lookbacks for the winning indicator/mode

Plus a v3.6-specific criterion: **the new variant must strictly dominate the v3.5 winner it's replacing.** OU-Z must beat ma_sigma at the same (lookback, alpha); SDR must beat z_ret. Otherwise just keep v3.5.

## Deliverables

- `hlanalysis/strategy/momentum_mr.py` — `_ou_z()` and `_sdr()` functions added; dispatcher accepts `"ou_z"` and `"sdr"`.
- `hlanalysis/strategy/theta_harvester.py` — `momentum_mr_jr_trust_weight` config field + wiring in gate/tilt paths.
- `tests/unit/test_momentum_mr.py` — extended with OU-Z and SDR tests (~6 tests).
- `tests/unit/test_theta_harvester_v36_jr.py` — JR-trust-weight unit + integration tests.
- `scripts/run_v36_sweep.py` — sweep driver mirroring v3.5 extension script.
- `docs/research/v36_ouz_sdr_jr_2026_05_28.md` — results writeup + ship decision.

## Out of scope

- CVD-Z (deferred to v3.7; needs new data plumbing for aggressor-tag bucket aggregation).
- VPIN, realized skewness — research note KILL signals.
- HL re-test of ma_sigma at expanded corpus — separate when corpus grows.
- Live deployment — backtest study only; v3.5 PR lands first.

## References

- v3.5 sweep results: `docs/research/v35_momentum_mr_2026_05_28.md`
- Alternative-indicators research: `docs/research/alternative_momentum_mr_indicators.md`
- BPV math: `hlanalysis/strategy/_numba/vol.py::bipower_variation_sigma`
