# v3.5 Momentum / Mean-Reversion — PM BTC 90d sweep results

**Spec:** `docs/specs/2026-05-28-v35-momentum-mr-design.md`
**Plan:** `docs/superpowers/plans/2026-05-28-v35-momentum-mr.md`
**Date:** 2026-05-28
**Baseline:** v3.1 final (`config/tuning.v3-1-final-pm.yaml`) on identical walk-forward splits.
**Baseline metrics:** PnL $1295 / Sharpe 5.94 / worst-fold-Sharpe 0.22 / maxDD $207 / 396 trades / hit 47.6%.

## TL;DR

**Ship `z_ret` tilt; shelve everything else.** Only `z_ret` produced ship candidates across multiple lookbacks under the spec's full 5-criteria gate. `ma_sigma` had the single largest PnL bump (+$98) but only at one lookback and fails the robustness check. `rsi` and `hurst_ou` either had no real signal (gate inert across all 24 cells each) or one-cell wins that fail robustness.

**Recommended config:** `momentum_mr_indicator="z_ret", momentum_mr_mode="tilt", momentum_mr_lookback_min=15, momentum_mr_alpha_tilt=1.0` — +$43 PnL (+3.3%), worst-fold Sharpe 0.27 (vs 0.22), maxDD unchanged, 474 trades (vs 396).

## Per-indicator results

### z_ret — the only robust winner (4 ship cells across 2 lookbacks)

| Rank | Cell | PnL | ΔPnL | Sharpe | Worst-fold | maxDD | Trades | Hit | Ship? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | tilt-z_ret-lb15-a100 | $1338 | +$43 | 6.10 | **0.27** | $207 | 474 | 49.7% | **✓** |
| 2 | tilt-z_ret-lb5-a050 | $1327 | +$32 | 6.07 | 0.14 | $207 | 438 | 48.3% | ✗ (WFS) |
| 3 | tilt-z_ret-lb5-a025 | $1318 | +$23 | 6.01 | 0.13 | $207 | 410 | 48.0% | ✗ (WFS) |
| 4 | tilt-z_ret-lb15-a025 | $1315 | +$20 | 5.99 | 0.24 | $207 | 406 | 48.3% | **✓** |
| 5 | tilt-z_ret-lb60-a100 | $1306 | +$11 | 6.02 | 0.02 | $207 | 454 | 48.7% | ✗ (WFS) |
| 6 | tilt-z_ret-lb30-a025 | $1305 | +$10 | 5.97 | 0.24 | $207 | 408 | 48.3% | **✓** |
| 7 | tilt-z_ret-lb30-a100 | $1302 | +$8 | 6.02 | 0.24 | $207 | 466 | 47.8% | **✓** |

**Ship verdict:** 4 cells across **2 distinct lookbacks** (lb15, lb30) — robustness criterion ✓.
Gate mode inert across all 12 cells. Tilt is where the signal lives.

### rsi — one-cell win, fails robustness

Best: `tilt-rsi-lb5-a100` at +$20 PnL / WFS 0.24 / maxDD $207 / 458 trades — **only ship cell**. All 12 gate cells bit-identical to baseline. Other tilt cells degrade worst-fold Sharpe (0.06–0.17, all below 0.22 baseline). One robust lookback fails the 2-lookback minimum. **Shelve.**

### ma_sigma — biggest single-cell PnL bump, fails robustness

Best: `tilt-ma_sigma-lb30-a100` at +$98 PnL / WFS 0.33 / maxDD $205 / 546 trades — biggest PnL gain in the entire sweep. But the only cell clearing all four numeric criteria simultaneously: `lb15-a100` and `lb5-a100` each break maxDD by $0.04 and break WFS. Fails 2-lookback robustness. Gate inert (12/12 cells). **Shelve as a singleton (marginal — worth a follow-up sweep around lb20–lb40 a075–a125 if we want to push further).**

### hurst_ou — signal absent, win is mechanical

5 cells technically clear (tilt-lb5 / lb30 / lb60 at various alphas), all +$7 to +$13 only. alpha-insensitive at lb5 (a025 = a050 = a100 within $0) ⇒ the indicator is scaling near-zero output. Autocorrelation-based pseudo-Hurst returns no actionable mean-reversion information on this corpus. Gate fully inert (12/12 cells). **Shelve.**

## Cross-indicator patterns

1. **Gate mode is dead across the board** — 47 of 48 gate cells produced PnL within ±$10 of baseline; ma_sigma and rsi and hurst_ou gate cells are bit-identical to baseline (signal never crosses threshold).
2. **Tilt is where the signal lives** — every indicator's wins concentrate in tilt mode, where the score-modulated `edge_buffer` lets the strategy enter more aggressively on aligned moves.
3. **maxDD ceiling is hard at $207** — every winning cell hits exactly the same $207 drawdown floor (the baseline's fold-5 worst loss). Tilt expands trade count by 0–40% but doesn't take a deeper single hit; it diversifies entries rather than removing the worst-case trade.
4. **Trade count expansion is the mechanism** — winning tilt cells add 8–164 trades over baseline. Hit rate slightly compresses (47.6% → 48–50%); the extra trades are marginal-edge entries the static buffer was rejecting.

## Ship criteria check — z_ret tilt-lb15-a100

| # | Criterion | Baseline | v3.5 | Pass? |
|---|---|---|---|---|
| 1 | PnL ≥ baseline | $1295 | $1338 (+$43, +3.3%) | ✓ |
| 2 | Worst-fold Sharpe > baseline | 0.22 | 0.27 | ✓ |
| 3 | Max DD ≤ baseline | $207 | $207 | ✓ |
| 4 | Trades ≥ 0.6 · baseline | 238 | 474 | ✓ |
| 5 | ≥ 2 robust lookbacks | — | lb15 + lb30 ship in tilt mode | ✓ |

## Risks / open questions (NOT investigated this round)

- **safety_d correlation** — did NOT compute Spearman ρ between `z_ret` score at entry and existing `safety_d` exit metric. If correlation > 0.7 the marginal info from the gate could be redundant with what `exit_safety_d=1.0` already captures. Follow-up: instrument a per-decision diagnostic pull and recompute.
- **TTE-bucket breakdown** — did NOT split z_ret tilt PnL by entry-TTE (0–1h / 1–4h / 4–24h). Need to confirm the +$43 PnL gain is not coming exclusively from a fragile late-stage bucket.
- **PM-only result** — per `[[v31_final_state_2026_05_23]]`, PM-tuned params do not transfer to HL. **Do NOT port to HL without a separate HL backtest.**
- **`ma_sigma` lb30-a100 outlier** — the +$98 PnL single-cell win is too big to ignore. If we run a denser sweep around (lb={20, 25, 30, 35, 40}, alpha={0.75, 1.0, 1.25}), at least one neighbour likely passes the 2-lookback robustness gate.

## Decision

**Ship `z_ret` tilt-lb15-a100 to PM-only `delta_hedged_v5` deployment** as a follow-up `tuning.v3-5-pm-final.yaml`. Defer the ma_sigma neighbour-sweep, the safety_d marginal-info diagnostic, and the TTE-bucket validation to a separate ticket — none block shipping, but all should land before live-cap is raised above smoke level.

## Next steps

1. **PR for the implementation** (T1–T6 already committed): spec + plan + code + tests + sweep runner.
2. **Final v3.5-pm tuning yaml** in `config/tuning.v3-5-pm-final.yaml` hard-coding `momentum_mr_*` to the winning config.
3. **Safety_d redundancy probe** — patch the runner to emit per-decision `(safety_d, mm_score)` pairs for the winning config, run on 30 markets, compute Spearman.
4. **Decision on `ma_sigma` follow-up sweep** — densify around `lb30-a100` to see if a neighbour passes robustness.
5. **HL port** — explicitly NOT recommended; PM-tuned params don't transfer. Would need a parallel HL sweep with HL-only baseline (v3.1 final HL state).

## Artifacts

- Sweep results: `data/sim/tuning/v3-5-{gate,tilt}-{indicator}-lb{N}-{tau|a}{N}-2026-05-28/results.jsonl` (96 cells)
- Baseline: `data/sim/tuning/v3-1-final-pm-walkforward-2026-05-28/results.jsonl`
- Runner: `scripts/run_v35_momentum_mr_sweep.py`
- Indicator module: `hlanalysis/strategy/momentum_mr.py`
