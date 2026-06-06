# v3.1 final-state analysis — what's necessary, what's dead weight

**Date:** 2026-05-23
**Goal:** From-first-principles ablation of v3.1 (`theta_harvester`)
production config. Identify knobs that no longer earn their keep under the
realistic PM-curve fee model, and converge on a minimal config that
Pareto-dominates prod on PnL × Sharpe × max-DD.

## Method

- **Corpus:** PM 364 markets (May 2025 → May 2026), realistic fee curve
  (`pm_binary` / `feeRate=0.07`), walk-forward 5 OOS splits (60/60/60).
- **HL confirmation:** 14 binary + 16 bucket questions (May 2026),
  full-corpus single run.
- **Decision rule:** keep a knob only if removing/loosening it strictly
  degrades on at least one of {total PnL, avg Sharpe, max DD}.

Prod baseline (PM, 24h tte, fav=0.85, d=1.0, all gates on):
**$855 PnL / Sharpe 3.07 / maxDD $508 / 584 trades / 47.8% hit**

## Per-knob verdict

| Knob | Prod | Verdict | Evidence |
|---|---|---|---|
| `edge_max=0.20` | on | **KILL** | identical with `edge_max=None` ($855/3.07/$508) — never binds on PM. |
| `min_distance_pct=0.002` | on (PM) | **KILL** | +$27 PnL, +0.03 Sharpe, identical DD when removed. Originally tuned on flat fees (memory `feedback_…`); under PM-curve the near-strike band no longer dominates losses. |
| `min_bid_notional_usd=10` | on (PM) | **KEEP** (safety gate) | bit-identical in backtest, but kept as live thin-book / spoof defense. Backtests don't replay adversarial microstructure; cost of keeping a non-firing safety gate is zero. |
| `vol_clip_min=0.05` | on | **KILL** | identical when removed. Post-σ-fix (commit `cc44bf6`) the 1-min ref bars never produce σ below the floor. Dead since May 2026. |
| `topup_enabled=True` | on | **KILL on PM** | identical when removed in PM sim (no partial fills). **KEEP on HL** — designed for HL HIP-4 partial-fill recovery (`fixes commit 2026-05-20`). |
| `half_spread_assumption=0.005` | 0.005 | **KEEP (cosmetic)** | mathematically equivalent to shifting `edge_buffer` by +0.005; kept for diagnostic interpretability. Killing it with absorbed edge_buffer is fine. |
| `fee_taker` (entry) | 0.0 | **KEEP** (zero) | already zeroed in commit `c2eea84`; the entry leg is venue-subsidized. |
| `drift_blend=0.0` | 0 | **KEEP** (zero) | drift contribution disabled since v2; the lookback config is dead but doesn't affect math. |
| `edge_buffer=0.02` | 0.02 | **TUNE → 0.03** | +$62 PnL, +0.13 Sharpe, −$23 DD on PM. Sweet spot is 0.025–0.03. |
| `favorite_threshold=0.85` | 0.85 | **TUNE → 0.90** | +$340 PnL, +2.81 Sharpe, −$301 DD vs prod. Memory `favorite_threshold_sweep_2026_05_23` (prefers 0.85) is invalidated by the PM fee-curve model — under realistic fees the high-p band carries less cost per trade so we can be more selective. |
| `exit_safety_d=1.0` | 1.0 | **KEEP** | d=0 catastrophic (−$450); d=0.9/1.0/1.1 cluster tightly; 1.0 has the cleanest per-split robustness. |
| `exit_take_profit_mode=True` | True | **KEEP** | tp_mode + `exit_fee=0.0007` adds ~$20 PnL / +0.06 Sharpe at the new fav=0.9 optimum. Small but non-zero. |
| `exit_fee=0.0007` | 0.0007 | **KEEP** | sensitivity sweep ±100% changes PnL by ≤$6. |
| `tte_max_seconds=86400` (PM) | 24h | **KEEP** | optimum confirmed; 48h cap identical (no PM markets longer than 24h). |
| `vol_lookback_seconds=3600` | 3600 | **KEEP** | 1800s → −$276 PnL, 7200s → no PnL gain. 1h is the sweet spot. |
| `vol_clip_max=5.0` | 5.0 | **KEEP** (defensive) | never binds in current corpus; cheap safety. |
| `stop_loss_pct=null` | null | **KEEP** (disabled) | memory `tuning v1-cap995-sl` showed stops cost net PnL. |
| `time_stop_seconds=0` | 0 | **KEEP** (disabled) | tte cap already provides time-bounding. |

## Final config (PM-tuned, HL-applicable)

| Knob | Prod | Final |
|---|---|---|
| `favorite_threshold` | 0.85 | **0.90** |
| `edge_buffer` | 0.02 | **0.03** |
| `edge_max` | 0.20 | **None** |
| `min_distance_pct` (PM only) | 0.002 | **None** |
| `vol_clip_min` | 0.05 | **0.0** |
| `topup_enabled` | True | **PM: False; HL: True** |
| everything else | unchanged |

## Results

| Config | PnL | Sharpe | maxDD | Trades | Hit |
|---|---:|---:|---:|---:|---:|
| **PM prod baseline** | $855 | 3.07 | $508 | 584 | 47.8% |
| **PM minimal (final)** | **$1,295** | **5.94** | **$207** | 396 | 47.6% |
| ΔPM | **+$440 (+51%)** | **+2.87** | **−$301 (−59%)** | −188 | −0.2pp |

Per-split robustness check (PM):

| Config | s0 | s1 | s2 | s3 | s4 | min |
|---|---:|---:|---:|---:|---:|---:|
| prod baseline | +3 | +272 | **−228** | +784 | +24 | −228 |
| final config | +174 | +318 | +218 | +564 | +21 | **+21** |

All 5 splits positive under the final config. The prod baseline relies on
split 3's $784 outlier; final config has a more uniform distribution.

## HL confirmation

HL HIP-4 corpus: 14 binary + 16 bucket questions, 2026-05-09 to 2026-05-23.
Single-run (corpus too small for walk-forward).

| Config | binary PnL | binary trades | bucket PnL | bucket trades |
|---|---:|---:|---:|---:|
| HL prod (fav=0.85, eb=0.02) | **$178.16** (71.4% hit) | 46 | **$211.06** (75.0% hit) | 46 |
| Conservative (prod + eb=0.03 + dead-knob kills) | $157.50 | 40 | $185.10 | 36 |
| PM-minimal (fav=0.90, eb=0.03 + kills) | $104.63 | 26 | $123.24 | 30 |

**Kill-only ablation on HL** — to isolate dead-knob kills from the `eb`
change, each kill applied to prod individually:

| Config | binary PnL | bucket PnL | Δ vs prod |
|---|---:|---:|---:|
| prod (control) | $178.16 | $211.06 | — |
| prod + `edge_max=None` | $178.16 | $211.06 | **bit-identical** |
| prod + `vol_clip_min=0.0` | $178.16 | $211.06 | **bit-identical** |
| prod + `topup_enabled=False` | $178.16 | $211.06 | **bit-identical** (in sim; live ≠ sim) |
| prod + all three kills | $178.16 | $211.06 | **bit-identical** |

Hit rate, Sharpe, maxDD, trade count — every metric matches to the cent.
The dead-weight knobs are definitively dead on the HL backtest corpus.

**Key HL findings:**

1. **`fav=0.90` is venue-specific to PM.** Applying it to HL strips out ~40%
   of trades and ~40% of PnL. HL HIP-4 favorites trade at 0.98–0.9999 but
   bid books often pull back to 0.85–0.90 territory; tightening to 0.90
   discards genuinely profitable bid-side entries. HL prefers `0.85`.
2. **`edge_buffer=0.03` is also venue-specific.** Conservative config
   (prod with eb 0.02→0.03 plus dead-knob kills) loses $21 on binary and
   $26 on bucket. HL prefers `0.02`.
3. **Dead-knob kills are safe on HL.** Killing `edge_max` and
   `vol_clip_min` produces bit-identical PnL/Sharpe/maxDD/trades/hit.
4. **Topup never fires in sim** (no partial fills modeled). MUST stay
   enabled in live HL — that's the actual point of the gate.

## Per-venue final configs

**PM (backtest-tuned, paper):**
- `favorite_threshold=0.90`, `edge_buffer=0.03`, `tte_max=24h`,
  `exit_safety_d=1.0`, all 4 dead knobs off.

**HL (production):**
- Keep prod `favorite_threshold=0.85`, `edge_buffer=0.02`, `tte_max=12h`.
- Kill dead-weight knobs: `edge_max=None`, `vol_clip_min=0.0`.
- Keep `topup_enabled=True` (live partial-fill recovery).
- Optionally also clean up dead config: `drift_lookback_seconds` is unused
  since `drift_blend=0`; `min_distance_pct`, `min_bid_notional_usd` already
  null/0 on HL.

## Knob count

- **Prod:** 16 tunable knobs in `theta_harvester` block of `strategy.yaml`
- **Final:** 12 active knobs (4 killed; `min_bid_notional_usd` kept as
  live safety gate even though backtest shows bit-identical)

## Recommendation

1. Ship `favorite_threshold=0.90`, `edge_buffer=0.03` on v3.1.
2. Set `edge_max=None`, `vol_clip_min=0.0` (kill dead config).
3. Keep `topup_enabled=True` on HL (real partial-fill issue), `False` on
   any PM deployment.
4. Re-run `v1` (`late_resolution`) ablation with the same methodology —
   different knobs but same dead-weight risk.
5. Investigate whether `exit_take_profit_mode` can be merged with
   `exit_edge_threshold` math: at the new minimal config the legacy and
   tp-mode formulations differ by ~$20 PnL — borderline worth the code
   complexity.
