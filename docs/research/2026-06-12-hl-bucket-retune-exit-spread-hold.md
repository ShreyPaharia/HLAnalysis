# HL v31 bucket retune — `exit_spread_hold` kills the doom-loop churn

**Date:** 2026-06-12 · **Scope:** ANALYSIS + PROPOSED config (NOT deployed, NOT live-changed).
**Data:** recorded HL hip4 corpus, settlement days 2026-06-06..06-10, all runs `--no-cache`
(the event-array raw-tick cache bug was fixed in `8cb4c08`; --no-cache is ground truth).
**Live truth:** engine `state.db` fill table, `source='venue'` (dedups the SHR-72 router+venue
double-count), PnL = Σ(closed_pnl−fee) incl. settlement.

## Problem
On persistently-wide bucket books the theta strategy round-trips at a loss: buy@ask →
`exit_safety_d` fires → IOC-sell the whole position @wide-bid → re-enter@ask → repeat
(SHR-102). Real money-loser on live too (06-07 #1670 lost −$133 doing this exact loop).

## Sweep (06-07, the churn day; LIVE −133.30/48 fills)
| config | fills | PnL |
| :-- | --: | --: |
| base (gates off, esd=1.0) | 49 | −185.39 |
| exit_safety_d=0.0 | 28 | −158.58 |
| **exit_spread_hold=0.04** | **23** | **+13.44** |
| exit_spread_hold=0.06 | 23 | +13.44 |
| entry_spread_gate=true | 49 | −185.39 (inert) |
| entry_spread_gate + hold=0.04 | 23 | +13.44 (gate adds nothing) |

The lever is `exit_spread_hold` (hold-to-settle when held half-spread > threshold). 0.04==0.06
here. `entry_spread_gate` is inert (entries weren't the problem); `exit_safety_d=0` only half-helps.

## Cross-day validation (v31 bucket, exit_spread_hold=0.04)
| day | LIVE | baseline (fills) | hold=0.04 (fills) |
| :-- | --: | --: | --: |
| 06-06 | −36.62 | 0.00 (0) | 0.00 (0) |
| 06-07 | −133.30 | −185.39 (49) | **+13.44 (23)** |
| 06-08 | 0.00 | +20.00 (3) | +20.00 (3) |
| 06-09 | +6.63 | −8.86 (15) | **+34.86 (4)** |
| 06-10 | +2.50 | +10.00 (5) | +10.00 (5) |
| **total** | **−160.79** | **−163.25 / 72** | **+78.30 / 35** |

**+$241 swing, churn 72→35 fills, zero day regressed.** The gate is conditional (fires only on
wide books) so clean/tight days are bit-identical — low overfit risk despite the thin 5-day sample.

## Recommendation
Promote to `config/strategy.yaml` `theta_overrides.priceBucket` (HL v31), keeping everything else:
```yaml
        exit_spread_hold: 0.04   # SHR-102: hold favorite to settle when held half-spread > 0.04
```
Keep `exit_safety_d: 1.0` (the gate suppresses it only on wide books; on tight books mid-hold exit
still works). Leave `entry_spread_gate` off (inert here).

## Caveats
- 5 settlement days; the win is concentrated on 2 (06-07, 06-09). Mechanism is sound and clean days
  are untouched, but re-validate at the engine's real cadence before deploy (per CLAUDE.md).
- The retuned sim now diverges from live in the *good* direction on 06-07 (sim +13 vs live −133)
  because live actually doom-looped — i.e. this gate is what stops live from losing there.
- Sweep driver: `tools/_bucket_retune_sweep.py`.

## Full-history validation (all 35 recorded settlement days, 2026-05-07..06-10)
Ran baseline vs exit_spread_hold=0.04 on EVERY recorded bucket day (`tools/_bucket_retune_fullsweep.py`,
main-worktree corpus). **Result: zero regressions.**

- Total sim PnL: **+591.84 → +921.76 (+$329.92)**; total fills 1041 → 941.
- **Inert on 30/35 days** (bit-identical — gate only fires on wide books).
- Helps on 5 wide-book days: 06-07 (−185.39→+13.44), 06-02 (−50.14→+35.88),
  06-09 (−8.86→+34.86), 05-28 (−89.44→−88.09), 05-24 (less churn, same PnL).
- **Leaves profitable liquid-day churn alone** (05-29 197 fills/+59, 05-19 164/+11 unchanged) —
  it distinguishes good theta churn from doom-loop spread-crossing. No day is hurt.

**Verdict: promote `exit_spread_hold: 0.04` to the live priceBucket config.** Strictly non-negative
across 35 days, conditional (no overfit), and it removes the only real bucket money-loser.
