# Sim-fidelity program — Spec 2 (unify) + Spec 3 (execution) implementation plan

**Date:** 2026-06-08  **Base:** main `d9a5eb5` (Spec 1 merged: SHR-78/79/80/56/57)
**Status:** plan → Linear tickets → parallel octomux tasks

## Why / context

Investigation (daily-report → live-vs-sim on HL buckets) established that, once the
comparison window is matched, the sim's *logic* agrees with live; the residual gaps are
**(a) input skew** between two MarketState implementations and **(b) execution** (real venue
vs simulated fills). Spec 1 made sim execution realistic (partial fills, latency knob, fees,
cadence). This plan covers the durable fixes:

- **Spec 2 — unify the shared core** (eliminate input/accounting skew by *code reuse*, not a
  diff harness). Normalize at the edges; share everything between.
- **Spec 3 — execution fidelity** (calibrate the un-shareable execution edge from a live
  trade journal: measured latency, reject/re-submit churn, state/halt replay).

Architecture (two thin context-specific edges, one shared core):

```
data-in edge        SHARED CORE (identical)             execution edge
WS/parquet → NormalizedEvent → MarketState → evaluate() → Decision → venue/sim Fill
                                                          Fill → SHARED accounting (pos/PnL/MTM)
```

## Work breakdown (dependency-ordered; Wave 1 is parallelizable now)

### Wave 1 — independent, parallelizable (own modules, no cross-conflict)

- **T1 — Shared MarketState core.** New `hlanalysis/marketdata/market_state.py`: the unified,
  pure event-driven core — L2 book, `recent_returns`, `recent_hl_bars`, σ estimators
  (parkinson/stdev/bipower), `recent_volume_usd` (1h window) — consuming a common event type.
  Self-contained + unit tests. Does NOT rewire engine/backtest yet (Wave 2). Foundational for
  Spec 2 (SHR-73).
- **T2 — Shared PnL/position accounting.** Extend `hlanalysis/marketdata/position_math.py`
  (already has `PositionState`/`apply_fill`) to the full pipeline: realized PnL on reduces,
  settlement payout, open-position MTM — with an optional **venue-`closedPnl` override flag**
  (live HL trusts venue truth; sim computes). Self-contained + tests. Independent of T1.
- **T3 — Live trade journal (engine write-path).** Per decision/order, emit a durable journal
  row: `decision_ts` + book-at-decision (top-N) + the evaluate() inputs (σ/returns/volume) +
  the `Decision`, then `send_ts`, `fill_ts`, fill px/sz, reject reason, and halt state. ONE
  journal serves decision-parity, latency calibration, AND halt replay. New module + hooks in
  scanner/router/runtime. Independent of T1/T2.
- **T4 — Recorder completeness / seq-gap reconciliation tool.** A tool that checks recorded vs
  live event coverage (per-symbol seq gaps, message counts, time gaps) so we *know* the
  recorded feed the sim replays is complete (the open v1 volume-source question). `tools/` +
  tests. Independent.
- **T5 — Sim state/halt replay + caps.** Backtest component that reads the engine halt/event
  log and (a) suppresses sim entries during recorded halt windows, (b) enforces the
  `daily_loss_cap_usd` slot-halt and inventory caps within the sim window. Reads existing
  engine logs; backtest-side. Mostly independent.

### Wave 2 — depends on Wave 1 (ticketed, NOT spawned until deps land)

- **T6 — Route backtest MarketState → shared core** (dep T1) + parity unit test.
- **T7 — Route engine MarketState → shared core + bit-identical replay gate** (dep T1; RISKY
  live money path — validate engine σ/volume/book bit-identical pre/post on recorded corpus
  before any deploy).
- **T8 — Route engine + backtest PnL accounting → shared `position_math`** (dep T2; live keeps
  the venue-`closedPnl` reconcile layer on top).
- **T9 — Sim reject/re-submit + measured-latency model** (dep T3 journal data + Spec 1): model
  IOC rejects when the limit is no longer marketable + re-fire; replace the constant 50ms with
  the measured δ distribution; fill on `book(decision_ts + δ)`.
- **T10 — Standing sim-vs-live validation pipeline** (dep T3/T8): per-market fill-truth
  reconciliation tracked over time, residual attributed to input/execution/halt.

## Cross-cutting guards (fold into the touching ticket)
- Coin-recycling: leg-book reads must be window-bounded (regression test). [T6]
- Settlement uses venue `settled_side_idx`, never a YES re-derivation. [T8/verify]
- Determinism: no process-salted hashes on the HL path. [T1/T6]

## Testing & acceptance
- Each Wave-1 ticket: TDD, full unit suite green (currently 1239), PM path unaffected.
- Spec 2 done when engine + backtest both run through the shared core/accounting and the
  engine output is **bit-identical** pre/post (T7 gate); the parity test is then the standing
  regression gate (replaces the dual-builder harness).
- Spec 3 done when the sim's fill count/size + reject churn + halt-suppressed windows match
  live on the matched markets (the 7th-market case: close the −133-vs-−65 execution residual
  to the irreducible chaotic floor).

## Mapping to existing Linear epics
- Spec 2 (T1/T2/T6/T7/T8) → **SHR-37** (duplication / train-serve skew); T1+T6+T7 realize **SHR-73**.
- Spec 3 (T3/T4/T5/T9/T10) → **SHR-33** (backtest fidelity).
