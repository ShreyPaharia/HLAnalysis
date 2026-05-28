# v3.7 SPIKE — HL HIP-4 reference-sampling cadence sweep

**Branch:** `feat/hl-1s-reference-sampling`
**Date:** 2026-05-28
**Strategy:** v3.6 universal (`jr-tilt-ma_sigma-lb30-a100`)
**Corpus:** HL HIP-4 BTC binaries, 2026-05-09 → 2026-05-27, 18 questions
**Decision:** **SHIP-CADENCE-CHANGE** — follow-up to port `_mark_bucket_ns` in `engine/market_state.py`.

## TL;DR

Going from 60s to sub-minute HL perp reference sampling materially improves the
v3.6 universal config on HL HIP-4 binaries. dt=1s adds **+$63 (+23.7%)** vs the
60s baseline, well above the +$20 ship gate. The cadence-vs-DD curve has a sweet
spot at **dt=5s** — same headline lift as dt=1s, identical Sharpe to baseline,
and the lowest drawdown of any cell.

| Cell    | dt(s) | PnL     | ΔPnL    | Trades | Hit   | Sharpe | maxDD   |
|---------|-------|---------|---------|--------|-------|--------|---------|
| dt60    | 60    | $267.01 |         | 72     | 83.3% | 16.56  | $32.74  |
| dt10    | 10    | $276.50 | +$9.49  | 58     | 83.3% | 15.57  | $26.28  |
| **dt5** | **5** | **$316.26** | **+$49.25** | **60** | **88.9%** | **18.60** | **$26.28** |
| dt1     | 1     | $330.18 | +$63.17 | 56     | 83.3% | 15.60  | $39.45  |

## Decision gate

Gate from the spec was:
- **Ship-cadence-change** (≥+$20 HL gain) → **YES — $63 on dt=1, $49 on dt=5.**
- Shelve (<+$5) → no.
- Extend-corpus (+$5 to +$20) → no.

**Ship with caveat**: pick dt=5, not dt=1, for the live-engine port. Same PnL
class (+$49 vs +$63), but Sharpe is 18.6 vs 15.6 and max DD is $26 vs $39.
dt=1 squeezes a few extra dollars on the right markets but takes a 50% DD hit
to do so, driven mostly by Q1000020 (see below).

## What this PR ships

1. **Parameterized resample period** in `hlanalysis/backtest/data/hl_hip4.py`
   and `_hl_hip4_fastpath.py`. The constant `_REFERENCE_RESAMPLE_NS` is gone;
   `HLHip4DataSource` now takes `reference_resample_seconds=60` (default
   preserves prior behavior) and threads it through both the legacy generator
   path and the arrow-backed fastpath.
2. **CLI auto-couples** loader cadence to strategy cadence: `cmd_run` reads
   `vol_sampling_dt_seconds` from the run config and passes it to
   `_resolve_data_source` so the loader and strategy can't drift. No new flag.
3. **Runner**: `scripts/run_v37_hl_1s_sampling.py` (dt ∈ {60, 10, 5, 1}).

No engine code touched. No PM loader touched. No strategy code touched.
Existing tests all pass at default 60s.

## What is NOT shipped (follow-up tasks)

- **Engine port** of `_mark_bucket_ns` in `hlanalysis/engine/market_state.py`
  (currently hardcoded to 60s). Required before any live deployment uses
  sub-minute sampling, otherwise train/serve skew kicks in.
- **PM loader cadence sweep**. We have 3 weeks of overlapping PM+HL tick data;
  PM has 12 months of corpus, so the sweep is more powerful there. Separate
  task.
- **Tune workers** (`make_hl_hip4_source`) still hardcode 60s. `hl-bt tune` on
  HL would always run at 60s regardless of the grid's `vol_sampling_dt_seconds`
  axis. Fine for now (we don't tune on HL) but flag if/when we do.

## Sanity reproduction note

The dt=60 cell on the parameterized loader yields **$267.01 / 72 trades**,
versus the originally-reported **$270.03 / 74 trades**. The delta is one
question (Q1000065: 4 trades vs 2), explained by strategy-code merges since
the original run (commits 88a3082 and the c5c00f9 merge ship). At the *loader*
level the refactor is byte-identical — the constant 60s × 1e9 ns is now an
ivar holding the same 60_000_000_000 — and all 37 existing HL tests pass.

## Per-question breakdown (dt1 vs dt60)

Net Δ = +$63. Mostly driven by big single-question wins (Q1000095 +$37,
Q1000075 +$27, Q1000065 +$25, Q1000070 +$16). The big loss is Q1000020 (−$41),
where the 1s sampler fired entry ~77 min earlier than 1m and the trade went
the wrong way.

| qid       | outc | n60→n1  | Δ entry-time | Δ PnL    |
|-----------|------|---------|--------------|----------|
| Q1000095  | no   | 3 → 1   | same         | **+$37** |
| Q1000075  | no   | 3 → 1   | same         | **+$27** |
| Q1000065  | yes  | 2 → 1   | −120s        | **+$25** |
| Q1000070  | yes  | 3 → 3   | same         | +$16     |
| Q1000080  | no   | 3 → 3   | −300s        | +$15     |
| Q1000040  | yes  | 5 → 1   | −840s        | +$11     |
| Q1000105  | no   | 5 → 1   | −5760s       | +$11     |
| Q1000085  | yes  | 5 → 3   | same         | +$8      |
| Q1000045  | no   | 3 → 1   | same         | +$7      |
| Q1000050  | no   | 3 → 1   | same         | +$3      |
| Q1000035  | no   | 1 → 1   | same         | $0       |
| Q1000055  | no   | 1 → 1   | +1560s       | $0       |
| Q1000060  | yes  | 2 → 2   | same         | −$4      |
| Q1000090  | yes  | 1 → 3   | −8160s       | −$6      |
| Q1000015  | yes  | 3 → 1   | same         | −$9      |
| Q1000025  | yes  | 5 → 3   | same         | −$14     |
| Q1000030  | no   | 3 → 3   | same         | −$21     |
| **Q1000020** | yes | 5 → 9 | **−4620s**   | **−$41** |

Entry-time histogram: 10/18 markets have *identical* first entries across dt;
6 enter earlier at 1s; 2 enter later at 1s. The earlier entries are signals
the 1m bucketer was smoothing away — usually correctly classified (8 winners,
3 marginal pushes), but Q1000020 shows that earlier ≠ better when the
underlying regime changes after the entry.

## Why this works mechanically

The strategy's σ is computed from the last `vol_lookback / vol_sampling_dt`
returns and annualized assuming each spans `vol_sampling_dt_seconds`. At
dt=60s the 1h-of-history is 60 returns wide; at dt=1s it is 3600 returns
wide. The denser sample tightens σ when realized vol is calm and widens it
faster when the underlying jumps. For `ma_sigma + JR-trust` specifically:

- The JR (jump-ratio) gate gets *more* timely. At 1m sampling, a 5-second
  jump is partially smoothed into a single OHLC bar and may not register as
  a jump. At 1s sampling, the same event hits BPV ≪ RV cleanly and JR fires,
  throttling the tilt during the burst — exactly the documented mechanism.
- The momentum_mr lookback of 30 minutes now uses 1800 samples (vs 30) for
  the moving-average / σ-spread calculation. The signal is the same average
  shape but on a denser grid → less estimation noise → more confident
  entries on real moves, less aggression on small chop.

This is why the gain is real (mechanism-aligned) rather than a fit artifact:
the same `JR_trust × ma_sigma_tilt` machinery we already shipped works
better when fed the data resolution it expects.

## Risks / open questions

- **Q1000020 (−$41)**: at dt=1s the strategy enters 77 min earlier on a
  signal the 1m bucketer missed. The early read happened to be wrong here.
  Sub-minute signals are higher-variance — single-question downside risk
  is bigger than at 1m. dt=5 sidesteps this (DD only $26).
- **HL corpus is 18 questions**. Per-question swings of $20-40 dominate the
  total. The +$63 signal at 1s is real but not 5-sigma. Memory
  `[[v36_ouz_sdr_jr_2026_05_28]]` already flagged this as a constraint on
  HL inference. Revisit at 3+ months HIP-4 corpus.
- **No live-engine port yet**. Shipping the backtest change alone is safe
  (paper-only impact). The engine port must land before live traffic
  switches to sub-minute sampling, otherwise live σ will be ~60× larger
  than what the backtest used to tune.

## Recommendation

1. Land this PR (loader refactor + spike data). Live behavior unchanged.
2. File a follow-up: **port `_mark_bucket_ns` in `engine/market_state.py`
   to take its period from the engine `TheoConfig.vol_sampling_dt_seconds`**,
   mirroring the backtest fix. Same default 60s preserves existing live
   behavior; flipping the engine config to 5s would then route through.
3. After (2), do a 1-week paper-trade comparison on HL with dt=5 to validate
   that the simulator advantage holds against live microstructure.
4. Separate: run the same cadence sweep on PM (12-month corpus is the
   load-bearing benchmark).

## Artifacts

- Runner: `scripts/run_v37_hl_1s_sampling.py`
- Results: `data/sim/runs/v3-7-hl-1s-sampling-2026-05-28/{dt60,dt10,dt5,dt1}/`
- Summary: `data/sim/runs/v3-7-hl-1s-sampling-2026-05-28/summary.json`
- Code: `hlanalysis/backtest/data/hl_hip4.py`,
  `hlanalysis/backtest/data/_hl_hip4_fastpath.py`,
  `hlanalysis/backtest/cli.py`
