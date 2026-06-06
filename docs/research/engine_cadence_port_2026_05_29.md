# Engine cadence port — v3.7 T13 sub-minute reference sampling (2026-05-29)

Follow-up to the backtest land (commit 6361512). Makes the live engine's
reference-mark bucketing period **config-driven** off `vol_sampling_dt_seconds`,
so the eventual HL dt=60→5 flip (T13, see `v37_hl_1s_sampling_2026_05_28.md`) can
be gated behind paper-trade validation without train/serve skew.

**No live/production config was flipped to dt=5.** `config/strategy.yaml` still
carries `vol_sampling_dt_seconds: 60` on every slot. This PR is plumbing only;
zero behaviour change for existing configs (every default-60 path is bit-for-bit
unchanged, verified by `test_registration_default_60s_preserves_behavior` and the
full 619-test suite).

## The bug being closed (train/serve skew)

The strategy's σ formula annualizes the last N log-returns assuming they are
spaced `vol_sampling_dt_seconds` apart. The **backtest** loader was already
parameterized (`reference_resample_seconds`, `hl_hip4.py`) to bucket reference
ticks at that period. The **live** engine, however, hardcoded the 60s assumption
in three places, none coupled to the strategy config:

1. `MarketState._mark_bucket_ns` — fixed `60 * 1e9`. The mark-bucketing equivalent
   of the backtest's `_resample_reference_*`.
2. `Scanner._required_returns_n` — `bars = (secs + 59) // 60`, i.e. the *count*
   of bars pulled per scan assumed 60s bars.
3. `MarketState` mark history — `deque(maxlen=256)`, fine for 60×60s bars but too
   short for sub-minute cadences (dt=5/3600s lookback needs ~720 bars), which
   would silently truncate the σ window vs the backtest's auto-growing ring buffer.

At dt=60 all three agree with the backtest (60 bars / 3600s window). If the live
strategy config were flipped to dt=5 while these stayed hardcoded, live σ would be
computed over a ~12× shorter, mis-spaced series — the exact skew this task exists
to prevent.

## What changed (HL)

Architecture constraint discovered first: **a single `MarketState` is shared
across all four live slots** (`runtime.py` `market_state` field → every slot's
`Scanner`). `vol_sampling_dt_seconds` is **per-slot** (lives in the `theta:` block;
`config.py`). HL slots read `reference_symbol="BTC"`, PM slots read `"BTCUSDT"`.

Coupling therefore had to be **per reference symbol**, not a single scalar:

- **`MarketState.set_reference_cadence(symbol, *, sampling_dt_seconds, lookback_seconds=None)`**
  registers a per-symbol bucket period (`_mark_bucket_ns_by_symbol`) and grows the
  per-symbol history deque (`_mark_history_by_symbol`) to `ceil(lookback/dt)+2`,
  never below the 256 default. Symbols with no override fall back to the 60s
  default → legacy behaviour preserved. `mark_bucket_ns_for(symbol)` exposes the
  effective period (used by the no-skew test).
  - **Conflict guard:** re-registering the *same* symbol at a *different* cadence
    raises. The one shared mark history for a symbol can only be bucketed one way,
    so two slots disagreeing is unsatisfiable — fail fast at startup beats silently
    skewing whichever slot loses.
- **`runtime.py`**: `reference_sampling_dt_seconds(cfg)` (theta → `theta.vol_sampling_dt_seconds`,
  else 60) and `reference_vol_lookback_seconds(cfg)` are the single source of truth.
  `_register_reference_cadences(slots)` runs in `run()` right after slots are built
  and before the ingest loop streams marks, registering each slot's
  `(reference_symbol → dt, lookback)` on the shared MarketState.
- **`scanner.py`**: `_required_returns_n` divisor now uses the slot's effective dt
  (`(secs + dt - 1)//dt`) instead of the hardcoded `//60`. Identical at dt=60.
- **`replay.py`**: `ReplayRunner` takes `sampling_dt_seconds=60`, registers it on its
  MarketState, and scales its returns request to hold the same wall-clock window
  (32 bars at 60s → 384 at 5s). Replay/paper validation now mirrors live cadence.

### ⚠️ Within-HL constraint that gates the flip (one line, human-gated)

Both **v1** (late_resolution) and **v31** (theta_harvester) read the same
`reference_symbol="BTC"`. They share one bucketed mark history. Flipping **only**
v31's `theta.vol_sampling_dt_seconds: 60 → 5` will make the engine **refuse to
start** (conflict guard) because v1 still assumes 60s on the same series.

To do the T13 flip, the operator must EITHER:
- move v1 and v31 to dt=5 in lockstep (v1's late_resolution σ at dt=5 is *not*
  validated — would need its own backtest), OR
- give v31 a distinct reference symbol so the two histories are independent.

This is intentional: it surfaces the shared-feed coupling instead of silently
skewing v1. **The actual `60 → 5` edit in `config/strategy.yaml` is the separate,
human-gated change** that must follow a ≥1-week paper-trade validation (below).

## PM gap analysis (investigate-first)

- **`vol_sampling_dt_seconds` is per-slot, NOT global.** It lives in each slot's
  `theta:` block. HL and PM are independently settable today. Because HL and PM use
  **different reference symbols** (`BTC` vs `BTCUSDT`), the per-symbol coupling makes
  them fully independent: flipping HL's BTC to 5s leaves PM's BTCUSDT at 60s
  (`test_hl_pm_independent`).
- **PM live DOES use MarketState bucketing for its reference feed.** `PMClient` is
  execution-only; the PM slots' σ/p_model reference is a Binance perp mark
  (`reference_symbol="BTCUSDT"`, emitted by the Binance adapter as `MarkEvent`),
  consumed through the same shared MarketState. So the **resample-period** concept
  applies to PM live and is now config-driven by the same mechanism — register
  `BTCUSDT → pm_dt`. No PM-specific plumbing was faked.
- **What PM live still lacks (the real gap):** the backtest's PM `reference_source`
  knob (`"klines"` default vs `"binance_bbo"` in `PolymarketDataSource`) selects the
  *source feed* for the reference, a separate axis from the resample *period*. In
  live, the reference source is fixed by whichever subscription/adapter feeds
  `BTCUSDT` marks; there is no config switch to choose klines-vs-BBO as the live
  reference. That feed-source selection is **out of scope here** and remains a gap
  to close when/if a sub-minute PM cadence is pursued — flagged, not implemented.

## Paper-trade validation step (gates the live flip)

1. **This PR (plumbing).** Merged with all slots at dt=60. Behaviour unchanged.
2. **Paper validation.** In a paper-mode config, set the HL theta slot(s) reading
   `BTC` to `vol_sampling_dt_seconds: 5` (in lockstep per the constraint above) and
   run ≥1 week. Confirm: engine starts (no conflict), live σ/p_model track the
   backtest's dt=5 series, no `STALE DATA HALT` regressions, gate-decision log
   shows entries consistent with the T13 backtest (+$49 vs 60s).
3. **Live cutover.** Only after (2): flip the production `config/strategy.yaml` dt
   to 5 for the validated HL slot(s) — one clearly-labelled line, human-approved.

## Tests

- `tests/unit/test_market_state.py` (+4): default 60s; per-symbol bucketing
  (HL/PM independence); conflict guard; sub-minute history sizing.
- `tests/unit/test_engine_runtime_cadence.py` (new): helper defaults; **no
  train/serve skew** (bucket period == strategy's assumed period per symbol);
  default-60 preservation; HL/PM independence; conflict raises at registration.
- Full suite: **619 unit/integration + 2 perf pass.**

## Safety gates

No defensive gate touched (`min_bid_notional`, `stop_loss`, `stale_data_halt`,
etc. all intact) — per `feedback_keep_safety_gates`.
