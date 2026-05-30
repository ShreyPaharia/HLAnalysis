# Engine: per-bucket OHLC + per-symbol σ source (dormant-Parkinson fix, PM bbo unblock)

Date: 2026-05-31
Branch: `feat/engine-bbo-sigma-source-parkinson-fix`
Status: **engine capability + tests only. No `config/strategy.yaml` change. No deploy. Human-gated.**

## What was broken (verified in code, not just claimed)

### 1. Dormant Parkinson (train/serve bug)
- `LateResolutionStrategy._sigma` (and theta's) only runs Parkinson when
  `recent_hl_bars` is non-empty; otherwise it **falls back to stdev**
  (`hlanalysis/strategy/late_resolution.py:357`).
- The **backtest** MarketState supplies HL bars and the runner passes them
  (`backtest/runner/market_state.py:recent_hl_bars`, `hftbt_runner`).
- The **live** path did not: `Scanner.scan` never passed `recent_hl_bars`
  (`engine/scanner.py:188`), and the live `MarketState` stored **close only**
  (`_marks: deque[float]`) with no H/L tracking.
- Net: the **HL v1 (late_resolution)** slot — the only one configured
  `vol_estimator: parkinson` (`config/strategy.yaml`, `paper_mode: false`) — was
  **silently running stdev in production**. (v31 / v31_pm are `theta_harvester`,
  whose `_sigma` uses return-based bipower/sample_std and **ignores**
  `recent_hl_bars`; they are unaffected.) The backtests that picked Parkinson
  were measuring a σ the live engine never computed — a real train/serve skew.

### 2. PM σ feed too sparse for dt=5
- PM slots read σ from the Binance perp **mark**, which `adapters/binance.py`
  REST-polls every ~3s (no `mark` entry in `_STREAMS_PERP`). At dt=5 that's
  ~1.6 points/bar → High≈Low≈Close → degenerate σ.
- The dense BBO (`bookTicker`, sub-second) the backtest used **is already
  subscribed live** — it just wasn't used for σ. So this is a re-point, not a
  data-availability problem (confirms `pm_bbo_dt5_verdict`).

## What changed (engine capability)

### A. Per-bucket OHLC + `recent_hl_bars` in the live `MarketState`
- `_marks` now stores per-bucket `(high, low, close)` bars instead of close
  only. Within a bucket: `high=max`, `low=min`, `close=last`; a new bucket
  appends a fresh bar. Per-symbol cadence / history-sizing / conflict-guard
  preserved.
- New `recent_hl_bars(symbol, n) -> tuple[(high, low), ...]` mirrors the
  backtest MarketState (KlineRingBuffer rows `[high, low]`). `(ln(H/L))²` is
  order-invariant, so σ is unaffected by the row order; the order matches the
  backtest for clarity.
- `recent_returns` is **bit-identical** (still close-to-close; the per-bucket
  close is still the last tick). Verified by the existing bucketing tests
  staying green.
- `Scanner` (and `ReplayRunner`) now thread
  `recent_hl_bars=ms.recent_hl_bars(ref_symbol, n=…)` into `strategy.evaluate`.

### B. Per-symbol σ source: `mark` | `bbo`
- New `StrategyConfig.reference_sigma_source: Literal["mark","bbo"] = "mark"`.
- `MarketState.set_reference_source(symbol, source)` + `reference_source_for`,
  with the same fail-fast conflict guard as the cadence (slots sharing a
  reference symbol must agree — one shared OHLC history).
- When `source="bbo"`, the `BboEvent` mid `=(bid+ask)/2` feeds the **same**
  per-bucket OHLC machinery and `last_mark` (so the strategy's reference price S
  is consistent with its σ source). A stray `MarkEvent` for a bbo-sourced symbol
  is ignored. When `source="mark"` (HL) behaviour is unchanged except that marks
  now also produce H/L (min/max within the bucket) — which is what activates
  Parkinson for HL.
- `runtime._register_reference_cadences` now also registers the source.

## Parity / no-skew evidence (tests)
- **bbo-mid OHLC parity** (`test_bbo_mid_ohlc_matches_load_binance_bbo_reference`):
  writes a recorded Binance perp BBO parquet, runs the **real backtest**
  `_load_binance_bbo_reference`, and asserts the live `MarketState`'s
  `recent_hl_bars` and close-to-close returns match **bar-for-bar** on the same
  tick sequence at dt=5. Same OHLC inputs ⇒ identical σ (the estimator is
  deterministic) ⇒ no train/serve skew.
- **Dormant-Parkinson regression** (`test_parkinson_differs_from_stdev_on_live_scanner_path`):
  on the live Scanner path, with intra-bucket H/L range and flat closes, a
  `vol_estimator: parkinson` slot now **HOLDs** (σ from range exceeds `vol_max`)
  while `stdev` **ENTERs** — proving Parkinson is actually live. Flat series →
  the two agree (`test_parkinson_equals_stdev_on_flat_series`).
- **Threading** (`test_scanner_threads_recent_hl_bars_into_evaluate`): the
  Scanner passes `recent_hl_bars` equal to `ms.recent_hl_bars(...)`, each bar
  with `high > low`.
- **Replay bbo path** (`test_replay_bbo_sourced_reference_drives_sigma_from_mid`):
  `ReplayRunner(reference_sigma_source="bbo")` sources σ from the BTCUSDT BBO mid.
- Full suite: **655 passed**.

## Default-behaviour guarantee
- No `config/strategy.yaml` change. With `source=mark` (default) and a
  `stdev` slot, behaviour is bit-identical **except** the intended H/L
  threading. HL stays `mark`; PM (`BTCUSDT`) is the only symbol that *can* opt
  into `bbo`, and nothing in this change flips it.
- Safety gates untouched (`min_bid_notional`, `stop_loss`, `stale_data_halt`,
  depth-walk slippage).

## Paper-validation (2026-05-31) — replay through the real engine path
`scripts/paper_validate_v1_parkinson.py` replays the **recorded HL corpus**
(perp BTC mark + HIP-4 leg bbo/trade) through `ReplayRunner` → the **live engine
`MarketState`** (the code changed here) with the **exact live v1 config**
(parkinson, dt=5, min_safety_d=3.0, λ=0.97, use_bid_gate), parkinson vs stdev.
Across 7 settled BTC binaries (expiries 2026-05-24→05-30, ~250k ticks):
- **Parkinson is genuinely active**: `σ_parkinson ≠ σ_stdev` every question
  (ratio **0.52–0.59×**) — bit-identical under the old dormant bug, so this
  confirms the fix flows live. dt=5 bars carry real range (~11 marks/5s; 720
  bars = full 3600s lookback, deque grew → no train/serve truncation).
- **Decisions diverge as expected**: lower Parkinson σ → larger safety_d → fewer
  `safety_d_below_min` blocks → more entry-eligible ticks (consistent with the
  backtest's higher trade count / +$48). No crashes end-to-end; full suite green.

## Operator decision — RESOLVED 2026-05-31
- **Merge to `main` + deploy LIVE, v1 kept on Parkinson** (operator: "the better
  version"). No `config/strategy.yaml` change. v1 stays `paper_mode: false` +
  `vol_estimator: parkinson` + `dt=5`; deploying this engine code is the live
  Parkinson cutover for HL v1. (v31 / v31_pm are theta → ignore H/L bars →
  unaffected. v1_pm is stdev → unaffected.)
- **Still open (separate, human-gated): enable PM bbo source for dt=5.** To get
  the dt=5 PM win (`pm_bbo_dt5_verdict`: v1_pm $358, theta clean-Pareto $229),
  set `reference_sigma_source: bbo` on **both** PM slots (v1_pm + v31_pm — they
  share the BTCUSDT feed, so the engine conflict-guards a mismatch) and flip
  their `vol_sampling_dt_seconds` to 5 in lockstep. Code supports it; nothing
  flipped here.
