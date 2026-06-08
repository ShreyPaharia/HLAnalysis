# HL HIP-4 backtest execution fidelity — design (Spec 1 of 2)

**Date:** 2026-06-08
**Status:** approved (brainstorming) → ready for implementation plan
**Author:** desk (via Claude)

## Why

Debugging the daily-report bucket result (v31 HL buckets **−$274.93** live, v1 **+$194.10**)
surfaced that the HL HIP-4 backtest does not match live, and the gap is dominated by
**execution-model fidelity**, not strategy logic. Concrete evidence (2026-06-08):

- **v31 full-corpus sim +$597.66 vs live −$274.93** — partly a period-comparison artifact
  (sim ran 31 questions over a month; the live C3 config is 2 days old), partly execution
  idealisation. Scoped to the matched C3 window (06-06→08): **sim −$216 vs live −$170** —
  same sign/magnitude, so the strategy logic is broadly right once windowed.
- **v1 HL bucket sim = 0 trades** vs live **+$51.35 / 26 fills** — root-caused to a hard bug:
  the runner passes `recent_volume_usd=0.0`, so v1's `min_recent_volume_usd=100` gate vetoes
  every entry. Disabling the gate → v1 trades **+$26.12 / 10 trades**.
- **Fill realism:** live v31 buckets churn **307 fills (median $45)** / **60 fills (median
  $17.6) in the C3 window**; the sim does clean full-size round-trips (**median notional $507
  = the $500 cap**). The sim has no partial-fill / queue / latency model on thin HL books.

Empirical inputs measured from prod (2026-06-08, read-only):

- **HL HIP-4 fees = $0** across all 767 live fills (v1 217, v31 550) — 0 bps. Fees are a
  non-factor for HL; the `pm_binary` curve is separate (out of scope).
- **HL order/network latency ≈ 46 ms median** (33–160, mean 58) from the engine box. Order
  latency is a small ~50 ms knob; the dominant timing effect is scan cadence + σ resample.

This is **Spec 1 of 2**. Spec 2 (separate) is a replay/decision-parity harness. Fill-by-fill
1:1 is **not** an achievable target for a re-deciding backtest (execution↔decision is a closed
loop; a single fill difference cascades — 06-07: live did 4 cycles on `#1670`, sim did 1). The
goal here is an **unbiased estimator**: realistic execution that removes the systematic
over-fill / wrong-exit-timing bias so the predictive backtest is trustworthy for tuning.

## Scope

In: HL HIP-4 backtest path — `hlanalysis/backtest/runner/hftbt_runner.py`,
`hlanalysis/backtest/data/hl_hip4.py`, `hlanalysis/backtest/cli.py`.

Out: PM / NBA execution model (own book/fill semantics); the Spec-2 replay-parity harness;
any strategy/engine behaviour change; any retuning. **PM must not regress** — shared-runner
changes are guarded to HL assets and the PM test suite stays green.

## Approach

**A — native hftbacktest partial-fill + latency** (chosen over a custom fill model). Lean on
the proven sim core; minimal custom code. Fall back to a thin custom IOC book-walk only if
hftbacktest's partial-IOC semantics can't reproduce the live fill-size distribution.

## Components (each maps to a Linear ticket under SHR-33)

### 1. Partial fills vs real depth — SHR-79
- `hftbt_runner.py` `_build_asset` / `_build_hedge_asset`: replace `.no_partial_fill_exchange()`
  with the partial-fill exchange model for HL assets.
- Line ~302 (`exec_qty = min(exec_qty, intent_size, book_depth_assumption)`): drop the
  `book_depth_assumption=10_000` default cap. `exec_qty` already comes from hftbacktest's
  depth-limited fill; with partial fills enabled it reflects real available depth at/through
  the IOC limit. IOC remainder cancels (existing IOC semantics).
- `--depth` becomes an *optional* explicit cap; default = unlimited (the real recorded book
  governs fill size). A deep order walks multiple levels → realised fill price worse than the
  touch (natural slippage).
- **Effect:** sim fill sizes track real top-of-book depth; no more uniform full-$500 fills.

### 2. Order latency — new knob
- Add `order_latency_ms` to `RunConfig` and `--order-latency-ms` to the CLI (default **50**).
- Configure hftbacktest's constant order-latency model (entry + response).
- **Effect:** an order decided on the book at T acts on the book at ~T+50 ms (book may have
  moved → different/again-partial fill).

### 3. Real `recent_volume_usd` — SHR-78
- `run_one_question` (≈ line 814): replace the literal `recent_volume_usd=0.0` with a real
  rolling-window traded notional, summed from the recorded per-leg `trade` events the runner
  already ingests, over the window the **live engine** uses (confirm the engine's window;
  match it). Add the accumulator to `MarketState` (alongside `recent_returns_and_hl`) or
  compute it inline.
- **Effect:** unblocks v1 and any future volume-gated strategy; removes the silent asymmetry
  (theta is unaffected today only because it doesn't gate on volume).

### 4. σ cadence — SHR-80
- `cli.py:_source_config_from_args`: derive `reference_resample_seconds` from the config
  JSON's `vol_sampling_dt_seconds` instead of hardcoding 60; carry it in `SourceConfig` so
  spawn workers don't drift via env. Add an assert that resample == `vol_sampling_dt` (the
  `hl_hip4.py` docstring already states they must match).
- Per-class dt (bucket dt=2 vs binary dt=5): a single `hl-bt run` uses one cadence; run
  buckets and binary as separate invocations. Document this.
- **Effect:** the sim evaluates σ / safety_d / p_model at the live cadence, so exit timing
  (the 06-07 driver) matches live far more closely.

### 5. Fees — SHR-57 (HL)
- HL fee = 0 is empirically correct; keep `fee_taker=0.0` default for HL but make it explicit
  and **log the effective fee** at run start (kill the silent-default aspect). `--fee-taker` /
  `--fee-model` overrides remain. PM's `pm_binary` curve untouched.

### 6. slippage_bps — SHR-56
- With the book-walk modelling execution slippage, repurpose `--slippage-bps` as an *explicit
  additive haircut* applied to the realised fill price (so it's no longer a no-op), default 0.

## Data flow

recorded `book_snapshot` + `trade` → hftbacktest depth (partial-fill + ~50 ms latency) +
`MarketState` (returns / HL bars / **volume**) → `strategy.evaluate(...)` → IOC submitted with
latency → fills walk the real book → `fills.parquet` / `report.md`.

## Error handling / safety

- New behaviour is the **default for HL** (the point of the spec), but each knob is config-
  surfaced so a run is reproducible. `--no-partial` / legacy escape hatch optional.
- Partial-fill flip is applied to **HL assets only**; PM build path unchanged. Verify PM unit
  tests unaffected.
- Cadence-mismatch is a loud assert, not a silent alias to 60.

## Testing (TDD — write tests first)

- **Partial fill:** book with top-of-book size < order size → fill == available, remainder
  cancelled (IOC).
- **Multi-level walk:** order spanning several levels → VWAP fill price strictly worse than
  the touch (slippage emerges from the book).
- **Latency:** order placed at T acts on the book at T+`order_latency_ms`; a moved book gives
  a different fill than the decision-time snapshot.
- **SHR-78 regression:** a v1 HL bucket run over a corpus where the live engine traded
  produces > 0 entries (today: 0).
- **Cadence:** `reference_resample_seconds` is derived from `vol_sampling_dt_seconds`; a
  mismatch raises.
- **PM regression:** existing PM/NBA tests unchanged; full unit suite (currently 1207) green.

## Validation (acceptance bar: bias gone + realism, on matched windows)

Re-run with the new model and capture in the run report:

- **v31 buckets** from the exact C3 deploy cutoff (2026-06-06 11:42 UTC) → 06-08.
- **v1 buckets** 2026-05-31 → 06-08.

Pass criteria:
1. Sim **fill-size distribution and fill count** in the live ballpark (no uniform full-$500
   fills; partials present).
2. Latency (~50 ms) and HL fee (=0, logged) applied.
3. Sim **aggregate PnL materially closer to live** than today, with the residual explicitly
   attributed to chaotic closed-loop divergence (and bounded — sign + rough magnitude agree).

## Out of scope

PM execution model; Spec-2 replay-parity harness; strategy/engine behaviour changes; retuning.

## References

- Linear: SHR-33 (epic), SHR-78 (volume hardcode), SHR-79 (full-size fill), SHR-80 (cadence),
  SHR-56 (slippage no-op), SHR-57 (HL fee default).
- Live-vs-sim evidence + artifacts: `data/sim/runs/v31_bucket_tailcheck`,
  `v31_bucket_c3window`, `v1_bucket_novol`; `config/backtest/v{1,31}_hl_bucket_*.json`.
- Memory: `bucket_overfit_rollback_c3_2026_06_06` (2026-06-08 update).
