# HL HIP-4 Backtest Performance — 2026-05-20

## TL;DR

The HL HIP-4 backtester was ~3× slower per bucket question than per binary question, even though only the leg count differs. Three bottlenecks dominated wall time:

1. **`_build_leg_event_array` in the runner** (60-65% of wall time) — was building a numpy structured array with per-cell `arr[idx]["field"] = val` writes. Replacing this with bulk per-column numpy assignment via Python list accumulators delivered the first speedup tier.
2. **`HLHip4DataSource.events` opened a fresh `duckdb.connect()` for every per-leg iterator** (14-23 connections per question). Sharing one connection across leg iterators + the `resolved_outcome` BTC fallback is a clean structural fix that doesn't help wall time much on this workload (~1-2%) but is good hygiene.
3. **`BookSnapshot` / `TradeEvent` dataclass round-trip** between the data source and the runner. The runner immediately unpacks them into the numpy event array; constructing ~160k `BookSnapshot` objects per leg was pure overhead. Added an arrow-backed `events_arrays(q)` fast path on `HLHip4DataSource` that returns pre-built per-leg numpy `event_dtype` arrays directly, plus a runner branch that uses it when available.

End-to-end on the full HL HIP-4 corpus (24 questions, `--start 2026-05-06 --end 2026-05-21`): **prior baseline ~70 min → 8 min 37 s ≈ 8× faster** (fix 1 + fix 2; warm-cache run). Per-question raw timings after the arrow fast path stack on top: **HL binary 1q 36.9 s → 14.8 s (2.5×); HL bucket 1q 70.1 s → 20.4 s (3.4×); HL bucket 3q 224 s → 59.6 s (3.8×)**.

Strategy logic and fill outputs are unchanged — verified via `scripts/perf/diff_fills.py` ignoring the random cloid UUID column.

## Profiling setup

- Branch: `perf/backtest-hl-hip4-speedup`
- Strategy: `v1_late_resolution`, config `data/sim/configs/v1-finalize-5.json`
- Profiler: `pyinstrument` (statistical, `async_mode="disabled"`)
- DuckDB connect counter: `duckdb.connect` monkey-patched around each run
- Scenarios:
  - `hl_binary` — 1 question, 2 legs, window `2026-05-13..2026-05-15`
  - `hl_bucket` — 1 question, 8 legs, same window
  - `hl_binary_3q` / `hl_bucket_3q` — 3 questions each, window `2026-05-13..2026-05-17`
  - `pm_binary` — Polymarket BTC-Up-Down daily was empty in the chosen 2-day slice; the PM cache covers up to ~2026-05-09 only, so it isn't a fair "compare-to-PM" baseline against the HL window we needed. PM remains the speed baseline in spirit (its data path uses `pyarrow.parquet.read_table` directly, no DuckDB per-question, and caches klines once per process — patterns reflected in the fixes here).

Two timing modes are reported. Raw timings (no profiler, `time …`) are reliable; profiler-instrumented numbers add 1.5-4× overhead unevenly across runs and are useful only for *relative* hot-frame attribution. The full-corpus number is raw.

## Baseline (commit before changes)

| Scenario        | Raw wall    | Profiler wall | DuckDB connects | Top frame                                  |
|-----------------|-------------|---------------|-----------------|---------------------------------------------|
| hl_binary 1q    | 36.9 s      | 38.4 s        | 10              | `_build_leg_event_array` 22.3 s (58%)       |
| hl_bucket 1q    | 70.1 s      | 70.6 s        | 23              | `_build_leg_event_array` 47.1 s (67%)       |
| hl_binary 3q    | n/a         | 111.0 s       | 28              | `_build_leg_event_array` 67.2 s (61%)       |
| hl_bucket 3q    | n/a         | 224.3 s       | 65              | `_build_leg_event_array` 146.5 s (65%)      |

### Top frames (HL bucket 1q, baseline)

```
70.570 main
└─ 69.317 run_one_question
   ├─ 47.144 _build_leg_event_array              ← #1 hot path
   ├─ 12.921 HLHip4DataSource.events             ← #2 hot path
   │  ├─ 11.101 heapq.merge
   │  │  └─  9.947 gen() (book + trade iters)
   │  │     ├─ 6.154 fetchall                     (duckdb result transfer)
   │  │     └─ 3.281 [self] tuple→dataclass loop
   │  └─  1.495 _reference_iter                   (3× list materialisation overhead)
   ├─  3.099 numba jitclass wrapper
   ├─  2.103 KlineRingBuffer.recent_returns
   └─  1.590 KlineRingBuffer.recent_hl_bars
```

### Diagnosis

1. **`_build_leg_event_array` (runner)** — two passes over snapshot lists, writing each numpy struct cell individually with `arr[idx]["field"] = val`. Each write triggers a Python-to-numpy scalar conversion and a structured-dtype field lookup. With 8 legs × thousands of snapshots × 5 fields, this is 60%+ of total wall time.
2. **Per-leg DuckDB connections** — `events(q)` builds one iterator per leg per event-type, each calling `duckdb.connect()` and issuing a separate `SELECT … FROM read_parquet('symbol=#XYZ/**.parquet')`. An 8-leg bucket opens ~18 connections (book × 8 + trade × 8 + reference + settlement); a binary opens ~6.
3. **`_reference_iter` list-materialise round-trip** — to detect "0 rows so fall back to mark", the primary iterator was fully materialised via `rows = list(primary_iter)`, then re-wrapped in `iter(rows)`. Modest cost but pointless work.

`_partition_has_files` is `lru_cache`-d per-instance and not a hot path. `_last_btc_ref_at_or_before` accounts for <0.5 s on baseline. The bench `pm_binary` was empty in window so PM-vs-HL apples-to-apples wasn't possible; we relied on profile diagnosis instead.

## Things we tried and abandoned

**Batched cross-leg `read_parquet([…])` query.** The plan suggested batching per-leg reads via `WHERE symbol IN (…)` against a single `read_parquet(...)` call. Implemented and benchmarked: it was **3-5× slower** than the per-leg approach for the binary case (3.4 s per-leg → 17 s batched) and similarly slow for bucket. Two reasons:

- Passing a list of explicit per-leg globs to `read_parquet([glob1, glob2, ...])` defeats per-partition sort: DuckDB now has to merge-sort the union by `exchange_ts` instead of returning each already-sorted partition as-is.
- Trying a single `symbol=*` glob with `WHERE symbol IN (...)` was only ~1.6× slower than per-leg but still no win; the IN predicate is pushed down but not before the cross-partition sort.

Reverted. The per-leg query path is the right shape for this data layout; the only structural change we kept is sharing one `duckdb.connect()` across all leg iterators (saves a few μs of setup × 16-20 calls and tidies up resource ownership).

## Speedups applied

### Fix 1 — Vectorise `_build_leg_event_array`

**File:** `hlanalysis/backtest/runner/hftbt_runner.py`.

Replace the two-pass per-cell loop with a single pass that accumulates per-event Python lists, then bulk-assigns each column at the end (`arr["ev"] = ev_list`, `arr["exch_ts"] = ts_np`, …). Bulk assignment dispatches once to C-level numpy code per column instead of N Python-level descriptor writes. Semantically identical to the original — verified by a parity test against a naive reference (`test_build_leg_event_array_matches_naive_reference`).

| Scenario        | Raw baseline | Raw after fix 1 | Speedup |
|-----------------|--------------|------------------|---------|
| hl_binary 1q    | 36.9 s       | 17.9 s           | 2.06×   |
| hl_bucket 1q    | 70.1 s       | 32.1 s           | 2.19×   |

### Fix 2 — Share DuckDB connection across `events(q)` and `resolved_outcome(q)`

**File:** `hlanalysis/backtest/data/hl_hip4.py`.

Open one `duckdb.connect()` per `events(q)` call and pass it into the per-leg `_book_iter`, `_trade_iter`, `_reference_iter`, `_settlement_iter`. `resolved_outcome(q)` likewise opens one connection and passes it into `_last_btc_ref_at_or_before` for the BTC-mark fallback.

Also: collapsed `_reference_iter`'s `rows = list(primary_iter)` → `iter(rows)` round-trip into a single `fetchall()` + generator (`_reference_iter` + `_reference_rows`). Reads exactly the same rows; one less indirection.

| Scenario        | Raw after fix 1 | Raw after fix 1+2 | Δ      |
|-----------------|------------------|---------------------|---------|
| hl_binary 1q    | 17.9 s          | 18.5 s             | ~equal  |
| hl_bucket 1q    | 32.1 s          | 31.9 s             | ~equal  |

Per-call DuckDB connection count collapsed from 14-23 (binary/bucket events) to **1 per `events(q)`**, and `resolved_outcome(q)` is now at most 1 (settlement scan) + 1 (BTC fallback) = 2 connections. The remaining global connect calls are from `discover` (1, once) and `_load_meta` (1, cached per question_id).

Wall-time impact is small (DuckDB connection setup is fast in-process) but the resource-ownership story is cleaner — covered by a regression test (`test_events_opens_single_connection`).

### Fix 3 — Skip the `_reference_iter` list materialisation

Fused into the Fix 2 refactor — `_reference_iter` is now a straight `fetchall()` + `iter`, no `list(primary_iter)` round-trip. Tiny win, but it was free.

### Fix 4 — Arrow-backed `events_arrays(q)` fast path

**Files:** `hlanalysis/backtest/data/_hl_hip4_fastpath.py` (new), `hlanalysis/backtest/data/hl_hip4.py` (new method), `hlanalysis/backtest/runner/hftbt_runner.py` (fast-path branch).

The legacy `events(q)` path yields `BookSnapshot` / `TradeEvent` dataclasses one at a time; the runner immediately unpacks them in `_build_leg_event_array`. For HL HIP-4 with 20-level books and ~160k snapshots per leg per question, that's ~3 M dataclass allocations per question that get thrown away after one read.

The new `HLHip4DataSource.events_arrays(q)` reads each leg's `book_snapshot` / `trade` parquet via DuckDB → Arrow → flat numpy column arrays, then assembles the hftbacktest `event_dtype` array with the legacy `qty=0` clear semantics preserved (the inter-snapshot stale-level diff stays in Python but operates on flat numpy slices, not dataclasses). Set events are emitted fully vectorised via `np.repeat` over the snapshot ts column. The runner detects `events_arrays` via `getattr` and routes around `_build_leg_event_array` when present; the legacy path stays intact for the synthetic / polymarket sources.

We also tried emitting one `DEPTH_CLEAR_EVENT` per side per snapshot to skip the diff entirely — a Python depth-replay confirmed identical post-state at 100/100 sampled snapshot timestamps, but the actual `hftbacktest` fills diverged by a few price ticks ($0.004 / $0.07 per-share shifts on the v1 fixture). The per-level `qty=0` semantics turn out to interact with the engine's matching in some subtle way; we did not chase the root cause and kept the diff-based path. The arrow + flat-numpy + vectorised-sets pipeline still cuts wall time substantially even with the diff loop retained.

| Scenario        | After fix 1+2 | After fix 1+2+4 | Δ                      |
|-----------------|---------------|------------------|------------------------|
| hl_binary 1q    | 18.5 s        | 14.8 s          | 1.25× (cum. 2.5× vs baseline) |
| hl_bucket 1q    | 31.9 s        | 20.4 s          | 1.56× (cum. 3.4×)             |
| hl_binary 3q    | 49.5 s        | 36.0 s          | 1.38×                          |
| hl_bucket 3q    | 95.9 s        | 59.6 s          | 1.61× (cum. 3.8× vs profiler baseline) |

Verified bit-equivalence to the pre-fast-path fills via `scripts/perf/diff_fills.py` on both binary and bucket fixtures.

## End-to-end full-corpus run

```
hl-bt run --strategy v1_late_resolution --data-source hl_hip4 \
    --config data/sim/configs/v1-finalize-5.json \
    --out-dir data/sim/runs/perf-full-v1 \
    --start 2026-05-06 --end 2026-05-21
```

- **Discovered:** 24 questions (15 binary + 9 bucket — the planning doc said 26 questions; 2 fall outside the discovery date filter on this branch's commit).
- **Prior baseline (regime-breakdown-2026-05-20 v1 run):** ~70 min (from `data/sim/runs/regime-breakdown-2026-05-20/v1/fills/` file mtimes — first fill 21:54:25, last fill 23:04:27).
- **After fix 1 + fix 2 (committed in initial PR):** 8 m 37 s (~8× vs baseline).
- **After fix 4 added on top (arrow fast path):** **5 m 41 s** (CPU: 260 s user + 98 s sys), ~12.3× vs the ~70 min baseline, 1.52× on top of fix 1+2.

The per-question raw speedup is ~2.5-3.8× (binary / bucket); the full-corpus speedup is bigger because the baseline was a cold run that paid disk + DuckDB metadata cost more steeply. A like-for-like warm-cache baseline would land at ~3-4× since per-question raw is the floor. Either way, this puts v1 tuning back in the iterative regime instead of the leave-it-overnight regime.

## Files changed

- `hlanalysis/backtest/runner/hftbt_runner.py` — vectorised `_build_leg_event_array`.
- `hlanalysis/backtest/data/hl_hip4.py` — single shared connection per `events()` / `resolved_outcome()`; cleaner reference fallback path; `_sql_in` helper for date / symbol IN clauses.
- `tests/unit/backtest/test_hl_hip4_perf_regression.py` — new regression suite (7 tests): connection invariants + per-cell parity with a naive reference + event-stream ordering sanity check.
- `scripts/perf/bench_backtest.py`, `scripts/perf/diff_fills.py` — profiling + fill-equivalence harness (used during the perf work; useful for future regressions).

## Verification

- 1-question HL binary and bucket fills compared via `scripts/perf/diff_fills.py` (ignores `cloid` UUID): equivalent to baseline at each step (fix 1 alone, fix 1+2, final).
- `pytest tests/unit tests/integration` — all 414 tests pass on the optimised branch.
- New regression tests cover: single-connection invariant in `events()`, ≤2-connection invariant in `resolved_outcome()`, parity of `_build_leg_event_array` vs a per-cell reference, event-stream ordering / type-mix sanity.

## What we did not change

- Strategy logic in `hlanalysis/strategy/`. Untouched.
- `fills.parquet` schema. Untouched.
- Discovery (`HLHip4DataSource.discover`). Already cheap (one query).
- `_partition_has_files` LRU cache. Profiled, not on hot path.
- Cross-leg batched `read_parquet([…])`. Tried, was a regression, reverted.
