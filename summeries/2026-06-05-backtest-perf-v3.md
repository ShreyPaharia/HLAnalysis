# Backtest Simulation Speedups (v3) — 2026-06-05

## TL;DR

Step-by-step profiling of the warm-cache HL binary path showed the v2 levers
(parallel run, event-array cache, PM fast path) left the wall dominated by costs
that are **not** the per-tick scan loop the v2 doc deferred as "lever 4". Two
targeted fixes:

1. **Memoize `resolved_outcome` per question** (instance cache on
   `HLHip4DataSource`). It was called **3× per binary question** at settlement
   (settlement block + `leg_payoff` + `QResult`), each a fresh `duckdb.connect()`
   + settlement scan + 2-day BTC-ref lookup. → `run_one_question` **18.24s →
   15.11s (−17%)** on a 5-market binary window (3 trials each); settlement work
   itself **−75%** (11.34s → 2.83s); real `resolved_outcome` computes 15 → 5.
2. **Opt-in process-level bundle memo** (`HLBT_INPROC_BUNDLE_MEMO`, set by
   `tune`). A sweep replays each question across many param cells; the
   param-independent event-array bundle was re-`cache_key`'d (file stat) +
   npz-inflated every cell. Memoizing on `(question_id, config_sig)` makes the
   replay **970.8ms → 117.8ms (8.3×)**. Default-off so single `run`s and the
   mtime-invalidation contract are untouched.

Full backtest unit + integration suite: **162 passed**. Both changes TDD'd.

## Why "lever 4" was a red herring

`scripts/perf/step_bench.py` (new) monkey-patches every boundary inside
`run_one_question` and accumulates per-step wall time, call counts, and
first-call (one-time JIT) cost. The native scan loop is the derived remainder.

Warm-cache, 5 HL binary questions, BEFORE these fixes:

| step | total | calls | % of run_one_question |
|------|-------|-------|------|
| `resolved_outcome` (settlement) | 8.5 s | 15 (3×/q) | 46% |
| data_prep (cache_key 1.4s + npz load 2.9s) | 5.0 s | 5 | 27% |
| `leg_payoff` (≈ its internal resolved_outcome) | 2.8 s | 5 | 15% |
| native loop + persist (derived) | 1.8 s | – | 10% |
| `strategy.evaluate` | **0.017 s** | **7195** | **0.1%** |

The Python per-tick strategy evaluation is **0.1%** — optimizing the scan loop
(v2's deferred lever 4) would have chased a rounding error. The real costs were
redundant settlement SQL and (for sweeps) repeated cache I/O.

## Lever 1 — `resolved_outcome` memo

`HLHip4DataSource.__init__` gains `self._outcome_cache: dict[str, outcome]`.
`resolved_outcome(q)` checks it, delegating the real work to a new
`_resolve_outcome_impl(q)` and caching the result by `question_id`. Settlement
is deterministic per question (settlement data / final BTC ref don't change
within a process), so this is behavior-preserving. Mirrors the existing
`_meta_cache` pattern. Helps single `run`s AND every tune cell (the 3× collapse
happens within one question's processing).

Test: `test_resolved_outcome_memoized_per_question` — two `resolved_outcome`
calls invoke `_last_btc_ref_at_or_before` exactly once.

## Lever 2 — opt-in process bundle memo

`cached_bundle` gains an in-process LRU (`_INPROC_MEMO`, `OrderedDict`, default
cap 512 via `HLBT_INPROC_BUNDLE_MEMO_MAX`) keyed on `(question_id, config_sig)`.
On a hit it returns the memoized bundle, skipping `_prune_stale_versions`,
`cache_key` (file stat), and `_load` (npz inflate). The disk-cache body moved to
`_cached_bundle_disk`.

**Why opt-in, not a `cache_key` memo:** memoizing `cache_key` by question would
break `test_mtime_change_invalidates` (it rewrites a file mid-process and asserts
the key changes — the cross-run invalidation contract). The process memo is a
*different* contract: "source files are immutable for this process's lifetime",
which `run`/`tune` over historical parquet satisfy but the test deliberately
does not. Default-off keeps both honest; `tune` sets it via `os.environ.setdefault`
(operator can still force `=0`). `config_sig` already encodes resample dt + source
mode (the v2 poisoning fix + its coverage test), so dt=5 / dt=60 cells get
distinct memo entries — no σ-poisoning.

Tests: `test_inproc_bundle_memo_skips_rebuild_when_enabled` (same object on
replay, no rebuild), `_off_by_default_loads_fresh` (default unchanged),
`_keys_on_config_sig` (dt variants don't share).

## Sweep math

For P param cells × Q questions, same-dt:
- Settlement: was 3 computes/q/cell → now 1 (lever 1), every cell.
- Data-prep: was ~970ms/q/cell → now 970ms first cell + ~118ms/q/cell after
  (lever 2). The 118ms floor is the `_fastpath_source_files` glob in
  `events_arrays`, which runs *before* `cached_bundle` — hoisting the memo above
  the glob is the obvious follow-up.

## Not done / deferred

- Hoist the process memo above the `events_arrays` glob (remove the 118ms floor).
- npz compression level (savez_compressed → uncompressed/lz4): the 586ms/q
  inflate is a disk-vs-CPU knob, only worth it if a real full-corpus sweep shows
  inflate binding after lever 2.
- Numba jitclass spawn recompile (~4.2s/worker under spawn) — only bites
  `--workers>1`/`tune` startup; fork-context / `NUMBA_CACHE_DIR` mitigation is
  higher-risk, not attempted.

## Files changed

- MOD: `hl_hip4.py` (outcome cache), `_event_array_cache.py` (process memo +
  `_cached_bundle_disk` split + `import os`/`OrderedDict`), `tuning.py`
  (enable memo in worker).
- TEST: `test_hl_hip4_source.py` (+1), `test_event_array_cache.py` (+3, autouse
  memo clear).
- NEW: `scripts/perf/step_bench.py` (reusable per-step profiler).
