# Backtest Simulation Speedups (v2) — 2026-06-05

## TL;DR

Three independent levers on top of the 2026-05-20 HL fast-path rework, plus a
dedup so HL and PM share one event-array assembler:

1. **Parallelize `hl-bt run`** across markets (was serial). **95.2 s → 27.3 s
   (3.5×)** on a 5-market HL binary window, warm cache; fills bit-equivalent to
   serial. Helps PM + HL. Speedup scales with `min(#markets, cores)` minus
   process overhead.
2. **Event-array disk cache** (param-independent built arrays, keyed on source
   file size/mtime + `BUILD_VERSION`). Warm cache hit **1081 ms → 17 ms (62.8×)**
   on the per-question data-prep stage (real PM market, 3932 snapshots),
   bit-equivalent bundle. This is the data-prep saving that compounds across a
   tuning sweep (built once per question, not once per param-combo).
3. **PM recorded-book fast path** (`events_arrays`) sharing the HL assembler —
   PM was still on the slow `events()` dataclass path. Bit-equivalent fills on
   real recorded data.

All changes preserve fills exactly (modulo the random `cloid` UUID). Full unit
suite: **138 passed**.

## Context

The 2026-05-20 rework (`2026-05-20-backtest-perf.md`) made the HL HIP-4
`events_arrays()` path ~18× on the full corpus but deliberately left three gaps:
`hl-bt run` ran serially; Polymarket had no fast path; nothing cached the
param-independent built arrays. This work closes all three and refactors the
shared assembly into `_fastpath_core.py` so HL and PM cannot drift.

## Architecture

- `backtest/data/_fastpath_core.py` (NEW) — source-agnostic assembler:
  `event_dtype`, `_diff_clears` (numba), `LegArrays`, `FastPathBundle`,
  `build_leg_event_array_from_columns`, `_resample_reference_rows`,
  `BUILD_VERSION`.
- `_hl_hip4_fastpath.py` / `_pm_fastpath.py` — venue-specific column loaders
  only; both call the shared assembler.
- `_event_array_cache.py` (NEW) — `cached_bundle(...)` wraps any bundle builder;
  key = sha256(question_id + sorted[(path,size,mtime_ns)] + BUILD_VERSION).
  Stat-only keying (never reads file bytes). One `.npz` per key; load failure or
  `HLBT_REBUILD_CACHE`/`--rebuild-cache` → rebuild.
- `runner/parallel.py` (NEW) — `run_questions_parallel(...)` fans questions
  across a spawn `ProcessPoolExecutor`, results returned in input order;
  `reconstruct_source` / `build_hedge_source` shared with the tuning worker.
- Wiring: `cmd_run` gains `--workers` (default 1 = serial) and `--rebuild-cache`;
  `tune` gains `--rebuild-cache`; both HL and PM `events_arrays` route through
  the cache.

## Lever 1 — Parallel `hl-bt run`

`hl-bt run --data-source hl_hip4 --kind binary --start 2026-05-13 --end
2026-05-18` (5 binary markets), warm event-array cache, 12-core M2 Pro:

| workers | wall  | speedup |
|---------|-------|---------|
| 1       | 95.2 s | 1.0×   |
| 8       | 27.3 s | **3.5×** |

`user` time 101 s ≫ `real` 27 s confirms genuine multi-core use. Ceiling here is
`min(5 markets, 8 workers) = 5×`; 3.5× reflects per-market work variance +
process spawn overhead. `diff_fills.py` reports the serial and parallel fills
**equivalent** (cloid ignored). The shared pool worker also de-duplicates the
source/hedge reconstruction that `tune` previously inlined.

## Lever 2 — Event-array cache

Per-question data-prep (read parquet → build per-leg `event_dtype` arrays),
real PM market `0x602f533f…` (3932 snapshots), recorded mode:

| stage | time | |
|-------|------|--|
| cold (build) | 1081 ms | |
| warm (cache hit) | **17 ms** | **62.8×** |

Cached bundle is bit-equivalent to a fresh build (event arrays + reference +
settlement). The cache wraps the **shared** assembler, so HL and PM cache with
one code path. PM `events_arrays` was refactored to defer ALL source reads
(trades, klines, book, settlement) into the build closure, so a cache hit skips
them entirely — without this, a hit still re-read trades/klines (557 ms).

Caveat (unchanged from the v1 analysis): a tuning sweep's wall time is still
bounded by the param-dependent scan replay, which is not cacheable; lever 2
removes the data-prep, not the replay. Net sweep speedup ≈ `1/scan_fraction`.

## Lever 3 — PM recorded-book fast path

`PolymarketDataSource.events_arrays(q)` (recorded mode only; synthetic raises
`NotImplementedError` and stays on the legacy path — low snapshot volume, not the
bottleneck). PM's recorded book parquet shares HL's schema, so `_pm_fastpath.py`
only adds a column loader + within-snapshot level normalization (bids px DESC,
asks px ASC) matching `_normalize_levels`; the shared assembler does the rest.

**Equivalence:** element-wise array order is NOT asserted (and does not hold once
stale-level clears appear — legacy uses Python set-subtraction order, the fast
path uses array order; identical situation to HL). The guaranteed invariant is
**multiset equality** (same events/values, possibly reordered within a
timestamp), verified on real 3932-snapshot data and in
`test_pm_fastpath_multiset_equiv_with_clears`. This is **fill-safe by
construction**: both builders stable-sort depth events before trades at equal
timestamps, and clears (stale prices) vs sets (current prices) touch disjoint
price levels, so any intra-timestamp reordering yields an identical final book
per timestamp → identical fills (the same basis on which HL was validated).

## Dedup

- One assembler (`_fastpath_core`) shared by HL + PM loaders.
- One process-pool worker (`runner/parallel`) shared by `run` + `tune`.
- One cache (`_event_array_cache`) wrapping the shared assembler → HL + PM cache
  via one path.

## Verification

- `uv run pytest tests/unit/backtest -q` → **138 passed** (incl. 4 cache tests,
  6 PM fast-path tests, parallel-run determinism, HL perf regression unchanged).
- HL serial vs `--workers 8`: `diff_fills.py` equivalent.
- PM recorded fast vs legacy: per-leg multiset-equal on real data; cached vs
  fresh bundle bit-equivalent.

## Not done / deferred

- Per-tick scan-loop optimization (lever 4 from the analysis) — higher risk,
  lower payoff; not attempted.
- Real-data end-to-end PM **fill** benchmark in recorded mode: only 2 PM markets
  currently have full two-leg book coverage and both are efficiently priced (0
  fills), so a non-vacuous live-fill diff awaits more recorded-book coverage.
  Correctness rests on the multiset + construction argument above, which is the
  same basis used for HL.
- `tune` cold-vs-warm full-sweep wall-clock not re-measured here; the 62.8×
  per-question data-prep saving is the component number.

## Files changed

- NEW: `_fastpath_core.py`, `_pm_fastpath.py`, `_event_array_cache.py`,
  `runner/parallel.py`, `tests/unit/backtest/{test_parallel_run,
  test_event_array_cache, test_pm_fastpath_equiv}.py`,
  `tests/fixtures/configs/v1-smoke.json`.
- MOD: `_hl_hip4_fastpath.py`, `hl_hip4.py`, `polymarket.py`, `hftbt_runner.py`
  (one-line `HLBT_DISABLE_FASTPATH` guard), `cli.py`, `tuning.py`, `.gitignore`.

## Addendum (2026-06-05, same day) — cache hardened + flipped default-ON

After a benchmark battery found two cache bugs, the cache was first made
**opt-in/default-OFF** (commit `7881738`): (1) the key omitted
`reference_resample_seconds`/source-mode so a dt=60 bundle could serve a dt=5
request (σ-inflation poisoning); (2) uncompressed bundles filled the disk. Both
are now fixed and the cache is **default-ON** again, with these safeguards:

- **Poisoning** — `config_sig` (resample dt + feed/book source) folded into the
  key on both venues; extracted to `HLHip4DataSource._bundle_config_sig()` /
  `PolymarketDataSource._bundle_config_sig()` with a coverage test
  (`test_bundle_config_sig.py`) asserting every bundle-affecting param changes
  the signature — guards the "forgot-a-knob" regression class.
- **Orphan eviction** — entries are filename-prefixed `v{BUILD_VERSION}_`;
  `_prune_stale_versions` deletes superseded-version files on the next run
  (cheap glob, no file reads). Disk is now bounded by *one entry per
  (question, config)*, not unbounded growth.
- **Size cap** — `_enforce_size_cap` LRU-evicts by mtime past a byte budget
  (default 20 GiB; `HLBT_CACHE_MAX_GB` / `HLBT_CACHE_MAX_BYTES`).
- **Compression / storage** — `np.savez` → `np.savez_compressed`, and the
  event arrays are **column-split** (each `event_dtype` field stored as its own
  homogeneous array) with **delta-encoded** monotone timestamp columns. On a
  realistic 160k-event leg this is **5.7× vs raw** (vs 4.0× for compressed
  struct layout). Tuned columnar parquet measured ~7× but needs a per-entry
  multi-file directory serializer; rejected as over-engineering since eviction
  already bounds disk — left as a documented follow-up if a real full-corpus
  sweep proves disk is still the binding constraint. `BUILD_VERSION` 2→3
  orphans the old struct-layout entries.
- **Escape hatch** — `--fresh` / `--no-cache` (sets `HLBT_NO_CACHE`) forces a
  guaranteed-fresh build; `--rebuild-cache` forces a one-time rebuild;
  `--cache-event-arrays` is now a deprecated no-op. Tests are kept hermetic via
  a global autouse conftest fixture that sets `HLBT_CACHE_EVENT_ARRAYS=0`.
