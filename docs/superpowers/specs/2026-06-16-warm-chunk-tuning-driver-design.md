# Warm-chunk tuning driver — design

**Date:** 2026-06-16
**Status:** Proposed
**Scope:** `scripts/perf/resumable_run.py` (+ a new in-process chunk worker entrypoint)

## Problem

`scripts/perf/resumable_run.py` runs **each `(config, question)` pair as its own
cold `uv run hl-bt run` subprocess** (`--skip-markets i --max-markets 1`). This
buys crash-resume + memory-bounding + orphan-reaping, but pays the two dominant
costs `M·N` times for an `M`-config × `N`-question sweep:

1. **Cold start** — `uv run` + import the whole backtest stack (~15–20 s) per job.
2. **Re-decoding the same data per config** — a question's recorded events are
   loaded, decoded, and its settlement DuckDB queried **fresh for every config**.
   Profiling (`backtest_perf_v3`) shows `strategy.evaluate` is ~0.1 % of runtime;
   the cost is data decode + settlement. Replaying one question under `M` configs
   in `M` cold subprocesses wastes `(M−1)/M` of the dominant cost.

`hl-bt tune` already avoids both via the **in-process bundle memo**
(`HLBT_INPROC_BUNDLE_MEMO`, `_event_array_cache.py`) — the param-independent event
bundle is decoded once per question and reused across grid cells in a warm worker
— plus warm imports and `--scan-mode event` support. The driver throws all of
this away by isolating each `(config, question)` in a cold process.

## Goal

Bring the `hl-bt` speedups **into** the driver while preserving its guarantees
(crash-resume, no-OOM on the t4g box, no orphans, configs.json generality,
event-mode cadence, bit-identical-to-live `--slot` path).

Non-goal: replacing the driver with `hl-bt tune` (loses configs.json `env` /
variant-`slot_config` generality and the per-config report-aggregation contract).

## Approach (Option 1: warm chunk × all configs)

Change the **job unit** from `(config, question)` to a **chunk** = `K` consecutive
questions run against the **entire config list** inside a single warm subprocess
with the in-process bundle memo enabled.

Within a chunk subprocess the loop is **question-outer / config-inner**:

```
for q in chunk_questions:          # K questions
    for cfg in configs:            # M configs
        run cfg on q in-process    # bundle for q decoded once (cfg 1),
                                    # memo-served for cfg 2..M
        write per-(cfg, q) report + .done
```

Because `config_sig` (`hl_hip4.py:_bundle_config_sig`, mirrored in
`polymarket.py`) folds in every bundle-affecting input (resample dt, ref
event/source, warmup, ticks), heterogeneous configs are handled correctly in one
process automatically: same-dt/source configs share `q`'s bundle; different-dt
configs get distinct bundles. No manual partitioning of the config list is needed.

### Why question-outer / config-inner

Keeps **at most one (or a few) question bundle(s) resident** at a time, so the
memo footprint per worker is bounded by a single question's bundle, not the whole
chunk. (Config-outer would force all `K` bundles resident to get reuse.)

### Parallelism & memory

- The supervisor's work-queue is unchanged in shape; only the queued unit becomes
  the chunk. `--workers N` runs **N chunk-subprocesses concurrently**, work-stealing
  the next undone chunk as each finishes.
- The memo is a **per-process module-global**, so each parallel chunk has its own
  memo — reuse happens *within* a chunk, never across them. No shared state, no
  contention.
- The supervisor sets `HLBT_INPROC_BUNDLE_MEMO_WORKERS=N` (as `tune` does) so each
  worker's memo budget is `total / N`; N parallel chunks cap aggregate memo RAM at
  the total budget, not `N×` it. This preserves the OOM protection the driver
  exists for.
- Each chunk is its own process group (`start_new_session=True`), so an OOM/crash
  kills one chunk, not the supervisor or other chunks. Orphan-reaping unchanged.

### Expected speedup

For `M` configs × `N` questions: data-decode + settlement drops from `M·N` →
~`N` (memo), and cold-starts from `M·N` → ~`⌈N/K⌉` (chunks). Since decode +
settlement dominate, this is roughly an `M×` win on the data path plus the
startup amortization — without changing fidelity (same per-`(config, question)`
`--slot` results).

## Components

### 1. Chunk worker — in-process multi-config loop, folded into `resumable_run.py`

`resumable_run.py` gains a **worker mode**: an internal flag (e.g.
`--_worker-chunk <chunk_idx>`) makes the script run *one chunk* in-process and
exit. The supervisor shells into **itself** once per chunk
(`subprocess.Popen([sys.executable, __file__, "--_worker-chunk", ...])`), so the
crash/OOM isolation (own process group) is preserved while everything lives in one
file — no separate entrypoint, no new CLI surface. The worker-mode branch is
hidden from normal `--help` usage (underscore-prefixed flag).

Worker-mode responsibilities:

- Parse: chunk bounds (`--skip-markets`, `--max-markets`), `--configs` (the same
  JSON the supervisor already loads), data-source + `--slot`/`--start`/`--end`/
  `--kind` args, and per-config scan defaults.
- Set `HLBT_INPROC_BUNDLE_MEMO=1` for the process.
- Discover descriptors **once**, slice to the chunk.
- For each question (outer) × config (inner): build that config's strategy +
  `SourceConfig` (params/scan vary; dt-coupling preserved) and run the single
  question in-process via the **existing** `run_questions_parallel(..., n_workers=1)`
  path used by `cmd_run` — i.e. bit-identical to today's per-question run.
- Write each cell to **`out_base/<config_id>/q{global_idx}/report.md` + `.done`** —
  the **exact same per-`(config, question)` layout as today** (the chunk only
  governs *which subprocess* computes those dirs, not the on-disk shape). This
  keeps `parse_pnl` and the parity test on byte-identical paths.
- Optional refinement: the worker skips any cell whose `.done` already exists, so
  a re-launched chunk recomputes only its missing cells (per-chunk *re-launch*,
  per-cell *skip*). Simplest fallback is full-chunk recompute; both satisfy the
  accepted per-chunk resume tradeoff.

Reuses `run_questions_parallel` + `write_single_run_report` from the run path; no
new fill/decision code, so the `--slot` parity guarantee is untouched.

### 2. Supervisor (`resumable_run.py`) — rework the job unit

- **Queue keyed by chunk index** (not `(config_id, idx)`). One job = one chunk =
  all configs over `K` questions. `--chunk-size K` **defaults to 25** (amortizes
  startup over a full chunk); `--chunk-size 1` reproduces the historical
  per-question granularity for resume, just warm instead of cold.
- `_launch` builds the **worker-mode self-invocation**
  (`[sys.executable, __file__, "--_worker-chunk", idx, ...]`, passing the full
  configs.json + chunk bounds) instead of a single-config `hl-bt run`.
- Resume: a chunk is `done` when its `.done` markers exist; partial chunks re-run
  in full (per-chunk resume — accepted tradeoff).
- `_finish` / `classify` / retry / timeout / orphan-reaping logic **unchanged**
  (operates on the chunk subprocess).
- `aggregate` is repointed to read the **per-config report dirs directly**
  (`out_base/<config_id>/q*/report.md`, the existing `parse_pnl`), summing per
  config — instead of reading per-job PnL from the manifest (job records are now
  chunk-keyed, so they no longer carry per-config PnL). The printed per-config
  output contract is preserved; only the source of the numbers moves from the
  manifest to the report dirs (which are the source of truth anyway).
- Set `HLBT_INPROC_BUNDLE_MEMO_WORKERS = --workers` before launching.

## Data flow

```
resumable_run.py (supervisor)
  ├─ discover N questions (once)            [config-independent]
  ├─ build chunk queue: ⌈N/K⌉ chunks
  ├─ pool of `--workers` slots:
  │    each slot → subprocess: resumable_run.py --_worker-chunk i --configs cfg.json
  │         ├─ HLBT_INPROC_BUNDLE_MEMO=1
  │         ├─ discover + slice to chunk
  │         └─ for q in chunk: for cfg in configs:
  │               run_questions_parallel(q, cfg, n_workers=1)  ← memo serves q's bundle
  │               write out/<cfg>/<dir>/report.md + .done
  ├─ on each finish: classify → done | retry | failed; save manifest
  └─ aggregate per-config totals across chunks
```

## Error handling

- Per-chunk crash/OOM/timeout: classified + retried exactly as today (the unit is
  bigger but the machinery is identical). A retry re-runs the whole chunk.
- Within a chunk, a single `(config, question)` that raises is caught and recorded
  as that cell's failure **without aborting the rest of the chunk** (so one bad
  cell doesn't cost the chunk's memo work); the chunk subprocess exits non-zero
  only if it cannot make progress, matching `classify`'s expectations.
- Memo correctness rests on `config_sig` capturing all bundle-affecting inputs —
  already contract-tested and comment-guarded in both data sources. **Verification
  task:** confirm no bundle-affecting input bypasses `SourceConfig` → `config_sig`
  (today none does; `env` knobs are runtime-only).

## Testing

- **Parity (the oracle):** for a small grid (e.g. 3 configs × 4 questions), the
  per-`(config, question)` `report.md` produced by the warm-chunk path must be
  **byte-identical** to the current cold per-question driver (serial, fixed cloid
  seed). This proves the memo/warm path changes nothing but speed.
- **Memo reuse:** assert (via a build counter / log) that a question's bundle is
  built once per chunk and served from memo for configs 2..M at the same dt, and
  rebuilt for a different-dt config.
- **Resume:** kill a chunk mid-flight; re-run; assert only undone chunks re-launch
  and completed `.done` cells are skipped.
- **Parallel safety:** `--workers >1` produces the same aggregate as `--workers 1`.
- **`--chunk-size 1` back-compat:** identical job keys / dirs / aggregate to the
  historical per-question behaviour.
- Existing `tests/perf/test_resumable_run.py` updated for the chunk-keyed queue.

## Risks / tradeoffs

- **Per-chunk resume** (accepted): a crashed chunk re-runs all its
  `K × M` cells. Tune `K` (small = finer resume + more startups; large = fewer
  startups + coarser resume).
- **Memo memory:** bounded by `total/N` per worker via the worker-aware budget;
  question-outer loop keeps only ~one bundle resident. Reuses `tune`'s proven
  bound, so no new OOM surface.
- **Worker mode in `resumable_run.py`**: a hidden self-invoked subprocess flag,
  not a new CLI surface; reuses run-path internals so no parity/fidelity surface
  is added. `resumable_run.py` grows — keep supervisor vs worker-mode code in
  clearly separated sections/functions.

## Out of scope

- Switching the sweep to `hl-bt tune` (Option 2) — rejected for configs.json
  generality + output-contract reasons.
- Cross-chunk memo sharing (would require shared memory; not worth the complexity).
- Changing fill/decision logic — this is a pure execution-orchestration change.
