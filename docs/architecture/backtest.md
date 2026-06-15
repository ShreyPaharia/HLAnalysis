# Backtester — replay + walk-forward tuning

`hlanalysis/backtest/` (~11.8k LOC). Read when running backtests, adding a data
source, or touching the fill/tuning path.

## Entry & subcommands

`backtest/cli.py` (`hl-bt`) → `run` / `tune` / `fetch` / `trace` / `report`.
CLI plumbing is split out: `_cli_args.py` (arg parsing), `_cli_commands.py`
(subcommand bodies), `_cli_plumbing.py` (data-root/cache resolution).

```bash
hl-bt run  --slot v31 --data-source hl_hip4 --start … --end … --kind binary
hl-bt tune --strategy v3_theta_harvester --grid config/tuning.yaml
```

The **`--slot`** path builds the strategy via `engine.config_builders` (bit-identical
to live); `slot_config.py` is the live-slot → params bridge, guarded by
`test_slot_config_parity`. That is *why* a passing backtest is trustworthy.

## File map

| Area | Files |
|------|-------|
| CLI | `cli.py`, `_cli_args.py`, `_cli_commands.py`, `_cli_plumbing.py` |
| Core protocols | `core/data_source.py` (`DataSource` protocol), `core/events.py`, `core/question.py`, `core/registry.py` (`@register`), `core/source_config.py` (spawn-worker config, replaces `HLBT_*` side-channel) |
| Data sources | `data/hl_hip4.py` (HL binaries+buckets), `data/polymarket.py` (PM L2), `data/pm_nba.py`, `data/synthetic.py`, `data/binance_klines.py`, plus `_*_fastpath.py` decoders |
| Caches | `data/_event_array_cache.py` (npz memo, ~5.7× compression), `data/_parquet_schema.py` (shared column constants + contract tests) |
| Runner (sim loop) | `runner/hftbt_runner.py` (per-question loop, `hftbacktest` fills), `runner/parallel.py` (worker init, shared inventory ledger), `runner/_fills.py`, `runner/_routing.py`, `runner/_fees.py`, `runner/_latency.py`, `runner/market_state.py` (wraps the shared `MarketState`) |
| Tuning | `tuning.py` (walk-forward grid, ProcessPool fan-out), `runner/walkforward.py` |
| Resumable sweep driver | `scripts/perf/resumable_run.py` — crash-tolerant **warm-chunk** sweep driver: runs K questions × all configs per warm subprocess (in-process bundle-memo reuse, `--chunk-size` default 25), per-chunk resume, no-OOM/no-orphan. Use for big sweeps on the memory-constrained box; supports `--scan-mode event` cadence |
| Halt replay | `halt_replay.py` (replays engine halt windows + caps in sim) |
| Output | `report.py` (markdown), `runner/result.py` (Sharpe/hit/DD), `plots/` |

## Gotchas

- **Validate on recorded data, not synthetic/stale klines.** Use the Binance spot BBO
  feed + recorded PM L2 book at the live cadence (e.g. `dt=5`). PM strike klines
  silently *freeze* (not error) when stale. (CLAUDE.md → "Validate on recorded data".)
- **Worker config-drop** — parallel `run`/`tune` workers rebuild data sources from the
  spawn config / `HLBT_*` env, **not** the in-memory object. Config-derived knobs
  (`book_source`, `reference_resample_seconds`, dt) can silently revert to defaults →
  "0 trades" / sigma inflation. **Re-baseline after any sim-infra change.**
- **Data lives in the main checkout** — never `make pull-data` from a worktree; point
  `--data-root` / `HLBT_HL_DATA_ROOT` / `HLBT_PM_CACHE_ROOT` at `../../data`.
  (CLAUDE.md → "Git worktrees & data reuse".)
- **Determinism** — serial runs (`--workers 1`) reproduce `report.md` + diagnostics
  byte-for-byte (fills differ only by random cloid); this is the refactor oracle.

## Related

The strategy object and decision inputs come from the [shared spine](shared-spine.md).
The grid YAMLs and live config files are in [recorder-data.md](recorder-data.md#config-files).
