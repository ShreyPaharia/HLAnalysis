# CLAUDE.md — conventions for working in HLAnalysis

Start with [README.md](README.md) for the architecture overview and repo map, and
[DEPLOYMENT.md](DEPLOYMENT.md) for the operational runbook. For a visual deep-dive,
open [docs/architecture.html](docs/architecture.html) (system architecture + the
shared sim/live spine). This file captures the load-bearing conventions that aren't
obvious from the code.

## Navigating the code (read before grepping)

[`docs/architecture/`](docs/architecture/README.md) is a concise, agent-oriented code
map — what every file is responsible for, entry points, and gotchas, per subsystem.
Open the one for the area you're touching first:

| Working on… | Read |
|-------------|------|
| Orientation, the 4 CLIs, data flow, HL/PM tracks, glossary | [docs/architecture/overview.md](docs/architecture/overview.md) |
| Live runtime (loops, slots, routing, risk, reconcile) | [docs/architecture/engine.md](docs/architecture/engine.md) |
| Backtesting / tuning (data sources, caches, fills, walk-forward) | [docs/architecture/backtest.md](docs/architecture/backtest.md) |
| Strategy logic / adding a strategy | [docs/architecture/strategy.md](docs/architecture/strategy.md) |
| Code that must stay sim≡live identical (MarketState, parity) | [docs/architecture/shared-spine.md](docs/architecture/shared-spine.md) |
| Recorder, adapters, events, on-disk layout, config files | [docs/architecture/recorder-data.md](docs/architecture/recorder-data.md) |
| Offline desk research (outcome-market win labels, cross-venue panels, desk metrics, HTML findings) | [docs/architecture/research.md](docs/architecture/research.md) |

## Desk research toolkit (`hlanalysis/research/`)

For offline data analysis — characterizing liquidity/mispricing/vol/adverse-selection
or hunting strategy edges — use `hlanalysis/research/` (read
[docs/architecture/research.md](docs/architecture/research.md)) rather than ad-hoc
DuckDB. It gives you the **outcome-market win-label resolver**, cached cross-venue
**panels**, standardized desk **metrics**, an HTML **card** builder, and the six
characterization cards from the 2026-06-13 desk study. Reusing it keeps analyses
reproducible and minutes-fast.

- **The load-bearing gotcha:** HL binary markets emit **no settlement event**;
  `settled_side_idx` is a constant artifact (always 0). Win labels are
  **oracle-derived** (`oracle_px(expiry)` vs `targetPrice` for binaries; band
  thresholds for buckets) — always go through `research/outcome_markets.py`, never
  re-derive winners from settlement.
- Run from a worktree with `HLBT_HL_DATA_ROOT=../../data` (data is main-checkout only;
  never `make pull-data`). Smoke gate: `uv run python -m hlanalysis.research.smoke`.
- Findings deck: `docs/research/hl-outcome-desk-<date>.html`. Every edge claim must
  carry n + date-span + **split-half** sign stability; treat n<15 as underpowered.

## Strategy naming

Strategies are registered via `@register(...)` in `hlanalysis/strategy/` and
selected by that key on the `hl-bt` / `hl-engine` CLIs.

- **`v1` = `v1_late_resolution`** (`strategy/late_resolution.py`) — near-resolution
  arbitrage on binary markets.
- **`v31` = `v3_theta_harvester`** (`strategy/theta_harvester.py`) — the theta /
  edge-harvesting strategy. "v31" / "v3.1" refers to the tuned **parameter
  generation** of this strategy code, not a separate class. Later variants
  (`v3_2_volclock`, `v3_4_lmgate`, `v3_5_momentum_mr`) are flag-gated extensions
  of the same module, off by default.
- Other registered strategies: `v2_model_edge`, `v4_binary_statarb`,
  `v5_delta_hedged`, and `v31_pm_nba` (a distinct Polymarket NBA win-probability
  strategy).

**HL vs PM tracks.** The *same* strategy code runs on two market tracks with
*different* tuned params: HL (Hyperliquid binaries, crypto-only) and PM
(Polymarket, including non-crypto). Live engine slots are named accordingly
(e.g. `v1`, `v31` for HL; `v1_pm`, `v31_pm` for PM). **Params do not transfer
between tracks** — a PM-tuned grid typically loses money on HL and vice versa.
Tune each track independently.

## Tuning workflow

1. `hl-bt tune --strategy <key> --grid config/tuning.<...>.yaml` runs a
   walk-forward parallel grid sweep over recorded data and reports per-cell PnL.
2. Promote the winning params into `config/strategy.yaml` (the live slot config).
3. Re-validate at the engine's real cadence before deploying.

`config/tuning.yaml` is the default/canonical grid. Per-experiment grids
(`tuning.v1-*`, `tuning.v3-*`, …) and their one-off runner scripts have been
archived under `experiments/`; they are historical, not maintained.

## Validate on recorded data, not synthetic klines

Backtests must run against **recorded** inputs — the Binance spot BBO feed and the
recorded Polymarket L2 book (at the live sampling cadence, e.g. `dt=5`) — because
that is what the engine actually consumes live. Do **not** validate config changes
against synthetic or stale kline data: PM strike klines silently *freeze* (not
error) when a cache is stale, and synthetic books don't reflect live
microstructure. A "passing" backtest on the wrong inputs is worse than none.

## Git worktrees & data reuse

Agent work usually happens in a worktree under `.worktrees/<name>/`. The ~18 GB of
recorded market data lives **only in the main checkout** (`data/` is gitignored and
is *not* copied into worktrees).

- **Never run `make pull-data` from a worktree.** It re-syncs gigabytes from S3 for
  no reason. The data is already on disk in the main checkout — reuse it. Point the
  backtester at it (from a `.worktrees/<name>/` checkout, main's data is `../../data`):
  ```bash
  export HLBT_HL_DATA_ROOT=../../data        # HL recorded venue data
  export HLBT_PM_CACHE_ROOT=../../data/sim   # PM sim caches
  # …or pass `--data-root ../../data` to hl-bt
  ```
  `make pull-data` is **only** for topping up the *main* checkout with newly-recorded
  days.
- **Rebase on local `main` before benchmarking.** Baselines drift fast as perf/fixes
  land; rebase the worktree onto local `main` first or you'll compare against a stale
  baseline.
- **Never `git checkout` a tracked file to "see" an old version** — that silently
  overwrites the working tree. Use `git show <ref>:<path>` / `git diff` instead.

## Deploy is SSM-only

Production deploys go through `scripts/deploy.sh`, which runs `git pull` + service
restart on the EC2 box via **AWS SSM `send-command`** (`make deploy` /
`deploy-recorder` / `deploy-engine`). **SSH to the box is blocked** — never assume
an interactive shell. Pushing config to SSM is `make push-engine-secrets`. The
engine has a drift gate (`restart_blocked`) that can suspend the scanner after a
deploy; check `make engine-status`.

## Ops & monitoring (all via SSM `send-command`, no SSH)

`make help` lists everything. SSH/interactive shells to the box are blocked — these
read-only diagnostics each run one `aws ssm send-command` and print the result:

| Command | What it does |
|---------|--------------|
| `make engine-status` | systemd state + per-slot `restart_blocked`/`halt` flags + last 30 journal lines |
| `make engine-diag [ALIAS=v1] [WINDOW=48] [PRETTY=1]` | JSON snapshot: positions, true PnL, rejects, config fingerprint, feed health |
| `make reconcile-report [JSON=1]` | per-slot realized + open-MTM **true PnL** vs venue truth; flags position drift |
| `make engine-events Q=<idx>` | full event trace (entry/exit/veto/reject) for one question across slots |
| `make engine-logs` / `make logs` | tail engine / recorder journal |
| `make data-summary` / `make query Q="…"` | DuckDB over recorded parquet on the box |
| `make parity-gate` | hermetic engine↔sim decision-parity gate (run before merging engine/sim changes) |

Gotchas:
- The box has **no `sqlite3` CLI** — use `/opt/hl-recorder/.venv/bin/python` for
  ad-hoc DB reads.
- Anything touching the live engine env runs via `.venv/bin/python` with
  `set -a; . /etc/hl-engine/env` — **not** `uv run` (uv isn't on the engine's PATH
  under that env).
- HL positions/PnL are **venue-authoritative** (the local HL ledger can drift —
  trust the venue); PM PnL is local-ledger. `reconcile-report` reflects this.

## Toolchain & CI

`ruff` (lint + format) and `mypy` are configured in `pyproject.toml`;
`.pre-commit-config.yaml` runs ruff on commit; `.github/workflows/ci.yml` **blocks**
on `ruff check`, `ruff format --check`, and `pytest` (mypy runs informationally).
Before pushing:

```bash
uvx ruff check hlanalysis && uvx ruff format --check hlanalysis tests
uv run pytest -q
```

## Working conventions

- **Don't kill defensive gates** (`min_bid_notional`, `stop_loss`, `stale_data_halt`,
  reject circuit-breakers, …) just because they never fire in a backtest. Backtests
  don't replay live adversarial microstructure; these gates earned their place from
  real incidents.
- **Triage tickets before coding.** For each ticket state *Problem* → *does it still
  exist in the current code (verified, with `file:line`)* → *Solution*, **before**
  implementing.
- **Re-baseline after any sim-infra change.** Parallel `run`/`tune` workers rebuild
  data sources from `HLBT_*` env, not the in-memory config, so config-derived knobs
  (`reference_resample_seconds`, `book_source`, dt, …) can silently revert to
  defaults — surfacing as "0 trades" or sigma-inflation. Always re-run a known
  baseline after touching the backtest data/worker path.

## Don't break the surface area

The four CLI entry points in `pyproject.toml` (`hl-recorder`, `hl-replay`,
`hl-engine`, `hl-bt`) and `hlanalysis` package imports must keep working. Tests
import a couple of repo-root helpers as modules (`scripts.train_nba_wp`,
`tools.backfill_daily_compaction`) — those files are load-bearing, not loose
scripts.

## Commits

Conventional-commit style (`feat:`, `fix:`, `chore:`, `docs:`, …). **Do not add
`Co-Authored-By` lines or any AI attribution** to commit messages or PR bodies.
