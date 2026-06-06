# CLAUDE.md — conventions for working in HLAnalysis

Start with [README.md](README.md) for the architecture overview and repo map, and
[DEPLOYMENT.md](DEPLOYMENT.md) for the operational runbook. This file captures the
load-bearing conventions that aren't obvious from the code.

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

## Deploy is SSM-only

Production deploys go through `scripts/deploy.sh`, which runs `git pull` + service
restart on the EC2 box via **AWS SSM `send-command`** (`make deploy` /
`deploy-recorder` / `deploy-engine`). **SSH to the box is blocked** — never assume
an interactive shell. Pushing config to SSM is `make push-engine-secrets`. The
engine has a drift gate (`restart_blocked`) that can suspend the scanner after a
deploy; check `make engine-status`.

## Don't break the surface area

The four CLI entry points in `pyproject.toml` (`hl-recorder`, `hl-replay`,
`hl-engine`, `hl-bt`) and `hlanalysis` package imports must keep working. Tests
import a couple of repo-root helpers as modules (`scripts.train_nba_wp`,
`tools.backfill_daily_compaction`) — those files are load-bearing, not loose
scripts.

## Commits

Conventional-commit style (`feat:`, `fix:`, `chore:`, `docs:`, …). **Do not add
`Co-Authored-By` lines or any AI attribution** to commit messages or PR bodies.
