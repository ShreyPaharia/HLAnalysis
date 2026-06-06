# HLAnalysis

Market-making research and trading infrastructure for **Hyperliquid** (HL) binary
markets and **Polymarket** (PM) daily markets, with reference/hedge data from
Binance. The repo spans three cooperating subsystems plus a backtester, all built
on a shared `hlanalysis` Python package.

## Architecture

```
                  Binance (reference + hedge venue)
                         │  WS trades + REST premium
                         ▼
  ┌────────────┐    ┌──────────┐    ┌────────────────────┐
  │  recorder  │───▶│   data   │───▶│     backtester     │
  │ (hl-recorder)   │ (Arrow/  │    │      (hl-bt)       │
  └────────────┘    │  DuckDB) │    └────────────────────┘
        │ records   └──────────┘            │ tunes params
        │ HL + PM + Binance L2/trades        ▼
        ▼                              config/strategy.yaml
  ┌────────────┐                             │ feeds
  │   engine   │◀────────────────────────────┘
  │ (hl-engine)│  live + paper trading, replay (hl-replay)
  └────────────┘
```

- **recorder** (`hlanalysis/recorder/`) — long-running collector. Subscribes to HL
  and Polymarket order-book / trade / settlement streams plus a Binance spot
  reference feed, and persists them as partitioned Arrow + DuckDB on EC2 (then
  archived to S3). The backtester replays this recorded data; PM L2 books are
  **not** backfillable, so the recorder is the only source of truth.
- **engine** (`hlanalysis/engine/`) — the trading runtime. Runs one strategy
  "slot" per market track, evaluates entry/exit each tick against a reference
  feed, and places orders (paper or live). Co-located with the recorder on a
  single EC2 box, isolated by systemd cgroups.
- **backtester** (`hlanalysis/backtest/`) — replays recorded data through the same
  strategy code the engine runs, scores PnL, and does walk-forward grid tuning to
  produce the parameters that ship in `config/strategy.yaml`.

See [DEPLOYMENT.md](DEPLOYMENT.md) for the full operational runbook and
[docs/](docs/) for design specs, research write-ups, and analysis notebooks.

## The four CLIs

All four are defined in `pyproject.toml` under `[project.scripts]` and installed
by `uv sync`. Run them with `uv run <cli>`.

| CLI | Entry point | Purpose | One-line example |
|-----|-------------|---------|------------------|
| `hl-recorder` | `recorder.main:main` | Record HL/PM/Binance streams to disk | `uv run hl-recorder --config config/symbols.yaml` |
| `hl-engine` | `engine.main:main` | Live/paper trading runtime | `uv run hl-engine --strategy-config config/strategy.yaml --deploy-config config/deploy.yaml --symbols-config config/symbols.yaml` |
| `hl-replay` | `engine.replay:_cli` | Replay recorded data through the engine | `uv run hl-replay --help` |
| `hl-bt` | `backtest.cli:main` | Backtest + walk-forward grid tuning | `uv run hl-bt tune --strategy v1_late_resolution --grid config/tuning.yaml` |

(`make engine-local` runs the engine on your machine in paper mode with the
canonical configs.)

## Repo map

| Path | What's in it |
|------|--------------|
| `hlanalysis/` | The Python package. Subpackages: `recorder/`, `engine/`, `backtest/`, `strategy/` (strategy implementations + registry), `adapters/` (venue clients), `analysis/`, `alerts/`, `tools/`, plus `config.py` / `events.py`. |
| `config/` | Canonical, **active** inputs only: `strategy.yaml` (live engine params), `symbols.yaml`, `deploy.yaml`, `tuning.yaml` (default backtest grid), `config/backtest/`, and the live run configs `run.v31_pm_nba.json` / `v1-final.json`. |
| `scripts/` | Durable, reusable tooling: ops shell scripts (`deploy.sh`, `pull-data.sh`, `status.sh`, …), fixture capture (`capture_*.py`), data fetch helpers (`fetch_*.py`), venue probes/smokes, and arg-taking analysis utilities (`equity_curve.py`, `plot_*.py`, `analyze_*.py`, `best_params.py`, …). |
| `experiments/` | **Archived** one-off experiment runners + their per-experiment tuning grids. Frozen lab notebook; not part of the live system. See `experiments/README.md`. |
| `tests/` | Unit + integration tests (`pytest`). |
| `docs/` | `reports/` (numbered analysis notebooks + polished reports), `research/` (dated experiment write-ups), `specs/` (design specs). |
| `deploy/` | CDK infrastructure (EC2 + EBS, Tokyo region). |

## Dev quickstart

This project uses [`uv`](https://docs.astral.sh/uv/) for env + dependency
management and a `Makefile` for ops shortcuts.

```bash
uv sync --all-extras          # create .venv and install deps
uv run python -c "import hlanalysis"   # smoke check
uv run pytest -q              # run the test suite
make help                     # list deploy / monitoring targets
```

Strategy naming, the tuning workflow, and deploy conventions are documented in
[CLAUDE.md](CLAUDE.md).
