# Overview

System orientation. Read first if you're new to the repo.

## The four subsystems

Everything is one package, `hlanalysis`, exposed through four CLI entry points
(declared in `pyproject.toml` → `[project.scripts]`, run with `uv run <cli>`):

| CLI | Entry point | Purpose |
|-----|-------------|---------|
| `hl-recorder` | `recorder.main:main` | Capture HL/PM/Binance streams to partitioned Parquet/DuckDB |
| `hl-engine` | `engine.main:main` | Live/paper trading runtime |
| `hl-replay` | `engine.replay:_cli` | Replay recorded data **through the engine path** |
| `hl-bt` | `backtest.cli:main` | Backtest + walk-forward grid tuning |

## Data flow

```
Venues (HL · PM · Binance)
        │ WS books/trades/settlement + Binance spot BBO
        ▼
   recorder  ──normalize──▶  Parquet / DuckDB  ──▶  S3 archive
        │                          │
        │                          ▼
        │                    backtester (hl-bt)  ──walk-forward grid──▶ winning params
        │                          │                                          │
        │                          └────────── promote ──▶ config/strategy.yaml
        ▼                                                          │ feeds
   engine (hl-engine) ◀───────────────────────────────────────────┘
   scan → risk gate → route → reconcile   (live or paper)
```

Both the tuner and the engine feed a market tick through the **same**
`strategy.evaluate(...)`. Only the data source differs (disk vs live socket).
PM L2 books are **not backfillable** — the recorder is the only source of truth.

## Shared vs independent — "one brain, two bodies"

The load-bearing design decision: the engine and backtester each own their own IO,
plumbing, and persistence, but both **import and call** one shared spine:

- `strategy/` — the registry and every `@register`'d strategy.
- `marketdata/` — `MarketState` core, OHLC bucketer, position math, decision kernel.
- `risk/caps.py` — the three entry-cap predicates (daily-loss, inventory, concurrency).

A shared `strategy_config_sig` fingerprint (`engine/config.py`) and the
`parity/` CI gate (`make parity-gate`) prove a sim run is comparable to a live slot.
See [shared-spine.md](shared-spine.md). When `hl-bt` runs with `--slot`, it builds
the strategy object via `engine.config_builders` — bit-for-bit the live config path.

## HL vs PM tracks (critical)

The same strategy **code** runs on two market **tracks** with independently-tuned
**params**:

- **HL** — Hyperliquid HIP-4 binaries, **crypto only**, 24h markets.
- **PM** — Polymarket daily markets, **incl. non-crypto** (WTI, NBA, …).

Params **do not transfer** between tracks — a PM-tuned grid typically loses money on
HL and vice-versa. Tune each track independently. Live engine slots are named
accordingly (`v1`, `v31` for HL; `v1_pm`, `v31_pm` for PM).

## Glossary

| Term | Meaning |
|------|---------|
| **slot** | One `(strategy, account)` pair the engine runs. Each has its own DAL, risk gate, kill switch, daily-loss cap, cloid prefix. |
| **track** | A market venue family with its own tuned params: HL or PM. |
| **v1** | `v1_late_resolution` — near-resolution arbitrage on binaries (`strategy/late_resolution.py`). |
| **v31 / v3.1** | `v3_theta_harvester` — theta/edge harvesting (`strategy/theta_harvester.py`). "v31" is a tuned *param generation*, not a separate class. |
| **binary vs bucket** | Two HIP-4 instrument classes (above/below a strike vs a price-range bucket). Tuned per-class. |
| **TTE** | Time-to-expiry of a market. |
| **favorite** | The outcome side priced above a threshold (e.g. 0.85) — the side strategies lean into. |
| **paper_mode** | Engine runs every path except the real order POST (logged only). Default true. |
| **reference feed** | The Binance spot price/σ feed strategies price against (not the traded venue). |
