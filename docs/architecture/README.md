# Code map (for agents)

Concise, navigation-first maps of the `hlanalysis` codebase. Read the one for the
area you're touching **before** grepping — each lists what every file is responsible
for, the entry points, and the gotchas that bite. These are maps, not ground truth:
file paths are kept current but always read the code before acting.

For the full prose architecture, see [`../architecture.html`](../architecture.html)
(rich, code-verified). For conventions (worktrees, data reuse, ops, deploy) see
[`../../CLAUDE.md`](../../CLAUDE.md).

## Which doc do I want?

| You're working on… | Read |
|--------------------|------|
| Anything — orientation, the 4 CLIs, data flow, HL/PM tracks, glossary | [overview.md](overview.md) |
| The **live trading runtime** (loops, slots, routing, risk, reconcile) | [engine.md](engine.md) |
| **Backtesting / tuning** (data sources, caches, fills, walk-forward) | [backtest.md](backtest.md) |
| **Strategy logic** or **adding a new strategy** | [strategy.md](strategy.md) |
| Code that must stay **sim≡live identical** (MarketState, position math, parity) | [shared-spine.md](shared-spine.md) |
| The **recorder**, venue **adapters**, event taxonomy, on-disk layout, config files | [recorder-data.md](recorder-data.md) |

## The one-line mental model

> **One strategy brain, two bodies.** The live **engine** and the **backtester** are
> two harnesses wrapped around one shared spine of `strategy/` + `marketdata/` +
> `risk/` code. A tuned parameter set behaves identically in sim and in production —
> the only variable is whether ticks come from a live socket or from disk. The
> **recorder** captures reality to disk so the backtester has something to replay.

## Package at a glance

| Subpackage | Role | ~LOC |
|------------|------|-----:|
| `hlanalysis/recorder/` | Long-running collector → partitioned Parquet/DuckDB | 0.4k |
| `hlanalysis/adapters/` | Venue WS/REST clients (HL, Binance, Polymarket) + normalizers | 2.5k |
| `hlanalysis/engine/` | Live/paper trading runtime | 10.5k |
| `hlanalysis/backtest/` | Replay + walk-forward grid tuning | 11.8k |
| `hlanalysis/strategy/` | Strategy registry + every registered strategy (shared) | 4.8k |
| `hlanalysis/marketdata/` | Shared spine: MarketState, OHLC, position math, decision kernel | 1.1k |
| `hlanalysis/risk/` | Shared entry-cap predicates (`caps.py`) | 0.1k |
| `hlanalysis/parity/` | Asserts engine ≡ sim (decision replay, position timeline) | 1.2k |
| `hlanalysis/analysis/` | Offline microstructure / markout analytics | 1.0k |
| `hlanalysis/alerts/` | Telegram alert rules | 0.4k |
| `hlanalysis/{config,events}.py` | Shared config loader + the normalized event taxonomy | — |
