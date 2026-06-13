# Shared spine — the sim≡live identity

The code that **both** the engine and the backtester import and call. Touching
anything here changes both paths at once — and several modules are guarded to be
byte-identical across them. Read this before editing `marketdata/`, `risk/`, or
`parity/`.

## `marketdata/` — the shared market-data core (~1.1k LOC)

| File | Role |
|------|------|
| `market_state.py` | `MarketState`: OHLC, recent returns, σ, volume, **multi-cadence per-(symbol,dt) buffers**. One feed fans into many buffers (e.g. v31 binary dt=5 + bucket dt=2). |
| `ohlc.py` | OHLC bucketer (the single bucketing implementation). |
| `position_math.py` | `apply_fill`, stop price, settlement payoff, dust tolerance. All engine + sim fills book through here. |
| `decision_kernel.py` | `build_decision_inputs()` — the shared σ/returns read; `scanner.py` and the sim runner both call it. |
| `decision_input.py` | The decision-input value types. |

The engine's `engine/market_state.py` is a thin adapter over this core; the
backtester's `runner/market_state.py` wraps it. σ is proven **bit-identical** across
both paths (the parity gate's headline result).

## `risk/caps.py` — shared entry caps (~0.1k LOC)

Three pure, zero-IO predicates — daily-loss, global-inventory, concurrent-positions
(top-up exempt) — plus `daily_window_start_ns`. Imported by `engine.risk`,
`engine.scanner`, and `backtest.halt_replay`. One source, not a hand-copy. The
engine's full 12-rule veto (`engine/risk.py`) wraps these.

## `parity/` — proves engine ≡ sim (~1.2k LOC)

| File | Role |
|------|------|
| `decision_replay.py` | Replays the engine path (`ReplayRunner`) and compares decisions to the sim. |
| `position_timeline.py` | Reconstructs the sim's held position per instant from its own fills (shared `apply_fill`) and feeds it to the engine replay → real decision-count parity. |
| `validation.py`, `sources.py` | Validation harness + source adapters. |

Run with **`make parity-gate`** (hermetic, no network) — assert before merging any
change to the engine evaluation path or the sim path. Asserts: `strategy_config_sig`
matches, σ is bit-identical at comparable scan points, and decisions match.

## `strategy_config_sig` — the fingerprint

Defined in `engine/config.py`. One hash makes any sim run comparable to any live slot;
the backtest stamps it into every `--slot` report and `make engine-diag` prints it.
If two sigs match, the params are provably identical.

## The invariant (don't break it)

> A passing backtest is trustworthy **only because** the simulated strategy is built
> from the same config path as live (`engine.config_builders`) and runs the same
> objects. The only intended variable is the data.

Therefore: refactors here must reproduce the deterministic backtest **byte-for-byte**
and keep every slot's `config_sig` identical. What *deliberately* differs is **params,
not code** — HL and PM tracks tune independently. The one thing that genuinely cannot
be shared is book assembly (live L2 book vs `hftbacktest` depth arrays); that split is
exactly why the decision-parity gate exists.

## Related

Who calls this: [engine.md](engine.md) (scanner/risk) and [backtest.md](backtest.md)
(runner). The strategies themselves: [strategy.md](strategy.md).
