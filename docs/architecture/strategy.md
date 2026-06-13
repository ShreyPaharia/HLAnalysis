# Strategy — registry, strategies, adding one

`hlanalysis/strategy/` (~4.8k LOC). The decision brain. Imported and called by **both**
the engine and the backtester (see [shared-spine.md](shared-spine.md)).

## The contract

`strategy/base.py` + `strategy/types.py` define the `evaluate()` contract:
`Decision` / `Action` / `BookState` / `QuestionView`. Every strategy is a registered
object that the engine and sim both build and call identically.

## Registry

- **`@register("<key>")`** (defined in `backtest/core/registry.py`) decorates the
  factory in each strategy module. `strategy/__init__.py` side-effect-imports every
  module so the decorators run.
- **`strategy/live_registry.py`** — the live-build registry both engine + backtest go
  through (`register_live_strategy`); no hardcoded per-strategy dispatch.

## Registered strategies

| Key | File | What |
|-----|------|------|
| `v1_late_resolution` | `late_resolution.py` | Near-resolution arbitrage on binaries (**v1**) |
| `v3_theta_harvester` | `theta_harvester.py` | Theta / edge harvesting (**v31 / v3.1**) |
| `v3_2_volclock`, `v3_4_lmgate`, `v3_5_momentum_mr` | `theta_params.py` (+ `momentum_mr.py`) | Flag-gated extensions of theta, **off by default** |
| `v2_model_edge` | `model_edge.py` | GBM model-edge strategy |
| `v4_binary_statarb` | `binary_statarb.py` | Binary stat-arb |
| `v5_delta_hedged` | `delta_hedged.py` | Delta-hedged variant |
| `v31_pm_nba` | `nba_wp.py` | Distinct Polymarket NBA win-probability strategy |

"v31"/"v3.1" is a tuned **param generation** of `v3_theta_harvester`, not a separate
class. The later `v3_*` variants are the same module behind flags.

## Params & shared helpers

| File | Role |
|------|------|
| `theta_params.py` | `ThetaHarvesterParams` pydantic base + the `v3_*` `@register`s |
| `late_resolution.py` | `LateResolutionParams` + the v1 logic |
| `vol.py`, `_numba/vol.py`, `_numba/returns_buffer.py` | σ estimators (Parkinson, bipower, …) |
| `fee.py` | Shared fee curves (incl. `pm_binary` fee) |
| `_theta_math.py`, `regions.py` | Edge math, leg geometry |
| `intents.py`, `topup.py` | IOC intent builders, top-up sizing |
| `render.py` | Human-readable decision rendering (alerts/logs) |

One pydantic params base per strategy is the single source; the YAML config model
inherits it (`extra='forbid'`), guarded by `test_single_source_property`.

## Adding a strategy

Post-R6/R7 this is **one new module + one YAML block** (no edits to `config.py`,
`config_builders.py`, or `slot_config.py`):

1. Write the strategy module; `@register("<key>")` the factory; define its pydantic
   params model and `register_live_strategy(...)` (params_model, build, evaluate,
   reference_requirements).
2. Add a slot block to `config/strategy.yaml` (engine) / a grid to `config/tuning.*`
   (tuning).
3. Verify: `strategy_config_sig` for existing slots byte-identical; `make parity-gate`
   green; suite green.

**Conventions:** params don't transfer between HL and PM tracks — tune separately.
Don't strip defensive gates that don't fire in backtests (they earned their place
from live incidents).

## Related

`MarketState` / position math / decision kernel: [shared-spine.md](shared-spine.md).
How the engine vs sim invokes `evaluate()`: [engine.md](engine.md) / [backtest.md](backtest.md).
