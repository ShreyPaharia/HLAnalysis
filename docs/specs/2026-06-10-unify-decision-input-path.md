# SHR-97 — Unify the decision-input path (engine ≡ backtest by construction)

**Status:** spec (Phase 1) · **Ticket:** SHR-97 (parent SHR-37) · **Date:** 2026-06-10

## 1. Problem

SHR-81/86/87 already share the MarketState **core math** (OHLC bucketer, σ
estimators, rolling-volume window) between the live engine and the backtest
(`hlanalysis/marketdata/market_state.py`). But the layer *around* the core — how
recorded/live events become reference ticks, at what cadence, from which price
source — is still **two parallel wirings** that drift. Every recent live-vs-sim σ
divergence (SHR-92/93/95/96) was the backtest re-deciding, by hand and with a
non-live default, a wiring choice the engine already makes from config.

| Wiring layer | Live engine | Backtest (today) | Divergent? |
| --- | --- | --- | --- |
| MarketState core (bucketer, σ) | shared | shared | ✅ shared (SHR-81/86/87) |
| Reference **source** (mark/bbo) | `reference_sigma_source` config, default **mark** | `--ref-event` CLI, default **bbo** | ❌ SHR-92 |
| Reference **price** (tick vs bar-close) | raw ticks → `apply_reference_tick` | bar-close, opt-in raw via `--reference-ticks`, default **bars** | ❌ SHR-93 |
| σ **cadence** (dt) | `reference_sampling_dt_seconds(cfg)` per symbol | `reference_resample_seconds` (coupled to `vol_sampling_dt_seconds`) + `set_reference_cadence` only in raw mode | ⚠️ coupled but conditionally wired (SHR-96 bug) |
| Scan cadence | event-driven (0.2/2s) | `--scan-mode`, default fixed | ❌ SHR-95 (**out of scope**, see §6) |

The bugs do not live in the shared core. They live in these divergent feeding
defaults. **The fix is to make the backtest derive its decision-input wiring
from the same strategy config the engine uses, so σ / p_model / recent_returns
are bit-identical to live by construction — not re-set as backtest-only CLI
knobs with non-live defaults.**

## 2. The two paths, precisely

### 2.1 Live (engine)

- **Ingest:** WS → adapter → `NormalizedEvent` (`MarkEvent` / `BboEvent` /
  `TradeEvent` / …) → `EngineRuntime._ingest_loop` → `MarketState.apply(ev)`
  (`hlanalysis/engine/market_state.py:244`).
- **Reference price:** `MarkEvent` (mark-sourced) or `BboEvent` mid (bbo-sourced)
  → `self._core.apply_reference_tick(symbol, ts_ns, price)` — **always a raw tick**,
  folded into the per-`(symbol, dt)` OHLC bucket in-place. `last_mark` is the
  instantaneous tick (`market_state.py:255-288`).
- **Config resolution (the "engine cadence port"):**
  `EngineRuntime._register_reference_cadences` (`runtime.py:394-420`) reads, per
  slot, from `StrategyConfig`:
  - cadence — `reference_sampling_dt_seconds(cfg)` (`config_builders.py:186`) plus
    every per-class theta-override cadence → `MarketState.set_reference_cadence`;
  - source — `cfg.reference_sigma_source` (default `"mark"`,
    `engine/config.py:292`) → `MarketState.set_reference_source` (fail-fast on
    conflict).
- **Read (Scanner):** at each scan `now_ns`, `scanner.py:361-386` calls
  `recent_returns` / `recent_hl_bars` with `now_ns` + `lookback_seconds` (the
  SHR-66 time-bounded window) at the class's resolved `dt`, then
  `strategy.evaluate(...)`.

### 2.2 Backtest

- **Ingest:** recorded parquet → `HLHip4DataSource` (`backtest/data/hl_hip4.py`)
  → `ReferenceEvent` bars **or** raw ticks → `hftbt_runner.run_one_question`
  drains them in the scan loop (`hftbt_runner.py:1305-1316`):
  - bars mode → `state.apply_reference(ev)` → `apply_reference_bar` (append
    pre-bucketed bar);
  - raw mode → `state.apply_reference_tick(ev)` → `apply_reference_tick` (live path).
- **Source / price / cadence are chosen independently of the engine config:**
  - source — `SourceConfig.hl_ref_event`, fed from `--ref-event` (default `"bbo"`),
    selects bbo-mid vs mark in `_hl_hip4_fastpath.py:159-223`;
  - price — `SourceConfig.hl_ref_ticks`, fed from `--reference-ticks`
    (default `"bars"`), selects bar-close vs raw tick;
  - cadence — `SourceConfig.reference_resample_seconds` is coupled to
    `params["vol_sampling_dt_seconds"]` (`cli.py:335`) ✅, but the runner only
    calls `state.set_reference_cadence(dt)` **when `ref_events_are_raw_ticks`**
    (`hftbt_runner.py:1186-1188`) — so in bars mode the buffer silently keeps the
    60s default. This conditional wiring is the SHR-96 interaction bug.
- **Read:** `hftbt_runner.py:1384-1398` — `state.recent_returns_and_hl(now_ns,
  lookback_seconds=cfg.vol_lookback_seconds)`, `state.latest_btc_close()`, then
  `strategy.evaluate(...)`. Same shared-core query surface as the engine.

### 2.3 Shared core

`hlanalysis/marketdata/market_state.py` (SHR-81). Both adapters wrap one instance.
The reference-tick path (`apply_reference_tick` → `_OhlcBuffer.ingest_tick` →
`slice_window`) is identical for both; given the **same tick (ts, price) stream
at the same dt**, σ / recent_returns / recent_hl / last_mark are bit-identical.
So the *only* thing that can make sim ≠ live is the **wiring that produces that
tick stream + dt** — which §3 unifies.

## 3. Design — single config-driven decision-input wiring

### 3.1 Shared resolver (new)

Add `hlanalysis/marketdata/decision_input.py`:

```python
@dataclass(frozen=True)
class DecisionInputConfig:
    reference_source: str          # "mark" | "bbo"  (which price feeds σ/p_model)
    sampling_dt_seconds: int       # OHLC bucket width == vol_sampling_dt_seconds
    vol_lookback_seconds: int      # σ window (history sizing / warm-up)
    reference_ticks: str           # "raw" (live) | "bars" (legacy override)
```

Two thin constructors onto the **same** struct — this is the "single
config-driven decision-input path" the acceptance asks for:

- `from_engine(cfg: StrategyConfig) -> DecisionInputConfig` — wraps the existing
  engine reads verbatim: `reference_source=cfg.reference_sigma_source`,
  `sampling_dt_seconds=reference_sampling_dt_seconds(cfg)`,
  `vol_lookback_seconds=reference_vol_lookback_seconds(cfg)`,
  `reference_ticks="raw"` (live always raw). **Returns exactly today's engine
  values — the engine stays bit-identical (§5 gate proves it).**
- `from_backtest_params(params: dict, *, track_default_source: str)
  -> DecisionInputConfig` — `reference_source = params.get("reference_sigma_source")
  or track_default_source` (HL → `"mark"`), `sampling_dt_seconds =
  params.get("vol_sampling_dt_seconds", 60)`, `vol_lookback_seconds = max
  vol_lookback across classes` (reuse `_derive_reference_warmup_seconds`'s logic),
  `reference_ticks = "raw"`.

The resolver is the **one** place that decides source / dt / ticks. The engine
calls it where it reads those fields today; the backtest calls it when building
`SourceConfig`.

### 3.2 Backtest derives, no longer hand-sets

In `cli.py` `_source_config_from_args` / `cmd_run` / `cmd_tune` (and
`parallel.py` if it rebuilds): populate `SourceConfig.hl_ref_event`,
`hl_ref_ticks`, `reference_resample_seconds` from
`DecisionInputConfig.from_backtest_params(...)` **instead of from the CLI flag
defaults**. The CLI flags (`--ref-event`, `--reference-ticks`) become *explicit
overrides*: when the user passes one it wins; when omitted, the value is
config-derived (no longer hard-defaulted to `bbo` / `bars`).

Implementation rule: change the arg `default` from `"bbo"`/`"bars"` to `None`,
and in `_source_config_from_args` use `getattr(args, "ref_event", None) or
resolved.reference_source` (and likewise for ticks). This makes the **derived
default live-faithful** (mark + raw) while keeping the override path for A/B.

Consequence (intended): the default HL backtest flips from bbo/bars to
**mark/raw** — i.e. live-faithful. This is the whole point of SHR-97; it WILL
change recorded HL numeric results. Fixture/golden tests that pinned the old
divergent default must be updated to the new config-derived expectation (§4).
Keep the override flags so a legacy A/B run is still one flag away.

### 3.3 Cadence always wired (SHR-96 structural fix)

With `reference_ticks="raw"` as the derived default, the runner's
`set_reference_cadence` call (`hftbt_runner.py:1186-1188`) always fires, so the
OHLC buffer is always dt-bucketed. The SHR-96 "raw × event ⇒ 0 trades" class of
bug (cadence left at 60s) is structurally gone — `set_reference_cadence` is no
longer gated on a mode that can be off. (Belt-and-braces: the runner may call
`set_reference_cadence` unconditionally from `cfg`/source dt, not only in raw
mode — verify bit-identity in bars mode, where it is a no-op per the existing
comment.)

## 4. Bit-identical GATE (the non-negotiable Phase-3 acceptance)

New test `tests/unit/test_decision_input_engine_sim_parity_shr97.py`:

1. Take a recorded HL reference tick stream (reuse the
   `tests/fixtures/hl_hip4` corpus the SHR-87 test loads), as `(ts_ns, price)`.
2. Build a `DecisionInputConfig` (source `mark`, some dt, e.g. 5 and 60).
3. **Engine path:** feed the stream as `NormalizedEvent`s
   (`MarkEvent`/`BboEvent` per source) into `engine.MarketState` configured via
   `set_reference_cadence` + `set_reference_source` from the resolver.
4. **Backtest path:** feed the *same* stream as raw `ReferenceEvent`s into
   `backtest.runner.MarketState` with `set_reference_cadence(dt)` from the same
   resolver, via `apply_reference_tick`.
5. At a grid of sampled `now_ns`, assert **bit-identical**:
   `recent_returns`, `recent_hl_bars`, `last_mark`, and `sigma` for each
   estimator (`stdev`, `bipower`, `parkinson`) — exact equality
   (`np.array_equal` / `==` on floats, not `allclose`).
6. Cover both a sub-minute dt (5s, the SHR-96 regime) and 60s, and both `mark`
   and `bbo` sources.

This extends the SHR-87 core gate to the **feeding**: same recorded events +
same resolved config ⇒ identical decision inputs on both paths.

Also assert (cheap unit test) that `from_engine(cfg)` and
`from_backtest_params(params)` agree on `(source, dt, lookback, ticks)` for a
representative HL slot config / params pair — i.e. the two constructors are one
config path.

## 5. Engine bit-identity

The engine read/ingest code does **not** change behaviourally. `from_engine`
returns the same values the engine reads inline today; `_register_reference_cadences`
may be refactored to call it but must produce identical `set_reference_cadence` /
`set_reference_source` calls. The standing SHR-87 replay-parity test
(`test_market_state_shr87_replay_parity.py`) must stay green unchanged, and the
new §4 gate is additive. Fill-sim (hftbacktest asset build, `hbt.elapse`, book
reads from the hftbacktest timeline) is untouched.

## 6. Scope boundaries

**In scope:** the decision-input wiring — reference **source** (SHR-92),
**price tick-vs-bar** (SHR-93), **cadence registration** (SHR-96) — collapsed
into / derived from the shared config; the §4 gate.

**Out of scope (keep separate, by ticket design):**
- **Fill/execution** path — hftbacktest's own L2 timeline; you cannot replay
  live fills. Untouched.
- **Event source** — recorded parquet vs live WS. The backtest keeps its loader;
  we do not route it through the live WS adapter.
- **Engine adapter reuse wholesale** — the engine `MarketState` owns the
  snapshot-coupled L2 **book** + question/settlement registry, which the backtest
  must NOT use (it reads books from the hftbacktest timeline). Reusing it whole
  would entangle the fill path. The correct seam is the shared *wiring config* +
  the two thin shared-core adapters, proven equal by §4 — not one adapter class.
- **Scan cadence (SHR-95 `--scan-mode`)** — changes *when* decisions fire, not
  the σ/p_model inputs at a given `now_ns`; it is fill-timing, not decision-input.
  Not in the SHR-97 acceptance knob list (92/93/96). Leave its default as-is;
  the §4 gate samples fixed `now_ns` and is scan-mode-independent. Note as a
  follow-up.

## 7. File-level change list

- **new** `hlanalysis/marketdata/decision_input.py` — `DecisionInputConfig` +
  `from_engine` + `from_backtest_params`.
- **edit** `hlanalysis/engine/runtime.py` — `_register_reference_cadences` calls
  `from_engine(slot.cfg)` (bit-identical output).
- **edit** `hlanalysis/backtest/cli.py` — `--ref-event` / `--reference-ticks`
  defaults → `None`; `_source_config_from_args` (+ run/tune paths) populate
  `SourceConfig` via `from_backtest_params`, flag = explicit override.
- **edit** `hlanalysis/backtest/core/source_config.py` — doc the derived
  (live-faithful) defaults; no field changes required.
- **edit** `hlanalysis/backtest/runner/hftbt_runner.py` — (optional) wire
  `set_reference_cadence` unconditionally from cfg dt; verify bars-mode no-op.
- **edit** `hlanalysis/backtest/runner/parallel.py` — ensure the per-cell
  `with_reference_resample` path still routes through the resolver.
- **new** `tests/unit/test_decision_input_engine_sim_parity_shr97.py` — §4 gate.
- **edit** affected fixture/golden tests that encoded the old bbo/bars default →
  update to the config-derived (mark/raw) expectation; keep override-flag tests.

## 8. Acceptance (mirrors SHR-97)

1. Single config-driven decision-input path (the resolver) shared by engine +
   backtest; backtest no longer hard-sets MarketState source/cadence/ticks
   independently.
2. **Bit-identical gate green:** σ / p_model(recent_returns) / recent_hl_bars /
   last_mark from the backtest path == the engine path on the same recorded
   events (§4).
3. SHR-92 `--ref-event`, SHR-93 `--reference-ticks`, SHR-96 `set_reference_cadence`
   collapse into / derive from the shared config (overrides retained, defaults
   live-faithful).
4. Full suite green; engine decision behaviour bit-identical (SHR-87 gate
   unchanged); fill-sim untouched.
