# Task A — Implementation plan

**Spec:** see the top of this branch: `feat/backtest-core` (specification reproduced in the dispatch prompt).
**Scope:** Core + hftbacktest runner + CLI + registry. The §3 interface contracts (`events`, `data_source`, `registry`, runner) are the seam tasks B/C/D consume.

## 1. hftbacktest API choice

Installed package: `hftbacktest==2.4.4` (`HashMapMarketDepthBacktest` + `BacktestAsset`, both jitclass-backed, callable from pure Python). I confirmed via a smoke test against `.venv/bin/python`:

- `BacktestAsset()` is fluent: `.data(np.ndarray[event_dtype])`, `.tick_size(...)`, `.lot_size(...)`, `.linear_asset(1.0)`, `.constant_order_latency(0, 0)`, `.risk_adverse_queue_model()`, `.no_partial_fill_exchange()`, `.trading_value_fee_model(maker, taker)`, `.last_trades_capacity(N)`.
- `HashMapMarketDepthBacktest([asset0, asset1, ...])` returns a jitted backtest object.
- Control flow: call `hbt.elapse(duration_ns)` to advance simulated time. Returns `0` on success, `1` at end-of-data.
- Submit orders with `hbt.submit_buy_order(asset_no, oid, px, qty, IOC, LIMIT, wait=True)` / `submit_sell_order(...)`. IOC = 3, LIMIT = 0 (per `hftbacktest.order`).
- Read state: `hbt.depth(asset_no).best_bid/.best_ask`, `hbt.position(asset_no)`, `hbt.orders(asset_no)`, `hbt.state_values(asset_no).fee` etc.

**This is sync, not callback-only — fits the spec's per-question loop with no surprises.** No fundamental control-flow mismatch (spec §8 risk #1).

### Event feed format

`event_dtype = (ev:u8, exch_ts:i8, local_ts:i8, px:f8, qty:f8, order_id:u8, ival:i8, fval:f8)` with `ev` a bitmask combining:

- A kind: `DEPTH_EVENT (1)`, `TRADE_EVENT (2)`, `DEPTH_CLEAR_EVENT (3)`, `DEPTH_SNAPSHOT_EVENT (4)`, `DEPTH_BBO_EVENT (5)`.
- A side: `BUY_EVENT (1<<29)` (bid in depth events; aggressor=buy in trades), `SELL_EVENT (1<<28)` (ask side).
- Visibility: `EXCH_EVENT (1<<31) | LOCAL_EVENT (1<<30)` — set both so feed is consumed at both processors.

Depth updates are absolute: `qty=0` at a price removes the level. For top-of-book replacement we emit a `qty=0` event at the prior best price before the new event, so stale levels don't accumulate.

### Per-question, one runtime per call

Each question gets its own `HashMapMarketDepthBacktest`. Legs (e.g. YES, NO) map 1:1 to hftbacktest assets in `QuestionDescriptor.leg_symbols` order. Reference-price events (BTC klines) are NOT pushed to hftbacktest — they stay in our `MarketState` for `recent_returns` / `recent_hl_bars` and `reference_price`, exactly mirroring sim/market_state's contract.

After `elapse` returns 1 (end of data), we synthesize settlement: pay 1.0 if the held leg won, 0.0 if it lost.

## 2. File-by-file deliverables

All paths relative to repo root.

### `hlanalysis/backtest/__init__.py`
Just the package marker; re-exports nothing user-visible (imports happen at module level inside cli.py).

### `hlanalysis/backtest/core/events.py`
Spec §3.1 verbatim. `MarketEvent` is `BookSnapshot | TradeEvent | ReferenceEvent | SettlementEvent`. All frozen, slots, `ts_ns: int` first.

### `hlanalysis/backtest/core/data_source.py`
Spec §3.2 verbatim. `QuestionDescriptor` dataclass + `DataSource` Protocol. Re-exports `MarketEvent` for the `events(...)` return type.

### `hlanalysis/backtest/core/registry.py`
Spec §3.3 verbatim. `register(strategy_id)` decorator, `build(strategy_id, params)`, `ids()`. Module-private `_REGISTRY: dict[str, Callable[[dict[str, Any]], Strategy]]`.

### `hlanalysis/backtest/core/question.py`
Re-exports `QuestionView` from `hlanalysis.strategy.types`. Provides one builder helper, `build_question_view(descriptor, *, strike, now_ns, settled, settled_side)`, that wraps the existing strategy `QuestionView` constructor with sensible defaults. The PM/HL sources will call this builder.

### `hlanalysis/backtest/runner/result.py`
- `Fill` (cloid, symbol, side, price, size, fee, partial) — value-equivalent to `sim/fills.Fill`.
- `DiagnosticRow`, `FillRow` (parquet-friendly dataclasses) — moved from `sim/diagnostics.py`.
- `RunResult` (fills, n_decisions, realized_pnl_usd, diagnostics rows).
- `RunSummary` + `summarise_run` — moved from `sim/metrics.py`.
- `write_diagnostics`, `write_fills` — pyarrow writers, identical schema to sim/.

### `hlanalysis/backtest/runner/market_state.py`
- `MarketState` — same `apply_l2/apply_trade_ts/apply_kline/book/recent_returns/recent_hl_bars/latest_btc_close` shape as `sim/market_state.SimMarketState`, but the L2 input is the abstract `BookSnapshot` instead of `L2Snapshot`. Field names and method signatures stay identical so task D can swap in numba helpers without touching the runner.
- Created by task A so D has a known target file.

### `hlanalysis/backtest/runner/hftbt_runner.py`
The heart. Public API:

```python
@dataclass(frozen=True, slots=True)
class RunConfig:
    scanner_interval_seconds: int
    tick_size: float
    lot_size: float
    slippage_bps: float     # added on the marketable IOC limit
    fee_taker: float
    book_depth_assumption: float
    underlying: str = "BTC"

def run_one_question(
    strategy: Strategy,
    data_source: DataSource,
    q: QuestionDescriptor,
    cfg: RunConfig,
    *,
    diagnostics_dir: Path | None = None,
    fills_dir: Path | None = None,
) -> RunResult: ...
```

Algorithm:

1. Collect all events from `data_source.events(q)` upfront. Partition into:
   - per-leg `BookSnapshot` / `TradeEvent` → hftbacktest event arrays (built leg-by-leg with prior-level removal logic).
   - `ReferenceEvent` → keep as list; fed to `MarketState` in chronological order during the scan loop.
   - `SettlementEvent` → keep as list; consumed at end-of-data.
2. Build one `BacktestAsset` per leg (`tick_size`, `lot_size`, `linear_asset(1.0)`, `risk_adverse_queue_model`, `no_partial_fill_exchange`, `constant_order_latency(0, 0)`, `trading_value_fee_model(0.0, cfg.fee_taker)`, `last_trades_capacity(256)`, `.data(arr)`).
3. Construct `HashMapMarketDepthBacktest([asset...])`.
4. Scan loop: in `cfg.scanner_interval_seconds` steps, call `hbt.elapse(interval_ns)`. After each step:
   - Replay any `ReferenceEvent`s whose ts_ns ≤ now into `MarketState`.
   - Compute `now_ns = hbt.current_timestamp`.
   - Build `QuestionView` via `data_source.question_view(q, now_ns=now_ns, settled=False)`.
   - Build `books: Mapping[str, BookState]` from each leg's depth (`best_bid/.best_ask/.best_bid_qty/.best_ask_qty`). Symbol comes from `q.leg_symbols[i]`.
   - Call `strategy.evaluate(...)` with `recent_returns`, `recent_hl_bars`, `reference_price=state.latest_btc_close()`.
   - If `Decision.ENTER` with one OrderIntent and no current position: submit IOC at `max(ask, intent.limit_price) * (1 + slippage_bps/1e4)` (capped to 1.0 for binary tokens). Record fill if `position(asset_no)` changes.
   - If `Decision.EXIT`: submit reduce-only IOC sell at `bid * (1 - slippage_bps/1e4)`.
   - Stop-loss check: if a held position's leg's best bid drops below `pos.stop_loss_price`, auto-submit a sell. Mirrors sim's `cfg.stop_loss_pct` pull from `strategy.cfg.stop_loss_pct`.
   - Persist diagnostics + fills metadata as in sim/.
5. End of data: settle any open position at 1.0/0.0 based on `data_source.resolved_outcome(q)`.

**Limit-vs-slippage formula** preserved from sim/fills.py: IOC limit binds when the ask + slippage cost would push above the limit; otherwise the slipped price is taken. Bound to [0, 1].

**hftbacktest fees** are configured to 0.0 — fee accounting uses the simple `px * size * fee_taker` formula from sim/fills.py and is added on top of the realized P&L. This matches sim/ behaviour exactly (sim never used hftbacktest's internal fee accounting because it never used hftbacktest at all).

### `hlanalysis/backtest/runner/walkforward.py`
Move from `sim/walkforward.py`, unchanged behaviour. Parameterised over an iterable of `QuestionDescriptor` (or any sequence).

### `hlanalysis/backtest/data/binance_klines.py`
Move from `sim/data/binance_klines.py`, unchanged. Provides `Kline` dataclass + `fetch_klines`.

### `hlanalysis/backtest/data/synthetic.py`
Test-only in-memory `DataSource`. Constructor takes a single hand-crafted `QuestionDescriptor` plus pre-built lists of `BookSnapshot`, `TradeEvent`, `ReferenceEvent` and a resolved outcome. `events()` yields them in ts_ns order. NOT a production source.

### `hlanalysis/backtest/tuning.py`
Grid + walk-forward + parallel runner. Same shape as `sim/tuning.py` but the strategy lookup goes through the registry (`hlanalysis.backtest.core.registry.build`), and the tuning YAML now reads `grid` keyed by strategy id (`grids: { v1_late_resolution: {...}, v2_model_edge: {...} }`). For task A we keep the layered structure but don't add strategies — the registry stays empty until E.

### `hlanalysis/backtest/report.py`
Move `sim/report.py` largely intact. Drops the plot imports that depend on PM-only data; for now we keep equity_curve and skip calibration/vol_realized plots (those reference PM-specific fill metadata and are non-essential to acceptance). The per-market table reads from the same `fills.parquet` schema and works for any `DataSource`.

### `hlanalysis/backtest/cli.py`
`hl-bt` entrypoint. Subcommands:

- `strategies` — print `registry.ids()`.
- `run --strategy <id> --data-source <name> --config <json> --out-dir <dir> [--start ...] [--end ...] [--scanner-interval-seconds N] [--slippage-bps] [--fee-taker] [--tick-size] [--lot-size] [--depth]`
  - `data-source` selects between `synthetic`, `polymarket`, `hl_hip4`. For task A only `synthetic` is wired (B+C land the rest).
  - Calls `discover()` then loops `run_one_question` over each descriptor; writes report + diagnostics + fills.

`fetch` and `tune` subcommands are scaffolded but data-source-specific fetching is left to B/C (the synthetic source needs no fetch).

### pyproject changes
Add `hl-bt = "hlanalysis.backtest.cli:main"`. Do NOT remove `hl-sim` (task E does that).

## 3. Synthetic DataSource shape (test only)

Hand-built one-question source. Question id `synth-q-0`, one BTC binary, ~10 minutes of life, 1 leg YES + 1 leg NO. Events include:
- Initial book snapshots for each leg.
- 5–10 trades.
- A handful of BTC reference klines.
- A settlement event at end_ts_ns picking outcome="yes".

A dummy strategy for the synthetic smoke test: in test, register a stub strategy that emits one `Decision.ENTER` on YES at scan #2 and HOLDs thereafter. This satisfies the "produces fills.parquet + diagnostics.parquet" acceptance without depending on `v1_late_resolution` (which task E registers).

## 4. Test strategy

- `tests/unit/backtest/test_core.py`: dataclass invariants (frozen, slot, ts_ns ordering), registry behaviour (register + duplicate detection + build + ids).
- `tests/unit/backtest/test_registry.py`: edge cases — unknown id, repeated register raises, isolation across test functions (clean registry per test via a fixture).
- `tests/unit/backtest/test_hftbt_runner.py`: build a one-leg synthetic source, run with a stub strategy, assert (a) at least one ENTER fill, (b) the recorded fill's price/size are sane, (c) settlement payout is applied, (d) realized_pnl_usd matches expected hand calc.
- `tests/integration/test_backtest_synthetic_smoke.py`: invoke the CLI end-to-end via `hlanalysis.backtest.cli.main(['run', ...])` and assert `report.md`, `fills.parquet`, `diagnostics.parquet` are created.

## 5. Commit sequencing

One commit per logical unit, all conventional-commits:

1. `docs(spec): task A implementation plan` — this doc.
2. `feat(backtest): core interface contracts (events, data_source, registry, question)` — pure interface, no runtime deps.
3. `feat(backtest): runner state, result types, walkforward` — `runner/{result,market_state,walkforward}.py`, includes `Fill`/`RunResult`/`RunSummary` and parquet writers.
4. `feat(backtest): hftbacktest-driven per-question runner` — `runner/hftbt_runner.py`.
5. `feat(backtest): binance klines + synthetic in-memory data source` — `data/{binance_klines,synthetic}.py`.
6. `feat(backtest): tuning, report, CLI scaffolding` — `tuning.py`, `report.py`, `cli.py`, pyproject `hl-bt` entrypoint.
7. `test(backtest): unit tests for core, registry, runner + synthetic smoke integration` — `tests/...`.

## 6. Open assumptions / spec deviations to flag at integration (E)

- **CLI `tune` subcommand**: I'm renaming the YAML key from `v1_grid`/`v2_grid` to a single `grids: { <strategy_id>: {...} }` map (spec §1 motivation #3 calls out the hardcoded keys as a pain point). If task E disagrees, the change is one-file (`cli.py`) and the loader (`tuning.py`).
- **`hbt.elapse(scan_interval_ns)`** is the *control* knob; if the recorder writes events at sub-second resolution and scan interval is 60s, every elapse may process thousands of events. This is the intended behaviour. The runner does NOT do per-event scan ticks — only every scanner_interval_seconds, identical to sim/.
- **fee_taker**: configured into hftbacktest's `trading_value_fee_model` AND mirrored as a multiplier on the recorded `Fill.fee`. To avoid double-counting, hftbacktest is configured with `(0.0, 0.0)` and the recorded Fill carries the full taker fee. The Fill's recorded P&L (used by `summarise_run`) matches sim/ exactly.
- **Stop-loss**: re-uses `strategy.cfg.stop_loss_pct` if present (mirrors sim). If absent, no stop. Strategies built by B/C/D agents shouldn't depend on the runner offering this — they can express stops via their own `Decision.EXIT` logic.
- **plots/**: not ported in this round. The acceptance only requires `report.md`; the calibration/vol-realized plots are PM-specific and follow in a later cleanup pass.
