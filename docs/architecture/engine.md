# Engine — live/paper trading runtime

`hlanalysis/engine/` (~10.5k LOC). Read when touching the live runtime: loops, slot
isolation, order routing, risk, reconciliation, diagnostics.

## Entry & shape

`engine/main.py` → `runtime.py`. The engine runs **N `(strategy, account)` slots in
one process**, sharing one WebSocket feed and one `MarketState`; everything else is
per-slot isolated. `runtime.py` drives five concurrent async loops:

| Loop | Responsibility |
|------|----------------|
| `_ingest_loop` | Single shared WS subscription → apply events to `MarketState`; reconnect backoff; fire `market_dirty` on price moves; latch all slots halted after repeated feed failure. |
| `_scan_loop` (per slot) | Event-driven. Wakes on `market_dirty`, coalesces bursts, floors at an idle interval. `scanner.scan()` → `strategy.evaluate()` → `router`. |
| `_reconcile_loop` (per slot) | Merge venue truth into the local ledger; detect drift. |
| `_checks_loop` | Watchdogs: stale-data, daily-loss, RSS/memory. |
| `_heartbeat_loop` | One heartbeat line per slot. |

## File map

| Area | Files |
|------|-------|
| Orchestration | `runtime.py` (loops), `main.py` (CLI), `_runtime_helpers.py`, `_slot_builder.py` (build an `AccountSlot`) |
| Config | `config.py` (schema + `strategy_config_sig`), `config_builders.py` (build strategy obj — reused by `hl-bt --slot`) |
| Scan / decision | `scanner.py` (per-slot scan → `RiskInputs`), `market_state.py` (thin adapter over `marketdata.MarketState`) |
| Risk / routing | `risk.py` (12-rule veto + breakers), `risk_events.py`, `router.py` (order routing, ~950 LOC) |
| Execution | `exec_client.py` / `exec_types.py` (venue-neutral protocol), `hl_client.py`, `pm_client.py`, `_venue_io.py` |
| PM specifics | `pm_strike.py` (strike resolution), `pm_watchdogs.py` |
| State / persistence | `state.py` (SQLite DAL), `trade_journal.py`, `event_bus.py`, `events_sink.py`, `migrations_alembic/` |
| Reconcile / drift | `reconcile.py`, `reconcile_report.py` (true-PnL vs venue), `restart_drift.py` (the `restart_blocked` gate) |
| Replay / diag | `replay.py` (`hl-replay`, engine path over recorded data), `diag.py` (`make engine-diag`) |

## Gotchas

- **`restart_blocked`** — on restart, the reconcile merge can detect ghost/orphan/
  position-mismatch drift and write `data/engine/<alias>/restart_blocked`; the scanner
  stays suspended until you investigate and `rm` it. Check `make engine-status`.
- **Per-alias namespacing** — with >1 account, state DB / kill switch / drift flag /
  cloid prefix are namespaced under `data/engine/<alias>/`. Single-account stays flat.
- **`paper_mode`** (default true) gates only the real REST POST; every other path runs.
- **Crash isolation** — a strategy exception is caught in `_scan_loop`; only that
  slot's tick drops, never the WS feed or sibling slots.
- **HL is venue-authoritative** for positions/PnL (local ledger can drift); PM is
  local-ledger. `reconcile_report.py` encodes this.

## Related

Decision/risk code that's *shared with the backtester* lives in [shared-spine.md](shared-spine.md).
Ops commands (`engine-status`, `engine-diag`, `reconcile-report`) are in [`../../CLAUDE.md`](../../CLAUDE.md).
