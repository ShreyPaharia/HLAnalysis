# Recorder, adapters, events, data & config

Read when touching data capture, venue clients, the event taxonomy, the on-disk
layout, or the config files.

## Recorder — `hlanalysis/recorder/` (~0.4k LOC)

| File | Role |
|------|------|
| `main.py` | `hl-recorder` entry point |
| `runner.py` | Subscribes adapters, fans events into the writer |
| `writer.py` | Partitioned Parquet/DuckDB writer (hive layout below) |
| `read.py` | Read helper (use `pq.ParquetFile(p).read()`, not `read_table` — see gotcha) |

Long-running collector on EC2; archives to S3. **PM L2 books are not backfillable** —
this is the only source of truth for them.

## Adapters — `hlanalysis/adapters/` (~2.5k LOC)

| File | Venue / role |
|------|--------------|
| `_ws_base.py` | Shared WS client base: reconnect, exponential backoff |
| `base.py`, `composite.py` | Adapter protocol + multiplexer |
| `hyperliquid.py` | HL WS (books/trades/settlement); HIP-4 listing semantics |
| `binance.py`, `binance_klines.py` | Binance spot reference feed (`@trade` + REST `premiumIndex`; perp `aggTrade`/`markPrice` are geo-blocked from some IPs) |
| `polymarket.py` | PM CLOB WS + Gamma poller |
| `polymarket_gamma.py` | Gamma REST (market listing/resolution) |
| `polymarket_normalize.py` | PM frame → event normalizers (book best-first sorting lives here) |

## Event taxonomy — `hlanalysis/events.py`

One normalized event model (msgspec, tagged unions). Key event types: `TradeEvent`,
`BookSnapshotEvent`, `BookDeltaEvent`, `BboEvent`, `MarkEvent`, `OracleEvent`,
`QuestionMetaEvent`, `SettlementEvent`, `HealthEvent`. Enums: `ProductType`,
`Mechanism`, `EventType`. The backtester has a parallel read taxonomy
(`backtest/core/events.py`); `backtest/data/_parquet_schema.py` pins the shared column
names with contract tests.

## On-disk layout (hive partitions)

```
data/
  venue=<hyperliquid|binance|polymarket>/
    product_type=<perp|prediction_binary|...>/
      mechanism=<clob|...>/
        event=<trades|book_snapshot|bbo|settlement|...>/
          symbol=<SYMBOL or 76-digit PM token id>/
            date=YYYY-MM-DD/
              hour=<HH|all>/        # `all` = sealed daily-compacted file
                *.parquet
```

The ~18 GB recorded corpus lives **only in the main checkout's `data/`** (gitignored).
From a worktree, reuse it via `--data-root ../../data` — never `make pull-data`.
See [`../../CLAUDE.md`](../../CLAUDE.md) → "Git worktrees & data reuse".

## Config files — `config/`

| File | What |
|------|------|
| `strategy.yaml` | Live engine slots: the `strategies: [...]` list (per-slot params, allowlist, risk block) |
| `deploy.yaml` | Per-account creds via env (`hl_accounts.<alias>`) |
| `symbols.yaml` | Recorder subscriptions (HL/PM/Binance) |
| `tuning.yaml` | Default/canonical backtest grid |
| `tuning.<…>.yaml` | Per-market grids (btc/eth multistrike, eth updown) |
| `config/backtest/*.json` | Saved backtest run configs (HL bucket/PM WTI variants) |
| `run.v31_pm_nba.json`, `v1-final.json`, `v31_pm.json` | Live/run configs |
| `pm_liquidity_profile.json` | PM liquidity assumptions |

Per-experiment grids + their one-off runners are archived under `experiments/`
(historical, not maintained). Loader: `hlanalysis/config.py`.

## Gotchas

- **`import websockets` does not load submodules** — adapters explicitly
  `import websockets.exceptions`.
- **PyArrow partition merge error** on `pq.read_table(path)` for partitioned files —
  read individual files via `pq.ParquetFile(path).read()` instead.
- **PM next-day listing lag** — PM lists the next `btc-up-or-down-daily` ~2.5h after
  the prior day settles; Gamma returns zero active markets in between (expected,
  emits one `HealthEvent kind="no_active_markets"`).

## Related

How recorded data feeds the sim: [backtest.md](backtest.md). Deployment/archival
runbook: [`../../DEPLOYMENT.md`](../../DEPLOYMENT.md).
