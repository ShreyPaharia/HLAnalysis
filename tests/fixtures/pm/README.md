# PM fixtures

`binary/` — one resolved BTC Up/Down daily market (`condition_id` in
`market.json`) plus its PM CLOB trades and the matching window of Binance
BTC/USDT 1m klines. Copied from `tests/fixtures/sim_smoke/`, originally
captured by an earlier sim-era smoke harness.

To re-capture:

```bash
hl-sim fetch \
    --start <YYYY-MM-DD> --end <YYYY-MM-DD> \
    --cache-root /tmp/pm-fixture-capture \
    --min-trades 30 --min-volume-usd 1000
```

then copy the per-market `manifest.json`'s entry, its `pm_trades/<cond>.parquet`,
and the kline JSON. The smoke test in `tests/integration/test_backtest_pm_smoke.py`
loads these three files (`market.json`, `trades.json`, `klines.json`).
