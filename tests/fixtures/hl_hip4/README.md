# HL HIP-4 test fixture

A small slice of the recorder's `data/venue=hyperliquid/...` parquet captured
for offline tests of `hlanalysis/backtest/data/hl_hip4.py`.

**Contents:** one `priceBinary` question (Q1000015, BTC, expiry 2026-05-10
06:00 UTC, `targetPrice=80354`) over the last 2 hours before its expiry.

- Window: `[2026-05-10 04:00 UTC, 2026-05-10 06:00 UTC)`
- `question_meta(Q1000015)` — 2 rows
- `market_meta(#150, #151)` — 4 rows per leg
- `book_snapshot(#150, #151)` — ~13.3k rows per leg (20 levels each)
- `trade(#150, #151)` — ~560 rows per leg
- `settlement(...)` — empty (recorder didn't capture this question's
  settlement event; the data source's fallback infers outcome from the last
  HL perp BTC reference price vs `targetPrice`)
- `perp BTC bbo` — ~33.7k rows
- `perp BTC mark` — ~8k rows
- Total size: ~1.2 MB (well under the 5 MB cap)

## Capture command

```bash
# Run from repo root with the project's .venv active.
START=$(python -c "import datetime as d; print(int(d.datetime(2026,5,10,4,tzinfo=d.timezone.utc).timestamp()*1e9))")
END=$(python -c   "import datetime as d; print(int(d.datetime(2026,5,10,6,tzinfo=d.timezone.utc).timestamp()*1e9))")

python scripts/capture_hl_hip4_fixture.py \
    --data-root data \
    --out-root tests/fixtures/hl_hip4 \
    --question-symbol Q1000015 \
    --start-ns "$START" --end-ns "$END"
```

## Why a binary (and not a bucket)

A binary has only 2 legs (`#150`/`#151`), keeping the fixture small without
sacrificing path coverage — `events()`, `question_view()`, and the
`resolved_outcome()` fallback are exercised end-to-end. Bucket-specific paths
are unit-tested separately against synthetic descriptors.
