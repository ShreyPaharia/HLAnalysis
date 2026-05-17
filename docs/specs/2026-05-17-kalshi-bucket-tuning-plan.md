# Kalshi bucket-tuning sanity check — follow-up plan

**Status:** Deferred. Not started. Pick this up when the sandbox network policy
allows `api.elections.kalshi.com`, or when a local fetch + commit workflow is
established.

**Context:** PR #8 (`feat/v1-buckets-and-sizing`) shipped the `size_cap_near_strike`
sizing rule tuned on the 1y PM BTC daily-binary corpus, and abandoned the bucket
half of the original plan because PM's `bitcoin-multi-strikes-hourly` series
uses an "above-X stack" layout incompatible with the strategy's HL
mutually-exclusive `priceBucket` convention (see write-up in
`docs/v1-bucket-sizing-results.md`, local-only). The strategy's `priceBucket`
allowlist entry in `config/strategy.yaml` therefore ships with `size_cap`
disabled, pending bucket-shaped data to tune against.

This plan covers the next-cheapest unlock: a Kalshi adapter on the existing
`hlanalysis/backtest/core/data_source.py` seam, just enough to confirm whether
the v1 bucket logic produces a non-zero edge on **any** HL-shaped corpus.

---

## 1. Why this and not something else

Three candidates were considered for unblocking bucket tuning. Ranked by
expected effort × confidence the result is informative:

| Option | Effort | Geometry match | Cadence match | Verdict |
|---|---|---|---|---|
| Wait for HL HIP-4 bucket history to accumulate | 0 (just time) | Perfect | Perfect | Best long-run answer; useless this quarter |
| Add `above_x_stack` layout mode to `_winning_region` and reuse PM hourly corpus | Days | Synthesized | Perfect | High value but couples strategy code to venue conventions; defer until we've validated the logic survives at all |
| Ingest Kalshi bracket corpus (this plan) | ~2–3 days | Native (mutually exclusive) | Hourly/daily available | **Sanity check only** — confirms the bucket rule isn't structurally broken before investing in (2) or building HL HIP-4 paper history |
| Synthesize HL buckets from PM stack (`P(a<BTC<b) = P(>a) − P(>b)`) | Days | Synthesized | Perfect | Clean math, but fill/slippage composition doesn't compose linearly across legs; high implementation risk |

**Scope of this plan = sanity check only.** Output is "yes the bucket logic
produces hit rate > 0 and PnL > 0 on a cleanly-shaped corpus" or "no it's still
broken." Not a shippable HL calibration.

## 2. Sandbox constraint

The remote execution environment used for development blocks outbound calls to
`api.elections.kalshi.com` (HTTP 403, host not in allowlist). PM is blocked
identically; the existing PM cache was populated outside the sandbox and
committed. Two unblock paths:

- **Preferred:** Add `api.elections.kalshi.com` to the environment's network
  policy. Then everything below runs end-to-end in one session.
- **Fallback:** Implement `fetch_and_cache(...)` exactly like
  `polymarket.py:fetch_and_cache` and run it on a local machine with network
  access, commit the resulting `data/sim/kalshi/manifest.json` + parquet, push.
  Sweep then runs in-sandbox against the cache.

Either way the adapter's read path must be sandbox-friendly: discovery and
event emission read from the on-disk manifest only.

## 3. Kalshi data shape (to confirm before writing the adapter)

Open questions that need a live API probe before any non-trivial code is
written. Each should be answered with a one-line `curl` against the public
docs at `https://trading-api.readme.io/`:

1. **Series identity.** Confirm the active BTC bracket series ticker. Most
   recent public reference is `KXBTCD` (daily) and the hourly variant; confirm
   from `/series?category=Crypto` and pick the highest-cadence one with > 30
   days of settled history.
2. **Outcome layout.** Each Kalshi market is one bracket. Verify that an event
   groups markets with strictly partitioning ranges (e.g. `<$60k`, `$60–65k`,
   `$65–70k`, …, `>$X`) so exactly one market per event settles YES. Reject
   the series if outcomes overlap.
3. **Settlement field.** Find the field that tells us which market in the event
   resolved YES. Likely `result` or `settle_value` on `/markets/{ticker}`.
4. **Trade history availability without auth.** `/markets/trades?ticker=...`
   is documented as public but pagination + rate limits unclear. Need a real
   call to estimate fetch cost for a 30-day corpus.
5. **Orderbook snapshots.** Almost certainly auth-only. Assume we synthesize
   L2 from trades using `hlanalysis/backtest/data/_synthetic_l2.py`, same
   approach `polymarket.py` already takes.
6. **Tick conventions.** Kalshi prices in cents (1–99); normalize to [0,1]
   ask/bid at parse time so downstream strategy code is venue-agnostic.

Do not start writing `kalshi.py` until these are resolved — a wrong assumption
on (2) is the entire bug we're trying to avoid repeating from the PM bucket
attempt.

## 4. File-by-file deliverables

All paths relative to repo root. Mirror the `polymarket.py` structure since
that adapter already solved 80% of the same problems (manifest layout,
synthetic L2, settlement parsing, discover/events split).

### `hlanalysis/backtest/data/kalshi.py` (new, ~400 lines)

- `_KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"`
- `_BTC_BRACKET_SERIES_TICKER = "KXBTCD"` (or whatever §3.1 confirms)
- HTTP helpers: `_http_get` matching the PM signature.
- `_fetch_series_events(ticker)` — paginated `/events?series_ticker=...`.
- `_parse_bracket_event(ev)` — return a manifest record with:
  - `kind = "bucket"`
  - `bucket.thresholds = [lo_of_market_1, lo_of_market_2, ...]` (N−1 thresholds for N markets, in HL convention)
  - `bucket.leg_tokens = [[yes_tok, no_tok], ...]` aligned to thresholds
  - `bucket.start_ts_ns`, `bucket.end_ts_ns`, `bucket.settlement_outcome_idx`
- `KalshiDataSource` (subclass of `QuestionDataSource` per
  `hlanalysis/backtest/core/data_source.py`):
  - `discover(start, end, kind="bucket")` — read `data/sim/kalshi/manifest.json`, filter by `endDate`
  - `events(q)` — yield `BookSnapshot`/`TradeEvent`/`ReferenceEvent`/`SettlementEvent` from cached parquet, exactly like `polymarket.py:_events_bucket`
- `fetch_and_cache(start, end, cache_root)` — offline fetcher; safe to run outside the sandbox. Reuses `_synthetic_l2.trade_to_l2`.

### `hlanalysis/backtest/data/_kalshi_settlement.py` (new, ~80 lines)

Just the settlement parser. Kept separate so the manifest-shape contract is
testable without an HTTP layer.

### `hlanalysis/backtest/cli.py` (modify ~5 lines)

Register `"kalshi"` as a `--data-source` choice in `cmd_tune` and `cmd_run`.
The discovery `kind` filter added in PR #8 already does the right thing.

### `config/tuning.v1-buckets-kalshi.yaml` (new, ~45 lines)

Mirror `config/tuning.v1-buckets.yaml` exactly — same 108-cell grid, same
splits (30/30/25). The whole point is to vary only the data source.

### `tests/unit/test_kalshi_adapter.py` (new, ~150 lines)

- Settlement parser tests using captured JSON fixtures (commit 2–3 real
  events under `tests/fixtures/kalshi/`).
- Manifest-round-trip test that the parsed thresholds match HL convention
  (mutually exclusive, ascending, one settlement outcome).
- Discover/events tests with a synthetic 2-market bucket event.

No live HTTP in tests — fetch tests live under `tests/integration/` and are
skipped when `KALSHI_API_KEY` (or just network) is unavailable.

## 5. Sweep & result interpretation

Once the corpus is cached:

```
uv run hl-bt tune \
  --data-source kalshi \
  --grid config/tuning.v1-buckets-kalshi.yaml \
  --kind bucket \
  --start 2026-04-01 --end 2026-05-01 \
  --out data/sim/tuning/v1-buckets-kalshi/
uv run python scripts/analyze_sweep.py \
  --run-dir data/sim/tuning/v1-buckets-kalshi/ \
  --sort profitable
```

**Pass criteria for "bucket logic generalizes":**

- ≥1 grid cell with mean hit_rate > 30% across splits (vs 0% on PM bucket attempt)
- ≥1 grid cell with total PnL > $0 across splits
- No structural anomaly in the per-leg diagnostic dump (use
  `scripts/plot_bucket_trace.py` on 3–5 trades — confirm the held leg's
  "winning region" matches the actual settled outcome)

**Fail criteria → do not invest further:**

- Hit rate still ~0%: the strategy logic itself has a deeper bug independent
  of layout. Block on debugging `_winning_region` and `_safety_d_for_region`
  before any more tuning.
- Hit rate high (>60%) but PnL negative: fill/slippage assumptions wrong;
  audit `_synthetic_l2` against real Kalshi book snapshots before trusting
  any backtest.

**Even on pass: do not ship Kalshi-tuned params to HL HIP-4.** Microstructure
differs (fees, lot sizes, taker dominance, intraday liquidity). The signal
this experiment produces is "the logic works in principle," not "these are
production params."

## 6. Overfitting + venue-transfer notes

Two layers of risk stack here, both worth flagging in the commit message and
in `config/strategy.yaml` if any params are ever ported from Kalshi:

1. **Same-corpus overfitting** — the v1 binary calibration already shows
   suspicious signs (see PR #8 discussion: 12-trade snipe at grid-extreme
   `pct=1.0`). Any Kalshi sweep needs the same scrutiny: held-out window,
   sensitivity test on threshold knobs, reject if optimum sits at a grid edge.
2. **Venue transfer** — Kalshi → HL is a cross-venue extrapolation. Tick
   conventions, fee model, lot sizing, and taker/maker mix all differ. A
   clean Kalshi backtest is a *prior* on HL behavior, not a calibration. The
   only honest path to a live HL bucket calibration is paper-trading on
   HIP-4 once enough markets have settled.

## 7. Done criteria for this follow-up

- [ ] Section 3 questions answered with API probe output committed to this doc
- [ ] `kalshi.py` adapter passes unit tests with fixture data
- [ ] One month of cached BTC bracket data exists at `data/sim/kalshi/`
- [ ] `data/sim/tuning/v1-buckets-kalshi/` results.jsonl produced
- [ ] Pass/fail verdict against §5 criteria appended to this doc
- [ ] If pass: separate ticket opened for the `above_x_stack` layout mode
  (which is the actual production unlock)

## 8. Out of scope

- Trading on Kalshi (this is a backtest-only data adapter)
- HL HIP-4 paper trading harness
- Strategy code changes — if Kalshi reveals a bug in `_winning_region` or
  sizing, that's a separate PR
- Cross-venue arbitrage analysis (Kalshi vs PM vs HL price dislocations)
