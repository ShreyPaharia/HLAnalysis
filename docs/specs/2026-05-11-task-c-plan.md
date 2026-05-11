# Task C — PM data adapter (binary + multi-outcome bucket) — implementation plan

> Parent spec: `docs/specs/2026-05-11-backtester-rebuild.md` (also pasted in the
> dispatch prompt). This document is the per-task plan that the implementer
> follows.

**Goal.** Replace `hlanalysis/sim/data/polymarket.py` + `hlanalysis/sim/synthetic_l2.py`
with a single `DataSource` implementation under `hlanalysis/backtest/data/polymarket.py`
that exposes both PM BTC daily Up/Down binaries (existing behaviour) and PM's
hourly multi-strike BTC bucket events (new), behind the §3 `DataSource` protocol.

**Architecture.** PM events are read from Gamma + CLOB. For binaries: one PM
market = one question. For buckets: one PM event (slug `bitcoin-above-on-...-et`)
= one question with N×2 legs (N strike markets × {YES, NO}). Trades from PM CLOB
are mapped to synthetic L2 (one tick per trade, single level), interleaved with
Binance BTC klines as `ReferenceEvent`s. Settlement is emitted per-leg at
`end_ts_ns` so the runner's held-leg outcome lookup is symmetric across
binary / bucket.

**Tech stack.** Python 3.11, `requests`, `pyarrow`/`pyarrow.parquet`, `loguru`.

---

## 0. Research notes on PM bucket markets

Gamma exposes BTC "Bitcoin above ___ on <date>, <hour><am|pm> ET" hourly events
under the canonical series slug:

```
series_slug=bitcoin-multi-strikes-hourly
```

Each event has ~10 sub-markets, each a binary "BTC above $X at hour H" with its
own `conditionId` and YES/NO `clobTokenIds`. The strike is in
`market.groupItemTitle` (e.g. `"79,400"` → `79400.0`). The PM event itself has
its own `slug` / `ticker` / `id`; the markets sort naturally by ascending strike
(though we re-sort ascending in code to be safe).

Discovery URL pattern (verified live 2026-05-11):

```
GET https://gamma-api.polymarket.com/events
    ?series_slug=bitcoin-multi-strikes-hourly
    &closed=true
    &limit=500
    &offset=0
```

This mirrors the binary-series flow (`series_slug=btc-up-or-down-daily`) — the
only structural difference is `len(markets) > 2` (binary events have 1 PM
market = 1 (YES, NO) token pair; bucket events have N PM markets each with
their own (YES, NO) pair).

**Key wire facts:**

- Within each PM market in a bucket event, YES + NO = 1 (it's still a binary at
  the CLOB level). Within-pair parity inference (p ↔ 1−p) applies.
- **Across** the strike markets in a bucket event, prices are independent. No
  cross-market parity.
- Multiple strike-markets can resolve YES (e.g. if BTC closes at $82k then
  "above 79.4k" = YES, "above 79.8k" = YES, ..., "above 82.6k" = NO). This is
  fine: each leg pair settles independently.

## 1. Identifier mapping

- **Binary question** (PM Up/Down): `question_id = market.condition_id`;
  `leg_symbols = (yes_token, no_token)`; `klass = "priceBinary"`.
- **Bucket question** (PM hourly multi-strikes): `question_id = event.slug`
  (e.g. `bitcoin-above-on-may-10-2026-5pm-et`); `leg_symbols` is
  `(yes_o0, no_o0, yes_o1, no_o1, ...)` with the markets sorted ascending by
  strike; `klass = "priceBucket"`. Thresholds (strikes) are stored as a
  comma-joined string under `kv["priceThresholds"]` for compatibility with
  `hlanalysis/strategy/render.py` and `late_resolution.py`'s bucket handling.

`QuestionDescriptor.question_idx` (numeric, matches
`QuestionView.question_idx`): for both kinds, use
`hash(question_id) & 0x7FFFFFFF` — same trick as the existing
`sim/question_builder.build_question_view`. This keeps the strategy's
question-keyed bookkeeping consistent across binaries and buckets.

## 2. Settlement contract (per-leg)

§3.1 `SettlementEvent` already has `ts_ns`, `question_idx`, `outcome`. The
§3.4 narrative says "for buckets the data source emits the per-leg outcome via
repeated events keyed to leg symbols" — but the dataclass as written has no
symbol field. **We add an optional `symbol: str = ""` field in our mirror**
(additive change per the §4 rules); Task A will pick this up at integration.
The runner-side semantics:

- Binary question: emit two `SettlementEvent`s at `end_ts_ns`, one per leg —
  e.g. for the YES leg, `outcome="yes"` iff `resolved_outcome == "yes"`; for
  the NO leg, `outcome="yes"` iff `resolved_outcome == "no"`. The runner picks
  the one matching its held symbol.
- Bucket question: same, applied per (YES, NO) pair × N strikes — 2N events
  total. For pair `k` with strike `S_k`: YES-leg `outcome="yes"` iff
  `close_price > S_k`, NO-leg the negation.

This is symmetric and avoids the runner having to special-case binary vs
bucket. Order: emit all 2N events at `end_ts_ns` in leg-symbol order; the
runner indexes by symbol.

## 3. Cache layout

We keep the existing on-disk cache shape under `data/sim/` (the rename to
`data/backtest/` is task E's call — not in our fence). The current manifest
holds one entry per binary `condition_id`. We extend it to:

```json
{
  "0x51d0...": {                            // binary condition_id
    "n_rows": 1850,
    "last_pull_ts_ns": 1778...,
    "market": { ...PMMarket... },
    "kind": "binary"
  },
  "bitcoin-above-on-may-10-2026-5pm-et": {  // bucket event slug
    "n_rows": 17234,
    "last_pull_ts_ns": 1778...,
    "kind": "bucket",
    "bucket": {
      "event_slug": "bitcoin-above-on-may-10-2026-5pm-et",
      "start_ts_ns": ...,
      "end_ts_ns": ...,
      "thresholds": [79400.0, 79800.0, ..., 83400.0],
      "leg_tokens": [
        ["<yes_token_o0>", "<no_token_o0>"],
        ...
      ],
      "leg_condition_ids": ["0xab..", "0xcd..", ...],
      "leg_resolutions": ["yes", "no", "no", "no", ...]
    }
  }
}
```

Per-leg trades are stored under `pm_trades/<condition_id>.parquet` as today
(one parquet per leg's `condition_id`). The bucket manifest entry's
`leg_condition_ids` is how `events(q)` re-locates them. The "thresholds" sidecar
required by the spec is the `bucket.thresholds` field — kept inside the same
`manifest.json` rather than a separate file to keep the cache to a single
metadata blob.

## 4. Event stream

`PolymarketDataSource.events(q)` yields `MarketEvent`s in monotone non-decreasing
`ts_ns`. Construction:

1. Load PM trades for each leg `condition_id` (binary: 1 leg-pair; bucket: N
   leg-pairs).
2. Load BTC 1m klines covering `[q.start_ts_ns, q.end_ts_ns]`.
3. For each trade:
   - Emit `BookSnapshot` derived from the trade via `_synthetic_l2.trade_to_l2`
     (half-spread / depth from config — passed by the runner via cfg).
   - Emit a `TradeEvent` (aggressor side, price, size).
   - **Within-pair parity inference:** for the trade's leg symbol, find the
     other leg in its (YES, NO) pair (from `leg_pairs` mapping). Emit a
     synthetic `BookSnapshot` for the complementary token at `(1 − price)`
     (clipped to `[1e-6, 1-1e-6]`). This preserves the existing
     `sim/hftbt_adapter.build_event_stream` parity logic. **No cross-pair
     inference for buckets** — only within each (YES, NO) pair.
4. For each kline: emit one `ReferenceEvent(ts_ns=k.ts_ns, symbol="BTC",
   high=k.high, low=k.low, close=k.close)`.
5. At `end_ts_ns`: emit per-leg `SettlementEvent`s as in §2.

Merge ordering uses `heapq.merge` keyed by `ts_ns` (same approach as the old
`build_event_stream`).

## 5. File layout (this branch)

Create (within fence):

- `hlanalysis/backtest/__init__.py` — empty.
- `hlanalysis/backtest/data/__init__.py` — empty.
- `hlanalysis/backtest/core/__init__.py` — empty (Task A owns content of
  `core/`; we put a stub here to make the package importable for our
  mirrored interfaces).
- `hlanalysis/backtest/core/events.py` — local mirror of §3.1 (Task A will
  drop this during integration).
- `hlanalysis/backtest/core/data_source.py` — local mirror of §3.2.
- `hlanalysis/backtest/data/_synthetic_l2.py` — moved from
  `hlanalysis/sim/synthetic_l2.py`, behaviour unchanged.
- `hlanalysis/backtest/data/polymarket.py` — the new `DataSource`.
- `tests/unit/backtest/__init__.py` — empty.
- `tests/unit/backtest/test_polymarket_source.py` — unit tests.
- `tests/integration/test_backtest_pm_smoke.py` — integration smoke,
  exercising the new source against the migrated fixture.
- `tests/fixtures/pm/binary/` — migrated PM binary fixture (copy of
  `tests/fixtures/sim_smoke/`).
- `tests/fixtures/pm/README.md` — capture command.

Do not touch (fence violation):

- `hlanalysis/sim/*` (task E deletes later).
- `hlanalysis/strategy/*`.
- `hlanalysis/backtest/runner/*`, `hlanalysis/backtest/cli.py` (task A).
- `hlanalysis/backtest/data/hl_hip4.py` (task B).

## 6. Tasks (bite-sized, TDD where it makes sense)

> Convention: each task ends with a commit using
> `git commit -m "feat(backtest): …"` or `test(backtest): …` per Conventional Commits.

### Task 6.1 — Plan commit

- [ ] **Step 1: Write this plan** at `docs/specs/2026-05-11-task-c-plan.md`.
- [ ] **Step 2: Commit:**

```bash
git add docs/specs/2026-05-11-task-c-plan.md
git commit -m "docs(backtest): plan for Task C — PM data adapter (binary + bucket)"
```

### Task 6.2 — Mirror §3 interfaces

- [ ] **Step 1: Create `hlanalysis/backtest/__init__.py` + sub-package
  `__init__.py` files** (empty).
- [ ] **Step 2: Write `hlanalysis/backtest/core/events.py`** with the §3.1
  dataclasses + the additive `symbol: str = ""` field on `SettlementEvent`.
- [ ] **Step 3: Write `hlanalysis/backtest/core/data_source.py`** with the
  §3.2 `QuestionDescriptor` + `DataSource` protocol. Import `QuestionView`
  from `hlanalysis.strategy.types`.
- [ ] **Step 4: Quick smoke** — `python -c "from
  hlanalysis.backtest.core.events import BookSnapshot, TradeEvent,
  ReferenceEvent, SettlementEvent; print('ok')"` must print `ok`.
- [ ] **Step 5: Commit:**

```bash
git add hlanalysis/backtest/__init__.py hlanalysis/backtest/core/
git commit -m "feat(backtest): mirror §3 core interfaces for Task C (drop at integration)"
```

### Task 6.3 — Move synthetic_l2

- [ ] **Step 1:** Copy `hlanalysis/sim/synthetic_l2.py` to
  `hlanalysis/backtest/data/_synthetic_l2.py`, rewriting the import of
  `PMTrade` to a local minimal trade dataclass to avoid importing from
  `hlanalysis/sim/` (since task E will delete that). The trade-to-L2 logic
  is pure: takes a `price`, `ts_ns`, `token_id`, returns L2 snapshot. No
  schema validation needed at this seam.
- [ ] **Step 2: Commit:**

```bash
git add hlanalysis/backtest/data/__init__.py hlanalysis/backtest/data/_synthetic_l2.py
git commit -m "feat(backtest): move synthetic L2 helper into backtest.data"
```

### Task 6.4 — PM source (binary path) — TDD

Tests first (subset of test file).

- [ ] **Step 1: Write failing tests in
  `tests/unit/backtest/test_polymarket_source.py`** covering:
    - `test_descriptor_binary_basic` — given a constructed `PMMarket` +
      cached trades + cached klines, `discover()` returns one
      `QuestionDescriptor` with `klass="priceBinary"`, `leg_symbols ==
      (yes_token, no_token)`, `underlying="BTC"`, idx stable.
    - `test_events_monotone_ts_binary` — `events(q)` emits in monotone
      `ts_ns`, has at least one `ReferenceEvent`, one `BookSnapshot`, one
      `TradeEvent`, and two `SettlementEvent` at `end_ts_ns`.
    - `test_binary_within_pair_parity` — if the only trade is on `yes_token`
      at price 0.6, a complementary `BookSnapshot` for `no_token` at price
      ≈ 0.4 is also emitted at the same `ts_ns`.
    - `test_resolved_outcome_binary` — returns `"yes"` / `"no"` / `"unknown"`
      matching the manifest's `PMMarket.resolved_outcome`.
- [ ] **Step 2: Run them, watch fail** — `pytest
  tests/unit/backtest/test_polymarket_source.py -q`.
- [ ] **Step 3: Implement `PolymarketDataSource` in
  `hlanalysis/backtest/data/polymarket.py`**, scoped to the binary path:
    - `__init__(cache_root: Path)` — load `manifest.json`, prepare cache
      reader functions inline.
    - `discover(start, end, kind="binary", **filters)` — for binary, reuse
      `sim/data/polymarket.discover_btc_updown_markets` logic by porting
      it into the new module (no import from `sim/`; copy the function and
      keep `_BTC_UPDOWN_SERIES_SLUG = "btc-up-or-down-daily"`).
    - `events(q)` — load cached trades + klines, build the merged stream
      (synthetic L2 + within-pair parity + reference + per-leg settlement).
    - `question_view(q, now_ns, settled)` — re-create the QuestionView used
      by the strategy.
    - `resolved_outcome(q)` — return manifest's stored outcome.
- [ ] **Step 4: Run tests, expect pass** — `pytest
  tests/unit/backtest/test_polymarket_source.py -q`.
- [ ] **Step 5: Commit:**

```bash
git add hlanalysis/backtest/data/polymarket.py tests/unit/backtest/
git commit -m "feat(backtest): PM data adapter — binary path"
```

### Task 6.5 — PM source (bucket path) — TDD

- [ ] **Step 1: Add failing bucket tests** in the same file:
    - `test_descriptor_bucket_ordering` — given a constructed bucket event
      with 3 PM markets at strikes [80000, 81000, 82000], `discover(...,
      kind="bucket")` returns one descriptor with
      `klass="priceBucket"` and `leg_symbols ==
      (yes_o0, no_o0, yes_o1, no_o1, yes_o2, no_o2)` in ascending-strike
      order.
    - `test_threshold_parsing_and_sidecar` — the bucket's manifest entry
      after `discover_btc_bucket_markets` contains
      `bucket.thresholds == [80000.0, 81000.0, 82000.0]`. Round-trip
      through `manifest.json` (read back, check stable).
    - `test_bucket_no_cross_pair_parity` — a trade on `yes_o0` at 0.6 emits
      a within-pair complementary `BookSnapshot` at price 0.4 on `no_o0`,
      but NO synthetic emission on `yes_o1`/`no_o1`/etc.
    - `test_bucket_per_leg_settlement` — given resolved bucket where leg
      pair 1 won (BTC closed above strike_0 and strike_1, below strike_2),
      the events stream ends with 6 `SettlementEvent`s (2N=6 for 3 strikes)
      whose `outcome` values are
      `(yes_o0=yes, no_o0=no, yes_o1=yes, no_o1=no, yes_o2=no, no_o2=yes)`
      and `symbol` matches the leg.
- [ ] **Step 2: Add `discover_btc_bucket_markets(start_iso, end_iso)`** in
  `polymarket.py` — uses `_fetch_series_events("bitcoin-multi-strikes-hourly")`,
  filters by `_event_in_window`, parses each event into a bucket descriptor
  + per-leg PMMarket-ish records. Update the cache manifest with the bucket
  sidecar (`kind="bucket"`, `bucket.thresholds`, `bucket.leg_tokens`,
  `bucket.leg_resolutions`, `start_ts_ns`, `end_ts_ns`).
- [ ] **Step 3: Extend `events(q)`** to branch on `q.klass`. For buckets,
  iterate all leg pairs and apply within-pair parity per pair. Emit 2N
  settlement events at `end_ts_ns` (per-leg, with `symbol`).
- [ ] **Step 4: Extend `question_view(q, ...)`** to populate
  `qv.kv = (("priceThresholds", "80000,81000,..."),)` for buckets so
  `strategy/render.py` and `strategy/late_resolution.py` work as-is.
- [ ] **Step 5: Run** `pytest tests/unit/backtest/test_polymarket_source.py
  -q` — expect pass.
- [ ] **Step 6: Commit:**

```bash
git add hlanalysis/backtest/data/polymarket.py tests/unit/backtest/test_polymarket_source.py
git commit -m "feat(backtest): PM bucket markets (multi-strike) with per-leg settlement"
```

### Task 6.6 — Migrate PM smoke fixture + integration test

- [ ] **Step 1: Copy fixture:**

```bash
mkdir -p tests/fixtures/pm/binary
cp tests/fixtures/sim_smoke/{market.json,trades.json,klines.json} tests/fixtures/pm/binary/
```

- [ ] **Step 2: Write `tests/fixtures/pm/README.md`** documenting how
  the fixture was captured (copied from `sim_smoke`, originally created by
  `tests/integration/test_sim_pm_smoke.py`'s capture script — refer to the
  PR that introduced sim_smoke). One-paragraph description; nothing fancy.
- [ ] **Step 3: Write `tests/integration/test_backtest_pm_smoke.py`** —
  builds a `PolymarketDataSource` pointed at a tmp cache with the fixture
  contents loaded, calls `discover()`, `events()`. Asserts:
  - event counts: total events match expected (computed from fixture).
  - the new event stream produces the same per-market realized P&L
    (within $0.01) as `sim/runner.run_one_market` does — by running BOTH
    in the test: old runner gives the baseline; a tiny in-test replay
    loop drives the new event stream through `strategy.evaluate` +
    `simulate_fill` and accumulates fill cash flows. (This isolates Task C
    from Task A's runner.)
- [ ] **Step 4: Run** `pytest tests/integration/test_backtest_pm_smoke.py -q`
  — expect pass.
- [ ] **Step 5: Commit:**

```bash
git add tests/fixtures/pm/ tests/integration/test_backtest_pm_smoke.py
git commit -m "test(backtest): migrate PM smoke fixture and integration test"
```

### Task 6.7 — Live bucket discovery sanity check (manual)

- [ ] **Step 1:** Manually run a tiny script that wraps
  `discover_btc_bucket_markets("2026-05-10", "2026-05-11")` and prints
  count + the first event's `(slug, n_legs, thresholds[:3])`. Confirm
  count ≥ 1.
- [ ] **Step 2:** Note the command + output in the PR body. (No commit —
  this is a runtime acceptance.)

### Task 6.8 — Open PR

- [ ] **Step 1:** `git push -u origin feat/backtest-pm-source`.
- [ ] **Step 2:** `gh pr create --title "feat(backtest): PM data adapter
  (binary + multi-outcome bucket)" --body @-` with body covering:
    - link to parent spec + this plan,
    - bucket discovery filter (`series_slug=bitcoin-multi-strikes-hourly`),
    - fixtures listed,
    - manual `hl-bt fetch --data-source polymarket --kind bucket ...`
      stand-in command output from 6.7.

## 7. Self-review

- **Spec coverage:** every Task C acceptance bullet maps:
  - "Existing PM binary smoke reproduces — ±$0.01" → Task 6.6 step 3.
  - "`hl-bt fetch --data-source polymarket --kind bucket ...` finds at
    least one BTC bucket market" → Task 6.7 (stand-in; the CLI itself is
    Task A's, so we exercise the discovery function directly and document
    in PR).
  - "Unit tests cover bucket leg ordering, threshold parsing, parity-not-
    applied for buckets" → Task 6.5 step 1.
- **Type consistency:** `QuestionDescriptor.question_idx` is `int` and
  must match `QuestionView.question_idx` — both use
  `hash(question_id) & 0x7FFFFFFF`. `leg_symbols` is `tuple[str, ...]` in
  both.
- **Placeholders:** none. Every task lists exact files, exact tests, exact
  commands.

## 8. Risk + escape hatch

Per parent spec §8 risk #3: if PM's Gamma API stops surfacing
`bitcoin-multi-strikes-hourly` cleanly (or rate-limits us), Task 6.5 lands
binary-only and we document the blocker in the PR. Binary parity (Task 6.4
+ 6.6) must still ship.

