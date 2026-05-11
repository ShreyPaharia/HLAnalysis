# Task B — HL HIP-4 recorded-data `DataSource` (implementation plan)

**Date:** 2026-05-11
**Branch:** `feat/backtest-hl-hip4-source`
**Spec:** `docs/superpowers/specs/2026-05-11-backtester-v2-design.md` (the parent design — agent B's tasking message reproduces it verbatim).
**Scope:** Implement `hlanalysis/backtest/data/hl_hip4.py` against the §3 contracts.

## 1. Goals + non-goals

**Goals**
- Provide a `DataSource` impl that walks the recorder's parquet partitions and yields the spec's `MarketEvent` union in monotonic-`ts_ns` order.
- Support both `priceBinary` and `priceBucket` HIP-4 questions over BTC.
- Reconstruct L2 from `event=book_snapshot` rows (full top-N levels per snapshot).
- Surface an HL reference-price stream for the runner's BTC return buffer.
- Carry settlement when present; fall back to mark-vs-strike inference otherwise.

**Non-goals (this task)**
- Anything outside `hlanalysis/backtest/data/hl_hip4.py` and the test/fixture paths in §4 of the design. The §3 modules belong to task A; until A merges I work against a local mirror.
- The runner, CLI, registry, plots, tuning — task A.
- The PM source — task C.
- Numba acceleration — task D.
- New strategies, recorder changes, engine changes.

## 2. Recorded data — confirmed schemas (UTC, ns)

`data/venue=hyperliquid/product_type=prediction_binary/mechanism=clob/event=<type>/symbol=<sym>/date=YYYY-MM-DD/hour=HH/*.parquet`

- `event=question_meta`, `symbol=Q<idx>` — Pydantic `QuestionMetaEvent` from `hlanalysis/events.py`:
  `question_idx`, `named_outcome_idxs[]`, `fallback_outcome_idx`, `settled_named_outcome_idxs[]`, parallel `keys[]/values[]`.
  Examples confirmed live: Q0–Q3 are `priceBucket` (`named=[7,8,9]/fallback=6`, etc.), Q1000020 is a `priceBinary` (`named=[20]`).
- `event=market_meta`, `symbol=#<outcome*10+side>` — `keys[]/values[]` carrying `outcome_idx`, `side_idx`, `class`, `underlying`, `expiry`, `targetPrice` (binary) or `priceThresholds` (bucket via owning question), `period`.
- `event=book_snapshot`, `symbol=#<leg>` — `bid_px[]`, `bid_sz[]`, `ask_px[]`, `ask_sz[]` (parallel arrays, best-first).
- `event=trade`, `symbol=#<leg>` — `price`, `size`, `side ∈ {buy,sell,unknown}`, plus `block_ts/buyer/seller/block_hash`. Recorder uses `exchange_ts = block_ts` for HL trades.
- `event=settlement`, `symbol=#<leg>` — `settled_side_idx`, `settle_price`, `settle_ts`. HL adapter sets `symbol` to the winning leg coin; the `_detect_polled_settlements` fallback writes one row per newly-settled outcome's YES coin with `settled_side_idx=0`.
- HL BTC perp reference: `product_type=perp/mechanism=clob/event=bbo/symbol=BTC` and `event=mark/symbol=BTC`. Perp BBO is the densest (≈2.3M rows historical, ~100–150 rows/h current); perp mark ≈half that.

**Leg-symbol naming** (verified): `#<outcome_idx*10 + side_idx>` where `side_idx=0` → YES, `=1` → NO. e.g. outcome 22 → `#220` YES / `#221` NO.

## 3. Reference-price source — pick one

The §3.5 contract emits `ReferenceEvent(high, low, close)` per tick. Choices:

| source | density (last 24h) | shape | latency | notes |
|---|---|---|---|---|
| perp `BTC` bbo | ~100–150 rows/h | mid = (bid+ask)/2 | tick-level | broadest, freshest |
| perp `BTC` mark | ~half of bbo | mark_px | tick-level | HL's official mark — but recorder samples it less |
| spot `@142` bbo | ~552k total / 24h-ish | mid | tick-level | spot ≠ what HIP-4 settles on |
| spot `@142` mark | ~263k total | mark_px | tick-level | spot — wrong reference |

**Decision: perp BTC BBO mid.** Rationale:
1. HIP-4 binaries reference HL perp mark (per the engine design and prior notes), so perp is the correct underlying — spot is a distractor.
2. Perp BBO is denser and lower-latency than perp mark. We synthesise `ReferenceEvent(high=last_mid, low=last_mid, close=last_mid)` per BBO tick (collapsing H/L to mid is honest at tick granularity — there is no 1m bar in the recorded stream).
3. The strategy's recent-returns buffer downsamples on the runner side, so the source can ship raw per-tick refs and let downstream sampling decide.

The runner's `recent_hl_bars` slot expects HLC. We satisfy the type but assume the runner builds bars from the dense stream rather than treating each `ReferenceEvent` as a 1m bar. If task A wants real OHLC, it can resample.

If perp BBO is empty over the window (gappy capture), the implementation falls back to perp mark; if both empty, it falls back to spot @142 BBO and logs a `Diagnostic`-style warning via `logger.warning`. Tests pin BBO as the primary path.

## 4. Module shape

```
hlanalysis/backtest/data/hl_hip4.py
├── HLHip4DataSource (class)
│     name = "hl_hip4"
│     __init__(data_root: Path = Path("data"), ...)
│     discover(start, end, underlying="BTC", ...)
│     events(q)            # generator
│     question_view(q, *, now_ns, settled)
│     resolved_outcome(q)
├── _local_mirror/  (inline at top of module — see §6)
│     QuestionDescriptor, BookSnapshot, TradeEvent, ReferenceEvent, SettlementEvent, DataSource (Protocol)
└── helpers:
      _parse_kv(keys, values) -> dict[str,str]
      _leg_outcome_side(symbol) -> (outcome_idx, side_idx)
      _resolve_klass_from_qm(qm_row) -> "priceBinary" | "priceBucket"
      _expiry_ns_from_kv(kv) -> int
      _strike_from_kv(kv, klass) -> float
```

## 5. Discovery

`discover(start: str, end: str, underlying: str = "BTC", *, kinds=("priceBinary","priceBucket")) -> list[QuestionDescriptor]`

Algorithm:
1. DuckDB-scan `event=question_meta/symbol=Q*/**/*.parquet` once with `hive_partitioning=1` filtering on date partition `[start, end]` (one-day pad on each side to catch boundary cases — the question_meta row is emitted at question-creation time, which can fall in the previous UTC day).
2. For each row, parse `values[]` keyed by `keys[]` and the question's `description` to recover `class`, `underlying`, `expiry`, `priceThresholds` (bucket) or `targetPrice` (binary), `period`.
3. Filter by `underlying`, `klass ∈ kinds`, and `expiry_ns ∈ [start_ns, end_ns)`.
4. Build leg symbols deterministically:
   - `priceBinary`: legs = (`#<o0*10+0>`, `#<o0*10+1>`) where `o0 = named_outcome_idxs[0]`.
   - `priceBucket`: legs ordered as `(o[0] YES, o[0] NO, o[1] YES, o[1] NO, ..., fallback YES, fallback NO)`. Fallback always last.
5. `start_ts_ns = min(question_meta.exchange_ts)` for the question. `end_ts_ns = expiry_ns`. If multiple `question_meta` rows are present (re-emit), pick the earliest.
6. Return a list sorted by `start_ts_ns`.

Edge case: if `question_meta` rows for a question span a period change (HL rolls), keep the first. The recorder re-emits `question_meta` only on `settled_named_outcome_idxs` change, so this is rare in our window.

## 6. Local interface mirror

Until task A's PR is merged, the §3.1/§3.2 dataclasses + `DataSource` Protocol live as a module-level block at the top of `hl_hip4.py`. Module docstring documents the temporary mirror with an explicit `TODO(task-E): drop mirror and import from hlanalysis.backtest.core` marker. The mirror types are NOT re-exported.

This avoids cross-branch coupling: if A names a field differently I just have to update the mirror — the public DataSource methods don't change.

Tests also do not import from the mirror — they import from the (eventual) public path through a tiny test-only shim or just from `hl_hip4` directly.

## 7. Event stream construction (`events(q)`)

Strategy: lazy DuckDB scans per event type, then a `heapq.merge` over typed iterators. We do NOT load all parquets into one big arrow table — for production windows that would blow RAM.

Steps:
1. For each leg `s ∈ q.leg_symbols`:
   - Stream `book_snapshot` parquet partitions for `s` filtered to `[q.start_ts_ns, q.end_ts_ns]`.
     Yield `BookSnapshot(ts_ns=exchange_ts, symbol=s, bids=tuple(zip(bid_px, bid_sz)), asks=tuple(zip(ask_px, ask_sz)))`.
   - Stream `trade` parquet partitions for `s` filtered same way.
     Yield `TradeEvent(ts_ns=exchange_ts, symbol=s, side="buy" if side in ("buy","unknown") else "sell", price=price, size=size)`.
     `side="unknown"` collapses to `"buy"` (taker default) — only used as aggressor flag downstream.
2. Stream HL perp BTC BBO partitions for `[q.start_ts_ns, q.end_ts_ns]`. Yield `ReferenceEvent(ts_ns=exchange_ts, symbol="BTC", high=mid, low=mid, close=mid)` with `mid = (bid_px+ask_px)/2`. Fall back to perp `mark` if BBO empty (and emit a warning).
3. Stream `event=settlement` for the question's legs in the same window. Yield `SettlementEvent(ts_ns=settle_ts, question_idx=q.question_idx, outcome="yes" if settled_side_idx==0 else "no")` for each row.
4. `heapq.merge` the per-source iterators by `ts_ns` (stable; intra-tick order: book → trade → ref → settlement, by source ordering at merge time).

DuckDB pushdown: each scan uses a parameterised `WHERE exchange_ts BETWEEN ? AND ?` plus a `date IN (...)` hive-partition predicate so we only touch hour-bucket files that intersect the window. Date-list derived from `start_ts_ns / end_ts_ns` rounded down/up.

Monotonicity guarantee: each per-source iterator is already sorted by `exchange_ts` (we add `ORDER BY exchange_ts`); `heapq.merge` preserves the global ordering. Test asserts monotone non-decreasing on a captured 2h slice.

## 8. `question_view(q, *, now_ns, settled)`

Build a fresh `QuestionView` from cached question_meta + market_meta rows:
- `question_idx = q.question_idx`
- `klass = q.klass`
- `underlying = q.underlying`
- `expiry_ns = q.end_ts_ns`
- `period` from kv (e.g. `"1d"`)
- For `priceBinary`: `yes_symbol = q.leg_symbols[0]`, `no_symbol = q.leg_symbols[1]`, `strike = float(targetPrice)`.
- For `priceBucket`: `yes_symbol=""`, `no_symbol=""`, `strike = float(priceThresholds[0])` (lowest threshold). Spec acknowledges binary-only `yes/no_symbol` fields; we mirror existing semantics from `hlanalysis/sim/question_builder.py`.
- `leg_symbols = q.leg_symbols`
- `settled` per arg; `settled_side` from `resolved_outcome(q)` only if `settled`.
- `name = "Recurring"` (from question_meta `question_name` kv); `kv` mirrors the question_meta keys/values pairs verbatim.

I/O: cached per-`question_id` on the source instance. First call scans `question_meta` + `market_meta` parquet for the question's `symbol=Q*` and `symbol=#<legs>` once; subsequent calls reuse.

## 9. `resolved_outcome(q)`

Order of resolution:
1. If `event=settlement` has any row with `symbol ∈ q.leg_symbols`: pick the earliest by `settle_ts` and return `"yes" if settled_side_idx==0 else "no"`. For buckets the runner consumes per-leg SettlementEvents and `resolved_outcome` is informational; we still return a coherent answer (`"yes"` if any leg's `settled_side_idx==0`).
2. Else (gap in recorder coverage): infer from the *last* HL perp BTC mark/BBO mid at-or-before `q.end_ts_ns`:
   - `priceBinary`: `"yes" if last_btc > strike else "no"`.
   - `priceBucket`: pick the bucket that contains `last_btc` (using `priceThresholds`); return `"yes"` if that bucket is the held leg's outcome, `"no"` otherwise. But since the source-level `resolved_outcome` is per-question (not per-leg), for buckets we return `"unknown"` and rely on the runner consuming per-leg SettlementEvents.
3. Else (no reference data): return `"unknown"`.

Implementation note: keep the fallback inference path tested at unit level (synthesise a tiny mark-stream parquet at fixture time).

## 10. Fixture capture

**Choice:** Q3 (priceBucket, BTC, expiry 2026-05-11 06:00 UTC, thresholds 79043/82270, named=[22,23,24], fallback=21). 8 legs. 2-hour window `[2026-05-10 14:00 UTC, 2026-05-10 16:00 UTC)` — mid-life, dense (~6660 book/h × 8 legs ≈ 106k snapshots; trades ~2k/h × 8 ≈ 32k trades; HL perp BTC bbo ~200 rows). Estimated total: ~3 MB parquet after compression. Under the 5 MB cap.

No `settlement` row in this window (expiry is at 06:00 UTC = 8h after window end). Smoke test exercises the mark-vs-thresholds fallback and the strategy's no-trade behavior (v1 is near-resolution-arb and won't fire mid-life). Report.md still gets written with zero-trade summary stats, satisfying "non-empty report.md".

**Capture command** (written into `tests/fixtures/hl_hip4/README.md`):

```bash
# Run from repo root. Requires the .venv with duckdb installed.
START=$(python -c "import datetime as d; print(int(d.datetime(2026,5,10,14,tzinfo=d.timezone.utc).timestamp()*1e9))")
END=$(python -c   "import datetime as d; print(int(d.datetime(2026,5,10,16,tzinfo=d.timezone.utc).timestamp()*1e9))")

python scripts/capture_hl_hip4_fixture.py \
    --data-root data \
    --out-root tests/fixtures/hl_hip4 \
    --question-symbol Q3 \
    --start-ns "$START" --end-ns "$END"
```

The capture script copies a hive-partitioned slice for each of: `question_meta(Q3)`, `market_meta(#21*, #22*, #23*, #24*)`, `book_snapshot(<legs>)`, `trade(<legs>)`, `settlement(<legs>)` (empty), `perp BTC bbo` (window slice). The slice preserves the hive partition layout so the data source can scan it with the same query.

Script lives at `scripts/capture_hl_hip4_fixture.py` (added by this PR; small, tested manually).

## 11. Tests

`tests/unit/backtest/test_hl_hip4_source.py`
- `test_leg_symbol_decode` — `_leg_outcome_side` parses `#220 → (22, 0)` etc.
- `test_discover_returns_q3` — discover() finds Q3 in the fixture's window and `leg_symbols` matches expected ordering.
- `test_events_monotonic_ts_ns` — iterate `events(q)` once; assert `ts[i] >= ts[i-1]` for all consecutive ticks.
- `test_book_snapshot_matches_parquet` — sample 10 random book_snapshot ticks from the fixture's raw parquet, find the corresponding `BookSnapshot` in `events()`, assert `bids/asks` lists match (within float exact equality).
- `test_resolved_outcome_fallback` — with a hand-built mark stream, `resolved_outcome` returns `"unknown"` for the bucket fixture (since the window doesn't include expiry).

`tests/integration/test_backtest_hl_hip4_smoke.py`
- Builds a `HLHip4DataSource(data_root=Path("tests/fixtures/hl_hip4"))`, runs `discover()`, picks the only question, consumes `events(q)` end-to-end into a list, asserts:
  - at least 100 events
  - ordering monotone
  - at least one `BookSnapshot`, one `TradeEvent`, one `ReferenceEvent`
- Stubs the runner via a tiny inline harness that tallies event types and writes a `report.md` to `tmp_path`. Asserts the file is non-empty.
- Marked `@pytest.mark.integration`.

Once task A lands, integration test rewrites to call `hl-bt run --strategy v1_late_resolution --data-source hl_hip4 ...` instead of the inline harness. The DataSource API doesn't change.

## 12. Commit plan

1. `docs(specs): task B plan for HL HIP-4 DataSource` — this doc.
2. `feat(backtest): HL HIP-4 recorded-data DataSource` — implementation + script.
3. `test(backtest): fixture + unit/integration tests for HL HIP-4 source` — fixture parquet + tests.

## 13. Open questions / risks

1. **Settlement semantics** — `_detect_polled_settlements` writes `settled_side_idx=0` for all newly-settled outcomes, but only the winning outcome should pay YES. The fallback inference handles this for binaries; for buckets the runner consumes per-leg settlements and the strategy decides. Documented in module docstring.
2. **HL perp BTC density** — confirmed dense enough (~100/h BBO). If a future capture is sparser, the mark fallback kicks in.
3. **Multiple question_meta rows** — Q0/Q2 had 3 rows; we use the earliest. Q3 had 1.
4. **Cross-branch drift** — local mirror tracks §3 exactly; task E will drop it when A merges. Mirror documented in module docstring.
