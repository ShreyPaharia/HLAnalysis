# Phase 1 — Reliability Hardening (on the current t4g.micro) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The engine cannot lose money through operational failure faster than edge makes it, **within the 1 GB box** (no infra upgrade). Close the open SHR reliability tickets, verify those that already landed, and prove the dangerous live paths with tests.

**Architecture:** Targeted fixes inside existing engine modules (`market_state`, `recorder/writer`, `router`, `reconcile`, `risk`, `runtime`, the PM adapter), each TDD'd against the existing paper-loop harness (`tests/integration/test_engine_paper_loop.py`, with `_FakeAdapter` / `_FakeTelegram`). Memory tickets come first because we are deliberately not buying RAM.

**Tech Stack:** Python 3.12, asyncio engine runtime, sqlmodel/sqlite, pyarrow recorder, pytest.

Spec: `docs/specs/2026-06-07-desk-scaling-phase-0-1-design.md` (Phase 1 §5).

---

## Investigation findings that reshaped this plan (read first)

The codebase moved ahead of the ticket text. Confirmed by reading the source:

- **SHR-45 (reject circuit-breaker) is ALREADY IMPLEMENTED** in `router.py`: `_reject_breaker_threshold` (default 5), per-`(question_idx, side)` counter `self._consecutive_rejects`, pre-place suppression, trip log, and reset-on-fill in `_book_fill`. → Task 3 is **verify + regression-test + close the ticket** (optionally add per-error-class visibility), NOT a rebuild.
- **SHR-46 is narrower than written.** Pending-row-before-`place()` and reconcile fill-replay already exist. The real gap: the reconcile local-ghost branch (`reconcile.py` ~138-179) marks the order `filled` and replays `Fill` rows but **does NOT apply the net delta to the `Position` table**, so an unbooked fill leaves the position open forever → re-exit loop. → Task 5 fixes exactly that branch.
- **SHR-48 confirmed real:** `risk.check_pre_trade` returns `approved_exit` *before* the depth-walk slippage clamp, so stop/exit IOCs bypass `max_slippage_pct`. → Task 6.
- **SHR-62 confirmed real:** `MarketState.apply`'s `BookSnapshotEvent` case full-replaces `bid_levels`/`ask_levels`; the PM adapter (`adapters/polymarket_normalize.py`) is stateless and packs a `price_change` delta as a snapshot of only changed levels. → Task 7 makes the adapter stateful.
- **SHR-66:** live `recent_returns` windows by COUNT (`market_state.py`), backtest by TIME (`backtest/runner/market_state.py` → `returns_buffer.slice_window`). Both already share `marketdata/ohlc.py` bucketing. → Task 8 converges the windowing.

---

## File structure

| File | Responsibility / change |
|------|--------------------------|
| `hlanalysis/engine/market_state.py` | Add `evict_settled_questions(...)`; converge `recent_returns` windowing (SHR-66). |
| `hlanalysis/engine/runtime.py` | Call eviction from a periodic loop; add in-flight exit guard + latching kill-switch + RSS memory guard. |
| `hlanalysis/recorder/writer.py` | Global buffer cap with drop-oldest + alert (SHR-63). |
| `hlanalysis/engine/reconcile.py` | Apply recovered net-delta to Position in the local-ghost branch (SHR-46). |
| `hlanalysis/engine/risk.py` | Apply depth-walk slippage clamp to exits (SHR-48). |
| `hlanalysis/adapters/polymarket_normalize.py` + `polymarket.py` | Stateful per-asset book; emit merged full snapshots (SHR-62). |
| `tests/...` | Regression + dangerous-path tests (SHR-45 guard, SHR-67 suite). |

---

## Task 1: Bound `_questions` — add eviction (SHR-44)

**Files:**
- Modify: `hlanalysis/engine/market_state.py` (add method near `mark_question_settled`, ~line 385)
- Test: `tests/unit/test_market_state_eviction.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_market_state_eviction.py
from hlanalysis.engine.market_state import MarketState
from hlanalysis.events import QuestionMetaEvent


def _meta(idx, expiry_ns, venue="hyperliquid"):
    return QuestionMetaEvent(
        question_idx=idx, venue=venue,
        keys=("class", "underlying", "expiry_ns"),
        values=("priceBinary", "BTC", str(expiry_ns)),
    )


def test_evict_settled_questions_removes_old_settled_only():
    ms = MarketState()
    ms.apply(_meta(1, expiry_ns=1_000))
    ms.apply(_meta(2, expiry_ns=10_000))
    ms.mark_question_settled(1)                 # settled, old
    # qidx 2 not settled
    n = ms.evict_settled_questions(now_ns=1_000 + 7 * 3600 * 1_000_000_000,
                                   retain_after_settle_ns=3600 * 1_000_000_000)
    idxs = {q.question_idx for q in ms.all_questions()}
    assert n == 1
    assert idxs == {2}                          # settled+old gone, unsettled kept


def test_evict_keeps_recently_settled():
    ms = MarketState()
    ms.apply(_meta(1, expiry_ns=1_000))
    ms.mark_question_settled(1)
    # now only 1 minute after expiry → within retain window → kept
    n = ms.evict_settled_questions(now_ns=1_000 + 60 * 1_000_000_000,
                                   retain_after_settle_ns=3600 * 1_000_000_000)
    assert n == 0
    assert {q.question_idx for q in ms.all_questions()} == {1}
```

> Confirm `QuestionMetaEvent` field names against `hlanalysis/events.py` and adjust `keys`/`values` to whatever `_update_question` parses for expiry; the test only needs a settled flag + an expiry to age against.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_market_state_eviction.py -v`
Expected: FAIL — `AttributeError: 'MarketState' object has no attribute 'evict_settled_questions'`.

- [ ] **Step 3: Implement eviction** (in `market_state.py`, after `mark_question_settled`)

```python
    def evict_settled_questions(
        self, *, now_ns: int, retain_after_settle_ns: int,
    ) -> int:
        """Drop questions that are settled AND whose expiry is older than the
        retain window. Bounds _questions on the 1 GB box (SHR-44) and shrinks
        the per-tick scan set. Returns the number evicted. Invalidates the
        symbol→question cache when anything is removed."""
        victims = [
            idx for idx, q in self._questions.items()
            if q.settled
            and q.expiry_ns
            and (now_ns - q.expiry_ns) > retain_after_settle_ns
        ]
        for idx in victims:
            del self._questions[idx]
        if victims:
            self._sym_to_q_cache = None
        return len(victims)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_market_state_eviction.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/engine/market_state.py tests/unit/test_market_state_eviction.py
git commit -m "feat(engine): evict old settled questions to bound MarketState (SHR-44)"
```

---

## Task 2: Call eviction from the reconcile loop (SHR-44 wiring)

**Files:**
- Modify: `hlanalysis/engine/runtime.py` (`_reconcile_loop`, ~814-931)
- Test: `tests/unit/test_market_state_eviction.py` (already covers the method; add a tiny wiring assert if a unit seam exists)

- [ ] **Step 1: Add the call** in `_reconcile_loop`, right after the existing `mark_question_settled` handling (one shared MarketState, so evict once per cycle is fine):

```python
                # SHR-44: bound the question set on the 1 GB box. Retain a
                # generous window after settlement so late reconciles / settlement
                # Exits still find the question, then drop it.
                self.market_state.evict_settled_questions(
                    now_ns=self._now_ns(),
                    retain_after_settle_ns=6 * 3600 * 1_000_000_000,  # 6h
                )
```

- [ ] **Step 2: Run the engine integration suite to confirm no regression**

Run: `uv run pytest tests/integration/test_engine_paper_loop.py -q`
Expected: PASS (the paper loop still enters+exits; eviction is a no-op within the short test window because retain=6h).

- [ ] **Step 3: Commit**

```bash
git add hlanalysis/engine/runtime.py
git commit -m "feat(engine): evict settled questions each reconcile cycle (SHR-44)"
```

---

## Task 3: Global recorder buffer cap with drop-oldest (SHR-63)

**Files:**
- Modify: `hlanalysis/recorder/writer.py` (`ParquetWriter.__init__`, `write`, `_flush_key`)
- Test: `tests/unit/test_recorder_writer_buffer_cap.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_recorder_writer_buffer_cap.py
from pathlib import Path
import pyarrow as pa  # noqa: F401  (writer imports it)
from hlanalysis.recorder.writer import ParquetWriter


def _row(sym="BTC", et="trade"):
    return {"venue": "v", "product_type": "p", "mechanism": "m",
            "event_type": et, "symbol": sym, "exchange_ts": 1}


def test_global_cap_drops_oldest_on_persistent_write_failure(tmp_path: Path, monkeypatch):
    w = ParquetWriter(tmp_path, max_buffer_rows=10, max_total_buffer_rows=25)

    # Force every flush to fail so rows re-buffer (the OOM path).
    import hlanalysis.recorder.writer as wr
    monkeypatch.setattr(wr.pq, "write_table",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    # Write far more than the global cap across several keys.
    for i in range(200):
        w.write(_row(sym=f"S{i % 5}"))

    total = sum(len(rows) for rows in w._buffers.values())
    assert total <= 25                       # global cap enforced
    assert w.dropped_rows > 0                 # and it tracked the drop
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_recorder_writer_buffer_cap.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword 'max_total_buffer_rows'` (or assertion fails: unbounded growth).

- [ ] **Step 3: Implement the global cap**

In `ParquetWriter.__init__`, add the param + counter:

```python
    def __init__(
        self,
        root: Path,
        flush_interval_s: float = 30.0,
        max_buffer_rows: int = 5000,
        max_total_buffer_rows: int = 500_000,   # global backstop (SHR-63)
    ) -> None:
        ...
        self.max_total_buffer_rows = max_total_buffer_rows
        self.dropped_rows = 0
```

Add a private enforcer and call it after any append/re-buffer:

```python
    def _enforce_global_cap(self) -> None:
        """SHR-63: on a persistent write failure _flush_key re-buffers rows; on
        the OOMScore=-500 recorder that would OOM-kill the +500 live engine.
        Cap total buffered rows across all keys; drop the OLDEST rows (FIFO) and
        count them so a monitor/alert can see data loss instead of an OOM."""
        total = sum(len(r) for r in self._buffers.values())
        if total <= self.max_total_buffer_rows:
            return
        # Drop oldest-first across keys until under the cap. Buffers are append
        # ordered, so index 0 of each key list is oldest for that key; drop from
        # the largest key first as a simple, bounded heuristic.
        while total > self.max_total_buffer_rows and self._buffers:
            key = max(self._buffers, key=lambda k: len(self._buffers[k]))
            rows = self._buffers[key]
            if not rows:
                del self._buffers[key]
                continue
            rows.pop(0)
            self.dropped_rows += 1
            total -= 1
        log.warning("recorder buffer cap hit; dropped_rows=%d", self.dropped_rows)
```

In `write`, after the append (line ~58) and in `_flush_key` after the re-buffer `extend` (line ~95), call `self._enforce_global_cap()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_recorder_writer_buffer_cap.py -v`
Expected: PASS.

- [ ] **Step 5: Run the recorder test suite**

Run: `uv run pytest tests/ -k recorder -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add hlanalysis/recorder/writer.py tests/unit/test_recorder_writer_buffer_cap.py
git commit -m "fix(recorder): global buffer cap with drop-oldest to stop OOM (SHR-63)"
```

---

## Task 4: Verify the reject circuit-breaker + regression test + close SHR-45

**Files:**
- Test: `tests/unit/test_router_reject_breaker.py` (create)

- [ ] **Step 1: Confirm the existing implementation** (read, do not change yet)

Read `hlanalysis/engine/router.py`: `__init__` (`_reject_breaker_threshold`, `_consecutive_rejects`), `_place` (pre-place suppression + reject increment + trip log), `_book_fill` (reset on fill). Confirm behavior matches the description.

- [ ] **Step 2: Write a regression test that pins the behavior**

```python
# tests/unit/test_router_reject_breaker.py
# Build a Router with a fake exec_client that always rejects, drive N entry
# intents for the same (question_idx, side), and assert:
#   (a) after `threshold` rejects, further placements are suppressed (no extra
#       exec_client.place calls);
#   (b) a fill on that question resets the counter (placements resume).
# Use the same Router construction the engine uses (see runtime._build_slot for
# the constructor kwargs: dal, gate, bus, exec_client, strategy_id, cloid_prefix,
# reject_breaker_threshold). Reuse a fake exec client returning
# OrderAck(status="rejected", error="insufficient margin").
```

Implement the test concretely against the real `Router.__init__` signature and `OrderAck`/`OrderIntent` shapes (copy field names from `router.py` + `exec_types.py`). Assert `fake_client.place_calls` stops growing past the threshold, then resumes after a synthesized fill.

- [ ] **Step 3: Run the test**

Run: `uv run pytest tests/unit/test_router_reject_breaker.py -v`
Expected: PASS (guards existing behavior). If it FAILS, the breaker regressed — fix `router.py` to satisfy the test.

- [ ] **Step 4: (Optional) per-error-class visibility** — only if cheap: include `ack.error` class in the `order_rejected` event payload so `engine-diag` reject counts bucket by reason. Skip if it risks scope creep.

- [ ] **Step 5: Commit + update the ticket**

```bash
git add tests/unit/test_router_reject_breaker.py
git commit -m "test(engine): pin reject circuit-breaker behavior; SHR-45 verified live"
```

Mark SHR-45 Done in Linear with a note that the breaker was already implemented and is now regression-covered.

---

## Task 5: Apply recovered net-delta to Position on reconcile fill-discovery (SHR-46)

**Files:**
- Modify: `hlanalysis/engine/reconcile.py` (local-ghost branch, ~138-179)
- Test: `tests/unit/test_reconcile_applies_recovered_fill.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_reconcile_applies_recovered_fill.py
# Set up a StateDAL with a PENDING OpenOrder whose cloid the venue does NOT
# return as open, but whose fills_lookup(cloid) returns a UserFillRow (the
# lost-ACK case). Run Reconciler.run with apply_position_changes=True and a
# venue_state that DOES contain the resulting position. Assert that after
# reconcile a Position row exists for the question_idx with qty == net_delta
# (i.e. the recovered fill was booked), not left absent.
```

Build it concretely from `tests/` reconcile fixtures (see existing reconcile tests for the `Reconciler` construction + `UserFillRow`/`ClearinghouseState` builders). The key assertion: `dal.get_position(qidx)` is not None and its `qty` equals the recovered net delta.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_reconcile_applies_recovered_fill.py -v`
Expected: FAIL — position is absent (current code marks order filled + replays Fill rows but never upserts the Position).

- [ ] **Step 3: Fix the local-ghost branch** in `reconcile.py`

After the existing `for f in fills: self.dal.append_fill(...)` + `update_order_status(cloid, status="filled")`, when `apply_position_changes` is True and the venue state confirms a matching position, upsert the Position from the recovered fills' net delta + avg entry (mirror how the venue `VenuePosition` reports qty/avg_entry for that symbol). Compute `net_delta` and `avg_entry` from the replayed fills; key by `db_o.question_idx`. Emit the existing drift record. (Keep the PM `apply_position_changes=False` path unchanged — alert only.)

```python
                if self.apply_position_changes:
                    vp = next((p for p in venue_state.positions
                               if p.symbol == db_o.symbol), None)
                    if vp is not None and abs(vp.qty) > 1e-9:
                        self.dal.upsert_position(Position(
                            question_idx=db_o.question_idx, symbol=db_o.symbol,
                            qty=vp.qty, avg_entry=vp.avg_entry,
                            realized_pnl=0.0, last_update_ts_ns=now_ns,
                            stop_loss_price=0.0,
                        ))
```

> Use the venue's reported qty/avg_entry (authoritative) rather than reconstructing from fills — simpler and matches the adopt-orphan path already in this file (~308-327). Confirm `Position` is imported in `reconcile.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_reconcile_applies_recovered_fill.py -v`
Expected: PASS.

- [ ] **Step 5: Run the reconcile suite**

Run: `uv run pytest tests/ -k reconcile -q`
Expected: PASS (no regression to vanish/orphan/adopt behavior).

- [ ] **Step 6: Commit**

```bash
git add hlanalysis/engine/reconcile.py tests/unit/test_reconcile_applies_recovered_fill.py
git commit -m "fix(engine): book recovered lost-ACK fill into Position (SHR-46)"
```

---

## Task 6: Slippage-clamp exits + in-flight exit guard (SHR-48)

**Files:**
- Modify: `hlanalysis/engine/risk.py` (`check_pre_trade` exit short-circuit, ~65)
- Modify: `hlanalysis/engine/runtime.py` (`_enforce_stop_losses`, ~1062-1112 — in-flight guard)
- Test: `tests/unit/test_risk_exit_slippage.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_risk_exit_slippage.py
# Build a RiskGate with global_.max_slippage_pct = 0.005. Construct an EXIT
# (reduce_only) sell intent for size that exceeds the inside bid depth, with a
# book whose deeper bid levels are >0.5% below the top. Call check_pre_trade.
# Assert the verdict CLAMPS size to the at-budget depth (verdict.clamped_size
# present and < intent.size) instead of approving the full walk.
```

Mirror the entry-side depth-walk test (`tests/` has one for `depth_walk_slip`/`clamped_size`); copy its `BookState` with `bid_levels` and `RiskInputs` construction, set the intent's `exit_reason`/`reduce_only` so `is_exit` is True.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_risk_exit_slippage.py -v`
Expected: FAIL — current code returns `approved_exit` with no clamp before the slippage gate runs.

- [ ] **Step 3: Fix `check_pre_trade`** — run the depth-walk clamp for exits too

Replace the bare short-circuit:

```python
        if is_exit:
            return RiskVerdict(True, "approved_exit")
```

with a version that still skips entry-only gates (allowlist, caps, cooldown) but runs the depth-walk slippage clamp so exits can't walk the book past budget:

```python
        if is_exit:
            # Exits skip entry gates, but MUST respect the slippage budget so a
            # stop/exit IOC can't walk a thin book down (SHR-48). Reuse the same
            # depth-walk clamp used for entries; on a budget breach, clamp size
            # rather than veto (a partial reduce is better than none).
            return self._depth_walk_clamp(intent, inp, approve_reason="approved_exit")
```

Extract the existing depth-walk block (risk.py ~153-188) into `_depth_walk_clamp(self, intent, inp, *, approve_reason) -> RiskVerdict` returning `RiskVerdict(True, approve_reason, clamped_size=...)` (or the existing `depth_walk_slip`/`depth_walk_no_fill` vetoes), and call it from both the entry path and the exit path. For exits, prefer clamping to the at-budget fill rather than vetoing outright (so a stop still reduces what it can).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_risk_exit_slippage.py -v` and `uv run pytest tests/ -k "risk or depth_walk" -q`
Expected: PASS (entry-side behavior preserved; exits now clamped).

- [ ] **Step 5: Add the in-flight exit guard** in `_enforce_stop_losses` (runtime.py)

Before submitting a stop IOC for a position, skip if there is already a live reduce-only order for that `question_idx`/`symbol` in `slot.dal.live_orders()` (status in pending/open/partially_filled). This stops the ~1 Hz loop re-firing a fresh full-size IOC before the prior ACK resolves (complements the entry-side SHR-47 guard).

```python
        live = {o.question_idx for o in slot.dal.live_orders()}
        for sp in breached:
            if sp.question_idx in live:
                continue   # an exit is already in flight; don't stack another IOC
            ...
```

- [ ] **Step 6: Run the stop-loss / runtime tests**

Run: `uv run pytest tests/ -k "stop_loss or paper_loop" -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add hlanalysis/engine/risk.py hlanalysis/engine/runtime.py tests/unit/test_risk_exit_slippage.py
git commit -m "fix(engine): clamp exits to slippage budget + in-flight exit guard (SHR-48)"
```

---

## Task 7: Stateful PM book — emit merged full snapshots (SHR-62)

**Files:**
- Modify: `hlanalysis/adapters/polymarket_normalize.py` (`parse_book_message`, `parse_price_change_message`)
- Modify: `hlanalysis/adapters/polymarket.py` (the `_handle` dispatch holds per-asset book state)
- Test: `tests/unit/test_polymarket_book_delta.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_polymarket_book_delta.py
from hlanalysis.adapters.polymarket_normalize import PmBook  # new

def test_price_change_delta_merges_into_full_book():
    bk = PmBook()
    # Full book snapshot: two bid levels, two ask levels.
    snap = bk.apply_book({
        "asset_id": "A", "timestamp": 1,
        "bids": [{"price": "0.40", "size": "100"}, {"price": "0.39", "size": "50"}],
        "asks": [{"price": "0.60", "size": "80"}, {"price": "0.61", "size": "40"}],
    })
    assert snap.bid_px[0] == 0.40 and len(snap.bid_px) == 2

    # price_change touches only the 0.39 level (resize) — must NOT wipe 0.40/asks.
    out = bk.apply_price_change({
        "asset_id": "A", "timestamp": 2,
        "changes": [{"price": "0.39", "size": "10", "side": "BUY"}],
    })
    assert set(out.bid_px) == {0.40, 0.39}          # full book preserved
    assert dict(zip(out.bid_px, out.bid_sz))[0.39] == 10.0   # delta applied
    assert len(out.ask_px) == 2                      # asks untouched

def test_price_change_zero_size_removes_level():
    bk = PmBook()
    bk.apply_book({"asset_id": "A", "timestamp": 1,
                   "bids": [{"price": "0.40", "size": "100"},
                            {"price": "0.39", "size": "50"}], "asks": []})
    out = bk.apply_price_change({"asset_id": "A", "timestamp": 2,
        "changes": [{"price": "0.39", "size": "0", "side": "BUY"}]})
    assert set(out.bid_px) == {0.40}                 # zero-size removed the level
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_polymarket_book_delta.py -v`
Expected: FAIL — `ImportError: cannot import name 'PmBook'`.

- [ ] **Step 3: Implement a `PmBook` per-asset state** in `polymarket_normalize.py`

```python
class PmBook:
    """Per-asset L2 book the PM adapter maintains so price_change deltas merge
    into a FULL book before emission (SHR-62). MarketState.apply treats
    BookSnapshotEvent as a full replace, so emitting only-changed levels as a
    snapshot corrupts the book to 1-2 phantom levels."""
    def __init__(self) -> None:
        self._bids: dict[float, float] = {}   # price -> size
        self._asks: dict[float, float] = {}

    def apply_book(self, payload, *, local_recv_ts=0) -> BookSnapshotEvent:
        self._bids = {float(l["price"]): float(l["size"]) for l in (payload.get("bids") or [])}
        self._asks = {float(l["price"]): float(l["size"]) for l in (payload.get("asks") or [])}
        return self._emit(payload, local_recv_ts)

    def apply_price_change(self, payload, *, local_recv_ts=0) -> BookSnapshotEvent | None:
        changes = payload.get("changes") or []
        if not changes:
            return None
        for c in changes:
            side = str(c.get("side", "")).upper()
            px, sz = float(c["price"]), float(c["size"])
            book = self._bids if side == "BUY" else self._asks
            if sz <= 0:
                book.pop(px, None)
            else:
                book[px] = sz
        return self._emit(payload, local_recv_ts)

    def _emit(self, payload, local_recv_ts) -> BookSnapshotEvent:
        bid_px = sorted(self._bids, reverse=True)          # best (highest) first
        ask_px = sorted(self._asks)                        # best (lowest) first
        return BookSnapshotEvent(
            venue=_VENUE, product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol=str(payload["asset_id"]),
            exchange_ts=_ts_ms_to_ns(payload.get("timestamp", 0)),
            local_recv_ts=local_recv_ts,
            bid_px=tuple(bid_px), bid_sz=tuple(self._bids[p] for p in bid_px),
            ask_px=tuple(ask_px), ask_sz=tuple(self._asks[p] for p in ask_px),
        )
```

- [ ] **Step 4: Wire it into the adapter** (`polymarket.py` `_handle`, ~190)

Hold a `dict[str, PmBook]` keyed by `asset_id` on the adapter instance; on `book` call `apply_book` (resets that asset), on `price_change` call `apply_price_change`. Emit the returned event (skip `None`). Reset the per-asset book on reconnect/resubscribe so a stale local book can't survive a gap.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_polymarket_book_delta.py -v` and `uv run pytest tests/ -k polymarket -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add hlanalysis/adapters/polymarket_normalize.py hlanalysis/adapters/polymarket.py tests/unit/test_polymarket_book_delta.py
git commit -m "fix(adapters): maintain PM book + emit merged full snapshots (SHR-62)"
```

---

## Task 8: Converge sim/live MarketState windowing (SHR-66)

**Files:**
- Modify: `hlanalysis/engine/market_state.py` (`recent_returns`, ~468-484)
- Test: `tests/unit/test_marketstate_windowing_parity.py` (create)

- [ ] **Step 1: Write the failing test** — same bar stream + a feed gap must yield the same return set under live and backtest windowing

```python
# tests/unit/test_marketstate_windowing_parity.py
# Feed a known tick stream (with a gap) into the live MarketState at a fixed
# cadence; compute recent_returns over a TIME window [now - lookback, now].
# Independently compute the expected returns by slicing the SAME bars by
# wall-clock time (the backtest rule). Assert they are equal.
# The bug today: live slices by COUNT ([-(n+1):]) so after a gap the count
# window reaches further back in wall-clock and includes returns the time
# window excludes.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_marketstate_windowing_parity.py -v`
Expected: FAIL (count window includes pre-gap bars the time window drops).

- [ ] **Step 3: Make live `recent_returns` time-bounded**

Add an optional `now_ns` + `lookback_seconds` path to live `recent_returns` (and `recent_hl_bars`) that filters bars to `ts >= now_ns - lookback_seconds*1e9` before taking returns — matching `returns_buffer.slice_window`. Keep the count-only path for callers that pass `n` without time, but have the Scanner pass `now_ns`+lookback (it already knows both). Prefer reusing a shared helper from `hlanalysis/marketdata/` if a time-slice util exists there; otherwise add one and call it from both live and backtest so there is ONE windowing rule.

> This is the bridge to Phase 2: sizing tuned in backtest is only safe if live σ matches. The acceptance is parity, so the implementation must converge on the backtest's TIME rule, not invent a third.

- [ ] **Step 4: Run test + the σ/sigma suites to verify**

Run: `uv run pytest tests/unit/test_marketstate_windowing_parity.py -v` and `uv run pytest tests/ -k "sigma or returns or market_state" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/engine/market_state.py hlanalysis/marketdata/ tests/unit/test_marketstate_windowing_parity.py
git commit -m "fix(engine): time-bounded recent_returns for sim/live parity (SHR-66)"
```

---

## Task 9: Latching kill-switch + RSS memory guard (W1.9)

**Files:**
- Modify: `hlanalysis/engine/runtime.py` (daily-loss halt path + a memory-watch in `_heartbeat_loop`)
- Test: `tests/unit/test_latching_killswitch.py` (create)

- [ ] **Step 1: Write the failing test** — a daily-loss breach must set a *persistent* halt that stays set across the next scan cycle (not a transient alert)

```python
# tests/unit/test_latching_killswitch.py
# Drive the daily-loss check past the cap on a slot; assert slot.halted is True
# AND the kill-switch flag file exists; assert a subsequent scan does not place
# new entries while halted; assert it clears only via explicit operator action
# (flag-file removal), not automatically on the next cycle.
```

Use the existing kill-switch flag path (`deploy_cfg.kill_switch_path_for(alias)`) that `diag.py` already reads as `halt`.

- [ ] **Step 2: Run test to verify it fails / passes**

Run: `uv run pytest tests/unit/test_latching_killswitch.py -v`
Expected: FAIL if the daily-loss halt is non-latching today (per spec: StaleDataHalt never sets `slot.halted`). If the daily-loss path already latches, this becomes a regression guard — keep it.

- [ ] **Step 3: Implement latching halt** — on a confirmed daily-loss breach (true PnL ≤ −cap), write the kill-switch flag file and set `slot.halted = True`; the scan/stop loops already honor `slot.halted`. Do NOT auto-clear; require flag-file removal (operator).

- [ ] **Step 4: Add an RSS self-halt guard** in `_heartbeat_loop` — read process RSS (e.g. `resource.getrusage(RUSAGE_SELF).ru_maxrss`, or `/proc/self/statm`); if it exceeds a configured ceiling (e.g. 80% of the box budget), flush DALs, set all slots halted + write flags, log + Telegram, and stop placing. This is the current-box safety net behind Tasks 1-3.

```python
            # SHR-44/63 backstop on the 1 GB box: self-halt before the kernel
            # OOM-killer fires, so we stop placing rather than die mid-position.
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if rss_kb > self.rss_halt_kb:
                logger.error("RSS {}kb over ceiling {}kb — self-halting", rss_kb, self.rss_halt_kb)
                for s in slots:
                    s.halted = True
                # write kill-switch flags + alert here
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_latching_killswitch.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add hlanalysis/engine/runtime.py tests/unit/test_latching_killswitch.py
git commit -m "feat(engine): latching daily-loss kill-switch + RSS self-halt guard (W1.9)"
```

---

## Task 10: Dangerous-path test suite (SHR-67)

**Files:**
- Create: `tests/integration/test_engine_dangerous_paths.py` (reuse `_FakeAdapter` / `_FakeTelegram` from `tests/integration/test_engine_paper_loop.py`)

- [ ] **Step 1: Write the tests** — one per dangerous path, each driving a finite event stream and asserting the safety behavior:

1. **Order rejection** — fake exec client returns `OrderAck(status="rejected")` repeatedly; assert the circuit-breaker suppresses after threshold (ties to Task 4) and no unbounded place() calls.
2. **Stop-loss IOC chain** — book breaches a stop; assert exactly one in-flight IOC at a time (Task 6 guard) and that size is slippage-clamped on a thin book.
3. **Reconcile finds venue drift** — venue state lacks a DB position (vanished) / has an extra (orphan); assert the reconciler emits drift and (for HL) applies the change; recovered lost-ACK fill is booked (Task 5).
4. **Restart with pre-existing DB+venue state** — construct a state.db with an open position + a venue state that matches; assert the RestartDriftGate adopts/aligns without double-booking.
5. **Feed disconnect/reconnect** — adapter raises then resumes; assert ingest reconnects (SHR-42 already landed) and PM book resets on reconnect (Task 7).

Each test: build `EngineRuntime.from_single(...)` with the fakes, `asyncio.create_task(runtime.run())`, drive events, `runtime.stop_event.set()`, assert on the `_FakeTelegram.messages` + DB state.

- [ ] **Step 2: Run the suite**

Run: `uv run pytest tests/integration/test_engine_dangerous_paths.py -v`
Expected: PASS (each asserts a Phase-1 fix).

- [ ] **Step 3: Run the FULL suite**

Run: `uv run pytest -q`
Expected: green (existing count + all new tests).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_engine_dangerous_paths.py
git commit -m "test(engine): dangerous-path coverage — reject/stop/reconcile/restart/reconnect (SHR-67)"
```

---

## Task 11: Deploy + start the Phase-1 soak gate

**Files:** none (ops). Deploy is SSM-only (CLAUDE.md).

- [ ] **Step 1: Merge the Phase-1 branch to `main`** after the full suite is green and the Phase-0 reconcile-report is in place (Phase 0 lands first per sequencing).
- [ ] **Step 2: Deploy** `make deploy-engine` + `make deploy-recorder`; verify `make engine-status` shows no `restart_blocked`/`halt`.
- [ ] **Step 3: Start the 7-day soak.** Record the start in SHR-30. The Phase-1 exit gate (spec §5) is met when, for 7+ continuous days: zero OOM kills (recorder+engine), zero unreconciled orphans (per the Phase-0 reconcile-report), a simulated reject storm auto-halts, a simulated daily-loss breach latches, the dangerous-path suite is green in CI, and live-vs-backtest σ agree within tolerance (Task 8).

---

## Self-review notes

- **Spec coverage:** W1.1→T1+T2; W1.2→T3; W1.3→T4 (verify/close); W1.4→T5 (narrowed); W1.5→T6; W1.6→T7; W1.7→T8; W1.8→T10; W1.9→T9; gate→T11.
- **Reality corrections folded in:** SHR-45 already implemented (verify, don't rebuild); SHR-46 narrowed to "book the recovered fill". These also need the spec's W1.3/W1.4 wording updated (done in the design doc).
- **Verify-before-coding flags:** event field names (T1), Router/OrderAck signatures (T4), reconcile fixtures + `Position` import (T5), the entry-side depth-walk block to extract (T6), PM payload field names (T7), shared marketdata time-slice util (T8), kill-switch flag path (T9), paper-loop fakes (T10). Each step says to match the real source.
- **Sequencing:** memory (T1-T3) → exec correctness (T4-T7) → parity (T8) → safety net (T9) → tests (T10) → soak (T11). T8 must be *done+verified* (not soaked) to unblock Phase 2; it does not gate Phase-1 operation.
```
