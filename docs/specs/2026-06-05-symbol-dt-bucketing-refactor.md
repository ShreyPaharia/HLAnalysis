# (symbol, dt) Reference Bucketing Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the engine bucket one shared reference-tick stream at multiple cadences simultaneously, so a single strategy slot can run different `vol_sampling_dt_seconds` per question class (e.g. v31 buckets at dt=2 while v31 binary + v1 stay at dt=5) without a second feed.

**Architecture:** Today `MarketState` keys its derived OHLC bar history by `symbol` and a `set_reference_cadence` conflict-guard forbids two cadences for the same symbol — the only thing forcing all slots on a feed into lockstep. We re-key the bucketing by `(symbol, dt)`, fan each incoming reference tick into every registered cadence for that symbol, and have the `Scanner` resolve the cadence **per question class** (reusing the `theta_overrides` config seam from the prior PR) and read the matching `(symbol, dt)` series. The raw WS subscription, books, trades, and `last_mark` stay per-symbol and untouched. The single-cadence path stays bit-identical.

**Tech Stack:** Python 3.12, pydantic v2 config, `MarketState` (`hlanalysis/engine/market_state.py`), `Scanner` (`hlanalysis/engine/scanner.py`), `EngineRuntime` cadence registration (`hlanalysis/engine/runtime.py`), `ReplayRunner` (`hlanalysis/engine/replay.py`), pytest. Run tests with `uv run python -m pytest`.

**Depends on:** `feat/per-class-theta-overrides` (this branch) — the `theta_overrides` config block + `build_theta_harvester_configs_by_class` are the per-class seam this refactor plugs cadence resolution into.

---

## Background: the surfaces this touches

Read these before starting; the tasks reference exact line ranges as of this branch.

- **`MarketState` state keyed by symbol** (`market_state.py:53-103`): `_marks: dict[str, deque]`, `_mark_bucket_ns_by_symbol: dict[str, int]`, `_mark_history_by_symbol: dict[str, int]`, `_last_mark_bucket: dict[str, int]`. The reference source (`mark`|`bbo`) and `last_mark`/`last_mark_ts` are also per-symbol — those stay per-symbol.
- **Cadence registration + conflict guard** (`market_state.py:114-162`): `set_reference_cadence(symbol, sampling_dt_seconds, lookback_seconds)` raises on a second distinct cadence for a symbol. `mark_bucket_ns_for(symbol)` returns the period.
- **Tick ingest / bucketing** (`market_state.py:252-283`): `_ingest_reference_price` appends/updates one `(high, low, close)` bar in `_marks[symbol]` using `_mark_bucket_ns_by_symbol`.
- **Series readers** (`market_state.py:477-508`): `recent_returns(symbol, n)` and `recent_hl_bars(symbol, n)` read `_marks[symbol]`.
- **Scanner per-slot read** (`scanner.py:115`, `scanner.py:117-143`, `scanner.py:264-283`): `self._recent_returns_n` is one per-slot value from `_required_returns_n(cfg)`; `scan()` pulls `recent_returns`/`recent_hl_bars` with `self.ref_symbol` + that single `n`. dt is read from `cfg.theta.vol_sampling_dt_seconds` (one value).
- **Runtime registration** (`runtime.py:611-628`): `_register_reference_cadences` calls `set_reference_cadence` once per slot with `reference_sampling_dt_seconds(slot.cfg)`.
- **Per-class dt guard to revert** (`runtime.py`, inside `build_theta_harvester_configs_by_class`): currently raises if a `theta_overrides[class]` sets `vol_sampling_dt_seconds`. After this refactor that override becomes legal.
- **Replay parity** (`replay.py:40-55`, `replay.py:76-93`): mirrors the live read path with a single cadence; must stay bit-identical for the single-cadence case.

**Design rule for bit-identity:** every new map keyed by `(symbol, dt)` must, when a symbol has exactly one registered cadence, produce byte-identical bars and returns to today's per-symbol path. The bucket id math (`ts // bucket_ns`, epoch-aligned) is already cadence-parameterized, so two cadences derive consistently from the same ticks.

---

## Task 1: `MarketState` stores bars keyed by (symbol, dt)

Re-key the bucketing maps and the ingest path so one symbol can hold multiple cadence series. Registration accepts repeated calls for the same symbol with *different* cadences (accumulates a set) but still rejects a re-registration that would *change* an existing (symbol, dt)'s history sizing inconsistently. `mark_bucket_ns_for`/`recent_returns`/`recent_hl_bars` gain an optional `dt` arg that defaults to preserving today's single-cadence behaviour.

**Files:**
- Modify: `hlanalysis/engine/market_state.py:53-103` (state decls), `:114-162` (`set_reference_cadence`), `:164-169` (`mark_bucket_ns_for`), `:252-283` (`_ingest_reference_price`), `:477-508` (`recent_returns`, `recent_hl_bars`)
- Test: `tests/unit/test_market_state_multi_cadence.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_market_state_multi_cadence.py`:

```python
# tests/unit/test_market_state_multi_cadence.py
"""(symbol, dt) bucketing: one reference-tick stream, two cadence series.

Lets a single slot run different vol_sampling_dt_seconds per question class
(v31 buckets dt=2 vs v31 binary/v1 dt=5) off the SAME feed. Single-cadence
reads must stay bit-identical to the legacy per-symbol path.
"""
from __future__ import annotations

import math

from hlanalysis.engine.market_state import MarketState
from hlanalysis.events import MarkEvent, Mechanism, ProductType


def _mark(symbol: str, px: float, ts_s: float) -> MarkEvent:
    ts = int(ts_s * 1_000_000_000)
    return MarkEvent(
        venue="hyperliquid", product_type=ProductType.PERP,
        mechanism=Mechanism.CLOB, symbol=symbol,
        exchange_ts=ts, local_recv_ts=ts, mark_px=px,
    )


def _feed(ms: MarketState, symbol: str, ticks: list[tuple[float, float]]) -> None:
    for ts_s, px in ticks:
        ms.apply(_mark(symbol, px, ts_s))


def test_two_cadences_bucket_same_stream_independently() -> None:
    """A symbol registered at dt=2 and dt=5 maintains two independent bar
    series from one tick stream: dt=2 closes every 2s, dt=5 every 5s."""
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=2)
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5)
    # ticks at t=0,1,2,3,4,5,6 → prices 100..106
    _feed(ms, "BTC", [(float(t), 100.0 + t) for t in range(7)])

    # dt=5 buckets: [0,5)->close@t4=104, [5,10)->close@t6=106 → 1 return
    rets5 = ms.recent_returns("BTC", n=10, dt=5)
    assert len(rets5) == 1
    assert math.isclose(rets5[0], math.log(106.0 / 104.0), rel_tol=1e-12)

    # dt=2 buckets: [0,2)->101, [2,4)->103, [4,6)->105, [6,8)->106 → closes
    # 101,103,105,106 → 3 returns. Independent series off the same ticks.
    rets2 = ms.recent_returns("BTC", n=10, dt=2)
    assert len(rets2) == 3
    assert math.isclose(rets2[0], math.log(103.0 / 101.0), rel_tol=1e-12)


def test_single_cadence_read_is_bit_identical_to_legacy() -> None:
    """A symbol with exactly one registered cadence yields identical
    recent_returns whether or not dt is passed (default resolves to the sole
    registered cadence)."""
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5)
    _feed(ms, "BTC", [(float(t), 100.0 + (t % 3)) for t in range(20)])
    assert ms.recent_returns("BTC", n=8) == ms.recent_returns("BTC", n=8, dt=5)
    assert ms.recent_hl_bars("BTC", n=8) == ms.recent_hl_bars("BTC", n=8, dt=5)


def test_same_cadence_reregistration_is_idempotent() -> None:
    """Re-registering the SAME (symbol, dt) only grows history sizing; it never
    raises (two slots can share class+cadence)."""
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5, lookback_seconds=1800)
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5, lookback_seconds=3600)
    assert ms.mark_bucket_ns_for("BTC", dt=5) == 5 * 1_000_000_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_market_state_multi_cadence.py -q`
Expected: FAIL — `set_reference_cadence` raises "conflicting mark-bucket cadence" on the second distinct cadence, and `recent_returns`/`recent_hl_bars`/`mark_bucket_ns_for` reject the `dt=` kwarg (TypeError).

- [ ] **Step 3: Re-key the state maps**

In `market_state.py`, change the per-symbol bucketing maps to be keyed by `(symbol, dt_ns)`. Keep `_reference_source_by_symbol`, `_last_mark`, `_last_mark_ts` per-symbol. Add a per-symbol set of registered cadences for default-resolution.

Replace the declarations around `:61-103`:

```python
        # Per (symbol, dt_ns): a deque of per-bucket OHLC bars (high, low, close).
        # One reference-tick stream fans into every cadence registered for the
        # symbol, so a single slot can run different vol_sampling_dt_seconds per
        # question class off the SAME feed (see the (symbol, dt) refactor plan).
        self._marks: dict[tuple[str, int], deque[tuple[float, float, float]]] = {}
        self._mark_history: int = mark_history  # default deque maxlen (legacy)
        self._mark_history_by_key: dict[tuple[str, int], int] = {}
        self._mark_bucket_ns: int = mark_bucket_ns  # default (60s) for unregistered symbols
        # Cadences registered per symbol, in registration order. The FIRST
        # registered cadence is the symbol's default (what a dt-less read
        # resolves to), preserving single-cadence bit-identity.
        self._cadences_by_symbol: dict[str, list[int]] = {}
        self._reference_source_by_symbol: dict[str, str] = {}
        # Last-bucket-id written per (symbol, dt_ns); coalesces within-bucket
        # reference-price updates into one bar per cadence.
        self._last_mark_bucket: dict[tuple[str, int], int] = {}
```

(Delete the old `_marks: dict[str, ...]`, `_mark_bucket_ns_by_symbol`, `_mark_history_by_symbol`, and the per-symbol `_last_mark_bucket` declarations these replace. Keep `_last_mark`/`_last_mark_ts` per-symbol as they are.)

- [ ] **Step 4: Accumulate cadences in `set_reference_cadence`**

Replace `set_reference_cadence` (`:114-162`) so it registers a *set* of cadences per symbol instead of guarding against a second one:

```python
    def set_reference_cadence(
        self,
        symbol: str,
        *,
        sampling_dt_seconds: int,
        lookback_seconds: int | None = None,
    ) -> None:
        """Register a mark-bucketing cadence for ``symbol``.

        A symbol may carry MULTIPLE cadences (e.g. dt=5 for v31 binary and dt=2
        for v31 buckets) — each is bucketed independently from the SAME shared
        reference-tick stream. Re-registering an existing (symbol, dt) only grows
        its history sizing (never shrinks, never raises). The first cadence
        registered for a symbol is its default, returned by dt-less reads.

        ``lookback_seconds`` sizes the (symbol, dt) history deque to hold
        ``ceil(lookback/dt)`` returns; never shrinks below the default maxlen.
        """
        if sampling_dt_seconds <= 0:
            raise ValueError(
                f"sampling_dt_seconds must be positive, got {sampling_dt_seconds!r}"
            )
        ns = int(sampling_dt_seconds) * 1_000_000_000
        cadences = self._cadences_by_symbol.setdefault(symbol, [])
        if ns not in cadences:
            cadences.append(ns)
        key = (symbol, ns)
        if lookback_seconds is not None:
            needed = int(lookback_seconds) // int(sampling_dt_seconds) + 2
            prev = self._mark_history_by_key.get(key, self._mark_history)
            self._mark_history_by_key[key] = max(prev, needed)
            hist = self._marks.get(key)
            if hist is not None and hist.maxlen != self._mark_history_by_key[key]:
                self._marks[key] = deque(hist, maxlen=self._mark_history_by_key[key])
```

- [ ] **Step 5: Resolve default dt + parameterize `mark_bucket_ns_for`**

Add a private resolver and update `mark_bucket_ns_for` (`:164-169`):

```python
    def _resolve_dt_ns(self, symbol: str, dt: int | None) -> int:
        """Bucket period (ns) for a (symbol, dt) read. ``dt=None`` resolves to
        the symbol's FIRST registered cadence (single-cadence bit-identity), or
        the global default if the symbol was never registered."""
        if dt is not None:
            return int(dt) * 1_000_000_000
        cadences = self._cadences_by_symbol.get(symbol)
        return cadences[0] if cadences else self._mark_bucket_ns

    def mark_bucket_ns_for(self, symbol: str, dt: int | None = None) -> int:
        """The bucket period (ns) applied to ``symbol`` at cadence ``dt`` (or the
        symbol's default cadence when dt is None)."""
        return self._resolve_dt_ns(symbol, dt)
```

- [ ] **Step 6: Fan each tick into every registered cadence**

Rewrite `_ingest_reference_price` (`:252-283`) to update `last_mark` once, then loop over the symbol's cadences:

```python
    def _ingest_reference_price(self, symbol: str, price: float, ts: int) -> None:
        """Feed one reference-price tick into ``last_mark`` plus the per-cadence
        OHLC bars. One shared tick stream fans into every cadence registered for
        the symbol; within a bucket the bar updates in place (high=max, low=min,
        close=last), a new bucket appends a fresh (price, price, price)."""
        self._last_mark[symbol] = price
        self._last_mark_ts[symbol] = ts
        # An unregistered symbol still gets the legacy single default bucket so
        # pre-registration ticks are not dropped (matches old behaviour).
        cadences = self._cadences_by_symbol.get(symbol) or [self._mark_bucket_ns]
        for bucket_ns in cadences:
            key = (symbol, bucket_ns)
            hist = self._marks.get(key)
            if hist is None:
                maxlen = self._mark_history_by_key.get(key, self._mark_history)
                hist = deque(maxlen=maxlen)
                self._marks[key] = hist
            bucket = ts // bucket_ns
            last_bucket = self._last_mark_bucket.get(key)
            if last_bucket is None or bucket != last_bucket or not hist:
                hist.append((price, price, price))
                self._last_mark_bucket[key] = bucket
            else:
                h, l, _c = hist[-1]
                hist[-1] = (h if h >= price else price, l if l <= price else price, price)
```

Note: the unregistered-symbol fallback uses `self._mark_bucket_ns` (the 60s default) as the bucket key, so a dt-less read on an unregistered symbol still resolves to that same key via `_resolve_dt_ns`.

- [ ] **Step 7: Parameterize the readers**

Update `recent_returns` (`:477-491`) and `recent_hl_bars` (`:493-508`) to take an optional `dt`:

```python
    def recent_returns(self, symbol: str, n: int, dt: int | None = None) -> tuple[float, ...]:
        """Last ``n`` close-to-close log returns for ``symbol`` at cadence ``dt``
        (default = the symbol's first registered cadence)."""
        key = (symbol, self._resolve_dt_ns(symbol, dt))
        hist = self._marks.get(key)
        if hist is None or len(hist) < 2:
            return ()
        bars = list(hist)[-(n + 1):]
        rets: list[float] = []
        for prev, curr in zip(bars, bars[1:], strict=False):
            prev_c, curr_c = prev[2], curr[2]
            if prev_c > 0 and curr_c > 0:
                rets.append(math.log(curr_c / prev_c))
        return tuple(rets)

    def recent_hl_bars(self, symbol: str, n: int, dt: int | None = None) -> tuple[tuple[float, float], ...]:
        """Last ``n`` per-bucket (high, low) bars for ``symbol`` at cadence ``dt``
        (default = the symbol's first registered cadence)."""
        key = (symbol, self._resolve_dt_ns(symbol, dt))
        hist = self._marks.get(key)
        if not hist:
            return ()
        bars = list(hist)[-n:]
        return tuple((b[0], b[1]) for b in bars)
```

- [ ] **Step 8: Run the new test to verify it passes**

Run: `uv run python -m pytest tests/unit/test_market_state_multi_cadence.py -q`
Expected: PASS (3 tests).

- [ ] **Step 9: Run the existing MarketState + cadence tests for regressions**

Run: `uv run python -m pytest tests/unit/ -k "market_state or cadence" -q`
Expected: PASS. If `set_reference_cadence` conflict-raise tests exist (they asserted the OLD guard), update them in Task 4 — note any failures here and carry them forward; do not delete assertions yet.

- [ ] **Step 10: Commit**

```bash
git add hlanalysis/engine/market_state.py tests/unit/test_market_state_multi_cadence.py
git commit -m "refactor(market-state): key reference bucketing by (symbol, dt)"
```

---

## Task 2: Scanner resolves cadence per question class

The scanner currently uses one `self._recent_returns_n` and reads with the slot's single dt. Make it resolve `(dt, n)` per question **class** from the strategy config, then read the matching `(symbol, dt)` series. Falls back to the slot default for classes without an override, so single-cadence slots are bit-identical.

**Files:**
- Modify: `hlanalysis/engine/scanner.py:115` (drop the single `_recent_returns_n` use in `scan`), `:117-143` (`_required_returns_n` → a per-class map), `:264-279` (the `recent_returns`/`recent_hl_bars` reads)
- Test: `tests/unit/test_scanner_per_class_cadence.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_scanner_per_class_cadence.py`:

```python
# tests/unit/test_scanner_per_class_cadence.py
"""Scanner reads σ-history at the per-question-class cadence: a v31 slot with a
priceBucket theta_override of vol_sampling_dt_seconds=2 must read the dt=2 bar
series for bucket questions while priceBinary reads the slot default dt=5."""
from __future__ import annotations

from hlanalysis.engine.config import (
    AllowlistEntry, GlobalRiskConfig, StrategyConfig, ThetaParams,
)
from hlanalysis.engine.scanner import Scanner


def _global() -> GlobalRiskConfig:
    return GlobalRiskConfig(
        max_total_inventory_usd=1100, max_concurrent_positions=5,
        daily_loss_cap_usd=100, max_strike_distance_pct=50,
        min_recent_volume_usd=100, stale_data_halt_seconds=30,
        reconcile_interval_seconds=15,
    )


def _entry(klass: str) -> AllowlistEntry:
    return AllowlistEntry(
        match={"class": klass, "underlying": "BTC"}, max_position_usd=500,
        stop_loss_pct=None, tte_min_seconds=0, tte_max_seconds=43200,
        price_extreme_threshold=0.0, distance_from_strike_usd_min=0, vol_max=100,
    )


def _cfg() -> StrategyConfig:
    defaults = AllowlistEntry(
        match={}, max_position_usd=500, stop_loss_pct=None, tte_min_seconds=0,
        tte_max_seconds=43200, price_extreme_threshold=0.0,
        distance_from_strike_usd_min=0, vol_max=100,
    )
    return StrategyConfig(
        name="theta_harvester", account_alias="v31", paper_mode=False,
        strategy_type="theta_harvester",
        allowlist=[_entry("priceBinary"), _entry("priceBucket")],
        blocklist_question_idxs=[], defaults=defaults,
        theta=ThetaParams(vol_lookback_seconds=3600, vol_sampling_dt_seconds=5),
        theta_overrides={"priceBucket": {"vol_sampling_dt_seconds": 2}},
        **{"global": _global()},
    )


def test_cadence_by_class_maps_each_class_to_its_dt() -> None:
    m = Scanner.cadence_by_class(_cfg())
    assert m["priceBinary"][0] == 5      # (dt_seconds, n)
    assert m["priceBucket"][0] == 2
    # dt=2 must request more bars than dt=5 for the same lookback
    assert m["priceBucket"][1] > m["priceBinary"][1]


def test_default_cadence_for_unmapped_class() -> None:
    m = Scanner.cadence_by_class(_cfg())
    dt_default, n_default = Scanner.default_cadence(_cfg())
    assert dt_default == 5
    # An unmapped class (e.g. a future "priceLadder") falls back to the default.
    assert "priceLadder" not in m
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_scanner_per_class_cadence.py -q`
Expected: FAIL — `Scanner.cadence_by_class` / `Scanner.default_cadence` do not exist.

- [ ] **Step 3: Add the cadence-resolution staticmethods**

In `scanner.py`, replace `_required_returns_n` (`:117-143`) with a shared bars-for-lookback helper plus the two classmethods. Keep the lookback inputs identical to today (defaults + allowlist + theta), only the **dt divisor** varies per class:

```python
    @staticmethod
    def _bars_for(secs: int, dt: int) -> int:
        """Number of dt-spaced bars covering ``secs`` of lookback, floored at 32
        (legacy) so downstream consumers assuming ≥32 (v3.4 LM K-of-N) keep working."""
        return max(32, (secs + dt - 1) // dt)

    @staticmethod
    def _lookback_secs(cfg: StrategyConfig) -> int:
        secs = cfg.defaults.vol_lookback_seconds
        for entry in cfg.allowlist:
            secs = max(secs, entry.vol_lookback_seconds)
        if cfg.theta is not None:
            secs = max(secs, cfg.theta.vol_lookback_seconds, cfg.theta.drift_lookback_seconds)
        return secs

    @classmethod
    def default_cadence(cls, cfg: StrategyConfig) -> tuple[int, int]:
        """(dt_seconds, n_bars) for the slot default cadence — used for any
        question class without a per-class theta override."""
        from .runtime import reference_sampling_dt_seconds  # avoid import cycle
        dt = reference_sampling_dt_seconds(cfg)
        return dt, cls._bars_for(cls._lookback_secs(cfg), dt)

    @classmethod
    def cadence_by_class(cls, cfg: StrategyConfig) -> dict[str, tuple[int, int]]:
        """Map question.klass -> (dt_seconds, n_bars) for classes whose
        theta_override sets vol_sampling_dt_seconds. Classes absent here use
        default_cadence(). Empty for non-theta slots or slots with no dt override."""
        out: dict[str, tuple[int, int]] = {}
        for klass, override in (cfg.theta_overrides or {}).items():
            if "vol_sampling_dt_seconds" not in override.model_fields_set:
                continue
            dt = int(override.vol_sampling_dt_seconds)
            out[klass] = (dt, cls._bars_for(cls._lookback_secs(cfg), dt))
        return out
```

- [ ] **Step 4: Wire the maps into `__init__` and the scan read**

In `Scanner.__init__`, replace `self._recent_returns_n = self._required_returns_n(cfg)` (`:115`) with:

```python
        self._default_cadence = self.default_cadence(cfg)        # (dt, n)
        self._cadence_by_class = self.cadence_by_class(cfg)       # klass -> (dt, n)
```

In `scan()` (`:264-279`), resolve per question then read at that cadence:

```python
            dt_s, ret_n = self._cadence_by_class.get(q.klass, self._default_cadence)
            decision = self.strategy.evaluate(
                question=q,
                books=books,
                reference_price=ref,
                recent_returns=self.ms.recent_returns(
                    self.ref_symbol, n=ret_n, dt=dt_s,
                ),
                recent_hl_bars=self.ms.recent_hl_bars(
                    self.ref_symbol, n=ret_n, dt=dt_s,
                ),
                recent_volume_usd=volume_total,
                position=strat_pos,
                now_ns=now_ns,
            )
```

- [ ] **Step 5: Run the new test + scanner suite**

Run: `uv run python -m pytest tests/unit/test_scanner_per_class_cadence.py tests/unit/test_scanner.py -q`
Expected: PASS. If any existing scanner test referenced `_required_returns_n` or `_recent_returns_n` by name, update it to `default_cadence(cfg)` (returns `(dt, n)`); the `n` is the second element.

- [ ] **Step 6: Commit**

```bash
git add hlanalysis/engine/scanner.py tests/unit/test_scanner_per_class_cadence.py tests/unit/test_scanner.py
git commit -m "feat(scanner): resolve reference cadence per question class"
```

---

## Task 3: Runtime registers every per-class cadence on the shared MarketState

`_register_reference_cadences` registers one cadence per slot today. It must additionally register each slot's per-class override cadences so the bucket series exist before the first tick.

**Files:**
- Modify: `hlanalysis/engine/runtime.py:611-628` (`_register_reference_cadences`)
- Test: `tests/unit/test_engine_runtime_cadence.py` (extend — existing file)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_engine_runtime_cadence.py`:

```python
def test_per_class_override_registers_extra_cadence(tmp_path) -> None:
    """A v31 theta slot with a priceBucket dt=2 override registers BOTH dt=5
    (default) and dt=2 on the shared MarketState for its reference symbol, so
    both bar series accumulate from the one BTC feed."""
    from hlanalysis.engine.config import ThetaParams
    cfg = _theta_cfg(alias="v31", reference_symbol="BTC", dt=5)
    cfg = cfg.model_copy(update={
        "theta_overrides": {"priceBucket": ThetaParams(vol_sampling_dt_seconds=2)},
    })
    rt = _runtime([cfg], tmp_path)
    rt._register_reference_cadences(rt.slots)
    assert rt.market_state.mark_bucket_ns_for("BTC", dt=5) == 5_000_000_000
    assert rt.market_state.mark_bucket_ns_for("BTC", dt=2) == 2_000_000_000
```

(Confirm `_runtime(...)` exposes `.slots` and `.market_state`; if the helper builds slots differently, register via the same path `_register_reference_cadences` consumes. `_theta_cfg` already exists in this file at `:51`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_engine_runtime_cadence.py::test_per_class_override_registers_extra_cadence -q`
Expected: FAIL — only dt=5 is registered; `mark_bucket_ns_for("BTC", dt=2)` returns the dt-less default (5s), not 2s.

- [ ] **Step 3: Register per-class cadences**

In `_register_reference_cadences` (`:611-628`), after the existing default-cadence registration, add the override cadences. Import the scanner resolver lazily to avoid a cycle:

```python
    def _register_reference_cadences(self, slots: list[AccountSlot]) -> None:
        """Register each slot's default reference cadence AND any per-class
        theta-override cadences on the shared MarketState, so every (symbol, dt)
        bar series exists and accumulates from the one shared feed."""
        from .scanner import Scanner
        for slot in slots:
            sym = slot.cfg.reference_symbol
            self.market_state.set_reference_cadence(
                sym,
                sampling_dt_seconds=reference_sampling_dt_seconds(slot.cfg),
                lookback_seconds=reference_vol_lookback_seconds(slot.cfg),
            )
            for dt_s, _n in Scanner.cadence_by_class(slot.cfg).values():
                self.market_state.set_reference_cadence(
                    sym,
                    sampling_dt_seconds=dt_s,
                    lookback_seconds=reference_vol_lookback_seconds(slot.cfg),
                )
            self.market_state.set_reference_source(
                sym, slot.cfg.reference_sigma_source,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/unit/test_engine_runtime_cadence.py -q`
Expected: PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/engine/runtime.py tests/unit/test_engine_runtime_cadence.py
git commit -m "feat(engine): register per-class override cadences on shared MarketState"
```

---

## Task 4: Lift the per-class dt guard + update the old conflict-guard tests

With multi-cadence support landed, a per-class `vol_sampling_dt_seconds` override is now legal. Remove the `ValueError` guard added in the prior PR and flip its test from "rejected" to "accepted". Update any MarketState test that asserted the old single-cadence conflict-raise.

**Files:**
- Modify: `hlanalysis/engine/runtime.py` (`build_theta_harvester_configs_by_class` — remove the `vol_sampling_dt_seconds` rejection block)
- Modify: `tests/unit/test_theta_per_class_overrides.py` (`test_vol_sampling_dt_override_rejected` → accepted)
- Modify: any `tests/unit/test_market_state*.py` asserting `set_reference_cadence` raises on a second cadence

- [ ] **Step 1: Flip the override test (write the new expectation first)**

In `tests/unit/test_theta_per_class_overrides.py`, replace `test_vol_sampling_dt_override_rejected` with:

```python
def test_vol_sampling_dt_override_now_allowed() -> None:
    """Post (symbol, dt) refactor: a per-class vol_sampling_dt_seconds override
    is legal — the engine maintains an independent bar series per cadence."""
    cfg = _theta_cfg(theta_overrides={"priceBucket": {"vol_sampling_dt_seconds": 2}})
    by_class = build_theta_harvester_configs_by_class(cfg)
    assert by_class["priceBucket"].vol_sampling_dt_seconds == 2
    base = build_theta_harvester_config(cfg)
    assert base.vol_sampling_dt_seconds == 5  # binary keeps the shared default
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run python -m pytest tests/unit/test_theta_per_class_overrides.py::test_vol_sampling_dt_override_now_allowed -q`
Expected: FAIL — the build still raises `ValueError(... vol_sampling_dt_seconds ...)`.

- [ ] **Step 3: Remove the guard**

In `build_theta_harvester_configs_by_class` (`runtime.py`), delete the block:

```python
        if "vol_sampling_dt_seconds" in set_fields:
            raise ValueError(
                f"strategy '{cfg.name}' (alias={cfg.account_alias}): "
                ...
            )
```

Update the function docstring: drop the "vol_sampling_dt_seconds is rejected per-class" paragraph; replace with a note that per-class cadences are realized via the (symbol, dt) MarketState bucketing + Scanner per-class resolution.

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run python -m pytest tests/unit/test_theta_per_class_overrides.py -q`
Expected: PASS.

- [ ] **Step 5: Update any stale MarketState conflict-guard tests**

Run: `uv run python -m pytest tests/unit/ -k "market_state or cadence" -q`
For any test asserting `set_reference_cadence` raises on a second distinct cadence (the OLD lockstep guard), change it to assert both cadences coexist (per Task 1's `test_same_cadence_reregistration_is_idempotent` / `test_two_cadences_bucket_same_stream_independently`). If a test asserted v1↔v31 lockstep specifically, retarget it to assert independence is now permitted.
Expected after edits: PASS.

- [ ] **Step 6: Commit**

```bash
git add hlanalysis/engine/runtime.py tests/unit/test_theta_per_class_overrides.py tests/unit/test_market_state*.py tests/unit/test_engine_runtime_cadence.py
git commit -m "feat(engine): allow per-class vol_sampling_dt now that bucketing is (symbol, dt)"
```

---

## Task 5: Replay parity (single-cadence bit-identity) + suite

`ReplayRunner` uses a single cadence and the dt-less readers, so it should be bit-identical after Task 1. Add an explicit parity test pinning that, then run the full suite.

**Files:**
- Test: `tests/unit/test_replay_cadence_parity.py` (create)

- [ ] **Step 1: Write the parity test**

Create `tests/unit/test_replay_cadence_parity.py`:

```python
# tests/unit/test_replay_cadence_parity.py
"""ReplayRunner single-cadence reads stay bit-identical after the (symbol, dt)
refactor: a dt-less read resolves to the sole registered cadence."""
from __future__ import annotations

from hlanalysis.engine.market_state import MarketState
from hlanalysis.events import MarkEvent, Mechanism, ProductType


def test_dtless_read_equals_explicit_dt_for_single_cadence() -> None:
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5, lookback_seconds=1800)
    for t in range(50):
        ts = t * 1_000_000_000
        ms.apply(MarkEvent(
            venue="hyperliquid", product_type=ProductType.PERP,
            mechanism=Mechanism.CLOB, symbol="BTC",
            exchange_ts=ts, local_recv_ts=ts, mark_px=100.0 + (t % 7) * 0.5,
        ))
    assert ms.recent_returns("BTC", n=32) == ms.recent_returns("BTC", n=32, dt=5)
    assert ms.recent_hl_bars("BTC", n=32) == ms.recent_hl_bars("BTC", n=32, dt=5)
```

- [ ] **Step 2: Run it**

Run: `uv run python -m pytest tests/unit/test_replay_cadence_parity.py -q`
Expected: PASS (Task 1 already implements the default resolution).

- [ ] **Step 3: Run the full suite**

Run: `uv run python -m pytest -q`
Expected: PASS (all tests, ~890+ green). Investigate and fix any regression before proceeding — the bit-identity guarantee means a single-cadence failure is a real defect in Task 1's default resolution.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_replay_cadence_parity.py
git commit -m "test(engine): pin single-cadence replay bit-identity after (symbol, dt) refactor"
```

---

## Out of scope (flag for a follow-up, do not implement here)

- **Backtest tuner per-class dt.** `ReplayRunner` and the backtest data path drive σ from a single cadence (`replay.py:44-55`, backtest `KlineRingBuffer`). The bucket tune that produced the dt=2 result ran a bucket-only backtest at dt=2. To *validate* a mixed dt=2-bucket / dt=5-binary config end-to-end before live, the backtest runner needs the same per-class cadence resolution. That is a separate plan — this one is the live-engine refactor only.
- **Flipping live config values.** This plan changes no `config/strategy.yaml` values. Realizing dt=2 for buckets = the operator adds `vol_sampling_dt_seconds: 2` under `theta_overrides.priceBucket` after backtest validation + the cadence-port paper-validation gate.
- **Decoupling v1↔v31 lockstep deliberately.** This refactor *permits* it (the conflict guard is gone), but no v1/v31 config is changed here.

---

## Self-review notes

- **Spec coverage:** (symbol, dt) re-key (Task 1) ✓; per-class scanner read (Task 2) ✓; runtime registration of override cadences (Task 3) ✓; revert dt guard (Task 4) ✓; single-cadence bit-identity (Tasks 1, 5) ✓.
- **Type consistency:** `cadence_by_class` and `default_cadence` both return/use `(dt_seconds, n_bars)` tuples; `recent_returns`/`recent_hl_bars`/`mark_bucket_ns_for` all take `dt: int | None = None` (seconds) and resolve to ns internally; registration keys are `(symbol, dt_ns)`.
- **Bit-identity invariant:** every new `(symbol, dt)` map defaults (dt=None) to the symbol's first registered cadence; with one cadence registered this is the only key, so reads equal the legacy per-symbol path (Tasks 1 step 7, Task 5).
