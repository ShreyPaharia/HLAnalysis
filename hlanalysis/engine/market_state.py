"""Engine MarketState — a thin adapter over the shared marketdata core (SHR-87).

Spec 2 / SHR-73 unifies the two MarketState implementations (live engine vs
backtest) onto ONE shared, pure event-driven core
(``hlanalysis/marketdata/market_state.py``, SHR-81) so the strategy sees
*identical* σ / returns / volume inputs on both paths — eliminating the
train/serve input skew by code reuse rather than a diff harness.

This module is the LIVE-MONEY adapter. It owns a shared-core ``MarketState``
and delegates the **book/returns/σ/volume math** to it:

  * reference-price ticks (perp mark / BBO mid) → ``apply_reference_tick`` →
    per-(symbol, dt) OHLC bars → ``recent_returns`` / ``recent_hl_bars`` /
    ``last_mark`` / ``last_mark_ts``;
  * trade prints → ``apply_trade`` → ``recent_volume_usd`` (1h rolling window).

It keeps the **engine-specific edges** on this side (the data-in edge of the
Spec-2 architecture), because they are not part of the shared core's contract:

  * the L2 **book** (BBO / snapshot / delta), whose HL semantics — a BBO updates
    top-of-book WITHOUT touching the deeper levels — the shared core's
    snapshot-coupled book model cannot represent. ``book()`` therefore reads
    this side's book and is byte-identical to the pre-refactor engine;
  * the HIP-4 / PM **question / strike / settlement** registry;
  * the per-symbol **reference source** (``mark`` | ``bbo``) selection and its
    fail-fast conflict guard;
  * the **cadence** registration bookkeeping (``mark_bucket_ns_for`` and the
    ``_cadences_by_symbol`` / ``_mark_history_by_key`` attributes other engine
    components and tests read), forwarded to the core so its bar buffers exist.

The legacy **count-path** reads (``recent_returns(symbol, n)`` with no
``now_ns``) are served from the shared core's full bar history — the
unbounded-window special case of the time-bounded slice — preserving the old
``[-(n+1):]`` semantics for callers that have not moved to the time-bounded
window (scanner/replay both pass ``now_ns`` + ``lookback_seconds``).

Bit-identity is proven against the pre-refactor engine on a real-engine replay
of the recorded HL corpus in
``tests/unit/test_market_state_shr87_replay_parity.py`` (the SHR-87 gate).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..events import (
    BboEvent,
    BookSnapshotEvent,
    BookDeltaEvent,
    MarkEvent,
    NormalizedEvent,
    QuestionMetaEvent,
    SettlementEvent,
    TradeEvent,
)
from ..marketdata.market_state import MarketState as _SharedMarketState
from ..strategy.types import BookState, QuestionView

# Legacy COUNT-path "infinite window": a now_ns above any real epoch-ns and a
# lookback wide enough that the cutoff precedes any bar, so the shared core's
# time-bounded slice returns the FULL bar history. The count path then takes the
# last ``n`` of that — reproducing the pre-SHR-87 deque ``[-(n+1):]`` read. The
# cutoff (now_ns − lookback·1e9) stays within int64 range so numpy.searchsorted
# is happy: (2^63−1) − 1e10·1e9 ≈ −7.8e17.
_COUNT_NOW_NS: int = (1 << 63) - 1
_COUNT_LOOKBACK_S: int = 10_000_000_000


@dataclass
class _MutableBook:
    bid_px: float | None = None
    bid_sz: float | None = None
    ask_px: float | None = None
    ask_sz: float | None = None
    last_trade_ts_ns: int = 0
    last_l2_ts_ns: int = 0
    ask_levels: tuple[tuple[float, float], ...] = ()
    bid_levels: tuple[tuple[float, float], ...] = ()


class MarketState:
    """In-memory live market state. Pure (no IO, no asyncio).

    Thin adapter over the shared ``marketdata.MarketState`` core (SHR-87). Owns
    the engine-specific edges (book, question registry, reference source /
    cadence config) and delegates the returns / σ / volume / last_mark math to
    the shared core.
    """

    def __init__(
        self,
        *,
        volume_window_ns: int = 60 * 60 * 1_000_000_000,  # 1 hour
        mark_history: int = 256,
        mark_bucket_ns: int = 60 * 1_000_000_000,  # 1 minute
    ) -> None:
        # The shared core owns the OHLC bar buffers (returns/HL/σ + last_mark)
        # and the rolling-volume window. The engine forwards reference ticks and
        # trades into it and reads through its query surface.
        self._core = _SharedMarketState(
            volume_window_ns=volume_window_ns,
            default_bucket_ns=mark_bucket_ns,
        )
        # ---- engine-side edges (NOT in the shared core's contract) ----
        self._books: dict[str, _MutableBook] = {}
        self._questions: dict[int, QuestionView] = {}
        self._mark_bucket_ns: int = mark_bucket_ns  # default (60s) for unregistered symbols
        self._mark_history: int = mark_history  # default history sizing (legacy)
        # Per (symbol, dt_ns) history-size bookkeeping. Vestigial for storage
        # (the shared core's buffers grow as needed) but preserved because
        # other engine components and tests read it as the configured σ-window
        # sizing. See ``set_reference_cadence``.
        self._mark_history_by_key: dict[tuple[str, int], int] = {}
        # Cadences registered per symbol, in registration order. The FIRST
        # registered cadence is the symbol's default (what a dt-less read
        # resolves to), preserving single-cadence bit-identity. Mirrors the
        # cadences forwarded to the shared core.
        self._cadences_by_symbol: dict[str, list[int]] = {}
        # Per reference symbol, which feed sources the σ/OHLC reference:
        #   "mark" (default) — the venue MarkEvent (HL perp mark; Binance perp
        #     mark REST-poll). Legacy behaviour, unchanged.
        #   "bbo"            — the dense BBO mid = (bid_px+ask_px)/2. Used to
        #     give PM (BTCUSDT_SPOT) a sub-second reference so dt=5 bars don't
        #     degenerate. The chosen feed drives BOTH the OHLC bars and
        #     ``last_mark``. A symbol can be fed only one way (shared history),
        #     so conflicting registrations raise — same fail-fast rule as
        #     ``set_reference_cadence``.
        self._reference_source_by_symbol: dict[str, str] = {}
        # Cached symbol→question_idx map. Rebuilt lazily on first access after
        # _update_question fires, and reused thereafter.
        self._sym_to_q_cache: dict[str, int] | None = None

    # ---- cadence registration ----

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

        ``lookback_seconds`` records the σ window the slot will request so other
        engine components can size against it; the shared core's bar buffers grow
        as needed, so this no longer bounds a fixed ring. The cadence is
        forwarded to the shared core so its (symbol, dt) bar buffer exists and
        the reference-tick stream fans into it.
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
            # floor + 2: +1 for the n+1 prices needed to form n returns, +1
            # margin so the oldest bar isn't evicted before the window fills.
            needed = int(lookback_seconds) // int(sampling_dt_seconds) + 2
            prev = self._mark_history_by_key.get(key, self._mark_history)
            self._mark_history_by_key[key] = max(prev, needed)
        # Forward to the shared core so the (symbol, dt) bar buffer exists and
        # reference ticks fan into it (the core dedups + orders identically).
        self._core.set_reference_cadence(
            symbol,
            sampling_dt_seconds=int(sampling_dt_seconds),
            lookback_seconds=lookback_seconds,
        )

    def _resolve_dt_ns(self, symbol: str, dt: int | None) -> int:
        """Bucket period (ns) for a (symbol, dt) read. ``dt=None`` resolves to
        the symbol's FIRST registered cadence (single-cadence bit-identity), or
        the global default if the symbol was never registered."""
        if dt is not None:
            return int(dt) * 1_000_000_000
        cadences = self._cadences_by_symbol.get(symbol)
        return cadences[0] if cadences else self._mark_bucket_ns

    def _read_dt_seconds(self, symbol: str, dt: int | None) -> int:
        """Cadence (seconds) to read the shared core at — the explicit ``dt`` or
        the symbol's resolved default cadence. Passing it explicitly to the core
        keeps the read on the exact buffer the engine resolved."""
        return self._resolve_dt_ns(symbol, dt) // 1_000_000_000

    def _history_maxlen(self, symbol: str, dt_ns: int) -> int:
        """Bar-history capacity for ``(symbol, dt)`` — the bounded ring the
        pre-SHR-87 engine used. The shared core's buffers grow unbounded, so the
        read paths re-impose this cap to stay byte-identical: a read sees at most
        the most-recent ``maxlen`` bars (hence ``maxlen-1`` returns), exactly as
        the old ``deque(maxlen=...)`` did. Live this is a no-op
        (``reference_vol_lookback_seconds`` sizes ``maxlen`` ≥ the read window);
        it only bites the legacy COUNT path over long histories."""
        return self._mark_history_by_key.get((symbol, dt_ns), self._mark_history)

    def mark_bucket_ns_for(self, symbol: str, dt: int | None = None) -> int:
        """The bucket period (ns) applied to ``symbol`` at cadence ``dt`` (or the
        symbol's default cadence when dt is None). Exposed so the runtime/tests
        can assert no train/serve skew against the strategy's configured
        vol_sampling_dt_seconds."""
        return self._resolve_dt_ns(symbol, dt)

    def set_reference_source(self, symbol: str, source: str) -> None:
        """Register which feed sources ``symbol``'s σ/OHLC reference.

        ``"mark"`` (default) reads the venue MarkEvent; ``"bbo"`` reads the
        dense BBO mid. The shared OHLC history for a symbol can only be fed
        one way, so two slots reading the same reference symbol must agree on
        the source — a conflicting re-registration raises (fail-fast) rather
        than silently feeding the shared history two ways. Note: unlike
        ``set_reference_source``, ``set_reference_cadence`` does NOT raise on
        a second distinct cadence; multiple cadences per symbol are accepted
        and bucketed independently.
        """
        if source not in ("mark", "bbo"):
            raise ValueError(
                f"reference source must be 'mark' or 'bbo', got {source!r}"
            )
        prev = self._reference_source_by_symbol.get(symbol)
        if prev is not None and prev != source:
            raise ValueError(
                f"conflicting reference source for symbol {symbol!r}: already "
                f"registered as {prev!r}, refused re-registration as {source!r}. "
                f"All strategy slots reading the same reference_symbol must share "
                f"one σ source (the shared OHLC history can only be fed one way)."
            )
        self._reference_source_by_symbol[symbol] = source

    def reference_source_for(self, symbol: str) -> str:
        """The σ/OHLC source for ``symbol`` — the per-symbol override if
        registered, else the ``"mark"`` default."""
        return self._reference_source_by_symbol.get(symbol, "mark")

    # ---- ingest ----

    def apply(self, ev: NormalizedEvent) -> None:
        match ev:
            case BboEvent():
                b = self._books.setdefault(ev.symbol, _MutableBook())
                b.bid_px, b.bid_sz = ev.bid_px, ev.bid_sz
                b.ask_px, b.ask_sz = ev.ask_px, ev.ask_sz
                b.last_l2_ts_ns = max(b.last_l2_ts_ns, ev.exchange_ts or ev.local_recv_ts)
                # When this symbol is bbo-sourced, the BBO mid feeds the σ/OHLC
                # reference (mirrors backtest `_load_binance_bbo_reference`:
                # mid=(bid+ask)/2, per-bucket OHLC). Mark-sourced symbols ignore
                # the bbo for σ (book only) — legacy behaviour.
                if self._reference_source_by_symbol.get(ev.symbol) == "bbo":
                    mid = (ev.bid_px + ev.ask_px) / 2.0
                    self._core.apply_reference_tick(
                        ev.symbol, ts_ns=ev.exchange_ts or ev.local_recv_ts, price=mid,
                    )
            case BookSnapshotEvent():
                b = self._books.setdefault(ev.symbol, _MutableBook())
                if ev.bid_px:
                    b.bid_px, b.bid_sz = ev.bid_px[0], ev.bid_sz[0]
                    b.bid_levels = tuple(zip(ev.bid_px, ev.bid_sz, strict=False))
                if ev.ask_px:
                    b.ask_px, b.ask_sz = ev.ask_px[0], ev.ask_sz[0]
                    b.ask_levels = tuple(zip(ev.ask_px, ev.ask_sz, strict=False))
                b.last_l2_ts_ns = max(b.last_l2_ts_ns, ev.exchange_ts or ev.local_recv_ts)
            case BookDeltaEvent():
                # Phase 1 strategy uses top-of-book only; deltas that hit the top
                # update it via subsequent bbo events. We still track recv ts for staleness.
                b = self._books.setdefault(ev.symbol, _MutableBook())
                b.last_l2_ts_ns = max(b.last_l2_ts_ns, ev.exchange_ts or ev.local_recv_ts)
            case TradeEvent():
                ts = ev.exchange_ts or ev.local_recv_ts
                # Volume accounting (1h rolling window) lives in the shared core.
                self._core.apply_trade(ev.symbol, ts_ns=ts, price=ev.price, size=ev.size)
                b = self._books.setdefault(ev.symbol, _MutableBook())
                b.last_trade_ts_ns = max(b.last_trade_ts_ns, ts)
            case MarkEvent():
                # bbo-sourced symbols take their σ/OHLC reference from the BBO
                # mid (handled in the BboEvent case); a stray MarkEvent for such
                # a symbol must NOT touch the reference price or bars.
                if self._reference_source_by_symbol.get(ev.symbol) != "bbo":
                    self._core.apply_reference_tick(
                        ev.symbol, ts_ns=ev.exchange_ts or ev.local_recv_ts,
                        price=ev.mark_px,
                    )
            case QuestionMetaEvent():
                self._update_question(ev)
            case SettlementEvent():
                self._mark_settled(ev)
            case _:
                pass  # other events ignored in Phase 1

    def _update_question(self, ev: QuestionMetaEvent) -> None:
        kv = dict(zip(ev.keys, ev.values, strict=False))
        try:
            strike = float(kv.get("strike", "nan"))
        except ValueError:
            strike = float("nan")
        # priceBinary expiry is keyed as "expiry" with format YYYYMMDD-HHMM
        # (HL convention). PM adapters mirror the same key so this path is
        # venue-agnostic; `expiry_ns` is accepted as an epoch-ns fallback
        # for older PM payloads.
        expiry_ns = self._parse_expiry_ns(kv.get("expiry", ""))
        if expiry_ns == 0:
            try:
                expiry_ns = int(kv.get("expiry_ns", "0") or 0)
            except ValueError:
                expiry_ns = 0
        # Leg symbols must equal the symbols the venue's book/trade frames are
        # keyed by, so the scanner's books.get(leg) and live orders use the
        # right id.
        #   - Polymarket: the ERC-1155 CLOB token ids (yes_token_id /
        #     no_token_id), which is exactly what the PM WS frames carry as
        #     asset_id and what create_order wants as token_id.
        #   - Hyperliquid HIP-4: coins keyed f"#{10*outcome_idx + side_idx}"
        #     where side_idx=0 is YES, 1 is NO (see adapters/hyperliquid.py).
        #     priceBinary: 1 outcome × 2 sides → 2 legs; priceBucket: N×2 legs.
        if ev.venue == "polymarket":
            if kv.get("class") == "priceBucket":
                leg_symbols = tuple(
                    t for t in kv.get("leg_token_ids", "").split(",") if t
                )
            else:
                yes_t = kv.get("yes_token_id", "")
                no_t = kv.get("no_token_id", "")
                leg_symbols = tuple(t for t in (yes_t, no_t) if t)
        else:
            outcomes = sorted(ev.named_outcome_idxs)
            leg_symbols = tuple(
                f"#{10 * o + s}" for o in outcomes for s in (0, 1)
            )
        yes_symbol = leg_symbols[0] if leg_symbols else ""
        no_symbol = leg_symbols[1] if len(leg_symbols) >= 2 else ""
        existing = self._questions.get(ev.question_idx)
        settled = bool(ev.settled_named_outcome_idxs) if not (existing and existing.settled) else True
        settled_side: str | None = None
        if existing and existing.settled and existing.settled_side:
            settled_side = existing.settled_side
        # Mirror keys/values pairs onto QuestionView for downstream rendering
        # (alerts, reports). Skip the bulky question_description so the tuple
        # stays compact.
        kv_pairs = tuple(
            (k, v) for k, v in zip(ev.keys, ev.values, strict=False)
            if k != "question_description"
        )
        question_name = kv.get("question_name", "")
        # leg_symbols can change between QuestionMetaEvent emissions (e.g. on
        # outcomeCreated rolls), so invalidate the cache on every meta update.
        # Settlement-only updates go through mark_question_settled and don't
        # need to invalidate (leg layout is unchanged).
        self._sym_to_q_cache = None
        self._questions[ev.question_idx] = QuestionView(
            question_idx=ev.question_idx,
            yes_symbol=yes_symbol,
            no_symbol=no_symbol,
            strike=strike,
            expiry_ns=expiry_ns,
            underlying=kv.get("underlying", ""),
            klass=kv.get("class", ""),
            period=kv.get("period", ""),
            settled=settled,
            settled_side=settled_side,
            leg_symbols=leg_symbols,
            name=question_name,
            kv=kv_pairs,
            venue=ev.venue,
        )

    def set_question_strike(self, question_idx: int, strike: float) -> bool:
        """Stamp a question's strike, but only if it isn't already set.

        Used to reload a persisted PM open-strike after a restart (see
        ``EngineRuntime._maybe_capture_pm_strike`` and StateDAL.get_pm_strike). Never clobbers
        a strike already known. Returns True iff the strike was stamped.
        """
        q = self._questions.get(question_idx)
        if q is None or q.strike == q.strike:  # unknown, or already non-NaN
            return False
        self._questions[question_idx] = dataclasses.replace(q, strike=strike)
        return True

    def mark_question_settled(self, question_idx: int) -> bool:
        """Mark a question settled by its question_idx.

        Used by the reconciler when it detects a local position has vanished
        from the venue — on HL HIP-4 that's overwhelmingly a settlement
        auto-close, and the polled SettlementEvent typically lags by tens of
        seconds. Marking the question settled here suppresses STALE DATA HALT
        on the now-silent legs and prevents the strategy from re-entering
        before the polled event arrives.

        Idempotent: returns True on the first state transition, False if the
        question was already marked settled (or the question_idx is unknown).
        Side is left as None — we have no way to know which side won from a
        position-vanished signal alone, and the alert payload doesn't need it
        (settled_side is informational for non-binary questions).
        """
        q = self._questions.get(question_idx)
        if q is None or q.settled:
            return False
        self._questions[question_idx] = dataclasses.replace(q, settled=True)
        return True

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

    def _mark_settled(self, ev: SettlementEvent) -> None:
        # SettlementEvent.symbol is one leg; find the question whose leg_symbols
        # contains it and mark it settled. Multi-outcome (priceBucket) has up to
        # 6 legs; the binary 2-leg shorthand fails for those.
        for idx, q in list(self._questions.items()):
            legs = q.leg_symbols or (
                (q.yes_symbol, q.no_symbol) if q.yes_symbol else ()
            )
            if ev.symbol not in legs:
                continue
            leg_idx = legs.index(ev.symbol)
            # side_idx 0 = YES, 1 = NO (per HL coin convention #{10*o + s}).
            # For non-binary questions, "side" is informational only — the
            # canonical "did our position win?" check uses `settled_symbol`
            # below (which identifies the exact winning leg, including
            # outcome index for buckets where side alone is ambiguous).
            side: str = "yes" if leg_idx % 2 == 0 else "no"
            prev = q.settled_symbols
            winners = prev if ev.symbol in prev else prev + (ev.symbol,)
            self._questions[idx] = dataclasses.replace(
                q, settled=True, settled_side=side,
                settled_symbol=ev.symbol, settled_symbols=winners,
            )

    @staticmethod
    def _parse_expiry_ns(expiry: str) -> int:
        # YYYYMMDD-HHMM -> ns since epoch (UTC). Returns 0 if unparseable.
        from datetime import datetime, timezone
        try:
            dt = datetime.strptime(expiry, "%Y%m%d-%H%M").replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1_000_000_000)
        except ValueError:
            return 0

    # ---- query ----

    def book(self, symbol: str) -> BookState | None:
        b = self._books.get(symbol)
        if b is None:
            return None
        return BookState(
            symbol=symbol,
            bid_px=b.bid_px,
            bid_sz=b.bid_sz,
            ask_px=b.ask_px,
            ask_sz=b.ask_sz,
            last_trade_ts_ns=b.last_trade_ts_ns,
            last_l2_ts_ns=b.last_l2_ts_ns,
            ask_levels=b.ask_levels,
            bid_levels=b.bid_levels,
        )

    def question(self, idx: int) -> QuestionView | None:
        return self._questions.get(idx)

    def all_questions(self) -> list[QuestionView]:
        return list(self._questions.values())

    def symbol_to_question_map(self) -> dict[str, int]:
        """sym → question_idx for every known leg. Cached, invalidated on
        QuestionMetaEvent ingestion. The reconciler uses this every cycle to
        attribute venue positions back to a qidx."""
        if self._sym_to_q_cache is not None:
            return self._sym_to_q_cache
        m: dict[str, int] = {}
        for q in self._questions.values():
            legs = q.leg_symbols or (
                (q.yes_symbol, q.no_symbol) if q.yes_symbol else ()
            )
            for sym in legs:
                if sym:
                    m[sym] = q.question_idx
        self._sym_to_q_cache = m
        return m

    def last_mark(self, symbol: str) -> float | None:
        return self._core.last_mark(symbol)

    def last_mark_ts(self, symbol: str) -> int | None:
        """Timestamp (ns) of the latest reference tick for ``symbol``, or None
        if none seen yet. Used by the Scanner to compute reference-feed age for
        the risk gate's stale-reference check (SHR-60)."""
        return self._core.last_mark_ts(symbol)

    def recent_returns(
        self,
        symbol: str,
        n: int,
        dt: int | None = None,
        *,
        now_ns: int | None = None,
        lookback_seconds: int | None = None,
    ) -> tuple[float, ...]:
        """Last ``n`` close-to-close log returns for ``symbol`` at cadence ``dt``
        (default = the symbol's first registered cadence).

        When ``now_ns`` and ``lookback_seconds`` are both provided, the bar
        sequence is TIME-bounded to the SHR-66 window (the live scanner/replay
        path) and delegated straight to the shared core. Without them the legacy
        COUNT path is used: the shared core's FULL bar history sliced to the last
        ``n`` returns — preserving the pre-SHR-87 ``[-(n+1):]`` semantics for
        callers not yet on the time-bounded window.
        """
        dt_s = self._read_dt_seconds(symbol, dt)
        maxlen = self._history_maxlen(symbol, dt_s * 1_000_000_000)
        if now_ns is not None and lookback_seconds is not None:
            arr = self._core.recent_returns(
                symbol, now_ns=now_ns, lookback_seconds=lookback_seconds, dt=dt_s,
            )
            # The old deque held ≤ maxlen bars → ≤ maxlen-1 windowed returns.
            cap = maxlen - 1
        else:
            arr = self._core.recent_returns(
                symbol, now_ns=_COUNT_NOW_NS, lookback_seconds=_COUNT_LOOKBACK_S,
                dt=dt_s,
            )
            # Legacy COUNT path: old read `[-(n+1):]` bars off a ≤ maxlen deque.
            cap = min(n, maxlen - 1)
        if cap < arr.shape[0]:
            arr = arr[-cap:] if cap > 0 else arr[:0]
        return tuple(arr.tolist())

    def recent_hl_bars(
        self,
        symbol: str,
        n: int,
        dt: int | None = None,
        *,
        now_ns: int | None = None,
        lookback_seconds: int | None = None,
    ) -> tuple[tuple[float, float], ...]:
        """Last ``n`` per-bucket ``(high, low)`` bars for ``symbol`` at cadence
        ``dt`` (default = the symbol's first registered cadence).

        Feeds the strategy's Parkinson σ estimator the same input shape on the
        live path as in the backtest. Time-bounded when ``now_ns`` +
        ``lookback_seconds`` are given (SHR-66, the live path), else the legacy
        COUNT path (last ``n`` bars of the full history). Empty tuple when the
        symbol has no bars yet.
        """
        dt_s = self._read_dt_seconds(symbol, dt)
        maxlen = self._history_maxlen(symbol, dt_s * 1_000_000_000)
        if now_ns is not None and lookback_seconds is not None:
            arr = self._core.recent_hl_bars(
                symbol, now_ns=now_ns, lookback_seconds=lookback_seconds, dt=dt_s,
            )
            cap = maxlen  # old deque held ≤ maxlen (high, low) bars
        else:
            arr = self._core.recent_hl_bars(
                symbol, now_ns=_COUNT_NOW_NS, lookback_seconds=_COUNT_LOOKBACK_S,
                dt=dt_s,
            )
            cap = min(n, maxlen)
        if cap < arr.shape[0]:
            arr = arr[-cap:]
        return tuple((float(h), float(low)) for h, low in arr)

    def recent_volume_usd(self, symbol: str, *, now: int) -> float:
        return self._core.recent_volume_usd(symbol, now_ns=now)
