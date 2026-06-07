from __future__ import annotations

import dataclasses
import math
from collections import deque
from dataclasses import dataclass, field

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
from ..marketdata.ohlc import bucket_index, update_bar
from ..strategy.types import BookState, QuestionView


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
    """In-memory market state. Pure (no IO, no asyncio).

    Owns:
      - per-symbol BBO/L2 (top-of-book is sufficient for Phase 1 strategy)
      - per-symbol recent trades (for last-hour volume)
      - per-symbol last mark
      - HIP-4 question registry (idx -> QuestionView)
      - per-perp tail of marks for realized-vol calc
    """

    def __init__(
        self,
        *,
        volume_window_ns: int = 60 * 60 * 1_000_000_000,  # 1 hour
        mark_history: int = 256,
        mark_bucket_ns: int = 60 * 1_000_000_000,  # 1 minute
    ) -> None:
        self._books: dict[str, _MutableBook] = {}
        self._questions: dict[int, QuestionView] = {}
        self._trades: dict[str, deque[TradeEvent]] = {}
        self._last_mark: dict[str, float] = {}
        # Timestamp of the latest reference tick per symbol; lets the risk gate
        # detect a stale reference feed (SHR-60).
        self._last_mark_ts: dict[str, int] = {}
        # 2026-05-21: bucket marks into ``mark_bucket_ns``-wide windows so the
        # strategy's σ formula (which assumes 60s-spaced returns) sees a
        # correctly-spaced series. Without this, HL's ~1.2/s markPx feed
        # produces 32 sub-second returns that the strategy then annualizes
        # as if they were 32 minutes — σ collapses to the floor and
        # p_model/safety_d go to extremes. See bug memo
        # `engine_sigma_sampling_bug_2026_05_21.md`.
        # Per (symbol, dt_ns): a deque of per-bucket OHLC bars (high, low, close).
        # One reference-tick stream fans into every cadence registered for the
        # symbol, so a single slot can run different vol_sampling_dt_seconds per
        # question class off the SAME feed (see the (symbol, dt) refactor plan).
        # Each bar stores (ts_ns, high, low, close) so the time-bounded
        # recent_returns / recent_hl_bars paths can filter by wall-clock time
        # to match the backtest's slice_window semantics (SHR-66).
        self._marks: dict[tuple[str, int], deque[tuple[int, float, float, float]]] = {}
        self._mark_history: int = mark_history  # default deque maxlen (legacy)
        self._mark_history_by_key: dict[tuple[str, int], int] = {}
        self._mark_bucket_ns: int = mark_bucket_ns  # default (60s) for unregistered symbols
        self._default_cadences: tuple[int, ...] = (self._mark_bucket_ns,)
        # Cadences registered per symbol, in registration order. The FIRST
        # registered cadence is the symbol's default (what a dt-less read
        # resolves to), preserving single-cadence bit-identity.
        self._cadences_by_symbol: dict[str, list[int]] = {}
        # Per reference symbol, which feed sources the σ/OHLC reference:
        #   "mark" (default) — the venue MarkEvent (HL perp mark; Binance perp
        #     mark REST-poll). Legacy behaviour, unchanged.
        #   "bbo"            — the dense BBO mid = (bid_px+ask_px)/2. Used to
        #     give PM (BTCUSDT_SPOT) a sub-second reference so dt=5 bars don't
        #     degenerate (the 3s mark poll yields ~1.6 pts/5s bar). The chosen
        #     feed drives BOTH the OHLC bars and ``last_mark`` so the strategy's
        #     reference price S is consistent with its σ source. A symbol can be
        #     fed only one way (shared history), so conflicting registrations
        #     raise — same fail-fast rule as ``set_reference_cadence``.
        self._reference_source_by_symbol: dict[str, str] = {}
        # Last-bucket-id written per (symbol, dt_ns); coalesces within-bucket
        # reference-price updates into one bar per cadence.
        self._last_mark_bucket: dict[tuple[str, int], int] = {}
        self._volume_window_ns = volume_window_ns
        # Cached symbol→question_idx map. Rebuilt lazily on first access after
        # _update_question fires, and reused thereafter. Reconciler runs every
        # ~15s and previously rebuilt this from O(N legs × M questions) on
        # every cycle; the cache makes that O(1) on the steady-state path.
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

        ``lookback_seconds`` sizes the (symbol, dt) history deque to hold at
        least ``lookback//dt + 2`` bars; never shrinks below the default maxlen.
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
            hist = self._marks.get(key)
            if hist is not None and hist.maxlen != self._mark_history_by_key[key]:
                self._marks[key] = deque(hist, maxlen=self._mark_history_by_key[key])

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
                    self._ingest_reference_price(
                        ev.symbol, mid, ev.exchange_ts or ev.local_recv_ts,
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
                dq = self._trades.setdefault(ev.symbol, deque())
                dq.append(ev)
                self._evict_old_trades(dq, now=ev.exchange_ts or ev.local_recv_ts)
                b = self._books.setdefault(ev.symbol, _MutableBook())
                b.last_trade_ts_ns = max(b.last_trade_ts_ns, ev.exchange_ts or ev.local_recv_ts)
            case MarkEvent():
                # bbo-sourced symbols take their σ/OHLC reference from the BBO
                # mid (handled in the BboEvent case); a stray MarkEvent for such
                # a symbol must NOT touch the reference price or bars.
                if self._reference_source_by_symbol.get(ev.symbol) != "bbo":
                    self._ingest_reference_price(
                        ev.symbol, ev.mark_px, ev.exchange_ts or ev.local_recv_ts,
                    )
            case QuestionMetaEvent():
                self._update_question(ev)
            case SettlementEvent():
                self._mark_settled(ev)
            case _:
                pass  # other events ignored in Phase 1

    def _ingest_reference_price(self, symbol: str, price: float, ts: int) -> None:
        """Feed one reference-price tick into ``last_mark`` plus the per-cadence
        OHLC bars.

        Shared by the mark-sourced (MarkEvent) and bbo-sourced (BboEvent mid)
        paths so both produce identical per-bucket ``(high, low, close)`` bars.
        ``last_mark`` tracks the absolute latest tick (unbucketed). One shared
        tick stream fans into every cadence registered for the symbol; within a
        bucket the bar updates in place (high=max, low=min, close=last), a new
        bucket appends a fresh (price, price, price). This keeps the strategy's
        ``recent_returns`` spanning the wall-clock window its σ formula assumes
        (not the raw sub-second tick rate) and supplies per-bar H/L for
        Parkinson. An unregistered symbol still gets the legacy single default
        bucket so pre-registration ticks are not dropped (matches old behaviour).
        """
        self._last_mark[symbol] = price
        self._last_mark_ts[symbol] = ts
        # An unregistered symbol still gets the legacy single default bucket so
        # pre-registration ticks are not dropped (matches old behaviour).
        cadences = self._cadences_by_symbol.get(symbol) or self._default_cadences
        for bucket_ns in cadences:
            key = (symbol, bucket_ns)
            hist = self._marks.get(key)
            if hist is None:
                maxlen = self._mark_history_by_key.get(key, self._mark_history)
                hist = deque(maxlen=maxlen)
                self._marks[key] = hist
            bucket = bucket_index(ts, bucket_ns)
            last_bucket = self._last_mark_bucket.get(key)
            if last_bucket is None or bucket != last_bucket or not hist:
                # New bucket: store (ts, high, low, close) so the time-bounded
                # read paths can slice by wall-clock (SHR-66).
                hist.append((ts, price, price, price))
                self._last_mark_bucket[key] = bucket
            else:
                # Same merge rule as the batch ``resample_ohlc`` loaders use: a
                # scalar tick is a degenerate (price, price, price) bar.
                # The stored ts is the latest tick in the bucket, matching
                # ``resample_ohlc`` semantics (bar ts = last sample in window).
                new_hlc = update_bar((hist[-1][1], hist[-1][2], hist[-1][3]), price, price, price)
                hist[-1] = (ts, new_hlc[0], new_hlc[1], new_hlc[2])

    def _evict_old_trades(self, dq: "deque[TradeEvent]", *, now: int) -> None:
        cutoff = now - self._volume_window_ns
        while dq and (dq[0].exchange_ts or dq[0].local_recv_ts) < cutoff:
            dq.popleft()

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
        return self._last_mark.get(symbol)

    def last_mark_ts(self, symbol: str) -> int | None:
        """Timestamp (ns) of the latest reference tick for ``symbol``, or None
        if none seen yet. Used by the Scanner to compute reference-feed age for
        the risk gate's stale-reference check (SHR-60)."""
        return self._last_mark_ts.get(symbol)

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
        sequence is TIME-bounded: only bars with ``ts >= now_ns -
        lookback_seconds * 1e9`` are kept BEFORE computing returns — matching
        the backtest's ``slice_window(now_ns, lookback_seconds)`` semantics
        (SHR-66). Both return endpoints must be inside the time window (the
        first return in the slice is skipped because its left endpoint is the
        boundary bar itself, which is inside the window — same rule as
        ``KlineRingBuffer.slice_window`` using ``ret_slice = ret[lo+1:hi]``).

        Without ``now_ns`` / ``lookback_seconds`` the legacy COUNT path is
        used (``[-(n+1):]``), preserving bit-identity for callers that have
        not been updated yet."""
        key = (symbol, self._resolve_dt_ns(symbol, dt))
        hist = self._marks.get(key)
        if hist is None or len(hist) < 2:
            return ()
        if now_ns is not None and lookback_seconds is not None:
            # TIME-bounded path: converge on backtest slice_window semantics.
            cutoff_ns = now_ns - lookback_seconds * 1_000_000_000
            # Each bar is (ts, high, low, close); keep bars in [cutoff, now].
            bars = [b for b in hist if b[0] >= cutoff_ns]
            rets: list[float] = []
            for prev, curr in zip(bars, bars[1:], strict=False):
                prev_c, curr_c = prev[3], curr[3]
                if prev_c > 0 and curr_c > 0:
                    rets.append(math.log(curr_c / prev_c))
            return tuple(rets)
        else:
            # Legacy COUNT path: keep for callers that don't pass time args.
            bars = list(hist)[-(n + 1):]
            rets = []
            for prev, curr in zip(bars, bars[1:], strict=False):
                prev_c, curr_c = prev[3], curr[3]
                if prev_c > 0 and curr_c > 0:
                    rets.append(math.log(curr_c / prev_c))
            return tuple(rets)

    def recent_hl_bars(
        self,
        symbol: str,
        n: int,
        dt: int | None = None,
        *,
        now_ns: int | None = None,
        lookback_seconds: int | None = None,
    ) -> tuple[tuple[float, float], ...]:
        """Last ``n`` per-bucket ``(high, low)`` bars for ``symbol`` at cadence ``dt``
        (default = the symbol's first registered cadence).

        Mirrors the backtest MarketState's ``recent_hl_bars`` (KlineRingBuffer
        rows ``[high, low]``) so the strategy's Parkinson σ estimator receives
        the same input shape on the live path as in the backtest. Empty tuple
        when the symbol has no bars yet.

        When ``now_ns`` and ``lookback_seconds`` are provided, only bars with
        ``ts >= now_ns - lookback_seconds * 1e9`` are returned — matching the
        backtest's time-bounded slice (SHR-66). Without them, the legacy COUNT
        path (``[-n:]``) is used."""
        key = (symbol, self._resolve_dt_ns(symbol, dt))
        hist = self._marks.get(key)
        if not hist:
            return ()
        if now_ns is not None and lookback_seconds is not None:
            # TIME-bounded: keep bars whose ts is inside the window.
            cutoff_ns = now_ns - lookback_seconds * 1_000_000_000
            # bar layout: (ts, high, low, close)
            return tuple(
                (b[1], b[2]) for b in hist
                if b[0] >= cutoff_ns and b[1] > 0 and b[2] > 0
            )
        else:
            # Legacy COUNT path.
            bars = list(hist)[-n:]
            return tuple((b[1], b[2]) for b in bars)

    def recent_volume_usd(self, symbol: str, *, now: int) -> float:
        dq = self._trades.get(symbol)
        if dq is None:
            return 0.0
        self._evict_old_trades(dq, now=now)
        return float(sum(t.price * t.size for t in dq))
