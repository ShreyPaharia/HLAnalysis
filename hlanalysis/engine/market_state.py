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
        # Per reference symbol, a deque of per-bucket OHLC bars stored as
        # ``(high, low, close)``. Close-to-close drives ``recent_returns`` (the
        # legacy stdev path, bit-identical); high/low drive ``recent_hl_bars``
        # which activates the strategy's Parkinson σ estimator. Open is not
        # retained (no consumer needs it). Within a bucket the bar is updated
        # in place (high=max, low=min, close=last); a new bucket appends a
        # fresh bar. Maxlen is sized per symbol via ``set_reference_cadence``.
        self._marks: dict[str, deque[tuple[float, float, float]]] = {}
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
        self._mark_bucket_ns = mark_bucket_ns
        # Per-symbol overrides of the bucket period, registered by the live
        # runtime via ``set_reference_cadence`` so each strategy slot's σ
        # formula (which assumes returns spaced ``vol_sampling_dt_seconds``
        # apart) sees a series bucketed at exactly that cadence. A single
        # MarketState is shared across all slots, but each slot reads its own
        # ``reference_symbol`` (HL=BTC, PM=BTCUSDT_SPOT), so per-symbol bucketing
        # gives HL/PM independence. Symbols with no override fall back to
        # ``_mark_bucket_ns`` (60s), preserving legacy behaviour. See
        # `engine_sigma_sampling_bug_2026_05_21.md` and the v3.7 cadence port.
        self._mark_bucket_ns_by_symbol: dict[str, int] = {}
        # Per-symbol deque maxlen overrides. At sub-minute cadences the default
        # 256-entry history is too short to cover ``vol_lookback_seconds`` of
        # returns (e.g. dt=5s / 3600s lookback needs ~720 bars), which would
        # silently truncate the σ window relative to the backtest (whose ring
        # buffer auto-grows). Registered alongside the bucket period.
        self._mark_history_by_symbol: dict[str, int] = {}
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
        # Last-bucket-id we wrote to ``_marks`` per symbol; used to coalesce
        # within-bucket reference-price updates into one bar.
        self._last_mark_bucket: dict[str, int] = {}
        self._volume_window_ns = volume_window_ns
        self._mark_history = mark_history
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
        """Register the mark-bucketing cadence for ``symbol``.

        Couples MarketState's per-symbol bucket period to the consuming
        strategy's ``vol_sampling_dt_seconds`` so there is no train/serve skew:
        the period the strategy's σ formula assumes equals the period the marks
        are bucketed at.

        Because a single MarketState is shared across all slots, two slots that
        read the *same* ``reference_symbol`` must agree on the cadence — the one
        mark history for that symbol can only be bucketed one way. Conflicting
        registrations raise, failing fast at startup rather than silently
        skewing whichever slot loses. (On HL today both v1 and v31 read "BTC",
        so flipping one to dt=5 requires the other to move in lockstep.)

        ``lookback_seconds`` (when given) sizes the per-symbol history deque so
        it can hold ``ceil(lookback/dt)`` returns; never shrinks below the
        default ``mark_history``.
        """
        ns = int(sampling_dt_seconds) * 1_000_000_000
        if ns <= 0:
            raise ValueError(
                f"sampling_dt_seconds must be positive, got {sampling_dt_seconds!r}"
            )
        prev = self._mark_bucket_ns_by_symbol.get(symbol)
        if prev is not None and prev != ns:
            raise ValueError(
                f"conflicting mark-bucket cadence for symbol {symbol!r}: "
                f"already registered at {prev // 1_000_000_000}s, refused "
                f"re-registration at {sampling_dt_seconds}s. All strategy slots "
                f"reading the same reference_symbol must share one "
                f"vol_sampling_dt_seconds (the shared mark history can only be "
                f"bucketed one way)."
            )
        self._mark_bucket_ns_by_symbol[symbol] = ns
        if lookback_seconds is not None:
            # +2: recent_returns needs n+1 prices for n returns, plus a margin
            # so the oldest bar isn't evicted before the window is full.
            needed = int(lookback_seconds) // int(sampling_dt_seconds) + 2
            self._mark_history_by_symbol[symbol] = max(
                self._mark_history_by_symbol.get(symbol, self._mark_history),
                needed,
            )

    def mark_bucket_ns_for(self, symbol: str) -> int:
        """The bucket period (ns) applied to ``symbol``'s marks. Returns the
        per-symbol override if registered, else the default. Exposed so the
        runtime/tests can assert no train/serve skew against the strategy's
        configured vol_sampling_dt_seconds."""
        return self._mark_bucket_ns_by_symbol.get(symbol, self._mark_bucket_ns)

    def set_reference_source(self, symbol: str, source: str) -> None:
        """Register which feed sources ``symbol``'s σ/OHLC reference.

        ``"mark"`` (default) reads the venue MarkEvent; ``"bbo"`` reads the
        dense BBO mid. Like ``set_reference_cadence`` this couples a per-symbol
        choice to the shared MarketState, so two slots reading the same
        reference symbol must agree — a conflicting re-registration raises
        rather than silently feeding the shared history two ways.
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
        """Feed one reference-price tick into the per-symbol OHLC bars +
        ``last_mark``.

        Shared by the mark-sourced (MarkEvent) and bbo-sourced (BboEvent mid)
        paths so both produce identical per-bucket ``(high, low, close)`` bars.
        ``last_mark`` tracks the absolute latest tick (unbucketed). Ticks are
        bucketed to ``vol_sampling_dt_seconds``-wide windows (default 60s;
        per-symbol override via ``set_reference_cadence``): within a bucket the
        bar updates in place (high=max, low=min, close=last); a new bucket
        appends a fresh ``(price, price, price)`` bar. This keeps the strategy's
        ``recent_returns`` spanning the wall-clock window its σ formula assumes
        (not the raw sub-second tick rate) and supplies per-bar H/L for
        Parkinson.
        """
        self._last_mark[symbol] = price
        self._last_mark_ts[symbol] = ts
        hist = self._marks.get(symbol)
        if hist is None:
            maxlen = self._mark_history_by_symbol.get(symbol, self._mark_history)
            hist = deque(maxlen=maxlen)
            self._marks[symbol] = hist
        bucket = ts // self._mark_bucket_ns_by_symbol.get(
            symbol, self._mark_bucket_ns,
        )
        last_bucket = self._last_mark_bucket.get(symbol)
        if last_bucket is None or bucket != last_bucket or not hist:
            hist.append((price, price, price))
            self._last_mark_bucket[symbol] = bucket
        else:
            h, l, _c = hist[-1]
            hist[-1] = (h if h >= price else price, l if l <= price else price, price)

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
            self._questions[idx] = dataclasses.replace(
                q, settled=True, settled_side=side, settled_symbol=ev.symbol,
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

    def recent_returns(self, symbol: str, n: int) -> tuple[float, ...]:
        """Last ``n`` close-to-close log returns for ``symbol``'s OHLC bars.

        Close-to-close is the legacy stdev input; bit-identical to the pre-OHLC
        behaviour since the per-bucket close is still the bucket's last tick."""
        hist = self._marks.get(symbol)
        if hist is None or len(hist) < 2:
            return ()
        bars = list(hist)[-(n + 1):]
        rets: list[float] = []
        for prev, curr in zip(bars, bars[1:], strict=False):
            prev_c, curr_c = prev[2], curr[2]
            if prev_c > 0 and curr_c > 0:
                rets.append(math.log(curr_c / prev_c))
        return tuple(rets)

    def recent_hl_bars(self, symbol: str, n: int) -> tuple[tuple[float, float], ...]:
        """Last ``n`` per-bucket ``(high, low)`` bars for ``symbol``.

        Mirrors the backtest MarketState's ``recent_hl_bars`` (KlineRingBuffer
        rows ``[high, low]``) so the strategy's Parkinson σ estimator receives
        the same input shape on the live path as in the backtest. Empty tuple
        when the symbol has no bars yet. Threading this into the Scanner is what
        activates Parkinson for ``vol_estimator: parkinson`` slots live (the
        dormant-Parkinson fix); ``(ln(H/L))²`` is order-invariant so the (high,
        low) row order does not affect σ, but it matches the backtest column
        order for clarity."""
        hist = self._marks.get(symbol)
        if not hist:
            return ()
        bars = list(hist)[-n:]
        return tuple((b[0], b[1]) for b in bars)

    def recent_volume_usd(self, symbol: str, *, now: int) -> float:
        dq = self._trades.get(symbol)
        if dq is None:
            return 0.0
        self._evict_old_trades(dq, now=now)
        return float(sum(t.price * t.size for t in dq))
