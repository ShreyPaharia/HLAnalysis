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
    ) -> None:
        self._books: dict[str, _MutableBook] = {}
        self._questions: dict[int, QuestionView] = {}
        self._trades: dict[str, deque[TradeEvent]] = {}
        self._marks: dict[str, deque[float]] = {}
        self._last_mark: dict[str, float] = {}
        self._volume_window_ns = volume_window_ns
        self._mark_history = mark_history

    # ---- ingest ----

    def apply(self, ev: NormalizedEvent) -> None:
        match ev:
            case BboEvent():
                b = self._books.setdefault(ev.symbol, _MutableBook())
                b.bid_px, b.bid_sz = ev.bid_px, ev.bid_sz
                b.ask_px, b.ask_sz = ev.ask_px, ev.ask_sz
                b.last_l2_ts_ns = max(b.last_l2_ts_ns, ev.exchange_ts or ev.local_recv_ts)
            case BookSnapshotEvent():
                b = self._books.setdefault(ev.symbol, _MutableBook())
                if ev.bid_px:
                    b.bid_px, b.bid_sz = ev.bid_px[0], ev.bid_sz[0]
                if ev.ask_px:
                    b.ask_px, b.ask_sz = ev.ask_px[0], ev.ask_sz[0]
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
                self._last_mark[ev.symbol] = ev.mark_px
                hist = self._marks.setdefault(ev.symbol, deque(maxlen=self._mark_history))
                hist.append(ev.mark_px)
            case QuestionMetaEvent():
                self._update_question(ev)
            case SettlementEvent():
                self._mark_settled(ev)
            case _:
                pass  # other events ignored in Phase 1

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
        expiry_ns = self._parse_expiry_ns(kv.get("expiry", ""))
        # HL HIP-4 emits L2/trades keyed by f"#{10*outcome_idx + side_idx}" where
        # side_idx=0 is YES and side_idx=1 is NO (see adapters/hyperliquid.py).
        # priceBinary: 1 outcome × 2 sides → 2 legs ([yes, no]).
        # priceBucket: N outcomes × 2 sides → 2N legs interleaved.
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
        )

    def _mark_settled(self, ev: SettlementEvent) -> None:
        # SettlementEvent.symbol is one leg; flip the matching question's settled flag.
        for idx, q in list(self._questions.items()):
            if ev.symbol in (q.yes_symbol, q.no_symbol):
                side = "yes" if ev.symbol == q.yes_symbol else "no"
                # Use dataclasses.replace since QuestionView is a frozen dataclass
                self._questions[idx] = dataclasses.replace(q, settled=True, settled_side=side)

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
        )

    def question(self, idx: int) -> QuestionView | None:
        return self._questions.get(idx)

    def all_questions(self) -> list[QuestionView]:
        return list(self._questions.values())

    def last_mark(self, symbol: str) -> float | None:
        return self._last_mark.get(symbol)

    def recent_returns(self, symbol: str, n: int) -> tuple[float, ...]:
        hist = self._marks.get(symbol)
        if hist is None or len(hist) < 2:
            return ()
        prices = list(hist)[-(n + 1):]
        rets: list[float] = []
        for prev, curr in zip(prices, prices[1:], strict=False):
            if prev > 0 and curr > 0:
                rets.append(math.log(curr / prev))
        return tuple(rets)

    def recent_volume_usd(self, symbol: str, *, now: int) -> float:
        dq = self._trades.get(symbol)
        if dq is None:
            return 0.0
        self._evict_old_trades(dq, now=now)
        return float(sum(t.price * t.size for t in dq))
