# hlanalysis/sim/hftbt_adapter.py
"""Chronological event stream over PM trades + Binance klines.

Named `hftbt_adapter` because the long-term plan (umbrella spec §9 Phase 2,
this spec §5.3) is to drive the runner via `hftbacktest`. For taker-IOC against
synthetic L2 derived from trades, the queue-position machinery in hftbacktest
buys us nothing today — the merge below is functionally what we'd hand
hftbacktest anyway. The dep stays in pyproject.toml for the HL HIP-4 swap;
this module is the seam where that swap will land.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

from .data.binance_klines import Kline
from .data.schemas import PMTrade
from .synthetic_l2 import L2Snapshot, trade_to_l2


class EventKind(str, Enum):
    L2 = "l2"
    TRADE_TS = "trade_ts"
    KLINE = "kline"


@dataclass(frozen=True, slots=True)
class SimEvent:
    ts_ns: int
    kind: EventKind
    payload: object


def build_event_stream(
    *,
    trades: list[PMTrade],
    klines: list[Kline],
    half_spread: float,
    depth: float,
    yes_token_id: str | None = None,
    no_token_id: str | None = None,
) -> Iterator[SimEvent]:
    """Chronological merge of L2 snapshots, trade-ts notifications, and BTC klines.

    Binary parity: when a trade lands on one leg, infer the complementary leg's
    price as `1 - p` and emit an L2 for it too. PM CLOB orders YES/NO trades
    independently, so without this inference the complementary leg's book stays
    empty during runs of one-sided trading. Strategies depending on both legs'
    quotes (e.g. v2 model-edge) would otherwise see ~no_book on most ticks.

    Pass yes_token_id and no_token_id to enable parity inference.
    """
    # PM CLOB returns trades newest-first. Sort ascending so heapq.merge can
    # interleave them with klines correctly.
    trades = sorted(trades, key=lambda t: t.ts_ns)
    klines = sorted(klines, key=lambda k: k.ts_ns)
    l2_events: list[SimEvent] = []
    for t in trades:
        l2_events.append(SimEvent(t.ts_ns, EventKind.L2, trade_to_l2(t, half_spread=half_spread, depth=depth)))
        if yes_token_id and no_token_id:
            other_id = no_token_id if t.token_id == yes_token_id else (
                yes_token_id if t.token_id == no_token_id else None
            )
            if other_id is not None:
                from .data.schemas import PMTrade as _PMTrade
                comp = _PMTrade(
                    ts_ns=t.ts_ns, token_id=other_id, side=t.side,
                    price=max(1e-6, min(1 - 1e-6, 1.0 - t.price)),
                    size=t.size,
                )
                l2_events.append(SimEvent(t.ts_ns, EventKind.L2, trade_to_l2(comp, half_spread=half_spread, depth=depth)))
    iters = [
        iter(l2_events),
        iter(SimEvent(t.ts_ns, EventKind.TRADE_TS, t) for t in trades),
        iter(SimEvent(k.ts_ns, EventKind.KLINE, k) for k in klines),
    ]
    yield from heapq.merge(*iters, key=lambda e: e.ts_ns)
