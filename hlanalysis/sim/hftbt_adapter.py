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
) -> Iterator[SimEvent]:
    """Chronological merge of L2 snapshots (from each trade), trade-ts notifications,
    and BTC kline closes.
    """
    iters = []
    iters.append(iter(SimEvent(t.ts_ns, EventKind.L2, trade_to_l2(t, half_spread=half_spread, depth=depth)) for t in trades))
    iters.append(iter(SimEvent(t.ts_ns, EventKind.TRADE_TS, t) for t in trades))
    iters.append(iter(SimEvent(k.ts_ns, EventKind.KLINE, k) for k in klines))
    yield from heapq.merge(*iters, key=lambda e: e.ts_ns)
