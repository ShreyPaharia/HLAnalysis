"""Market event dataclasses (spec §3.1).

Additive vs the original §3.1: ``SettlementEvent.symbol: str = ""`` is added
(default empty) so multi-outcome (bucket) markets can key per-leg outcomes to
the leg symbol, as required by §3.4. Binary sources may leave it as the empty
string; the runner falls back to ``q.leg_symbols`` ordering in that case.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    ts_ns: int
    symbol: str
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class TradeEvent:
    ts_ns: int
    symbol: str
    side: Literal["buy", "sell"]
    price: float
    size: float


@dataclass(frozen=True, slots=True)
class ReferenceEvent:
    ts_ns: int
    symbol: str
    high: float
    low: float
    close: float


@dataclass(frozen=True, slots=True)
class SettlementEvent:
    ts_ns: int
    question_idx: int
    outcome: Literal["yes", "no", "unknown"]
    symbol: str = ""


MarketEvent = Union[BookSnapshot, TradeEvent, ReferenceEvent, SettlementEvent]


__all__ = [
    "BookSnapshot",
    "TradeEvent",
    "ReferenceEvent",
    "SettlementEvent",
    "MarketEvent",
]
