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


MarketEvent = Union[BookSnapshot, TradeEvent, ReferenceEvent, SettlementEvent]


__all__ = [
    "BookSnapshot",
    "TradeEvent",
    "ReferenceEvent",
    "SettlementEvent",
    "MarketEvent",
]
