"""Market event dataclasses. Local mirror of §3.1 from
docs/specs/2026-05-11-backtester-rebuild.md until Task A's PR merges.

The §3 contract is frozen: tasks B/C/D mirror these definitions locally and
Task E (integration) drops the mirrors and re-targets the real imports.

Additive vs §3.1: `SettlementEvent.symbol: str = ""` is added (default empty)
so multi-outcome (bucket) markets can key per-leg outcomes to the leg symbol,
as required by §3.4. This is the only deviation; the field is optional so
Task A's mirror remains forward-compatible.
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
