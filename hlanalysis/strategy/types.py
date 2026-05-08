from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class Action(str, Enum):
    HOLD = "hold"
    ENTER = "enter"
    EXIT = "exit"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """Structured signal emitted by the strategy. The runner logs / metrics these.
    Kept dependency-free so strategy/ can stay pure."""
    level: Literal["info", "warn", "debug"]
    message: str
    fields: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class OrderIntent:
    question_idx: int
    symbol: str  # HL market symbol of the leg (e.g. "@30")
    side: Literal["buy", "sell"]
    size: float  # always positive; direction is in `side`
    limit_price: float  # IOC marketable limit
    cloid: str  # client order id, "hla-..." prefix
    time_in_force: Literal["ioc", "gtc"] = "ioc"
    reduce_only: bool = False


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    intents: tuple[OrderIntent, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class BookState:
    """Snapshot of one market's L2 + last trade context as the strategy sees it."""
    symbol: str
    bid_px: float | None
    bid_sz: float | None
    ask_px: float | None
    ask_sz: float | None
    last_trade_ts_ns: int  # 0 if no trade yet
    last_l2_ts_ns: int     # 0 if no l2 yet


@dataclass(frozen=True, slots=True)
class Position:
    question_idx: int
    symbol: str
    qty: float            # signed
    avg_entry: float
    stop_loss_price: float
    last_update_ts_ns: int


@dataclass(frozen=True, slots=True)
class QuestionView:
    """What the strategy knows about a HIP-4 question at evaluation time.
    Built by the engine from QuestionMetaEvent + MarketMetaEvent + adapter cache."""
    question_idx: int
    yes_symbol: str       # HL leg symbol for the YES side
    no_symbol: str        # HL leg symbol for the NO side
    strike: float
    expiry_ns: int        # nanoseconds since epoch
    underlying: str       # e.g. "BTC"
    klass: str            # e.g. "priceBinary"
    period: str           # e.g. "1h"
    settled: bool = False
    settled_side: Literal["yes", "no", "unknown"] | None = None


__all__ = [
    "Action",
    "BookState",
    "Decision",
    "Diagnostic",
    "OrderIntent",
    "Position",
    "QuestionView",
]
