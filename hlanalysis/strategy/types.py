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
    # Strategy-supplied exit reason. Plumbed to Exit.reason on close so Telegram
    # alerts distinguish safety_d / edge / time_stop / true stop_loss exits.
    # Empty on entries; router falls back to legacy "stop_loss" if reduce_only
    # but no reason was supplied (for older strategies).
    exit_reason: str = ""


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
    # Top-N L2 levels (price, size) for entry-leg slippage estimation.
    # Empty tuple disables the depth-walk gate (legacy HL BboEvent path);
    # MarketState fills these from BookSnapshotEvent on PM/HL.
    ask_levels: tuple[tuple[float, float], ...] = ()
    bid_levels: tuple[tuple[float, float], ...] = ()


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
    Built by the engine from QuestionMetaEvent + MarketMetaEvent + adapter cache.

    `leg_symbols` is the full set of tradable leg coins for this question,
    in canonical order (priceBinary: [yes, no]; priceBucket: [yes_o0, no_o0,
    yes_o1, no_o1, ...]). `yes_symbol` / `no_symbol` are convenience aliases
    for the first two legs (binaries) and remain empty strings for non-binary
    questions where they have no meaning.
    """
    question_idx: int
    yes_symbol: str       # HL leg symbol for the YES side (binary only)
    no_symbol: str        # HL leg symbol for the NO side (binary only)
    strike: float
    expiry_ns: int        # nanoseconds since epoch
    underlying: str       # e.g. "BTC"
    klass: str            # e.g. "priceBinary" or "priceBucket"
    period: str           # e.g. "1d"
    settled: bool = False
    settled_side: Literal["yes", "no", "unknown"] | None = None
    # Symbol of the winning leg, stamped by MarketState._mark_settled from
    # SettlementEvent.symbol. Needed to compute realized PnL on
    # held-to-settlement positions for multi-outcome buckets where
    # settled_side alone (yes/no) doesn't identify which outcome won.
    settled_symbol: str = ""
    # For multi-leg (priceBucket) questions every leg-pair resolves at once;
    # each leg's SettlementEvent names its pair's winning token. We accumulate
    # all winners here so a held leg's win/loss is decided by membership, not by
    # whichever leg's event happened to land last (which clobbers settled_symbol).
    settled_symbols: tuple[str, ...] = ()
    leg_symbols: tuple[str, ...] = ()
    # Display fields from QuestionMetaEvent, used by alerts/reports for human
    # readable rendering. `name` is HL's question name (e.g. "Recurring").
    # `kv` mirrors QuestionMetaEvent.keys/values pairs (description fields like
    # priceThresholds, targetPrice, expiry, etc.).
    name: str = ""
    kv: tuple[tuple[str, str], ...] = ()
    # Originating venue ("hyperliquid" / "polymarket"). Lets strategy slots be
    # scoped to a single venue so a PM slot never matches an HL question (and
    # vice-versa) — both share class/underlying but resolve to different books
    # and order token namespaces.
    venue: str = ""


__all__ = [
    "Action",
    "BookState",
    "Decision",
    "Diagnostic",
    "OrderIntent",
    "Position",
    "QuestionView",
]
