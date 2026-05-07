from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class ProductType(str, Enum):
    PERP = "perp"
    SPOT = "spot"
    PREDICTION_BINARY = "prediction_binary"
    PREDICTION_CATEGORICAL = "prediction_categorical"


class Mechanism(str, Enum):
    CLOB = "clob"
    RFQ = "rfq"


class EventType(str, Enum):
    BOOK_SNAPSHOT = "book_snapshot"
    BOOK_DELTA = "book_delta"
    BBO = "bbo"
    TRADE = "trade"
    FUNDING = "funding"
    MARK = "mark"
    ORACLE = "oracle"
    OPEN_INTEREST = "open_interest"
    LIQUIDATION = "liquidation"
    MARKET_META = "market_meta"
    QUESTION_META = "question_meta"
    QUOTE_REQUEST = "quote_request"
    QUOTE = "quote"
    AUCTION_OPEN = "auction_open"
    AUCTION_CLOSE = "auction_close"
    FILL = "fill"
    DEALER_STATE = "dealer_state"
    HEALTH = "health"
    SETTLEMENT = "settlement"


class _BaseEvent(BaseModel):
    model_config = ConfigDict(use_enum_values=True, frozen=True)

    venue: str
    product_type: ProductType
    mechanism: Mechanism
    symbol: str
    # exchange_ts: nanoseconds since epoch, best-effort from the venue. Per-venue caveats:
    #   - binance spot @bookTicker / partial-depth: no ts in payload → recorded as 0.
    #   - binance perp @bookTicker / @depth* / @trade / @aggTrade: msg["T"] (trade/event ms).
    #   - hyperliquid trade: data[i]["time"] = on-chain BLOCK time (NOT ws transport time).
    #     HL's ws API does not expose a message-level send timestamp, and HL ships
    #     trades in batches some seconds after block finalization. For HL trades, use
    #     `block_ts` for explicit block semantics; `exchange_ts` is set to the same value.
    #   - hyperliquid bbo / l2Book: data["time"] = block time at the BBO/book change.
    exchange_ts: int
    local_recv_ts: int  # nanoseconds since epoch, captured from time.time_ns() on this host
    seq: int | None = None


class TradeEvent(_BaseEvent):
    event_type: Literal["trade"] = "trade"
    price: float
    size: float
    side: Literal["buy", "sell", "unknown"]  # taker side
    trade_id: str | None = None
    # On-chain block timestamp (ns since epoch). Currently set only for hyperliquid
    # trades, where HL's ws `trades` channel timestamps each fill with the block-finality
    # time of the matching block, not the time the message was sent over the wire. Other
    # venues leave this null. Keep `exchange_ts` for backward-compat (it's populated to
    # the same value on HL); use `block_ts` when you need to be explicit that this is
    # block time vs transport time.
    block_ts: int | None = None
    # HL-only: WsTrade.users = [buyer_address, seller_address]. Other venues null.
    # Used for wallet-attribution / adverse-selection analysis (HIP-4 strategy P4).
    buyer: str | None = None
    seller: str | None = None
    # HL-only: WsTrade.hash = L1 tx hash of the matching block. Useful for dedup
    # across reconnects, since HL re-publishes recent trades after stream re-attach.
    block_hash: str | None = None


class BookSnapshotEvent(_BaseEvent):
    event_type: Literal["book_snapshot"] = "book_snapshot"
    bid_px: list[float]
    bid_sz: list[float]
    ask_px: list[float]
    ask_sz: list[float]


class BookDeltaEvent(_BaseEvent):
    event_type: Literal["book_delta"] = "book_delta"
    side: Literal["bid", "ask"]
    price: float
    size: float  # 0 means level removed


class BboEvent(_BaseEvent):
    event_type: Literal["bbo"] = "bbo"
    bid_px: float
    bid_sz: float
    ask_px: float
    ask_sz: float


class FundingEvent(_BaseEvent):
    event_type: Literal["funding"] = "funding"
    funding_rate: float
    premium: float | None = None
    next_funding_ts: int | None = None  # ns


class MarkEvent(_BaseEvent):
    event_type: Literal["mark"] = "mark"
    mark_px: float


class OracleEvent(_BaseEvent):
    event_type: Literal["oracle"] = "oracle"
    oracle_px: float


class OpenInterestEvent(_BaseEvent):
    event_type: Literal["open_interest"] = "open_interest"
    open_interest: float
    day_ntl_vlm: float | None = None
    prev_day_px: float | None = None
    mid_px: float | None = None  # ctx-reported mid; not the bbo mid


class LiquidationEvent(_BaseEvent):
    event_type: Literal["liquidation"] = "liquidation"
    price: float
    size: float
    side: Literal["long", "short"]


class MarketMetaEvent(_BaseEvent):
    event_type: Literal["market_meta"] = "market_meta"
    # Free-form metadata as parallel arrays so we keep parquet-friendly.
    # For prediction_binary: keys like "strike", "expiry_ns", "settlement_source", "outcome".
    keys: list[str]
    values: list[str]


class QuestionMetaEvent(_BaseEvent):
    event_type: Literal["question_meta"] = "question_meta"
    # Question identifier from outcomeMeta. Stable per recurring question class.
    question_idx: int
    # Sibling outcomes (e.g., the 3 buckets of a priceBucket question).
    named_outcome_idxs: list[int]
    # Catch-all bucket if the truth doesn't match any named outcome.
    fallback_outcome_idx: int | None = None
    # Question-level settlement: outcome_idx values (joinable to MarketMetaEvent.outcome_idx).
    # Empty before settlement; one element after. We re-emit this event when this changes.
    settled_named_outcome_idxs: list[int] = Field(default_factory=list)
    # Free-form parsed description (class, underlying, expiry, priceThresholds, period).
    # Same parallel-arrays shape as MarketMetaEvent.
    keys: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)


class SettlementEvent(_BaseEvent):
    event_type: Literal["settlement"] = "settlement"
    settled_side_idx: int
    settle_price: float | None = None
    settle_ts: int
    keys: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)


class HealthEvent(_BaseEvent):
    event_type: Literal["health"] = "health"
    kind: str  # "stall", "gap", "reconnect", "subscribed", ...
    detail: str = ""


# RFQ-shaped events kept here so the schema is venue-agnostic from day one,
# even though no current adapter emits them.
class QuoteRequestEvent(_BaseEvent):
    event_type: Literal["quote_request"] = "quote_request"
    rfq_id: str
    side: Literal["buy", "sell", "two_way"]
    size: float


class QuoteEvent(_BaseEvent):
    event_type: Literal["quote"] = "quote"
    rfq_id: str
    bid_px: float | None = None
    ask_px: float | None = None
    bid_sz: float | None = None
    ask_sz: float | None = None
    dealer: str | None = None


NormalizedEvent = Annotated[
    Union[
        TradeEvent,
        BookSnapshotEvent,
        BookDeltaEvent,
        BboEvent,
        FundingEvent,
        MarkEvent,
        OracleEvent,
        OpenInterestEvent,
        LiquidationEvent,
        MarketMetaEvent,
        QuestionMetaEvent,
        SettlementEvent,
        HealthEvent,
        QuoteRequestEvent,
        QuoteEvent,
    ],
    Field(discriminator="event_type"),
]
