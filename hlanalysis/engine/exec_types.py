"""Venue-neutral execution-layer types. Shared by HLClient, PMClient, and
any future ExecutionClient implementations. Lifted out of hl_client.py
during the v3.1-PM refactor so the engine wiring can carry one Protocol
type instead of a concrete HL class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class PlaceRequest:
    cloid: str
    symbol: str
    side: Literal["buy", "sell"]
    size: float
    price: float
    reduce_only: bool
    time_in_force: Literal["ioc", "gtc"]


@dataclass(frozen=True, slots=True)
class OrderAck:
    cloid: str
    venue_oid: str
    status: Literal["pending", "open", "filled", "rejected"]
    fill_price: float | None = None
    fill_size: float | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class VenuePosition:
    symbol: str
    qty: float
    avg_entry: float
    unrealized_pnl: float


@dataclass(frozen=True, slots=True)
class ClearinghouseState:
    positions: tuple[VenuePosition, ...]
    account_value_usd: float
    # False when the venue position set could NOT be fetched (e.g. PM data-api
    # error). The reconciler must then SKIP position reconciliation rather than
    # treat the empty set as truth and vanish-delete every live position.
    # Always True for HL (its clearinghouse fetch either succeeds or raises).
    positions_known: bool = True


@dataclass(frozen=True, slots=True)
class OpenOrderRow:
    cloid: str
    venue_oid: str
    symbol: str
    side: Literal["buy", "sell"]
    price: float
    size: float
    placed_ts_ns: int


@dataclass(frozen=True, slots=True)
class UserFillRow:
    fill_id: str
    cloid: str
    symbol: str
    side: Literal["buy", "sell"]
    price: float
    size: float
    fee: float
    ts_ns: int
    # HL-reported realized PnL on this fill (0 for opens, signed for reduces).
    # Sourced from the `closedPnl` field on /info user_fills. The daily-loss
    # gate sums (closed_pnl - fee) across today's fills instead of trying to
    # reconstruct PnL from the local DB, which loses information whenever a
    # position closes (the row is deleted, taking its realized_pnl with it).
    closed_pnl: float = 0.0
