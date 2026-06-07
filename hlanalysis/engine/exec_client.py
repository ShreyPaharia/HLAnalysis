from __future__ import annotations

from typing import Protocol, runtime_checkable

from .exec_types import (
    ClearinghouseState,
    OpenOrderRow,
    OrderAck,
    PlaceRequest,
    UserFillRow,
)


@runtime_checkable
class ExecutionClient(Protocol):
    """Venue-neutral order-execution interface.

    Implementations: HLClient (Hyperliquid HIP-4 + perp + spot), PMClient
    (Polymarket CLOB). The Router and Reconciler depend on this Protocol,
    not on any concrete class.
    """

    paper_mode: bool

    # True when the venue reports position settlement as a fill carrying
    # closedPnl (HL HIP-4/bucket: `dir="Settlement"`), so `realized_pnl_since`
    # already includes the settlement payout. False when settlement is an
    # out-of-band redeem (PM CLOB), in which case settlement PnL must be
    # tracked separately. Drives whether the daily-loss gate and the settlement
    # Exit source PnL from the venue or from a re-derivation.
    settlement_reported_as_fill: bool

    def place(self, req: PlaceRequest) -> OrderAck: ...
    def cancel(self, *, cloid: str, symbol: str) -> bool: ...
    def open_orders(self) -> list[OpenOrderRow]: ...
    def clearinghouse_state(self) -> ClearinghouseState: ...
    def user_fills(self, *, since_ts_ns: int = 0) -> list[UserFillRow]: ...
    def realized_pnl_since(self, since_ts_ns: int, *, outcome_only: bool = False) -> float: ...

    def realized_pnl_for_symbol(
        self, symbol: str, *, since_ts_ns: int = 0
    ) -> float:
        """Venue-truth realized PnL for one leg: Σ(closedPnl − fee) over this
        account's fills on `symbol` since the cutoff. On settlement-as-fill
        venues this captures the settlement payout exactly, so a settlement
        Exit can be booked from venue truth rather than a re-derived winner."""
        ...
