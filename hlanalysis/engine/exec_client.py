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

    def place(self, req: PlaceRequest) -> OrderAck: ...
    def cancel(self, *, cloid: str, symbol: str) -> bool: ...
    def open_orders(self) -> list[OpenOrderRow]: ...
    def clearinghouse_state(self) -> ClearinghouseState: ...
    def user_fills(self, *, since_ts_ns: int = 0) -> list[UserFillRow]: ...
    def realized_pnl_since(self, since_ts_ns: int) -> float: ...
