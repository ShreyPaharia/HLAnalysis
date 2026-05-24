"""Polymarket execution client. Implements ExecutionClient.

This is the paper-mode-only stub in Phase 5. Live `py-clob-client-v2`
wiring lands in Phase 8 — the SDK import is intentionally deferred so
the engine boots cleanly even without the dependency installed and so
construction in `paper_mode=False` does not hit the network.
"""
from __future__ import annotations

from .exec_types import (
    ClearinghouseState,
    OpenOrderRow,
    OrderAck,
    PlaceRequest,
    UserFillRow,
)


class PMClient:
    paper_mode: bool

    def __init__(
        self,
        *,
        paper_mode: bool,
        clob_host: str | None = None,
        chain_id: int = 137,
        private_key: str | None = None,
        clob_api_key: str | None = None,
        clob_api_secret: str | None = None,
        clob_api_passphrase: str | None = None,
    ) -> None:
        self.paper_mode = paper_mode
        self._cfg = dict(
            clob_host=clob_host,
            chain_id=chain_id,
            private_key=private_key,
            clob_api_key=clob_api_key,
            clob_api_secret=clob_api_secret,
            clob_api_passphrase=clob_api_passphrase,
        )
        # py-clob-client-v2 wiring lands in Phase 8. We do NOT import the SDK
        # here — even in live mode — so a misconfigured environment fails
        # loudly at first order rather than at process startup.
        self._live = None

    def place(self, req: PlaceRequest) -> OrderAck:
        if not self.paper_mode:
            raise NotImplementedError(
                "PMClient live mode lands in Phase 8 of the v3.1-PM plan."
            )
        # Paper-mode fill semantics arrive in Phase 6; the stub returns a
        # rejected ack so any accidental live wiring is visible immediately.
        return OrderAck(
            cloid=req.cloid,
            venue_oid=f"pm-paper-stub-{req.cloid}",
            status="rejected",
            error="pm_client_paper_stub_phase5",
        )

    def cancel(self, *, cloid: str, symbol: str) -> bool:
        if not self.paper_mode:
            raise NotImplementedError(
                "PMClient live mode lands in Phase 8 of the v3.1-PM plan."
            )
        return False

    def open_orders(self) -> list[OpenOrderRow]:
        return []

    def clearinghouse_state(self) -> ClearinghouseState:
        return ClearinghouseState(positions=(), account_value_usd=0.0)

    def user_fills(self, *, since_ts_ns: int = 0) -> list[UserFillRow]:
        return []

    def realized_pnl_since(self, since_ts_ns: int) -> float:
        return 0.0
