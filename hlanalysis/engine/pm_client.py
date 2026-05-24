"""Polymarket execution client. Implements ExecutionClient.

Paper mode here is the same shape as HLClient.paper — synthesized
OrderAcks, in-memory positions and fills. The strategy doesn't know it's
paper; the engine wiring decides.

Live mode (paper_mode=False) bootstrap and order I/O land in Phase 8 via
py-clob-client-v2.
"""
from __future__ import annotations

import time
import uuid

from .exec_types import (
    ClearinghouseState,
    OpenOrderRow,
    OrderAck,
    PlaceRequest,
    UserFillRow,
    VenuePosition,
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
        # Paper bookkeeping
        self._paper_acks: dict[str, OrderAck] = {}
        self._paper_open: dict[str, OpenOrderRow] = {}
        self._paper_positions: dict[str, VenuePosition] = {}
        self._paper_fills: list[UserFillRow] = []
        # Live SDK wiring lands in Phase 8. We do NOT import py-clob-client-v2
        # here — even in live mode — so a misconfigured environment fails
        # loudly at first order rather than at process startup.
        self._live = None

    def place(self, req: PlaceRequest) -> OrderAck:
        if self.paper_mode:
            return self._paper_place(req)
        return self._live_place(req)

    def cancel(self, *, cloid: str, symbol: str) -> bool:
        if self.paper_mode:
            return self._paper_open.pop(cloid, None) is not None
        return self._live_cancel(cloid=cloid, symbol=symbol)

    def open_orders(self) -> list[OpenOrderRow]:
        # Live read-only paths return empty until Phase 8 wires in the CLOB
        # client. This keeps engine boot (restart-drift, reconciler) clean
        # for a misconfigured live slot — order I/O still raises on first use.
        if self.paper_mode:
            return list(self._paper_open.values())
        return []

    def clearinghouse_state(self) -> ClearinghouseState:
        if self.paper_mode:
            return ClearinghouseState(
                positions=tuple(self._paper_positions.values()),
                account_value_usd=0.0,
            )
        return ClearinghouseState(positions=(), account_value_usd=0.0)

    def user_fills(self, *, since_ts_ns: int = 0) -> list[UserFillRow]:
        if self.paper_mode:
            return [f for f in self._paper_fills if f.ts_ns >= since_ts_ns]
        return []

    def realized_pnl_since(self, since_ts_ns: int) -> float:
        fills = self.user_fills(since_ts_ns=since_ts_ns)
        return sum(f.closed_pnl - f.fee for f in fills)

    # ---- paper internals ----

    def _paper_place(self, req: PlaceRequest) -> OrderAck:
        # Idempotency: same cloid → same ack, no double-fill.
        if req.cloid in self._paper_acks:
            return self._paper_acks[req.cloid]
        if req.price <= 0:
            ack = OrderAck(
                cloid=req.cloid, venue_oid=f"paper-{req.cloid}",
                status="rejected", error="non_marketable_price",
            )
            self._paper_acks[req.cloid] = ack
            return ack
        ack = OrderAck(
            cloid=req.cloid, venue_oid=f"paper-{uuid.uuid4().hex[:16]}",
            status="filled", fill_price=req.price, fill_size=req.size,
        )
        self._paper_acks[req.cloid] = ack
        # Position bookkeeping — same shape as HLClient._paper_place.
        signed = req.size if req.side == "buy" else -req.size
        existing = self._paper_positions.get(req.symbol)
        if existing is None:
            self._paper_positions[req.symbol] = VenuePosition(
                symbol=req.symbol, qty=signed, avg_entry=req.price,
                unrealized_pnl=0.0,
            )
        else:
            tot = existing.qty + signed
            if abs(tot) < 1e-9:
                self._paper_positions.pop(req.symbol, None)
            else:
                avg = (
                    (existing.qty * existing.avg_entry + signed * req.price) / tot
                )
                self._paper_positions[req.symbol] = VenuePosition(
                    symbol=req.symbol, qty=tot, avg_entry=avg, unrealized_pnl=0.0,
                )
        ts = time.time_ns()
        self._paper_fills.append(UserFillRow(
            fill_id=f"f-{req.cloid}-{ts}", cloid=req.cloid, symbol=req.symbol,
            side=req.side, price=req.price, size=req.size, fee=0.0, ts_ns=ts,
        ))
        return ack

    # ---- live stubs (filled in Phase 8) ----

    def _live_place(self, req: PlaceRequest) -> OrderAck:
        raise NotImplementedError("Phase 8")

    def _live_cancel(self, *, cloid: str, symbol: str) -> bool:
        raise NotImplementedError("Phase 8")
