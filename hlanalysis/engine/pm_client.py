"""Polymarket execution client. Implements ExecutionClient.

Paper mode synthesizes OrderAcks, positions, and fills in-memory — same
shape as HLClient.paper. Live mode wraps `py-clob-client-v2`: order
placement uses `OrderType.FAK` for `time_in_force="ioc"` and
`OrderType.GTC` for `"gtc"`. The SDK is imported and constructed lazily
on first live call so paper-only tests don't need the dep installed.
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
        # Live wiring. SDK is lazily constructed on first live call so paper
        # tests and stub-credential constructions don't touch py-clob-client-v2.
        # Tests inject a fake by setting `self._sdk` directly before any call.
        self._sdk = None
        self._cloid_to_oid: dict[str, str] = {}

    def place(self, req: PlaceRequest) -> OrderAck:
        if self.paper_mode:
            return self._paper_place(req)
        return self._live_place(req)

    def cancel(self, *, cloid: str, symbol: str) -> bool:
        if self.paper_mode:
            return self._paper_open.pop(cloid, None) is not None
        return self._live_cancel(cloid=cloid, symbol=symbol)

    def open_orders(self) -> list[OpenOrderRow]:
        # Live read-only paths land in Task 8.4; place/cancel ship first
        # so the engine boot path is clean.
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

    # ---- live internals ----

    def _ensure_sdk(self):
        if self._sdk is not None:
            return self._sdk
        from py_clob_client_v2 import ApiCreds, ClobClient
        self._sdk = ClobClient(
            host=self._cfg["clob_host"],
            chain_id=self._cfg["chain_id"],
            key=self._cfg["private_key"],
            creds=ApiCreds(
                api_key=self._cfg["clob_api_key"],
                api_secret=self._cfg["clob_api_secret"],
                api_passphrase=self._cfg["clob_api_passphrase"],
            ),
        )
        return self._sdk

    def _live_place(self, req: PlaceRequest) -> OrderAck:
        from py_clob_client_v2 import (
            OrderArgs,
            OrderType,
            PartialCreateOrderOptions,
            Side,
        )
        side = Side.BUY if req.side == "buy" else Side.SELL
        order_type = OrderType.FAK if req.time_in_force == "ioc" else OrderType.GTC
        try:
            sdk = self._ensure_sdk()
            resp = sdk.create_and_post_order(
                order_args=OrderArgs(
                    token_id=req.symbol,
                    price=req.price,
                    side=side,
                    size=req.size,
                ),
                options=PartialCreateOrderOptions(tick_size="0.01"),
                order_type=order_type,
            )
        except Exception as e:
            return OrderAck(
                cloid=req.cloid, venue_oid="", status="rejected",
                error=str(e)[:200],
            )
        if not resp.get("success"):
            return OrderAck(
                cloid=req.cloid,
                venue_oid=str(resp.get("orderID") or ""),
                status="rejected",
                error=str(resp.get("errorMsg", "unknown"))[:200],
            )
        oid = str(resp.get("orderID") or "")
        if oid:
            self._cloid_to_oid[req.cloid] = oid
        making = float(resp.get("makingAmount") or 0)
        taking = float(resp.get("takingAmount") or 0)
        # For a BUY: maker spends USDC (taking), receives tokens (making).
        #   size = makingAmount (tokens); price = takingAmount/makingAmount.
        # For a SELL: maker spends tokens (making), receives USDC (taking).
        #   size = makingAmount (tokens); price = takingAmount/makingAmount.
        fill_size = making
        if req.side == "buy":
            fill_price = (taking / making) if making > 0 else req.price
        else:
            fill_price = (taking / making) if making > 0 else req.price
        status = "filled" if fill_size > 0 else "open"
        return OrderAck(
            cloid=req.cloid, venue_oid=oid,
            status=status, fill_price=fill_price, fill_size=fill_size,
        )

    def _live_cancel(self, *, cloid: str, symbol: str) -> bool:
        # PM cancels by orderID, not cloid. We track cloid→orderID locally in
        # _live_place. Orphans (no mapping) fail-soft to False.
        oid = self._cloid_to_oid.get(cloid)
        if not oid:
            return False
        from py_clob_client_v2 import OrderPayload
        try:
            sdk = self._ensure_sdk()
            resp = sdk.cancel_order(OrderPayload(orderID=oid))
        except Exception:
            return False
        if isinstance(resp, dict):
            # CLOB returns {"canceled": [...], "not_canceled": {...}}.
            canceled = resp.get("canceled") or []
            if oid in canceled:
                self._cloid_to_oid.pop(cloid, None)
                return True
            if resp.get("success"):
                self._cloid_to_oid.pop(cloid, None)
                return True
            return False
        return bool(resp)

