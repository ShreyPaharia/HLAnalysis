from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger
from tenacity import (
    retry, retry_if_exception_type, stop_after_attempt, wait_exponential,
)


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


class RestError(Exception):
    pass


class HLClient:
    """Wraps the official hyperliquid-python-sdk for engine use.

    `paper_mode=True` short-circuits all order REST calls and synthesizes acks.
    Read-only calls (open_orders, clearinghouse_state, user_fills) reflect the
    in-memory paper book in paper mode; in live mode they hit HL.
    """

    def __init__(
        self,
        *,
        account_address: str,
        api_secret_key: str,
        base_url: str,
        paper_mode: bool,
    ) -> None:
        self.account_address = account_address
        self.base_url = base_url
        self.paper_mode = paper_mode
        # Paper bookkeeping
        self._paper_orders: dict[str, OrderAck] = {}
        self._paper_open: dict[str, OpenOrderRow] = {}
        self._paper_positions: dict[str, VenuePosition] = {}
        self._paper_fills: list[UserFillRow] = []
        self._exchange = None
        self._info = None
        if not paper_mode:
            # Lazy SDK import so paper-only test runs don't need credentials.
            from hyperliquid.exchange import Exchange  # type: ignore[import-not-found]
            from hyperliquid.info import Info  # type: ignore[import-not-found]
            import eth_account  # type: ignore[import-not-found]
            try:
                wallet = eth_account.Account.from_key(api_secret_key)
                self._exchange = Exchange(wallet, base_url=base_url, account_address=account_address)
            except Exception:
                # Allow construction with invalid credentials for testing;
                # any live call will fail at the SDK level.
                pass
            self._info = Info(base_url=base_url, skip_ws=True)
            self._patch_info_for_hip4()

    def _patch_info_for_hip4(self) -> None:
        """Inject HIP-4 outcome legs into the SDK's coin↔asset map.

        SDK v0.23.0 only knows perp + spot assets; HIP-4 coins (`#N`) raise
        KeyError in `Info.name_to_asset`. Encoding observed from a UI-placed
        trade: asset_id = 100_000_000 + N for coin `#N` (e.g. #150 →
        100000150). Pull live outcomeMeta and register a mapping for every
        side of every active outcome so the engine can place IOC orders
        through the standard SDK path.
        """
        if self._info is None:
            return
        try:
            import requests
            r = requests.post(
                self._exchange.base_url.rstrip("/") + "/info"
                if self._exchange else "https://api.hyperliquid.xyz/info",
                json={"type": "outcomeMeta"}, timeout=5,
            )
            r.raise_for_status()
            data = r.json()
        except Exception:
            return  # Best-effort; engine will retry on next reconnect
        for o in data.get("outcomes", []) or []:
            outcome_idx = o.get("outcome")
            if outcome_idx is None:
                continue
            for side_idx in range(len(o.get("sideSpecs", []) or [])):
                n = 10 * int(outcome_idx) + side_idx
                coin = f"#{n}"
                asset_id = 100_000_000 + n
                self._info.coin_to_asset[coin] = asset_id
                self._info.name_to_coin[coin] = coin
                # HIP-4 sizes appear integer-quantised; szDecimals=0.
                self._info.asset_to_sz_decimals[asset_id] = 0

    # ---- write path ----

    def place(self, req: PlaceRequest) -> OrderAck:
        if self.paper_mode:
            return self._paper_place(req)
        return self._live_place_safe(req)

    def cancel(self, *, cloid: str, symbol: str) -> bool:
        if self.paper_mode:
            row = self._paper_open.pop(cloid, None)
            return row is not None
        return self._live_cancel(cloid=cloid, symbol=symbol)

    def _paper_place(self, req: PlaceRequest) -> OrderAck:
        # Idempotency: same cloid → same ack.
        if req.cloid in self._paper_orders:
            return self._paper_orders[req.cloid]
        # Crude paper fill model: IOC at price > 0 is treated as immediately
        # filled at the requested price; price <= 0 is rejected. This is a
        # placeholder — fidelity improves in Plan 1C with a fake L2.
        if req.price <= 0:
            ack = OrderAck(cloid=req.cloid, venue_oid=f"paper-{req.cloid}",
                           status="rejected", error="non_marketable_paper_default")
            self._paper_orders[req.cloid] = ack
            return ack
        ack = OrderAck(
            cloid=req.cloid, venue_oid=f"paper-{req.cloid}",
            status="filled", fill_price=req.price, fill_size=req.size,
        )
        self._paper_orders[req.cloid] = ack
        # Track virtual position (only on entries; paper exit logic netts out)
        signed = req.size if req.side == "buy" else -req.size
        if req.reduce_only:
            existing = self._paper_positions.get(req.symbol)
            if existing is not None:
                new_qty = existing.qty + signed
                if abs(new_qty) < 1e-9:
                    self._paper_positions.pop(req.symbol, None)
                else:
                    self._paper_positions[req.symbol] = VenuePosition(
                        symbol=req.symbol, qty=new_qty,
                        avg_entry=existing.avg_entry, unrealized_pnl=0.0,
                    )
        else:
            existing = self._paper_positions.get(req.symbol)
            if existing is None:
                self._paper_positions[req.symbol] = VenuePosition(
                    symbol=req.symbol, qty=signed, avg_entry=req.price, unrealized_pnl=0.0,
                )
            else:
                # Average up
                tot = existing.qty + signed
                avg = (existing.qty * existing.avg_entry + signed * req.price) / tot if tot else 0.0
                self._paper_positions[req.symbol] = VenuePosition(
                    symbol=req.symbol, qty=tot, avg_entry=avg, unrealized_pnl=0.0,
                )
        ts = time.time_ns()
        self._paper_fills.append(UserFillRow(
            fill_id=f"f-{req.cloid}-{ts}", cloid=req.cloid, symbol=req.symbol,
            side=req.side, price=req.price, size=req.size, fee=0.0, ts_ns=ts,
        ))
        return ack

    def _live_place_safe(self, req: PlaceRequest) -> OrderAck:
        """Outer wrapper: converts any escaped ConnectionError to RestError."""
        try:
            return self._live_place(req)
        except ConnectionError as e:
            raise RestError(str(e)) from e

    @retry(
        retry=retry_if_exception_type(ConnectionError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.2, max=2.0),
        reraise=True,
    )
    def _live_place(self, req: PlaceRequest) -> OrderAck:
        try:
            assert self._exchange is not None
            # SDK signature differs across versions; this is the contract we
            # rely on. Tweak as needed when the SDK is locked.
            resp = self._exchange.order(
                req.symbol, req.side == "buy", req.size, req.price,
                {"limit": {"tif": "Ioc" if req.time_in_force == "ioc" else "Gtc"}},
                reduce_only=req.reduce_only,
                cloid=req.cloid,
            )
        except ConnectionError:
            raise
        except Exception as e:
            raise RestError(str(e)) from e
        data = resp.get("response", {}).get("data", {}).get("statuses", [{}])[0]
        if "error" in data:
            return OrderAck(
                cloid=req.cloid, venue_oid="", status="rejected", error=str(data["error"]),
            )
        if "filled" in data:
            f = data["filled"]
            return OrderAck(
                cloid=req.cloid, venue_oid=str(f.get("oid", "")),
                status="filled", fill_price=float(f.get("avgPx", 0)),
                fill_size=float(f.get("totalSz", 0)),
            )
        if "resting" in data:
            return OrderAck(
                cloid=req.cloid, venue_oid=str(data["resting"].get("oid", "")), status="open",
            )
        return OrderAck(cloid=req.cloid, venue_oid="", status="rejected", error="unknown_response")

    @retry(
        retry=retry_if_exception_type(ConnectionError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.2, max=2.0),
        reraise=True,
    )
    def _live_cancel(self, *, cloid: str, symbol: str) -> bool:
        assert self._exchange is not None
        try:
            resp = self._exchange.cancel_by_cloid(symbol, cloid)
        except ConnectionError:
            raise
        except Exception as e:
            raise RestError(str(e)) from e
        return resp.get("status") == "ok"

    # ---- read path ----

    def open_orders(self) -> list[OpenOrderRow]:
        if self.paper_mode:
            return list(self._paper_open.values())
        assert self._info is not None
        try:
            rows = self._info.open_orders(self.account_address)
        except Exception as e:
            raise RestError(str(e)) from e
        out: list[OpenOrderRow] = []
        for r in rows or []:
            out.append(OpenOrderRow(
                cloid=str(r.get("cloid", "")),
                venue_oid=str(r.get("oid", "")),
                symbol=str(r.get("coin", "")),
                side="buy" if r.get("side") == "B" else "sell",
                price=float(r.get("limitPx", 0)),
                size=float(r.get("sz", 0)),
                placed_ts_ns=int(r.get("timestamp", 0)) * 1_000_000,
            ))
        return out

    def clearinghouse_state(self) -> ClearinghouseState:
        if self.paper_mode:
            return ClearinghouseState(
                positions=tuple(self._paper_positions.values()),
                account_value_usd=0.0,
            )
        assert self._info is not None
        try:
            data = self._info.user_state(self.account_address)
        except Exception as e:
            raise RestError(str(e)) from e
        positions: list[VenuePosition] = []
        for ap in data.get("assetPositions", []):
            p = ap.get("position", {})
            qty = float(p.get("szi", 0))
            if qty == 0:
                continue
            positions.append(VenuePosition(
                symbol=str(p.get("coin", "")),
                qty=qty,
                avg_entry=float(p.get("entryPx", 0)),
                unrealized_pnl=float(p.get("unrealizedPnl", 0)),
            ))
        return ClearinghouseState(
            positions=tuple(positions),
            account_value_usd=float(data.get("marginSummary", {}).get("accountValue", 0)),
        )

    def user_fills(self, *, since_ts_ns: int = 0) -> list[UserFillRow]:
        if self.paper_mode:
            return [f for f in self._paper_fills if f.ts_ns >= since_ts_ns]
        assert self._info is not None
        try:
            rows = self._info.user_fills(self.account_address)
        except Exception as e:
            raise RestError(str(e)) from e
        out: list[UserFillRow] = []
        for r in rows or []:
            ts_ms = int(r.get("time", 0))
            ts_ns = ts_ms * 1_000_000
            if ts_ns < since_ts_ns:
                continue
            out.append(UserFillRow(
                fill_id=str(r.get("hash", r.get("tid", ""))),
                cloid=str(r.get("cloid", "")),
                symbol=str(r.get("coin", "")),
                side="buy" if r.get("side") == "B" else "sell",
                price=float(r.get("px", 0)),
                size=float(r.get("sz", 0)),
                fee=float(r.get("fee", 0)),
                ts_ns=ts_ns,
            ))
        return out
