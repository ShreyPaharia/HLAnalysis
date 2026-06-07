"""Polymarket execution client. Implements ExecutionClient.

Paper mode synthesizes OrderAcks, positions, and fills in-memory — same
shape as HLClient.paper. Live mode wraps `py-clob-client-v2`: order
placement uses `OrderType.FAK` for `time_in_force="ioc"` and
`OrderType.GTC` for `"gtc"`. The SDK is imported and constructed lazily
on first live call so paper-only tests don't need the dep installed.
"""
from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable

import requests
from tenacity import (
    retry, retry_if_exception, stop_after_attempt, stop_after_delay,
    wait_exponential,
)

from .exec_types import (
    ClearinghouseState,
    OpenOrderRow,
    OrderAck,
    PlaceRequest,
    UserFillRow,
    VenuePosition,
)


_PM_DATA_API = "https://data-api.polymarket.com"

# Polymarket CLOB accepts prices in [0.01, 0.99] at tick 0.01. Outside this
# band the venue rejects with "invalid price (X), min: 0.01 - max: 0.99". Near
# resolution the favorite's bid sits >0.99, so an exit at limit_price=bid_px
# (theta_harvester / late_resolution) auto-rejects every tick unless clamped.
_PM_PRICE_MIN = 0.01
_PM_PRICE_MAX = 0.99


def _clamp_pm_price(p: float) -> float:
    return min(_PM_PRICE_MAX, max(_PM_PRICE_MIN, p))


def _real_data_api_get(url: str) -> list[dict]:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def _round_down_2(x: float) -> float:
    """Floor to 2 decimal places (cents). PM caps a market BUY's USDC (maker)
    amount at 2 decimals; rounding *down* never spends more than intended."""
    return math.floor(x * 100.0) / 100.0


def _pm_is_transient(exc: BaseException) -> bool:
    """Retry predicate mirroring hl_client's transient/business split.

    Retry only known-transient failures so a momentary data-api / CLOB blip
    backs off and recovers instead of being silently soft-failed to an empty
    book / rejected order (which historically caused reconcile false-settles).
    Everything else — genuine business rejections, malformed requests, 4xx —
    fails fast so a real error is surfaced immediately, not after pointless
    retries.

    Transient:
      - connection drops / timeouts (requests + builtin)
      - HTTP 5xx (server-side) and 429 (rate limit)
    """
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        resp = getattr(exc, "response", None)
        code = getattr(resp, "status_code", None)
        return code is not None and (code >= 500 or code == 429)
    return False


# READ-path retry (open_orders / clearinghouse_state / data-api positions /
# user_fills). Mirrors hl_client._read_retry: bounded in BOTH attempts and total
# elapsed time so a sustained outage can't pin the worker thread.
_PM_READ_RETRY = retry(
    retry=retry_if_exception(_pm_is_transient),
    stop=stop_after_attempt(4) | stop_after_delay(8.0),
    wait=wait_exponential(multiplier=0.2, max=2.0),
    reraise=True,
)

# WRITE-path retry (place / cancel). Mirrors hl_client._WRITE_RETRY — slightly
# tighter bound than the read path so a flapping connection can't keep an order
# submission alive indefinitely.
_PM_WRITE_RETRY = retry(
    retry=retry_if_exception(_pm_is_transient),
    stop=stop_after_attempt(3) | stop_after_delay(5.0),
    wait=wait_exponential(multiplier=0.2, max=2.0),
    reraise=True,
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
        funder_address: str | None = None,
        signature_type: str | None = None,
    ) -> None:
        # PM-UI-onboarded accounts: the EOA only SIGNS; a Safe-style proxy
        # is the on-chain MAKER. PM rejects orders signed by the EOA acting
        # as its own maker with "maker address not allowed". Set
        # funder_address to the proxy ("Receive / Deposit" address on
        # polymarket.com/wallet) and signature_type="POLY_1271" to use the
        # ERC-1271 contract-signature flow.
        # signature_type values: "EOA" | "POLY_PROXY" | "POLY_1271".
        # None defaults to EOA which only works for direct-deployment EOAs.
        self.paper_mode = paper_mode
        self._cfg = dict(
            clob_host=clob_host,
            chain_id=chain_id,
            private_key=private_key,
            clob_api_key=clob_api_key,
            clob_api_secret=clob_api_secret,
            clob_api_passphrase=clob_api_passphrase,
            funder_address=funder_address,
            signature_type=signature_type,
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
        # PM has no positions endpoint in py-clob-client; holdings live on-chain
        # as conditional-token balances, surfaced by the public data-api. The
        # reconciler needs these as venue truth (adopt/keep/vanish PM positions).
        # Injectable so tests don't hit the network.
        self._data_api_get: "Callable[[str], list[dict]]" = _real_data_api_get

    def place(self, req: PlaceRequest) -> OrderAck:
        if self.paper_mode:
            return self._paper_place(req)
        return self._live_place(req)

    def cancel(self, *, cloid: str, symbol: str) -> bool:
        if self.paper_mode:
            return self._paper_open.pop(cloid, None) is not None
        return self._live_cancel(cloid=cloid, symbol=symbol)

    def open_orders(self) -> list[OpenOrderRow]:
        if self.paper_mode:
            return list(self._paper_open.values())
        return self._live_open_orders()

    def clearinghouse_state(self) -> ClearinghouseState:
        if self.paper_mode:
            return ClearinghouseState(
                positions=tuple(self._paper_positions.values()),
                account_value_usd=0.0,
            )
        return self._live_clearinghouse_state()

    def user_fills(self, *, since_ts_ns: int = 0) -> list[UserFillRow]:
        if self.paper_mode:
            return [f for f in self._paper_fills if f.ts_ns >= since_ts_ns]
        return self._live_user_fills(since_ts_ns=since_ts_ns)

    # PM settles via on-chain redeem (CTF), which is NOT a CLOB fill, so the
    # venue fill stream misses settlement PnL — it is tracked separately and
    # the settlement Exit re-derives it from the (price-sourced) winner.
    settlement_reported_as_fill = False

    def realized_pnl_since(self, since_ts_ns: int, *, outcome_only: bool = False) -> float:
        # outcome_only is a no-op for PM: every PM market the engine trades IS an
        # outcome market, so there's no non-strategy activity to filter out.
        fills = self.user_fills(since_ts_ns=since_ts_ns)
        return sum(f.closed_pnl - f.fee for f in fills)

    def realized_pnl_for_symbol(
        self, symbol: str, *, since_ts_ns: int = 0
    ) -> float:
        """Σ(closedPnl − fee) over this account's CLOB fills on `symbol`.
        Excludes the redeem payout (not a fill); the settlement path does not
        rely on this for PM."""
        return sum(
            f.closed_pnl - f.fee
            for f in self.user_fills(since_ts_ns=since_ts_ns)
            if f.symbol == symbol
        )

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
        from py_clob_client_v2 import ApiCreds, ClobClient, SignatureTypeV2
        kwargs: dict = dict(
            host=self._cfg["clob_host"],
            chain_id=self._cfg["chain_id"],
            key=self._cfg["private_key"],
            creds=ApiCreds(
                api_key=self._cfg["clob_api_key"],
                api_secret=self._cfg["clob_api_secret"],
                api_passphrase=self._cfg["clob_api_passphrase"],
            ),
        )
        funder = self._cfg.get("funder_address")
        sig_type = self._cfg.get("signature_type")
        if funder:
            kwargs["funder"] = funder
            # Default to POLY_1271 (Safe / smart-contract proxy) — that's
            # what polymarket.com-UI accounts use. Override via config if
            # using POLY_PROXY (older proxy factory variant).
            type_name = (sig_type or "POLY_1271").upper()
            kwargs["signature_type"] = getattr(SignatureTypeV2, type_name)
        self._sdk = ClobClient(**kwargs)
        return self._sdk

    @_PM_WRITE_RETRY
    def _post_order_raw(self, req: PlaceRequest, price: float) -> dict:
        """Submit the order via the SDK, retrying transient network/5xx blips.
        A genuine business rejection (non-transient exception) fails fast and
        the caller converts it to a rejected ack. Returns the raw SDK response.
        """
        from py_clob_client_v2 import (
            MarketOrderArgs,
            OrderArgs,
            OrderType,
            PartialCreateOrderOptions,
            Side,
        )
        side = Side.BUY if req.side == "buy" else Side.SELL
        opts = PartialCreateOrderOptions(tick_size="0.01")
        sdk = self._ensure_sdk()
        if req.time_in_force == "ioc":
            # Marketable (FAK) orders must go through PM's market-order
            # endpoint, which enforces market-order precision: the USDC
            # (maker) amount of a BUY is capped at 2 decimals and the share
            # (taker) amount at 4. The limit path (create_and_post_order)
            # rounds the USDC amount to 4 decimals (tick-0.01 amount=4), so
            # PM rejected every buy with "invalid amounts ... max accuracy
            # of 2 decimals". For a BUY the market `amount` is the USDC to
            # spend (price·size); for a SELL it is the share count. Round
            # down to 2 dp so the maker amount is always within PM's cap.
            amount = (
                _round_down_2(price * req.size)
                if req.side == "buy"
                else _round_down_2(req.size)
            )
            return sdk.create_and_post_market_order(
                order_args=MarketOrderArgs(
                    token_id=req.symbol,
                    amount=amount,
                    side=side,
                    price=price,
                    order_type=OrderType.FAK,
                ),
                options=opts,
                order_type=OrderType.FAK,
            )
        return sdk.create_and_post_order(
            order_args=OrderArgs(
                token_id=req.symbol,
                price=price,
                side=side,
                size=req.size,
            ),
            options=opts,
            order_type=OrderType.GTC,
        )

    def _live_place(self, req: PlaceRequest) -> OrderAck:
        # Clamp to PM's [0.01, 0.99] tick band before the venue sees it. An
        # exit priced at the favorite's bid_px (>0.99 near resolution) would
        # otherwise auto-reject every tick. Affects the order price and the
        # BUY's USDC `amount` (price·size); the SELL amount is share-count.
        price = _clamp_pm_price(req.price)
        try:
            resp = self._post_order_raw(req, price)
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
        # Polymarket CLOB returns makingAmount/takingAmount from the
        # TAKER'S perspective:
        #   BUY taker:  making = USDC paid, taking = outcome tokens received.
        #   SELL taker: making = tokens sold, taking = USDC received.
        # Empirically confirmed against a $19.50 fill 2026-05-26: ack
        # reported making=19.5, taking=30.47 → 30.47 shares at $0.64/share.
        if req.side == "buy":
            fill_size = taking
            fill_price = (making / taking) if taking > 0 else price
        else:
            fill_size = making
            fill_price = (taking / making) if making > 0 else price
        status = "filled" if fill_size > 0 else "open"
        return OrderAck(
            cloid=req.cloid, venue_oid=oid,
            status=status, fill_price=fill_price, fill_size=fill_size,
        )

    @_PM_WRITE_RETRY
    def _cancel_raw(self, oid: str):
        """Cancel by orderID via the SDK, retrying transient blips."""
        from py_clob_client_v2 import OrderPayload
        sdk = self._ensure_sdk()
        return sdk.cancel_order(OrderPayload(orderID=oid))

    def _live_cancel(self, *, cloid: str, symbol: str) -> bool:
        # PM cancels by orderID, not cloid. We track cloid→orderID locally in
        # _live_place. Orphans (no mapping) fail-soft to False.
        oid = self._cloid_to_oid.get(cloid)
        if not oid:
            return False
        try:
            resp = self._cancel_raw(oid)
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

    @_PM_READ_RETRY
    def _open_orders_raw(self):
        sdk = self._ensure_sdk()
        return sdk.get_open_orders()

    def _live_open_orders(self) -> list[OpenOrderRow]:
        try:
            rows = self._open_orders_raw()
        except Exception:
            return []
        oid_to_cloid = {v: k for k, v in self._cloid_to_oid.items()}
        out: list[OpenOrderRow] = []
        for r in rows or []:
            oid = str(r.get("id") or r.get("orderID") or "")
            symbol = str(r.get("asset_id") or r.get("token_id") or "")
            side_raw = str(r.get("side") or "BUY").upper()
            side = "buy" if side_raw == "BUY" else "sell"
            try:
                price = float(r.get("price") or 0.0)
            except (TypeError, ValueError):
                price = 0.0
            try:
                original = float(r.get("original_size") or r.get("size") or 0.0)
                matched = float(r.get("size_matched") or 0.0)
            except (TypeError, ValueError):
                original, matched = 0.0, 0.0
            remaining = max(original - matched, 0.0)
            created = r.get("created_at") or 0
            try:
                placed_ts_ns = int(float(created) * 1_000_000_000)
            except (TypeError, ValueError):
                placed_ts_ns = 0
            cloid = oid_to_cloid.get(oid, oid)
            out.append(OpenOrderRow(
                cloid=cloid, venue_oid=oid, symbol=symbol,
                side=side, price=price, size=remaining,
                placed_ts_ns=placed_ts_ns,
            ))
        return out

    @_PM_READ_RETRY
    def _balance_allowance_raw(self):
        from py_clob_client_v2 import AssetType, BalanceAllowanceParams
        sdk = self._ensure_sdk()
        return sdk.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL),
        )

    def _live_clearinghouse_state(self) -> ClearinghouseState:
        try:
            resp = self._balance_allowance_raw()
        except Exception:
            return ClearinghouseState(positions=(), account_value_usd=0.0)
        try:
            # USDC is 6-decimal; balance comes back as a stringified integer.
            bal_raw = float(resp.get("balance") or 0.0)
        except (TypeError, ValueError):
            bal_raw = 0.0
        account_value_usd = bal_raw / 1_000_000.0
        # Positions: PM holdings are on-chain conditional-token balances, not in
        # balance-allowance. The public data-api exposes them. We MUST surface
        # them so the reconciler can adopt/keep/vanish PM positions against venue
        # truth; returning an empty set unconditionally (the old behaviour) made
        # the reconciler treat every live PM position as vanished and delete it.
        positions, positions_known = self._fetch_pm_positions()
        return ClearinghouseState(
            positions=positions,
            account_value_usd=account_value_usd,
            positions_known=positions_known,
        )

    @_PM_READ_RETRY
    def _data_api_get_retrying(self, url: str) -> list[dict]:
        """Call the injected data-api getter, retrying transient blips. A
        sustained outage still soft-fails to positions_known=False at the
        caller (so the reconciler skips rather than vanishing positions)."""
        return self._data_api_get(url)

    def _fetch_pm_positions(self) -> tuple[tuple[VenuePosition, ...], bool]:
        """Return (positions, positions_known) from the PM data-api.

        positions_known=False signals the fetch FAILED — the reconciler then
        skips position reconciliation rather than vanishing every position.
        A successful fetch that returns no rows is positions_known=True with an
        empty tuple (genuinely flat).
        """
        funder = self._cfg.get("funder_address")
        if not funder:
            # Direct-EOA deploys have no proxy wallet to query by. We can't
            # establish venue truth, so report "unknown" to keep the reconciler
            # from vanishing positions. (Our PM slots all set a funder.)
            return (), False
        url = (
            f"{_PM_DATA_API}/positions?user={funder}"
            "&sizeThreshold=0.01&limit=500"
        )
        try:
            rows = self._data_api_get_retrying(url)
        except Exception:
            return (), False
        out: list[VenuePosition] = []
        for p in rows or []:
            try:
                qty = float(p.get("size") or 0.0)
            except (TypeError, ValueError):
                continue
            if abs(qty) < 1e-9:
                continue
            try:
                avg = float(p.get("avgPrice") or 0.0)
            except (TypeError, ValueError):
                avg = 0.0
            try:
                upnl = float(p.get("cashPnl") or 0.0)
            except (TypeError, ValueError):
                upnl = 0.0
            out.append(VenuePosition(
                symbol=str(p.get("asset") or ""), qty=qty,
                avg_entry=avg, unrealized_pnl=upnl,
            ))
        return tuple(out), True

    @_PM_READ_RETRY
    def _trades_raw(self, after_s: int):
        from py_clob_client_v2 import TradeParams
        sdk = self._ensure_sdk()
        params = TradeParams(after=after_s) if after_s else None
        return sdk.get_trades(params=params)

    def _live_user_fills(self, *, since_ts_ns: int = 0) -> list[UserFillRow]:
        after_s = since_ts_ns // 1_000_000_000 if since_ts_ns else 0
        try:
            trades = self._trades_raw(after_s)
        except Exception:
            return []
        oid_to_cloid = {v: k for k, v in self._cloid_to_oid.items()}
        out: list[UserFillRow] = []
        for t in trades or []:
            fill_id = str(t.get("id") or "")
            taker_oid = str(t.get("taker_order_id") or "")
            symbol = str(t.get("asset_id") or "")
            side_raw = str(t.get("side") or t.get("trader_side") or "BUY").upper()
            side = "buy" if side_raw == "BUY" else "sell"
            try:
                price = float(t.get("price") or 0.0)
            except (TypeError, ValueError):
                price = 0.0
            try:
                size = float(t.get("size") or 0.0)
            except (TypeError, ValueError):
                size = 0.0
            try:
                fee_bps = float(t.get("fee_rate_bps") or 0.0)
            except (TypeError, ValueError):
                fee_bps = 0.0
            fee = fee_bps / 10_000.0 * price * size
            ts = t.get("match_time") or t.get("last_update") or 0
            try:
                ts_ns = int(float(ts) * 1_000_000_000)
            except (TypeError, ValueError):
                ts_ns = 0
            cloid = oid_to_cloid.get(taker_oid, taker_oid or fill_id)
            out.append(UserFillRow(
                fill_id=fill_id, cloid=cloid, symbol=symbol,
                side=side, price=price, size=size, fee=fee, ts_ns=ts_ns,
                closed_pnl=0.0,
            ))
        return [f for f in out if f.ts_ns >= since_ts_ns]
