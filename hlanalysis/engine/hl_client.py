from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, NoReturn

from loguru import logger
from tenacity import (
    retry, retry_if_exception_type, stop_after_attempt, stop_after_delay,
    wait_exponential, wait_random,
)

from .exec_types import (
    ClearinghouseState,
    OpenOrderRow,
    OrderAck,
    PlaceRequest,
    UserFillRow,
    VenuePosition,
)


class RestError(Exception):
    pass


class RateLimitError(RestError):
    """HL REST returned HTTP 429. A subclass of RestError so existing callers
    that catch RestError are unaffected on final failure, but distinct enough
    for the retry predicate to single it out for backoff."""


def _reraise_rest(e: Exception) -> NoReturn:
    """Map a venue read-path exception onto our error hierarchy, preserving the
    distinctions the read-retry decorator needs:
      - ConnectionError → re-raised as-is (transient; retried)
      - HTTP 429        → RateLimitError (transient; retried)
      - anything else   → RestError (not retried — fail fast)
    """
    if isinstance(e, ConnectionError):
        raise e
    if getattr(e, "status_code", None) == 429:
        raise RateLimitError(str(e)) from e
    raise RestError(str(e)) from e


# Shared retry policy for the READ path (open_orders / clearinghouse_state /
# user_fills). Reconcile polls these every cycle from two HL slots on one IP, so
# a transient 429 or connection blip must back off and recover rather than crash
# the reconcile pass (it previously propagated immediately — only ConnectionError
# was retried, and only on the write path). Bounded in both attempts and elapsed
# time (SHR-41 discipline) so a sustained outage can't pin the worker thread.
# The +wait_random jitter desynchronises the two HL slots: they reconcile on the
# same cadence from one IP, so without jitter their retries collide in lockstep
# and re-trigger HL's 429 rate limit on every backoff. A small random offset
# spreads the retries so one slot's backoff usually clears before the other's.
_read_retry = retry(
    retry=retry_if_exception_type((ConnectionError, RateLimitError)),
    stop=stop_after_attempt(4) | stop_after_delay(8.0),
    wait=wait_exponential(multiplier=0.2, max=2.0) + wait_random(0, 0.3),
    reraise=True,
)


# Shared retry policy for the WRITE path (_live_place / _live_cancel). A flapping
# connection retries with bounded backoff rather than propagating immediately.
# Bound BOTH the attempt count and total elapsed retry time (SHR-41) so a
# flapping connection can't keep the call alive indefinitely. The call is
# offloaded off the event loop by the runtime, so this guards the worker thread /
# order-resolution latency rather than the loop. Consolidated from two
# copy-pasted inline decorators so the policy lives in exactly one place.
_WRITE_RETRY = retry(
    retry=retry_if_exception_type(ConnectionError),
    stop=stop_after_attempt(3) | stop_after_delay(5.0),
    wait=wait_exponential(multiplier=0.2, max=2.0),
    reraise=True,
)


_HEX_CHARS = set("0123456789abcdefABCDEF")

# USD-stable spot coins counted as cash in account_value_usd. HIP-4 binaries are
# spot-classified on HL and funded from spot USDC, so the perp marginSummary is
# ~0; the real account value is this spot cash plus any perp value.
_SPOT_STABLE_COINS = frozenset({"USDC", "USDH", "USDT0", "USDT", "USDE", "USD"})


def _extract_cloid_hex32(internal_cloid: str) -> str:
    """Pull the trailing 32-char hex run out of a cloid in any of the forms
    we encounter on the wire.

    Forms handled:
      hla-<uuid>                      → uuid hex (uuid str has 4 hyphens; strip them)
      hla-<alias>-<hex>               → <hex>; alias may contain any chars
      0x<hex>                         → <hex> (HL's normalized cloid)
      <bare-hex>                      → already hex

    Strategy: strip the leading `hla-` or `0x` anchor, then prefer the substring
    after the last remaining hyphen (multi-account form). If that tail is
    shorter than 32 hex chars (legacy uuid str with internal dashes), fall back
    to "strip all hyphens and take the first 32". Final string is left-padded
    with zeros to 32 hex chars so equality always compares fixed-width.
    """
    s = internal_cloid
    if s.startswith("hla-"):
        s = s[len("hla-"):]
    elif s.startswith("0x") or s.startswith("0X"):
        s = s[2:]
    if "-" in s:
        tail = s.rsplit("-", 1)[1]
        if len(tail) >= 32 and all(c in _HEX_CHARS for c in tail[:32]):
            return tail[:32].lower()
    flat = s.replace("-", "")
    return flat[:32].zfill(32).lower()


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
        pnl_cache_ttl_s: float = 10.0,
    ) -> None:
        self.account_address = account_address
        self.base_url = base_url
        self.paper_mode = paper_mode
        # Paper bookkeeping
        self._paper_orders: dict[str, OrderAck] = {}
        self._paper_open: dict[str, OpenOrderRow] = {}
        self._paper_positions: dict[str, VenuePosition] = {}
        self._paper_fills: list[UserFillRow] = []
        # realized_pnl_since() short-TTL cache (bounded REST load when scanner
        # ticks at 1Hz). Layout: (since_ts_ns, cached_at_monotonic, pnl).
        self._pnl_cache: tuple[int, float, float] | None = None
        self._pnl_cache_ttl_s = pnl_cache_ttl_s
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
        100000150). The encoding is deterministic, so pre-register a wide
        range up-front — survives rollover without a refresh.

        Range covers `#0..#9999` (1000 outcomes × 2 sides) which is years of
        HIP-4 rollovers given current cycle cadence. Tiny memory footprint
        (~10k dict entries × 3 dicts × 2 Info instances = ~60k entries).

        Note: Exchange has its OWN Info instance (Exchange.info, distinct
        from the Info we instantiate here). Both must be patched.
        """
        targets = [t for t in (self._info, getattr(self._exchange, "info", None)) if t is not None]
        if not targets:
            return
        for n in range(10000):  # #0..#9999
            coin = f"#{n}"
            asset_id = 100_000_000 + n
            for info in targets:
                info.coin_to_asset[coin] = asset_id
                info.name_to_coin[coin] = coin
                # HIP-4 sizes appear integer-quantised; szDecimals=0.
                info.asset_to_sz_decimals[asset_id] = 0

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

    @_WRITE_RETRY
    def _live_place(self, req: PlaceRequest) -> OrderAck:
        try:
            assert self._exchange is not None
            # HL rejects reduce_only on HIP-4 (treated as spot): "Reduce-only is
            # invalid for spot trading." The strategy uses reduce_only=True on
            # stop-loss exits; for HIP-4 we strip that flag and rely on the
            # router's position bookkeeping (which verifies sell size doesn't
            # exceed held qty before submitting).
            req_reduce_only = req.reduce_only and not req.symbol.startswith("#")
            # Floor size to the asset's szDecimals. The strategy emits 2-decimal
            # sizes (PM/Binance lot=0.01 convention) but HIP-4 has szDecimals=0
            # (integer sizes only); HL rejects "Order has invalid size" for any
            # fractional input. The asset map is already populated by
            # _patch_info_for_hip4 so we just look it up and floor.
            sz_decimals = 2
            try:
                asset_id = self._exchange.info.coin_to_asset[req.symbol]
                sz_decimals = self._exchange.info.asset_to_sz_decimals[asset_id]
            except (KeyError, AttributeError):
                pass
            import math as _math
            quant = 10 ** sz_decimals
            sized = _math.floor(req.size * quant) / quant if quant > 1 else _math.floor(req.size)
            if sized <= 0:
                return OrderAck(
                    cloid=req.cloid, venue_oid="", status="rejected",
                    error=f"size {req.size} flooring to {sz_decimals} decimals → 0",
                )
            # SDK expects cloid as a Cloid object (32-byte hex). Accept three
            # internal cloid shapes:
            #   - 0x<hex32>                  (already wire-form)
            #   - hla-<uuid>                 (legacy single-account)
            #   - hla-<alias>-<hex>          (multi-account; alias may contain non-hex chars)
            # The wire form drops the alias — HL only needs the 32-char hex tail.
            from hyperliquid.utils.types import Cloid  # type: ignore[import-not-found]
            cloid_str = req.cloid
            if not cloid_str.startswith("0x"):
                cloid_str = f"0x{_extract_cloid_hex32(cloid_str)}"
            cloid_obj = Cloid.from_str(cloid_str)
            resp = self._exchange.order(
                req.symbol, req.side == "buy", sized, req.price,
                {"limit": {"tif": "Ioc" if req.time_in_force == "ioc" else "Gtc"}},
                reduce_only=req_reduce_only,
                cloid=cloid_obj,
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

    @_WRITE_RETRY
    def _live_cancel(self, *, cloid: str, symbol: str) -> bool:
        assert self._exchange is not None
        from hyperliquid.utils.types import Cloid  # type: ignore[import-not-found]
        cloid_str = cloid if cloid.startswith("0x") else f"0x{_extract_cloid_hex32(cloid)}"
        cloid_obj = Cloid.from_str(cloid_str)
        try:
            resp = self._exchange.cancel_by_cloid(symbol, cloid_obj)
        except ConnectionError:
            raise
        except Exception as e:
            raise RestError(str(e)) from e
        return resp.get("status") == "ok"

    # ---- read path ----

    @_read_retry
    def open_orders(self) -> list[OpenOrderRow]:
        if self.paper_mode:
            return list(self._paper_open.values())
        assert self._info is not None
        try:
            rows = self._info.open_orders(self.account_address)
        except Exception as e:
            _reraise_rest(e)
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

    @_read_retry
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
            _reraise_rest(e)
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
        # HIP-4 outcome shares are classified as spot on HL and live in
        # spotClearinghouseState.balances as coin="+N", NOT in the perp
        # assetPositions list above. Without merging them the reconciler's
        # "vanished from venue" branch deletes every locally-booked HIP-4
        # position each cycle, the strategy then sees position=None and
        # re-fires entries that the venue rejects for insufficient USDH —
        # which is exactly the rejection storm we observed in live.
        try:
            spot = self._info.spot_user_state(self.account_address)
        except Exception as e:
            _reraise_rest(e)
        spot_cash_usd = 0.0
        for bal in spot.get("balances", []):
            coin = str(bal.get("coin", ""))
            # USD-stable spot cash is the real funding for HIP-4 (spot-classified)
            # accounts; the perp marginSummary.accountValue is ~0 for them. Without
            # summing it, account_value_usd reads $0 even with $1000s of USDC.
            if coin in _SPOT_STABLE_COINS:
                spot_cash_usd += float(bal.get("total", 0))
                continue
            if not coin.startswith("+"):
                continue
            n_str = coin[1:]
            if not n_str.isdigit():
                continue
            qty = float(bal.get("total", 0))
            if qty == 0:
                continue
            entry_ntl = float(bal.get("entryNtl", 0))
            avg_entry = entry_ntl / qty if qty > 0 else 0.0
            positions.append(VenuePosition(
                symbol=f"#{n_str}",
                qty=qty,
                avg_entry=avg_entry,
                unrealized_pnl=0.0,
            ))
        perp_value = float(data.get("marginSummary", {}).get("accountValue", 0))
        return ClearinghouseState(
            positions=tuple(positions),
            account_value_usd=perp_value + spot_cash_usd,
        )

    def _fetch_fills_raw(self, since_ts_ns: int) -> list[dict]:
        """Fetch raw fills [since_ts_ns, now] via user_fills_by_time, paginating
        past HL's 2000-per-response cap (dedup by tid). Plain user_fills returns
        only the 2000 MOST-RECENT fills, silently truncating realized PnL for
        accounts with >2000 lifetime fills (e.g. v1 had 2399 → capped figure
        overstated realized by ~$36). For a recent cutoff this is a single page,
        so the daily-loss gate pays no extra cost."""
        assert self._info is not None
        start_ms = max(0, since_ts_ns // 1_000_000)
        out: list[dict] = []
        seen: set = set()
        for _ in range(100):  # safety bound: 100 pages × 2000 = 200k fills
            batch = self._info.user_fills_by_time(self.account_address, start_time=start_ms)
            if not batch:
                break
            out.extend(r for r in batch if r.get("tid") not in seen)
            seen.update(r.get("tid") for r in batch)
            if len(batch) < 2000:
                break
            start_ms = max(int(r.get("time", 0)) for r in batch)
        return out

    @_read_retry
    def user_fills(self, *, since_ts_ns: int = 0) -> list[UserFillRow]:
        if self.paper_mode:
            return [f for f in self._paper_fills if f.ts_ns >= since_ts_ns]
        try:
            rows = self._fetch_fills_raw(since_ts_ns)
        except Exception as e:
            _reraise_rest(e)
        out: list[UserFillRow] = []
        for r in rows or []:
            ts_ms = int(r.get("time", 0))
            ts_ns = ts_ms * 1_000_000
            if ts_ns < since_ts_ns:
                continue
            out.append(UserFillRow(
                fill_id=str(r.get("tid", r.get("hash", ""))),
                cloid=str(r.get("cloid", "")),
                symbol=str(r.get("coin", "")),
                side="buy" if r.get("side") == "B" else "sell",
                price=float(r.get("px", 0)),
                size=float(r.get("sz", 0)),
                fee=float(r.get("fee", 0)),
                ts_ns=ts_ns,
                closed_pnl=float(r.get("closedPnl", 0) or 0),
            ))
        return out

    # HL reports HIP-4 / bucket settlement as a fill (dir="Settlement",
    # closedPnl populated), so realized_pnl_since already includes settlement.
    settlement_reported_as_fill = True

    def realized_pnl_for_symbol(
        self, symbol: str, *, since_ts_ns: int = 0
    ) -> float:
        """Venue-truth realized PnL for one leg: Σ(closedPnl − fee) over this
        account's fills on `symbol` since the cutoff. Because HIP-4 settlement
        is a fill, this captures the settlement payout exactly — used to book
        settlement Exits from HL truth instead of re-deriving a winning leg
        (the latter mislabels multi-leg buckets, booking winners as losses)."""
        return sum(
            f.closed_pnl - f.fee
            for f in self.user_fills(since_ts_ns=since_ts_ns)
            if f.symbol == symbol
        )

    def account_pnl_all_time(self) -> float | None:
        """All-time account PnL exactly as HL's Portfolio UI shows it: the
        `portfolio` endpoint's `allTime` pnlHistory tail. This is EQUITY-based
        (current account value net of deposits/withdrawals) and so includes perp
        + spot + funding — NOT just closed-trade closedPnl. It can diverge a lot
        from `realized_pnl_since` for accounts with perp activity (e.g. v1:
        equity PnL +$362.68 vs realized-closedPnl +$161.90 because perpAllTime is
        −$350). Returns None if unavailable (e.g. paper mode)."""
        if self.paper_mode:
            return None
        assert self._info is not None
        try:
            pf = self._info.portfolio(self.account_address)
        except Exception as e:
            _reraise_rest(e)
        for period, d in pf or []:
            if period == "allTime":
                ph = d.get("pnlHistory") or []
                if ph:
                    return float(ph[-1][1])
        return None

    def realized_pnl_since(self, since_ts_ns: int) -> float:
        """Sum (closedPnl - fee) across this account's fills since the cutoff.

        Source of truth for the daily-loss gate. Reads directly from HL so the
        cap survives DB rotations (e.g. multi-account migration leaving v31
        with a fresh state.db while live positions persist).

        Cached for `_pnl_cache_ttl_s` to avoid hammering HL every scan tick.
        """
        cache = getattr(self, "_pnl_cache", None)
        now = time.time()
        if cache is not None:
            cached_since, cached_at, cached_val = cache
            if cached_since == since_ts_ns and (now - cached_at) < self._pnl_cache_ttl_s:
                return cached_val
        fills = self.user_fills(since_ts_ns=since_ts_ns)
        pnl = sum(f.closed_pnl - f.fee for f in fills)
        self._pnl_cache = (since_ts_ns, now, pnl)
        return pnl
