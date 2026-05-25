"""One-shot live PM order smoke test.

Picks the soonest-expiring active BTC Up/Down daily binary market via Gamma,
chooses the favored leg, fetches the current best ask from the CLOB orderbook,
and places one FAK (immediate-or-cancel) buy sized to a configurable USD
notional (default $20).

Bypasses the engine entirely — drives PMClient.place() directly. Use this to
validate the EIP-712 signing + FAK execution path before flipping the
production engine to paper_mode=false.

Required env vars:
  PM_PRIVATE_KEY
  PM_CLOB_API_KEY
  PM_CLOB_API_SECRET
  PM_CLOB_API_PASSPHRASE

Usage:
  uv run python scripts/pm_single_order_smoke.py            # $20 default
  uv run python scripts/pm_single_order_smoke.py --usd 50   # custom notional
  uv run python scripts/pm_single_order_smoke.py --dry-run  # show plan, no order
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

from hlanalysis.adapters.polymarket_gamma import GammaClient
from hlanalysis.engine.exec_types import PlaceRequest
from hlanalysis.engine.pm_client import PMClient


def _pick_favored_leg(market: dict) -> tuple[str, str, float]:
    """Return (token_id, side_label, indicative_price) for the favored leg.

    Uses Gamma's `outcomePrices` (last-trade mids) for the leg pick; we'll
    fetch fresh ask from the CLOB book for the actual limit price.
    """
    tokens = json.loads(market["clobTokenIds"])
    prices = json.loads(market["outcomePrices"])
    yes_t, no_t = str(tokens[0]), str(tokens[1])
    yes_p, no_p = float(prices[0]), float(prices[1])
    if yes_p >= no_p:
        return yes_t, "YES", yes_p
    return no_t, "NO", no_p


def _best_ask_from_orderbook(client: PMClient, token_id: str) -> float | None:
    """Fetch live best ask via py-clob-client-v2.get_orderbook.

    Returns None if the book is empty (no asks) or the SDK errors out.
    `_sdk` is lazy-initialized in PMClient; force it now.
    """
    try:
        sdk = client._ensure_sdk()  # type: ignore[attr-defined]
        ob = sdk.get_order_book(token_id)
    except Exception as e:
        print(f"  ! get_orderbook failed: {e}")
        return None
    asks = getattr(ob, "asks", None) or (ob.get("asks") if isinstance(ob, dict) else None)
    if not asks:
        return None
    # PM returns asks sorted DESCENDING by price — the "best" ask (lowest
    # price, most attractive to a buyer) is the MIN, not asks[0].
    def _px(a):
        return float(getattr(a, "price", None) or a["price"])
    return min(_px(a) for a in asks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", type=float, default=20.0, help="notional in USD")
    ap.add_argument("--limit-buffer", type=float, default=0.005,
                    help="add this to best_ask for FAK limit (default 0.5¢)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan; do NOT send the order")
    args = ap.parse_args()

    required = ("PM_PRIVATE_KEY", "PM_CLOB_API_KEY", "PM_CLOB_API_SECRET",
                "PM_CLOB_API_PASSPHRASE")
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        sys.exit(f"missing env vars: {missing}")
    # Polymarket-web-UI accounts use a proxy/safe contract as the on-chain
    # maker; the EOA only signs. Set PM_FUNDER_ADDRESS to the address shown
    # at polymarket.com/wallet (the "Receive / Deposit" address). When set,
    # we use the deposit-wallet flow (signature_type=POLY_1271 + funder=...).
    funder = os.environ.get("PM_FUNDER_ADDRESS")

    # 1. Discover active BTC Up/Down daily market with future endDate.
    # Gamma's `closed=false` keeps returning markets that have expired but
    # haven't been recorded as resolved on-chain yet, so we filter ourselves.
    print("[1/4] fetching active BTC Up/Down daily markets via Gamma...")
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    gc = GammaClient()
    events = gc.fetch_events(series_slug="btc-up-or-down-daily", closed=False)

    def _ends_in_future(mkt: dict) -> bool:
        try:
            end = datetime.fromisoformat(mkt.get("endDate", "").replace("Z", "+00:00"))
        except ValueError:
            return False
        return end > now

    active = [m for m in gc.iter_binary_markets(events) if _ends_in_future(m)]
    markets = sorted(active, key=lambda m: m["endDate"])
    if not markets:
        sys.exit("no future-ending BTC Up/Down daily markets")
    mk = markets[0]
    print(f"      market: {mk['conditionId']}")
    print(f"      ends:   {mk.get('endDate')} ({(datetime.fromisoformat(mk['endDate'].replace('Z','+00:00')) - now).total_seconds()/3600:.1f}h from now)")

    # 2. Pick favored leg
    token_id, side_label, indicative = _pick_favored_leg(mk)
    print(f"[2/4] favored leg: {side_label} @ indicative={indicative:.4f}")
    print(f"      token_id: {token_id}")

    # 3. Build live PMClient + fetch fresh ask
    print("[3/4] building PMClient (live), fetching orderbook ask...")
    client = PMClient(
        paper_mode=False,
        clob_host=os.environ.get("PM_CLOB_HOST", "https://clob.polymarket.com"),
        chain_id=int(os.environ.get("PM_CHAIN_ID", "137")),
        private_key=os.environ["PM_PRIVATE_KEY"],
        clob_api_key=os.environ["PM_CLOB_API_KEY"],
        clob_api_secret=os.environ["PM_CLOB_API_SECRET"],
        clob_api_passphrase=os.environ["PM_CLOB_API_PASSPHRASE"],
    )
    # If we have a deposit-wallet funder, override PMClient's lazy SDK
    # construction with one that uses the POLY_1271 deposit-wallet flow.
    # Permanent PMConfig+PMClient support for this lands in a follow-up
    # commit; this is the smoke-test bridge.
    if funder:
        print(f"      using deposit-wallet flow; funder={funder}")
        from py_clob_client_v2 import ApiCreds, ClobClient, SignatureTypeV2
        client._sdk = ClobClient(  # type: ignore[attr-defined]
            host=os.environ.get("PM_CLOB_HOST", "https://clob.polymarket.com"),
            chain_id=int(os.environ.get("PM_CHAIN_ID", "137")),
            key=os.environ["PM_PRIVATE_KEY"],
            creds=ApiCreds(
                api_key=os.environ["PM_CLOB_API_KEY"],
                api_secret=os.environ["PM_CLOB_API_SECRET"],
                api_passphrase=os.environ["PM_CLOB_API_PASSPHRASE"],
            ),
            signature_type=SignatureTypeV2.POLY_1271,
            funder=funder,
        )
    best_ask = _best_ask_from_orderbook(client, token_id)
    if best_ask is None:
        # Fall back to indicative + 1¢ — better than aborting on transient
        # SDK get_orderbook hiccup.
        best_ask = indicative + 0.01
        print(f"      no live ask returned; falling back to indicative+1¢ = {best_ask:.4f}")
    else:
        print(f"      best ask: {best_ask:.4f}")

    # Compute limit. The live `best_ask` from get_order_book is the source
    # of truth — Gamma's `outcomePrices` is stale (last trade, not mid).
    # Sanity guard: refuse to lift an ask > 0.95 against a much lower
    # indicative — that's the stale-lonely-seller pattern, not a real
    # quote. The strategy gate (price_extreme_max) catches this in prod;
    # this script enforces the same defensively.
    if best_ask > 0.95 and best_ask - indicative > 0.20:
        sys.exit(
            f"best_ask {best_ask:.4f} is much wider than indicative "
            f"{indicative:.4f} — likely stale lonely seller; refusing to "
            f"trade. Re-run later when book tightens."
        )
    # PM tick = 0.01; round limit to 2 decimals.
    limit_price = round(best_ask + args.limit_buffer, 2)
    # PM share-size is an integer (1-share granularity); floor so we don't
    # exceed notional.
    size = math.floor(args.usd / limit_price)
    if size < 5:  # PM min is typically 5 shares
        sys.exit(f"size {size} below PM minimum 5; raise --usd or check price")
    realized_notional = size * limit_price
    print(f"[4/4] order plan:")
    print(f"      side=BUY  size={size}  limit={limit_price}")
    print(f"      max notional = {realized_notional:.2f} USDC")

    if args.dry_run:
        print("\n--dry-run set, exiting without placing order")
        return

    cloid = f"hla-smoke-{int(time.time())}"
    print(f"\nplacing FAK order... cloid={cloid}")
    ack = client.place(PlaceRequest(
        cloid=cloid,
        symbol=token_id,
        side="buy",
        size=float(size),
        price=limit_price,
        reduce_only=False,
        time_in_force="ioc",   # → FAK in PMClient._live_place
    ))
    print(f"\nack: status={ack.status}")
    print(f"     venue_oid={ack.venue_oid}")
    if ack.status == "filled":
        print(f"     fill_price={ack.fill_price} fill_size={ack.fill_size}")
        print(f"     realized notional ≈ ${(ack.fill_price or 0) * (ack.fill_size or 0):.2f}")
    elif ack.status == "open":
        print("     (open = resting on book; FAK should have filled or "
              "cancelled — investigate)")
    elif ack.status == "rejected":
        print(f"     error: {ack.error}")


if __name__ == "__main__":
    main()
