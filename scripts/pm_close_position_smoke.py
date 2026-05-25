"""One-shot live PM position-close.

Mirror of `pm_single_order_smoke.py` but flipped to sell-side. Discovers the
account's open positions by querying `get_balance_allowance` for each asset
seen in recent trades, then places a FAK sell of the full balance at
best_bid (or `--token-id` to target a specific position; `--size` to
partial-close).

Bypasses the engine — drives PMClient.place() directly via the same
deposit-wallet flow used to open.

Required env vars: same as `pm_single_order_smoke.py` plus
`PM_FUNDER_ADDRESS` for polymarket.com-UI accounts.

Usage:
  uv run python scripts/pm_close_position_smoke.py             # close most-recent buy
  uv run python scripts/pm_close_position_smoke.py --dry-run   # plan only
  uv run python scripts/pm_close_position_smoke.py --token-id <id>  # specific position
  uv run python scripts/pm_close_position_smoke.py --size 10   # partial close
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from hlanalysis.engine.exec_types import PlaceRequest
from hlanalysis.engine.pm_client import PMClient


def _best_bid_from_orderbook(client: PMClient, token_id: str) -> float | None:
    """Return the max-priced bid for a token. PM orderbook returns bids
    sorted ascending by price — best (highest) is the MAX, not bids[0]."""
    try:
        sdk = client._ensure_sdk()  # type: ignore[attr-defined]
        ob = sdk.get_order_book(token_id)
    except Exception as e:
        print(f"  ! get_order_book failed: {e}")
        return None
    bids = getattr(ob, "bids", None) or (ob.get("bids") if isinstance(ob, dict) else None)
    if not bids:
        return None
    def _px(b):
        return float(getattr(b, "price", None) or b["price"])
    return max(_px(b) for b in bids)


def _balance_shares(client: PMClient, token_id: str, funder: str) -> float:
    """Return integer-share balance for an outcome token under the funder."""
    from py_clob_client_v2 import AssetType, BalanceAllowanceParams
    sdk = client._ensure_sdk()  # type: ignore[attr-defined]
    bal = sdk.get_balance_allowance(BalanceAllowanceParams(
        asset_type=AssetType.CONDITIONAL,
        token_id=token_id,
        signature_type=2,  # 2 = POLY_1271 per SDK enum
    ))
    # bal is an object/dict with `balance` in 6-decimal USDC-style units for
    # ERC1155 outcome shares. PM returns raw integer balances (no decimals
    # for conditional tokens — 1 share = 1 unit).
    raw = (getattr(bal, "balance", None)
           or (bal.get("balance") if isinstance(bal, dict) else None))
    if raw is None:
        return 0.0
    val = float(raw)
    # CONDITIONAL tokens use 6 decimals on PM (USDC.e collateral parity);
    # divide to get the share count.
    return val / 1_000_000.0


def _discover_recent_position(client: PMClient, funder: str) -> tuple[str, str] | None:
    """Return (token_id, market) of the most recent BUY taker trade for the funder."""
    from py_clob_client_v2 import TradeParams
    sdk = client._ensure_sdk()  # type: ignore[attr-defined]
    trades = sdk.get_trades(TradeParams(maker_address=funder))
    if not trades:
        return None
    # Filter to our taker BUYs, take the most recent.
    buys = []
    for t in trades:
        side = getattr(t, "side", None) or (t.get("side") if hasattr(t, "get") else None)
        ts = getattr(t, "match_time", None) or (t.get("match_time") if hasattr(t, "get") else None)
        if str(side).upper() == "BUY" and ts:
            buys.append((int(ts), t))
    if not buys:
        return None
    _, latest = max(buys, key=lambda x: x[0])
    token_id = getattr(latest, "asset_id", None) or (latest.get("asset_id") if hasattr(latest, "get") else None)
    market = getattr(latest, "market", None) or (latest.get("market") if hasattr(latest, "get") else None)
    return (str(token_id), str(market)) if token_id else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-id", type=str, default=None,
                    help="token to sell; default = auto-discover from recent BUYs")
    ap.add_argument("--size", type=float, default=None,
                    help="shares to sell; default = full balance")
    ap.add_argument("--limit-buffer", type=float, default=0.01,
                    help="subtract from best_bid for FAK sell limit (default 1¢)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan; do NOT send the order")
    args = ap.parse_args()

    required = ("PM_PRIVATE_KEY", "PM_CLOB_API_KEY", "PM_CLOB_API_SECRET",
                "PM_CLOB_API_PASSPHRASE", "PM_FUNDER_ADDRESS")
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        sys.exit(f"missing env vars: {missing}")
    funder = os.environ["PM_FUNDER_ADDRESS"]

    # 1. Build live PMClient
    print("[1/4] building PMClient (live, deposit-wallet flow)...")
    client = PMClient(
        paper_mode=False,
        clob_host=os.environ.get("PM_CLOB_HOST", "https://clob.polymarket.com"),
        chain_id=int(os.environ.get("PM_CHAIN_ID", "137")),
        private_key=os.environ["PM_PRIVATE_KEY"],
        clob_api_key=os.environ["PM_CLOB_API_KEY"],
        clob_api_secret=os.environ["PM_CLOB_API_SECRET"],
        clob_api_passphrase=os.environ["PM_CLOB_API_PASSPHRASE"],
        funder_address=funder,
        signature_type="POLY_1271",
    )

    # 2. Discover or use the supplied token
    token_id = args.token_id
    if not token_id:
        print("[2/4] discovering most-recent BUY position...")
        discovered = _discover_recent_position(client, funder)
        if not discovered:
            sys.exit("no recent BUY trades found; pass --token-id")
        token_id, market = discovered
        print(f"      token: {token_id}")
        print(f"      market: {market}")
    else:
        print(f"[2/4] using --token-id {token_id}")

    # 3. Look up balance + best bid
    print("[3/4] reading balance + best bid...")
    bal_shares = _balance_shares(client, token_id, funder)
    print(f"      balance: {bal_shares:.4f} shares")
    if bal_shares < 1:
        sys.exit("balance < 1 share; nothing to close")
    sell_size = args.size if args.size is not None else bal_shares
    if sell_size > bal_shares + 1e-6:
        sys.exit(f"--size {sell_size} exceeds balance {bal_shares:.4f}")

    best_bid = _best_bid_from_orderbook(client, token_id)
    if best_bid is None:
        sys.exit("no bids on the book; can't sell at any price (try GTC limit lower)")
    limit_price = round(best_bid - args.limit_buffer, 2)
    if limit_price <= 0.01:
        sys.exit(f"best_bid {best_bid:.4f} too low; FAK sell would land at {limit_price:.4f}")

    notional = sell_size * limit_price
    print(f"[4/4] order plan:")
    print(f"      side=SELL  size={sell_size:.4f}  limit={limit_price}")
    print(f"      best bid: {best_bid:.4f}  (selling 1¢ below = ~immediate fill)")
    print(f"      min notional to receive ≈ ${notional:.2f}")

    if args.dry_run:
        print("\n--dry-run set, exiting without placing order")
        return

    cloid = f"hla-close-{int(time.time())}"
    print(f"\nplacing FAK SELL order... cloid={cloid}")
    ack = client.place(PlaceRequest(
        cloid=cloid,
        symbol=token_id,
        side="sell",
        size=float(sell_size),
        price=limit_price,
        reduce_only=False,  # PM doesn't have reduce_only; PMClient ignores
        time_in_force="ioc",
    ))
    print(f"\nack: status={ack.status}")
    print(f"     venue_oid={ack.venue_oid}")
    if ack.status == "filled":
        print(f"     fill_size={ack.fill_size} shares @ ${ack.fill_price}")
        print(f"     realized notional ≈ ${(ack.fill_price or 0) * (ack.fill_size or 0):.2f}")
    elif ack.status == "rejected":
        print(f"     error: {ack.error}")


if __name__ == "__main__":
    main()
