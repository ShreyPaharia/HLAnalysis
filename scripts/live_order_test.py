"""One-off live HL HIP-4 IOC buy for testing the signing path.

Reads HL_ACCOUNT_ADDRESS / HL_API_SECRET_KEY from env. Pulls the top-of-book
ask for the target coin from HL's info endpoint. Computes size from a USDH
budget. Prints the full order details and waits for `yes` on stdin before
sending the signed transaction.

Usage:
    set -a; . ./.env.local; set +a
    uv run python scripts/live_order_test.py --coin '#150' --usdh 11

Defaults to a tiny budget so this stays a smoke test.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import uuid

import requests


HL_INFO = "https://api.hyperliquid.xyz/info"
HL_BASE = "https://api.hyperliquid.xyz"


def get_top_of_book(coin: str) -> tuple[float, float, float]:
    """Return (best_bid, best_ask, ask_size). Raises on error."""
    r = requests.post(HL_INFO, json={"type": "l2Book", "coin": coin}, timeout=10)
    r.raise_for_status()
    levels = r.json().get("levels") or [[], []]
    if not levels[0] or not levels[1]:
        raise RuntimeError(f"no book for {coin}: {levels}")
    bid = levels[0][0]
    ask = levels[1][0]
    return float(bid["px"]), float(ask["px"]), float(ask["sz"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--coin", required=True, help="HL coin, e.g. '#150'")
    p.add_argument("--usdh", type=float, default=11.0,
                   help="Budget in USDH (HIP-4 settles in USDH)")
    p.add_argument("--max-slippage-bps", type=float, default=20.0,
                   help="Cap fill price = ask × (1 + bps/1e4); reject if breaches")
    args = p.parse_args()

    addr = os.environ.get("HL_ACCOUNT_ADDRESS", "").strip()
    key = os.environ.get("HL_API_SECRET_KEY", "").strip()
    if not addr or addr.startswith("0xdeadbeef") or not key or key.startswith("0xdeadbeef"):
        print("ERROR: HL_ACCOUNT_ADDRESS / HL_API_SECRET_KEY not set or are dummy values", file=sys.stderr)
        sys.exit(1)

    bid, ask, ask_sz = get_top_of_book(args.coin)
    if ask <= 0 or ask > 1.0:
        print(f"ERROR: nonsensical ask {ask} for {args.coin}", file=sys.stderr)
        sys.exit(1)
    # Round size to whole shares (HIP-4 sz is integer-sized in the books we've seen).
    size = math.floor(args.usdh / ask)
    if size <= 0:
        print(f"ERROR: budget too small — usdh={args.usdh} / ask={ask:.4f} = {args.usdh/ask:.2f} shares", file=sys.stderr)
        sys.exit(1)

    # IOC limit at ask (no slippage budget added; if it doesn't fill at ask, skip).
    limit_px = ask
    notional = limit_px * size
    cloid_hex = uuid.uuid4().hex[:32]  # SDK accepts a 32-char hex string
    cloid = f"0x{cloid_hex}"

    print("=== LIVE ORDER (paper_mode=false) ===")
    print(f"  account:      {addr[:6]}...{addr[-4:]}")
    print(f"  coin:         {args.coin}")
    print(f"  side:         BUY")
    print(f"  size:         {size} shares")
    print(f"  limit price:  ${limit_px:.4f}  (top-of-book ask; ask_sz={ask_sz})")
    print(f"  bid/ask:      ${bid:.4f} / ${ask:.4f}")
    print(f"  notional:     ${notional:.2f} USDH (budget ${args.usdh})")
    print(f"  TIF:          IOC")
    print(f"  cloid:        {cloid}")
    print()
    answer = input("Send this signed order? Type 'yes' to confirm, anything else to abort: ").strip()
    if answer != "yes":
        print("Aborted. No tx sent.")
        return

    # Sign + send via HL SDK.
    from hyperliquid.exchange import Exchange  # type: ignore[import-not-found]
    from hyperliquid.utils.signing import OrderType  # type: ignore[import-not-found]
    import eth_account  # type: ignore[import-not-found]

    wallet = eth_account.Account.from_key(key)
    ex = Exchange(wallet, base_url=HL_BASE, account_address=addr)
    print("Submitting...")
    resp = ex.order(
        args.coin,
        True,  # is_buy
        float(size),
        float(limit_px),
        {"limit": {"tif": "Ioc"}},
        reduce_only=False,
        cloid=cloid,
    )
    print("\n=== HL response ===")
    import json
    print(json.dumps(resp, indent=2))


if __name__ == "__main__":
    main()
