"""Close the diagnostic probe position on #551 (125 shares).

Run on EC2:
    /opt/hl-recorder/.venv/bin/python scripts/close_probe.py
"""
from __future__ import annotations

import os
import sys
import uuid

import requests


def main() -> int:
    addr = os.environ.get("HL_ACCOUNT_ADDRESS")
    key = os.environ.get("HL_API_SECRET_KEY")
    base = os.environ.get("HL_BASE_URL", "https://api.hyperliquid.xyz")
    if not addr or not key:
        print("ERROR: HL_ACCOUNT_ADDRESS / HL_API_SECRET_KEY missing", file=sys.stderr)
        return 2

    # Look up current spot balance to determine exact size to sell.
    r = requests.post(f"{base}/info",
                      json={"type": "spotClearinghouseState", "user": addr.lower()},
                      timeout=10)
    r.raise_for_status()
    balances = {b["coin"]: float(b["total"]) for b in r.json().get("balances", [])}
    held = int(balances.get("+551", 0.0))  # HIP-4 sizes are integer
    if held <= 0:
        print(f"no +551 balance to close (have {balances.get('+551', 0)})")
        return 1
    print(f"holding {held} shares of #551 — selling all")

    # Get current bid to choose marketable limit.
    rb = requests.post(f"{base}/info", json={"type": "l2Book", "coin": "#551"}, timeout=10)
    rb.raise_for_status()
    bids = rb.json().get("levels", [[], []])[0]
    if not bids:
        print("no bid on #551 — cannot close at market")
        return 1
    bid = float(bids[0]["px"])
    # Sell IOC at bid * 0.95 — gives ~5% slippage tolerance while staying marketable.
    limit = round(bid * 0.95, 4)
    print(f"top bid {bid}, IOC sell limit {limit}, expected fill ~${bid * held:.2f}")

    from hyperliquid.exchange import Exchange
    from hyperliquid.utils.types import Cloid
    import eth_account

    wallet = eth_account.Account.from_key(key)
    ex = Exchange(wallet, base_url=base, account_address=addr)
    sym = "#551"
    asset_id = 100_000_000 + 551
    ex.info.coin_to_asset[sym] = asset_id
    ex.info.name_to_coin[sym] = sym
    ex.info.asset_to_sz_decimals[asset_id] = 0

    cloid_obj = Cloid.from_str(f"0x{uuid.uuid4().hex}")
    resp = ex.order(sym, False, float(held), limit,
                    {"limit": {"tif": "Ioc"}},
                    reduce_only=False, cloid=cloid_obj)
    import json as _j
    print(_j.dumps(resp, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
