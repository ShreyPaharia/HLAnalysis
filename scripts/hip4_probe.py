"""One-shot probe: place a tiny IOC HIP-4 order using the engine's agent key
and print the raw HL response. Designed to surface the rejection reason when
the engine reports `13,741 rejected, 0 filled`.

Run on EC2 (where /opt/hl-recorder/.engine.env has the secrets):
    sudo /opt/hl-recorder/.venv/bin/python scripts/hip4_probe.py

Targets the cheapest available BTC-related HIP-4 leg (lowest ask), buys ~$1
worth via IOC. Safe: total spend bounded to ~$1.
"""
from __future__ import annotations

import os
import sys
import time
import uuid

import requests


def main() -> int:
    addr = os.environ.get("HL_ACCOUNT_ADDRESS")
    key = os.environ.get("HL_API_SECRET_KEY")
    base = os.environ.get("HL_BASE_URL", "https://api.hyperliquid.xyz")
    if not addr or not key:
        print("ERROR: HL_ACCOUNT_ADDRESS / HL_API_SECRET_KEY env vars missing",
              file=sys.stderr)
        return 2

    # Discover today's active HIP-4 outcomes.
    r = requests.post(f"{base}/info", json={"type": "outcomeMeta"}, timeout=10)
    r.raise_for_status()
    meta = r.json()
    outcomes = meta.get("outcomes", [])
    btc_outcomes = [
        o for o in outcomes
        if "underlying:BTC" in (o.get("description") or "")
        or "BTC" in (o.get("description") or "")
    ]
    if not btc_outcomes:
        print("no BTC HIP-4 outcomes today")
        return 1

    # Probe each leg's ask, pick the cheapest one (lowest ask).
    candidates: list[tuple[str, float, float]] = []
    for o in btc_outcomes:
        for side_idx, side_name in ((0, "yes"), (1, "no")):
            sym = f"#{o['outcome'] * 10 + side_idx}"
            rr = requests.post(f"{base}/info",
                               json={"type": "l2Book", "coin": sym}, timeout=10)
            if rr.status_code != 200:
                continue
            j = rr.json()
            asks = j.get("levels", [[], []])[1]
            if not asks:
                continue
            ask = float(asks[0]["px"])
            sz = float(asks[0]["sz"])
            candidates.append((sym, ask, sz))

    if not candidates:
        print("no asks available on any BTC HIP-4 leg")
        return 1

    candidates.sort(key=lambda x: x[1])
    sym, ask, depth = candidates[0]
    # Round-up to a marketable IOC limit: 1.10x ask, capped at 1.0 (leg price max)
    limit = min(1.0, round(ask * 1.10, 4))
    # Test fractional size to confirm/reject the "szDecimals=0" hypothesis.
    # If this rejects with 'invalid size', that's the engine's bug — its
    # strategy emits floor(usd/px*100)/100 → 100.10-style two-decimal sizes.
    size = float(os.environ.get("PROBE_SIZE", "100.10"))
    print(f"probe: BUY IOC {sym}  size={size}  limit={limit}  "
          f"(top ask {ask} sz {depth}, estimated cost ${ask:.4f})")

    # Sign + place via the HL SDK (same path as the engine).
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils.types import Cloid
    import eth_account

    wallet = eth_account.Account.from_key(key)
    ex = Exchange(wallet, base_url=base, account_address=addr)

    # Patch the SDK asset map for HIP-4 (same logic as engine/hl_client.py).
    asset_id = 100_000_000 + int(sym[1:])
    for info_obj in (ex.info,):
        info_obj.coin_to_asset[sym] = asset_id
        info_obj.name_to_coin[sym] = sym
        info_obj.asset_to_sz_decimals[asset_id] = 0

    cloid_hex = uuid.uuid4().hex
    cloid_obj = Cloid.from_str(f"0x{cloid_hex}")

    t0 = time.time()
    resp = ex.order(
        sym, True, size, limit,
        {"limit": {"tif": "Ioc"}},
        reduce_only=False,
        cloid=cloid_obj,
    )
    dt = time.time() - t0

    import json as _j
    print(f"\nraw response (elapsed {dt:.3f}s):")
    print(_j.dumps(resp, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
