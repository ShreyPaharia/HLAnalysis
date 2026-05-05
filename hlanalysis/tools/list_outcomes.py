"""List currently active HIP-4 outcome markets.

Run: python -m hlanalysis.tools.list_outcomes
"""

from __future__ import annotations

import json

import requests

URL = "https://api.hyperliquid.xyz/info"


def main() -> None:
    r = requests.post(URL, json={"type": "outcomeMeta"}, timeout=10)
    r.raise_for_status()
    data = r.json()
    outcomes = data.get("outcomes", [])
    if not outcomes:
        print("no active outcome markets")
        return
    for o in outcomes:
        outcome_idx = o.get("outcome")
        name = o.get("name")
        desc = o.get("description") or ""
        fields = dict(p.split(":", 1) for p in desc.split("|") if ":" in p)
        print(f"# outcome={outcome_idx} name={name!r}")
        for k, v in fields.items():
            print(f"#   {k}: {v}")
        for side_idx, side in enumerate(o.get("sideSpecs", [])):
            coin = f"#{10 * outcome_idx + side_idx}"
            print(f"  - venue: hyperliquid")
            print(f"    product_type: prediction_binary")
            print(f"    mechanism: clob")
            print(f'    symbol: "{coin}"  # {side.get("name")}')
            print("    channels: [trades, l2Book, bbo, activeAssetCtx]")
        print()


if __name__ == "__main__":
    main()
