"""One-time bootstrap: derive PM L2 API credentials from a Polygon EOA
and print them as deploy.yaml-shaped env-var values.

Usage:
  PM_PRIVATE_KEY=0x... uv run python scripts/pm_derive_api_key.py

The three printed `KEY=value` lines are suitable for piping into a
systemd EnvironmentFile or copy-pasting into a secrets store. Run this
once per EOA; the L2 creds it produces are stable and reusable across
restarts.
"""
from __future__ import annotations

import os
import sys


def main() -> None:
    key = os.environ.get("PM_PRIVATE_KEY")
    if not key:
        sys.exit("set PM_PRIVATE_KEY env var")
    from py_clob_client_v2 import ClobClient

    host = os.environ.get("PM_CLOB_HOST", "https://clob.polymarket.com")
    chain_id = int(os.environ.get("PM_CHAIN_ID", "137"))
    client = ClobClient(host=host, chain_id=chain_id, key=key)
    creds = client.create_or_derive_api_key()
    print(f"PM_CLOB_API_KEY={creds.api_key}")
    print(f"PM_CLOB_API_SECRET={creds.api_secret}")
    print(f"PM_CLOB_API_PASSPHRASE={creds.api_passphrase}")


if __name__ == "__main__":
    main()
