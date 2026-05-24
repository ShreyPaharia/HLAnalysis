"""One-shot: connect to PM CLOB market WS, subscribe to the YES/NO tokens of
the next-expiring BTC-Up-or-Down market, dump 60s of raw frames to JSONL.

Usage:
  uv run python scripts/capture_pm_ws_fixture.py \
      --out tests/fixtures/pm/ws_book_frames.jsonl --seconds 60
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import websockets

from hlanalysis.adapters.polymarket_gamma import GammaClient

_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


def _discover_active_crypto_tokens(limit_events: int = 20) -> list[str]:
    """Probe Gamma for high-24h-volume crypto binary markets and return the
    YES/NO token IDs. Used only when the canonical `btc-up-or-down-daily`
    series is illiquid (e.g. between settlement and the next listing) — the
    capture fixture just needs *any* real PM WS frames so the adapter's parser
    can be exercised against the live wire format.
    """
    import requests
    r = requests.get(
        "https://gamma-api.polymarket.com/events",
        params={"closed": "false", "limit": 100, "order": "volume24hr",
                "ascending": False, "tag_slug": "crypto"},
        timeout=30,
    )
    r.raise_for_status()
    events = r.json()
    tokens: list[str] = []
    for ev in events[:limit_events]:
        for mk in ev.get("markets") or []:
            raw = mk.get("clobTokenIds")
            if not raw:
                continue
            try:
                tokens.extend(json.loads(raw) if isinstance(raw, str) else raw)
            except json.JSONDecodeError:
                continue
    return tokens


async def _capture(out_path: Path, seconds: int) -> None:
    gc = GammaClient()
    events = gc.fetch_events(series_slug="btc-up-or-down-daily", closed=False)
    markets = list(gc.iter_binary_markets(events))
    token_ids: list[str] = []
    for mk in markets:
        token_ids.extend(json.loads(mk["clobTokenIds"]))
    # The canonical BTC Up/Down daily market often goes near-flat (no book
    # updates) in the gap between resolution and next-day listing. We also pull
    # in the top-volume crypto binary markets so the captured fixture contains
    # enough real WS frames to exercise the adapter's parser. PM WS accepts a
    # mixed list of asset_ids — there's no per-market subscription requirement.
    extra = _discover_active_crypto_tokens()
    token_ids.extend(extra)
    # De-duplicate while preserving order.
    seen: set[str] = set()
    token_ids = [t for t in token_ids if not (t in seen or seen.add(t))]
    if not token_ids:
        sys.exit("No tokens to subscribe to.")
    print(f"capturing {len(token_ids)} tokens "
          f"({len(markets)} btc-up-or-down + ≤{len(extra)} top-crypto)")

    deadline = time.time() + seconds
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with out_path.open("w") as fout:
        async with websockets.connect(_WS_URL, ping_interval=30) as ws:
            await ws.send(json.dumps({"type": "market", "assets_ids": token_ids}))
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue
                if isinstance(msg, bytes):
                    msg = msg.decode("utf-8", errors="replace")
                # PM sometimes sends "PONG" keepalives; skip them.
                if msg.strip() in {"PONG", "PING"}:
                    continue
                fout.write(msg if msg.endswith("\n") else msg + "\n")
                fout.flush()
                frames_written += 1
    print(f"wrote {out_path} ({frames_written} frames)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=int, default=60)
    args = ap.parse_args()
    asyncio.run(_capture(args.out, args.seconds))


if __name__ == "__main__":
    main()
