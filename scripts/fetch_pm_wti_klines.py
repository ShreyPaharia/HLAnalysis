#!/usr/bin/env python3
"""Populate the WTI klines cache for a PM WTI Up/Down backtest.

Walks <cache_root>/manifest.json (populate it first via:
    `hl-bt fetch --data-source polymarket --pm-flavor wti_updown ...`)
and fetches per-market 1m Pyth klines from the active CL contract,
writing one JSON file per market endDate.

Idempotent: skips files that already exist unless --refresh.

Usage:
    python scripts/fetch_pm_wti_klines.py \\
        --cache-root data/sim/pm_wti \\
        [--refresh] [--lookback-seconds 259200]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from hlanalysis.backtest.data._pyth_klines import fetch_window_for_market


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", required=True)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--lookback-seconds", type=int, default=86400 * 3)
    args = p.parse_args(argv)

    cache = Path(args.cache_root)
    manifest_path = cache / "manifest.json"
    if not manifest_path.exists():
        logger.error(
            f"No manifest at {manifest_path}; run `hl-bt fetch ... --pm-flavor wti_updown` first"
        )
        return 2

    manifest = json.loads(manifest_path.read_text())
    klines_dir = cache / "wti_klines"
    klines_dir.mkdir(parents=True, exist_ok=True)

    n_fetched, n_skipped, n_empty = 0, 0, 0
    for qid, entry in manifest.items():
        if entry.get("kind") != "binary":
            continue
        end_ns = int(entry["market"]["end_ts_ns"])
        date_iso = datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")
        out = klines_dir / f"{date_iso}.json"
        if out.exists() and not args.refresh:
            n_skipped += 1
            continue
        rows = fetch_window_for_market(end_ns, lookback_seconds=args.lookback_seconds)
        if not rows:
            logger.warning(f"empty kline window for {qid} @ {date_iso}")
            n_empty += 1
            continue
        out.write_text(json.dumps(rows))
        n_fetched += 1
        logger.info(f"  wrote {len(rows):>5d} bars → {out.name}")

    logger.info(f"done: fetched={n_fetched} skipped={n_skipped} empty={n_empty}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
