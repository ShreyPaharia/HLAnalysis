#!/usr/bin/env python3
"""Pull genuine Binance SPOT 1s klines into the PM cache (`btc_klines_1s/`).

Feeds the `--pm-reference-source klines_1s` path for the BTC-ref-equivalence
experiment (recorded Binance BBO vs pulled 1s klines, both at dt=5). Writes one
JSON file per UTC day under ``<cache>/btc_klines_1s/`` in the same row shape as
the 1m ``btc_klines`` cache ({ts_ns, open, high, low, close, volume}), so
``PolymarketDataSource._load_all_klines_1s`` reads it directly. Idempotent: skips
days already present (pass --refresh to re-pull).

Spot 1s is NOT geo-blocked from this IP (perp fapi IS — do not use it here).

Usage:
  uv run python scripts/fetch_binance_1s_klines.py --start 2026-05-26 --end 2026-06-05
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from hlanalysis.backtest.data.binance_klines import fetch_klines

_DEFAULT_CACHE = Path(
    "/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data/sim"
)
_DAY_MS = 86_400_000


def _day_ms(d: datetime) -> int:
    return int(d.replace(tzinfo=timezone.utc).timestamp() * 1000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="ISO date YYYY-MM-DD (UTC, inclusive)")
    ap.add_argument("--end", required=True, help="ISO date YYYY-MM-DD (UTC, exclusive)")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--cache-root", default=str(_DEFAULT_CACHE))
    ap.add_argument("--subdir", default="btc_klines_1s")
    ap.add_argument("--refresh", action="store_true", help="re-pull days already cached")
    args = ap.parse_args()

    out_dir = Path(args.cache_root) / args.subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    day = start
    total_rows = 0
    while day < end:
        tag = day.strftime("%Y-%m-%d")
        fpath = out_dir / f"{tag}.json"
        if fpath.exists() and fpath.stat().st_size > 100 and not args.refresh:
            print(f"[skip] {tag} (cached)")
            day += timedelta(days=1)
            continue
        start_ms = _day_ms(day)
        end_ms = start_ms + _DAY_MS
        t0 = time.time()
        rows = fetch_klines(start_ms, end_ms, symbol=args.symbol, interval="1s")
        out = [
            {"ts_ns": k.ts_ns, "open": k.open, "high": k.high,
             "low": k.low, "close": k.close, "volume": k.volume}
            for k in rows
        ]
        fpath.write_text(json.dumps(out))
        total_rows += len(out)
        print(f"[ok] {tag}: {len(out)} 1s bars ({time.time() - t0:.1f}s)")
        day += timedelta(days=1)

    print(f"\ndone: {total_rows} new 1s bars under {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
