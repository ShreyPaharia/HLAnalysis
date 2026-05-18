#!/usr/bin/env python3
"""One-shot puller: Binance USDM perp 1m klines for the PM corpus window.

Writes a single JSON list to data/sim/btc_perp_klines/<start>_to_<end>.json,
matching the layout of the existing data/sim/btc_klines/ spot file.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from hlanalysis.backtest.data.binance_klines import fetch_perp_klines


def _parse_iso_date(s: str) -> int:
    dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-05-01", help="UTC date YYYY-MM-DD")
    p.add_argument("--end", default="2026-05-09", help="UTC date YYYY-MM-DD")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--out", default="data/sim/btc_perp_klines")
    args = p.parse_args()

    start_ms = _parse_iso_date(args.start)
    end_ms = _parse_iso_date(args.end)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.start}_to_{args.end}.json"

    rows = fetch_perp_klines(start_ms, end_ms, symbol=args.symbol, interval="1m")
    serialised = [
        dict(ts_ns=r.ts_ns, open=r.open, high=r.high, low=r.low, close=r.close, volume=r.volume)
        for r in rows
    ]
    out_path.write_text(json.dumps(serialised))
    print(f"wrote {len(serialised)} rows to {out_path}")


if __name__ == "__main__":
    main()
