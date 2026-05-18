"""One-shot fixture builder. Run manually:

    uv run python tests/fixtures/kalshi/_build_fixture.py

Produces a deterministic small manifest + 3 trade parquets covering a single
synthetic Kalshi BTC bucket event with 3 markets.
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    root = Path(__file__).resolve().parent
    (root / "kalshi_trades").mkdir(parents=True, exist_ok=True)

    # Three contiguous markets covering BTC < 79000, 79000-80000, > 80000.
    # Settled with BTC = 79800 ⇒ market M1 wins.
    manifest = {
        "KXBTCD-FIXTURE": {
            "n_rows": 30,
            "last_pull_ts_ns": 0,
            "kind": "bucket",
            "bucket": {
                "event_ticker": "KXBTCD-FIXTURE",
                "series_ticker": "KXBTCD",
                "start_ts_ns": 1_700_000_000_000_000_000,
                "end_ts_ns":   1_700_086_400_000_000_000,
                "thresholds": [79000.0, 80000.0],
                "leg_markets": ["M0", "M1", "M2"],
                "leg_strike_ranges": [
                    [None, 79000.0],
                    [79000.0, 80000.0],
                    [80000.0, None],
                ],
                "leg_settlements": ["no", "yes", "no"],
                "mutex_verified": True,
                "settlement_close_price": 79800.0,
            },
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    base_ts = 1_700_000_000_000_000_000
    # M1 is staggered by +30s so its trades interleave with M0/M2.
    # M0/M2: base, base+60s, base+120s
    # M1:    base+30s, base+90s, base+150s
    # This ensures the leg_events.sort() in _build_stream is exercised by the
    # integration smoke test — reverting the sort would break the monotone-ts
    # assertion.
    market_offsets = {"M0": 0, "M1": 30_000_000_000, "M2": 0}
    for market, prices in (
        ("M0", [0.12, 0.08, 0.06]),  # below 79k, fading
        ("M1", [0.55, 0.70, 0.92]),  # winning bucket, rising
        ("M2", [0.30, 0.20, 0.05]),  # above 80k, fading
    ):
        rows = []
        for i, p in enumerate(prices):
            rows.append({
                "ts_ns": base_ts + market_offsets[market] + i * 60_000_000_000,
                "yes_price": p,
                "size": 5.0,
                "taker_side": "yes" if i % 2 == 0 else "no",
            })
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, root / "kalshi_trades" / f"{market}.parquet")


if __name__ == "__main__":
    main()
