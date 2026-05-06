"""Quick gap audit across recorded streams.

For each (venue, product_type, event, symbol) it reports:
  - first/last timestamp (local_recv_ts)
  - row count
  - p50 / p99 / max inter-event seconds
  - count of inter-event gaps > thresholds (5s, 30s, 60s)
  - top 5 longest gap windows

Run: .venv/bin/python tools/check_gaps.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

# (label, glob, gap-threshold-seconds-for-flagging)
STREAMS = [
    # (venue, ptype, mech, event, symbol, expected_cadence_hint_s)
    ("hyperliquid", "perp", "clob", "trade", "BTC", None),
    ("hyperliquid", "perp", "clob", "bbo", "BTC", None),
    ("hyperliquid", "perp", "clob", "book_snapshot", "BTC", None),
    ("hyperliquid", "perp", "clob", "mark", "BTC", None),
    ("hyperliquid", "perp", "clob", "oracle", "BTC", None),
    ("hyperliquid", "perp", "clob", "funding", "BTC", None),
    ("hyperliquid", "spot", "clob", "trade", "*", None),
    ("hyperliquid", "spot", "clob", "bbo", "*", None),
    ("hyperliquid", "spot", "clob", "book_snapshot", "*", None),
    ("hyperliquid", "prediction_binary", "clob", "trade", "*", None),
    ("hyperliquid", "prediction_binary", "clob", "bbo", "*", None),
    ("hyperliquid", "prediction_binary", "clob", "book_snapshot", "*", None),
    ("hyperliquid", "prediction_binary", "clob", "mark", "*", None),
    ("binance", "perp", "clob", "trade", "BTCUSDT", None),
    ("binance", "perp", "clob", "bbo", "BTCUSDT", None),
    ("binance", "perp", "clob", "book_snapshot", "BTCUSDT", None),
    ("binance", "perp", "clob", "mark", "BTCUSDT", None),
    ("binance", "perp", "clob", "funding", "BTCUSDT", None),
    ("binance", "spot", "clob", "trade", "BTCUSDT", None),
    ("binance", "spot", "clob", "bbo", "BTCUSDT", None),
    ("binance", "spot", "clob", "book_snapshot", "BTCUSDT", None),
]


def glob_for(venue, ptype, mech, ev, sym):
    return str(
        DATA
        / f"venue={venue}"
        / f"product_type={ptype}"
        / f"mechanism={mech}"
        / f"event={ev}"
        / f"symbol={sym}"
        / "date=*"
        / "hour=*"
        / "*.parquet"
    )


def fmt_ts(ns):
    if ns is None:
        return "-"
    import datetime as dt
    return dt.datetime.fromtimestamp(int(ns) / 1e9, tz=dt.timezone.utc).strftime("%H:%M:%S")


def main():
    con = duckdb.connect()
    print(f"{'stream':<70} {'rows':>9} {'first':>10} {'last':>10}  {'p50_s':>7} {'p99_s':>7} {'max_s':>7}  {'>5s':>4} {'>30s':>4} {'>60s':>4}")
    print("-" * 140)
    summary = []
    long_gaps = []
    for venue, ptype, mech, ev, sym, _ in STREAMS:
        g = glob_for(venue, ptype, mech, ev, sym)
        # are there any files?
        try:
            n = con.execute(
                f"SELECT count(*) FROM read_parquet('{g}', hive_partitioning=1)"
            ).fetchone()[0]
        except duckdb.IOException:
            n = 0
        if n == 0:
            print(f"{venue}/{ptype}/{ev}/{sym:<10} -- no data")
            continue
        df = con.execute(
            f"""
            WITH t AS (
              SELECT local_recv_ts AS ts
              FROM read_parquet('{g}', hive_partitioning=1)
              WHERE local_recv_ts IS NOT NULL
            ),
            ord AS (
              SELECT ts, lag(ts) OVER (ORDER BY ts) AS prev_ts FROM t
            ),
            d AS (
              SELECT ts, prev_ts, (ts - prev_ts)/1e9 AS gap_s FROM ord WHERE prev_ts IS NOT NULL
            )
            SELECT
              (SELECT count(*) FROM t) AS n,
              (SELECT min(ts) FROM t) AS first_ts,
              (SELECT max(ts) FROM t) AS last_ts,
              quantile_cont(gap_s, 0.50) AS p50,
              quantile_cont(gap_s, 0.99) AS p99,
              max(gap_s) AS max_g,
              count(*) FILTER (WHERE gap_s > 5)  AS g5,
              count(*) FILTER (WHERE gap_s > 30) AS g30,
              count(*) FILTER (WHERE gap_s > 60) AS g60
            FROM d
            """
        ).fetchone()
        n_rows, first_ts, last_ts, p50, p99, mx, g5, g30, g60 = df
        label = f"{venue}/{ptype}/{ev}/{sym}"
        print(
            f"{label:<70} {n_rows:>9} {fmt_ts(first_ts):>10} {fmt_ts(last_ts):>10}  "
            f"{(p50 or 0):>7.2f} {(p99 or 0):>7.2f} {(mx or 0):>7.1f}  "
            f"{g5:>4} {g30:>4} {g60:>4}"
        )
        # collect long gaps (top 5 per stream, > 30s)
        rows = con.execute(
            f"""
            WITH t AS (
              SELECT local_recv_ts AS ts
              FROM read_parquet('{g}', hive_partitioning=1)
              WHERE local_recv_ts IS NOT NULL
            ),
            ord AS (
              SELECT ts, lag(ts) OVER (ORDER BY ts) AS prev_ts FROM t
            )
            SELECT prev_ts, ts, (ts-prev_ts)/1e9 AS gap_s
            FROM ord WHERE prev_ts IS NOT NULL AND (ts-prev_ts)/1e9 > 30
            ORDER BY gap_s DESC LIMIT 5
            """
        ).fetchall()
        for prev_ts, ts, gap_s in rows:
            long_gaps.append((label, prev_ts, ts, gap_s))

    print()
    print("=== Long gaps (>30s), top per stream ===")
    long_gaps.sort(key=lambda r: -r[3])
    for label, prev_ts, ts, gap_s in long_gaps[:40]:
        print(f"  {gap_s:>7.1f}s  {fmt_ts(prev_ts)} -> {fmt_ts(ts)}  {label}")


if __name__ == "__main__":
    main()
