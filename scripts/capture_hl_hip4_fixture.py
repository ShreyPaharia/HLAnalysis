"""Capture a small fixture slice of HL HIP-4 recorded parquet for tests.

Picks a single question's legs over a [start_ns, end_ns) window and copies the
matching hive partitions (date+hour buckets that intersect the window) under
``--out-root``, filtering rows by ``exchange_ts`` so files are tight.

Also copies the HL perp BTC bbo/mark partitions for the same window so the
fixture's reference-price stream is self-contained.

Usage:
    python scripts/capture_hl_hip4_fixture.py \
        --data-root data \
        --out-root tests/fixtures/hl_hip4 \
        --question-symbol Q1000015 \
        --start-ns 1778378400000000000 \
        --end-ns   1778385600000000000
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb


HL_PRED = "venue=hyperliquid/product_type=prediction_binary/mechanism=clob"
HL_PERP = "venue=hyperliquid/product_type=perp/mechanism=clob"


def _leg_for(outcome_idx: int, side_idx: int) -> str:
    return f"#{outcome_idx * 10 + side_idx}"


def _date_partitions(start_ns: int, end_ns: int) -> list[str]:
    start = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).date()
    end = datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc).date()
    out = []
    d = start
    while d <= end + timedelta(days=1):
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _question_legs(con: duckdb.DuckDBPyConnection, data_root: Path, qid: str) -> tuple[list[str], int]:
    glob = str(data_root / HL_PRED / "event=question_meta" / f"symbol={qid}" / "**" / "*.parquet")
    row = con.sql(
        f"""
        SELECT question_idx, named_outcome_idxs, fallback_outcome_idx
        FROM read_parquet('{glob}', hive_partitioning=1)
        ORDER BY exchange_ts LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise SystemExit(f"question_meta not found for {qid}")
    qidx, named, fb = row
    legs = []
    ordering = list(named)
    if fb is not None:
        ordering.append(int(fb))
    for o in ordering:
        legs.extend([_leg_for(int(o), 0), _leg_for(int(o), 1)])
    return legs, int(qidx)


def _copy_event_slice(
    con: duckdb.DuckDBPyConnection,
    src_glob: str,
    out_path: Path,
    *,
    start_ns: int,
    end_ns: int,
    date_list: list[str],
    extra_where: str = "",
) -> int:
    """Copy a row slice into a single parquet at ``out_path`` (zstd).

    Returns the row count. The fixture collapses each (event, symbol) to a single
    file; the data source reads via hive partitioning, which is the same path it
    would take against the real layout because we keep partition columns.
    """
    # check existence first
    import glob as _glob
    if not _glob.glob(src_glob, recursive=True):
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    date_clause = ",".join(repr(d) for d in date_list)
    where = f"date IN ({date_clause}) AND exchange_ts BETWEEN {start_ns} AND {end_ns}"
    if extra_where:
        where = f"{where} AND {extra_where}"
    con.sql(
        f"""
        COPY (
            SELECT *
            FROM read_parquet('{src_glob}', hive_partitioning=1)
            WHERE {where}
            ORDER BY exchange_ts
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    return con.sql(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]


def _question_meta_slice(
    con: duckdb.DuckDBPyConnection,
    src_glob: str,
    out_path: Path,
) -> int:
    """Question metadata is small; copy in full (no row filter)."""
    import glob as _glob
    if not _glob.glob(src_glob, recursive=True):
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.sql(
        f"""
        COPY (
            SELECT * FROM read_parquet('{src_glob}', hive_partitioning=1)
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    return con.sql(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--question-symbol", required=True, help="e.g. Q1000015")
    p.add_argument("--start-ns", required=True, type=int)
    p.add_argument("--end-ns", required=True, type=int)
    args = p.parse_args()

    data_root: Path = args.data_root
    out_root: Path = args.out_root
    qid: str = args.question_symbol
    start_ns: int = args.start_ns
    end_ns: int = args.end_ns

    con = duckdb.connect()
    legs, qidx = _question_legs(con, data_root, qid)
    print(f"question={qid} idx={qidx} legs={legs}")
    print(f"window: {start_ns}..{end_ns}")
    date_list = _date_partitions(start_ns, end_ns)

    # question_meta (copy all rows for the question — small)
    n_qm = _question_meta_slice(
        con,
        str(data_root / HL_PRED / "event=question_meta" / f"symbol={qid}" / "**" / "*.parquet"),
        out_root / HL_PRED / "event=question_meta" / f"symbol={qid}" / "fixture.parquet",
    )
    print(f"  question_meta: {n_qm} rows")

    # per-leg slices
    total_rows = 0
    for leg in legs:
        # market_meta (no exchange_ts filter — small)
        nm = _question_meta_slice(
            con,
            str(data_root / HL_PRED / "event=market_meta" / f"symbol={leg}" / "**" / "*.parquet"),
            out_root / HL_PRED / "event=market_meta" / f"symbol={leg}" / "fixture.parquet",
        )
        nb = _copy_event_slice(
            con,
            str(data_root / HL_PRED / "event=book_snapshot" / f"symbol={leg}" / "**" / "*.parquet"),
            out_root / HL_PRED / "event=book_snapshot" / f"symbol={leg}" / "fixture.parquet",
            start_ns=start_ns, end_ns=end_ns, date_list=date_list,
        )
        nt = _copy_event_slice(
            con,
            str(data_root / HL_PRED / "event=trade" / f"symbol={leg}" / "**" / "*.parquet"),
            out_root / HL_PRED / "event=trade" / f"symbol={leg}" / "fixture.parquet",
            start_ns=start_ns, end_ns=end_ns, date_list=date_list,
        )
        ns = _copy_event_slice(
            con,
            str(data_root / HL_PRED / "event=settlement" / f"symbol={leg}" / "**" / "*.parquet"),
            out_root / HL_PRED / "event=settlement" / f"symbol={leg}" / "fixture.parquet",
            start_ns=start_ns, end_ns=end_ns, date_list=date_list,
        )
        total_rows += nm + nb + nt + ns
        print(f"  {leg}: meta={nm} book={nb} trade={nt} settle={ns}")

    # HL perp BTC reference price (bbo + mark for robustness)
    for evt in ("bbo", "mark"):
        nr = _copy_event_slice(
            con,
            str(data_root / HL_PERP / f"event={evt}" / "symbol=BTC" / "**" / "*.parquet"),
            out_root / HL_PERP / f"event={evt}" / "symbol=BTC" / "fixture.parquet",
            start_ns=start_ns, end_ns=end_ns, date_list=date_list,
        )
        total_rows += nr
        print(f"  perp BTC {evt}: {nr} rows")

    print(f"total rows: {total_rows}")
    # bytes
    sz = sum(f.stat().st_size for f in out_root.rglob("*.parquet"))
    print(f"fixture size: {sz/1024:.1f} KB ({sz/1024/1024:.2f} MB)")


if __name__ == "__main__":
    main()
