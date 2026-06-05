"""One-time backfill: re-compact historical S3 data from hourly to daily layout.

Going-forward, scripts/compact-data.sh merges each sealed day into a single
`date=D/hour=all/compacted.parquet`. This tool applies the same reduction to the
data already archived in S3 under the legacy `date=D/hour=HH/...` layout, so
historical backtests open ~24x fewer files and the bucket's object count (and
its LIST/transition costs) drops accordingly.

It is OPTIONAL — daily and hourly files coexist for reads (see
tests/unit/test_daily_compaction_layout.py), so nothing breaks if you never run
it. Recommended: run AFTER the going-forward change has been live a few days.

Run it ON THE EC2 BOX so the S3 traffic goes through the free gateway endpoint:

    # dry run (default) — prints the plan, touches nothing
    uv run python tools/backfill_daily_compaction.py --bucket hl-recorder-archive-819175935435

    # execute
    uv run python tools/backfill_daily_compaction.py --bucket ... --apply

Safety: for each day partition it merges ONLY the hour=HH source files (never an
existing hour=all, so re-runs can't double-count), verifies the merged row count
equals the sum of the sources, and only THEN deletes the hourly objects. A
mismatch aborts that partition with the sources left intact.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import re
import sys
from collections import defaultdict
from collections.abc import Iterable

_DATE_RE = re.compile(r"(?P<prefix>.*/date=(?P<date>\d{4}-\d{2}-\d{2}))/hour=(?P<hour>[^/]+)/[^/]+\.parquet$")


@dataclasses.dataclass
class DayPartition:
    """A single (stream, date) partition eligible for hourly->daily merge."""

    date_prefix: str          # ".../symbol=X/date=YYYY-MM-DD"
    date: str                 # "YYYY-MM-DD"
    sources: list[str]        # hour=HH/*.parquet keys (HH != "all")

    @property
    def target_key(self) -> str:
        return f"{self.date_prefix}/hour=all/compacted.parquet"


def plan_migration(keys: Iterable[str], today: str) -> list[DayPartition]:
    """Pure: group parquet keys into day partitions needing an hourly->daily merge.

    A partition is included iff it has at least one `hour=HH` (HH != "all")
    source file and its date is strictly before `today` (never touch the
    in-progress day). Already-migrated partitions (only `hour=all`) yield no
    sources and are skipped. Sources are returned sorted for deterministic SQL.
    """
    by_partition: dict[str, list[str]] = defaultdict(list)
    dates: dict[str, str] = {}
    for key in keys:
        m = _DATE_RE.match(key)
        if not m:
            continue
        if m.group("hour") == "all":
            continue  # never fold an existing daily file back into the merge
        prefix = m.group("prefix")
        by_partition[prefix].append(key)
        dates[prefix] = m.group("date")

    plan: list[DayPartition] = []
    for prefix, srcs in by_partition.items():
        date = dates[prefix]
        if date >= today:
            continue  # skip the in-progress (or future-dated) day
        plan.append(DayPartition(date_prefix=prefix, date=date, sources=sorted(srcs)))
    plan.sort(key=lambda p: (p.date, p.date_prefix))
    return plan


# --------------------------------------------------------------------------
# Execution (I/O — exercised on the box, not in unit tests)
# --------------------------------------------------------------------------

def _list_keys(s3, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                keys.append(obj["Key"])
    return keys


def _duck(region: str, mem_limit: str = "256MB"):
    import duckdb

    con = duckdb.connect()
    # Under SSM/root there is no $HOME, so httpfs install/cache has nowhere to
    # live; point DuckDB at a writable dir explicitly.
    con.execute("SET home_directory='/tmp';")
    # Cap memory so a merge never OOM-competes with the co-located live engine
    # on the 1GB box; DuckDB spills to disk if a partition ever needs more.
    con.execute(f"SET memory_limit='{mem_limit}';")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(
        f"CREATE SECRET s3backfill (TYPE S3, PROVIDER credential_chain, REGION '{region}');"
    )
    return con


def _merge_partition(con, bucket: str, part: DayPartition) -> tuple[int, int]:
    """Merge sources -> hour=all, return (source_rows, merged_rows). No deletes."""
    src_list = ", ".join(f"'s3://{bucket}/{k}'" for k in part.sources)
    target = f"s3://{bucket}/{part.target_key}"
    src_rows = con.execute(
        f"SELECT count(*) FROM read_parquet([{src_list}])"
    ).fetchone()[0]
    con.execute(
        f"COPY (SELECT * FROM read_parquet([{src_list}])) "
        f"TO '{target}' (FORMAT 'PARQUET', COMPRESSION 'ZSTD');"
    )
    merged_rows = con.execute(
        f"SELECT count(*) FROM read_parquet('{target}')"
    ).fetchone()[0]
    return src_rows, merged_rows


def _delete(s3, bucket: str, keys: list[str]) -> None:
    for i in range(0, len(keys), 1000):
        batch = keys[i : i + 1000]
        s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="", help="restrict to a key prefix (e.g. venue=binance/)")
    ap.add_argument("--region", default="ap-northeast-1")
    ap.add_argument("--apply", action="store_true", help="execute (default: dry run)")
    ap.add_argument("--limit", type=int, default=0, help="cap partitions processed (0 = all)")
    ap.add_argument("--mem-limit", default="256MB", help="DuckDB memory cap (gentle on the live box)")
    args = ap.parse_args(argv)

    import boto3

    s3 = boto3.client("s3", region_name=args.region)
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    print(f"listing s3://{args.bucket}/{args.prefix} ...", file=sys.stderr)
    keys = _list_keys(s3, args.bucket, args.prefix)
    plan = plan_migration(keys, today)
    if args.limit:
        plan = plan[: args.limit]

    objs_before = sum(len(p.sources) for p in plan)
    print(
        f"plan: {len(plan)} day-partitions, {objs_before} hourly objects -> "
        f"{len(plan)} daily objects (delete {objs_before - len(plan)} objects)"
    )
    if not args.apply:
        for p in plan[:20]:
            print(f"  DRY  {p.date_prefix}  ({len(p.sources)} hourly files)")
        if len(plan) > 20:
            print(f"  ... and {len(plan) - 20} more")
        print("\n(dry run — pass --apply to execute)")
        return 0

    con = _duck(args.region, args.mem_limit)
    done = failed = 0
    for p in plan:
        try:
            src_rows, merged_rows = _merge_partition(con, args.bucket, p)
            if src_rows != merged_rows:
                print(f"  SKIP {p.date_prefix}: row mismatch src={src_rows} merged={merged_rows}")
                failed += 1
                continue
            _delete(s3, args.bucket, p.sources)
            done += 1
            print(f"  OK   {p.date_prefix}: {len(p.sources)} files -> 1 ({merged_rows} rows)")
        except Exception as e:  # noqa: BLE001 — one bad partition must not abort the run
            print(f"  FAIL {p.date_prefix}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\ndone: migrated={done} failed/skipped={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
