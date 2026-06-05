#!/bin/bash
# Two-stage compaction of the per-minute parquet files written by the recorder.
#
#   PASS 1 — hourly, for DISK leanness:
#     Merge each SEALED hour's minute-files into hour=HH/compacted.parquet.
#     Runs for today's already-elapsed hours too, so the EBS working set never
#     holds more than the single in-progress hour as raw minute-files. The
#     current UTC hour is skipped (still being written).
#
#   PASS 2 — daily, for S3 leanness:
#     Once a DAY is sealed (date < today UTC), merge its hour=*/compacted.parquet
#     into ONE hour=all/compacted.parquet and delete the hour=HH dirs.
#     sync-to-s3.sh uploads ONLY hour=all, so the archive holds ~1 object per
#     day per stream while the on-disk layout stays hour-compacted.
#
# Why upload cadence and disk usage are decoupled: disk is bounded by Pass 1
# (always hour-compacted) + cleanup-ebs retention, NOT by how often we push to
# S3. The daily rollup only changes the S3 object granularity.
#
# Why `hour=all` and not "drop the hour= level": every analysis/backtest reader
# globs `.../date=*/hour=*/*.parquet` and reads `hour=` as a DuckDB hive
# partition column. A daily file with NO hour= dir makes DuckDB raise
# "Hive partition mismatch" next to legacy hourly data. The sentinel keeps old
# hourly and new daily files readable together with zero reader changes (the
# `hour` column is just the string "all", which nothing consumes).
#
# Reentrant + atomic throughout: a partition already in its target shape is
# skipped; merges write compacted.parquet.tmp then mv over the final name; if
# duckdb fails the partition is left untouched.
#
# Optional env: DATA_ROOT (default /data), DUCKDB_BIN (default /usr/local/bin/duckdb)
#               LOOKBACK_DAYS (default 2 — bounds the walk so it stays fast as
#               the EBS volume grows; covers the most recent date= partitions).

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data}"
DUCKDB_BIN="${DUCKDB_BIN:-/usr/local/bin/duckdb}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-2}"
SENTINEL="hour=all"

if [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: $DATA_ROOT does not exist" >&2
  exit 1
fi
if ! command -v "$DUCKDB_BIN" >/dev/null 2>&1; then
  echo "ERROR: duckdb not found at $DUCKDB_BIN" >&2
  exit 1
fi

NOW_DATE=$(date -u +%Y-%m-%d)
NOW_HOUR=$(date -u +%H)

# today + the previous LOOKBACK_DAYS.
DATES=()
for offset in $(seq 0 "$LOOKBACK_DAYS"); do
  DATES+=("$(date -u -d "$offset days ago" +%Y-%m-%d)")
done

# merge_parquet <output_file> <read_glob> — atomic duckdb COPY. Returns 0 on a
# committed merge, 1 on failure (output left untouched). The .tmp suffix is not
# matched by '*.parquet' so an existing target is never read mid-write.
merge_parquet() {
  local out="$1" glob="$2" tmp="$1.tmp"
  # union_by_name aligns columns by name across files so a stream whose schema
  # drifted between hours (e.g. a Polymarket column added/dropped mid-day) still
  # merges — the same promotion the readers do (read_recorded promote_options).
  if "$DUCKDB_BIN" -c \
    "COPY (SELECT * FROM read_parquet('$glob', union_by_name=true)) TO '$tmp' (FORMAT 'PARQUET', COMPRESSION 'ZSTD');"
  then
    if [ ! -s "$tmp" ]; then rm -f "$tmp"; return 1; fi
    mv -f "$tmp" "$out"
    return 0
  fi
  rm -f "$tmp"
  return 1
}

COMPACTED_HOURS=0
ROLLED_DAYS=0
SKIPPED=0
FAILED=0

# ── PASS 1: hourly compaction (disk leanness) ──────────────────────────────
for d in "${DATES[@]}"; do
  while IFS= read -r hourdir; do
    [ -z "$hourdir" ] && continue
    hour_part="${hourdir##*hour=}"
    [ "$hour_part" = "all" ] && continue            # never re-compact the daily rollup

    # Skip the in-progress hour (still being written).
    if [ "$d" = "$NOW_DATE" ] && [ "$hour_part" = "$NOW_HOUR" ]; then
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    count=0
    while IFS= read -r f; do [ -n "$f" ] && count=$((count + 1)); done \
      < <(find "$hourdir" -maxdepth 1 -type f -name '*.parquet')
    if [ "$count" -le 1 ]; then                     # already compact (or empty)
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    if merge_parquet "$hourdir/compacted.parquet" "$hourdir/*.parquet"; then
      # Delete the source minute-files (everything except the merged file).
      while IFS= read -r f; do
        [ -z "$f" ] && continue
        [ "$f" = "$hourdir/compacted.parquet" ] && continue
        rm -f "$f"
      done < <(find "$hourdir" -maxdepth 1 -type f -name '*.parquet')
      COMPACTED_HOURS=$((COMPACTED_HOURS + 1))
    else
      FAILED=$((FAILED + 1))
      echo "compact (hour) failed for ${hourdir#"$DATA_ROOT"/}" >&2
    fi
  done < <(find "$DATA_ROOT" -type d -path "*date=$d/hour=*" -name 'hour=*')
done

# ── PASS 2: daily rollup of SEALED days (S3 leanness) ──────────────────────
for d in "${DATES[@]}"; do
  [ "$d" = "$NOW_DATE" ] && continue                 # never roll up the in-progress day

  while IFS= read -r datedir; do
    [ -z "$datedir" ] && continue
    rel="${datedir#"$DATA_ROOT"/}"
    sentinel_file="$datedir/$SENTINEL/compacted.parquet"

    # Source files = every hour=HH parquet (HH != all). Counted portably.
    count=0
    while IFS= read -r f; do
      [ -z "$f" ] && continue
      case "$f" in "$datedir/$SENTINEL/"*) continue ;; esac
      count=$((count + 1))
    done < <(find "$datedir" -type f -path "*hour=*/*.parquet")

    if [ "$count" -eq 0 ]; then                      # already rolled up (or empty)
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    mkdir -p "$datedir/$SENTINEL"
    # Read all hour=HH files; the sentinel dir is excluded from the glob below
    # because the hour-glob matches hour=<something>/... and we delete only
    # hour=HH dirs afterward. To avoid folding a prior hour=all back in, read
    # the explicit hour=HH glob (HH is one or two digits — never "all").
    if merge_parquet "$sentinel_file" "$datedir/hour=[0-9]*/*.parquet"; then
      while IFS= read -r hourdir; do
        [ -z "$hourdir" ] && continue
        case "$hourdir" in "$datedir/$SENTINEL") continue ;; esac
        rm -rf "$hourdir"
      done < <(find "$datedir" -mindepth 1 -maxdepth 1 -type d -name 'hour=*')
      ROLLED_DAYS=$((ROLLED_DAYS + 1))
    else
      FAILED=$((FAILED + 1))
      echo "compact (day) failed for $rel" >&2
    fi
  done < <(find "$DATA_ROOT" -type d -path "*date=$d")
done

echo "==> compact: hours=$COMPACTED_HOURS days=$ROLLED_DAYS skipped=$SKIPPED failed=$FAILED"
