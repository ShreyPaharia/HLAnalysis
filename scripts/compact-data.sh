#!/bin/bash
# Compact small parquet files written by the live recorder into one file per
# sealed-hour partition. Runs from the hourly hl-recorder-sync.service before
# the S3 sync, so each upload pushes one merged file per partition instead of
# hundreds of tiny ones.
#
# - Only sealed hours are touched (current UTC hour is skipped).
# - Reentrant: a partition that already has a single `compacted.parquet`
#   (or only one parquet in any name) is left alone.
# - Atomic: writes `compacted.parquet.tmp`, then `mv` over the final name,
#   then deletes the originals. If duckdb fails the partition is untouched.
#
# Optional env: DATA_ROOT (default /data), DUCKDB_BIN (default /usr/local/bin/duckdb)
#               LOOKBACK_DAYS (default 2 — bound the walk so it stays fast as
#               the EBS volume grows; covers the most recent two date= partitions).

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data}"
DUCKDB_BIN="${DUCKDB_BIN:-/usr/local/bin/duckdb}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-2}"

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

# Build the list of date= directories to consider: today and the previous LOOKBACK_DAYS.
DATES=()
for offset in $(seq 0 "$LOOKBACK_DAYS"); do
  DATES+=("$(date -u -d "$offset days ago" +%Y-%m-%d)")
done

COMPACTED=0
SKIPPED=0
FAILED=0

# Iterate all hour=HH partitions under the recent date= partitions.
# Layout: $DATA_ROOT/venue=*/product_type=*/mechanism=*/event=*/symbol=*/date=YYYY-MM-DD/hour=HH/
for d in "${DATES[@]}"; do
  while IFS= read -r dir; do
    [ -z "$dir" ] && continue
    rel="${dir#"$DATA_ROOT"/}"
    hour_part="${rel##*hour=}"
    hour_part="${hour_part%%/*}"

    # Skip partition currently being written.
    if [ "$d" = "$NOW_DATE" ] && [ "$hour_part" = "$NOW_HOUR" ]; then
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    # Count parquet files; <=1 means already compact (or empty).
    count=$(find "$dir" -maxdepth 1 -name '*.parquet' -type f | wc -l | tr -d ' ')
    if [ "$count" -le 1 ]; then
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    # Compact via duckdb. Read all parquets in the partition (including any
    # previously-merged compacted.parquet, so a re-run is correct), write to
    # a tmp file, then atomically replace.
    if (
      cd "$dir" &&
      "$DUCKDB_BIN" -c "COPY (SELECT * FROM read_parquet('*.parquet')) TO 'compacted.parquet.tmp' (FORMAT 'PARQUET', COMPRESSION 'ZSTD');"
    ); then
      if [ ! -s "$dir/compacted.parquet.tmp" ]; then
        rm -f "$dir/compacted.parquet.tmp"
        FAILED=$((FAILED + 1))
        continue
      fi
      # Capture originals BEFORE the rename so we don't delete the new merged file.
      mapfile -t originals < <(find "$dir" -maxdepth 1 -name '*.parquet' -type f -not -name 'compacted.parquet.tmp')
      mv -f "$dir/compacted.parquet.tmp" "$dir/compacted.parquet"
      # Delete originals (excluding the freshly renamed compacted.parquet).
      for f in "${originals[@]}"; do
        # The previous compacted.parquet, if any, is in `originals`; remove it
        # only if it's not literally the new file we just renamed (it isn't,
        # because we used `mv -f` from a different tmp name).
        case "$f" in
          "$dir/compacted.parquet") continue ;;
        esac
        rm -f "$f"
      done
      COMPACTED=$((COMPACTED + 1))
    else
      rm -f "$dir/compacted.parquet.tmp"
      FAILED=$((FAILED + 1))
      echo "compact failed for $rel" >&2
    fi
  done < <(find "$DATA_ROOT" -type d -path "*date=$d/hour=*" -name 'hour=*')
done

echo "==> compact: compacted=$COMPACTED skipped=$SKIPPED failed=$FAILED"
