#!/bin/bash
# Push recently-sealed /data partitions to S3.
#
# Cost design: only sealed-DAY partitions are uploaded — one merged file per
# day under the `hour=all` sentinel that compact-data.sh produces. Raw hour=HH
# minute-files (and the in-progress day) stay EBS-local and never reach S3.
# Each day is synced at the date prefix with --delete, so the upload both pushes
# hour=all AND removes any legacy hour=HH objects left in S3 for that day — the
# archive ends up hour=all-only per sealed day (no double-counted rows on read).
# Only partitions modified since the last successful run are touched (tracked via
# a marker file). This keeps S3 LIST cost O(partitions changed this run) instead
# of O(total objects in the bucket), and the per-day granularity keeps the object
# count ~24x lower than hourly uploads.
#
# The previous implementation ran `aws s3 sync $DATA_ROOT/ s3://$BUCKET/` with
# client-side --include filters. The CLI lists the ENTIRE destination bucket
# before applying those filters, so LIST cost grew without bound as the archive
# filled (it dominated the monthly S3 request bill). The --include scope only
# narrowed what was uploaded, never what was listed.
#
# Required env: ARCHIVE_BUCKET
# Optional env: DATA_ROOT (default /data)
#               DAYS_BACK (default 2 — bounds the partition walk so it stays
#                 fast as the volume grows; covers the most recent sealed days,
#                 with a one-day margin so a missed run still gets uploaded)
#               SYNC_MARKER (default $DATA_ROOT/.s3-sync-marker)

set -euo pipefail

BUCKET="${ARCHIVE_BUCKET:?ARCHIVE_BUCKET must be set}"
DATA_ROOT="${DATA_ROOT:-/data}"
DAYS_BACK="${DAYS_BACK:-2}"
MARKER="${SYNC_MARKER:-$DATA_ROOT/.s3-sync-marker}"

if [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: $DATA_ROOT does not exist" >&2
  exit 1
fi

# Compact small parquets in sealed-hour partitions before the push, so each
# upload pushes one merged file per partition instead of dozens of tiny ones.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$SCRIPT_DIR/compact-data.sh" ]; then
  DATA_ROOT="$DATA_ROOT" LOOKBACK_DAYS="$DAYS_BACK" "$SCRIPT_DIR/compact-data.sh" || \
    echo "WARN: compact-data.sh exited non-zero; continuing with sync"
fi

# Stamp a fresh marker at the START of the run. We commit it (atomic rename)
# only after every partition sync succeeds, so a mid-run failure leaves the
# previous marker in place and the next run retries the same window. Because it
# is stamped at start, any partition written *during* this run is still newer
# than the committed marker and gets re-evaluated next run (re-upload is cheap:
# --size-only skips unchanged objects).
NEW_MARKER="$(mktemp "${MARKER}.XXXXXX")"

# Only consider partitions modified since the last successful run. On the first
# run (no marker) the predicate is empty and the full DAYS_BACK window is swept.
newer_pred=()
if [ -f "$MARKER" ]; then
  newer_pred=(-newer "$MARKER")
fi

# Candidate date= dirs to walk (bounds the find as the archive grows).
DATES=()
for offset in $(seq 0 "$DAYS_BACK"); do
  DATES+=("$(date -u -d "$offset days ago" +%Y-%m-%d)")
done

SYNCED=0
for d in "${DATES[@]}"; do
  while IFS= read -r datedir; do
    [ -z "$datedir" ] && continue
    # Only sync days already rolled up to the daily sentinel. This guard
    # excludes the in-progress day (no hour=all yet) and any day whose rollup
    # failed — we must never date-sync a directory still full of raw hour=HH
    # minute-files (that would both explode the object count and, with --delete,
    # be the wrong authoritative set).
    [ -f "$datedir/hour=all/compacted.parquet" ] || continue
    rel="${datedir#"$DATA_ROOT"/}"
    echo "==> sync $rel"
    # Date-level sync with --delete: pushes hour=all AND removes any legacy
    # hour=HH objects in S3 for this day, so the archive is hour=all-only per
    # sealed day. LIST is scoped to this one date prefix (~one day of objects),
    # never the whole bucket.
    aws s3 sync "$datedir/" "s3://$BUCKET/$rel/" \
      --size-only --no-progress --delete
    SYNCED=$((SYNCED + 1))
  done < <(find "$DATA_ROOT" -type d -path "*date=$d" ${newer_pred[@]+"${newer_pred[@]}"})
done

# Commit the marker now that all syncs succeeded (mv preserves the start-of-run
# mtime, which is what the next run's -newer test compares against).
mv -f "$NEW_MARKER" "$MARKER"

echo "==> sync complete: synced=$SYNCED"
