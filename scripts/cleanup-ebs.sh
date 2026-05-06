#!/bin/bash
# Daily cleanup of EBS data partitions older than RETENTION_DAYS.
# Safety: runs a full S3 sync first, then for each candidate partition
# verifies file-count parity before any deletion.
#
# Required env: ARCHIVE_BUCKET
# Optional env: DATA_ROOT (default /data), RETENTION_DAYS (default 3)
# Test-only env: SKIP_SYNC=1 (skip the upfront full-sync), SKIP_S3_CHECK=1
#                (skip S3 LIST verification, assume parity). Used by
#                tests/scripts/test_cleanup_ebs.sh; never set in production.

set -euo pipefail

BUCKET="${ARCHIVE_BUCKET:?ARCHIVE_BUCKET must be set}"
DATA_ROOT="${DATA_ROOT:-/data}"
RETENTION_DAYS="${RETENTION_DAYS:-3}"

if [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: $DATA_ROOT does not exist" >&2
  exit 1
fi

# Step 1: full sync first - cleanup must never delete unsynced data.
# Skip in test mode so unit tests don't require AWS credentials.
if [ "${SKIP_SYNC:-0}" != "1" ]; then
  echo "==> full sync to s3://$BUCKET/"
  aws s3 sync "$DATA_ROOT/" "s3://$BUCKET/" \
    --size-only --exclude 'logs/*' --no-progress
fi

# Step 2: walk date= partitions strictly older than RETENTION_DAYS.
CUTOFF=$(date -u -d "$RETENTION_DAYS days ago" +%Y-%m-%d)
DELETED=0
KEPT=0
SKIPPED=0

while IFS= read -r partition; do
  partition_date="${partition##*date=}"
  partition_date="${partition_date%%/*}"

  # Strictly older than cutoff (lexicographic comparison works for ISO dates)
  if [[ ! "$partition_date" < "$CUTOFF" ]]; then
    KEPT=$((KEPT + 1))
    continue
  fi

  # Step 3: verify same file count exists in S3 before deleting locally.
  # Assumes the recorder writes only *.parquet files (no _SUCCESS, _metadata,
  # etc.); revisit this filter if the writer ever emits sidecar files.
  rel="${partition#"$DATA_ROOT"/}"
  local_count=$(find "$partition" -type f -name '*.parquet' | wc -l | tr -d ' ')

  if [ "${SKIP_S3_CHECK:-0}" = "1" ]; then
    s3_count="$local_count"  # test mode: assume parity
  else
    s3_count=$(aws s3 ls "s3://$BUCKET/$rel/" --recursive 2>/dev/null \
               | grep -c '\.parquet$' || true)
  fi

  if [ "$local_count" -ne "$s3_count" ]; then
    echo "SKIP $partition: local=$local_count s3=$s3_count"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  rm -rf "$partition"
  DELETED=$((DELETED + 1))
done < <(find "$DATA_ROOT" -type d -path '*date=*' -not -path '*hour=*')

echo "==> cleanup complete: deleted=$DELETED kept=$KEPT skipped=$SKIPPED cutoff=$CUTOFF"
