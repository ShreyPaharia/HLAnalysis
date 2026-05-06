#!/bin/bash
# Push recently-written /data partitions to S3.
# Scoped to the last DAYS_BACK days so LIST cost stays flat as the bucket grows.
#
# Required env: ARCHIVE_BUCKET
# Optional env: DATA_ROOT (default /data), DAYS_BACK (default 1)

set -euo pipefail

BUCKET="${ARCHIVE_BUCKET:?ARCHIVE_BUCKET must be set}"
DATA_ROOT="${DATA_ROOT:-/data}"
DAYS_BACK="${DAYS_BACK:-1}"

if [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: $DATA_ROOT does not exist" >&2
  exit 1
fi

for offset in $(seq 0 "$DAYS_BACK"); do
  d=$(date -u -d "$offset days ago" +%Y-%m-%d)
  echo "==> sync date=$d"
  aws s3 sync "$DATA_ROOT/" "s3://$BUCKET/" \
    --size-only \
    --no-progress \
    --exclude '*' \
    --include "*/date=$d/*"
done

echo "==> sync complete"
