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

# Compact small parquets in sealed-hour partitions before the S3 push.
# `--delete` is then safe on the date-scoped sync below: local is authoritative
# and the scope is narrow.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$SCRIPT_DIR/compact-data.sh" ]; then
  DATA_ROOT="$DATA_ROOT" LOOKBACK_DAYS="$DAYS_BACK" "$SCRIPT_DIR/compact-data.sh" || \
    echo "WARN: compact-data.sh exited non-zero; continuing with sync"
fi

for offset in $(seq 0 "$DAYS_BACK"); do
  d=$(date -u -d "$offset days ago" +%Y-%m-%d)
  echo "==> sync date=$d"
  aws s3 sync "$DATA_ROOT/" "s3://$BUCKET/" \
    --size-only \
    --no-progress \
    --delete \
    --exclude '*' \
    --include "*/date=$d/*"
done

echo "==> sync complete"
