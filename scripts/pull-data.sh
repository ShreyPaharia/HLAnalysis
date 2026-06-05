#!/bin/bash
# Pull archived parquet data from S3 to a local directory for backtesting.
#
# The local copy is kept in the DAILY layout (one date=*/hour=all/compacted.parquet
# per sealed day) — see scripts/compact-data.sh. By default this pulls ONLY the
# daily (hour=all) objects, so it tops up new days without re-downloading the
# legacy hourly objects that still exist in S3 for historical dates (which would
# either duplicate or, with --delete, clobber the locally-compacted daily files).
#
# One-time setup / resync: run with MIRROR=1 to make the local tree exactly
# match S3 (hourly + daily, --delete prunes local orphans). Follow it with
#   LOOKBACK_DAYS=400 scripts/compact-data.sh
# to roll the freshly-mirrored hourly data into the local daily layout.
#
# Optional env: DEST (default ./data), STACK_NAME (default HLRecorderStack),
#               MIRROR=1 (full --delete mirror instead of daily-only top-up)

set -euo pipefail

STACK_NAME="${STACK_NAME:-HLRecorderStack}"
DEST="${DEST:-./data}"

BUCKET=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?ExportName=='HLRecorderArchiveBucket'].OutputValue" \
  --output text)

if [ -z "$BUCKET" ] || [ "$BUCKET" = "None" ]; then
  echo "ERROR: Could not resolve archive bucket from stack $STACK_NAME" >&2
  exit 1
fi

mkdir -p "$DEST"

if [ "${MIRROR:-0}" = "1" ]; then
  echo "==> MIRROR: full sync s3://$BUCKET/ -> $DEST/ (--delete prunes local orphans)"
  aws s3 sync "s3://$BUCKET/" "$DEST/" \
    --exclude 'logs/*' \
    --delete \
    --size-only
else
  echo "==> pulling daily (hour=all) data s3://$BUCKET/ -> $DEST/"
  aws s3 sync "s3://$BUCKET/" "$DEST/" \
    --exclude '*' \
    --include '*/hour=all/*' \
    --size-only
fi

echo "==> done"
du -sh "$DEST"/venue=* 2>/dev/null || true
