#!/bin/bash
# Pull all archived parquet data from S3 to a local directory.
# Idempotent and incremental.
#
# Optional env: DEST (default ./data), STACK_NAME (default HLRecorderStack)

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

echo "==> pulling s3://$BUCKET/ -> $DEST/"
aws s3 sync "s3://$BUCKET/" "$DEST/" \
  --exclude 'logs/*' \
  --size-only

echo "==> done"
du -sh "$DEST"/venue=* 2>/dev/null || true
