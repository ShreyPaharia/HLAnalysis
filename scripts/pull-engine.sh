#!/bin/bash
# Pull the daily engine snapshots (state.db.gz, gate_decisions.jsonl.gz, filtered
# logs) from S3 to a local directory — the SSM-free replacement for dumping venue
# user_fills via send-command. Counterpart to scripts/sync-engine-to-s3.sh.
#
# Lands the `engine/` prefix only (additive to scripts/pull-data.sh, which pulls
# the recorder's market-data partitions). Idempotent and incremental.
#
#   scripts/pull-engine.sh                 # -> ./data/engine/
#   DEST=/tmp/desk scripts/pull-engine.sh  # -> /tmp/desk/engine/
#
# Then inspect a slot for a date with:
#   scripts/inspect_engine_state.py ./data/engine --date 2026-06-10 --alias v1
#
# Optional env: DEST (default ./data), STACK_NAME (default HLRecorderStack),
#               S3_ENGINE_PREFIX (default engine)

set -euo pipefail

STACK_NAME="${STACK_NAME:-HLRecorderStack}"
DEST="${DEST:-./data}"
S3_ENGINE_PREFIX="${S3_ENGINE_PREFIX:-engine}"

BUCKET=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?ExportName=='HLRecorderArchiveBucket'].OutputValue" \
  --output text)

if [ -z "$BUCKET" ] || [ "$BUCKET" = "None" ]; then
  echo "ERROR: Could not resolve archive bucket from stack $STACK_NAME" >&2
  exit 1
fi

mkdir -p "$DEST/$S3_ENGINE_PREFIX"

echo "==> pulling engine snapshots s3://$BUCKET/$S3_ENGINE_PREFIX/ -> $DEST/$S3_ENGINE_PREFIX/"
aws s3 sync "s3://$BUCKET/$S3_ENGINE_PREFIX/" "$DEST/$S3_ENGINE_PREFIX/" \
  --size-only

echo "==> done"
du -sh "$DEST/$S3_ENGINE_PREFIX" 2>/dev/null || true
