#!/bin/bash
# Fixture test for cleanup-ebs.sh: build a fake /data tree with
# partitions of varying ages, run cleanup with SKIP_SYNC=1 and
# SKIP_S3_CHECK=1, verify the right partitions are deleted.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
FIXTURE=$(mktemp -d)
trap 'rm -rf "$FIXTURE"' EXIT

# Use GNU date if available (gdate on macOS via coreutils), else assume Linux date
DATE_BIN=$(command -v gdate || command -v date)

make_partition() {
  local days_ago="$1"
  local d
  d=$("$DATE_BIN" -u -d "$days_ago days ago" +%Y-%m-%d)
  local part="$FIXTURE/venue=test/product_type=perp/mechanism=clob/event=trades/symbol=BTC/date=$d"
  mkdir -p "$part/hour=00"
  echo "fixture" > "$part/hour=00/test.parquet"
}

for days in 0 1 2 4 10; do make_partition "$days"; done

# Override `date` for the cleanup script's own use (it also calls `date -u -d "N days ago"`).
# If gdate exists, prepend a shim dir to PATH that aliases date->gdate.
if command -v gdate >/dev/null 2>&1; then
  SHIM=$(mktemp -d)
  trap 'rm -rf "$FIXTURE" "$SHIM"' EXIT
  cat > "$SHIM/date" <<EOF
#!/bin/bash
exec gdate "\$@"
EOF
  chmod +x "$SHIM/date"
  export PATH="$SHIM:$PATH"
fi

ARCHIVE_BUCKET="test-bucket" \
DATA_ROOT="$FIXTURE" \
RETENTION_DAYS=3 \
SKIP_SYNC=1 \
SKIP_S3_CHECK=1 \
bash "$SCRIPT_DIR/scripts/cleanup-ebs.sh"

KEPT_DAYS=(0 1 2)
GONE_DAYS=(4 10)

fail=0
for days in "${KEPT_DAYS[@]}"; do
  d=$("$DATE_BIN" -u -d "$days days ago" +%Y-%m-%d)
  if [ ! -d "$FIXTURE/venue=test/product_type=perp/mechanism=clob/event=trades/symbol=BTC/date=$d" ]; then
    echo "FAIL: partition date=$d should have been kept (days_ago=$days)"
    fail=1
  fi
done

for days in "${GONE_DAYS[@]}"; do
  d=$("$DATE_BIN" -u -d "$days days ago" +%Y-%m-%d)
  if [ -d "$FIXTURE/venue=test/product_type=perp/mechanism=clob/event=trades/symbol=BTC/date=$d" ]; then
    echo "FAIL: partition date=$d should have been deleted (days_ago=$days)"
    fail=1
  fi
done

if [ "$fail" -eq 0 ]; then
  echo "PASS: cleanup-ebs.sh fixture test"
else
  exit 1
fi
