#!/bin/bash
# Integration test for sync-to-s3.sh (daily compaction + incremental upload).
# Verifies the cost-critical behavior:
#   1. Compaction runs, then ONLY the sealed-day hour=all partition is uploaded,
#      scoped to its exact prefix (NOT a full-bucket `aws s3 sync`, and NOT the
#      raw hour=HH scratch).
#   2. The in-progress (current UTC date) day is never uploaded.
#   3. A second run with no new data re-uploads nothing (incremental marker).
#
# `aws` is stubbed (logs args). `duckdb` is REAL (python-backed shim) so the
# compact step actually merges the fixtures and the test exercises the true flow.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
FIXTURE=$(mktemp -d)
SHIM=$(mktemp -d)
AWS_LOG="$FIXTURE/aws.log"
trap 'rm -rf "$FIXTURE" "$SHIM"' EXIT

DATE_BIN=$(command -v gdate || command -v date)

# --- stubs -----------------------------------------------------------------
cat > "$SHIM/aws" <<EOF
#!/bin/bash
echo "\$*" >> "$AWS_LOG"
exit 0
EOF
chmod +x "$SHIM/aws"

cat > "$SHIM/duckdb" <<EOF
#!/bin/bash
if [ "\$1" = "-c" ]; then
  exec uv run --quiet --project "$REPO" python -c 'import duckdb,sys; duckdb.connect().execute(sys.argv[1])' "\$2"
fi
exit 1
EOF
chmod +x "$SHIM/duckdb"

if command -v gdate >/dev/null 2>&1; then
  cat > "$SHIM/date" <<'EOF'
#!/bin/bash
exec gdate "$@"
EOF
  chmod +x "$SHIM/date"
fi
export PATH="$SHIM:$PATH"

mk_parquet() {
  local dir="$1" n="$2"
  mkdir -p "$dir"
  uv run --quiet --project "$REPO" python - "$dir/data.parquet" "$n" <<'PY'
import sys, pandas as pd
path, n = sys.argv[1], int(sys.argv[2])
pd.DataFrame({"ts": range(n), "px": [1.0]*n}).to_parquet(path)
PY
}

# --- fixtures --------------------------------------------------------------
PREFIX="venue=test/product_type=perp/mechanism=clob/event=trades/symbol=BTC"
YDAY=$("$DATE_BIN" -u -d "1 day ago" +%Y-%m-%d)
TODAY=$("$DATE_BIN" -u +%Y-%m-%d)
NOW_HOUR=$("$DATE_BIN" -u +%H)

# Sealed day with two raw hours — compaction should merge to hour=all, then
# sync uploads only that.
mk_parquet "$FIXTURE/$PREFIX/date=$YDAY/hour=05" 3
mk_parquet "$FIXTURE/$PREFIX/date=$YDAY/hour=09" 4
# In-progress day — must never be uploaded.
mk_parquet "$FIXTURE/$PREFIX/date=$TODAY/hour=$NOW_HOUR" 2

run_sync() {
  ARCHIVE_BUCKET="test-bucket" DATA_ROOT="$FIXTURE" DAYS_BACK=2 DUCKDB_BIN="duckdb" \
    bash "$REPO/scripts/sync-to-s3.sh"
}

fail=0
expect()    { grep -q -- "$1" "$AWS_LOG" || { echo "FAIL: expected aws call matching: $1"; fail=1; }; }
expect_no() { grep -q -- "$1" "$AWS_LOG" && { echo "FAIL: unexpected aws call matching: $1"; fail=1; } || true; }

# --- run 1 -----------------------------------------------------------------
: > "$AWS_LOG"
run_sync

# Date-level sync with --delete: pushes hour=all and prunes legacy hour=HH in S3.
expect "s3 sync $FIXTURE/$PREFIX/date=$YDAY/ s3://test-bucket/$PREFIX/date=$YDAY/ --size-only --no-progress --delete"
expect_no "s3://test-bucket/ "                 # no full-bucket sync
expect_no "sync $FIXTURE/ s3://test-bucket/"   # no full-tree sync
expect_no "date=$TODAY"                        # in-progress day never uploaded

# --- run 2: nothing changed -> incremental no-op ---------------------------
: > "$AWS_LOG"
run_sync
if [ -s "$AWS_LOG" ]; then
  echo "FAIL: second run re-uploaded with no new data:"; cat "$AWS_LOG"; fail=1
fi

if [ "$fail" -eq 0 ]; then
  echo "PASS: sync-to-s3.sh daily fixture test"
else
  exit 1
fi
