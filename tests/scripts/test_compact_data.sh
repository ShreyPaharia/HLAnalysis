#!/bin/bash
# Fixture test for the two-stage compact-data.sh. Uses a real-duckdb-backed
# `duckdb` CLI shim (python duckdb) so merges actually run and rows are verified.
#
# Asserts:
#   PASS 1 (disk leanness): a sealed hour's multiple parquets are merged into
#     one hour=HH/compacted.parquet; the in-progress (current UTC) hour is left
#     as raw files.
#   PASS 2 (S3 leanness): a sealed DAY's hours are rolled into ONE
#     hour=all/compacted.parquet and the hour=HH dirs are deleted; row count
#     preserved.
#   The in-progress day is never rolled up to hour=all.
#   Reentrant: a second run is a no-op.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
FIXTURE=$(mktemp -d)
SHIM=$(mktemp -d)
trap 'rm -rf "$FIXTURE" "$SHIM"' EXIT

DATE_BIN=$(command -v gdate || command -v date)

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

mk_parquet() {  # mk_parquet <dir> <filename> <nrows>
  mkdir -p "$1"
  uv run --quiet --project "$REPO" python - "$1/$2" "$3" <<'PY'
import sys, pandas as pd
path, n = sys.argv[1], int(sys.argv[2])
pd.DataFrame({"ts": range(n), "px": [1.0]*n}).to_parquet(path)
PY
}
count_files() { find "$1" -maxdepth 1 -type f -name '*.parquet' 2>/dev/null | grep -c . || true; }
count_rows() {
  uv run --quiet --project "$REPO" python - "$1" <<'PY'
import sys, duckdb
print(duckdb.connect().execute(f"SELECT count(*) FROM read_parquet('{sys.argv[1]}')").fetchone()[0])
PY
}

PART="$FIXTURE/venue=test/product_type=perp/mechanism=clob/event=trade/symbol=BTC"
YDAY=$("$DATE_BIN" -u -d "1 day ago" +%Y-%m-%d)
TODAY=$("$DATE_BIN" -u +%Y-%m-%d)
NOW_HOUR=$("$DATE_BIN" -u +%H)

# Sealed day: hour=05 has 2 raw files (3+2 rows), hour=09 has 1 file (4 rows).
mk_parquet "$PART/date=$YDAY/hour=05" 000.parquet 3
mk_parquet "$PART/date=$YDAY/hour=05" 001.parquet 2
mk_parquet "$PART/date=$YDAY/hour=09" 000.parquet 4

# In-progress hour today: 2 raw files, must remain untouched.
mk_parquet "$PART/date=$TODAY/hour=$NOW_HOUR" 000.parquet 1
mk_parquet "$PART/date=$TODAY/hour=$NOW_HOUR" 001.parquet 1

# A sealed hour today (only when NOW_HOUR != 00 so it doesn't collide with the
# in-progress hour): Pass 1 must compact it to a single file, with NO hour=all.
TEST_TODAY_SEAL=0
if [ "$NOW_HOUR" != "00" ]; then
  TEST_TODAY_SEAL=1
  mk_parquet "$PART/date=$TODAY/hour=00" 000.parquet 1
  mk_parquet "$PART/date=$TODAY/hour=00" 001.parquet 1
fi

run() { DATA_ROOT="$FIXTURE" DUCKDB_BIN="duckdb" LOOKBACK_DAYS=2 bash "$REPO/scripts/compact-data.sh"; }

fail=0
run

# PASS 2: yesterday rolled up to a single hour=all, hours gone, rows preserved.
SENT="$PART/date=$YDAY/hour=all/compacted.parquet"
[ -f "$SENT" ] || { echo "FAIL: expected $SENT"; fail=1; }
[ -d "$PART/date=$YDAY/hour=05" ] && { echo "FAIL: hour=05 should be gone"; fail=1; }
[ -d "$PART/date=$YDAY/hour=09" ] && { echo "FAIL: hour=09 should be gone"; fail=1; }
if [ -f "$SENT" ]; then
  r=$(count_rows "$SENT"); [ "$r" -eq 9 ] || { echo "FAIL: rolled rows=$r, expected 9"; fail=1; }
fi

# In-progress hour: still 2 raw files, untouched.
n=$(count_files "$PART/date=$TODAY/hour=$NOW_HOUR")
[ "$n" -eq 2 ] || { echo "FAIL: in-progress hour files=$n, expected 2"; fail=1; }
# Today never rolled up.
[ -d "$PART/date=$TODAY/hour=all" ] && { echo "FAIL: today must not have hour=all"; fail=1; }

# PASS 1 on today's sealed hour: compacted to one file (disk leanness).
if [ "$TEST_TODAY_SEAL" = "1" ]; then
  n=$(count_files "$PART/date=$TODAY/hour=00")
  [ "$n" -eq 1 ] || { echo "FAIL: today sealed hour files=$n, expected 1 (compacted)"; fail=1; }
fi

# Reentrant: second run does nothing new.
out=$(run); echo "$out"
echo "$out" | grep -q "hours=0 days=0" || { echo "FAIL: second run not a no-op: $out"; fail=1; }

if [ "$fail" -eq 0 ]; then
  echo "PASS: compact-data.sh two-stage fixture test"
else
  exit 1
fi
