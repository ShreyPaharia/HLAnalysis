#!/bin/bash
# Integration test for sync-engine-to-s3.sh. Runs the REAL script end-to-end in a
# sandbox (no AWS, no box) and verifies the load-bearing behaviour:
#   1. CONSISTENT snapshot: a state.db with an open uncommitted transaction (a
#      live WAL) is `.backup`-ed — the uploaded copy opens cleanly and contains
#      the COMMITTED rows only (not the in-flight write, not a torn file).
#   2. S3 KEY LAYOUT: engine/date=<DATE>/<alias>/{state.db.gz,gate_decisions.jsonl.gz}.
#   3. EXCLUSIONS: stale state.db.bak-*, -wal and -shm are never uploaded.
#   4. The top-level state.db is captured under the _root alias.
#   5. LOG FILTER: the filtered log keeps signal lines (WARN/halt/reject/…) and
#      drops the 1 Hz PnL-poll / heartbeat NOISE — i.e. not the 97 MB/day raw log.
#
# `aws` is never called (ENGINE_SYNC_S3_BASE points at a local dir). `sqlite3`
# and `gzip` are REAL so the snapshot path is genuinely exercised. `journalctl`
# is stubbed to emit a canned mix of signal + noise.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SANDBOX=$(mktemp -d)
trap 'rm -rf "$SANDBOX"' EXIT

ENGINE_ROOT="$SANDBOX/engine-data"
S3_LOCAL="$SANDBOX/s3"
SHIM="$SANDBOX/shim"
mkdir -p "$ENGINE_ROOT" "$S3_LOCAL" "$SHIM"
DATE="2026-06-10"

# --- stub journalctl: a realistic mix of signal and noise -------------------
cat > "$SHIM/journalctl" <<'EOF'
#!/bin/bash
cat <<'LINES'
Jun 10 06:00:01 box hl-engine[1]: INFO data/trades pnl=12.30 account_value=1000.0 poll
Jun 10 06:00:02 box hl-engine[1]: INFO heartbeat ok scanning markets
Jun 10 06:00:03 box hl-engine[1]: WARNING stale_data_halt symbol=BTC quiet book
Jun 10 06:00:04 box hl-engine[1]: ERROR order reject reason=insufficient balance
Jun 10 06:00:05 box hl-engine[1]: INFO data/trades pnl=12.31 account_value=1000.1 poll
Jun 10 06:00:06 box hl-engine[1]: CRITICAL restart_blocked drift detected on v31
Jun 10 06:00:07 box hl-engine[1]: INFO 🚨 FEED STALE no ticks 30s
LINES
EOF
chmod +x "$SHIM/journalctl"

# --- build a slot state.db with a live WAL (uncommitted write open) ----------
# A second connection holds an open transaction that INSERTs an extra row but
# never commits, leaving the db in WAL mode mid-write — exactly the torn-read
# hazard a raw cp would capture. `.backup` must yield committed rows only.
make_slot_db() {  # make_slot_db <db_path> <committed_symbol>
  local db="$1" sym="$2"
  sqlite3 "$db" <<SQL
PRAGMA journal_mode=WAL;
CREATE TABLE fill (fill_id TEXT PRIMARY KEY, symbol TEXT, side TEXT, price REAL,
                   size REAL, ts_ns INTEGER, closed_pnl REAL, source TEXT);
INSERT INTO fill VALUES ('f1','$sym','buy',0.42,100,1,0.0,'router');
INSERT INTO fill VALUES ('f2','$sym','sell',0.55,100,2,13.0,'router');
SQL
}

mkdir -p "$ENGINE_ROOT/v1" "$ENGINE_ROOT/v31_pm"
make_slot_db "$ENGINE_ROOT/v1/state.db" "BTC"
make_slot_db "$ENGINE_ROOT/v31_pm/state.db" "ETH"
# Top-level (un-namespaced) db -> _root alias.
make_slot_db "$ENGINE_ROOT/state.db" "ROOT"

# gate decision logs
printf '{"q":1,"decision":"enter"}\n{"q":2,"decision":"veto"}\n' > "$ENGINE_ROOT/v1/gate_decisions.jsonl"

# things that MUST be excluded
echo "stale" > "$ENGINE_ROOT/v1/state.db.bak-1700000000"
echo "wal"   > "$ENGINE_ROOT/v1/state.db-wal"
echo "shm"   > "$ENGINE_ROOT/v1/state.db-shm"

# Hold an uncommitted transaction open against v1's db for the duration of the
# sync, so the .backup runs against a genuinely live WAL.
python3 - "$ENGINE_ROOT/v1/state.db" <<'PY' &
import sqlite3, sys, time
c = sqlite3.connect(sys.argv[1], isolation_level=None)
c.execute("PRAGMA journal_mode=WAL;")
c.execute("BEGIN;")
c.execute("INSERT INTO fill VALUES ('uncommitted','BTC','buy',0.99,1,9,0.0,'router');")
time.sleep(5)   # never commits; process exits, rolling back
PY
WRITER_PID=$!
sleep 0.5  # let the writer open its transaction before we snapshot

# --- run the real script ----------------------------------------------------
PATH="$SHIM:$PATH" \
ARCHIVE_BUCKET="test-bucket" \
ENGINE_DATA_ROOT="$ENGINE_ROOT" \
ENGINE_SYNC_S3_BASE="$S3_LOCAL" \
SYNC_DATE="$DATE" \
LOG_UNIT="hl-engine" \
  bash "$REPO/scripts/sync-engine-to-s3.sh"

kill "$WRITER_PID" 2>/dev/null || true
wait "$WRITER_PID" 2>/dev/null || true

# --- assertions -------------------------------------------------------------
fail=0
BASE="$S3_LOCAL/date=$DATE"
exists()     { [ -f "$1" ] || { echo "FAIL: expected object missing: $1"; fail=1; }; }
not_exists() { [ -e "$1" ] && { echo "FAIL: unexpected object present: $1"; fail=1; } || true; }

# (2) key layout
exists "$BASE/v1/state.db.gz"
exists "$BASE/v1/gate_decisions.jsonl.gz"
exists "$BASE/v31_pm/state.db.gz"
exists "$BASE/_root/state.db.gz"          # (4) top-level db
exists "$BASE/engine/log-filtered.gz"

# (3) exclusions — only the two named files per slot, nothing else
not_exists "$BASE/v1/state.db.bak-1700000000.gz"
not_exists "$BASE/v1/state.db-wal.gz"
not_exists "$BASE/v1/state.db-shm.gz"
not_exists "$BASE/v1/state.db-wal"

# (1) consistent snapshot: gunzip + open -> committed rows only, no torn read
SNAP="$SANDBOX/check.db"
gunzip -c "$BASE/v1/state.db.gz" > "$SNAP"
integ=$(sqlite3 "$SNAP" "PRAGMA integrity_check;")
[ "$integ" = "ok" ] || { echo "FAIL: snapshot integrity_check=$integ"; fail=1; }
n_committed=$(sqlite3 "$SNAP" "SELECT COUNT(*) FROM fill;")
[ "$n_committed" = "2" ] || { echo "FAIL: expected 2 committed rows, got $n_committed"; fail=1; }
n_uncommitted=$(sqlite3 "$SNAP" "SELECT COUNT(*) FROM fill WHERE fill_id='uncommitted';")
[ "$n_uncommitted" = "0" ] || { echo "FAIL: snapshot leaked the uncommitted row"; fail=1; }

# (5) log filter: signal kept, noise dropped
LOG="$SANDBOX/log.txt"
gunzip -c "$BASE/engine/log-filtered.gz" > "$LOG"
grep -q "stale_data_halt" "$LOG" || { echo "FAIL: log filter dropped a WARN line"; fail=1; }
grep -q "order reject"    "$LOG" || { echo "FAIL: log filter dropped an ERROR line"; fail=1; }
grep -q "restart_blocked" "$LOG" || { echo "FAIL: log filter dropped a CRITICAL line"; fail=1; }
grep -q "FEED STALE"      "$LOG" || { echo "FAIL: log filter dropped a FEED STALE line"; fail=1; }
grep -q "pnl=12.30"       "$LOG" && { echo "FAIL: log filter kept 1 Hz PnL-poll noise"; fail=1; } || true
grep -q "heartbeat"       "$LOG" && { echo "FAIL: log filter kept heartbeat noise"; fail=1; } || true

if [ "$fail" -eq 0 ]; then
  echo "PASS: sync-engine-to-s3.sh integration test"
else
  exit 1
fi
