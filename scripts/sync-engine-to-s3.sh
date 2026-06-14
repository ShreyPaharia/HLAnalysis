#!/bin/bash
# Push a consistent daily snapshot of the LIVE engine's operational data to S3,
# so analysts/workers can `aws s3 sync` it locally and query state.db directly —
# replacing the slow per-run SSM `user_fills` roundtrips.
#
# Runs ON the box (invoked by hl-engine-s3-sync.timer, daily after the 06:00 UTC
# settlement). Additive to the recorder sync (scripts/sync-to-s3.sh): a separate
# `engine/` prefix, never touches the recorder's partitions.
#
# For each slot alias under $ENGINE_DATA_ROOT/<alias>/ it uploads, under
#   s3://$BUCKET/engine/date=YYYY-MM-DD/<alias>/
#   - state.db.gz          a CONSISTENT snapshot (sqlite `.backup`, NOT cp) —
#                          the engine is live-writing and a raw cp of .db + -wal
#                          can be torn/corrupt-on-read. `.backup` uses SQLite's
#                          online backup API: it folds the WAL in and yields a
#                          single self-contained file with only committed rows.
#   - gate_decisions.jsonl.gz   the live decision log (high analysis value).
#   - log-filtered.gz      journald hl-engine lines matching the signal regex
#                          (WARN/ERROR/halt/reject/…) — KB/day, NOT the ~97 MB/day
#                          raw 1 Hz PnL-poll + 429 noise. Piped journalctl|grep|gzip
#                          so the raw log is never buffered in memory.
#
# EXCLUDED by construction (we name the two files exactly, never glob the dir):
# stale `state.db.bak-*`, the live `-wal`/`-shm`, raw logs.
#
# Lean by design (the box is a t4g.micro that OOMs): one slot at a time, snapshot
# to /tmp and clean up, stream-gzip. CPU/IO niceness is set by the systemd unit
# (Nice=19 / IOSchedulingClass=idle) so this script stays portable.
#
# Idempotent: daily granularity, one date= prefix per run; re-running overwrites
# that day's objects in place.
#
# Required env: ARCHIVE_BUCKET (same bucket as the recorder; supplied via
#               EnvironmentFile=/etc/hl-recorder/env on the box).
# Optional env (defaults target the live box; overridden by the test harness):
#   ENGINE_DATA_ROOT     (default /opt/hl-recorder/data/engine)
#   SYNC_DATE            (default today UTC, YYYY-MM-DD — the date= prefix)
#   LOG_SINCE            (default "yesterday" — journalctl --since window)
#   LOG_UNIT             (default hl-engine — journald unit to filter)
#   ENGINE_LOG_REGEX     (default below — the signal-not-noise grep)
#   S3_ENGINE_PREFIX     (default engine — the top-level key prefix)
#   ENGINE_SYNC_S3_BASE  (default s3://$BUCKET/$S3_ENGINE_PREFIX; tests point this
#                         at a local dir so no real AWS is needed)
#   SQLITE3 / JOURNALCTL (binaries; overridable for tests)

set -euo pipefail

BUCKET="${ARCHIVE_BUCKET:?ARCHIVE_BUCKET must be set}"
ENGINE_ROOT="${ENGINE_DATA_ROOT:-/opt/hl-recorder/data/engine}"
DATE_BIN=$(command -v gdate || command -v date)
DATE="${SYNC_DATE:-$("$DATE_BIN" -u +%Y-%m-%d)}"
LOG_SINCE="${LOG_SINCE:-yesterday}"
LOG_UNIT="${LOG_UNIT:-hl-engine}"
S3_ENGINE_PREFIX="${S3_ENGINE_PREFIX:-engine}"
S3_BASE="${ENGINE_SYNC_S3_BASE:-s3://$BUCKET/$S3_ENGINE_PREFIX}"
SQLITE3="${SQLITE3:-sqlite3}"
# The t4g.micro box ships NO sqlite3 CLI (by design — analysis uses .venv python),
# so the consistent snapshot falls back to Python's online backup API when the CLI
# is absent. Tests can still force the CLI path by exporting SQLITE3.
PYBIN="${PYBIN:-/opt/hl-recorder/.venv/bin/python}"
JOURNALCTL="${JOURNALCTL:-journalctl}"

# Consistent online snapshot of a live-written sqlite db → $2. Prefers the
# sqlite3 CLI `.backup` (folds the WAL, committed rows only); falls back to the
# Python sqlite3 module's Connection.backup() when the CLI is unavailable. Both
# use SQLite's online backup API, so a concurrent engine writer is safe.
sqlite_backup() {  # sqlite_backup <src_db> <dest_file>
  local src="$1" dest="$2"
  if command -v "$SQLITE3" >/dev/null 2>&1; then
    "$SQLITE3" "$src" ".timeout 10000" ".backup '$dest'"
    return $?
  fi
  "$PYBIN" - "$src" "$dest" <<'PYEOF'
import sqlite3, sys
src, dest = sys.argv[1], sys.argv[2]
s = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=10)
d = sqlite3.connect(dest)
with d:
    s.backup(d)
d.close(); s.close()
PYEOF
}

# Signal, not noise. The raw hl-engine journal is dominated by the 1 Hz PnL poll,
# HL 429s, and reject spam (~97 MB/day). Keep only operationally interesting
# lines. Case-insensitive so both log levels (WARN/ERROR) and lowercase message
# keywords (halt/reject/drift) match.
ENGINE_LOG_REGEX="${ENGINE_LOG_REGEX:-WARN|ERROR|CRITICAL|halt|restart_blocked|reject|OOM|FEED STALE|FEED RECOVERED|drift|Traceback|Exception}"

if [ ! -d "$ENGINE_ROOT" ]; then
  echo "ERROR: engine data root $ENGINE_ROOT does not exist" >&2
  exit 1
fi

# The snapshot temp dir MUST be disk-backed, not tmpfs. The default mktemp dir
# ($TMPDIR or /tmp) is a tmpfs sized at ~50% of RAM (~945 MB on the 2 GB box),
# so the online .backup of a >1 GB slot DB failed with SQLITE_FULL and the slot
# was silently skipped from S3 (the 2026-06-14 eth_ms gap — 1.4 GB trade_journal
# never archived). Stage on /var/tmp (root xfs, 16 GB) instead; overridable.
BACKUP_TMPDIR="${ENGINE_BACKUP_TMPDIR:-/var/tmp}"
TMP=$(mktemp -d -p "$BACKUP_TMPDIR" 2>/dev/null || mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Upload (or, in tests, local-copy) one file to its dated key. Switching on the
# s3:// scheme keeps the script runnable end-to-end in a sandbox without AWS.
put() {  # put <local_file> <relative_key>
  local src="$1" key="$2"
  local dest="$S3_BASE/date=$DATE/$key"
  case "$dest" in
    s3://*) aws s3 cp --no-progress "$src" "$dest" ;;
    *) mkdir -p "$(dirname "$dest")"; cp "$src" "$dest" ;;
  esac
}

# --- per-slot Tier 1: state.db snapshot + gate_decisions.jsonl --------------
SLOTS=0
# A slot is any immediate subdir holding a state.db, PLUS the top-level state.db
# (the un-namespaced default account) keyed under the alias "_root".
shopt -s nullglob
declare -a SLOT_DBS=()
declare -a SLOT_ALIASES=()
for db in "$ENGINE_ROOT"/*/state.db; do
  SLOT_DBS+=("$db")
  SLOT_ALIASES+=("$(basename "$(dirname "$db")")")
done
if [ -f "$ENGINE_ROOT/state.db" ]; then
  SLOT_DBS+=("$ENGINE_ROOT/state.db")
  SLOT_ALIASES+=("_root")
fi
shopt -u nullglob

for i in "${!SLOT_DBS[@]}"; do
  db="${SLOT_DBS[$i]}"
  alias="${SLOT_ALIASES[$i]}"
  dir="$(dirname "$db")"
  echo "==> slot $alias"

  # Consistent snapshot via the online backup API (folds the WAL, committed rows
  # only). Waits out a transient writer lock instead of failing.
  snap="$TMP/$alias-state.db"
  if sqlite_backup "$db" "$snap"; then
    gzip -c "$snap" > "$snap.gz"
    put "$snap.gz" "$alias/state.db.gz"
    rm -f "$snap" "$snap.gz"
  else
    echo "WARN: .backup failed for $alias ($db); skipping state.db" >&2
  fi

  # Live decision log (sibling of state.db). May be absent on a fresh slot.
  gd="$dir/gate_decisions.jsonl"
  if [ -f "$gd" ]; then
    gzip -c "$gd" > "$TMP/$alias-gd.jsonl.gz"
    put "$TMP/$alias-gd.jsonl.gz" "$alias/gate_decisions.jsonl.gz"
    rm -f "$TMP/$alias-gd.jsonl.gz"
  fi

  SLOTS=$((SLOTS + 1))
done

# --- Tier 2: filtered engine log (engine-wide, not per-slot) ----------------
# Pipe journalctl|grep|gzip so the 97 MB/day raw log never lands on disk or in
# memory. The subshell localizes `set +o pipefail` so grep's exit-1 on zero
# matches (or a missing journalctl in a sandbox) doesn't abort the run.
logout="$TMP/log-filtered.gz"
(
  set +o pipefail
  "$JOURNALCTL" -u "$LOG_UNIT" --since "$LOG_SINCE" --no-pager 2>/dev/null \
    | grep -iE "$ENGINE_LOG_REGEX" \
    | gzip -c
) > "$logout"
put "$logout" "engine/log-filtered.gz"
rm -f "$logout"

echo "==> engine sync complete: slots=$SLOTS date=$DATE base=$S3_BASE"
