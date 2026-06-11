#!/bin/bash
# Install (or refresh) the daily engine→S3 snapshot timer on this host.
#
# Runs on EC2 as root via SSM. Idempotent: safe to re-run after every repo change.
# Installs the two units from deploy/systemd/ so the live box matches the repo:
#   - hl-engine-s3-sync.service  (oneshot: scripts/sync-engine-to-s3.sh)
#   - hl-engine-s3-sync.timer    (daily 06:45 UTC, Persistent=true)
#
# This is the piece #19 shipped the script + unit files for but never wired onto
# the box, leaving s3://<bucket>/engine/ empty. After this runs, the snapshot
# (state.db + gate_decisions + filtered log) lands in S3 daily and is pullable
# SSM-free via scripts/pull-engine.sh.
#
# Pre-conditions:
#  - /opt/hl-recorder is checked out at a commit that has the patched
#    scripts/sync-engine-to-s3.sh (Python online-backup fallback — the box has no
#    sqlite3 CLI) and deploy/systemd/hl-engine-s3-sync.{service,timer}.
#  - ARCHIVE_BUCKET is set in /etc/hl-recorder/env (the unit's EnvironmentFile).

set -euo pipefail

REPO_DIR="/opt/hl-recorder"
SRC_DIR="${REPO_DIR}/deploy/systemd"
SYNC_SCRIPT="${REPO_DIR}/scripts/sync-engine-to-s3.sh"

for f in hl-engine-s3-sync.service hl-engine-s3-sync.timer; do
  if [ ! -f "$SRC_DIR/$f" ]; then
    echo "ERROR: $SRC_DIR/$f not found. Pull the latest repo on this host first:"
    echo "  cd $REPO_DIR && sudo -u ec2-user git pull"
    exit 1
  fi
done

if [ ! -x "$SYNC_SCRIPT" ]; then
  echo "ERROR: $SYNC_SCRIPT not found or not executable (the unit's ExecStart)."
  exit 1
fi

for f in hl-engine-s3-sync.service hl-engine-s3-sync.timer; do
  install -m 0644 -o root -g root "$SRC_DIR/$f" "/etc/systemd/system/$f"
  echo "==> installed /etc/systemd/system/$f"
done

systemctl daemon-reload
systemctl enable --now hl-engine-s3-sync.timer

echo
echo "==> timer status:"
systemctl list-timers hl-engine-s3-sync.timer --all --no-pager || true
echo
echo "==> last service run (if any):"
systemctl status hl-engine-s3-sync.service --no-pager || true
