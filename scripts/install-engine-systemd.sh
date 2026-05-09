#!/bin/bash
# Install (or refresh) /etc/systemd/system/hl-engine.service on this host.
#
# Runs on EC2 as root via SSM (see `make install-engine-on-ec2`). Idempotent:
# safe to re-run after every CDK or repo change. Mirrors the unit content
# embedded in deploy/cdk/stack.go, so the live box and a freshly-CDK'd box
# end up with byte-identical units.
#
# Pre-conditions:
#  - /opt/hl-recorder is checked out at a commit that has
#    scripts/fetch-engine-secrets.sh (the unit references it via ExecStartPre).
#  - SSM parameters /hl-engine/{account-address,api-secret-key,tg-bot-token,tg-chat-id}
#    are populated. Fetch is deferred to first start, so an empty SSM here
#    will surface as a clean failure on `systemctl start`.
#  - paper_mode in config/strategy.yaml is true (this script does not flip it).
#
# Post-conditions: hl-engine.service is enabled, started, and has an active
# (or failed-with-clear-message) state. `make deploy` will work.

set -euo pipefail

REPO_DIR="/opt/hl-recorder"
SECRETS_SCRIPT="${REPO_DIR}/scripts/fetch-engine-secrets.sh"
UNIT_FILE="/etc/systemd/system/hl-engine.service"

if [ ! -x "$SECRETS_SCRIPT" ]; then
  echo "ERROR: $SECRETS_SCRIPT not found or not executable."
  echo "Pull the latest repo on this host before running this script:"
  echo "  cd $REPO_DIR && sudo -u ec2-user git pull"
  exit 1
fi

# Engine state lives at /opt/hl-recorder/data/engine/ — relative to
# WorkingDirectory so the in-repo deploy.yaml's relative paths
# (data/engine/state.db, data/engine/halt) resolve correctly without an
# /etc/hl-engine/deploy.yaml override.
mkdir -p "${REPO_DIR}/data/engine" "${REPO_DIR}/data/logs"
chown -R ec2-user:ec2-user "${REPO_DIR}/data"

mkdir -p /etc/hl-engine

# Engine systemd unit. Resource limits give the recorder priority for memory
# (load-bearing for replay/calibration) and the engine priority for CPU
# (latency-sensitive stop-loss path). OOMScoreAdjust=500 vs recorder's -500
# means the kernel reaches for the engine first under memory pressure.
#
# Restart=on-failure (NOT always) so a clean exit from the §5.5 restart-drift
# gate doesn't flap. ExecStartPre re-fetches SSM secrets on every restart so
# rotated keys land without a CDK redeploy.
cat > "$UNIT_FILE" <<'SERVICEEOF'
[Unit]
Description=Hyperliquid MM Engine (Phase 1)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/hl-recorder
ExecStartPre=+/opt/hl-recorder/scripts/fetch-engine-secrets.sh
EnvironmentFile=/etc/hl-engine/env
ExecStart=/opt/hl-recorder/.venv/bin/hl-engine \
  --strategy-config /opt/hl-recorder/config/strategy.yaml \
  --deploy-config /opt/hl-recorder/config/deploy.yaml \
  --symbols-config /opt/hl-recorder/config/symbols.yaml \
  --log-level INFO
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
MemoryMax=384M
CPUWeight=200
OOMScoreAdjust=500
ReadWritePaths=/opt/hl-recorder/data

[Install]
WantedBy=multi-user.target
SERVICEEOF

chmod 644 "$UNIT_FILE"
chown root:root "$UNIT_FILE"

systemctl daemon-reload
systemctl enable hl-engine.service

# Start (or restart) so the new unit is in effect. ExecStartPre will fetch
# SSM secrets; if any are missing the service fails fast and the operator
# can fix them and re-run this script.
if systemctl is-active --quiet hl-engine.service; then
  systemctl restart hl-engine.service
else
  systemctl start hl-engine.service || true
fi

sleep 3
systemctl status hl-engine.service --no-pager || true
echo
echo "==> Recent journal:"
journalctl -u hl-engine.service -n 30 --no-pager || true
