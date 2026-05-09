#!/bin/bash
set -euo pipefail

# Deploy code: git pull on EC2 and restart service(s) via SSM.
# Workflow: Push to GitHub locally, then run this script.
#
# Usage:
#   ./scripts/deploy.sh                          # both services, main branch
#   ./scripts/deploy.sh my-branch                # both services, my-branch
#   ./scripts/deploy.sh --service recorder       # recorder only, main
#   ./scripts/deploy.sh --service engine         # engine only, main
#   ./scripts/deploy.sh --service engine my-br   # engine only, my-br
#
# Restarting the engine triggers its restart-drift gate (spec §5.5). If any
# ghost/orphan/position-mismatch fires at startup, the engine writes
# /data/engine/restart_blocked and the scanner stays suspended. SSH in and
# investigate before clearing the flag.

STACK_NAME="HLRecorderStack"
REGION="${AWS_REGION:-$(aws configure get region)}"
SERVICE="both"
BRANCH="main"

while [ $# -gt 0 ]; do
  case "$1" in
    --service)
      SERVICE="${2:?--service requires recorder|engine|both}"
      shift 2
      ;;
    --service=*)
      SERVICE="${1#--service=}"
      shift
      ;;
    -h|--help)
      sed -n '4,18p' "$0"
      exit 0
      ;;
    *)
      BRANCH="$1"
      shift
      ;;
  esac
done

case "$SERVICE" in
  recorder|engine|both) ;;
  *)
    echo "ERROR: --service must be one of: recorder, engine, both (got: $SERVICE)"
    exit 1
    ;;
esac

INSTANCE_ID=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --query "Stacks[0].Outputs[?ExportName=='HLRecorderInstanceID'].OutputValue" \
  --output text)

if [ -z "$INSTANCE_ID" ]; then
  echo "ERROR: Could not fetch instance ID. Is the stack deployed?"
  exit 1
fi

# Show local commit being deployed
LOCAL_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
LOCAL_STATUS=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')

echo "==> Region:        $REGION"
echo "==> Instance:      $INSTANCE_ID"
echo "==> Branch:        $BRANCH"
echo "==> Service(s):    $SERVICE"
echo "==> Local HEAD:    $LOCAL_COMMIT"
if [ "$LOCAL_STATUS" -gt 0 ]; then
  echo "==> WARNING: You have $LOCAL_STATUS uncommitted change(s). Push to GitHub first!"
fi

# Verify GitHub is up to date
echo ""
echo "==> Checking remote..."
git fetch origin "$BRANCH" --quiet
REMOTE_COMMIT=$(git rev-parse --short "origin/$BRANCH")
echo "==> Remote HEAD:   $REMOTE_COMMIT"

if [ "$LOCAL_COMMIT" != "$REMOTE_COMMIT" ]; then
  echo ""
  echo "WARNING: Local ($LOCAL_COMMIT) and remote ($REMOTE_COMMIT) differ."
  echo "Run 'git push origin $BRANCH' to deploy your latest changes."
  echo ""
  read -p "Continue with remote $REMOTE_COMMIT? [y/N] " -n 1 -r
  echo ""
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

# Build the per-service restart block. Recorder restarts always print fast;
# engine restarts run the §5.5 restart-drift gate which can take a few seconds
# longer to settle, so we sleep 5 instead of 3 before is-active.
case "$SERVICE" in
  recorder)
    RESTART_BLOCK='"systemctl restart hl-recorder.service","sleep 3","systemctl is-active hl-recorder.service"'
    ;;
  engine)
    RESTART_BLOCK='"systemctl restart hl-engine.service","sleep 5","systemctl is-active hl-engine.service","journalctl -u hl-engine.service -n 30 --no-pager"'
    ;;
  both)
    RESTART_BLOCK='"systemctl restart hl-recorder.service","sleep 3","systemctl is-active hl-recorder.service","systemctl restart hl-engine.service","sleep 5","systemctl is-active hl-engine.service","journalctl -u hl-engine.service -n 30 --no-pager"'
    ;;
esac

# Pull and restart on EC2
echo ""
echo "==> Deploying on EC2..."
COMMAND_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --region "$REGION" \
  --document-name "AWS-RunShellScript" \
  --parameters '{
    "commands": [
      "set -xe",
      "cd /opt/hl-recorder",
      "sudo -u ec2-user git fetch origin",
      "sudo -u ec2-user git checkout '"$BRANCH"'",
      "sudo -u ec2-user git reset --hard origin/'"$BRANCH"'",
      "/root/.local/bin/uv pip install --python /opt/hl-recorder/.venv/bin/python -e /opt/hl-recorder",
      "chown -R ec2-user:ec2-user /opt/hl-recorder",
      '"$RESTART_BLOCK"',
      "sudo -u ec2-user git -C /opt/hl-recorder log --oneline -1"
    ]
  }' \
  --query "Command.CommandId" \
  --output text)

echo "  Command ID: $COMMAND_ID"
echo ""
echo "==> Waiting for command to complete..."

aws ssm wait command-executed \
  --command-id "$COMMAND_ID" \
  --instance-id "$INSTANCE_ID" \
  --region "$REGION" 2>/dev/null || true

STATUS=$(aws ssm get-command-invocation \
  --command-id "$COMMAND_ID" \
  --instance-id "$INSTANCE_ID" \
  --region "$REGION" \
  --query "Status" \
  --output text)

if [ "$STATUS" = "Success" ]; then
  echo "==> Deploy complete!"
  echo ""
  aws ssm get-command-invocation \
    --command-id "$COMMAND_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query "StandardOutputContent" \
    --output text | tail -10
else
  echo "==> Command status: $STATUS"
  echo ""
  echo "==> Output:"
  aws ssm get-command-invocation \
    --command-id "$COMMAND_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query "[StandardOutputContent, StandardErrorContent]" \
    --output text
  exit 1
fi
