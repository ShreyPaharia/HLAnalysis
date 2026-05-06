#!/bin/bash
set -euo pipefail

# Deploy hl-recorder: git pull on EC2 and restart service via SSM.
# Workflow: Push to GitHub locally, then run this script.
#
# Usage: ./scripts/deploy.sh [branch]

STACK_NAME="HLRecorderStack"
REGION="${AWS_REGION:-$(aws configure get region)}"
BRANCH="${1:-main}"

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
      "systemctl restart hl-recorder.service",
      "sleep 3",
      "systemctl is-active hl-recorder.service",
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
