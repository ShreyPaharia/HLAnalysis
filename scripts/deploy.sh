#!/bin/bash
set -euo pipefail

# Deploy hl-recorder: package code, push to S3, restart service on EC2 via SSM
# Usage: ./scripts/deploy.sh

STACK_NAME="HLRecorderStack"
REGION="${AWS_REGION:-$(aws configure get region)}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="hl-recorder-deploy-${ACCOUNT}"

echo "==> Account: $ACCOUNT"
echo "==> Region:  $REGION"
echo "==> Bucket:  $BUCKET"

INSTANCE_ID=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --query "Stacks[0].Outputs[?ExportName=='HLRecorderInstanceID'].OutputValue" \
  --output text)

if [ -z "$INSTANCE_ID" ]; then
  echo "ERROR: Could not fetch instance ID. Is the stack deployed?"
  exit 1
fi

echo "==> Instance: $INSTANCE_ID"

# Ensure deploy bucket exists
if ! aws s3 ls "s3://${BUCKET}" --region "$REGION" > /dev/null 2>&1; then
  echo "==> Creating S3 deploy bucket..."
  aws s3 mb "s3://${BUCKET}" --region "$REGION"
fi

# Package code (excluding venv, data, logs, CDK output, git)
echo ""
echo "==> Packaging code..."
TARBALL=$(mktemp -t hl-recorder-code.XXXXXX.tar.gz)
tar --exclude='.venv' \
    --exclude='data' \
    --exclude='logs' \
    --exclude='.git' \
    --exclude='deploy/cdk/cdk.out' \
    --exclude='deploy/cdk/cdk.context.json' \
    --exclude='__pycache__' \
    --exclude='.DS_Store' \
    --exclude='*.pyc' \
    -czf "$TARBALL" .

SIZE=$(du -h "$TARBALL" | cut -f1)
echo "  Size: $SIZE"

# Upload to S3
echo ""
echo "==> Uploading to S3..."
aws s3 cp "$TARBALL" "s3://${BUCKET}/hl-recorder-code.tar.gz" --region "$REGION"
rm -f "$TARBALL"

# Restart service via SSM (downloads new code, reinstalls, restarts)
echo ""
echo "==> Restarting service on EC2..."
COMMAND_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --region "$REGION" \
  --document-name "AWS-RunShellScript" \
  --parameters '{
    "commands": [
      "set -xe",
      "rm -rf /opt/hl-recorder.new && mkdir -p /opt/hl-recorder.new",
      "aws s3 cp s3://'"${BUCKET}"'/hl-recorder-code.tar.gz /tmp/code.tar.gz --region '"${REGION}"'",
      "tar -xzf /tmp/code.tar.gz -C /opt/hl-recorder.new",
      "rm -rf /opt/hl-recorder.bak && mv /opt/hl-recorder /opt/hl-recorder.bak 2>/dev/null || true",
      "mv /opt/hl-recorder.new /opt/hl-recorder",
      "/root/.local/bin/uv venv /opt/hl-recorder/.venv --python 3.12",
      "/root/.local/bin/uv pip install --python /opt/hl-recorder/.venv/bin/python -e /opt/hl-recorder",
      "chown -R ec2-user:ec2-user /opt/hl-recorder",
      "systemctl restart hl-recorder.service",
      "sleep 3",
      "systemctl status hl-recorder.service --no-pager"
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
    --output text | tail -20
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
