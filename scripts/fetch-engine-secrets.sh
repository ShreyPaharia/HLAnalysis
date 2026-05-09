#!/bin/bash
# Pull hl-engine secrets from SSM Parameter Store into /etc/hl-engine/env.
#
# Wired in via `ExecStartPre=+` on hl-engine.service so it re-runs on every
# `systemctl restart hl-engine.service` — that's how rotated secrets in SSM
# get picked up without a CDK redeploy.
#
# Runs as root (the `+` prefix on ExecStartPre) because /etc/hl-engine/env is
# 0600 root:root. Failures here intentionally fail the engine start: the
# engine has no business running without the signing key.
#
# Region resolution order: $AWS_REGION env, IMDSv2, hardcoded ap-northeast-1.

set -euo pipefail

REGION="${AWS_REGION:-}"
if [ -z "$REGION" ]; then
  TOKEN=$(curl -fsS -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || true)
  if [ -n "$TOKEN" ]; then
    REGION=$(curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" \
      http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || true)
  fi
fi
REGION="${REGION:-ap-northeast-1}"

mkdir -p /etc/hl-engine

# Fetch each parameter; abort cleanly if any required one is missing so the
# engine startup fails loudly instead of running with empty secrets.
get_param() {
  local name="$1"
  local decrypt="$2"
  local args=(--name "$name" --region "$REGION" --query Parameter.Value --output text)
  if [ "$decrypt" = "yes" ]; then
    args+=(--with-decryption)
  fi
  aws ssm get-parameter "${args[@]}"
}

HL_ACCOUNT_ADDRESS=$(get_param /hl-engine/account-address no)
HL_API_SECRET_KEY=$(get_param /hl-engine/api-secret-key yes)
TG_BOT_TOKEN=$(get_param /hl-engine/tg-bot-token yes)
TG_CHAT_ID=$(get_param /hl-engine/tg-chat-id no)

# Write atomically so a partial write can never be sourced by EnvironmentFile.
TMP=$(mktemp /etc/hl-engine/env.XXXXXX)
chmod 600 "$TMP"
cat > "$TMP" <<ENVEOF
HL_ACCOUNT_ADDRESS=${HL_ACCOUNT_ADDRESS}
HL_API_SECRET_KEY=${HL_API_SECRET_KEY}
TG_BOT_TOKEN=${TG_BOT_TOKEN}
TG_CHAT_ID=${TG_CHAT_ID}
PYTHONUNBUFFERED=1
ENVEOF
mv "$TMP" /etc/hl-engine/env
chown root:root /etc/hl-engine/env
chmod 600 /etc/hl-engine/env
