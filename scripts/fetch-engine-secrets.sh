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

# v1 (late_resolution) — required.
HL_ACCOUNT_ADDRESS=$(get_param /hl-engine/account-address no)
HL_API_SECRET_KEY=$(get_param /hl-engine/api-secret-key yes)
TG_BOT_TOKEN=$(get_param /hl-engine/tg-bot-token yes)
TG_CHAT_ID=$(get_param /hl-engine/tg-chat-id no)

# v31 (theta_harvester) — multi-account. Optional so a fresh install with only
# the legacy v1 params doesn't fail-start. If either v31 param is missing the
# engine will still boot v1; the runtime errors loudly when load_deploy_config
# tries to substitute the missing env var, surfacing the misconfiguration in
# journalctl rather than silently running with empty credentials.
HL_ACCOUNT_ADDRESS_V31=$(get_param /hl-engine/account-address-v31 no 2>/dev/null || true)
HL_API_SECRET_KEY_V31=$(get_param /hl-engine/api-secret-key-v31 yes 2>/dev/null || true)

# v31_pm (theta_harvester on Polymarket) — optional, same pattern as v31. The
# engine constructs the v31_pm slot only when deploy.yaml's accounts.v31_pm
# block resolves these env vars; missing here means the engine still boots
# v1 + v31 but the v31_pm slot fails at load_deploy_config with a clear
# substitution error in journalctl.
PM_PRIVATE_KEY=$(get_param /hl-engine/pm-private-key yes 2>/dev/null || true)
PM_CLOB_API_KEY=$(get_param /hl-engine/pm-clob-api-key yes 2>/dev/null || true)
PM_CLOB_API_SECRET=$(get_param /hl-engine/pm-clob-api-secret yes 2>/dev/null || true)
PM_CLOB_API_PASSPHRASE=$(get_param /hl-engine/pm-clob-api-passphrase yes 2>/dev/null || true)
PM_FUNDER_ADDRESS=$(get_param /hl-engine/pm-funder-address no 2>/dev/null || true)

# v1_pm (late_resolution on Polymarket) — optional, same pattern as v31_pm.
# Separate funder/keys so the two PM slots have independent PnL accounting and
# can be halted independently.
PM_PRIVATE_KEY_V1=$(get_param /hl-engine/pm-private-key-v1 yes 2>/dev/null || true)
PM_CLOB_API_KEY_V1=$(get_param /hl-engine/pm-clob-api-key-v1 yes 2>/dev/null || true)
PM_CLOB_API_SECRET_V1=$(get_param /hl-engine/pm-clob-api-secret-v1 yes 2>/dev/null || true)
PM_CLOB_API_PASSPHRASE_V1=$(get_param /hl-engine/pm-clob-api-passphrase-v1 yes 2>/dev/null || true)
PM_FUNDER_ADDRESS_V1=$(get_param /hl-engine/pm-funder-address-v1 no 2>/dev/null || true)

# Write atomically so a partial write can never be sourced by EnvironmentFile.
TMP=$(mktemp /etc/hl-engine/env.XXXXXX)
chmod 600 "$TMP"
cat > "$TMP" <<ENVEOF
HL_ACCOUNT_ADDRESS=${HL_ACCOUNT_ADDRESS}
HL_API_SECRET_KEY=${HL_API_SECRET_KEY}
HL_ACCOUNT_ADDRESS_V31=${HL_ACCOUNT_ADDRESS_V31}
HL_API_SECRET_KEY_V31=${HL_API_SECRET_KEY_V31}
PM_PRIVATE_KEY=${PM_PRIVATE_KEY}
PM_CLOB_API_KEY=${PM_CLOB_API_KEY}
PM_CLOB_API_SECRET=${PM_CLOB_API_SECRET}
PM_CLOB_API_PASSPHRASE=${PM_CLOB_API_PASSPHRASE}
PM_FUNDER_ADDRESS=${PM_FUNDER_ADDRESS}
PM_PRIVATE_KEY_V1=${PM_PRIVATE_KEY_V1}
PM_CLOB_API_KEY_V1=${PM_CLOB_API_KEY_V1}
PM_CLOB_API_SECRET_V1=${PM_CLOB_API_SECRET_V1}
PM_CLOB_API_PASSPHRASE_V1=${PM_CLOB_API_PASSPHRASE_V1}
PM_FUNDER_ADDRESS_V1=${PM_FUNDER_ADDRESS_V1}
TG_BOT_TOKEN=${TG_BOT_TOKEN}
TG_CHAT_ID=${TG_CHAT_ID}
PYTHONUNBUFFERED=1
ENVEOF
mv "$TMP" /etc/hl-engine/env
chown root:root /etc/hl-engine/env
chmod 600 /etc/hl-engine/env
