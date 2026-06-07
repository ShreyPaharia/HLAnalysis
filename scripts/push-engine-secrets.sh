#!/bin/bash
# Push hl-engine secrets from .env.local into AWS SSM Parameter Store.
#
# Idempotent: existing parameters are overwritten in place, missing
# parameters are created. Empty/unset vars are skipped (so a partial
# .env.local that has only the v1 keys but not v31 won't error — useful
# while staging v31 separately).
#
# Mapping (.env var -> SSM param, type):
#   HL_ACCOUNT_ADDRESS      -> /hl-engine/account-address       String
#   HL_API_SECRET_KEY       -> /hl-engine/api-secret-key        SecureString
#   HL_ACCOUNT_ADDRESS_V31  -> /hl-engine/account-address-v31   String
#   HL_API_SECRET_KEY_V31   -> /hl-engine/api-secret-key-v31    SecureString
#   TG_BOT_TOKEN            -> /hl-engine/tg-bot-token          SecureString
#   TG_CHAT_ID              -> /hl-engine/tg-chat-id            String
#   PM_PRIVATE_KEY          -> /hl-engine/pm-private-key            SecureString  (v31_pm)
#   PM_CLOB_API_KEY         -> /hl-engine/pm-clob-api-key           SecureString  (v31_pm)
#   PM_CLOB_API_SECRET      -> /hl-engine/pm-clob-api-secret        SecureString  (v31_pm)
#   PM_CLOB_API_PASSPHRASE  -> /hl-engine/pm-clob-api-passphrase    SecureString  (v31_pm)
#   PM_FUNDER_ADDRESS       -> /hl-engine/pm-funder-address         String        (v31_pm)
#   PM_PRIVATE_KEY_V1       -> /hl-engine/pm-private-key-v1         SecureString  (v1_pm)
#   PM_CLOB_API_KEY_V1      -> /hl-engine/pm-clob-api-key-v1        SecureString  (v1_pm)
#   PM_CLOB_API_SECRET_V1   -> /hl-engine/pm-clob-api-secret-v1     SecureString  (v1_pm)
#   PM_CLOB_API_PASSPHRASE_V1 -> /hl-engine/pm-clob-api-passphrase-v1 SecureString (v1_pm)
#   PM_FUNDER_ADDRESS_V1    -> /hl-engine/pm-funder-address-v1      String        (v1_pm)
#
# The engine's ExecStartPre (scripts/fetch-engine-secrets.sh) reads these
# back from SSM on every `systemctl restart hl-engine.service` — so after
# this finishes, `make deploy-engine` picks up the rotated keys.
#
# Usage:
#   ./scripts/push-engine-secrets.sh                # reads .env.local in cwd
#   ENV_FILE=path/to/other.env ./scripts/push-engine-secrets.sh
#   AWS_REGION=us-east-1 ./scripts/push-engine-secrets.sh

set -euo pipefail

ENV_FILE="${ENV_FILE:-.env.local}"
REGION="${AWS_REGION:-ap-northeast-1}"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found." >&2
  echo "  Copy .env.local.example -> .env.local and fill in real values," >&2
  echo "  or set ENV_FILE=path/to/other.env" >&2
  exit 1
fi

# Extract a single var's value from $ENV_FILE without sourcing the file
# (sourcing would clobber the operator's shell with whatever extras live in
# .env.local). Returns empty if the var is unset or commented out. Portable to
# macOS bash 3.2 — no associative arrays.
get_env() {
  local var="$1"
  # Match the LAST assignment in the file so a later override wins. Strip
  # surrounding single/double quotes from the value.
  local line value
  line=$(grep -E "^[[:space:]]*${var}=" "$ENV_FILE" | tail -n 1 || true)
  if [ -z "$line" ]; then
    return 0
  fi
  value="${line#*=}"
  case "$value" in
    \"*\") value="${value#\"}"; value="${value%\"}" ;;
    \'*\') value="${value#\'}"; value="${value%\'}" ;;
  esac
  printf '%s' "$value"
}

# Mapping table. Each row is: env-var | ssm-name | type (String|SecureString) | required(yes|no)
MAPPING=(
  "HL_ACCOUNT_ADDRESS|/hl-engine/account-address|String|yes"
  "HL_API_SECRET_KEY|/hl-engine/api-secret-key|SecureString|yes"
  "HL_ACCOUNT_ADDRESS_V31|/hl-engine/account-address-v31|String|no"
  "HL_API_SECRET_KEY_V31|/hl-engine/api-secret-key-v31|SecureString|no"
  "TG_BOT_TOKEN|/hl-engine/tg-bot-token|SecureString|yes"
  "TG_CHAT_ID|/hl-engine/tg-chat-id|String|yes"
  # Polymarket v31_pm slot. All required when the slot is configured in
  # strategy.yaml — fetch-engine-secrets.sh hard-fails if they're missing.
  "PM_PRIVATE_KEY|/hl-engine/pm-private-key|SecureString|no"
  "PM_CLOB_API_KEY|/hl-engine/pm-clob-api-key|SecureString|no"
  "PM_CLOB_API_SECRET|/hl-engine/pm-clob-api-secret|SecureString|no"
  "PM_CLOB_API_PASSPHRASE|/hl-engine/pm-clob-api-passphrase|SecureString|no"
  "PM_FUNDER_ADDRESS|/hl-engine/pm-funder-address|String|no"
  # Polymarket v1_pm slot. Separate funder/keys from v31_pm so the slots have
  # independent PnL accounting and can be halted independently.
  "PM_PRIVATE_KEY_V1|/hl-engine/pm-private-key-v1|SecureString|no"
  "PM_CLOB_API_KEY_V1|/hl-engine/pm-clob-api-key-v1|SecureString|no"
  "PM_CLOB_API_SECRET_V1|/hl-engine/pm-clob-api-secret-v1|SecureString|no"
  "PM_CLOB_API_PASSPHRASE_V1|/hl-engine/pm-clob-api-passphrase-v1|SecureString|no"
  "PM_FUNDER_ADDRESS_V1|/hl-engine/pm-funder-address-v1|String|no"

  # Polymarket multi-strike bucket slots (v31_pm_btc_ms / v31_pm_eth_ms).
  # deploy.yaml references these, and load_deploy_config validates EVERY
  # referenced env var even in paper_mode — so they must be present in SSM
  # before deploying the bucket slots. paper_mode never signs, so dummy
  # (non-placeholder) values are fine until the live flip.
  "PM_PRIVATE_KEY_BTC_MS|/hl-engine/pm-private-key-btc-ms|SecureString|no"
  "PM_CLOB_API_KEY_BTC_MS|/hl-engine/pm-clob-api-key-btc-ms|SecureString|no"
  "PM_CLOB_API_SECRET_BTC_MS|/hl-engine/pm-clob-api-secret-btc-ms|SecureString|no"
  "PM_CLOB_API_PASSPHRASE_BTC_MS|/hl-engine/pm-clob-api-passphrase-btc-ms|SecureString|no"
  "PM_FUNDER_ADDRESS_BTC_MS|/hl-engine/pm-funder-address-btc-ms|String|no"
  "PM_PRIVATE_KEY_ETH_MS|/hl-engine/pm-private-key-eth-ms|SecureString|no"
  "PM_CLOB_API_KEY_ETH_MS|/hl-engine/pm-clob-api-key-eth-ms|SecureString|no"
  "PM_CLOB_API_SECRET_ETH_MS|/hl-engine/pm-clob-api-secret-eth-ms|SecureString|no"
  "PM_CLOB_API_PASSPHRASE_ETH_MS|/hl-engine/pm-clob-api-passphrase-eth-ms|SecureString|no"
  "PM_FUNDER_ADDRESS_ETH_MS|/hl-engine/pm-funder-address-eth-ms|String|no"
)

# Dummy-value guard: the .env.local.example ships placeholder addresses like
# 0xdeadbeef... — if those leaked into .env.local we'd push them to SSM and
# then the engine would crash on first sign with HL. Refuse to push placeholders.
PLACEHOLDER_PATTERNS=(
  "0xdeadbeef00000000000000000000000000000000"
  "0xdeadbeef00000000000000000000000000000000000000000000000000000000"
  "0x..."
  "fake"
)

echo "==> Region:   $REGION"
echo "==> Source:   $ENV_FILE"
echo

PUSHED=0
SKIPPED=0
for row in "${MAPPING[@]}"; do
  IFS='|' read -r env_var ssm_name type required <<<"$row"
  value=$(get_env "$env_var")

  if [ -z "$value" ]; then
    if [ "$required" = "yes" ]; then
      echo "ERROR: required $env_var is missing or empty in $ENV_FILE" >&2
      exit 1
    fi
    echo "  skip $ssm_name ($env_var unset in $ENV_FILE)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  for placeholder in "${PLACEHOLDER_PATTERNS[@]}"; do
    if [ "$value" = "$placeholder" ]; then
      echo "ERROR: $env_var is set to a placeholder value from .env.local.example." >&2
      echo "       Replace with the real value before pushing." >&2
      exit 1
    fi
  done

  # Only show a short preview of secret values — never the full payload.
  preview="${value:0:6}…${value: -4}"
  echo "  push $ssm_name [$type] = $preview"
  aws ssm put-parameter \
    --region "$REGION" \
    --name "$ssm_name" \
    --type "$type" \
    --value "$value" \
    --overwrite \
    >/dev/null
  PUSHED=$((PUSHED + 1))
done

echo
echo "==> Done: $PUSHED pushed, $SKIPPED skipped."
echo "==> Trigger a restart to pick up the new values:"
echo "      make deploy-engine"
