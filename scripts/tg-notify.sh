#!/usr/bin/env bash
# Read stdin and send to Telegram. Requires TG_BOT_TOKEN and TG_CHAT_ID env vars.
set -euo pipefail
python3 - <<'PY'
import os, sys, urllib.request, json
text = sys.stdin.read().strip()
if not text:
    sys.exit(0)
token = os.environ['TG_BOT_TOKEN']
chat_id = os.environ['TG_CHAT_ID']
data = json.dumps({
    'chat_id': chat_id,
    'text': text,
    'parse_mode': 'HTML',
    'disable_web_page_preview': True,
}).encode()
req = urllib.request.Request(
    f'https://api.telegram.org/bot{token}/sendMessage',
    data=data,
    headers={'Content-Type': 'application/json'},
)
urllib.request.urlopen(req, timeout=5)
PY
