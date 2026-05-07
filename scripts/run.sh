#!/usr/bin/env bash
# Start the recorder as a background process.
# Idempotent: refuses to start if a healthy process is already running.
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs

PIDFILE=logs/recorder.pid
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "recorder already running (pid=$(cat "$PIDFILE"))"
    exit 1
fi

# Drop any stale pid file before forking.
rm -f "$PIDFILE"

nohup .venv/bin/python -m hlanalysis.recorder.main \
    --config config/symbols.yaml \
    --data-root data-local \
    --log-file logs/recorder.log \
    --log-level INFO \
    > logs/stdout.log 2>&1 &

PID=$!
echo "$PID" > "$PIDFILE"
sleep 0.5
if ! kill -0 "$PID" 2>/dev/null; then
    rm -f "$PIDFILE"
    echo "recorder failed to start; see logs/stdout.log"
    tail -20 logs/stdout.log >&2
    exit 1
fi
echo "started pid=$PID"
echo "logs:    logs/recorder.log (rotated daily)"
echo "stdout:  logs/stdout.log"
echo "stop:    scripts/stop.sh"
