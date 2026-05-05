#!/usr/bin/env bash
# Send SIGINT to the running recorder and wait for clean shutdown.
set -euo pipefail

cd "$(dirname "$0")/.."

PIDFILE=logs/recorder.pid
if [ ! -f "$PIDFILE" ]; then
    echo "no pid file at $PIDFILE; recorder probably not running"
    exit 0
fi

PID=$(cat "$PIDFILE")
if ! kill -0 "$PID" 2>/dev/null; then
    echo "process $PID not running; clearing stale pid file"
    rm -f "$PIDFILE"
    exit 0
fi

echo "stopping recorder (pid=$PID)"
kill -INT "$PID"

# Wait up to 30s for graceful shutdown (parquet flushers need time on busy buffers).
for i in $(seq 1 30); do
    if ! kill -0 "$PID" 2>/dev/null; then
        rm -f "$PIDFILE"
        echo "stopped"
        exit 0
    fi
    sleep 1
done

echo "did not stop in 30s; sending SIGTERM"
kill -TERM "$PID" || true
sleep 2
rm -f "$PIDFILE"
