#!/usr/bin/env bash
# Print recorder status: pid, runtime, recent log activity, last hour row counts.
set -euo pipefail

cd "$(dirname "$0")/.."

PIDFILE=logs/recorder.pid

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    PID=$(cat "$PIDFILE")
    echo "recorder: RUNNING pid=$PID"
    if [ "$(uname)" = "Darwin" ]; then
        ps -p "$PID" -o etime= -o rss= 2>/dev/null | awk '{printf "  uptime: %s   rss: %dMB\n", $1, $2/1024}'
    else
        ps -p "$PID" -o etime= -o rss= 2>/dev/null | awk '{printf "  uptime: %s   rss: %dMB\n", $1, $2/1024}'
    fi
else
    echo "recorder: STOPPED"
fi

echo
echo "--- last 5 log lines ---"
if [ -f logs/recorder.log ]; then
    tail -5 logs/recorder.log
fi

echo
echo "--- row counts (current hour, all venues) ---"
.venv/bin/python - <<'PY' 2>/dev/null || echo "(no parquet data yet)"
import duckdb, datetime, os
if not os.path.isdir("data"):
    raise SystemExit
con = duckdb.connect()
hour = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d/hour=%H")
date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
try:
    rows = con.execute(f"""
        SELECT venue, product_type, symbol, event, count(*) c
        FROM read_parquet('data/**/date={date}/hour={datetime.datetime.now(datetime.timezone.utc).strftime("%H")}/*.parquet', hive_partitioning=true)
        GROUP BY venue, product_type, symbol, event
        ORDER BY venue, product_type, symbol, event
    """).fetchall()
    for r in rows:
        print(f"  {r[0]:12s} {r[1]:18s} {r[2]:10s} {r[3]:15s} {r[4]:6d}")
except Exception:
    pass
PY
