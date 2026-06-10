#!/usr/bin/env bash
# One-off driver: re-run the all-days HL live-vs-sim matrix with the SHR-79/89
# IOC re-fire floor (ioc_ arm) vs no floor (base_ arm). Single-process per
# SHR-100 (spawned workers load main + drop config). NOT a maintained tool.
# bash 3.2-compatible (macOS) — no associative arrays.
set -uo pipefail
export HLBT_HL_DATA_ROOT=/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data

window() {  # echo "<start> <end>" for a settle-day key
  case "$1" in
    0606) echo "2026-06-05T07:00:00Z 2026-06-06T07:00:00Z" ;;
    0607) echo "2026-06-06T07:00:00Z 2026-06-07T07:00:00Z" ;;
    0608) echo "2026-06-07T07:00:00Z 2026-06-08T07:00:00Z" ;;
    0609) echo "2026-06-08T07:00:00Z 2026-06-09T07:00:00Z" ;;
    *) echo "BAD BAD" ;;
  esac
}

run_cell() {
  local prefix=$1 floor=$2 slot=$3 kind=$4 day=$5
  local w start end out
  w=$(window "$day"); start=${w% *}; end=${w#* }
  out="data/sim/runs/${prefix}_${slot}_${kind}_${day}"
  if [[ -f "$out/fills.parquet" ]]; then echo "skip $out"; return; fi
  echo ">>> $prefix $slot $kind $day floor=$floor win=$start..$end"
  uv run hl-bt run --slot "$slot" --slot-class "$kind" --kind "$kind" \
    --data-source hl_hip4 --ref-source hl_perp --ref-event mark \
    --reference-ticks raw --scan-mode event --fee-taker 0.0 --slippage-bps 0 \
    --min-inter-order-seconds "$floor" \
    --start "$start" --end "$end" --out-dir "$out" >/dev/null 2>&1 \
    && echo "  ok $out" || echo "  FAILED $out"
}

for day in 0606 0607 0608 0609; do
  for slot in v1 v31; do
    for kind in binary bucket; do
      run_cell ioc  0.75 "$slot" "$kind" "$day"
      run_cell base 0.0  "$slot" "$kind" "$day"
    done
  done
done
echo "ALL DONE"
