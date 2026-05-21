#!/usr/bin/env bash
# P0 — HL HIP-4 5-variant test: does mid-hold safety_d let us safely
# remove (or relax) the 4h TTE cap?
#
# Configs under data/sim/configs/v3.1-mid-hold-tte-stack/
# Runs under   data/sim/runs/v3.1-mid-hold-tte-stack-2026-05-21/hl/
set -euo pipefail
cd "$(dirname "$0")/.."

CFG_ROOT="data/sim/configs/v3.1-mid-hold-tte-stack"
OUT_ROOT="data/sim/runs/v3.1-mid-hold-tte-stack-2026-05-21/hl"
VARIANTS=(
  "hl_d0.0_tte4h"
  "hl_d1.0_tte4h"
  "hl_d0.0_tte24h"
  "hl_d1.0_tte12h"
  "hl_d1.0_tte24h"
)

export HLBT_HL_DATA_ROOT="/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data"

mkdir -p "$OUT_ROOT"

for v in "${VARIANTS[@]}"; do
  cfg="$CFG_ROOT/${v}.json"
  out="$OUT_ROOT/${v}"
  if [[ -f "$out/report.md" ]]; then
    echo "skip $out (already exists)"
    continue
  fi
  mkdir -p "$out"
  echo "===> [$(date +%H:%M:%S)] HL $v"
  uv run hl-bt run \
    --strategy v3_theta_harvester \
    --data-source hl_hip4 \
    --config "$cfg" \
    --out-dir "$out" \
    --start 2026-05-06 --end 2026-05-22 \
    --kind both \
    --fee-taker 0.00035 \
    --slippage-bps 5.0 \
    2>&1 | tail -8
done

echo "===> [$(date +%H:%M:%S)] HL sweep done"
