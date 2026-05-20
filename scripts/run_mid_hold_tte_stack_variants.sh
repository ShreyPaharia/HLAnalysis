#!/usr/bin/env bash
# P3/P4 — stacking experiments at exit_safety_d=1.0
#   v3_theta_harvester (control)  +  d=1.0   (PM full year + HL 4h cap)
#   v3_2_volclock                  +  d=1.0   (PM full year + HL 4h cap)
#   v3_4_lmgate         k_jump=4.0 +  d=1.0   (PM full year)
set -euo pipefail
cd "$(dirname "$0")/.."

CFG_ROOT="data/sim/configs/v3.1-mid-hold-tte-stack"
OUT_ROOT="data/sim/runs/v3.1-mid-hold-tte-stack-2026-05-21"

export HLBT_HL_DATA_ROOT="/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data"
export HLBT_PM_CACHE_ROOT="/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data/sim"

run_pm() {
  local strategy="$1" cfg="$2" outdir="$3"
  if [[ -f "$outdir/report.md" ]]; then echo "skip $outdir"; return; fi
  mkdir -p "$outdir"
  echo "===> [$(date +%H:%M:%S)] PM $strategy ← $cfg"
  uv run hl-bt run \
    --strategy "$strategy" \
    --data-source polymarket \
    --config "$cfg" \
    --out-dir "$outdir" \
    --start 2025-05-08 --end 2026-05-07 \
    --kind both \
    --fee-taker 0.00035 \
    --slippage-bps 5.0 \
    2>&1 | tail -3
}

run_hl() {
  local strategy="$1" cfg="$2" outdir="$3"
  if [[ -f "$outdir/report.md" ]]; then echo "skip $outdir"; return; fi
  mkdir -p "$outdir"
  echo "===> [$(date +%H:%M:%S)] HL $strategy ← $cfg"
  uv run hl-bt run \
    --strategy "$strategy" \
    --data-source hl_hip4 \
    --config "$cfg" \
    --out-dir "$outdir" \
    --start 2026-05-06 --end 2026-05-22 \
    --kind both \
    --fee-taker 0.00035 \
    --slippage-bps 5.0 \
    2>&1 | tail -3
}

# P3 PM: v3_theta_harvester @ d=1.0 (control — already in prior sweep) + v3_2_volclock @ d=1.0
run_pm v3_theta_harvester "$CFG_ROOT/pm_v3_theta_d1.0.json"     "$OUT_ROOT/pm/v3_theta_d1.0"
run_pm v3_2_volclock      "$CFG_ROOT/pm_v3_2_volclock_d1.0.json" "$OUT_ROOT/pm/v3_2_volclock_d1.0"

# P4 PM: v3_4_lmgate @ k_jump=4.0 + d=1.0
run_pm v3_4_lmgate        "$CFG_ROOT/pm_v3_4_lmgate_d1.0.json"  "$OUT_ROOT/pm/v3_4_lmgate_d1.0"

# P3 HL: v3_2_volclock @ d=1.0, 4h cap (mirror prod HL config)
run_hl v3_2_volclock      "$CFG_ROOT/hl_v3_2_volclock_d1.0_tte4h.json" "$OUT_ROOT/hl/hl_v3_2_volclock_d1.0_tte4h"

echo "===> [$(date +%H:%M:%S)] variants done"
