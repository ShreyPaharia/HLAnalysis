#!/usr/bin/env bash
# v1 (late_resolution) post-σ-fix HL retune.
# Grid: tte_max ∈ {7200, 14400, 43200, 86400} × exit_safety_d ∈ {0.0, 0.5, 1.0, 1.5}
# = 16 cells. Each run is the HL HIP-4 14-day corpus (2026-05-06 → 2026-05-22, kind=both).
set -euo pipefail
cd "$(dirname "$0")/.."

CFG_ROOT="data/sim/configs/v1-postfix-sweep"
OUT_ROOT="data/sim/runs/v1-postfix-sweep-2026-05-21"
TTES=(7200 14400 43200 86400)
DS=(0.0 0.5 1.0 1.5)

export HLBT_HL_DATA_ROOT="/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data"
mkdir -p "$CFG_ROOT" "$OUT_ROOT"

build_cfg() {
  local tte="$1" d="$2"
  local out="$CFG_ROOT/v1_tte${tte}_d${d}.json"
  cat > "$out" <<JSON
{
  "tte_min_seconds": 0,
  "tte_max_seconds": ${tte},
  "price_extreme_threshold": 0.85,
  "price_extreme_max": 0.99,
  "distance_from_strike_usd_min": 0,
  "vol_max": 100,
  "stop_loss_pct": null,
  "max_position_usd": 100,
  "min_safety_d": 1.0,
  "vol_lookback_seconds": 3600,
  "exit_safety_d": ${d},
  "exit_bid_floor": 0,
  "drift_aware_d": false,
  "vol_ewma_lambda": 0.85,
  "size_cap_near_strike_pct": 1.0,
  "size_cap_max_dist_pct": 1.5,
  "size_cap_min_ask": 0.88,
  "use_bid_for_entry_gate": true,
  "min_bid_notional_usd": 25.0,
  "entry_cooldown_seconds": 60
}
JSON
  echo "$out"
}

for tte in "${TTES[@]}"; do
  for d in "${DS[@]}"; do
    cfg=$(build_cfg "$tte" "$d")
    out="$OUT_ROOT/v1_tte${tte}_d${d}"
    if [[ -f "$out/report.md" ]]; then
      echo "skip $out"
      continue
    fi
    mkdir -p "$out"
    echo "===> [$(date +%H:%M:%S)] v1 tte=$tte d=$d"
    uv run hl-bt run \
      --strategy v1_late_resolution \
      --data-source hl_hip4 \
      --config "$cfg" \
      --out-dir "$out" \
      --start 2026-05-06 --end 2026-05-22 \
      --kind both \
      --fee-taker 0.00035 \
      --slippage-bps 5.0 \
      2>&1 | tail -3
  done
done

echo "===> [$(date +%H:%M:%S)] v1 sweep done"
