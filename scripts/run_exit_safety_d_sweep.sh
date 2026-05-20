#!/usr/bin/env bash
# Sweep exit_safety_d ∈ {0.0, 0.25, 0.5, 0.75, 1.0, 1.5} on HL + PM.
# Runs sequentially; writes per-variant configs and out-dirs under
# data/sim/{configs,runs}/v3.1-exit-safety-d-2026-05-21/...
#
# Usage:
#   scripts/run_exit_safety_d_sweep.sh hl   # HL HIP-4 sweep
#   scripts/run_exit_safety_d_sweep.sh pm   # Polymarket sweep
#   scripts/run_exit_safety_d_sweep.sh all  # HL first, then PM
set -euo pipefail
cd "$(dirname "$0")/.."

CORPUS="${1:-hl}"
BASE_CFG="data/sim/configs/v3.1-exit-safety-d/base.json"
OUT_ROOT="data/sim/runs/v3.1-exit-safety-d-2026-05-21"
CFG_ROOT="data/sim/configs/v3.1-exit-safety-d"
VARIANTS=(0.0 0.25 0.5 0.75 1.0 1.5)

# Repo-local cache roots so the worktree reads the primary repo's data.
export HLBT_HL_DATA_ROOT="/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data"
export HLBT_PM_CACHE_ROOT="/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data/sim"

build_cfg() {
  local v="$1"
  local out="$CFG_ROOT/exit_safety_d_${v}.json"
  python3 - "$BASE_CFG" "$v" "$out" <<'PY'
import json, sys, pathlib
base = pathlib.Path(sys.argv[1]).read_text()
val = float(sys.argv[2])
d = json.loads(base)
d["exit_safety_d"] = val
pathlib.Path(sys.argv[3]).write_text(json.dumps(d, indent=2) + "\n")
PY
  echo "$out"
}

run_one() {
  local corpus="$1" cfg="$2" outdir="$3"
  local ds start end kind_flag
  if [[ "$corpus" == "hl" ]]; then
    ds="hl_hip4"
    start="2026-05-06"
    end="2026-05-22"
    kind_flag="--kind both"
  elif [[ "$corpus" == "pm" ]]; then
    ds="polymarket"
    start="2025-05-08"
    end="2026-05-07"
    kind_flag="--kind both"
  else
    echo "Unknown corpus: $corpus" >&2
    return 2
  fi
  echo "===> [$(date +%H:%M:%S)] $corpus exit_safety_d=$cfg out=$outdir"
  mkdir -p "$outdir"
  uv run hl-bt run \
    --strategy v3_theta_harvester \
    --data-source "$ds" \
    --config "$cfg" \
    --out-dir "$outdir" \
    --start "$start" --end "$end" \
    $kind_flag \
    --fee-taker 0.00035 \
    --slippage-bps 5.0 \
    2>&1 | tail -5
}

sweep() {
  local corpus="$1"
  for v in "${VARIANTS[@]}"; do
    local cfg
    cfg=$(build_cfg "$v")
    local outdir="$OUT_ROOT/$corpus/exit_safety_d_${v}"
    if [[ -f "$outdir/report.md" ]]; then
      echo "skip $outdir (already exists)"
      continue
    fi
    run_one "$corpus" "$cfg" "$outdir"
  done
}

case "$CORPUS" in
  hl) sweep hl ;;
  pm) sweep pm ;;
  all)
    sweep hl
    sweep pm
    ;;
  *)
    echo "Usage: $0 {hl|pm|all}" >&2
    exit 2
    ;;
esac
