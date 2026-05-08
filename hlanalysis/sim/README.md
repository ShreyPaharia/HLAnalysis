# `hlanalysis/sim/` — backtester

Spec: `docs/superpowers/specs/2026-05-09-backtester-polymarket-design.md`
Plan: `docs/superpowers/plans/2026-05-09-backtester-polymarket.md`

## End-to-end

1. Fetch ~6 months of data into `data/sim/`:
   ```
   hl-sim fetch --start 2025-12-01 --end 2026-05-09 --cache-root data/sim
   ```
2. Run a single config:
   ```
   echo '{"vol_lookback_seconds": 3600, "edge_buffer": 0.02, "stop_loss_pct": 10, "drift_lookback_seconds": 0}' > /tmp/v2.json
   hl-sim run --strategy v2 --config /tmp/v2.json --cache-root data/sim --out-dir data/sim/runs/manual
   ```
3. Tune over the grid:
   ```
   hl-sim tune --strategy v2 --grid config/tuning.yaml --cache-root data/sim --run-id v2-2026-05-09 --workers 8
   ```
4. View report at `data/sim/tuning/v2-2026-05-09/report.md`.

## Notes

- Synthetic L2 fidelity ≠ real L2. `slippage_bps`, `half_spread`, `depth` defaults are guesses; refine via the spec's §9 step 7 live audit.
- v2 (`model_edge`) uses `scipy.stats.norm`; allowed in `strategy/` via the isolation test allowlist.
- v2's GBM `d`-statistic includes the Itô drift correction `−½σ²·τ` (see `model_edge.py`). Without it, `p_model` is biased by ~σ²τ/2.
- All sim code runs in-process; multiprocessing is via `ProcessPoolExecutor` with `spawn` context for clean state.
- Walk-forward splits in `tuning.py` pass `drop_short_tail=True` so OOS test windows are uniform; the standalone `walkforward.walk_forward_splits` defaults to `False` for callers that want partial tails.
- Runner contract: strategies must expose `cfg.stop_loss_pct` (`float | None`); the runner uses it to set `Position.stop_loss_price` after a fill. Both `late_resolution` (v1) and `model_edge` (v2) satisfy this.

## Smoke test

`tests/integration/test_sim_pm_smoke.py` runs end-to-end against a captured fixture (one PM BTC Up/Down market, real trades + real Binance klines). Re-capture by running the script in plan Task 26.
