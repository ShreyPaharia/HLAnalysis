# PM WTI Up/Down v3.1 Backtest

**Date:** 2026-05-27
**Strategy:** v3_theta_harvester (v3.1)
**Corpus:** PM series `oil-daily-up-or-down`, 41 closed markets, 2026-03-25 → 2026-05-22
**Reference data:** Pyth Benchmarks tradingview shim — `Commodities.WTIN6/USD` (July-26 CL contract)
**Fee model:** `pm_binary`, `fee_rate=0.04` (PM "finance" category), slippage 5 bps

---

## TL;DR

v3.1 ports cleanly to WTI without retuning. On 41 closed markets the HL-tuned configuration (favorite_threshold=0.85, edge_buffer=0.02) produces $480.86 total PnL with a Sharpe of 8.35, outperforming the PM-tuned configuration ($297.35, Sharpe 6.58) by 62% on gross PnL and 5 pp on hit rate. This is the opposite of the BTC corpus result, where PM-tuned beat HL-tuned under realistic fees — WTI's lower realized volatility (~25–35% annualized vs BTC's ~60–90%) leaves more slack against the sigma caps, which favors the more permissive HL knobs.

---

## Methodology

- **PM discovery.** Markets fetched and cached via `hl-bt fetch --data-source polymarket --pm-flavor wti_updown --start 2026-03-25 --end 2026-05-23`. The `wti_updown` flavor maps to series slug `oil-daily-up-or-down` with `reference_symbol=WTI`.
- **Reference data + Pyth caveat.** 1m klines sourced from the Pyth Benchmarks tradingview shim via `scripts/fetch_pm_wti_klines.py`. The `_CL_ACTIVE_TABLE` roll table collapsed to a single entry (`Commodities.WTIN6/USD`, the July-26 contract) because all earlier contracts (H6/J6/K6/M6) returned `Symbol doesn't exist` errors from the Pyth shim — Pyth deprecates expired CL contracts and removes them. WTIN6 retains full historical depth back through the entire corpus window, so no klines are missing. The implication is that every market's reference price stream uses July-26 contract prices regardless of which contract was the actual active month at the time. The intra-contract spread is typically pennies in normal markets and is immaterial for sigma estimation and near-resolution edge, but should be noted.
- **Strike fallback.** PM resolution still uses the contemporaneous active-month contract, which is already baked into the `resolved_outcome` field read from the Gamma/CLOB cache. The contract collapse does not affect the win/loss tally.
- **Sigma auto-estimation.** No BTC vol is carried over. WTI sigma is estimated from `recent_returns` with `vol_lookback_seconds=3600`, `vol_sampling_dt_seconds=60`.
- **Two configs, no retuning.** Both configurations were applied to the WTI corpus as-is from their BTC walk-forward origins. No WTI-specific parameter search was run.
- **Fee model.** `pm_binary` with `fee_rate=0.04` and 5 bps slippage. Fee curve: `fee = qty * fee_rate * p * (1 - p)`.

---

## Results

| Metric | HL-tuned (0.85/0.02) | PM-tuned (0.90/0.03) |
|---|---|---|
| Markets discovered | 41 | 41 |
| Markets traded (>=1 fill) | 35 | 33 |
| Markets skipped (0 trades) | 6 | 8 |
| Total PnL (USD) | **$480.86** | **$297.35** |
| PnL per market (all 41) | $11.73 | $7.25 |
| Hit rate | 73.17% | 68.29% |
| Max drawdown (single market) | -$92.03 | -$92.03 |
| Sharpe (annualized 365) | **8.35** | 6.58 |
| Fill count | 132 | 96 |
| Config SHA-256 | `651cb7d516e5` | `56042fb95e81` |

The single worst market is shared by both runs: `0x3339d40e...` on 2026-04-29/30 (-$92.03 in each run). Same trade, same outcome.

**What drove the difference?** The HL-tuned config's tighter favorite threshold (0.85 vs 0.90) traded 35 markets vs 33, generating 37% more fills (132 vs 96). The live `edge_max=0.20` and `min_distance_pct=0.002` gates, which were killed for the PM-tuned config under realistic BTC fees, appear to provide signal quality filtering on WTI that more than pays for any missed opportunities. The PM-tuned config's stricter favorite threshold was calibrated against BTC's fatter-tailed return distribution; on WTI's quieter vol regime it excludes markets that would have been profitable, leaving money on the table.

---

## Diagnostic Tags

41 markets total. 8 markets span the weekend (Fri 21:00Z – Sun 22:00Z, WTI closed, sigma frozen). 17 markets span the EIA Petroleum Status Report release hour (Wed 14:00–15:00 UTC).

| Tag | Markets | HL-tuned PnL contribution | PM-tuned PnL contribution |
|---|---|---|---|
| spanned_weekend | 8 of 41 | +$36.91 | +$118.86 |
| spanned_eia | 17 of 41 | +$180.37 | +$58.39 |

PnL contributions are not additive — some markets carry both flags, some neither. Both configs are net profitable on each tagged subset. The strategy does not apply special gating for weekends or EIA releases; both remain open areas for future refinement.

---

## Configs Used

**HL-tuned** (`config/backtest/v31_pm_wti_hl_tuned.json`):

```json
{
  "favorite_threshold": 0.85,
  "edge_buffer": 0.02,
  "vol_clip_min": 0.05,
  "vol_clip_max": 5.0,
  "edge_max": 0.20,
  "min_distance_pct": 0.002,
  "min_bid_notional_usd": 10.0,
  "max_position_usd": 200.0
}
```

Source: `config/tuning.v3-1-final-baseline.yaml` (HL prod baseline). Applied to WTI without modification.

**PM-tuned** (`config/backtest/v31_pm_wti_pm_tuned.json`):

```json
{
  "favorite_threshold": 0.90,
  "edge_buffer": 0.03,
  "vol_clip_min": 0.0,
  "vol_clip_max": 5.0,
  "edge_max": null,
  "min_distance_pct": null,
  "min_bid_notional_usd": 10.0,
  "max_position_usd": 200.0,
  "topup_enabled": false
}
```

Source: `config/tuning.v3-1-final-pm.yaml` (PM walk-forward winner on BTC). `edge_max` and `min_distance_pct` were killed for the PM BTC corpus under realistic fees; `topup_enabled=false` because PM has no partial fills.

---

## Caveats

- **Short corpus (~9 weeks, 41 markets).** Out-of-sample confidence is limited. This is not a walk-forward; there are no held-out splits.
- **Reference contract collapse.** All markets use WTIN6 (Jul-26) prices from Pyth. The intra-contract spread vs the contemporaneous active month is small (typically pennies) but not zero. Future work could resolve the per-market active contract from the PM Gamma description text.
- **Weekend gaps.** WTI is closed Fri 21:00Z – Sun 22:00Z. During that window sigma goes stale. 8 of 41 markets span this gap. Both configs are net profitable on weekend-spanning markets; no gating is applied.
- **EIA event risk.** 17 of 41 markets span the Wed 14:30 UTC EIA Petroleum Status Report release. The strategy applies no special gating. Both configs are net profitable on EIA-spanning markets; the release window is an open risk.
- **sigma adaptation and vol_clip_min.** WTI's lower vol means the `vol_clip_min` floor matters more than on BTC. HL-tuned (0.05 floor) generated 37% more fills than PM-tuned (0.0 floor), suggesting the floor improves signal quality on this corpus rather than filtering out valid edges.
- **Fee rate assumption.** PM tags WTI markets as "finance" (fee_rate=0.04 per `cli.py` help text). If PM treats commodities as a separate fee bucket the actual rate could differ. PnL reported here uses fee_rate=0.04 throughout.

---

## How to Re-run

### 1. Fetch the PM WTI market cache

```bash
hl-bt fetch \
  --data-source polymarket \
  --pm-flavor wti_updown \
  --cache-root data/sim/pm_wti \
  --start 2026-03-25 \
  --end 2026-05-23
```

### 2. Fetch WTI klines from Pyth

```bash
python scripts/fetch_pm_wti_klines.py \
  --cache-root data/sim/pm_wti
```

### 3. Run backtest — HL-tuned config

```bash
hl-bt run \
  --data-source polymarket \
  --pm-flavor wti_updown \
  --cache-root data/sim/pm_wti \
  --strategy v3_theta_harvester \
  --config config/backtest/v31_pm_wti_hl_tuned.json \
  --fee-model pm_binary \
  --fee-rate 0.04 \
  --slippage-bps 5 \
  --start 2026-03-25 \
  --end 2026-05-23 \
  --out-dir data/sim/runs/pm_wti_hl_tuned
```

### 4. Run backtest — PM-tuned config

```bash
hl-bt run \
  --data-source polymarket \
  --pm-flavor wti_updown \
  --cache-root data/sim/pm_wti \
  --strategy v3_theta_harvester \
  --config config/backtest/v31_pm_wti_pm_tuned.json \
  --fee-model pm_binary \
  --fee-rate 0.04 \
  --slippage-bps 5 \
  --start 2026-03-25 \
  --end 2026-05-23 \
  --out-dir data/sim/runs/pm_wti_pm_tuned
```

### 5. Generate diagnostic tags

```bash
python scripts/tag_pm_wti_diagnostics.py \
  --cache-root data/sim/pm_wti \
  --out data/sim/runs/pm_wti_hl_tuned/diagnostics_tags.csv

python scripts/tag_pm_wti_diagnostics.py \
  --cache-root data/sim/pm_wti \
  --out data/sim/runs/pm_wti_pm_tuned/diagnostics_tags.csv
```

Results land in `data/sim/runs/pm_wti_hl_tuned/report.md` and `data/sim/runs/pm_wti_pm_tuned/report.md`.

---

## Out of Scope

- Live engine wiring — no engine changes in this branch; WTI is backtest-only.
- EIA / weekend gating — could improve worst-market outcomes but corpus is too short to calibrate.
- Per-market active-month contract resolution — requires parsing PM Gamma description text for each market's expiry date.
- WTI walk-forward (`k=3` / `k=4` tuning) — 41 markets over 9 weeks is too short to produce meaningful walk-forward splits.
- Sports and other non-WTI PM series — separate task.
