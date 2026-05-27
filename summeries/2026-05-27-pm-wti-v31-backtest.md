# PM WTI Up/Down — v3.1 portability test

**Date:** 2026-05-27
**Goal:** Test whether the v3.1 stack (`v3_theta_harvester`), tuned on BTC,
ports to a non-crypto PM underlying without retuning. First finance-category
target per the non-crypto extension plan.

**Status:** Research only. Not taken live. Will revisit once BTC PM
production fixes land.

## TL;DR

v3.1 ports cleanly to WTI. The **HL-tuned config (`fav=0.85 / eb=0.02`)
wins on WTI by +62%**, the opposite of the BTC finding where PM-tuned won.
Driver: WTI's lower realized vol (~25–35% annualized vs BTC's ~60–90%)
makes the tighter PM favorite gate (0.90) over-selective, leaving money
on the table that the more permissive HL gate (0.85) captures.

## Corpus

- **Series:** PM `oil-daily-up-or-down`, 41 closed binary markets,
  2026-03-25 → 2026-05-22 (~9 weeks).
- **Reference data:** Pyth Benchmarks tradingview shim 1m. **Caveat:**
  Pyth deprecates expired CL contracts and removes them from the shim, so
  the active-contract table collapsed to a single entry (`WTIN6`, the
  Jul-26 contract). Calendar-spread vs the contemporaneous active month
  is typically pennies — immaterial for σ and edge in backtest. PM
  resolution uses the contemporaneous active month, but the corpus is
  already-resolved so `outcomePrices` is canonical.
- **Fee model:** `pm_binary` with `fee_rate=0.04` (PM "finance" tag
  category — WTI events tagged `finance` on gamma). Slippage 5 bps.

## Method

- No strategy code changes; existing `v3_theta_harvester` (a.k.a. v3.1).
- σ auto-estimated from `recent_returns` on the WTI reference stream
  (3600s lookback, 60s sampling). No BTC vol carried over.
- Two configs, run as-is from existing repo YAMLs (Tasks 5/6):
  - **HL-tuned**: `config/tuning.v3-1-final-baseline.yaml` flat-JSONed.
    `fav=0.85, eb=0.02, vol_clip_min=0.05, edge_max=0.20,
    min_distance_pct=0.002, topup_enabled=True`.
  - **PM-tuned**: `config/tuning.v3-1-final-pm.yaml` flat-JSONed.
    `fav=0.90, eb=0.03, vol_clip_min=0.0, edge_max=null,
    min_distance_pct=null, topup_enabled=False`.
- Diagnostic tagging only (not filtering): `spanned_weekend`
  (Fri 21:00Z – Sun 22:00Z, WTI closed → σ frozen) and `spanned_eia`
  (Wed 14:00–15:00 UTC, EIA Petroleum Status Report release).

## Results

| Metric                       | HL-tuned (0.85/0.02) | PM-tuned (0.90/0.03) |
|------------------------------|---------------------:|---------------------:|
| Markets traded (of 41)       | 35                   | 33                   |
| Total PnL (USD)              | **$480.86**          | **$297.35**          |
| PnL per market               | $11.73               | $7.25                |
| Hit rate                     | 73.17%               | 68.29%               |
| Max drawdown (single market) | –$92.03              | –$92.03              |
| Sharpe (annualized 365)      | **8.35**             | 6.58                 |
| Fill count                   | 132                  | 96                   |

The single worst market (`0x3339d40e…`, 2026-04-29/30, –$92.03) is shared
by both runs — same trade, same outcome — so the DD figure is event-driven,
not parameter-sensitive.

## Diagnostic tags

| Tag             | Markets  | HL-tuned PnL  | PM-tuned PnL  |
|-----------------|---------:|--------------:|--------------:|
| spanned_weekend | 8 of 41  | +$36.91       | +$118.86      |
| spanned_eia     | 17 of 41 | +$180.37      | +$58.39       |

Both configs are net profitable on weekend-spanning and EIA-spanning
markets; no obvious case to add explicit gating yet.

## Why HL knobs win on WTI (and lose on BTC)

WTI's lower realized vol means:
- The `vol_clip_min=0.05` floor (HL) actually adds trades the PM-tuned
  config (`vol_clip_min=0.0`) misses — 132 vs 96 fills. Those extra
  trades are net profitable on this corpus.
- The tighter PM favorite gate (0.90) skips ~5% of WTI markets that
  price in the 0.85–0.90 band where HL would trade. Most of those are
  late-day favorites that resolve as expected.
- `edge_max=0.20` (HL) and `min_distance_pct=0.002` (HL) never bind
  meaningfully on WTI, so the cost of keeping them is zero. They're
  also kept on HL prod for the same reason (safety gates, see
  `[Keep safety gates 2026-05-24]`).

The PM-tune was calibrated against BTC's fatter-tailed return distribution
where the high-p band carries less cost per trade so being more selective
paid off (see `[v3.1 final state 2026-05-24]`). The mechanism doesn't
transfer to WTI.

## Caveats

- Corpus is 9 weeks / 41 markets. No walk-forward. Treat as a directional
  signal, not a regime claim.
- All markets use WTIN6 reference prices via Pyth (contract-collapse
  caveat above). Intra-contract spread is small but non-zero.
- `fee_rate=0.04` assumes PM's "finance" category schedule per cli.py
  help text. If PM puts commodities in a different bucket, headline
  PnL shifts proportionally with `p·(1–p)·feeRate`.

## Next steps (deferred)

- Hold off on live until BTC PM production fixes land.
- WTI-native walk-forward (`k=3`/`k=4`) once the corpus is meaningfully
  longer (≥3 months).
- Sports markets (separate task — different microstructure,
  resolution semantics, and fee schedule).
- Active-month contract resolution per market if Pyth ever exposes
  historical bars for expired contracts.

## Artifacts

- Plan: `docs/superpowers/plans/2026-05-27-pm-wti-up-down-backtest.md`
- Report: `docs/research/pm-wti-up-down-backtest.md`
- Code: `hlanalysis/backtest/data/_pyth_klines.py`,
  `hlanalysis/backtest/data/polymarket.py` (parametrized),
  `hlanalysis/backtest/cli.py` (`--pm-flavor`),
  `scripts/fetch_pm_wti_klines.py`,
  `scripts/tag_pm_wti_diagnostics.py`.
- Configs: `config/backtest/v31_pm_wti_{hl,pm}_tuned.json`.
