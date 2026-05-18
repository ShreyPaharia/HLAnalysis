# Strategy Comparison: v3 / v4 / v5 vs v2 baseline

PM BTC daily Up/Down corpus, 2025-05-08 → 2026-05-08, 363-364 markets.
Walk-forward selection: `train=60, test=60, step=60, drop_short_tail=True` (5 OOS folds).
Objective: OOS Sharpe with `n_trades ≥ 20` floor on train.

Date: 2026-05-18.

## Headline metrics — full-corpus run at the OOS-aggregate-best config per strategy

| Strategy | Trades | Total PnL ($) | Sharpe | Hit rate | Max DD ($) |
| :------- | -----: | ------------: | -----: | -------: | ---------: |
| v2 model_edge (baseline, no MTM exits) | 614 | **−103.76** | −0.110 | 65.93% | 1,435 |
| **v3 theta_harvester** | 2,194 | **+1,560.78** | **+2.10** | 62.26% | **649** |
| v4 binary_statarb (model-free) | 6,446 | +81.02 | +0.08 | 43.53% | 1,597 |
| v5 delta_hedged ⚠ | 19,990 | +169,651.78 | +1.66 | 55.65% | 57,573 |

⚠ v5's full-corpus number is **not trustworthy** — see "v5 status" below.

## OOS-aggregate metrics (sum across 5 test folds, 300 markets total)

| Strategy | Best config (modal) | OOS PnL ($) | OOS trades | Avg fold Sharpe |
| :------- | :------------------ | ----------: | ---------: | --------------: |
| v3 theta_harvester | `exit_edge_threshold=0.0, take_profit=None, time_stop=0, stop_loss=None` | **+1,012.35** | 1,792 | +2.07 |
| v4 binary_statarb | `lookback=7200, λ=0.95, z_entry=2.0, z_exit=0.5, mid=(0.2,0.8)` | +72.65 | 5,172 | −0.24 |
| v5 delta_hedged ⚠ | `interval=300s, threshold=0.1` (post-fix; accounting still suspect) | +103,598 | 16,570 | +1.45 |

## Recommendation

**Ship v3 theta_harvester.** Move v2 → v3 in `config/strategy.yaml` for the next live engine iteration, after the HL paper-trade gate below.

Why v3:

- **Decisive PnL improvement vs v2 baseline:** +$1,665 swing on the full PM corpus (−$104 → +$1,561), with Sharpe going from −0.11 to +2.10. This is the largest signal-vs-noise effect in the comparison.
- **Mechanism confirmed:** the only change is mark-to-market re-evaluation of the GBM edge each tick and an EXIT when `edge_held < exit_edge_threshold`. The winning threshold is 0 (exit as soon as the edge has been captured) — exactly the "harvest theta, run from losers" instinct that the spec hypothesised was the v2 bleed source.
- **Risk envelope shrinks:** v2 max DD $1,435 → v3 $649 (−55%). Hit rate dips slightly (66% → 62%) because v3 takes more small wins and cuts losers earlier, but expected value per trade jumps materially.
- **Robust across folds:** OOS aggregate $1,012 across 5 folds, avg per-fold Sharpe 2.07, no single fold dominates. Full-corpus number ($1,561) extrapolates cleanly from the per-fold average.
- **Stability across nearby configs:** the top 3 v3 configs by OOS PnL all share `take_profit=None, time_stop=0, stop_loss=None` and differ only in `exit_edge_threshold ∈ {0.0, −0.005, −0.01}` — a wide flat optimum, not a knife-edge tune.

Why not v4: the model-free binary stat-arb's best aggregate is **$73 across 5 folds at avg Sharpe −0.24**. This is not a "needs more tuning" result — it is the negative control working as designed. **The BTC reference signal is essential.** GBM-based strategies (v2/v3) can extract edge from binaries; pure binary-price z-score reversion cannot. This rules out an entire class of "ignore the underlying" strategies for HL HIP-4 BTC binaries.

Why not v5: see below.

## v5 status — needs engineering before it can be evaluated

The v5 backtest surfaced two issues during execution:

1. **Hedge-state leak across questions (fixed in this branch):** the runner reuses one strategy instance across all 363 markets, but `DeltaHedgedStrategy._hedge_state` (the `last_rebalance_ns` and `hedge_qty_btc` ledger) was not reset between questions. This caused spurious rebalances on stale state, visible as **85 hedge trades/market** in the pre-fix full-corpus run vs ~6 trades/market expected. Pre-fix v5 full-corpus: **−$2,774 PnL with $42k max DD on a $100/market position cap** — clearly broken. Fix committed (`8156ced`, reset `_hedge_state` when `question.question_idx` changes).

2. **Hedge accounting flagrantly off (still open):** after the fix, v5 OOS aggregate jumped to +$103k and full-corpus to +$170k, but **max drawdown is $58k on a $100/market binary position cap** — the hedge book is recording PnL roughly **1,000× the binary's risk envelope**. Likely causes (not yet diagnosed):
   - Hedge positions not closed at question end-of-data — accumulating BTC inventory across markets, so mark-to-market PnL becomes a directional BTC bet rather than a hedge.
   - Hedge fill price-vs-size accounting: `Fill(price=104000, size=0.006)` may be summed as notional, not `(exit − entry) × size`.
   - Settlement code in the runner may be applying binary-outcome settlement (`1.0` or `0.0` price) to hedge positions.

   The v5 PnL number cannot be interpreted as alpha until this is fixed.

**Recommended next steps for v5** (separate work, not blocking v3 deployment):

1. Add an integration test asserting `sum(hedge_fill.size * (exit_price − entry_price)) ≈ realized_hedge_pnl` per question.
2. Audit `_settle_px_for_outcome` and the end-of-data settlement loop in `hftbt_runner.py` for hedge-symbol handling.
3. Re-run v5 walk-forward and full-corpus once the accounting is verified by a focused test.
4. Sanity check: v5 PnL should be **within roughly 1× of v3 PnL** when the binary is delta-hedged, ± a gamma-scalp contribution which for 24h-tenor BTC binaries at $100 position size should be on the order of $1-10/market, not $400/market.

## Caveats and known limitations

- **PM is a proxy for HL HIP-4.** Wins on PM may not transfer to HL. Required gate before any live commit: paper-trade v3 on a HL HIP-4 corpus for ≥ 30 days. PM markets are documented to be priced richer than realized vol (see memory `calibration_findings_2026_05_09`), which works in v3's favour by extending the favourite-only entry window — HL HIP-4 favourites trade tighter (0.98–0.9999), which may reduce v3 entry frequency.
- **5-fold walk-forward is noisy.** Effective sample size is smaller than 5 because consecutive BTC daily binaries are autocorrelated. The v3 effect ($1,665 swing vs v2, all five folds positive) is large enough to be robust; the v4 effect (~$73) is within sampling noise. v5's effect is contaminated by the accounting bug.
- **Backtest assumes** `fee_taker=0`, `slippage_bps=5`, `half_spread=0.005`. HL HIP-4 fills will likely see higher effective costs once order routing latency is added.
- **Hedge data fallback for v5:** the v5 hedge venue is Binance USDM perp in spec, but `fapi.binance.com` is geo-blocked from the dev IP. The backtest uses the existing 1y Binance **spot** kline file (`data/sim/btc_klines/2025-05-01_to_2026-05-09.json`) with `hedge_slippage_bps=15` to absorb spot/perp basis (~5 bps) + headroom. Production target is HL perp; the adapter is pluggable.

## Artifacts

- v3 tuning report: `data/sim/tuning/v3-walkforward-2026-05-18/report.md` (1,200 cells, 240 configs × 5 folds)
- v3 full-corpus run: `data/sim/runs/v3-walkforward-2026-05-18-full/report.md`
- v4 tuning report: `data/sim/tuning/v4-walkforward-2026-05-18/report.md` (1,920 cells)
- v4 full-corpus run: `data/sim/runs/v4-walkforward-2026-05-18-full/report.md`
- v5 tuning report (post-fix): `data/sim/tuning/v5-walkforward-2026-05-18-fix/report.md` (60 cells)
- v5 full-corpus run (post-fix, accounting-untrustworthy): `data/sim/runs/v5-fix-full-300-0.1/report.md`
- Design spec: `docs/superpowers/specs/2026-05-18-strategy-comparison-v3-v4-v5-design.md`
- Implementation plan: `docs/superpowers/plans/2026-05-18-strategy-comparison-v3-v4-v5.md`
