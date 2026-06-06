# v3.5 — Momentum / Mean-Reversion Gate & Tilt on PM Binaries

**Date:** 2026-05-28
**Branch baseline:** v3.1 final state (`favorite_threshold=0.9`, `edge_min_base=0.03`)
**Scope:** Polymarket BTC daily Up/Down (90d corpus). HL and WTI explicitly out of scope.

## Motivation

v3.1's favorite-side rule (`p_market > 0.9 ⇒ take favorite`) is static. It assumes the late-stage favorite is always systematically underpriced relative to GBM. But when the underlying (BTC) is overextended on a short-horizon basis — large unidirectional move into the favorite, no pullback — mean reversion may eat the model edge before resolution. Conversely, when momentum aligns with the favorite, we're under-allocating.

Add a momentum / mean-reversion (MR) score on the underlying BTC spot price and use it to either **gate** trades (skip when MR contradicts favorite) or **tilt** the required edge (looser when momentum aligns).

## Hypothesis

A momentum/MR score on BTC 1m log-returns carries marginal information over `model_edge`'s GBM drift estimate, sufficient to:
- (gate) avoid losing trades where MR is about to bite, OR
- (tilt) capture additional volume where momentum is well-aligned.

We expect this to improve **risk-adjusted** returns (worst-fold Sharpe, max DD) more than raw PnL.

## Design

### Architecture

New module `hlanalysis/strategy/momentum_mr.py` exposes:

```python
def momentum_mr_score(
    asset_bars: np.ndarray,   # 1m OHLC array, time-aligned to t
    t_index: int,             # current bar index
    lookback_min: int,        # one of {5, 15, 30, 60}
    indicator: str,           # "z_ret" | "rsi" | "ma_sigma" | "hurst_ou"
    favorite_side: int,       # +1 = Up, -1 = Down
) -> tuple[float, str]:
    """Returns (score in [-1, +1], regime in {"momentum","mr","neutral"}).
    Score is signed so positive = trend aligned with favorite_side."""
```

Wire into `hlanalysis/strategy/delta_hedged.py` behind two new configs (default off ⇒ v3.1 bit-identical):

```python
@dataclass
class MomentumMRConfig:
    enabled: bool = False
    indicator: str = "z_ret"
    lookback_min: int = 15
    mode: str = "gate"        # "gate" | "tilt"
    tau_gate: float = 1.0     # gate threshold on |score| when regime == "mr"
    alpha_tilt: float = 0.5   # tilt coefficient: edge_min *= (1 - alpha * score)
```

Behavior:
- **gate**: if `regime == "mr"` and `score < -tau_gate` (against favorite), **skip** trade. Otherwise unchanged.
- **tilt**: replace `edge_min_effective = edge_min_base * max(0, 1 - alpha_tilt * score)`. Direction (favorite side) unchanged.

### Indicators

All four computed on BTC 1m log-return series `r_t = log(close_t / close_{t-1})`.

| Indicator | Score (signed to favorite) | Regime call |
|---|---|---|
| `z_ret` | `z = sum(r over LB) / sqrt(LB) / σ_240m`; sign-aligned to favorite | `\|z\|>2.5` and sign-flip in last 3 bars ⇒ MR; `\|z\|>1.0` aligned ⇒ momentum; else neutral |
| `rsi` | `(RSI_14 - 50) / 50`, signed to favorite | RSI∈(30,70) neutral; >70 or <30 ⇒ MR (overbought/oversold relative to favorite direction) |
| `ma_sigma` | `(close − MA_LB) / σ_LB`, signed to favorite | `\|d\|>2.0` ⇒ MR; aligned and `\|d\|>0.5` ⇒ momentum |
| `hurst_ou` | OU half-life on log-price over LB window; score = `2·(H−0.5)` clipped to [-1,1] | `H<0.45` ⇒ MR; `H>0.55` ⇒ momentum |

Pure NumPy implementation (no Numba required — this runs offline in the sweep, not in the engine hot path yet).

### Sweep grid

Walk-forward `k=3` and `k=4` folds (per `[[v3_4_lmgate_walkforward_2026_05_21]]`).

- `lookback_min ∈ {5, 15, 30, 60}` — 4 values
- `tau_gate ∈ {0.5, 1.0, 1.5}` — 3 values (gate mode only)
- `alpha_tilt ∈ {0.25, 0.5, 1.0}` — 3 values (tilt mode only)
- Per indicator: 4×3 (gate) + 4×3 (tilt) = 24 cells
- Total: 4 indicators × 24 = **96 cells × 2 fold configs = 192 backtest runs**

Baseline: v3.1 final on identical splits.

### Data

Use existing PM BTC 90d corpus + BTC 1m kline data already loaded by the backtest runner (per `[[v5_hedge_data_fallback]]` — spot 1m via `data/binance_klines_btc_1m_*.parquet` or equivalent). No new ingestion.

## Ship criteria

A variant ships only if **all five** hold vs v3.1 final on PM 90d:

1. **PnL/market** ≥ v3.1 baseline on full sample.
2. **Worst-fold Sharpe** strictly better than v3.1's (no single split collapses).
3. **Max drawdown** ≤ v3.1's worst.
4. **Trade count** ≥ 60% of v3.1's (gate variant must not shrivel the book).
5. **Robustness:** ≥ 2 of 4 lookback values beat v3.1 for the chosen indicator (single-lookback wins are overfit per `[[v3_4_lmgate_walkforward_2026_05_21]]`).

If no variant clears all five ⇒ write up the negative result and shelve, mirroring `[[dynamic_sizing_negative_2026_05_19]]`.

## Risks / things to watch

- **Overfitting:** 96 cells × 4 indicators per fold-config is a large surface. Criterion #5 is the first cut.
- **Double-counting drift:** `model_edge`'s `safety_d` already encodes GBM drift toward the strike. A pure z-score MR signal may carry no marginal info beyond `safety_d` — report Spearman correlation between final score and `safety_d` in the writeup.
- **PM-tune contamination:** Per `[[v31_final_state_2026_05_23]]`, PM-tuned params don't transfer to HL. **Do not** modify any HL config in this branch. All results PM-only.
- **TTE interaction:** Late-stage PM (TTE < 1h) is where v3.1 makes most of its money. If the gate kills late-stage trades, the variant is dead; instrument TTE buckets (0–1h, 1–4h, 4–24h) in the writeup.
- **GBM Itô bias:** d-statistic must include `−½σ²·τ` per `[[strategy_v2_gbm_ito]]`. Confirm baseline uses corrected form before comparing.

## Deliverables

- `hlanalysis/strategy/momentum_mr.py` — indicator module + score function.
- `hlanalysis/strategy/delta_hedged.py` — `MomentumMRConfig` wired through; default off keeps v3.1 bit-identical.
- `hlanalysis/backtest/runner/pm_sweep_momentum_mr.py` — sweep orchestrator (192 runs, parallelizable).
- `tests/strategy/test_momentum_mr.py` — unit tests for each indicator (deterministic inputs, expected scores).
- `tests/strategy/test_delta_hedged_v35.py` — integration: default-off path is bit-identical to v3.1; gate skips correctly; tilt modulates `edge_min` correctly.
- `docs/research/v35_momentum_mr_2026_05_28.md` — sweep results, regime-bucket breakdown, decision.

## Out of scope

- HL HIP-4 (crypto perp binaries) — separate port if PM result is positive.
- WTI / other PM non-crypto markets — separate port.
- Hedge-leg interaction — pure naked PM study.
- Live engine integration — backtest study only; live wiring happens after a ship decision.

## References

- `[[strategy_phase1]]` — overall PM strategy framing.
- `[[v31_final_state_2026_05_23]]` — v3.1 baseline definition.
- `[[v3_4_lmgate_walkforward_2026_05_21]]` — walk-forward methodology and overfit guardrails.
- `[[feedback_keep_safety_gates]]` — preserve existing safety gates.
- `[[pm_fee_curve_2026_05_22]]` — `pm_binary` fee model used in backtest.
- `[[strategy_v2_gbm_ito]]` — Itô correction in d-statistic.
- `[[v5_hedge_data_fallback]]` — existing BTC 1m data source.
