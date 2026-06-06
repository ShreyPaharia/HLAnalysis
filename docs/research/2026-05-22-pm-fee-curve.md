# PM fee curve fix — v3.1 prod walk-forward impact

## What changed

PM's published taker fee is `fee = C · feeRate · p · (1−p)`, not the flat 3.5 bps
our PM backtests were using. Crypto feeRate = 0.07; fees peak at $1.75/100
shares (p=0.5) and decay toward the tails where v3.1 actually trades.

`RunConfig` now carries `fee_model ∈ {flat, pm_binary}` + `fee_rate`. Branch in
`hftbt_runner._binary_fee`. HL/synthetic continue to use the flat scalar (HL
fee schedule is venue-specific anyway; we keep that knob until we model it).

Surfaced via `hl-bt run/tune --fee-model {flat,pm_binary} --fee-rate <f>`.

## Walk-forward A/B (PM, 2025-05-08 → 2026-05-07)

Both runs: prod-mirror v3.1 (`tte_max=12h`, `exit_take_profit_mode=true`,
`exit_fee=0.0007`), 5 OOS windows × 4 `exit_safety_d` values, identical seeds.

| d    | no-fee PnL | pm-fee PnL | Δ PnL    | Δ %    | pm-fee avg Sharpe | pm-fee max DD |
|------|-----------:|-----------:|---------:|-------:|------------------:|--------------:|
| 0.0  | $1357      | $629       | −$729    | −53.7% | 1.04              | $438          |
| 0.50 | $2086      | **$1314**  | −$773    | −37.0% | 2.60              | $425          |
| 0.75 | $2023      | $1225      | −$798    | −39.4% | 2.81              | $425          |
| 1.00 | $2049      | $1188      | −$861    | −42.0% | 3.11              | $525          |

Implied effective fee ≈ $0.80 per trade ≈ 0.4% of $200 notional — consistent
with the strategy trading at p ∈ [0.85, 0.95] (theoretical: 0.35–1.05%).

## Findings

1. **Strategy survives.** All d-values still produce positive PnL over the
   year. Sharpe drops by ~1 point but stays positive everywhere.

2. **The optimum shifts.** Under flat fees the published recommendation was
   d=0.75 (and prod ships d=1.0). Under the realistic curve:
   - **d=0.5 wins on total PnL** ($1314) and tied for lowest max DD ($425).
   - d=1.0 keeps the best avg Sharpe (3.11) but pays the most fees per trade
     (more topup-driven re-entries) and has the worst max DD ($525).

3. **Higher d → more fees paid.** Per-trade fee climbs from $0.75 (d=0) to
   $0.84 (d=1.0); not the trade count alone, but the additional re-entries
   the mid-hold gate generates push the average up.

4. **HL is not yet correct either.** This run only fixed PM. HL uses a
   different schedule we haven't modeled — separate task. The flat scalar
   may still be a reasonable HL approximation (HIP-4 fee is tiny) but
   should be verified.

## Recommendation

- **Live-config change:** lower v3.1 `exit_safety_d` from 1.0 → 0.5 on the
  PM-facing leg (if/when PM is added live). Keep HL at d=1.0 until we
  re-run HL with a venue-accurate fee model.
- **Backtest hygiene:** every new PM tuning yaml must run with
  `--fee-model pm_binary --fee-rate 0.07`. Add to `Makefile` or wrap in a
  helper to prevent regressions.
- **Follow-ups:** (a) re-run v3.4 LM-gate with realistic fees — the
  $2.51/market figure was on flat fees; (b) model HL HIP-4 fee schedule
  and add an `hl_binary` fee model variant.

## Artifacts

- Code: `hlanalysis/backtest/runner/hftbt_runner.py:_binary_fee`,
  `hlanalysis/backtest/cli.py` (`--fee-model`, `--fee-rate`)
- Tests: `tests/unit/backtest/test_hftbt_runner.py::test_binary_fee_*` (3 cases)
- Tuning: `config/tuning.v3-1-prod-pm-fee.yaml`
- Runs: `data/sim/tuning/v3-1-prod-pm-{fee,nofee}-2026-05-22/`
