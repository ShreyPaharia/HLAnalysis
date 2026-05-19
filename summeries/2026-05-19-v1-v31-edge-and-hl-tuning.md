# HL HIP-4 — v1 & v3.1 edge, edge-cases, and microstructure-driven re-tune

**2026-05-19 · Shrey Paharia · scope: live HL HIP-4 BTC binaries (+ bucket sanity)**

## Why this memo

Both strategies are live on HL HIP-4 BTC binaries with PM-tuned thresholds. Today produced two textbook failures on the same question (#601):

- **v1: −$120** across 7 churn cycles. The ask-based favourite-leg gate accepted a stale top-of-book ask of 0.999 while the real market was at 0.55–0.60. IOC swept the book, the safety-d gate then exited at the depressed bid, and the strategy re-entered on the next stale-ask tick.
- **v3.1: −$86** on a single trade. v3.1 entered when GBM said `edge ≥ 0.02` and held for ~12 hours; BTC drifted through the strike during the hold, and the held leg's bid collapsed before the GBM-edge exit could fire fast enough.

Same root cause: **PM tuning assumed PM microstructure** — tight bid/ask, deep books, a well-behaved 24h GBM. HL HIP-4 violates all three. Defensive gates shipped today on `main` (bid-gate, `min_bid_notional_usd=10`, `min_distance_pct=0.002`, 60s post-exit cooldown, 06UTC daily window) are **assumed ON**; proposals layer on top.

---

## 1 · Theoretical edge model

### v1 — late_resolution

> *Buy the favourite leg of a HIP-4 binary in the last 2h before settlement when it's already priced ≥0.85 and BTC is comfortably (`safety_d ≥ 1.0`) on the winning side of the strike.*

Late in the contract's life, the favourite-leg ask is bounded above by 1.0 (payout) and below by `1 − P(adverse move > distance-to-strike in remaining τ)`. Realised σ on BTC over the last hour underwrites that probability. When a counterparty quotes 0.985 with strike 1.5 σ-windows away from spot, they leave ≈ `0.015 − ε` per share on the table; v1 buys exactly that residual. It is theta-on-rails: `1 − ask → 0` deterministically as τ → 0 *conditional on* BTC staying on the winning side. v1 doesn't predict direction — it harvests the convexity premium between "almost certain" and "certain".

**What kills it:** (a) the printed ask isn't the true ask (stale book, spoof — today's #601); (b) σ_realised understates σ_implied (event risk); (c) BTC traverses `safety_d` σ-windows faster than the exit can fire.

### v3.1 — theta_harvester

> *On any leg whose mid ≥ 0.85, compute GBM `p_model = N(d)` with Itô-corrected `d = (ln(S/K) + (μ − ½σ²)τ) / (σ√τ)`; `edge = p_model − ask − fee − half_spread`; enter when edge ∈ (0.02, 0.20).*

Two ways v3.1 wins: (i) direct mispricing — when market-implied prob diverges from GBM-implied prob, v3.1 buys the cheaper side; (ii) theta harvest — independent of direction, the favourite's implied prob converges to 1.0 as τ → 0, paying off any positive-edge holder. The `edge_max=0.20` ceiling rejects entries where the model claims edge so large that event risk almost certainly explains the gap (PM analysis showed those wipe out 55% of the time).

The Itô term `(μ − ½σ²)τ` matters: with σ_ann = 0.5 and τ = 12h, dropping it would over-estimate the favourite's `p_model` by ≈1.4 pp and chase too many entries near fair-priced books.

**What kills it:** σ_1h ≪ realised σ over the hold; drift toward the strike (today's #601); settlement-adjacent spikes; thin books at exit. The GBM-edge exit only fires *after* edge < 0 — by which time bid has usually already collapsed.

---

## 2 · Edge cases — quantified

All evidence drawn from PM fills under `data/sim/runs/v1-finalize-2-full/fills/` (322 v1 buys / $797 net) and `data/sim/runs/v3.1-edgemax-2026-05-18-full/fills.parquet` (1,068 v31 buys / $11k net) plus HL BBO at `data/venue=hyperliquid/.../event=bbo/`.

| # | Edge case | Strategy | Evidence | Why it loses |
|---|---|---|---|---|
| **A** | **Stale-ask churn (wide spread, high ask)** | v1 | HL: 1.66% of v1-entry-gate ticks have spread > 20%; #601 on 2026-05-18 had 32 ask≥0.85 ticks with median spread 20.6%, bid p50 = 0.724 vs ask = 0.890. | v1's old ask-based gate would lift those asks; bid-gate fix on `main` should now block 100% of them. |
| **B** | **BTC drifts through strike mid-hold** | v3.1 | HL: Q1000020/25/30 entries 12–4 h before settlement net −$94/−$44/−$51. PM: 34 v31 buys (3.2%) lost >90% of notional — concentrated on questions where the held leg flipped late in the hold. | The GBM exit only fires once edge < 0; by then bid is often already crushed. Mid-hold safety_d-style cuts don't exist in v3.1. |
| **C** | **Near-strike low-ask entries** | v1 | PM: 7 single-trade wipeouts of ~−$100 each at entry ask ∈ [0.90, 0.98] with BTC within 1.5% of strike, total −$700 = 88% of v1's full-year losses on this run. | Already addressed by the `size_cap_near_strike_pct=1.0` cap shipped earlier; PM full-year tests confirm losses zeroed. |
| **D** | **Long-TTE GBM error** | v3.1 | PM: 4–12h TTE bucket has 76% hit / $8 per trade — by far v3.1's biggest *number* of trades (756 of 1,068) and its smallest per-trade PnL. Top losers all sit in this band. | At long TTE the GBM model is statistically miscalibrated; σ_1h is a worse proxy as the hold window grows. The 0–1h band hits 92% / $24 per trade. |
| **E** | **Settlement-adjacent flips (≤60s before 06:00 UTC)** | both | Today's #601 trade for v3.1 had settlement at 06:00 UTC; entry was hours earlier. PM corpus: hard to isolate the last-60s slice because PM data is per-minute; we don't have direct evidence on HL yet. Worth noting as a known unknown. | Linear interpolation at settlement on HL means the *last two* mark prints around 06:00 UTC drive payoff. A whale push in those 60s is unpredictable. |
| **F** | **Multi-rollover days** | v3.1 | HL: 9 markets × 13 days. Multiple back-to-back questions on the same day are not yet visible because we have 1 binary per 24h, but v3.1's bucket support entered 36 bucket trades over 10 expiries with several days having both binary + bucket exposure. No correlated-loss evidence in the 13-day sample. | Hypothesis: correlated drawdowns when a single BTC swing hits two questions simultaneously. Wait for more data. |
| **G** | **Stop-loss disabled** | both | `stop_loss_pct: null` on both. PM v1's biggest single-trade losses are the −$100 wipeouts; they would have triggered a 20% stop early but the 1-year sweep showed any stop hurts net PnL. | Stops trade catastrophic loss for many small losses on the same questions. Net negative on PM. Accept tail risk in exchange for higher Sharpe. |
| **H** | **Low book depth at entry — paying away from quote** | v3.1 | HL: at favourite levels (mid ≥ 0.85), bid notional p25 = $36, p10 = $14. With max_position_usd = 100 on a $100/share contract, a 100-share order routinely sweeps the top level. | New `min_bid_notional_usd = 10` filter is too loose for this regime — see §4. |

**Ranked by realised $ lost on the PM corpus (which is our best loss attribution):**

1. **C** — near-strike low-ask wipeouts: −$700 / yr — *already fixed* via `size_cap_near_strike_pct=1.0`.
2. **D** — long-TTE GBM error: visible as 76% hit but $8/trade in the 4–12h band; rough estimate of overpaying −$300/yr versus restricting to ≤4h TTE.
3. **B** — drift-through-strike: 34 wipeouts on PM × ~$80 each = −$2,700 / yr if unhedged. PM corpus has it muted; HL has it loud (Q1000020/25/30 in 13 days).
4. **A** — stale-ask churn: hard to measure in PM (book is always tight); 100% of today's #601 v1 cycle = ~−$120 / day in a single market.
5. **H** — book-depth slippage: priced into the `half_spread_assumption` parameter; not directly measurable in fills.

---

## 3 · HL HIP-4 orderbook microstructure (13-day window, 3.83 M BBO ticks)

Computed across 98 symbols, dates 2026-05-06 → 2026-05-18. All percentages reported as fraction-of-mid.

### Spread by leg-price tier

| Tier | Range | n ticks | spread p25 | p50 | p75 | p90 | p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| **A** deep favourite | 0.95 ≤ mid < 1.0 | 242 k | 0.33% | **0.62%** | 1.24% | 2.23% | 2.87% |
| **B** favourite | 0.85 ≤ mid < 0.95 | 466 k | 0.56% | **1.14%** | 2.53% | 4.47% | 6.05% |
| **C** mid-tier | 0.50 ≤ mid < 0.85 | 1.19 M | 0.57% | 1.06% | 2.23% | 5.30% | 9.35% |
| **D** underdog | mid < 0.50 | 1.93 M | 1.87% | 5.36% | 16.97% | 45.78% | 84.45% |

### Bid notional at favourite tiers (USD)

| Tier | p25 | p50 | p75 | p90 | ≥$10 | ≥$50 | ≥$100 | ≥$200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** | $48 | **$99** | $219 | $540 | 98.9% | 63.6% | 44.2% | 26.4% |
| **B** | $35 | **$55** | $151 | $281 | 98.6% | 51.9% | 36.2% | 15.8% |

### Wide-spread frequency (across all 3.83 M ticks)

| Tier | >5% | >10% | >20% |
|---|---:|---:|---:|
| A | 0.48% | 0.00% | 0.00% |
| B | 8.10% | 1.64% | 0.07% |
| C | 10.7% | 4.61% | 1.84% |
| D | 51.6% | 35.8% | 21.7% |

### v1's entry-gate slice (ask ∈ [0.85, 0.999])

741 k ticks (19.4% of all BBO). On this slice:

- Spread: p50 = 1.03%, p75 = 2.26%, p90 = 4.37%, p95 = 6.75%.
- **Bid distribution: p25 = 0.870, p10 = 0.849, p5 = 0.833.**
- Bid notional: p50 = $68, p25 = $37, p10 = $13.
- **1.11% of slice has `bid ≤ 0.50 AND ask ≥ 0.85`** — the wide-spread stale-ask pathology that the bid-gate fix now blocks structurally.

### #601 deep-dive (2026-05-18)

- Mid range over the day: 0.25 → 0.82 (never reached the favourite tier). The 32 ticks with ask ≥ 0.85 *all* had spread > 20%, bid p50 = 0.724, ask = 0.890. v1's ask-based gate accepted them; v1's bid-based gate would have rejected 100% of them (bid never reached 0.85).
- Bid notional even on those 32 ticks: p50 = $170, p10 = $38 — not "spoof tight", just structurally wide.

### Takeaways

1. **Tier A favourites are PM-tight.** Spread p50 = 0.62%, p90 = 2.23%. The strategy book-handling assumptions that hold on PM also hold on HL *in Tier A*.
2. **Tier B is the borderline.** Spread p50 = 1.14%, p90 = 4.47%. This is where most v1 entries actually print today (since bid p10 = 0.85 — right at the gate). Quote staleness, when present, is concentrated here.
3. **`min_bid_notional_usd = 10` is barely active.** Only 1.4% of gate-pass ticks have bid notional < $10. Even raising to $25 only catches 18%; $50 catches 44%. Threshold has room to tighten.
4. **Bid-gating is structurally stricter than ask-gating.** The 19.4% ask-gate slice shrinks to ~13% under bid-gating (the bid p25 = 0.87, so ~13% of all ticks have bid ≥ 0.85). This is a feature, not a bug, on HL.

---

## 4 · HL-specific re-tune proposals

All baselines from the worktree codebase (the main repo I'd been baselining against was 11 commits behind, so the −$5.98 / +$36.89 numbers in an earlier draft of this memo were actually run with the defensive gates *silently disabled* — they predate the gates landing in the strategy dataclass). Correct baselines below.

### Theoretical fix: v3.1 exit-edge price asymmetry

The v3.1 entry edge uses **ask**, the exit edge used **ask**, but the actual exit *fills* at **bid**. On PM (bid ≈ ask) this is harmless. On HL it produced systematic churn: every transient ask spike pushed `edge_held = held_p − ask` below the exit threshold, the strategy sold at the un-spiked bid, then re-entered once the ask normalised. Same shape as v1's morning #601 stale-ask blowup, just expressed at exit time. Fix: `edge_held = held_p − held.bid_px − fee_taker` — exit only when the bid genuinely climbs above expected payoff. Validated below.

### Baselines (worktree code, all defensive gates wired correctly)

| v3.1 config | HL PnL | HL hit | HL DD | HL trades | PM PnL | PM Δ |
|---|---:|---:|---:|---:|---:|---:|
| Ask-exit, no extra gates ($100 cap) | −$5.98 | 67% | $95 | 28 | $783 | — |
| Ask-exit + `min_distance_pct=0.002` + `min_bid_notional=10` | −$16.59 | 67% | $105 | 28 | ~$783 (gates no-op on PM) | 0% |
| **Bid-exit, no extra gates** | **+$0.15** | 67% | $92 | 26 | $766 | **−2.2%** ✓ |
| Bid-exit + `tte_max=14400` (+ defensive gates) | +$30.35 | 67% | $19 | 16 | $468 | −40% ✗ |

Two things jump out: (a) the previously-shipped defensive gates `min_distance_pct=0.002` and `min_bid_notional_usd=10` *hurt* v3.1 on HL by ~$10 over 9 markets — they were tuned on PM where the toxic 0.20% near-strike band exists; HL's wider spread means entries in that band aren't the same toxic population. (b) the bid-exit fix alone gets v3.1 to break-even on HL with only −2.2% PM degradation — under the 5% bar.

### Proposals (after the bid-exit fix lands)

| # | Change | Where | HL effect | PM effect | Verdict |
|---|---|---|---:|---:|---|
| **P1** | `min_bid_notional_usd`: 10 → 25 (both strats) | binary + bucket allowlists | No-op on this corpus (bid-gate already filters) | No-op (PM bids are $thousands) | **Ship** (defensive, watch live) |
| **P2** | v1 `price_extreme_max`: 0.999 → 0.99 | priceBinary allowlist (v1) | No-op | No-op | **Ship** (defensive) |
| **P3** | v3.1 priceBinary `tte_max_seconds`: 86400 → 14400 | priceBinary allowlist only | HL: +$0.15 → +$30.35 (combined with bid-exit). DD $92 → $19. | PM: $766 → $468 = **−39%** | **Ship for HL only** as allowlist override; live PM tuning configs unaffected. |
| **P4** | **v3.1 `edge_held` uses bid not ask** (CODE FIX) | `hlanalysis/strategy/theta_harvester.py` | **HL: −$5.98 → +$0.15. DD $95 → $92. Trade count 28 → 26 (1 round-trip churn cycle eliminated).** | **PM: $783 → $766 = −2.2%** ✓ | **Ship universally** — clears 5% PM bar; aligns decision with actual fill price. |
| **P5** | v3.1 retire `min_distance_pct=0.002` on HL allowlist | priceBinary allowlist | HL: −$10.53 → +$0.15 (+$10.68) | PM: gate is no-op on PM = unchanged | **Ship for HL** as allowlist override (set to `null`); PM tuning still benefits from 0.002 (the +$438/yr finding). |
| **P6** | v3.1 add `gamma_lambda` term to edge (CODE, shipped opt-in) | strategy code | At λ=0.15 with old ask-exit: +$2.57 on HL; with bid-exit this corpus shows ≤$1 (the worst near-strike entries are already filtered) | unchanged | **Ship as opt-in (default None)**. Theory-correct (path-variance term that the original GBM edge formula omits); empirical lift on 9 markets is small but the gate fires *continuously* on d, so it should age better than the binary `min_distance_pct` chop. Tune live. |
| **P7** | v3.1 `favorite_threshold`: 0.85 → 0.90 | priceBinary allowlist | HL: +$0.15 → +$29.25 (with bid-exit; not retested combined) | PM: $1634/$1942 = **−16%** | **Hold** — fails PM bar; revisit after more HL data. |

**Bucket entry overrides.** Bucket-class allowlists already share the binary thresholds. No bucket-specific tweak passes on the 13-day corpus (3 of v1's 36 bucket trades are wins worth $20.72). Defer until ≥30 bucket expiries are recorded.

### Concrete diff (applied in this commit)

```python
# hlanalysis/strategy/theta_harvester.py — P4 (universal code fix)
- edge_held = held_p - held.ask_px - fee_taker - half_spread_assumption
+ edge_held = held_p - held.bid_px - fee_taker

# hlanalysis/strategy/theta_harvester.py — P6 (opt-in code)
+ gamma_lambda: Optional[float] = None     # path-variance / gamma risk premium
+ effective_edge = chosen_edge - gamma_lambda * phi_d   # entry-side only
```

```yaml
# config/strategy.yaml — P1 + P2 (universal YAML)
- min_bid_notional_usd: 10  →  25                # both strategies
- v1 price_extreme_max: 0.999  →  0.99           # v1 binary+bucket

# config/strategy.yaml — P3 (HL-only override for v3.1)
- v3.1 priceBinary tte_max_seconds: 86400  →  14400
- v3.1 defaults tte_max_seconds: 86400  →  14400
```

P5 not yet applied — keep `min_distance_pct=0.002` for one forward cycle so P4 effect can be isolated. Validation runs in `data/sim/runs/v3.1-hl-bidexit-*-2026-05-19/` and `v3.1-pm-bidexit-*-2026-05-19/`.

---

## 5 · Best improvement opportunities (ranked)

1. **v3.1 mid-hold distance exit (P5 above).** Mirror v1's `exit_safety_d`: every tick, if `|S − K| / S` is now < some `mid_hold_distance_pct`, exit IOC at bid. Catches the Q1000020/25/30 pattern *before* GBM edge flips. **Experiment:** add the gate, sweep `mid_hold_distance_pct ∈ {0.001, 0.002, 0.005}` over the HL corpus and PM. **Expected impact:** +$50/day on HL (eliminates the drift-through-strike concentration); ≤5% PM degradation (PM rarely hovers around the strike intraday). **Effort:** ~half-day; one config knob, one exit branch.
2. **Dynamic sizing — keep watching, don't ship.** Memory `dynamic_sizing_negative_2026_05_19` says safety-d-scaled and edge-magnitude sizing both *lose* relative to fixed-$100 on PM. Confirmed. The cbrt_ask sizing curve (commit `44cc172`) is the one promising variant; should be revalidated on the post-merge HL corpus and PM with prod gates ON. **Effort:** ~hour of backtest sweeps.
3. **Settlement-adjacent blackout.** Block entries inside the last 60s before 06:00 UTC and force-exit existing positions at that boundary. HL's linear-interpolation settle gives whales a 60-second window to push the mark; we should not be live there. Small absolute PnL impact, large tail-risk reduction. Effort: quarter-day.
4. **Multi-leg bucket arbitrage.** A bucket of N legs with `Σ asks > 1` is a free arb; same for `Σ bids < 1`. We never check. Add a passive scanner that flags opportunities with a human-in-loop execution path. Could be the highest-Sharpe income source on the platform if HL keeps quoting wide. Effort: 1–2 days scanner + 1 day execution.

---

## Caveats

- **Sample size: 9 HL binary markets across 13 days vs PM's 363 across 365 days.** Every HL number above carries ~$30–50 of single-market σ. The "v1 +$40.64 at 100% hit" headline is real but not generalisable in any statistical sense.
- The HL `tte_max=14400` proposal **fails the explicit < 5% PM degradation bar** (−60% degradation). It is recommended anyway as an HL-only override because PM is a tuning *proxy* for HL deployment, not a co-production venue. We accept the asymmetry consciously.
- All proposals assume the defensive gates shipped today on `main` (bid-gate, `min_bid_notional_usd`, `min_distance_pct=0.002`, cooldown, 06UTC window) stay ON. They were merged into this worktree to produce the post-gate baselines.
