# Alternative Momentum / Mean-Reversion Indicators for v3.6

**Context**: v3.5 tilt uses `(1 - alpha * score)` to scale `edge_buffer`.
Tested: `z_ret` (winner, +$57 PM, transfers HL), `ma_sigma` (best PM +$98, HL fails),
`rsi` (inconsistent), `hurst_ou/R-S` (no signal). Gate mode was dead on all four.
This note surveys what to try next.

---

## Candidate Indicators

### 1. OU Half-Life z-Score (OU-Z)

**Why this, not R/S Hurst**: R/S Hurst estimates global memory over all lags — it answers
"is this series fractal?" not "how strongly is it currently pulling back?". The
OU z-score answers the right question: "how far is price from equilibrium, scaled by
how fast it historically reverts?" This is ma_sigma in the same coordinate system but
with a data-driven mean-reversion anchor instead of an arbitrary rolling window.

**Formula**:

```
kappa = -log(beta_1) / dt           # AR(1) OLS: delta_p = beta_0 + beta_1 * p_lag + eps
mu    = beta_0 / (1 - beta_1)       # long-run mean
sigma_eq = sqrt(var(eps) / (1 - beta_1^2))
ou_z  = (p_last - mu) / sigma_eq    # signed: + means above mean => expect fall
score = clip(ou_z / 2.0, -1, 1)     # normalize to [-1,+1]
```

**Data requirement**: 30–120 hourly closes of the BTC spot or perp price (already
in recorder). Fit AR(1) on a rolling window matching `lookback` param.

**Why it might beat ma_sigma**: ma_sigma uses a fixed rolling MA and rolling std —
both tunable hyperparameters. OU-Z instead *estimates* the equilibrium level (mu)
and the natural scale (sigma_eq) directly from the data. The equilibrium is more
stable than a trailing MA when price has been drifting. In the GBM framework we
already use Itô-corrected drift — OU-Z is the mean-reversion analogue: same
coordinate system, path-adaptive anchor.

Academic backing: Avellaneda & Lee (2010, *Statistical Arbitrage in the US Equities
Market*) use exactly this z-score for equity stat-arb entry/exit sizing. Hudson &
Thames ArbitrageLab confirms the half-life = -log(2)/kappa formula is standard and
well-behaved when kappa > 0 (mean reverting). SSRN 5310321 warns that half-life
itself (a point estimate of kappa) can be misleading — the z-score using the full
OU parameterization is more robust.

**Kill signals**: Fails when kappa ≤ 0 (trending/unit-root regime). Guard: if AR(1)
beta_1 ≥ 0.99, fallback to ma_sigma or zero score.

---

### 2. Signed CVD-Z (Cumulative Volume Delta, z-normalized)

**Formula**:

```
cvd_t  = cumsum(volume * sign(close - open))  # or tick-rule aggressors
cvd_z  = (cvd_t - rolling_mean(cvd, lb)) / rolling_std(cvd, lb)
score  = clip(cvd_z / 2.0, -1, 1)   # + means buy pressure => favor UP
```

**Data requirement**: Tick-level aggressor tags already recorded on the BTC perp
stream. Aggregate into N-minute buckets; feed as additional feature alongside price.

**Why it might beat z_ret / ma_sigma**: z_ret and ma_sigma are price-only; CVD-Z
adds *order-flow* information. Price can remain flat while buy pressure accumulates
(stealth accumulation) — CVD divergence from price is a classic leading indicator.
Practitioners at Buildix and VisualHFT confirm CVD divergence precedes price moves
on HL/Binance. For 24h binaries, a persistent one-sided CVD over the prior few hours
plausibly increases or decreases the probability of an UP resolution.

Academic backing: Cont, Kukanov & Stoikov (2014, *The Price Impact of Order Book
Events*, J. Financial Econometrics) formalise that OFI is the primary predictor of
short-term mid-price moves. The Hybrid VAR-FNN paper (arXiv 2411.08382) shows
superlinear OFI predictability in high-volatility regimes.

**Kill signals**: Signal decay is fast — raw OFI is accurate over seconds to minutes
(R² ~3%, hit rate 53% in Markwick 2022 BTC analysis). Aggregated over hours it
becomes much noisier. If lookback is short (< 1h), transaction cost dominates and
signal is useless for our 24h horizon. Recommendation: aggregate into 4–8h buckets;
test lb=1 (≈ last 4h CVD window). Do NOT use raw tick-level CVD as a signal for
24h binary resolution.

**Structural advantage specific to our setup**: The BTC perp tape is always-on;
CVD captures the *directional conviction* of the marginal participant. A 24h binary
asking "will BTC close up?" is partly a question about whether buyers or sellers have
controlled the past 8h — CVD-Z formalizes that intuition.

---

### 3. Funding Rate Z-Score (FR-Z) as Regime Tag

**Formula**:

```
fr_z   = (funding_rate_current - rolling_mean(fr, lb)) / rolling_std(fr, lb)
score  = clip(fr_z / 2.0, -1, 1)   # + = extreme longs = contrarian short signal
```

**Data requirement**: 8h funding snapshots from HL REST (`/info` fundingHistory
endpoint). Already partially in recorder for hedge data. HL perp funding is 1h;
Binance USDT perp is 8h.

**Why it might add info**: Funding rate measures the *willingness of longs to pay
to stay long* — a direct cost-of-carry signal for sentiment extremes. When funding
is extreme positive, overleveraged longs are crowded: even a small adverse move
triggers cascading liquidations, making an UP resolution less likely. The contrarian
direction (high positive FR => favor DOWN) has practitioner backing from CryptoQuant,
QuantJourney, and the Gate.com derivatives guide. The signal is regime-level (works
at extremes, near-zero is noise) which is a good fit for the tilt-not-gate mode we
already use.

Academic / practitioner backing: QuantJourney Substack confirms "Sudden funding spikes
often precede sharp price movements" and notes the signal "strengthens at extremes."
The Phemex Academy page quantifies: >0.1%/8h = extreme long; <-0.1%/8h = extreme
short. Gate derivatives guide recommends combining with open interest to filter false
positives.

**Structural advantage specific to our setup**: Funding is orthogonal to price — two
assets can have the same price move but completely different funding. FR-Z adds a
sentiment/positioning dimension that ma_sigma and z_ret cannot capture. It should be
near-zero correlation with those signals except during momentum extremes, where it
provides the most value.

**Kill signals**: Funding rate as a standalone predictor has no published alpha beyond
practitioner observation; no rigorous academic backtest exists for binary markets.
Known problem: during ranging markets (low |FR-Z|), the signal is useless and may
add noise. Recommendation: use as a *secondary tilt* or *regime gate* only when
|fr_z| > 1.5, else pass-through (score = 0).

---

### 4. Realized Skewness (RS-Z)

**Formula**:

```
rs    = (1/N) * sum((r_i - r_bar)^3) / realized_vol^3  # 5-min returns over lookback
score = clip(-rs / 2.0, -1, 1)   # negative rs (left tail heavy) => favor DOWN
```

**Data requirement**: 5-min OHLCV already recorded. Lookback 1–3 days.

**Why it might add info**: Realized skewness captures jump asymmetry — the sign and
magnitude of tail events in the recent return distribution. In the path-variance-aware
class (topic 1), skewness is complementary to BV and RV: if recent returns have a
strong left tail (negative skew), the underlying may be in a fragile uptrend — more
"up moves with a few hard drops." This is orthogonal to the level-based ma_sigma.

Amaya et al. (2015, *Does Realized Skewness Predict the Cross-Section of Equity
Returns?*, JFE) find significant cross-sectional return predictability from weekly
realized skewness. Bali et al. (2021, *Realized Skewness and Short-Term Predictability
for Aggregate Volatility*, Economic Modelling) confirm short-term vol predictability.
Importantly, skewness predicts the *volatility* of the next period (negative skew →
higher vol), which for a binary with a GBM overlay means a wider d-statistic range
and potentially higher edge on the current market price.

**Kill signals**: Daily return horizon studies find *no* individual-stock directional
predictability (Bali et al. 2021 explicitly notes this). Cross-section works; time
series directional signal for a single asset does not. For our single-asset (BTC)
24h binary, realized skewness is more useful as a *vol regime* input to adjust the
GBM σ estimate than as a directional entry tilt. Rank: lower priority for v3.6
direction tilt, but consider as a vol-scaling upgrade to v3.2 BV clock.

---

### 5. Jump Ratio (JR) Tilt

**Formula**:

```
RV   = sum(r_i^2)                    # realized variance
BPV  = (pi/2) * sum(|r_i| * |r_{i+1}|)  # bipower variation (already computed)
JR   = max(RV - BPV, 0) / RV        # fraction of variance from jumps, [0,1]
score = clip(1.0 - 2*JR, -1, 1)     # JR near 0 = pure diffusion = tilt with ma_sigma
                                     # JR near 1 = jump-dominated = reduce tilt / flatten
```

**Data requirement**: BPV is already part of the v3.2 volclock σ-estimator. JR is
one extra line.

**Why it might add info**: Not a directional indicator — it's a *signal quality filter*.
When recent variance is jump-dominated (JR high), the continuous drift signal (ma_sigma,
z_ret, OU-Z) is unreliable because the current price level reflects a jump, not drift.
JR acts as a *trust weight* on the other indicators: `final_score = signal * (1 - JR)`.
Barndorff-Nielsen & Shephard (2004, *Power and Bipower Variation*, J. Financial
Econometrics) establish BV/RV decomposition rigorously. Corsi, Pirino & Renò (2010,
*Threshold Bipower Variation*, JoE) confirm jump vs continuous separation improves
forecasting.

**Kill signals**: None — this is a meta-signal (quality weight), not a standalone
predictor. Risk: if jumps are persistent (trend), suppressing the signal every day
would reduce all tilts. Add hysteresis: only suppress when JR > 0.4 over the last
window.

---

### 6. Options Skew (25Δ RR)

**Formula**:

```
skew_rr  = IV_call_25d - IV_put_25d   # from Deribit/Binance options
score    = clip(skew_rr / 5.0, -1, 1) # + = calls pricier = bullish implied sentiment
```

**Data requirement**: Deribit or Binance options API; not currently in recorder.
Would require adding a new data source.

**Why it might add info**: Options skew is the market's *priced probability* of
tail outcomes. A strongly negative RR (puts pricier) says the options market assigns
more probability mass to down moves — exactly the signal we want for fading a binary
priced at 50%.

Deribit Insights (2024) analysis of 4 years of BTC vol regimes shows that selling
RR offers best risk-adjusted return in BTC options, implying the skew signal has
predictive content. Flyonthewall and MenthorQ confirm RR as a sentiment leading
indicator at extremes.

**Kill signals**: Strike-Watch analysis confirms 25Δ RR literature is equity-focused
and has not been rigorously backtested on BTC short-horizon binary prediction.
Implementation cost is high (new data source, options APIs are unreliable/pricey).
Rank: research interest only for v3.6; not recommended for near-term implementation.

---

### 7. ETH/BTC Ratio z-Score (Cross-Asset Regime)

**Formula**:

```
ratio   = price_ETH / price_BTC
ratio_z = (ratio - rolling_mean(ratio, lb)) / rolling_std(ratio, lb)
score   = clip(ratio_z / 2.0, -1, 1)  # + = ETH outperforming = risk-on = favor BTC UP
```

**Data requirement**: ETH perp price (already in recorder alongside BTC).

**Why it might add info**: ETH/BTC flipping momentum historically signals broad
crypto risk-on/risk-off rotations. If ETH is surging vs BTC, it implies a risk-on
regime where BTC bulls are already positioned — mild momentum signal. Cross-asset
regime signals are orthogonal to single-asset price signals and have low correlation
with z_ret and ma_sigma.

**Kill signals**: The ETH-BTC correlation averaged 0.89 in 2025; the ratio is
slow-moving. On a 24h binary horizon, the ratio z-score is likely a very low-frequency
signal that provides minimal incremental edge over price-only indicators. Additionally,
ETH can outperform BTC for idiosyncratic reasons (DEX volume, protocol upgrades) fully
unrelated to BTC directional outcome. Rank: low priority.

---

### 8. VPIN (Volume-Synchronized Probability of Informed Trading)

**Formula**:

```
# Volume buckets of size V_bar
buy_vol_bucket  = sum of aggressor buy volume per bucket
sell_vol_bucket = V_bar - buy_vol_bucket
vpin = rolling_mean(|buy_vol - sell_vol| / V_bar, n=50)  # 50-bucket window
score = clip((vpin - 0.3) / 0.4, -1, 1)  # high VPIN = stressed = reduce edge
```

**Data requirement**: Aggressor-tagged tick data (already recorded).

**Why it might add info**: VPIN is NOT a directional signal — it's an *adverse
selection* / toxicity gauge. High VPIN means informed flow dominates; for a market-
maker this means "I'm on the wrong side of an informed trader." In the binary context:
high VPIN near the entry time → the market has better information than our model
→ widen edge_buffer (reduce aggressiveness), not tilt direction.

Academic backing: Easley, López de Prado & O'Hara (2012, *Flow Toxicity and Liquidity
in a Flash Crash*) originate VPIN. VPIN predicted the 2010 Flash Crash. For BTC spot
(Lucas Astorian analysis, August 2020–June 2021), VPIN achieves ROC-AUC 0.55 only
in high-vol regimes; 0.49 overall (below random). This is a clear signal that VPIN
works for risk management (vol scaling) not directional prediction.

**Kill signals**: As a directional signal, VPIN on BTC spot has near-zero predictive
power under normal conditions (AUC 0.49 in academic study). Use ONLY as a risk-scaling
overlay to *widen* edge_buffer during high-toxicity windows, not to tilt direction.

---

### 9. Signed Drift Ratio (Continuous-to-Total Variance Tilt)

**Formula**:

```
RV   = sum(r_i^2)
BPV  = (pi/2) * sum(|r_i| * |r_{i+1}|)
C    = min(BPV, RV)         # continuous component (jump-robust)
drift = sum(r_i) / sqrt(C)  # drift scaled by jump-robust vol = jump-robust d-stat
score = clip(drift / 2.0, -1, 1)
```

**Data requirement**: Same as BPV (v3.2 volclock). One extra sum.

**Why it might beat z_ret**: z_ret uses raw cumulative return divided by reference
std (4h std). The Signed Drift Ratio replaces the denominator with the jump-robust
continuous volatility estimate — same idea as our Itô-corrected d-statistic, but
applied to the *indicator* rather than just the model. This is the path-variance-aware
indicator from research topic 1: jump-robust realized drift.

Barndorff-Nielsen & Shephard (2004) establish that BPV = continuous IV; dividing
realized drift by sqrt(BPV) instead of sqrt(RV) removes jump contamination from the
momentum signal. This is directly analogous to how we use BV-σ instead of sample-σ
in the volclock. The signal should be more stable than z_ret in jump-heavy BTC
regimes.

**Kill signals**: Same direction as z_ret — this is an upgrade to z_ret, not a
replacement. Risk of overfitting if lookback window is tuned separately from the
volclock window. Recommendation: inherit the same window as the volclock BV estimator.

---

### 10. Accumulated Funding z-Score (AF-Z, slower regime signal)

**Formula**:

```
af   = rolling_sum(funding_rate, 30d)  # accumulated cost of carry
af_z = (af - rolling_mean(af, 90d)) / rolling_std(af, 90d)
score = clip(-af_z / 2.0, -1, 1)  # high accumulated funding = crowded longs = contrarian
```

**Data requirement**: Same as FR-Z (1h or 8h funding snapshots).

**Why separate from FR-Z**: FR-Z is a snapshot of current positioning; AF-Z measures
how *long* the market has been in this regime. Sustained high funding (30d accumulated)
signals structural long bias — a more powerful contrarian indicator than a single
spike. CryptoQuant's "Accumulated Funding" metric is a standard practitioner tool.

**Kill signals**: 30-day horizon is very slow for a 24h binary — this is a regime
filter, not an entry signal. Use to gate the *alpha* of FR-Z (only apply FR-Z tilts
when AF-Z agrees), not as a standalone score.

---

## Rankings

### (a) Likely Incremental Info Over ma_sigma / z_ret

| Rank | Indicator | Incremental Info | Reason |
|------|-----------|-----------------|--------|
| 1 | **OU-Z** | High | Same coordinate system as ma_sigma but data-adaptive equilibrium; directly measures mean-reversion force |
| 2 | **Signed Drift Ratio** | High | Jump-robust upgrade to z_ret; uses BPV already computed |
| 3 | **CVD-Z** | Medium-High | Orthogonal data source (order flow ≠ price); adds conviction signal |
| 4 | **FR-Z** | Medium | Sentiment/positioning; truly orthogonal to price |
| 5 | **Jump Ratio** | Medium | Meta-weight, not directional; reduces noise in jump regimes |
| 6 | **Realized Skewness** | Low-Medium | Vol regime, not direction; better as volclock input |
| 7 | **VPIN** | Low | Risk mgmt / edge-buffer scale, not direction |
| 8 | **ETH/BTC Ratio** | Low | Slow-moving, high correlation with price signals |
| 9 | **AF-Z** | Low | Regime filter only; use as gate on FR-Z |
| 10 | **Options Skew (RR)** | Unknown | High potential but requires new data source |

### (b) Ease of Implementation (Given Existing Pipelines)

| Rank | Indicator | Effort | Notes |
|------|-----------|--------|-------|
| 1 | **Signed Drift Ratio** | Trivial | BPV already computed; one new sum |
| 2 | **Jump Ratio** | Trivial | Direct from existing BPV and RV |
| 3 | **OU-Z** | Low | AR(1) OLS on price series; ~10 lines |
| 4 | **Realized Skewness** | Low | 5-min returns already available |
| 5 | **FR-Z** | Low-Medium | Need funding history pull; recorder already has endpoint |
| 6 | **CVD-Z** | Medium | Aggressor tags in recorder but aggregation pipeline needed |
| 7 | **AF-Z** | Medium | Same as FR-Z; slower window |
| 8 | **ETH/BTC Ratio** | Medium | ETH price in recorder; need to add ratio series |
| 9 | **VPIN** | Medium-High | Volume bucket logic; more complex than OFI |
| 10 | **Options Skew** | High | New data source (Deribit API); not worth for v3.6 |

---

## Top-3 Picks for v3.6

### Pick 1: OU-Z (Ornstein-Uhlenbeck z-Score)

**Why**: This is the right version of the mean-reversion signal that ma_sigma was
approximating. ma_sigma won PM by 2× over z_ret because it uses σ-normalized distance
— the same coordinate system as the GBM d-statistic. OU-Z does this better: the
denominator is the equilibrium standard deviation estimated from the AR(1) process,
not a rolling std with arbitrary window. It directly answers "how many equilibrium
standard deviations is price from its OU mean?" — the most natural measure of
mean-reversion excess.

**Pseudocode**:

```python
def ou_z_score(prices, lb=48):  # lb in hours, e.g. 48h lookback
    p = prices[-lb:]
    dp = np.diff(p)
    p_lag = p[:-1]
    # OLS: dp = beta0 + beta1 * p_lag
    beta1, beta0 = np.polyfit(p_lag, dp, 1)
    if beta1 >= 0:           # non mean-reverting; guard
        return 0.0
    mu = -beta0 / beta1      # long-run mean
    kappa = -beta1           # speed of reversion (kappa > 0)
    sigma_eq = np.std(dp - beta0 - beta1 * p_lag) / np.sqrt(2 * kappa)
    ou_z = (p[-1] - mu) / max(sigma_eq, 1e-8)
    return float(np.clip(ou_z / 2.0, -1.0, 1.0))
    # + score = above OU mean = expect fall = tilt AGAINST current favorite if favorite is UP
```

---

### Pick 2: Signed Drift Ratio (Jump-Robust Momentum)

**Why**: This is a drop-in upgrade to z_ret with near-zero implementation cost —
BPV is already computed in the volclock. The key property: dividing drift by
sqrt(BPV) instead of sqrt(RV) removes jump contamination from the momentum signal.
If BTC jumped up +5% on news, z_ret sees a large positive drift; Signed Drift Ratio
sees a smaller signal because the jump variance is stripped from the denominator.
This is the path-variance-aware indicator most aligned with the existing GBM model.

**Pseudocode**:

```python
def signed_drift_ratio(returns, lb=48):  # returns = 1h log-returns
    r = returns[-lb:]
    rv = np.sum(r ** 2)
    bpv = (np.pi / 2) * np.sum(np.abs(r[:-1]) * np.abs(r[1:]))
    c = max(min(bpv, rv), 1e-10)  # continuous component
    drift = np.sum(r) / np.sqrt(c)
    return float(np.clip(drift / 2.0, -1.0, 1.0))
    # + = positive drift relative to jump-robust vol = momentum UP
```

---

### Pick 3: CVD-Z (Cumulative Volume Delta, z-normalized, 4h bucket)

**Why**: The only candidate that uses a different data dimension (order flow, not
price). z_ret and ma_sigma and OU-Z are all price-derived and will share information
in trend regimes. CVD-Z captures *directional conviction of aggressors* — persistent
buy-side delta even when price is flat (stealth accumulation) or persistent sell-side
delta during apparent price stability. For a 24h binary, the CVD over the prior 4–8h
before entry adds non-price evidence about which side is in control.

**Pseudocode**:

```python
def cvd_z_score(aggressor_buy_vol, aggressor_sell_vol, lb=6):  # lb in 4h buckets
    # aggressor_buy/sell: arrays of 4h-binned volumes
    delta = aggressor_buy_vol[-lb:] - aggressor_sell_vol[-lb:]
    cvd = np.cumsum(delta)
    if len(cvd) < 4:
        return 0.0
    mu_cvd = np.mean(cvd)
    sd_cvd = max(np.std(cvd), 1e-8)
    score = (cvd[-1] - mu_cvd) / sd_cvd
    return float(np.clip(score / 2.0, -1.0, 1.0))
    # + = net buy pressure accumulating = momentum UP
```

**Implementation note**: requires aggregating aggressor tags into 4h buckets in the
recorder's on-disk parquet. The HL perp tape already has aggressor side per trade.

---

## Known Kill Signals from Literature

1. **Raw tick-level OFI / order-book imbalance for 24h binary**: Predictive decay
   within seconds to 10 minutes (Markwick 2022, Towardsdatascience BTC OBI study).
   At the 24h horizon, raw OBI has zero standalone alpha after transaction costs.
   Only use aggregated (4h+) CVD, not raw orderbook imbalance.

2. **R/S Hurst exponent as a tilt signal**: Tested and failed (our own v3.5 result).
   The R/S estimator is biased for short samples and has no directional content.
   The academic literature (MDPI 2024, Macrosynergy review) uses it only as a
   regime classifier (H < 0.5 / H > 0.5), not a signed score. We should use
   regime classification via Hurst only to switch between OU-Z (mean-reversion) and
   Signed Drift Ratio (momentum) — not as a direct signal.

3. **RSI on 24h binary**: Confirmed dead in our v3.5 sweep. Literature consistent:
   RSI is a lagging rescaling of the price move already captured by z_ret; no
   independent information.

4. **Options 25Δ RR (skew) as a short-term directional signal**: Strike-Watch
   analysis confirms the equity-based skew-as-leading-indicator framework has not
   been validated for BTC binary horizons. The signal requires new data infrastructure
   and has unquantified crypto-specific failure modes.

5. **VPIN as a directional signal for individual 24h entries**: ROC-AUC = 0.49 on
   BTC spot under normal conditions (Astorian 2020–2021). VPIN is a risk-management
   tool (adversity gauge), not an alpha signal. Using it to tilt direction would
   add noise.

6. **Realized skewness as a *directional* tilt for a single asset**: Bali et al.
   (2021) explicitly find no daily directional predictability from realized skewness
   for individual assets. The cross-sectional result (Amaya et al. 2015) doesn't
   transfer to single-asset time series. Use realized skewness only to *adjust σ*
   (vol regime), not to tilt entry direction.

---

## References

### Academic Papers
- Barndorff-Nielsen, O.E. & Shephard, N. (2004). *Power and Bipower Variation with
  Stochastic Volatility and Jumps*. Journal of Financial Econometrics.
  https://public.econ.duke.edu/~get/browse/courses/883/Spr16/COURSE-MATERIALS/Z_Papers/BNSJFEC2004.pdf
- Corsi, F., Pirino, D. & Renò, R. (2010). *Threshold Bipower Variation and the
  Impact of Jumps on Volatility Forecasting*. Journal of Econometrics.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1115783
- Amaya, D., Christoffersen, P., Jacobs, K. & Vasquez, A. (2015). *Does Realized
  Skewness Predict the Cross-Section of Equity Returns?* Journal of Financial
  Economics. https://www.sciencedirect.com/science/article/abs/pii/S0304405X15001257
- Cont, R., Kukanov, A. & Stoikov, S. (2014). *The Price Impact of Order Book
  Events*. Journal of Financial Econometrics.
- Avellaneda, M. & Lee, J.H. (2010). *Statistical Arbitrage in the US Equities
  Market*. Quantitative Finance.
- Easley, D., López de Prado, M. & O'Hara, M. (2012). *Flow Toxicity and Liquidity
  in a Flash Crash*. The Review of Financial Studies.

### Practitioner References
- Astorian, L. (2021). *Order Flow Toxicity in the Bitcoin Spot Market*.
  https://medium.com/@lucasastorian/empirical-market-microstructure-f67eff3517e0
- Markwick, D. (2022). *Order Flow Imbalance — A High Frequency Trading Signal*.
  https://dm13450.github.io/2022/02/02/Order-Flow-Imbalance.html
- QuantJourney Substack. *Funding Rates in Crypto: The Hidden Cost, Sentiment Signal,
  and Strategy Trigger*.
  https://quantjourney.substack.com/p/funding-rates-in-crypto-the-hidden
- VisualHFT. *Volume-Synchronized Probability of Informed Trading (VPIN)*.
  https://www.visualhft.com/blog/volume-synchronized-probability-of-informed-trading-vpin/
- Hudson & Thames ArbitrageLab. *Half-Life of Mean Reversion*.
  https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/cointegration_approach/half_life.html
- Buildix Blog. *What Is VPIN? Flow Toxicity Detection for Crypto Traders*.
  https://www.buildix.trade/blog/what-is-vpin-flow-toxicity-crypto-trading
