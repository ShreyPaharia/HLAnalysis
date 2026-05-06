# How a perp price turns into MM PnL on the HIP-4 binary

**A from-scratch explainer.**
Ground rules: I assume you can read math but I won't skip the physical
intuition. We'll build it up: what the binary is, how you price it, what each
Greek tells you, how the perp hedge works mechanically, and finally how that
shakes out into expected dollars and a sizing rule. Numbers are the live
ones from our recorded data on 2026-05-05.

---

## 0. The setup

We're looking at one specific market and one specific hedge:

- **Binary**: HL HIP-4 `BTC > 80,930 by 2026-05-06 06:00 UTC`. Two CLOBs:
  `#30` pays $1 USDC if YES wins, `#31` pays $1 if NO wins. They're scaled to
  prices in $[0,1]$ and `bid_yes + bid_no` should sit just under 1
  (notebook 05 confirms it never crossed 1 in our window).
- **Hedge**: HL BTC perp. Linear USD-quoted perp; ticks in $0.5 USD.

Two questions we want answers to:

1. **What's the right price for the binary** if we're given the BTC perp price?
2. **Can we trade in a way that actually generates a positive expected PnL,**
   or are we just transferring risk between books?

The thing that makes the binary not-a-coinflip is volatility. So we'll start
there.

---

## 1. The binary is a digital cash-or-nothing call. What's its fair price?

### 1.1 If BTC didn't move at all

If the BTC price were frozen at $S$, the binary would settle:

- $1 if $S > K$
- $0 if $S \le K$

Its price now would be 1 or 0. No volatility, no math, no edge.

### 1.2 If BTC moves by a fixed Gaussian increment

Now suppose BTC changes by a single step $\Delta = S_T - S_t$ that's normally
distributed: $\Delta \sim \mathcal{N}(0, \sigma_\$^2 \tau)$, where
$\sigma_\$$ is volatility in dollars per $\sqrt{\text{year}}$ and
$\tau = T-t$ is the remaining life in years.

The probability YES wins is

$$\Pr(S_T > K) \;=\; \Pr(\Delta > K - S_t) \;=\; 1 - \Phi\!\Big(\tfrac{K-S_t}{\sigma_\$\sqrt{\tau}}\Big) \;=\; \Phi\!\Big(\tfrac{S_t - K}{\sigma_\$\sqrt{\tau}}\Big)$$

That's already the right shape. As $S_t \to \infty$ → 1; as $S_t \to -\infty$ → 0;
at $S_t = K$ → 0.5.

### 1.3 The Black–Scholes correction: returns are log-normal, not normal

The actual model is that **log-returns** are Gaussian. So
$\ln(S_T/S_t) \sim \mathcal{N}((-\sigma^2/2)\tau,\; \sigma^2 \tau)$ under the
risk-neutral measure with zero rates. The "$-\sigma^2/2$" is the standard
log-normal drift correction (a.k.a. Itô correction).

Solving for $\Pr(S_T > K)$ gives the textbook digital formula:

$$\boxed{\;P_\text{YES}(S, t) \;=\; \Phi(d_2)\,, \quad d_2 = \frac{\ln(S/K) - \tfrac12 \sigma^2 \tau}{\sigma\sqrt{\tau}}\;}$$

Read it like this:

- **$\ln(S/K)$**: how far above (positive) or below (negative) the strike we
  currently are, in log-units. For our state $S=81{,}359, K=80{,}930$:
  $\ln(S/K) = +0.0053$.
- **$\sigma\sqrt\tau$**: the standard deviation of the log-return between now
  and expiry. $\sigma=0.25, \tau=13\text{h}/8760\text{h} = 0.00148\text{ y}$:
  $\sigma\sqrt\tau = 0.0096$. So a 1-σ move of BTC over 13 hours is about
  ±0.96% in log-terms (~$780).
- **$d_2$**: how many σ's we are above the strike, with a small drift
  correction. Positive = ITM YES. For us $d_2 \approx 0.55$.
- **$\Phi(d_2)$**: probability that a standard normal exceeds $-d_2$, i.e. the
  probability YES wins. $\Phi(0.55) \approx 0.71$.

That's the recipe. **At $\sigma=25\%$, the model says YES is worth $0.71$**.

The market is paying $0.635$. So either:

- **Realized σ is overstated** (BTC may move *more* than 25% has implied),
- **The market is paying a vol risk premium** (it always over-prices σ for
  short-dated options because nobody wants to be short gamma into expiry),
- **There's a settlement-rule premium** (oracle source could deviate from
  perp mid in a stress event), or
- **The market is plain wrong** (least likely, but worth keeping in the bin).

We'll quantify each below.

### 1.4 The key knob: σ

**$\sigma$ is the only free parameter** — once you commit to a value, the
price is mechanical. There are two flavours:

- **Realized σ**: from past log-returns of S. We estimate
  $\hat\sigma = \text{stdev}(r_t) \times \sqrt{n}$ where $n$ is the number of
  return-intervals per year. From our HL perp 1-second log-returns over 9 h:
  ~25%–27% annualized.
- **Implied σ**: invert the binary's market price to find the σ that makes
  $\Phi(d_2(\sigma)) = P_\text{market}$. Notebook 06 gives a 39% median.

The gap between these — **σ_imp − σ_real ≈ 14 percentage points** — is the
vol risk premium. It's the lever for the trade.

---

## 2. Greeks: how the binary price moves when the world moves

A "Greek" is a partial derivative of the option price. Each one tells you a
risk you have, and how to neutralise it.

### 2.1 Δ (Delta) — sensitivity to spot

$$\Delta \;=\; \frac{\partial P}{\partial S} \;=\; \frac{\phi(d_2)}{S\,\sigma\sqrt\tau}$$

where $\phi$ is the standard normal pdf.

**What it means**: a $1 move in BTC moves the binary by $\Delta dollars.
This is **always positive** for a YES (binary call) — pushing BTC up makes
YES more likely.

For our state ($S=81{,}359, \sigma=39\%, \tau=13$ h): $\Delta \approx 3.08\times 10^{-4}$
**per $1 of BTC**. So if BTC moves up $1, the binary YES price goes up
0.000308 (3 hundredths of a cent on the $0.635 price).

If you owned 1000 YES contracts (each with $1 max payoff), your dollar
delta exposure is

$$\$\text{-delta} \;=\; 1000 \times 0.000308 \;=\; 0.308\text{ \$/\$}$$

Meaning: a $1 move in BTC moves your portfolio by 30.8 cents. You'd hedge
this by **selling 0.308 USD-notional of BTC perp** — which on HL perp is
$\$0.308 / S = 0.0000038$ BTC per YES contract held. Tiny, but additive.

### 2.2 Γ (Gamma) — sensitivity of Δ to spot

$$\Gamma \;=\; \frac{\partial^2 P}{\partial S^2} \;=\; -\frac{\phi(d_2)\,(d_2 + \sigma\sqrt\tau)}{S^2\,\sigma^2\,\tau}$$

**What it means**: a $1 move in BTC changes your delta by $\Gamma$. So your
hedge needs to be re-done as S moves — that re-hedging is what we call
**dynamic delta hedging**.

For a binary, $\Gamma$ flips sign at the strike:

- **$S < K$**: $\Gamma > 0$ (long YES is long-gamma)
- **$S > K$**: $\Gamma < 0$ (long YES is short-gamma)
- **$|\Gamma|$ is biggest near $S = K$ and grows as $\tau \to 0$** — this is
  the dreaded "**pin risk**".

For our state at $\sigma=39\%$: $\Gamma \approx -9.04\times 10^{-8}$ per $1²
per YES contract.

This sign matters more than its magnitude for what trade we should put on.

### 2.3 Θ (Theta) — time decay

$$\Theta \;=\; \frac{\partial P}{\partial t}$$

How much the binary price changes over a unit of time, holding S and σ
fixed. For a binary, near-strike Θ can be massive close to expiry — the
S-curve sharpens. Far from strike, Θ is small.

### 2.4 ν (Vega) — sensitivity to σ

$$\nu \;=\; \frac{\partial P}{\partial \sigma} \;=\; -\phi(d_2) \cdot d_2 / \sigma$$

Sign depends on $d_2$:

- **$S > K$ (ITM YES, $d_2 > 0$)**: $\nu < 0$. **Higher σ** *lowers* YES
  price.
- **$S < K$ (OTM YES, $d_2 < 0$)**: $\nu > 0$. Higher σ raises YES price.

This is the **opposite** of vanilla calls (always Vega>0). The intuition:
when ITM, more uncertainty pulls the prob toward 50/50; when OTM, more
uncertainty does the same thing but raises the prob away from 0.

For our state: $\nu \approx -0.33$ per unit σ at σ=39%. Meaning, σ goes from
39% to 40% (+0.01) → YES price drops 0.0033 (~0.5% of the price).

---

## 3. The hedging mechanic, step by step

Suppose we're going to **buy 1 YES contract at $0.635**. Steps:

1. **Pay $0.635** to own 1 YES contract. Pays out $1 if BTC > 80,930 at
   06:00 UTC tomorrow, else $0.
2. **Compute current Δ**. At $\sigma_\text{imp}=39\%$, Δ ≈ $3.08\times10^{-4}$ /\$
   per contract (we use the implied σ because we want to hedge consistently
   with the price we paid).
3. **Hedge**: sell short $\Delta \times 1 \times \$1\text{ payoff}$ worth of
   BTC perp. For 1 contract that's $0.000308 of BTC perp = $S \times 0$ basically.
   So we'd never actually hedge a single contract — we hedge thousands at a
   time. For 10,000 contracts (max payoff $10,000): short 10,000 × 3.08e-4
   = **$3.08 of BTC perp** (~3.8e-5 BTC).
4. **As S moves, Δ changes** (that's Γ kicking in). We need to keep our perp
   short equal to the current Δ. So as S goes up: Δ for ITM YES is
   *decreasing* (Γ<0), so we need to **buy back** some perp short. As S
   goes down: Δ is *increasing*, sell more perp. **Always opposite to S
   move when short gamma.**
5. At expiry: binary settles to $0 or $1. We close the perp. Total PnL is
   the sum of:
   - Payoff received: 0 or 1
   - Premium paid: $0.635
   - Net hedge PnL: sum of all the perp re-balancing trades

That sum is what we'll analyze next.

### 3.1 Why the hedge "works" — the BS theorem in three lines

If we hedge **at the same σ that priced the binary** (here: σ_imp), and S
follows lognormal dynamics with vol σ_real, then by Itô on $V(S,t)$ and
standard manipulations:

$$\mathrm{d}\text{PnL} \;=\; \frac{1}{2}\,\Gamma\,S^2\,(\sigma_\text{real}^2 - \sigma_\text{imp}^2)\,\mathrm{d}t \;+\; \text{noise (mean 0)}$$

Integrating along the path:

$$\boxed{\;\mathbb{E}[\text{Total PnL}] \;=\; \tfrac12 \int_0^\tau \Gamma_t\, S_t^2\, (\sigma_\text{real}^2 - \sigma_\text{imp}^2)\,\mathrm{d}t\;}$$

This is the **gamma-rent / variance-swap formula**, the most important
identity in vol arbitrage. Read it as:

- **Long Γ position** (e.g. long YES at S<K): make money when σ_real >
  σ_imp.
- **Short Γ position** (e.g. long YES at S>K, like us): make money when
  σ_real < σ_imp.

### 3.2 Which direction do we trade right now?

Our state: ITM YES (S>K), σ_imp = 39%, σ_real = 25%. So σ_imp > σ_real.

Long YES at S>K → Γ < 0 → **short gamma**. Short gamma + (σ_imp > σ_real)
→ **positive expected PnL**.

So the right play is **buy YES, short BTC perp** (or equivalently for the
NO leg: sell NO, long BTC perp — same trade, different leg).

If instead we were OTM (S<K), the rule reverses: sell YES, long perp.

The general rule: **whichever direction makes you net short gamma when
σ_imp > σ_real**.

### 3.3 The simpler way to see the same thing

You don't have to think in Greeks if you don't want to. The same answer
falls out of:

> "The market is trading YES at $0.635. Under our best estimate of σ (25%),
> fair value is $0.71. We are buying something for $0.635 that is worth
> $0.71. That's $+0.075 of expected edge per unit, paid for by absorbing the
> jump risk between now and expiry."

The dynamic hedge is just a way to **cash out that edge gradually** instead
of waiting for the binary outcome. Without the hedge, we'd get either
$1−$0.635 = $+0.365 or $−$0.635 (a coin flip with positive EV but high
variance). With the hedge, we converge to roughly the same $+0.075 in
expectation but with much lower variance.

---

## 4. The two trades, side by side

You can run **either** of these on the same market — they have different
risk profiles.

### Trade A — Pure market-making (spread capture)

**Idea**: quote a bid and an ask on #30 (or #31) inside the existing spread,
get filled randomly on both sides, hedge any net inventory on perp.

**Where the money comes from**: the ~42-bp median quoted spread on #30.
Each round-trip (one buy, one sell) earns the full spread.

**Big change vs CEXs**: HL HIP-4 has **0 fees**. On Binance you'd pay
~0.5–2 bps in fees, often netted against rebates. On HIP-4 you keep the
**entire spread**. This single fact moves passive MM from "marginal" to
"genuinely positive EV" on this market — it's roughly a 4× improvement.

**What can go wrong**:

- **Adverse selection**: when BTC starts rallying hard, all the buy-YES
  orders hit your ask before you can move it. You sell low, then hedge by
  buying perp at a higher price. The realised spread you capture is
  smaller than the quoted spread — typically 30–60% smaller in retail-flow
  binaries.
- **Inventory risk**: between fills, your inventory creates Γ exposure.
  You re-hedge each new fill on perp but the residual gamma costs you.

### Trade B — Vol-risk-premium harvest

**Idea**: take a one-sided position in the direction that makes you short
gamma when σ_imp > σ_real. Dynamically delta-hedge on perp. Profit ≈
$\frac{1}{2}\Gamma S^2 (\sigma_\text{real}^2 - \sigma_\text{imp}^2)\tau$
from the σ gap.

**Where the money comes from**: the **14-percentage-point σ premium**
between implied (~39%) and realized (~25%).

**What can go wrong**:

- **Pin risk**: if S oscillates near K close to expiry, |Γ| is enormous
  and re-hedging trades scrape vega/gamma faster than σ premium accrues.
  This is the classic short-gamma blow-up.
- **σ_real estimation error**: the 25% number comes from a 9-hour window.
  If the true σ over the next 13 hours is actually 40% (BTC moves more than
  expected), the trade loses.
- **Discrete-hedging slippage**: theory assumes continuous hedging. In
  practice you re-hedge every $X seconds and pay 0.12 bps of perp spread
  each time.

You can run **both** at once — they don't interfere. In fact running them
together helps with adverse selection (the σ-premium leg gives you a
fundamental view of fair value, so your MM quotes can be skewed
intelligently rather than centred on the mid).

---

## 5. Putting numbers on it

Live state on 2026-05-05 17:23 UTC (from notebook 06):

| input | value |
|---|---:|
| $S$ — HL BTC perp mid | 81,359 |
| $K$ — strike | 80,930 |
| $\tau$ — time to expiry | 13 h = 0.001484 y |
| $\sigma_\text{imp}$ — from market price | 39 % |
| $\sigma_\text{real}$ — 9 h realized | 25 % |
| YES market price | 0.635 |
| YES theo price ($\sigma = 25\%$) | 0.7068 |
| Δ at $\sigma_\text{imp}$ | $3.08\times10^{-4}$ /\$ |
| Γ at $\sigma_\text{imp}$ | $-9.04\times10^{-8}$ /\$² |

### 5.1 Trade B PnL forecast — buy 10,000 YES, hedge on perp

**Capital deployed (binary leg)**: $10{,}000 \times 0.635 = \$6{,}350$.

**Hedge (perp leg)**: Δ × 10,000 = $3.08$ of BTC notional → short **$3.08
worth of BTC perp**. (Yes, that small. Binary delta is small in $-units
because the binary itself is bounded $[0,1]$.)

**Two ways to estimate expected PnL**:

1. **Simple**: pay 0.635 per unit, fair value 0.7068 per unit at
   σ_real=25%. Expected edge = 0.0718 × 10,000 = **+$718**.
2. **Variance swap**: $\frac12 \Gamma S^2 (\sigma_r^2 - \sigma_i^2) \tau$
   per unit:
   - $\frac12 \times (-9.04\times10^{-8}) \times 81{,}359^2 \times (0.0625 - 0.1521) \times 0.001484$
   - $= \frac12 \times (-9.04\times10^{-8}) \times 6.62\times10^9 \times (-0.0896) \times 0.001484$
   - $= +0.0398$ per unit × 10,000 = **+$398**.

The two numbers don't match because the variance-swap one uses the *current*
Γ for the whole 13 h, but as we approach expiry Γ explodes near K. The
simple estimate is closer to the truth.

So **expected PnL ≈ $400–$700 on $6,350 deployed**, over 13 hours. Call it
**~6–11% over a 13-hour period in expectation**.

### 5.2 ...and the variance of that PnL

The PnL distribution is roughly:

- 70% of the time (BTC settles above K, our σ=25% prior): payoff $1, total
  return $1 − $0.635 + hedge PnL ≈ $+0.40 per unit
- 30% of the time: payoff $0, total return $-0.635 + hedge PnL ≈ $-0.50 per unit

Without the hedge, it's a $+/-50¢$ coin-flip. Hedging shrinks the variance
substantially but doesn't eliminate it — discrete hedging error +
σ_real-estimation error.

A reasonable assumption: σ of the PnL outcome is ~40% of the unhedged σ
under decent execution. So 1 std dev on $10,000 contracts is roughly $±$1,500–2,000.

### 5.3 Trade A PnL forecast — pure passive MM

Assume:

- We capture **20%** of the median 42-bp spread on round-trips (after
  adverse selection).
- We participate in **5%** of the daily flow.
- HIP-4 daily flow on the active binary = **$1.8 M**.

PnL ≈ $1{,}800{,}000 \times 5\% \times 0.42\% \times 20\% = \$$**75/day**.

That's **with zero fees** (which is already a 4× lift from CEX baseline). On
a 24-hour basis. Per market-pair (can run on both #30 and #31).

The flow-share assumption is the most uncertain part. If we get to **15%**
flow share (plausible if we're the fastest quoter), $225/day. Annualized:
$25k–$80k per binary — small but not zero, with low capital lock-up.

### 5.4 Both trades together

Trade A funds Trade B's gamma rebates and provides a constant view of
realized σ. Trade B's view of fair value lets Trade A skew its quotes
intelligently. Capital usage on Trade B (~$6k per cycle) is the constraint.

**At-scale assumption**: $50k of binary-side capital × 13-h cycle × 2 cycles
per day (one before settlement, one after the next roll) → 100 binary-cycle
units per month. Expected gross at the conservative $400/$6.35k figure:
~$25k/month before adverse selection on Trade A.

These are paper estimates. Real figures need the markout study and the
post-settlement microstructure observation.

---

## 6. Trade sizing — how big should we go?

Two binding constraints, take the tighter one.

### 6.1 Capital constraint

Binary side capital per cycle = $N \times P_\text{YES}$ where N is the
number of contracts. For $50k capital, $N = 50000 / 0.635 \approx 78{,}740$
contracts.

That's a lot of contracts on a market with $1.8 M/day flow and thin
top-of-book. Realistically you can only deploy this many over many hours of
gradual fills, not one shot.

### 6.2 Pin-risk / gamma-budget constraint

Pick the maximum BTC move you're prepared to absorb without re-hedging
(call it $\Delta S^*$, e.g. $200 = ~25 bps). Your unhedged exposure to a
move that big is

$$\Delta\text{PnL} \approx \Delta \cdot N \cdot \Delta S^* \;+\; \tfrac12 \Gamma N (\Delta S^*)^2$$

For $N=78{,}740, \Delta S^* = 200$:

- Delta term: $3.08\times10^{-4} \times 78{,}740 \times 200 = \$$4,852
- Gamma term: $\tfrac12 \times 9.04\times 10^{-8} \times 78{,}740 \times 40{,}000 = \$$142

Total ≈ $5,000 per ungedged $200 BTC move. If we re-hedge every 30 seconds
and BTC has 1-minute σ of ~$50, this is a fine budget.

### 6.3 Liquidity constraint (the one that actually binds first)

Looking at notebook 03 §4 (depth profiles), HIP-4 top-of-book depth is
small — order of magnitude $1k of contracts at the best price. To accumulate
10,000+ contracts you'd need to either:

- Walk the book (paying 200+ bps in slippage on ITM YES, making the trade
  unprofitable in one shot), or
- Spread entries over hours, or
- Provide liquidity (be the bid yourself) — which is Trade A.

So in practice **Trade A is the entry mechanism for Trade B**: you accumulate
the binary inventory by winning fills passively, then hedge on perp as
inventory builds.

### 6.4 Suggested starting size

For a first live test:

- **$5,000–10,000 binary capital** (≈ $8k–16k YES contracts at current price)
- **$5–10 of perp delta hedge** (literally $5)
- Re-hedge every **30 s**, or whenever |residual Δ × S| > $50
- Cycle for one full day (13 h pre-roll + 13 h post-roll = 26 h)
- Compare actual vs theoretical PnL after one cycle, adjust σ_real
  estimator (rolling window length) based on hedge slippage

That's small enough that one wrong σ-estimate costs $500 not $5,000, but
big enough to be statistically meaningful.

---

## 7. Funding and zero-fees — the cost ledger

### 7.1 HIP-4 binary side: 0 fees, 0 funding

You confirmed HIP-4 has **0 fees**. There's also **no funding** on the
binary itself (it's a real binary contract that pays out at settlement, not
a perpetual). So on the binary leg the only costs are:

- **Bid-ask spread paid on entry/exit** if we cross. For Trade A we *don't*
  cross — we're the one earning the spread. For Trade B if we're
  accumulating passively, also no spread paid. Only when we get desperate
  to exit (e.g. forced to dump pre-settlement) do we pay the spread.

This is a huge advantage. CEX MMs pay 0.5–2 bps per fill in fees alone.

### 7.2 HL perp hedge side

Costs accumulated on the perp:

| component | size | cost over 13 h |
|---|---|---|
| Spread paid on each rebalance | 0.12 bps × 2 (round trip) | depends on # rebalances |
| Funding (annualized ~0.66 bps premium) | × 13 h | ≈ 0.0010 bps of perp notional |

Worked example for a 10,000-contract Trade B over 13 h with re-hedging
every 60 s (780 rebalances), each shifting $5 of perp notional:

- Spread cost: $5 × 0.0024 bps × 780 = **$0.94 total**
- Funding cost: $5 × 0.66/365 × 13/24 bps = **$0.005 total**

Negligible. The perp hedge is essentially free at this position size. Even
at 10× larger, it stays under $20.

### 7.3 What about Binance as a hedge?

We could also hedge on Binance perp instead of HL perp. Binance has tighter
spread (~0.01 bps), better depth, and similar funding. But then you're
paying basis risk: HL/BN basis σ is ~1 bps (notebook 02), so over 13 h the
hedge mismatch can be ±5 bps per cycle. For a $5 hedge that's negligible;
for a $5,000 hedge it's $0.25.

For now, hedge on HL perp — eliminates basis risk entirely, costs slightly
more in spread but it's still basically free.

---

## 8. Risks specific to this trade

### 8.1 Pin risk

If BTC mid sits within ~0.1% of K (~$80) close to expiry, |Γ| explodes.
Your hedge ratio swings violently and discrete hedging error eats the
σ premium. The mitigation: **size down or close out** in the last hour if
within ½% of strike. Empirically, traders wind down 1–2 hours before expiry
if pinning.

### 8.2 Settlement source ≠ perp mid

HIP-4 binaries settle on **HL oracle price** at expiry. The oracle is a
median of multiple CEX prices, so it can deviate from HL perp mid by a few
bps in stress. For pricing the *settlement* we should use oracle, not perp
mid; for *hedging* perp mid is cleaner because it's directly tradable.

To bound this: the gap between perp mid and oracle in our 9-h window was
under 5 bps almost all the time (visible in P0). So on $10k binary capital,
the oracle-vs-perp basis costs at most $5 of slippage.

### 8.3 Realized σ regime change

Our σ_real = 25% is from a calm 9-hour BTC stretch. If a Fed announcement
or ETF flow event hits during the binary's life, σ_real could spike to
50–80%. Then σ_imp (39%) was actually too LOW, and our short-gamma trade
loses.

Mitigation: **don't run Trade B through scheduled high-volatility events**
(Fed days, CPI, large funding times). Use Trade A only during those.

### 8.4 Liquidity gaps

Notebook 01 showed multiple multi-minute gaps in our recorder during DNS
issues. If our hedge engine has the same brittleness, a 5-minute gap during
re-hedging could cost more than the σ premium pays. Mitigation:
**watchdog + redundant network** before any live size.

### 8.5 Adverse selection on Trade A

The 42-bp quoted spread is gross. Net spread captured will be lower because
flow that hits us tends to be informed (if BTC just printed up, the buyer
who bought our YES ask is the same person bidding perp up). The plan §P4
markout study will quantify this; until then assume **net spread = 30–50%
of quoted spread**.

---

## 9. Decision

**Go list:**

- ✅ Mathematical setup is sound: binary is digital cash-or-nothing call,
  perp is its hedge, dynamic delta hedging produces expected gamma rent.
- ✅ σ premium is real and large (39% vs 25%, **14 pts**).
- ✅ HIP-4 fees = 0 makes Trade A genuinely interesting on its own.
- ✅ Perp hedge is basically free (cost <$5 per cycle on small size).
- ✅ HL perp tick + spread (0.12 bps) is fine for sub-minute hedging.

**Wait list:**

- ⏳ Need ≥1 settlement cycle observed (next: **2026-05-06 06:00 UTC**, ~13h
  out as of the write).
- ⏳ Need markout / adverse-selection study (plan §P4) to size Trade A
  realistically.
- ⏳ Need oracle-vs-perp gap distribution (we have it implicitly in the
  oracle event stream — should plot).

**What to do this week:**

1. **Tonight**: let the recorder run through tomorrow's 06:00 UTC roll.
2. **Tomorrow**: run notebook 06 again on the post-roll window and
   compare implied σ before/after the settlement print — this is the single
   most informative number for sizing.
3. **In parallel**: build a tiny back-tester that replays our recorded BBO
   and simulates a passive MM (Trade A) over the captured day. Output:
   net spread captured per cycle, fill-rate, inventory profile.
4. **After tomorrow's roll**: deploy Trade A with **$5k binary cap** for
   one full cycle. Don't run Trade B until the back-tester says Trade A
   is positive after slippage.

**Concrete starting size for Trade A** when ready:

- Quote 1 tick inside best bid and best ask on #30 and #31.
- Order size: $1k contracts each side, max $5k total inventory.
- Inventory limit: ±$2k of net YES exposure, hedged on HL perp at every
  500-contract step.
- Expected daily PnL at this size: $20–60. Boring on purpose.
- Scale up only after 1 week of real fills with markout > 0.

That's the whole loop. The math is friendly here: 0 fees, free hedge, fat
σ premium, and we already have the data infrastructure. The bottleneck is
**execution discipline** + **observed settlement behavior**.
