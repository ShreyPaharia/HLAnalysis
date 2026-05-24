# v3.1 on Polymarket — Live-Trading Investigation

**Status:** investigation only — no code changes.
**Branch:** `docs/v31-polymarket-live-investigation` (rebased on `main` @ 385dd9d)
**Author / date:** Shrey, 2026-05-22 (updated 2026-05-24 with operator answers + post-ablation PM params)

**Decisions locked in (from operator):**
- **Trading host: Tokyo** — PM is reachable; no geo-block workaround needed.
- **Wallet model: single Polygon EOA**, operator-funded.
- **Daily loss cap: $100** (prod parity with HL slots — no $25/$50 ramp on the cap).
- **Scope: v3.1 only.** v1 (`late_resolution`) is **not** going to PM. No paper-only
  step is requested; shadow infra still needed for risk-gate verification, but
  the rollout target is real money on a dedicated third HL/PM account.
- **Account topology: third (strategy, account) slot** alongside the existing
  `v1` (HL) and `v31` (HL) slots — `account_alias: v31_pm` (new), runs the
  same strategy class as `v31` but against PM.
- **PM params: locked from post-ablation final-state (2026-05-23, commit `562e8f1`).**
  Source: `config/tuning.v3-1-final-pm.yaml` + memory
  [[v31-final-state-2026-05-23]]. See §3.5 / Appendix C for the exact knob
  delta vs HL `v31`.

This report scopes the work needed to take the **v3.1 strategy** (currently the
`theta_harvester` strategy_type with the v3.1-tuned config — see
`config/strategy.yaml:151-283`) from a Polymarket-backtest-validated state to
**live shadow + live real-money** on Polymarket BTC Up/Down daily markets.

The engine today is a Hyperliquid-first live system; the Polymarket code path
is **read-only and historical** (cache → backtester). Live PM trading requires
a new venue adapter on the **market-data side** and a new HL-equivalent
**order-execution client** on the write side, plus a tighter risk envelope to
match PM's much thinner books.

---

## 0. Naming clarification (load-bearing)

The user prompt calls the strategy "v3.1". In the codebase **v3.1 is the v3.1
parameter pack of the `theta_harvester` strategy**, not `late_resolution`
(which is v1).

- Source: `hlanalysis/strategy/theta_harvester.py:312` (`ThetaHarvesterStrategy`)
- Config slot: `config/strategy.yaml:151-283` (`name: theta_harvester`,
  `account_alias: v31`)
- Defining v3.1 params (PM-tuned, see file):
  - `exit_take_profit_mode: true`, `exit_fee: 0.0007` (`strategy.yaml:258-259`)
  - `fee_taker: 0.0`, `half_spread_assumption: 0.005` (`:240-241`)
  - `edge_max: 0.20` (the v3.1 upper-edge filter, `:245`)
  - `exit_safety_d: 1.0` (`:282`) — **see §3 below; prod is 1.0 but the PM
    fee-curve memory recommends 0.5**
  - `tte_max_seconds: 43200` (12h, HL value) — PM walk-forward used
    `tte_max=14400` (4h) per user note; **needs reconciliation** before live
    PM runs

`late_resolution.py:991` (v1) is a different strategy and is **not** the
candidate for this rollout, even though some of the engine wiring (per-class
configs, gates, topup) is shared.

---

## 1. What already exists and can be reused

### 1.1 Strategy core — reusable verbatim

- **`ThetaHarvesterStrategy.evaluate(...)`**
  (`hlanalysis/strategy/theta_harvester.py:318-382`) — pure synchronous
  function of `(QuestionView, books, reference_price, recent_returns,
  position, now_ns, …)`. Has no venue coupling.
- Helpers (`_winning_region`, `_p_leg_win_prob_and_phi`, `_safety_d_for_region`,
  `_sigma`, `_mu`) all live in the same file (`:36-410`). They use only
  numpy / numba and the strategy types.
- Strategy types are venue-agnostic: `OrderIntent`, `BookState`, `QuestionView`,
  `Position` in `hlanalysis/strategy/types.py:1-107`.

**Implication:** the same `evaluate()` will drive PM orders unchanged — the
adapter layer must surface PM token-IDs as `symbol`, PM CLOB top-of-book as
`BookState`, and PM market metadata as `QuestionView`.

### 1.2 Engine glue — mostly reusable, partly HL-coupled

- **`EngineRuntime` (`hlanalysis/engine/runtime.py:200-323`)** is multi-strategy /
  multi-account; can host a `pm` account_alias slot alongside the existing `v31`
  HL slot without code changes — *if* the per-slot `HLClient` becomes generic.
- **`Scanner` (`hlanalysis/engine/scanner.py:26-99`)** — pure: walks
  `MarketState.all_questions()`, applies allowlist, calls
  `strategy.evaluate()`. No venue coupling.
- **`Router` (`hlanalysis/engine/router.py:27-372`)** — calls
  `self.hl.place(...)` and `self.hl.cancel(...)` only via duck-typed methods
  (`PlaceRequest`, `OrderAck`). Stamps cloid prefixes, handles fills,
  cooldowns, settlement. **Reusable if a `PMClient` exposes the same API
  surface as `HLClient`.**
- **`RiskGate` (`hlanalysis/engine/risk.py:33-162`)** — venue-agnostic. Reads
  allowlist, caps, stale-data, etc. Reuse unchanged.
- **`Reconciler` (`hlanalysis/engine/reconcile.py:36-238`)** — venue-agnostic
  *interface*: takes `venue_open: list[OpenOrderRow]`, `venue_state:
  ClearinghouseState`, `fills_lookup`. **HL-shaped payloads are produced by
  `HLClient` adapters at lines `runtime.py:489-499`**, but the reconcile
  algorithm itself is generic — wire in equivalent PM payloads and it works.
- **`MarketState` (`hlanalysis/engine/market_state.py:31-300`)** is
  venue-agnostic: ingests `NormalizedEvent` (`hlanalysis/events.py:43-218`).
  PM events must be normalized into these types; once they are,
  `MarketState.book()`, `recent_returns()`, `last_mark()` all work as-is.

### 1.3 Market data: PM read-only path exists

- `hlanalysis/backtest/data/polymarket.py:282-836` — full Polymarket data
  pipeline: discovery via Gamma `/events`, trades via Data-API
  `/trades?market=<conditionId>`, synthetic L2 from each trade. Strike-ref
  parsing for "BTC Up or Down" daily markets (`:142-164`).
- DNS bootstrap: `scripts/fetch_pm_with_doh.py:18-99` — Cloudflare DoH
  fallback for blocked resolvers (NextDNS / Pi-hole). **PM may be geo-blocked
  for US IPs; verify operator location before deploying.**
- Endpoints referenced (`polymarket.py:43-45`):
  - Gamma: `https://gamma-api.polymarket.com`
  - Data-API (trades): `https://data-api.polymarket.com`
  - **CLOB (write): `https://clob.polymarket.com`** — referenced as a constant
    but **never called**; this is the surface the live writer needs to drive.

### 1.4 Recorder & ops infra — reusable shell, PM adapter missing

- `hlanalysis/recorder/runner.py:1-60` registers adapters by name; adding a
  `polymarket` entry is mechanical.
- `config/symbols.yaml:1-50` already names HL + Binance subscriptions; PM
  subscriptions slot in cleanly using existing `Subscription` model
  (`hlanalysis/config.py:19-34`).
- Deploy infra (`deploy/`, `DEPLOYMENT.md`): EC2 t4g.micro topology already
  hosts recorder + engine; PM adds an additional WS endpoint but no new
  hardware required.

### 1.5 Paper-mode primitive — reusable pattern, must be re-implemented per venue

`HLClient` has a built-in paper-mode branch (`hlanalysis/engine/hl_client.py:200-248`):
synthesizes fills, tracks virtual positions, returns `OrderAck` with
`status="filled"`. **Pattern reuses cleanly for `PMClient`** — same idea,
PM-specific quirks (token-id symbols, on-chain finalization delay) layered
in.

---

## 2. What is missing for PM live execution

### 2.1 Live market-data adapter (read-side) — **does not exist**

Required:
- `PolymarketAdapter(VenueAdapter)` at `hlanalysis/adapters/polymarket.py`,
  implementing the `VenueAdapter` interface (`adapters/base.py:10-28`).
- WebSocket subscription to PM CLOB book + trades. PM provides
  `wss://ws-subscriptions-clob.polymarket.com/ws/market` for unauth feeds.
- Translation from PM events to `NormalizedEvent`:
  - PM `book` update → `BookSnapshotEvent` (top of book) or `BboEvent`
  - PM `last_trade_price` / trade → `TradeEvent`
  - PM market-creation / resolution → `QuestionMetaEvent` / `SettlementEvent`
- Market discovery via Gamma `/events?series_slug=btc-up-or-down-daily&closed=false`
  (the live counterpart of `polymarket.py:_fetch_series_events`).
- Strike-reference timestamp parsing already exists
  (`backtest/data/polymarket.py:_parse_strike_ref_ts_ns:142-164`) — promote to
  shared util.

### 2.2 Live order client (write-side) — **does not exist**

PM order entry runs over an EIP-712 signed payload to the CLOB REST API on
Polygon, plus on-chain settlement for fills. Concretely:

- **Auth model**: Polygon EOA + L2 API key registered with the PM CLOB. Two
  layers (per `py-clob-client-v2` README): **L1** EIP-712 wallet signature to
  create / derive API key; **L2** HMAC with the derived `(api_key, api_secret,
  api_passphrase)` for order placement / cancel / account-read.
  - Library: **`py-clob-client-v2` 1.0.1** (PyPI, official). The original
    `py-clob-client` was **archived on GitHub 2026-05-11** alongside the TS
    `clob-client` — both superseded by `-v2` variants. Deps:
    `eth-account>=0.13`, `eth-abi>=5.0`, `poly_eip712_structs>=0.0.1`,
    `py-order-utils>=0.3.2`, `httpx[http2]>=0.27`.
  - **No Python WS SDK from Polymarket** — `real-time-data-client` is
    TypeScript-only. The Python live adapter must wrap PM's WS directly on
    `websockets` (already a project dep via `hyperliquid.py`).
- **Funding model**: USDC.e on Polygon for collateral; small MATIC bag for
  gas on approvals/withdrawals. No margin: it is fully collateralized — each
  contract you hold costs $1, fills clamp into [0,1]. Maps cleanly onto the
  existing `max_position_usd` cap.
- **Order types** (per `py-clob-client-v2` `examples/orders/`):
  - `OrderType.GTC` / `OrderType.GTD` (resting limits).
  - `OrderType.FOK` (all-or-cancel immediate).
  - `OrderType.FAK` ("fills as much as possible, remainder cancelled") —
    **direct map for our `time_in_force="ioc"`**. The `marketable_limit_buy.py`
    / `marketable_limit_sell.py` examples show the exact pattern (a marketable
    limit with FAK semantics = our IOC).
  - Market orders also supported (`create_and_post_market_order` with
    `MarketOrderArgs.amount` in USDC). The strategy still emits a price; stick
    with FAK on a marketable limit so slippage stays bounded.
  - `reduce_only` doesn't exist on PM (long-only collateralized YES/NO
    contracts). Same workaround the HL client uses for HIP-4
    (`hl_client.py:271`): strip the flag, rely on Router's position
    bookkeeping to size sells correctly.
- **Symbol shape**: PM legs are 76-digit ERC-1155 token IDs (`tokenId`
  strings). Already used as `symbol` in the backtest layer
  (`polymarket.py:414`). The engine doesn't care about symbol shape — works.
- **Order min size / tick**: PM minimum is 5 shares (and ~$1 USDC notional);
  price ticks are 0.001 (= 0.1¢). Tighter than HL; the strategy emits
  2-decimal sizes already (`theta_harvester.py:_size_floor_for...`), but the
  `PMClient.place()` must round price to 3 decimals and floor size to whole
  shares.
- **Fee handling**: **PM published fee is non-flat — `fee = C · feeRate · p ·
  (1−p)`** (docs.polymarket.com/trading/fees). Crypto category feeRate=0.07.
  - The strategy currently uses `fee_taker=0.0`, `exit_fee=0.0007`. These were
    calibrated on **flat-fee assumptions** — see [[pm-fee-curve-2026-05-22]]
    in memory: under the real PM curve, v3.1 prod takes a 37–54% PnL haircut.
  - **Backtester now supports `--fee-model pm_binary --fee-rate 0.07`** —
    confirm config has been re-tuned under that model before live.
- **Settlement detection**: PM markets resolve via on-chain `condition.resolve`
  event (UMA). Two paths:
  1. Poll Gamma `/events?slug=...` and watch `resolved` flag (used by backtest
     already, `polymarket.py:_parse_binary_event:188-196`).
  2. Subscribe to UMA proposal/resolution events on Polygon.
  Path 1 is simpler and fits the existing `SettlementEvent` semantics.

### 2.3 Account-side wiring

- `HLConfig` (`engine/config.py:177-181`) is HL-specific. Needs a sibling
  `PMConfig` matching the `py-clob-client-v2.ClobClient` constructor surface:
  `clob_host` (e.g. `https://clob.polymarket.com`), `chain_id` (137 mainnet
  / 80002 Amoy), `private_key` (EOA L1 key), and the persisted L2 triple
  `clob_api_key` / `clob_api_secret` / `clob_api_passphrase` (derived once
  via `client.create_or_derive_api_key()` and saved to `deploy.yaml` so the
  engine doesn't re-derive every restart). Polygon RPC URL is **not** needed
  for CLOB I/O (the SDK talks HTTPS to PM's REST/WS — Polygon RPC is only
  needed for direct token approvals / redemptions, which can be a manual
  one-time setup step).
- `DeployConfig.hl_accounts` (`engine/config.py:200`) becomes
  `accounts: dict[str, HLConfig | PMConfig]` — or split into separate
  dicts keyed by venue. The runtime's per-slot factory
  (`runtime.py:_build_slot:326`) needs to dispatch on venue type.
- Telegram alerts (`hlanalysis/alerts/`) already venue-agnostic.

### 2.4 Settlement reconciler (write-side mirror)

The reconciler `vanished_positions` path
(`engine/reconcile.py:163-178`) treats "local position not on venue" as a
settlement signal. On HL HIP-4 this is right. **On PM, that branch needs
adapting** — PM positions don't vanish on settlement; the `position.balance`
remains zero on the losing leg and stays at notional on the winning leg until
redemption. Either:
- Reconciler treats `vp.qty == 0` the same way HL treats "missing", or
- New code: detect "market.resolved == true via Gamma poll" → close locally
  and book PnL = `qty × (settled_outcome == leg_side ? 1.0 : 0.0) − avg_entry`.

The second option is cleaner because PM exposes the resolved outcome
directly; the `polymarket.py:_parse_binary_event` already reads it.

### 2.5 Recorder PM topic

For the shadow phase (no orders) we need PM book/trade history captured
locally so post-mortem analysis is symmetrical with HL/Binance. New
subscription block in `config/symbols.yaml`:

```yaml
- venue: polymarket
  product_type: prediction_binary
  mechanism: clob
  symbol: "*"
  channels: [trades, book, bbo]
  match:
    series_slug: btc-up-or-down-daily
    underlying: BTC
    class: priceBinary
```

Plus a writer schema extension — PM token-IDs are 76-digit strings; ensure
the parquet column type is `string`, not `int64`.

---

## 3. Risk-control gaps specific to Polymarket

### 3.1 Market-depth–aware sizing (highest priority)

PM books for "BTC Up or Down daily" are **routinely thin at the favorite leg**
once `p > 0.95`. Backtest assumed `_HALF_SPREAD_DEFAULT = 0.005`, `_DEPTH_DEFAULT
= $10,000` (`backtest/data/polymarket.py:51-52`), but live depth at >0.95 is
often **$100–$500 per level**. Hitting the book with `max_position_usd=200`
will routinely consume 1–3 levels of asks.

**New gate needed**:
- Pre-trade depth-walk: simulate filling `intent.size` against the live ask
  ladder; reject (or downsize) if the realized fill price would exceed
  `intent.limit_price + max_slippage`.
- Slippage cap candidate: 0.5–1.0¢ (≈ 0.5–1.0% on a 0.95-priced favorite).

This does not exist today on the HL side because HIP-4 books have a different
microstructure (single-level "fat ask" is the failure mode, not 5-level
consumption). The `min_bid_notional_usd` gate
(`theta_harvester.py:249`) is the only PM-relevant tool currently and it
checks bid notional, not ask depth.

### 3.2 Per-market exposure cap

Existing: `max_position_usd` (per-position) and `max_total_inventory_usd: 500`
(global) — `strategy.yaml:220-221`. These transfer to PM cleanly.

**Additional PM-only cap suggested**:
- `max_position_pct_of_book = 0.15` — never take more than 15% of the
  visible ask depth at entry. Prevents the same trade from being the entire
  book.

### 3.3 Daily-loss cap

`daily_loss_cap_usd: 100` is already in place (`strategy.yaml:222`). For PM,
the cap reads from the venue: `Scanner._pnl_provider` is wired to
`HLClient.realized_pnl_since` (`runtime.py:372`). `PMClient.realized_pnl_since`
must compute the equivalent from `user_fills` + (resolved-market redemption
flow). Caveat: PM redemption is on-chain and lags; until the position
redeems, "realized PnL" should mark settled positions to 1.0 / 0.0 based on
the on-chain resolution flag rather than waiting for the redemption tx.

### 3.4 Kill switch

Per-slot kill-switch path already exists (`runtime.py:543`, file-based).
Reuse unchanged. Operator workflow stays: `touch <kill_switch_path>` to halt
a single (strategy, venue) pair without affecting the HL slot.

### 3.5 PM-specific catastrophic failure modes

- **Gas exhaustion**: MATIC balance on the trading EOA can run dry. Need a
  pre-trade balance check (`if gas_balance < threshold → halt`). Doesn't
  exist on HL because HL is off-chain for the user.
- **Polygon RPC outage / mempool stall**: an order can be signed and
  broadcast but never confirmed. Need an order-state timeout (e.g. 30s) and
  a reconcile pass that picks up "submitted but never acknowledged".
- **UMA-disputed resolutions**: rare but real. Don't book PnL until the
  dispute window closes. Detected by Gamma `resolved=true && resolved_at -
  now > 48h`.
- **`exit_safety_d` value for PM — SETTLED post-ablation (2026-05-23):** the
  earlier d=0.5-vs-1.0 question is resolved. The final-state ablation memory
  [[v31-final-state-2026-05-23]] and `config/tuning.v3-1-final-pm.yaml:33`
  ship **`exit_safety_d: 1.0`** for PM ("d=0.9/1.0/1.1 cluster, 1.0 most
  robust per-split"). The d=0.5 finding from [[pm-fee-curve-2026-05-22]] was
  superseded by the round-2-through-5 ablations once `favorite_threshold` and
  `edge_buffer` were jointly retuned. So **PM and HL share `exit_safety_d=1.0`**
  — no per-venue override needed on this knob. The per-venue split *is* still
  required for `favorite_threshold` / `edge_buffer` / `topup_enabled` /
  `min_distance_pct` — see Appendix C.

---

## 4. Paper-mode (shadow) plan

### 4.1 Goal

Run v3.1 against live PM book + trade feeds, produce decisions and "would-have-placed"
orders, **never submit a signed CLOB transaction**. Validate:
1. Decision parity with the backtester on overlapping bars.
2. Realized fill prices (book-walk simulation) close to backtest assumptions.
3. Latency from PM book update → engine decision → would-be-place ≤ 1s on the
   1Hz scan loop.
4. Risk gates fire under realistic live conditions (especially depth-walk
   slippage cap and post-exit cooldown).

### 4.2 Concrete shadow mode steps

1. **Add PM market-data adapter only** (no PMClient yet).
2. **Synthetic PMClient** using the same `paper_mode=True` pattern as
   `HLClient` (`hl_client.py:200-248`). Methods:
   - `place(req)` → `OrderAck(status="filled", fill_price=req.price,
     fill_size=req.size)` (or model the depth-walk and partial-fill).
   - `open_orders()` / `clearinghouse_state()` → in-memory mirrors.
   - `user_fills()` / `realized_pnl_since()` → from synthesized fills.
3. **Engine config**:
   ```yaml
   - name: theta_harvester
     account_alias: v31_pm_paper
     strategy_type: theta_harvester
     paper_mode: true                  # ← key
     # … rest mirrors the v31 entry but with PM-tuned exit_safety_d=0.5
   ```
4. **Logging — what to capture per decision**:
   - Tick timestamp, BTC reference price, σ, μ, τ.
   - Per-leg `(bid, ask, bid_sz, ask_sz, mid)` from PM book.
   - `(p_win, edge_raw, phi_d, gamma_penalty)`.
   - Decision action + diagnostic codes.
   - For ENTER decisions: simulated fill price (book walk), simulated fill
     size, would-be cloid.
   - For EXIT decisions: same plus realized PnL under the simulated entry.
5. **Storage**: extend the existing gate-decision JSONL (`scanner.py:_gate_log_path`)
   with PM-specific fields, written to `data/sim/live-shadow/pm/`.

### 4.3 Success criteria — must hit all of these before any real-money rollout

- ≥ 14 calendar days of continuous shadow operation, ≥ 10 closed markets
  with at least one decision per market.
- Mean abs decision-time deviation from backtest re-run on same bars < 100 ms.
- Simulated fill price within ±0.005 of backtest's `_HALF_SPREAD_DEFAULT`
  assumption on ≥ 80% of trades. (If not, retune `half_spread_assumption`
  before going live.)
- No `OrderRejected` or `RiskVeto.daily_loss_cap` firing on otherwise valid
  decisions (i.e. risk envelope sized correctly).
- ≥ 1 settlement-driven exit cycle observed, reconcile path produces a single
  clean Exit event (no spurious DRIFT alerts).

---

## 5. Real-money rollout plan

### 5.1 Capital sizing — operator-locked

- **Wallet balance (operator-funded)**: ≥ $1,000 USDC on Polygon. Sized to
  support `max_total_inventory_usd=1000` (= main HL prod, per
  `strategy.yaml:139`) without bumping the cap into the live wallet.
- **`max_position_usd: 200`** for PM — matches the post-ablation final-PM
  config (`tuning.v3-1-final-pm.yaml:40`). The main `strategy.yaml`
  HL slots ship `300` post-rebase (`:31, :82, :167, :191, :209`); PM stays
  at the ablation-validated `200`. Walk-forward at `200` produced $1,295 PnL
  / Sharpe 5.94 / maxDD $207 across 5 OOS splits.
- **Daily loss cap: $100** (operator decision; matches HL prod
  `strategy.yaml:222`). Trips after ~5 catastrophic full-position losers.
- **Per-position stop-loss**: disabled (PM walk-forward best per memory
  [[dynamic_sizing_negative_2026_05_19]]).
- **Gas float**: 20 MATIC (~$20). Monitor; top up when < 5.

### 5.2 Market filtering

For the first 30 days, restrict allowlist to **`btc-up-or-down-daily`** only.

Pre-trade filters (built into the existing `RiskGate` + the new PM-only
depth gate from §3.1):
- Allowlist match-class `priceBinary` only.
- `min_recent_volume_usd: 100` (main config default, sufficient on this
  liquid daily market).
- Best-ask depth at entry leg ≥ `intent.size × 1.5` (depth-walk gate; see §3.1).
- Spread ≤ 1.5¢ at entry (PM ticks are 0.001 → ≤ 15 ticks wide).
- Operator-side gate: `favorite_threshold=0.90` (PM-tuned, vs HL's 0.85),
  `edge_buffer=0.03` (vs HL's 0.02).

### 5.3 Ramp schedule (revised — no paper-only phase)

Operator answer "real only v3.1" + "third account": skip the paper-only
calendar weeks. **Position-size ramp still applies** to validate live
microstructure (depth, slippage, on-chain finalization) before sitting at the
ablation-recommended cap. The dedicated `v31_pm` account isolates blast
radius — the existing HL `v1` and `v31` slots are unaffected.

| Phase | Duration | `max_position_usd` | Daily cap | `max_concurrent_positions` | Notes |
|-------|----------|-------------------:|----------:|----------------------:|-------|
| Burn-in | 3 d  | 50                 | $25       | 1                     | Single market at a time; verify fill prices match backtest assumption ≤ 50 bps slip |
| Ramp 1 | 7 d   | 100                | $50       | 2                     | Spread across consecutive daily markets |
| Prod   | open  | 200 (ablation)     | $100      | 5 (HL parity)         | Final state — matches `tuning.v3-1-final-pm.yaml` |

Promotion gates between phases (all must hold):
- No `OrderRejected` storms (≤ 1 per day, manually triaged).
- No `RiskVeto.daily_loss_cap` firing.
- Realized fill price within ±50 bps of intent `limit_price` on ≥ 80% of fills.
- ≥ 1 settlement-driven exit cycle observed without spurious DRIFT alerts.
- No manual intervention.

### 5.4 Monitoring dashboard requirements

The existing Telegram alert pipeline (`hlanalysis/alerts/rules.py`) covers:
Entry / Exit / OrderRejected / RiskVeto / KillSwitchActivated /
DailyLossHalt / StaleDataHalt / ReconcileDrift. Add PM-specific:
- `GasLow` (event published when MATIC balance < threshold).
- `OnchainPending` (order signed but unconfirmed > 60s).
- `BookDepthInsufficient` (depth-walk gate fired N times in last 10m).

Web dashboard / Grafana panels recommended:
- PnL: realized today (from PM client), unrealized (mark to mid), cumulative.
- Per-market: current position, entry px, current mid, p_model, edge.
- Decision throughput: scans/sec, entries/hour, vetoes/hour by reason.
- Wallet: USDC balance, MATIC balance, last withdrawal/deposit.
- Latency: book-tick to decision, decision to acked-place.

Reuse: the existing heartbeat loop (`runtime.py:446-479`) already publishes
per-slot scan/event/decision counts to logs — extend with PM-specific fields
and pipe into the same destination.

---

## 6. Prioritized punch list

**P0 — blockers for any PM activity**
1. `PolymarketAdapter` (read-side WS + REST + meta-poller).
2. `Subscription` config + recorder writer schema for PM topics.
3. `PMConfig` + per-venue `_build_slot` dispatch in `EngineRuntime`.
4. `PMClient(paper_mode=True)` implementing the `HLClient` surface
   (place / cancel / open_orders / clearinghouse_state / user_fills /
   realized_pnl_since). **No live signing required at this step.**

**P1 — required before real money**
5. PM-specific fee model in the strategy entry path, OR confirm strategy
   config has been re-tuned under `--fee-model pm_binary --fee-rate 0.07`
   (per memory note, this is already supported in the backtester).
6. Depth-walk slippage gate in `RiskGate` (PM-only; HL is unaffected).
7. `PMClient(paper_mode=False)` — `py-clob-client-v2` integration:
   `create_or_derive_api_key()` bootstrap (one-time, persisted to
   `deploy.yaml`), `create_and_post_order(OrderArgs, OrderType.FAK)` for
   IOC, `cancel_order(order_id)`, `get_orders` / `get_balance_allowance` /
   `get_trades` for the read path.
8. PM settlement detection path (Gamma `/events` resolution poll →
   `SettlementEvent` → existing `_close_settled` flow).
9. `GasLow` / `OnchainPending` alert rules.

**P2 — operational polish**
10. Per-venue config override mechanism for `exit_safety_d` (so PM=0.5 and
    HL=1.0 coexist without forking the strategy module).
11. PM dashboard / Grafana — at minimum, swap the existing alert messages
    to include PM fields.
12. Withdrawal automation (low-priority — PM allows manual withdrawal
    weekly).

---

## 7. Resolved decisions + remaining open questions

### Resolved (operator answers, 2026-05-24)

| # | Question | Answer |
|---|----------|--------|
| 1 | Operator location | **Tokyo** — PM reachable, no geo-block workaround |
| 2 | `exit_safety_d` PM value | **1.0** — settled by 2026-05-23 ablation, same as HL |
| 3 | PM ablation params | Take from `config/tuning.v3-1-final-pm.yaml` (see Appendix C) |
| 4 | Wallet model | **Single Polygon EOA**, operator-managed |
| 5 | Funding source | **Operator-funded** — USDC.e procurement is operator's |
| 6 | Daily loss cap | **$100** (HL prod parity; no ramp on the cap) |
| 7 | Scope | **v3.1 only**, real money — no v1 PM slot |
| 8 | Account topology | **Third (strategy, account) slot** — `account_alias: v31_pm` |

### Still open (need answers before P0 ships)

1. **`tte_max_seconds` for PM**: the ablation file pins `tte_max=86400`
   (24h, `tuning.v3-1-final-pm.yaml:38`). HL ships 43200 (12h). PM daily
   markets *expire* at 24h regardless; the question is whether we want to
   accept entries with > 12h to expiry. Recommend keeping the
   ablation value (`86400`) since that's what the OOS PnL was measured under.
   Confirm.

2. **Concurrent-position cap on PM**: ablation tested per-market not
   cross-market. With `max_concurrent_positions=5` (HL prod parity) and PM
   running one binary per day, the cap only binds across rollover windows
   (a few minutes per day). OK to ship at 5, or tighten to e.g. 2?

3. **EIP-712 key custody**: where does the Polygon private key live? Options:
   - (a) Encrypted env var on the EC2 instance (matches HL key handling
     today, `engine/config.py:HLConfig.api_secret_key`).
   - (b) Hardware signer (Frame / Fireblocks) — slower but stronger.
   Recommend (a) for parity with existing HL key handling, given single-EOA
   model and ≤ $1,000 exposure cap.

4. **Tokyo host: re-use existing EC2 or new instance?** PM WS + Polygon RPC
   add ~1–3 MB/s extra ingress. The existing t4g.micro should hold (the
   memory note `deployment_topology` confirms recorder + engine already
   share it with 1G swap headroom), but if Polygon RPC adds latency to the
   1Hz scan loop, consider a separate `t4g.small` for the PM slot.

5. **Polygon RPC provider** *(may be unnecessary)*: `py-clob-client-v2` does
   all order I/O over HTTPS against `clob.polymarket.com` — no direct
   Polygon RPC calls. RPC is only needed for one-off operations: USDC
   approval to the CTF exchange (one tx, can be done via MetaMask manually)
   and redemption sweeps after settlement (also one-off per market). For
   ≤ 5 orders/day this is *not* a hot-path dependency. Recommend: skip
   RPC integration for v1; do the approvals manually via the PM UI.
   Revisit only if we need programmatic redemption.

6. **Settlement-PnL bookkeeping vs redemption lag**: PM resolves the
   on-chain `condition.resolve` but USDC redemption can take minutes to
   hours. Should the engine mark settled positions to 1.0/0.0 immediately
   (and book PnL), or wait for the actual redemption tx? Recommend
   immediate marking (matches existing `_close_settled` flow,
   `router.py:353`), with a separate `RedemptionTimeout` alert if the tx
   doesn't land within 6 hours.

---

## Appendix A — Key file:line references

| Topic | Location |
|-------|----------|
| v3.1 strategy class | `hlanalysis/strategy/theta_harvester.py:312-984` |
| v3.1 entry/exit core | `hlanalysis/strategy/theta_harvester.py:411-984` |
| v3.1 prod config | `config/strategy.yaml:151-283` |
| Late-resolution (v1) | `hlanalysis/strategy/late_resolution.py:260-991` |
| Engine runtime + multi-account | `hlanalysis/engine/runtime.py:200-696` |
| HL client (template for PMClient) | `hlanalysis/engine/hl_client.py:113-475` |
| Router (decision → place) | `hlanalysis/engine/router.py:27-372` |
| Risk gate | `hlanalysis/engine/risk.py:33-162` |
| Reconciler | `hlanalysis/engine/reconcile.py:36-238` |
| Market-state | `hlanalysis/engine/market_state.py:31-300` |
| PM read-only path (backtest) | `hlanalysis/backtest/data/polymarket.py:1-836` |
| PM DoH bootstrap | `scripts/fetch_pm_with_doh.py:1-99` |
| Adapter interface | `hlanalysis/adapters/base.py:10-28` |
| Existing adapters (HL, Binance) | `hlanalysis/adapters/hyperliquid.py:99-918`, `binance.py` |
| Event schema (NormalizedEvent) | `hlanalysis/events.py:43-218` |
| Subscription config | `hlanalysis/config.py:19-44`, `config/symbols.yaml:1-50` |
| Engine deploy config | `hlanalysis/engine/config.py:149-234` |
| Alerts pipeline | `hlanalysis/alerts/rules.py`, `telegram.py` |

## Appendix C — PM-specific config delta (post-2026-05-23 ablation)

Source: `config/tuning.v3-1-final-pm.yaml` + memory [[v31-final-state-2026-05-23]].
Walk-forward PM result vs prior PM prod baseline (5 OOS splits, `pm_binary`/0.07):

| Metric | PM prod baseline | Final-PM (ship) |
|--------|-----------------:|----------------:|
| PnL | $855 | **$1,295** (+51%) |
| Sharpe | 3.07 | **5.94** |
| Max DD | $508 | **$207** (-59%) |
| Trades | 584 | 396 |
| Per-split min | -$228 | **+$21** (all positive) |

### Knob delta — what changes between HL `v31` and the new `v31_pm` slot

| Knob | HL prod (current `v31`) | PM prod (new `v31_pm`) | Why |
|------|------------------------|-----------------------|-----|
| `favorite_threshold` | 0.85 | **0.90** | PM-curve fees reward selectivity; +$340 PnL / +2.81 Sharpe |
| `edge_buffer` | 0.02 | **0.03** | Tighter entry bar; +$62 PnL on PM |
| `topup_enabled` | true | **false** | PM has no partial fills; HL live MUST keep `true` |
| `min_distance_pct` | null | null | Already disabled both sides |
| `tte_max_seconds` | 43200 (12h) | **86400 (24h)** | PM markets are 24h; ablation tested at 24h |
| `max_position_usd` | 300 (post-rebase) | **200** | Ablation OOS validated at 200; bump deferred for PM |
| `min_bid_notional_usd` | 0 (theta defaults) / 10 (PM ablation) | **10** | Live safety gate per [[feedback-keep-safety-gates]] — kept even though bit-identical in backtest |
| `exit_safety_d` | 1.0 | 1.0 | Same — d=0.9/1.0/1.1 cluster |
| `exit_take_profit_mode` | true | true | Same |
| `exit_fee` | 0.0007 | 0.0007 | Same |
| `edge_max` | null (killed) | null | Same — killed 2026-05-24 |
| `vol_clip_min` | 0.0 | 0.0 | Same — killed 2026-05-24 |

### Concrete YAML stub (to drop into `config/strategy.yaml`)

```yaml
  # --- v3.1 Polymarket ---
  - name: theta_harvester
    account_alias: v31_pm                   # → deploy.hl_accounts.v31_pm (new)
    strategy_type: theta_harvester
    paper_mode: false

    allowlist:
      - match:
          class: priceBinary
          underlying: BTC
        max_position_usd: 200
        stop_loss_pct: null
        tte_min_seconds: 0
        tte_max_seconds: 86400              # PM 24h (vs HL 43200)
        price_extreme_threshold: 0.0
        price_extreme_max: 1.0
        distance_from_strike_usd_min: 0
        vol_max: 100
        entry_cooldown_seconds: 60

    blocklist_question_idxs: []

    defaults:
      match: {}
      max_position_usd: 200
      stop_loss_pct: null
      tte_min_seconds: 0
      tte_max_seconds: 86400
      price_extreme_threshold: 0.0
      price_extreme_max: 1.0
      distance_from_strike_usd_min: 0
      vol_max: 100
      entry_cooldown_seconds: 60

    global:
      max_total_inventory_usd: 1000
      max_concurrent_positions: 5
      daily_loss_cap_usd: 100               # operator-locked
      max_strike_distance_pct: 50
      min_recent_volume_usd: 100
      stale_data_halt_seconds: 30
      reconcile_interval_seconds: 15
      daily_window_start_hour_utc: 6

    theta:
      vol_lookback_seconds: 3600
      vol_sampling_dt_seconds: 60
      vol_clip_min: 0.0
      vol_clip_max: 5.0
      edge_buffer: 0.03                     # PM-tuned (vs HL 0.02)
      fee_taker: 0.0
      half_spread_assumption: 0.005
      drift_lookback_seconds: 3600
      drift_blend: 0.0
      favorite_threshold: 0.90              # PM-tuned (vs HL 0.85)
      edge_max: null
      exit_edge_threshold: 0.0
      time_stop_seconds: 0
      exit_take_profit_mode: true
      exit_fee: 0.0007
      min_distance_pct: null
      min_bid_notional_usd: 10.0            # safety gate, kept (see feedback memo)
      topup_enabled: false                  # PM has no partial fills
      topup_threshold_pct: 0.2
      topup_min_notional_usd: 11.0
      exit_safety_d: 1.0
```

The corresponding `deploy.yaml` needs an `hl_accounts.v31_pm` entry — but the
field shape must change from `HLConfig` to a venue-typed config (`PMConfig`)
per §2.3. Exact fields (mirroring `py-clob-client-v2.ClobClient`):
`clob_host`, `chain_id`, `private_key`, `clob_api_key`, `clob_api_secret`,
`clob_api_passphrase`.

## Appendix B — Cross-references to project memory

- [[v31-final-state-2026-05-23]] — **load-bearing**: end-to-end v3.1 ablation
  under PM-curve fees. Source of the PM `favorite_threshold=0.90`,
  `edge_buffer=0.03`, `topup_enabled=false`, killed `edge_max` /
  `vol_clip_min`. Confirms PM-tune does NOT transfer to HL (40% PnL hit).
- [[feedback-keep-safety-gates]] — `min_bid_notional_usd=10` is bit-identical
  in backtest but kept as live safety gate; same logic applies to other
  defensive gates.
- [[pm-fee-curve-2026-05-22]] — PM fee formula derivation. The original
  d=0.5-vs-1.0 finding was superseded by the round-2-through-5 ablations once
  `favorite_threshold` / `edge_buffer` were jointly retuned.
- [[strategy_phase1]] — original near-resolution arb scope; v3.1 inherits.
- [[dynamic_sizing_negative_2026_05_19]] — fixed-$100 beats safety-d-scaled
  sizing on PM; informs §5.1 sizing.
- [[v3_2_volclock_smoke_2026_05_20]] / [[tau_removal_pm_2026_05_20]] —
  alternative vol estimators tested; v3.1 stays on sample_std.
- [[deployment_topology]] — EC2 shared between recorder + engine; PM slot
  fits inside existing footprint (caveat in §7-Q4 if Polygon RPC latency
  pressures the 1Hz scan loop).
