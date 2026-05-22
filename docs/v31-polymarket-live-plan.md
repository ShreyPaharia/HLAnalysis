# v3.1 on Polymarket — Live-Trading Investigation

**Status:** investigation only — no code changes.
**Branch:** `docs/v31-polymarket-live-investigation`
**Author / date:** Shrey, 2026-05-22

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
  signatures per order (CLOB API auth signature; order-payload EIP-712).
  Library: `py-clob-client` (PyPI, official) — **not currently in
  `pyproject.toml`**.
- **Funding model**: USDC.e on Polygon for collateral; small MATIC bag for
  gas on approvals/withdrawals. No margin: it is fully collateralized — each
  contract you hold costs $1, fills clamp into [0,1]. Maps cleanly onto the
  existing `max_position_usd` cap.
- **Order types**:
  - GTC limit, GTD limit, FOK, FAK (PM's term for IOC-like).
  - The strategy emits `time_in_force="ioc"` — must map to PM `FAK`.
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
  `PMConfig` with: `polygon_rpc_url`, `private_key` (or signer reference),
  `clob_api_key`, `clob_secret`, `clob_passphrase`, `usdc_address`,
  `clob_exchange_address`.
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
- **`exit_safety_d` value for PM**: the production `strategy.yaml:282` sets
  `exit_safety_d: 1.0` (HL-tuned). The PM-fee-curve memory
  ([[pm-fee-curve-2026-05-22]]) shows under realistic PM fees **d=0.5 wins
  total PnL** ($438 net vs $396 at d=1.0). **Decide before live: use
  d=0.5 PM config or d=1.0 HL config?** Recommend: per-venue config
  override mirroring the existing per-`question.klass` override pattern
  (`runtime.py:_cfg_by_class`), so PM gets `exit_safety_d=0.5` without
  disturbing HL's `1.0`.

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

### 5.1 Capital sizing — start conservative

- **Initial wallet balance**: $500 USDC on Polygon (covers 5 concurrent
  $100 positions or 2.5 concurrent $200 positions; `max_total_inventory_usd:
  500` already enforced).
- **Initial `max_position_usd`**: **$50** (down from prod $200). Reason: PM
  depth at favorites is thin; one $200 fill at 0.95 needs $190 of ask depth at
  $0.95 — uncommon. Start at the depth that >80% of live ticks support.
- **Daily loss cap**: $25 for first 2 weeks (down from $100). Twice the
  expected per-trade loss; trips after ~3 catastrophic losers.
- **Per-position stop-loss**: leave disabled (matches PM walk-forward best;
  see memory `dynamic_sizing_negative_2026_05_19`).
- **Gas float**: 20 MATIC (~$20). Monitor and top up when < 5.

### 5.2 Market filtering

For the first 30 days, restrict allowlist to **the most liquid PM market
slug only**: `btc-up-or-down-daily` (binary, single market per day).

Additional pre-trade filters:
- `total_volume_usd >= 50_000` over market lifetime (filters away thin
  weekend / off-cycle markets).
- Best-bid depth at entry leg ≥ $250 USDC.
- Spread ≤ 1.5¢ at entry.

Reconcile against `pm_fee_curve_2026_05_22` recommendation: ship v3.1 PM with
**`exit_safety_d=0.5`** (PM-optimal) rather than 1.0 (HL prod value).

### 5.3 Ramp schedule

| Phase | Duration | `max_position_usd` | Daily cap | Notes |
|-------|----------|-------------------:|----------:|-------|
| Paper | 14 d     | 100 (simulated)    | n/a       | Shadow only |
| Real T0 | 7 d    | 50                 | $25       | One leg at a time max (concurrent=1) |
| Real T1 | 14 d   | 100                | $50       | Concurrent ≤ 2 |
| Real T2 | open   | 200 (prod)         | $100      | Match HL prod caps |

Promotion gate between phases: no `RiskVeto` storms, no manual interventions,
realized hit-rate within 2σ of backtest expectation, fill-price slippage
under 50 bps.

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
7. `PMClient(paper_mode=False)` — `py-clob-client` integration, EIP-712
   signing, FAK order mapping, status polling, on-chain finalization
   handling.
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

## 7. Open questions / decisions for the user

1. **Operator location**: Are you in a jurisdiction that can transact on PM?
   PM blocks US IPs. The DoH workaround in `scripts/fetch_pm_with_doh.py`
   handles **DNS** blocks but not **IP geo-blocks**. If geo-blocked,
   shadow mode is fine but real-money requires a VPS in an allowed region.

2. **`exit_safety_d` value for PM**: ship 0.5 (PM-fee-curve memory's
   PnL-best) or 1.0 (current prod, HL value)? Recommend **0.5** with a
   per-venue override mechanism so HL keeps 1.0.

3. **`tte_max_seconds` for PM**: the user mentioned PM was tuned with
   `tte_max=14400` (4h); current prod config is 43200 (12h, HL-tuned).
   Confirm which value to ship for PM — affects how early in a daily market
   we accept entries.

4. **Wallet model**: a single Polygon EOA for trading, or hot/cold split
   (cold custody, hot wallet topped up from cold weekly)? Cold split is
   safer but adds operational overhead; given the $500 starting capital,
   single hot EOA is fine.

5. **Account funding source**: do you already have USDC.e on Polygon, or
   does the rollout need to budget bridge time/fees from another chain?

6. **Risk-cap appetite during ramp**: §5.3 proposes $25 → $50 → $100 daily
   caps. Is that aggressive enough, or too aggressive? Backtest expectation
   is ~ -$5 max worst trade at $50 size; a $25 cap permits ~5 consecutive
   losers before halt.

7. **Shadow → real promotion authority**: who signs off on each ramp
   transition? Likely just you, but worth being explicit so the criteria in
   §4.3 / §5.3 aren't quietly waived.

8. **Multi-strategy-on-PM scope**: does v1 (`late_resolution`) also get a PM
   shadow slot, or is this v3.1-only? Recommend v3.1-only for now;
   `late_resolution` was HL-tuned and would need a separate PM walk-forward.

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

## Appendix B — Cross-references to project memory

- [[pm-fee-curve-2026-05-22]] — PM fee formula and v3.1-prod walkforward
  impact (drives the d=0.5 vs d=1.0 decision in §3.5 / §7).
- [[strategy_phase1]] — original near-resolution arb scope; v3.1 inherits
  from this.
- [[dynamic_sizing_negative_2026_05_19]] — fixed-$100 beats safety-d-scaled
  sizing on PM (informs §5.1 conservative cap choice).
- [[v3_2_volclock_smoke_2026_05_20]] / [[tau_removal_pm_2026_05_20]] —
  alternative vol estimators tested; v3.1 stays on sample_std.
- [[deployment_topology]] — EC2 t4g.micro shared between recorder + engine;
  PM slot fits inside existing footprint.
