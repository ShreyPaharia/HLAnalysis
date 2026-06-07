# PM multi-strike live enablement (BTC + ETH) — design

**Date:** 2026-06-07
**Status:** approved for spec → plan
**Author:** engine work follow-up to PR #12 (multi-strike backtest + tune)

## Background

PR #12 added backtest/tuning support for Polymarket daily/weekly multi-strike
"above-X" ladders and tuned `v3_theta_harvester` on 6 months of recorded data
(synthetic, real-venue-calibrated liquidity; Binance 1s klines @ dt=5 reference):

| Market | Best params | sum PnL | worst split | maxDD | hit | Verdict |
|---|---|---|---|---|---|---|
| BTC multi-strike | fav0.75/eb0.02/vlb1800/esd0.5 | +$870 | −$24 | $49 | 72% | **GO** |
| ETH multi-strike | fav0.75/eb0.02/vlb1800/esd0.5 | +$832 | −$31 | $61 | 74% | **GO** |
| ETH up/down (binary) | best cell | −$1.4 | −$130 | $161 | 58% | no edge — drop |

The series are recorded **record-only** today; no engine slot trades them. PR #12
explicitly deferred live enablement as a separate engine effort.

## Goal

Trade BTC + ETH multi-strike ladders live, **reusing the already-merged and
already-tuned bucket strategy path verbatim**. Ship in `paper_mode` first;
flip to live after a multi-day paper burn-in against the real CLOB.

## Core principle

Maximize reuse. The strategy already handles the `bucketLayout=above_ladder`
layout (merged `cce0d82`) and was tuned as a `priceBucket`. The live work is
**plumbing the same bucket question shape into the engine**, not new strategy
logic. Each "above-X" leg is economically an independent binary with a static
strike; grouping the legs into one `priceBucket` question is what lets the
identical tuned strategy code (and the same params) run live and match the
validated backtest behavior and risk profile.

## Components

### 1. Shared bucket grouping (de-duplicate)
Extract the backtest's `_parse_bucket_event`
(`hlanalysis/backtest/data/polymarket.py`) into a shared helper consumed by
**both** the backtest data source and the live adapter. It sorts an event's
sub-markets by strike ascending and produces:
- `thresholds`: `[X0, X1, …]`
- `leg_tokens`: flat `[[yes0,no0], [yes1,no1], …]`
- `leg_condition_ids`, `leg_resolutions`

Single source of truth; the backtest path must remain byte-identical.

### 2. Live PM adapter bucket path
In `adapters/polymarket.py::_gamma_poll_once`, branch on the subscription's
`match.class`:
- `priceBinary` → existing per-market path, **unchanged** (up/down stays
  bit-identical).
- `priceBucket` → group the event's sub-markets into **one** `priceBucket`
  `QuestionMetaEvent`:
  - flat `leg_symbols = [yes0,no0,yes1,no1,…]`
  - kv: `priceThresholds="X0,X1,…"`, `bucketLayout="above_ladder"`,
    `series_slug`, `condition_id` (event-level), expiry from event `endDate`
  - `strike = 0.0` (buckets carry per-leg thresholds, not a single strike) — so
    the up/down strike-capture loop skips them (`qv.strike == qv.strike` is True)
  - subscribe **all** leg tokens to the CLOB WS
  - emit per-leg settlement (see Component 6)

This mirrors `_question_view_bucket` in the backtest so the live `QuestionView`
matches the tuned shape exactly.

### 3. symbols.yaml
Flip the two enabled multistrike subscriptions from `class: priceBinary` →
`class: priceBucket`:
- `btc-multi-strikes-weekly` (underlying BTC)
- `ethereum-multi-strikes-weekly` (underlying ETH)

This is the signal the adapter branches on. Token-level recording is unchanged.
SOL/XRP multistrikes stay `priceBinary` (not enabled). Add the engine ETH spot
reference is code-constructed (Component 4), not a symbols.yaml entry.

### 4. ETHUSDT_SPOT reference feed
Generalize the existing BTC-only reference wiring:
- `engine/main.py::binance_spot_reference_subscription(symbol)` → parameterized;
  `build_engine_subscriptions` appends both `BTCUSDT` and `ETHUSDT` bbo feeds.
- `engine/runtime.py::_remap_reference_symbol` + `_SPOT_REF_SYMBOL` → replace the
  single constant with a map `{BTCUSDT: BTCUSDT_SPOT, ETHUSDT: ETHUSDT_SPOT}`.

The BTC path stays behaviorally identical. ETH markets resolve against the
Binance spot 1m close, so the ETH bucket slot references `ETHUSDT_SPOT` for
price + σ (strike thresholds are static from `groupItemTitle`, so no candle
capture is needed for buckets).

### 5. strategy.yaml — two dedicated PM bucket slots (paper_mode)
Both `paper_mode: true`, `strategy_type: theta_harvester`, tuned cell
**fav0.75 / eb0.02 / vlb1800 / esd0.5**, burn-in caps ($50/leg, $100 inventory,
mirroring existing PM slots), `tte_max_seconds` = bucket window, `fee_model:
pm_binary`, `fee_rate: 0.07`:

- **`v31_pm_btc_ms`** — `reference_symbol: BTCUSDT_SPOT`, `reference_sigma_source:
  bbo`, allowlist match `{class: priceBucket, underlying: BTC, venue: polymarket,
  series_slug: btc-multi-strikes-weekly}`.
- **`v31_pm_eth_ms`** — `reference_symbol: ETHUSDT_SPOT`, `reference_sigma_source:
  bbo`, allowlist match `{… underlying: ETH, series_slug:
  ethereum-multi-strikes-weekly}`.

Each needs a `deploy.yaml` account block with a distinct alias for state-DB /
cloid / kill-switch isolation. **Paper mode transacts nothing**, so dedicated
PM funded wallets + SSM secrets are the operator step **at the live flip**, not
now; the account blocks can reference new env placeholders until then.

### 6. PM per-leg settlement (highest risk)
The engine settles HL buckets via venue fills/closedPnl; PM legs resolve from
Gamma `outcomePrices` per sub-market. Grouping N legs into one question means
settlement must mark each leg independently. This path has never run live.
- Reuse the existing HL `priceBucket` position/PnL accounting for the
  position-side bookkeeping.
- Source per-leg resolution from Gamma `outcomePrices` (mirror the backtest's
  `leg_resolutions` / `PolymarketDataSource.leg_payoff`).
- Most TDD scrutiny goes here.

## Go-live posture
1. Land all of the above on a branch, full suite green, open PR.
2. **Operator** runs the SSM paper-mode deploy to EC2; verify `engine-status`
   (no `restart_blocked`/halt) and that the new slots ingest both reference
   feeds and group buckets (gate logs show `no_favorite`/entries, not
   `vol_insufficient_data` forever).
3. Multi-day paper burn-in validating live grouping, strike thresholds, σ from
   the ETH feed, settlement, and order routing against the real CLOB.
4. Flip `paper_mode: false` (operator) + provision dedicated PM funders, after
   the burn-in is clean.

## Out of scope
- Hit-price barrier markets (need a barrier pricer), SOL/XRP multistrikes,
  sports — untouched.
- ETH up/down (no edge) — not enabled.
- Any change to existing HL or PM up/down slots (must stay bit-identical).

## Validation / tests
- Shared grouping helper: parity test that backtest output is unchanged.
- Adapter: bucket-series event → one `priceBucket` QuestionMeta with correct
  flat `leg_symbols` + `priceThresholds` + `bucketLayout`; up/down series still
  emit per-market binaries (regression).
- Reference feed: ETHUSDT remap → `ETHUSDT_SPOT`; BTC unchanged.
- Settlement: per-leg resolution from Gamma `outcomePrices`; winning/losing legs
  booked correctly (the bug class PR #12 fixed in sim — guard it live).
- Config: both new slots load, resolve their per-class theta override, and carry
  the tuned cell; `extra='forbid'` parity (per SHR-65 discipline).
- Full suite green.
