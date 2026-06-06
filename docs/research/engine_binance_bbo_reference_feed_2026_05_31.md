# Engine: dedicated Binance BTCUSDT bbo reference feed (the missing PM σ feed)

Date: 2026-05-31
Branch: `feat/engine-binance-bbo-reference-feed`
Status: **engine wiring + tests only. No `config/strategy.yaml` change. No deploy. Human-gated.**

## What was missing (verified in code)

The live engine ingested **no Binance data at all**. `engine/main.py` built
`engine_subs` from only `venue in ("hyperliquid","polymarket")` and ran a
`CompositeAdapter([HyperliquidAdapter(), PolymarketAdapter()])` — Binance was
**recorder-only**.

So the PM slots' `reference_symbol: BTCUSDT` (`config/strategy.yaml`,
`paper_mode: false`) had **no live σ reference feed**:
`MarketState.recent_returns("BTCUSDT")` / `last_mark("BTCUSDT")` were always
empty. PM σ / p_model had effectively never worked live (consistent with the PM
slots taking ~0 trades). The `btc=$...` in PM heartbeats is misleading —
`_heartbeat_loop` prints `last_mark("BTC")` (the HL mark) for every slot.

## What this change does (FEED side)

- `engine/main.py`:
  - `binance_reference_subscription()` — a dedicated, code-constructed
    `Subscription(venue=binance, product_type=perp, mechanism=clob,
    symbol="BTCUSDT", channels=("bbo",))`. **bbo ONLY.**
  - `build_engine_subscriptions(sym_cfg)` — HL + PM subs from `symbols.yaml`
    **verbatim**, plus exactly one appended binance bbo reference sub. Binance
    entries in `symbols.yaml` are skipped (recorder-only, untouched).
  - `build_engine_adapter()` — `CompositeAdapter([HyperliquidAdapter(),
    PolymarketAdapter(), BinanceAdapter()])`. The composite filters subs by
    `venue`, so the `BinanceAdapter` receives only the bbo reference sub.
  - `main()` now uses both helpers (logic extracted from the inline body so it's
    unit-testable).

### Why bbo-only is pure WS (no REST poll)
`adapters/binance.py` spawns `_poll_perp_premium` **only** when a perp sub
requests `mark` or `funding` (`_PERP_REST_CHANNELS` intersection non-empty). A
bbo-only sub subscribes just `btcusdt@bookTicker` over WS and spawns **no** REST
task. `bookTicker` is not geo-blocked from the EC2/Tokyo IP (only `markPrice` WS
is — which we are not using). Pinned by
`test_bbo_only_perp_sub_spawns_no_rest_premium_poll` with a positive control.

### Staleness / watchdog
`MarketState.apply` routes the BTCUSDT `BboEvent` into `book("BTCUSDT")` and
stamps `last_l2_ts_ns` from the event's exchange/recv ts — the same key the
`stale_books` / `stale_data_halt` machinery already uses. The binance feed runs
as its own child task inside the `CompositeAdapter` with the adapter's existing
reconnect/backoff, so a binance disconnect cannot wedge the HL/PM event flow
(they share a queue but each child reconnects independently). No HL/PM slot
holds a BTCUSDT position, so this is purely additive to existing watchdog
semantics — no safety gate was added or removed.

## Inert for existing slots (verified)

Adding the feed does **not** change live trading: `MarketState` only feeds the
σ/OHLC reference from a BBO mid when `reference_source_for(symbol) == "bbo"`. The
default is `"mark"`, so with the feed added the BTCUSDT book is populated but
`recent_returns("BTCUSDT")` / `last_mark("BTCUSDT")` stay empty until a slot
opts in via `reference_sigma_source: bbo`. Pinned by
`test_btcusdt_book_populated_but_unread_for_sigma_by_default`.

## Dependency framing (corrected vs. the original task prompt)

The original task assumed the σ-source **consumer** (`reference_sigma_source:
mark|bbo`, per-bucket OHLC, `recent_hl_bars`) was on a separate unmerged branch
(`__WrkvCgsO8p`). **That is stale.** Local `main` already has the consumer
merged at `dc38388` (from `feat/engine-bbo-sigma-source-parkinson-fix`), which is
this branch's base. So this feed **completes** the PM-σ-from-bbo path end to end:

- Feed side (this change): BTCUSDT BboEvents now flow into MarketState.
- Consumer side (already on main): a slot with `reference_sigma_source: bbo`
  reads `book`/BBO mid → per-bucket OHLC → σ.

No further branch merge is required for the capability. **Activation is an
operator step only:** flipping a PM slot to `reference_sigma_source: bbo` (and
the separate dt=5 cadence decision) in `config/strategy.yaml` + deploy. Both PM
slots share the `BTCUSDT` reference, so the source/cadence must move in lockstep
(MarketState fails fast on conflicting per-symbol registration).

## Resource note (t4g.micro, 1G swap)

The feed is one extra WS stream (`btcusdt@bookTicker`) and one extra child task.
No REST poll, no trades/book/mark/funding ingest. bookTicker on BTCUSDT is a
modest message rate; memory cost is one `_MutableBook` + the per-symbol OHLC
deque (sized by the consuming slot's lookback/cadence, only allocated once a slot
is bbo-sourced). Negligible against the existing HL + PM ingest.

## Tests

`tests/unit/test_engine_reference_feed.py` (10 tests, all green):
- subscription shape (bbo-only BTCUSDT perp), exactly one binance sub, HL/PM
  preserved verbatim, composite routes binance → BinanceAdapter;
- bbo-only ⇒ no REST premium poll (+ positive control: `mark` does spawn it);
- real adapter `_handle` perp bookTicker → BTCUSDT BboEvent;
- BboEvent → `book("BTCUSDT")` keyed on recv/exchange ts, incl. through the
  CompositeAdapter;
- default mark-source ⇒ book populated but σ series empty.

Full suite: **671 passed** (was 661; +10 new).

## Follow-ups / dependencies

- **No deploy.** Activation = operator edit to `config/strategy.yaml`
  (`reference_sigma_source: bbo` on a PM slot, ± dt=5) + deploy. Real money.
- Consumer already on `main` (`dc38388`) — no further merge needed.
- When a PM slot is flipped to bbo, both PM slots sharing `BTCUSDT` must agree on
  source AND cadence (fail-fast guard in `MarketState`).
