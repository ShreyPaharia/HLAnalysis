# HL Outcome-Market Desk Analysis & Strategy Hunt — Design

**Date:** 2026-06-13
**Status:** approved, in implementation
**Live progress:** `docs/research/hl-outcome-desk-progress.html`

## Goal

As a desk manager, characterize 36 days (2026-05-06 → 2026-06-10) of granular HL data — perp
BTC + binary and bucket BTC outcome markets — let the data nominate tradable edges, and propose
strategies scored against hard KPIs. Build a reusable `hlanalysis/research/` toolkit so reruns
take minutes. **Ordering: taker edges first, market-making second.**

## Operator decisions

- Taker first → MM second.
- Full landscape, no prior thesis.
- KPIs calibrated to the live $1k→$25k desk on the t4g.micro box; realistic fills/fees; deployable.
- Exhaustive multi-agent build (sonnet subagents).
- MM hedge leg modeled on **HL perp** (Binance perp geo-blocked from dev IP).
- This run stops at **backtest + KPI scorecard** — no live-slot registration.
- Data read-only from the main checkout at `../../data`; never `make pull-data`.

## Data layout (verified)

Hive-partitioned parquet under `../../data` (from this worktree):

```
data/venue=hyperliquid/product_type=perp/mechanism=clob/event={bbo,trade,mark,book_snapshot,funding,open_interest,oracle}/symbol=BTC/date=YYYY-MM-DD/hour=all/compacted.parquet
data/venue=hyperliquid/product_type=spot/.../event={bbo,trade,mark,book_snapshot}/symbol=BTC/...
data/venue=hyperliquid/product_type=prediction_binary/mechanism=clob/event={bbo,trade,mark,book_snapshot,settlement,market_meta,question_meta,health}/symbol=#NNN|QNNNN/date=.../
data/venue=binance/product_type={spot,perp}/.../event={bbo,trade,book_snapshot,mark,funding}/symbol=.../
```

Key facts:
- Outcome markets are daily (`period=1d`), expiry `YYYYMMDD-0600` UTC.
- `question_meta` has `class ∈ {priceBinary, priceBucket}`. Binary → single `targetPrice`
  (above/below). Bucket → `priceThresholds="lo,hi"` (price band).
- Per-leg order data keyed by `symbol=#NNN`; map `#NNN` → outcome_idx → question/class/leg via
  `market_meta` (keys: outcome_idx, side_idx, side_name∈{Yes,No}, outcome_name, class, underlying,
  expiry, targetPrice, period) and `question_meta`.
- Timing: use `local_recv_ts` (ns) for ordering, NOT `exchange_ts` (HL trade exchange_ts = block
  time, lagged ~5s; Binance spot exchange_ts is sentinel 0). HL perp transport latency ~225ms
  median — subtract per-venue median latency before cross-venue lead-lag claims.
- Settlement events (99) carry resolution outcomes; join for win labels.
- Existing helpers to build on: `hlanalysis/analysis/{helpers,book,markouts,microstructure}.py`
  (DuckDB-over-parquet). Backtest framework: `hlanalysis/backtest/` with `book_source=recorded`.

## Reusable toolkit — `hlanalysis/research/`

- `outcome_markets.py` — resolver: `#NNN`/`QNNNN` ↔ question ↔ class ↔ expiry ↔ strike/thresholds
  ↔ Yes/No leg ↔ settlement outcome. Returns tidy reference frames. This is the dependency for
  every downstream card.
- `dataset.py` — cached cross-venue research panels: aligned, resampled (configurable `dt`) frames
  of perp mid/σ + each outcome market mid/depth/trades + settlement label. Parquet-cached, keyed by
  (date-range, dt, symbols, version). Reuses `../../data` via `HLBT_*` env. Build once, reuse.
- `metrics.py` — one tested definition each: spread (bps), depth@N, trade markout curve,
  cross-venue lead-lag xcorr, Yes+No overround, GBM implied-prob (with Itô −½σ²τ), theta-decay
  curve, settlement-convergence curve, realized-vol term structure (Parkinson/bipower).
- `report.py` — self-contained HTML builder: card registry, base64-embedded matplotlib plots,
  no server. Each analysis appends a card; one command regenerates the deck.
- CLI/make target to regenerate the full report from cache.

## Analysis cards (Wave 1)

Taker-relevant first: **E** settlement/resolution convergence, **D** mispricing surface
(implied-prob vs GBM, Yes+No overround, bucket↔binary no-arb), **F** vol term structure & theta.
Then MM-leaning: **B** adverse selection/markouts, **A** liquidity & book shape, **C** cross-venue
lead-lag (perp→outcome mid).

## Strategy candidates (Wave 2)

Taker first: (1) cross-market static arb (Yes+No overround, bucket↔binary no-arb, fee-adjusted),
(2) v1/v31 refinement (new gates/sizing from B/D/E/F), (3) settlement-convergence capture. Then
MM: quoter + HL-perp delta hedge with inventory + adverse-selection gating. Plus a reusable
sizing/risk module (edge-proportional, adverse-selection haircut, Kelly-capped, inventory-aware).

## KPIs

**Analysis (trust gate):** one-command reproducible from recorded data; every claim has n +
date-span + sanity cross-check; ≥30 expiries / ≥30 days; split-half (first 18d vs last 18d) sign
stability for edge claims.

**Strategy (promote-to-paper gate):** positive net PnL after realistic fees on `book_source=recorded`
fills, walk-forward OOS (≥7d held out); daily Sharpe ≥1.0 OOS; max DD within desk inventory cap;
split-half sign stability; survives adversarial verification (no lookahead, realistic fills, safety
gates intact); capacity ≥ current desk; deployable on existing engine surface.

## Deliverables

- `hlanalysis/research/` toolkit + tests.
- `docs/research/hl-outcome-desk-progress.html` — live dashboard (kept current throughout).
- `docs/research/hl-outcome-desk-2026-06-13.html` — final report: characterization cards + ranked
  strategy scorecard + prioritized recommendations.

## Execution

Wave 0 (toolkit, 1 agent, gate on smoke HTML + tests) → Wave 1 (6 parallel cards) → Wave 2
(strategy candidates) → Wave 3 (synthesis + one final review pass). All sonnet. No live/prod
changes; no `pull-data`.
