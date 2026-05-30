# PM recorded-book fills — synthetic vs recorded smoke (2026-05-31)

## What changed
Polymarket BTC Up/Down binary backtests can now fill against the **real recorded
multi-level L2 order book** instead of the synthetic single-level book, matching
the fill realism HL HIP-4 already has.

- New `PolymarketDataSource(book_source="synthetic"|"recorded")`
  (default `"synthetic"`, bit-identical to prior behavior).
- `"recorded"` reads `event=book_snapshot` parquet per leg token over the
  market window (duckdb, mirroring `_load_binance_bbo_reference`), normalizes
  level ordering (best bid = `max(bid_px)`, best ask = `min(ask_px)`; bids px
  DESC, asks px ASC), and emits real multi-level `BookSnapshot`s. It DROPs the
  synthetic `trade_to_l2` book + the `1−p` within-pair parity synthesis — we
  have the real book for both legs. Real PM `TradeEvent`s and `ReferenceEvent`s
  are unchanged.
- The matching engine was **not** touched — `_build_leg_event_array` already
  depth-walks every level it's fed. The fix was purely the data source: HL fed
  real multi-level books, PM fed synthetic 1-level. That asymmetry is closed.
- CLI: `--pm-book-source {synthetic,recorded}` (default `synthetic`).

## Coverage caveat
The native PM book recorder is new — `book_snapshot` coverage starts
**2026-05-27**. Only BTC binary markets open in that window have real-book data:
**exactly 2** markets (settle 2026-05-27 and 2026-05-28) currently have both-leg
coverage. The 05-27 market's 24h trading window largely predates the recorder,
so its recorded book is partial. This is a **capability + smoke-eval, not a
strategy decision** — the window is far too small to be load-bearing.

## Smoke result (v1 `late_resolution` prod slot, pm_binary fees)
`scripts/run_pm_recorded_book_smoke.py`, window 2026-05-27 → 2026-05-29:

| mode      | mkts | trades | total PnL | hit  | max DD | taker fills | avg slip (¢) | avg fill | avg mid |
|-----------|-----:|-------:|----------:|-----:|-------:|------------:|-------------:|---------:|--------:|
| synthetic |    2 |      6 |   $76.00  | 100% |  $0.00 |           4 |        0.500 |  0.8850  | 0.8825  |
| recorded  |    2 |      4 |   $26.44  |  50% |  $0.00 |           3 |        0.500 |  0.8933  | 0.8917  |

**Reading it:** recorded mode produces fewer fills and different PnL. The
synthetic book offers a flat 10k-deep level at `price ± 0.5¢` at every trade
print, so the strategy always fills at a generous, deterministic price. The real
book is sparser (no synthetic depth between prints; partial coverage on the
05-27 market) and the realized entries land at a slightly different price level,
flipping one of the two markets' outcomes (100% → 50% hit) and cutting PnL. Avg
taker slippage-vs-mid is ~0.5¢ in both modes (synthetic by construction = the
half-spread; recorded coincidentally similar at p≈0.88 where these markets
trade). With n=2 this is illustrative only.

## Winner calculation now driven by the recorder `settlement` event
`PolymarketDataSource.resolved_outcome` (binary) is now sourced from the
recorder's on-chain `event=settlement` parquet — the winning leg token redeems
at `settle_price≈1.0`, and only the winning side is published, so the leg with
the higher recorded settle price (≥0.5) is the winner. The manifest's
Gamma-derived `resolved_outcome` is the **fallback** for markets with no
recorder coverage (the whole pre-2026-05-27 corpus), so legacy backtests stay
bit-identical. This is independent of `book_source` (it governs settlement
payoff, not fills).

Validated on the 2 book-covered markets (independent ground-truth cross-check):

| market | recorder settlement | manifest | strike-implied | backtest payoff |
|--------|--------------------|----------|----------------|-----------------|
| 0xe09b (May 27) | `no` (no_px=1.0) | `no` | `no` (75,326 < 76,461) | NO→1.0 / YES→0.0 |
| 0x602f (May 28) | `no` (no_px=1.0) | `no` | `no` (72,915 < 75,326) | NO→1.0 / YES→0.0 |

All four independent sources agree; smoke PnL is unchanged ($76 synthetic /
$26.44 recorded) because settlement == manifest for both markets today. The
change makes the on-chain settlement authoritative going forward.

## Acceptance
- `book_source="synthetic"` (default) event stream / fills / PnL bit-identical
  to `main` — asserted in `test_synthetic_default_bit_identical` plus the full
  existing `test_polymarket_source.py` suite (13 tests, unchanged).
- `book_source="recorded"` emits multi-level `BookSnapshot`s from real parquet
  with correct best-bid/ask normalization and per-leg join — new unit tests.
- Recorded-mode backtest runs end-to-end on the 2 covered markets, produces
  fills, and skips missing-coverage legs cleanly (logged, no crash).
- Full suite green: `uv run pytest` → 631 passed.
