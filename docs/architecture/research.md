# Research toolkit — `hlanalysis/research/`

Read when doing **desk-level data analysis** on recorded data: characterizing
liquidity / mispricing / vol / adverse-selection, hunting strategy edges, or
producing a self-contained HTML findings report. This is an **offline, read-only,
additive** subsystem — it never touches the live engine, the strategy registry, or
prod config. It exists so analyses that used to take hours of ad-hoc DuckDB take
minutes and stay reproducible.

It is distinct from `hlanalysis/analysis/` (the older notebook helpers): `research/`
builds *on* those (`analysis/book.py`, `markouts.py`, `microstructure.py`,
`helpers.py`) and adds outcome-market label resolution, cached cross-venue panels,
standardized desk metrics, an HTML card builder, and ready-made analysis "cards".

## The one fact that bites: outcome-market win labels

**HL binary markets emit NO settlement event.** The `settlement` parquet only
contains bucket *band* legs, and its `settled_side_idx` is a **constant artifact
(always 0)** — NOT the winner. Deriving winners from settlement presence/absence or
from `settled_side_idx` is wrong and silently corrupts every win-dependent metric.

Authoritative win labels (verified, baked into the resolver):
- **Binary:** `yes_won = oracle_px(expiry) > targetPrice`, where `oracle_px` is the
  HL perp oracle (`product_type=perp/event=oracle/symbol=BTC`, col `oracle_px`)
  as-of the expiry ns, and `targetPrice`/`expiry` come from `market_meta` for the
  binary Yes leg (`outcome_name='Recurring'`, `side_name='Yes'`).
- **Bucket:** exactly one band wins; `band_won = lo < oracle_px(expiry) <= hi` from
  `question_meta.priceThresholds`. Cross-checked against terminal book mid (~100%
  agreement in the 36-day corpus).
- Symbol decode: `#NNN` → `outcome_idx = NNN // 10`, `side_idx = NNN % 10`
  (0=Yes, 1=No). Timing is always `local_recv_ts` (ns), never `exchange_ts`.

Sanity numbers on the 2026-05-06..06-10 corpus: 38 binary expiries, **47.4%**
Yes-win; 37 bucket questions × 3 bands, one winner each.

## File map

| File | Responsibility |
|------|----------------|
| `outcome_markets.py` | The resolver. `resolve_binary_outcomes` / `resolve_bucket_outcomes` (oracle-derived win labels), `load_market_reference`, `load_settlements`. Every win-dependent analysis depends on this. |
| `dataset.py` | `build_panel(symbols, start, end, dt_seconds, data_root, cache_dir, fresh)` — aligned LOCF-resampled cross-venue panel (perp mid/vol + per-leg mid/bid/ask/depth/trade-count + static attrs + settlement label). Parquet-cached by content hash; memory-bounded (day-by-day). |
| `metrics.py` | One tested function per desk metric: `spread_bps`, `depth_at_n`, `trade_markout_curve`, `leadlag_xcorr`, `yes_no_overround`, `implied_prob_gbm` (GBM with Itô −½σ²τ), `theta_decay_curve`, `settlement_convergence_curve`, `realized_vol_termstructure` (Parkinson + bipower). Pure; unit-tested with known-answer synthetic inputs. |
| `report.py` | `Report` (card registry → one standalone HTML, base64-embedded matplotlib, no server) + `fig_to_base64`. Dark theme matching `docs/research/*.html`. |
| `cards/card_{a..f}_*.py` | The six characterization cards (liquidity, adverse-selection, lead-lag, mispricing, convergence, vol/theta). Each exposes `build_card(con, data_root) -> (html, findings_dict)` and a `__main__` that writes `docs/research/_cards/card_X.{html,json}`. |
| `cards/strategy_taker_flb.py`, `cards/strategy_mm.py` | Strategy backtest cards (favorite-longshot taker; market-maker + perp hedge). Same `build_card` contract. |
| `smoke.py` / `__main__.py` | `python -m hlanalysis.research.smoke` → smoke report from a 2-day slice (toolkit acceptance gate). |

Also: `scripts/research_clean_flb.py` — a standalone, dependency-light DuckDB
backtest of the clean favorite-longshot rule (the authoritative taker scorecard,
independent of the live-strategy gates). Good template for a quick, trustworthy edge
check that doesn't route through the engine.

## How to run it (from a worktree)

Data lives **only in the main checkout**; from `.worktrees/<name>/` it is `../../data`.
**Never `make pull-data`.**

```bash
export HLBT_HL_DATA_ROOT=../../data            # HL recorded venue data

# Smoke (acceptance gate) → docs/research/_smoke_report.html
HLBT_HL_DATA_ROOT=../../data uv run python -m hlanalysis.research.smoke

# Regenerate one characterization card → docs/research/_cards/card_e.{html,json}
HLBT_HL_DATA_ROOT=../../data uv run python -m hlanalysis.research.cards.card_e_convergence

# Authoritative clean-FLB taker backtest → docs/research/_cards/clean_flb_authoritative.json
HLBT_HL_DATA_ROOT=../../data uv run python scripts/research_clean_flb.py

# Tests (fast; data-dependent ones skip cleanly if ../../data is absent)
uv run pytest -q tests/research
uvx ruff check hlanalysis/research tests/research && uvx ruff format --check hlanalysis/research tests/research
```

The findings deck is `docs/research/hl-outcome-desk-<date>.html` (final report) +
`docs/research/hl-outcome-desk-progress.html` (live log). Each card's
`findings` dict carries `{title, headline, metrics:[{name,value,n,date_span,sanity}],
split_half, verdict}`.

## Conventions for adding an analysis

1. **Reuse the resolver for any win/label join** — never re-derive winners from
   settlement. Reuse `metrics.py` definitions rather than inventing a second spread /
   markout / implied-prob formula.
2. **Every quantitative claim carries `n` + date-span + a sanity cross-check**, and
   any *edge* claim must report **split-half** (first 18d vs last 18d) sign
   stability. Treat metrics on n<15 as underpowered, not a pass.
3. **Validate on recorded inputs only** (recorded book/bbo at the live cadence) — no
   synthetic/stale klines. A "passing" result on the wrong inputs is worse than none.
4. **Don't trust unreviewed simulated PnL.** Backtest cards that loop over markets
   can emit inflated numbers; anchor edge claims in *measured* quantities (realized
   spread, markout) and review sim output before quoting a dollar figure.
5. New card → new `cards/card_*.py` with the `build_card` contract + a light test in
   `tests/research/`. Keep modules small and focused.

## Findings index (2026-06-13 desk study)

The first full run of this toolkit characterized 36 days of HL BTC perp + binary +
bucket data. One-line conclusions (full report:
`docs/research/hl-outcome-desk-2026-06-13.html`):

- **Edge:** favourite-longshot bias — favorites (mid 0.80–0.95) underpriced ~6–9pp,
  root-caused to a sign-stable **+14% ann implied−realized vol premium**. Survives
  adverse selection (Card B) and fees (HL exchange fee = 0).
- **Vehicle:** real but capacity-starved as a single-name BTC-binary bet (~1
  favorite/day, fat left tail → OOS-fragile). Deployable only via **breadth** across
  many markets.
- **Prod:** **v31 (theta) is the scalable workhorse** (+$449 IS / +$50 OOS, 293/44
  trades); v1 keep-but-don't-scale. Scaling bottleneck is capacity ($107 top-of-book)
  → grow by adding markets, not clip size.
- **MM:** maker side +EV (Card B: +0.95pp realized spread net of adverse selection).
  Latency requirement: cancel/replace < ~1s (binary reprices to perp with 1–2s
  half-life). The live engine is event-driven **~5 Hz** (`scan_min_interval_seconds
  0.2`), so MM is **viable on current infra** pending a one-time end-to-end
  cancel-replace RTT measurement — not infra-blocked.
- **Dropped on evidence:** static Yes+No arb (0 of 3.18M ticks), perp-momentum taker
  (R²=0.08, dies to the spread).
