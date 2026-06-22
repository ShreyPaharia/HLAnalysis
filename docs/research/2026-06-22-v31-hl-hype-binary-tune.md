# v31/theta HL **HYPE** binary — first walk-forward characterisation

**Date:** 2026-06-22 · **Scope:** HL HIP-4 `priceBinary`, underlying **HYPE**,
v3_theta_harvester ("v31"/theta). **Decision: DO NOT open a live HYPE binary slot.
Insufficient data + no demonstrated edge.** Analysis only — no config change, no
deploy. This is the FIRST look at HYPE binaries (recorder-only since 2026-06-15);
there is no existing live HYPE slot.

## Headline (read this first)

- **Corpus is far too small to tune.** Only **n = 6** settled HYPE `priceBinary`
  questions exist, expiries **2026-06-16 → 2026-06-21** (one per day, 06:00 UTC;
  recording began 2026-06-15 20:35). A 6-question walk-forward split (3 early / 3
  late) is statistically meaningless. **n < 15 ⇒ underpowered**; every number below
  is anecdotal, dominated by 1–2 questions.
- **The BTC-tuned config loses money on HYPE.** The current live HL-binary block
  (`msd2.5/esd1.5, vlb900, dt5, fav0.85, eb0.02, tte43200`) returns **−$48.26** full
  / Sharpe −6.8 / hit 17% / maxDD $66 / 44 trades, and is **P(loss)=100%** under the
  loss-injection tail stress. Params do **not** transfer from BTC.
- **Reference feed works.** HYPE is **not on Binance**; the backtest sources price/σ
  from HL's own perp (`--ref-source hl_perp`, the default). σ computes **non-zero**
  (annualised **0.62–0.73**, plausibly higher than BTC's ~0.5) — results are not a
  frozen/absent-reference artifact.
- **The only "profitable" cells trade almost nothing.** Raising the entry floor to
  `min_safety_d=3.0` flips full PnL to +$36 — but only by entering 2 ultra-deep
  favorites and skipping the whipsaw-prone ones. That is overfitting to which of 6
  questions got skipped, not an edge.

## Data existence + span (acceptance #1)

`HLHip4DataSource.discover(underlying="HYPE", kinds=("priceBinary",))` over
2026-05-01→2026-07-01:

| underlying | settled `priceBinary` n | span (expiries) |
|---|---|---|
| **HYPE** | **6** | 2026-06-16 → 2026-06-21 |
| ETH | 6 | (same window, also newly recording) |
| SOL | 6 | (same window) |
| BTC | 45 | 2026-05-10 → 2026-06-21 |

HYPE question ids: `Q1000404, Q1000412, Q1000420, Q1000428, Q1000468, Q1000476`.
HL-perp reference data (mark + bbo) is present for all 6 dates.

## Method

- **Runner:** `experiments/scripts/run_v31_hl_hype_binary_sweep.py` — mirrors the BTC
  band sweep (`run_v31_hl_safetyd_sweep.py`): ThreadPool over `hl-bt run --kind binary
  --underlying HYPE`, parse `report.md`, 3/3 walk-forward split, rank by **worst-half
  PnL**. Data reused from the main checkout (`HLBT_HL_DATA_ROOT=…/HLAnalysis/data`).
- **Windows:** full 06-15→06-22 (n=6); early 06-15→06-19 (Q…404/412/420); late
  06-19→06-22 (Q…428/468/476).
- **Anchor:** the **actual** current live HL-binary theta block (config/strategy.yaml,
  2026-06-22). NB: the task brief quoted `vlb1800/fav0.80/eb0.01`; the real committed
  config is `vlb900/fav0.85/eb0.02` — we anchored on the real config (source of truth).
- **Axes swept (OAT around anchor + the safety band 2D grid):** `min_safety_d ∈
  {0,1.5,2,2.5,3}` × `exit_safety_d ∈ {0,1,1.5,2}`; `favorite_threshold ∈
  {0.75,0.8,0.9}`; `edge_buffer ∈ {0,0.01,0.03}`; `vol_lookback_seconds ∈
  {1800,3600}`; `vol_sampling_dt_seconds ∈ {60}`; `tte_max_seconds ∈ {21600,86400}`.
- **Tail stress:** `experiments/scripts/run_v31_hl_hype_binary_tail_stress.py` —
  the 2026-06-18 loss-injection model (each entered-and-held winning favorite flips to
  a loss at its implied rate `1−entry_price`; full-stake under buy-and-hold, empirical
  adverse-exit loss when soft exits on). Validated by reproducing the buy-and-hold
  collapse.

A code fix was required to run any non-BTC underlying: the parallel/serial backtest
worker re-discovers `question_id → descriptor` with `discover()`'s **default
underlying=BTC**, so it raised *"worker could not re-map question_id"* for every HYPE
question. Fixed by carrying `discover_underlying` on the picklable `SourceConfig` and
replaying it in both workers (`parallel.py`, `tuning.py`) — the discovery-filter
instance of the documented worker-factory config-drop bug class. Unit-tested in
`tests/unit/backtest/test_source_config.py`.

## Anchor reproduced on HYPE (acceptance #2)

Per-question (anchor `msd2.5/esd1.5`):

| qid | outcome | trades | realized |
|---|---|---:|---:|
| Q1000404 | yes | 0 | $0.00 (no entry) |
| Q1000412 | yes | 2 | **+$17.78** |
| Q1000420 | no | 18 | −$18.27 |
| Q1000428 | no | 0 | $0.00 (no entry) |
| Q1000468 | yes | 21 | **−$47.77** |
| Q1000476 | yes | 0 | $0.00 (no entry) |
| **total** | | 44 | **−$48.26** (Sharpe −6.8, hit 17%, maxDD $66) |

**Loss mechanism = soft-exit whipsaw on volatile HYPE.** Q1000468 *resolved YES* yet
lost $47.77: the strategy bought 1495 sh of the YES favorite at avg 0.948, then the
σ√τ-distance mid-hold exit (`esd=1.5`) dumped 1471 sh at avg **0.915** as HYPE drifted
toward the adverse boundary — and it then recovered to settle YES. The exit sold a
winner at a loss. HYPE's higher realised σ pushes the soft exit into whipsaw far more
than on BTC.

## Ranked sweep (acceptance #3) — n=6, EXPLORATORY ONLY

Sorted by worst-half PnL (full / Sharpe / maxDD / early / late / worst-half / trades / hit):

| config | full | Shrp | maxDD | early | late | worst½ | trd | hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **anchor msd2.5/esd1.5** | **−48.26** | −6.8 | 66.05 | −0.49 | −47.77 | −47.77 | 44 | 17% |
| msd3.0_esd0.0 / 1.0 / 1.5 | 35.92 | 11.9 | 0.00 | 21.92 | 14.00 | **+14.00** | 14 | 33% |
| msd3.0_esd2.0 | 30.97 | 11.0 | 0.00 | 21.92 | 9.05 | +9.05 | 18 | 33% |
| msd0.0_esd0.0 | 87.94 | 10.7 | 3.48 | 91.42 | −3.48 | −3.48 | 48 | 33% |
| vol_lookback=1800 | 27.91 | 6.6 | 4.01 | 31.91 | −4.01 | −4.01 | 9 | 17% |
| msd1.5_esd0.0 | 100.43 | 10.9 | 4.57 | 105.00 | −4.57 | −4.57 | 36 | 33% |
| msd2.0_esd0.0 (buy&hold) | 59.16 | 7.9 | 15.36 | 72.98 | −13.82 | −13.82 | 29 | 50% |
| tte_max=21600 (6h) | −16.43 | −3.1 | 34.22 | −16.43 | 0.00 | −16.43 | 24 | 17% |
| edge_buffer=0.0 | −41.79 | −5.9 | 59.57 | −0.49 | −41.30 | −41.30 | 49 | 33% |
| fav 0.75 / 0.80, eb 0.01, tte 86400 | −48.26 | −6.8 | 66.05 | −0.49 | −47.77 | −47.77 | 44 | 17% |
| vol_sampling_dt=60 | −46.89 | −6.0 | 57.68 | 10.79 | −57.68 | −57.68 | 45 | 17% |
| msd2.0_esd1.0 | −147.59 | −7.4 | 169.42 | −132.43 | −15.15 | −132.43 | 43 | 33% |
| msd0.0_esd2.0 | −413.95 | −13.9 | 438.90 | −178.17 | −235.78 | −235.78 | 321 | 17% |
| msd1.5_esd2.0 | −680.11 | −11.4 | 711.08 | −444.42 | −235.70 | −444.42 | 300 | 17% |

(Full 32-cell grid in `data/sim/runs/v31-hl-hype-binary-2026-06-22/results.json`.)

What the grid actually shows — all of it small-n noise:

- **`favorite_threshold` and `edge_buffer` are inert** (0.75/0.80/0.01 are *bit-identical*
  to the anchor): every HYPE favorite the strategy enters is priced ≥0.92, far above
  the favorite gate, so loosening it changes nothing.
- **`exit_safety_d` is inert at `msd=3.0`** (esd 0.0/1.0/1.5 all +$35.92): the 2 deep
  favorites it enters never approach the soft-exit boundary, so exit policy can't be
  distinguished there.
- The single lever that matters is the **entry floor**: `msd 2.5→3.0` flips −$48→+$36
  purely by gating out the two whipsaw-prone entries (Q1000420, Q1000468). The
  `msd3.0` family is the only group **positive in both halves** — but on **2 entries**.
- **Aggressive exits cascade** (`esd=2.0` with low floors: −$266 to −$680, 200–320
  trades): on a cliff-payoff instrument with HYPE-level vol, low-floor + high-exit
  thrashes.

## Tail stress (acceptance #4)

Loss-injection, N=20000, full corpus (n=6):

| config | esd | realized | wins | EV | 5th-pct | P(loss) |
|---|---:|---:|---:|---:|---:|---:|
| **anchor msd2.5/esd1.5** | 1.5 | −$48 | 1 | **−$50** | −$48 | **100.0%** |
| msd3.0_esd1.5 (grid winner) | 1.5 | +$36 | 2 | +$30 | −$28 | 8.5% |
| msd3.0_esd2.0 | 2.0 | +$31 | 1 | +$28 | +$31 | 4.1% |
| msd2.0_esd1.5 | 1.5 | −$43 | 1 | −$45 | −$43 | 100.0% |
| msd2.0_esd0.0 (**VALIDATION**, buy&hold) | 0.0 | +$59 | 3 | −$19 | **−$522** | 17.3% |

**Validation passes:** the buy-and-hold cell collapses to a 5th-pct of **−$522** from
full-stake flips, reproducing the 2026-06-18 model behaviour — so the relative tail
numbers are trustworthy. The anchor is a certain loser (P(loss)=100%); only the
trade-almost-nothing `msd3.0` cells survive the tail, on 1–2 winning entries.

## Verdict (acceptance #5)

**No tradeable edge has been demonstrated, and the data is too thin to claim one
either way. Do not open a live HYPE `priceBinary` slot now.**

1. n=6 over 6 days is ~3× below the n≥15 power floor; a walk-forward split is noise.
2. The BTC-tuned config is an outright loser on HYPE (−$48, P(loss)=100% stressed),
   driven by a concrete, repeatable mechanism (soft-exit whipsaw under HYPE's higher
   σ), not just sample luck — so "lift-and-shift the BTC block to HYPE" is rejected.
3. The least-bad config (`min_safety_d=3.0`, i.e. enter only ≥3σ√τ favorites) is the
   *direction* a real HYPE tune would likely take — HYPE's higher vol wants a higher
   entry floor than BTC — but +$36 from 2 entries is anecdotal, not promotable.

**Recommended next step:** re-run this exact harness **on/after ~2026-06-29** (≈2 more
weeks of settled HYPE questions → n≈20), and only then consider a *paper* HYPE slot.
When tuning for real, start the entry floor **higher than BTC** (`min_safety_d ≥ 3.0`)
and treat the soft exit cautiously — on HYPE it whipsawed winners. Do not pre-commit
any HYPE params from this run.

## Caveats (acceptance #6)

- **Corpus size: n=6, 6 days.** The dominant caveat. Treat all $ figures as
  illustrative, not estimates.
- **No Binance reference for HYPE.** σ/p_model come from the HL perp only; there is no
  cross-venue feed to validate the HL reference against. σ is non-zero and sane, but
  unverified against an independent source.
- **Fill realism.** Backtest fills the recorded HL book; `--slippage-bps` is a no-op on
  recorded-book fills. The high-turnover cascade cells (200–320 trades over 6 days) are
  the most fill-model-sensitive and least trustworthy.
- **Tail-deficiency.** Like the BTC corpus, the realized HYPE sample under-samples
  losing favorites; the loss-injection stress is the partial correction, but with only
  1–3 winners to flip it is itself low-resolution here.
- **Single underlying, single corpus.** ETH and SOL binaries are at the same n=6 — a
  later multi-underlying study (BTC/ETH/SOL/HYPE) once each has n≥15 would be the
  proper basis for a non-BTC slot.

## Artifacts

- Sweep runner: `experiments/scripts/run_v31_hl_hype_binary_sweep.py`
- Tail stress: `experiments/scripts/run_v31_hl_hype_binary_tail_stress.py`
- Raw runs: `data/sim/runs/v31-hl-hype-binary-2026-06-22/` (per-cell `report.md` +
  `fills.parquet`, `results.json`)
- Worker discovery fix: `hlanalysis/backtest/core/source_config.py`
  (`discover_underlying` + `discover_kwargs()`), wired in `runner/parallel.py`,
  `tuning.py`, `_cli_plumbing.py`; tested in
  `tests/unit/backtest/test_source_config.py`.
