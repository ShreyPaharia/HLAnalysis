# v31/theta HL **HYPE** binary — first walk-forward characterisation

**Date:** 2026-06-22 · **Scope:** HL HIP-4 `priceBinary`, underlying **HYPE**,
v3_theta_harvester ("v31"/theta). **Decision: still DO NOT open a live HYPE binary
slot yet (n=6), but a tail-safe profitable candidate DOES exist** — the full
cartesian grid (added below) found one the OAT sweep missed. Analysis only — no
config change, no deploy. First look at HYPE binaries (recorder-only since
2026-06-15); no existing live HYPE slot.

> **Update (full grid).** This doc has two passes. **Pass 1 (OAT)** perturbed one
> axis at a time around the BTC anchor (frozen at the live `vol_lookback=900`) and
> concluded "no edge / whipsaw". **Pass 2 (full 4608-config cartesian)** — added at
> the user's request because HYPE is a new token whose joint optimum need not sit
> near BTC — **overturns that**: `vol_lookback` is the decisive lever the OAT held
> fixed. With `vol_lookback=3600` (1h, not the live 900) the soft-exit whipsaw
> disappears and a **tail-safe** cell emerges:
> `vl3600/dt5/tte43200/msd2.5/esd1.0` → EV **+$104** / 5th-pct **+$35** / P(loss)
> **1.7%**, positive in both walk-forward halves. See **"Full cartesian grid"** below;
> it supersedes Pass 1's verdict. Everything is still n=6 — promising, not promotable.

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
- **But the full grid finds a tail-safe profitable cell** the OAT missed: with
  `vol_lookback=3600` (not the live 900), `vl3600/dt5/tte43200/msd2.5/esd1.0` returns
  +$123 (both halves positive, Sharpe 23.8) and survives the tail stress (EV +$104 /
  5th-pct +$35 / P(loss) 1.7%). The lever is the σ-lookback, not the entry floor.
  Still n=6 — promising, not promotable.

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

## Pass 1 — OAT ranked sweep (acceptance #3) — n=6, EXPLORATORY ONLY

*One-axis-at-a-time around the BTC anchor, with `vol_lookback` frozen at the live
900s. Superseded by the full grid below, which frees `vol_lookback` — read Pass 2
for the actual recommendation.*

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
numbers are trustworthy. The anchor is a certain loser (P(loss)=100%); within the
vl900-anchored OAT slice only the trade-almost-nothing `msd3.0` cells survive the tail.

## Pass 2 — FULL cartesian grid (the complete sweep)

Because HYPE is a new token, the OAT pass (which froze `vol_lookback=900`) can't see
joint optima. Pass 2 sweeps the **full 7-axis cartesian product = 4608 configs**:

| axis | levels |
|---|---|
| favorite_threshold | 0.75, 0.80, 0.85, 0.90 |
| edge_buffer | 0.0, 0.01, 0.02, 0.03 |
| vol_lookback_seconds | 900, 1800, 3600 |
| vol_sampling_dt_seconds | 5, 60 |
| tte_max_seconds | 21600, 43200, 86400 |
| min_safety_d | 1.5, 2.0, 2.5, 3.0 |
| exit_safety_d | 0.0, 1.0, 1.5, 2.0 |

Run via the shared-decode warm-chunk driver `scripts/perf/resumable_run.py`
(`--underlying HYPE`, added for this study): one pass over the full window, each
question decoded once and replayed against all 4608 configs (≈50 min wall, 27,648
cells); early/late halves split offline (q0–2 / q3–5). Runner:
`experiments/scripts/run_v31_hl_hype_binary_fullgrid.py`.

**Decisive lever = `vol_lookback`** (best achievable worst-half by level): **900 →
$19, 1800 → $47, 3600 → $112**. The live config inherited BTC's short `vl900`; that
short σ window is what made `safety_d` jumpy and drove the Pass-1 whipsaw. With
`vl3600` (1h) σ stabilises and the soft exit stops dumping winners. Other axis
marginals: `dt 5 ($112) ≫ 60 ($23)`; `tte 43200 ($112)` interior-best; `favorite_threshold`
and `edge_buffer` remain **inert** (the `vl3600/dt5/tte43200/msd2.5/esd1.0` cell is
$118–$123 across *all* fav/eb — keep them at the live 0.85/0.02).

Top of the full grid (worst-half), with the tail verdict from the stress below:

| config | full | early | late | worst½ | Sharpe | maxDD | trd | wins | tail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `vl3600 dt5 tte43200 msd1.5 esd0.0` (fav/eb any) | 224.78 | 112.60 | 112.18 | 112.18 | 20.7 | 0.00 | 21 | 4/6 | **REJECT — buy&hold, 5th −$788** |
| `vl3600 dt5 tte86400 msd1.5 esd0.0` | 213.50 | 102.59 | 110.91 | 102.59 | 15.7 | 0.00 | 51 | 4/6 | reject (esd=0) |
| **`vl3600 dt5 tte43200 msd2.5 esd1.0`** | **123.15** | 69.96 | 53.19 | **53.19** | 23.8 | 0.00 | 9 | 4/6 | **SURVIVES — EV +$104 / 5th +$35 / P(loss) 1.7%** |
| `vl3600 dt5 tte43200 msd2.5 esd1.5` | 66.7 | — | — | ~31 | — | 0.00 | — | 3/6 | survives — EV +$55 / 5th −$1 / P(loss) 8.3% |
| anchor `vl900 … msd2.5 esd1.5` | −48.26 | −0.49 | −47.77 | −47.77 | −6.8 | 66.04 | 44 | 1/6 | reject — EV −$50 / P(loss) 100% (**rank 2418/4608**) |

`min_safety_d` and `exit_safety_d` only *look* monotone-toward-zero in the raw
worst-half marginals because the unconstrained max is the (tail-blind) buy-and-hold
corner; among **tail-safe** (`esd>0`) configs the optimum is interior:
`msd2.5/esd1.0`. 713 of the 3456 `esd>0` configs have positive worst-half, but **999
of them post realized maxDD=$0** — the corpus simply never sampled a losing held
favorite, which is exactly why the loss-injection tail stress (not realized maxDD) is
the deciding metric.

### Full-grid tail stress (acceptance #4, Pass 2)

Loss-injection, N=20000, n=6 (`experiments/scripts/run_v31_hl_hype_binary_fullgrid_tail_stress.py`):

| config | esd | realized | wins | EV | 5th-pct | P(loss) |
|---|---:|---:|---:|---:|---:|---:|
| `vl3600 … msd1.5 esd0.0` (grid winner, buy&hold) | 0.0 | +$225 | 4 | +$5 | **−$788** | 35.4% |
| **`vl3600 … msd2.5 esd1.0`** (best tail-safe) | 1.0 | +$123 | 4 | **+$104** | **+$35** | **1.7%** |
| `vl3600 … msd2.5 esd1.5` | 1.5 | +$67 | 3 | +$55 | −$1 | 8.3% |
| `vl3600 … msd2.0 esd1.0` | 1.0 | −$47 | 3 | −$104 | −$285 | 100.0% |
| anchor `vl900 … msd2.5 esd1.5` | 1.5 | −$48 | 1 | −$50 | −$48 | 100.0% |

The literal grid winner (buy-and-hold) is the tail-blind trap — EV collapses to +$5
with a 5th-pct of −$788 and a 35% loss probability (validation: behaves like the
2026-06-18 BAH collapse). The standout is the **tail-safe** cell
`vol_lookback=3600 / dt=5 / tte=43200 / min_safety_d=2.5 / exit_safety_d=1.0`
(fav/eb at the live 0.85/0.02): positive in both halves, Sharpe 23.8, and the only
candidate with a **positive 5th-pct** (+$35) and ~zero loss probability under stress.

## Verdict (acceptance #5)

**A tail-safe, profitable HYPE-binary candidate exists, but n=6 is still far below
the power floor — characterise + paper-soak, do NOT open a live slot yet.**

1. **Recommended HYPE-binary cell (when one is opened):** the live BTC theta block
   with **two changes** — `vol_lookback_seconds 900 → 3600` and `exit_safety_d
   1.5 → 1.0`; keep `dt=5, tte_max=43200, min_safety_d=2.5, favorite_threshold=0.85,
   edge_buffer=0.02` (the last two are inert on this corpus). This cell:
   full +$123 / both halves positive / Sharpe 23.8 / stressed EV +$104 / 5th-pct
   +$35 / P(loss) 1.7%, robust across all favorite/edge levels.
2. **Why it differs from BTC:** HYPE's higher realised σ (0.62–0.73) needs a **longer
   σ-lookback (1h)** to keep `safety_d` stable; the live `vl900` made the soft exit
   whipsaw winners (the −$48 anchor / Pass-1 "no edge"). Params do not transfer.
3. **Reject the literal grid winner** (`esd=0` buy-and-hold, +$225): tail-blind
   (5th-pct −$788), the same trap rejected for BTC on 2026-06-18.
4. **Do not deploy on n=6.** Every "winner" posts realized maxDD=$0 because the corpus
   has no losing held favorite; the positive-tail verdict rests on the injection model.

**Recommended next step:** re-run the full grid **on/after ~2026-06-29** (≈n≈20), and
if `vl3600/dt5/tte43200/msd2.5/esd1.0` (or its neighbourhood) holds, stand up a
**paper** HYPE slot before any live capital. Do not pre-commit live params from n=6.

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

- **Pass 1 (OAT):** runner `experiments/scripts/run_v31_hl_hype_binary_sweep.py`,
  tail stress `…_tail_stress.py`; raw runs
  `data/sim/runs/v31-hl-hype-binary-2026-06-22/` (`results.json`).
- **Pass 2 (full grid):** runner
  `experiments/scripts/run_v31_hl_hype_binary_fullgrid.py` (drives
  `scripts/perf/resumable_run.py`), tail stress `…_fullgrid_tail_stress.py`; raw runs
  `data/sim/runs/v31-hl-hype-binary-fullgrid-2026-06-22/` (`aggregate.json` =
  4608-config ranked table; per-cell `<id>/qNNNN/report.md`).
- **Worker discovery fix** (enables any non-BTC `--underlying`):
  `hlanalysis/backtest/core/source_config.py` (`discover_underlying` +
  `discover_kwargs()`), wired in `runner/parallel.py`, `tuning.py`,
  `_cli_plumbing.py`; tested in `tests/unit/backtest/test_source_config.py`.
- **`resumable_run.py --underlying`** added (discover + argv + worker forwarding);
  tested in `tests/perf/test_resumable_run.py`.
