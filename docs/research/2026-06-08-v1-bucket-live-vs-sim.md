# v1 (late_resolution) HL priceBucket — live vs sim, fill-by-fill divergence

**Window:** 2026-05-31 → 2026-06-08 UTC
**Strategy:** `v1_late_resolution`, HL HIP-4 **priceBucket** track (the live `v1` slot)
**Date:** 2026-06-08

This mirrors the v31 live-vs-sim exercise: take the live v1 bucket fills off the
venue, reproduce the same window in the backtester, and explain — leg by leg —
*where* and *why* the two diverge. **Analysis only; no strategy/engine behaviour
was changed.**

## TL;DR

| | PnL | positions | hit |
| :-- | --: | --: | --: |
| **Live** (venue `closedPnl − fee`, 05-31→06-08) | **+$51.35** | 7 coins / 26 fills | 7/7 = 100% |
| **Sim** (`hl-bt run`, 05-31→06-07 corpus) | **+$36.32** | 6 traded / 12 fills | 6/8 questions resolved-win, 75% |

The headline gap ($51 vs $36) is **not** a single effect — it is four separate
divergences that partly cancel:

1. **Sim misses 06-04 #1510 (live +$8.28).** The 60 s scan grid's first eligible
   sample landed inside a momentary high-vol `safety_d` veto; live entered in the
   first 7 seconds and won.
2. **Sim trades 06-07 #1670 (sim +$10.56) that live never took.** This is the
   **SHR-78** artifact: the volume gate is disabled in sim, so the sim takes an
   entry set live's `min_recent_volume_usd=100` filtered out. This single phantom
   trade is ~29% of the sim's total PnL.
3. **On markets both took, sim under-performs live by ~$11** because live's
   sub-second cadence + partial-fill ladder (**SHR-79**) captured better entry
   prices (06-02 and 06-06 especially).
4. **06-08 #2230 (live +$6.12) is outside the corpus** — the archive's daily
   `hour=all` partition for 06-08 was not yet sealed at analysis time, and
   `--end 2026-06-08` is exclusive regardless.

**Corrected read:** strip the phantom 06-07 trade and the sim is +$25.77 on the
five shared coins, versus live's +$36.94 on those same coins. So once the
entry-set artifact is removed, **the sim is *pessimistic*, not optimistic** — the
idealized full-size-at-touch fill on a 60 s grid earns *less* than the live
engine's fast, laddered execution on these near-resolution favorites.

## Method & caveats

- **Live fills:** venue `user_fills`, read-only via AWS SSM (SSH blocked) on
  `i-0dc4c0abec85a9eda`, classified `priceBucket` by the engine's
  `coin_klass_map`. Script: `tools/dump_v1_bucket_fills.py`. HL HIP-4 settlement
  is exposed as a *fill* with `closedPnl` (the `sell @ 1.0000` rows are
  settlements, not market exits), so live net = Σ(`closedPnl − fee`); observed
  `fee = 0` on every HL bucket fill (SHR-57 a non-factor here).
- **Sim:** `hl-bt run --strategy v1_late_resolution --kind bucket --ref-source
  hl_perp --fee-taker 0 --slippage-bps 0`, config
  `config/backtest/v1_hl_bucket_since0531.json` (live v1 params).
- **SHR-78 caveat (load-bearing):** `hftbt_runner.py:814` hardcodes
  `recent_volume_usd=0.0`, so v1's `min_recent_volume_usd=100` gate would veto
  *every* entry → 0 trades. The config sets `min_recent_volume_usd=0.0` to make
  v1 trade at all. **Consequence:** the sim evaluates a *different (larger) entry
  set than live* — it cannot see the volume condition that live actually gates
  on. This is the single biggest interpretability caveat and directly causes
  divergence #2 above.
- **Corpus:** recorded HL feed synced from
  `s3://hl-recorder-archive-819175935435/venue=hyperliquid/` (the `make
  pull-data` set), 3.2 GB, `prediction_binary` book coverage 05-31→**06-07**.
  06-08 not present.

## Per-day / per-coin comparison

One bucket market resolves per day at **06:00 UTC**. Coin `#N` is the HL HIP-4
market id; live and sim agree on the coin id, so legs line up exactly.

| Day | Coin | Live net | Live VWAP (n buys) | Sim net | Sim entry px | Δ (sim−live) | Root cause |
| :-- | :-- | --: | :-- | --: | --: | --: | :-- |
| 05-31 | #1290 | +6.87 | 0.9775 (11) | +5.19 | 0.983 | −1.68 | SHR-79 ladder + earlier entry (live 00:00:04 vs sim 00:01:10) |
| 06-01 | #1340 | +10.23 | 0.9670 (2) | +10.24 | 0.967 | +0.01 | **match** (both entered ~0.967 at open) |
| 06-02 | #1380 | +10.39 | 0.9665 (1) | +4.57 | 0.985 | **−5.82** | scan cadence: live caught favorite at 01:35:51@0.9665, sim's grid passed at 01:53:55@0.985 |
| 06-03 | #1460 | +4.11 | 0.9865 (1) | +4.26 | 0.986 | +0.15 | ~match |
| 06-04 | #1510 | +8.28 | 0.9730 (2) | **0.00** | — | **−8.28** | sim **no trade**: `safety_d_below_min` veto on 60 s grid; live entered 00:00:03/07 |
| 06-06 | #1610 | +5.34 | 0.9825 (1) | +1.51 | 0.995 | **−3.83** | scan cadence: live entered 04:19:06@0.9825, sim 05:00:11@0.995 (favorite emerged intraday) |
| 06-07 | #1670 | **0.00** | — (live skipped) | +10.56 | 0.966 | **+10.56** | **SHR-78**: volume gate disabled in sim → sim takes a market live's `min_recent_volume_usd` blocked |
| 06-08 | #2230 | +6.12 | 0.9800 (1) | — | — | n/a | corpus coverage gap (06-08 not in archive; `--end` exclusive) |
| **Total** | | **+51.35** | | **+36.32** | | **−15.03** | |

### Reconciliation of the −$15.03 gap

| Component | Δ |
| :-- | --: |
| Shared coins (#1290,#1340,#1380,#1460,#1610): sim 25.77 vs live 36.94 | **−11.17** |
| 06-04 #1510 live-only winner (sim safety_d veto) | **−8.28** |
| 06-08 #2230 live-only (corpus coverage) | **−6.12** |
| 06-07 #1670 sim-only winner (SHR-78 phantom) | **+10.56** |
| **Net** | **−15.01** |

(rounds to the −$15.03 headline; residual is sub-cent fill rounding.)

## Fill-by-fill on the two biggest divergences

### 06-04 #1510 — sim never enters (−$8.28)

Live, in the first seven seconds of the market:

```
00:00:03  buy 166 @ 0.9700
00:00:07  buy 141 @ 0.9766   → 307 sh, VWAP 0.9730 → settles +$8.28
```

Sim diagnostics for the same market (`question_idx=28`) — **first eligible sample
is 00:00:52**, and every in-window sample thereafter is a veto:

```
00:00:52  hold  safety_d_below_min   yes_ask 0.977  ref 64292
...        (BTC drifts 64384 → 63076 over the hour, ~2% range)
360 in-window samples: 303 no_extreme_leg, 57 safety_d_below_min, 0 enter
```

Root cause: at market open (00:00:00) time-to-expiry equals `tte_max=21600` (6 h
to the 06:00 settle), so the first ~52 s are `tte_out_of_window`; the sim's
**60 s scan grid** therefore takes its first real look at **00:00:52**. By then
BTC had begun a volatile slide, the Parkinson σ had risen, and `safety_d`
(distance-to-bucket-edge in σ units) sat below the `min_safety_d=3.0` gate for
the rest of the window. **Live evaluated at 00:00:03**, inside the brief window
where the favorite was both extreme and far enough from the edge in σ terms, and
got filled. This is a *cadence-granularity* miss (scan grid), compounded by the
`safety_d` gate's sensitivity to the σ estimate during a volatile session — not a
disagreement about the trade's merit.

### 06-02 #1380 — same trade, 18 minutes late, −$5.82

```
LIVE  01:35:51  buy 310 @ 0.9665                → settles +$10.39
SIM   01:53:55  enter    304.56 @ 0.985 (1 fill) → settles +$4.57
```

The #1380 favorite did not exist at 00:00 — it emerged ~01:35 as BTC moved. Live
caught it the moment it crossed the gate; the sim's 60 s grid + σ/`safety_d`
timing did not clear until 01:53, by which point the ask had run from 0.9665 to
0.985 (1.85 pp less edge on 310 sh ≈ the $5.82 difference). Same family as 06-06
#1610 (live 04:19@0.9825 → +$5.34 vs sim 05:00@0.995 → +$1.51).

## Does the sim reproduce hold-to-settlement? — Yes.

Live is 19 buys / 7 sells, and all 7 sells are settlement fills at 1.0000: every
position is bought and held to the 06:00 resolution, never market-exited. The sim
mirrors this exactly — all 6 sim exits are `settle` rows at price 1.000; there are
**zero mid-hold exits**. With `exit_safety_d=0.0` (mid-hold exit disabled) and
`stop_loss_pct=null`, the strategy has no early-exit path, so sim and live agree
on the *hold* behaviour. The divergence is entirely on the **entry** side
(whether, when, and at what price each leg is entered), never on the exit.

## Root-cause attribution (summary)

| Divergence | $ impact | Root cause | Ticket |
| :-- | --: | :-- | :-- |
| 06-07 phantom winner (sim-only) | +$10.56 | volume gate disabled → sim takes wider entry set than live | **SHR-78** |
| Shared-coin entry-price shortfall | −$11.17 | 60 s scan grid + single full-size touch fill vs live's fast laddered partials | **SHR-79**, scan cadence |
| 06-04 missed winner (live-only) | −$8.28 | 60 s grid's first sample hit a transient `safety_d` veto; live entered in 7 s | scan cadence + σ/`safety_d` sensitivity |
| 06-08 missed winner (live-only) | −$6.12 | corpus coverage (archive 06-08 unsealed; `--end` exclusive) | data coverage |
| σ cadence (dt) | (subsumed above) | `hl-bt run` *does* couple `reference_resample_seconds`→`vol_sampling_dt_seconds=5` on this branch; σ is dt=5. The residual cadence gap is the **scanner** interval (60 s), not σ resample. | **SHR-80** (σ path already wired; scanner interval still coarse) |

### Note on SHR-80

The original SHR-80 framing ("`hl-bt run` hardcodes
`reference_resample_seconds=60`") is **already addressed for the σ path** on this
branch: `cli.py:239` sets `reference_resample_seconds =
params["vol_sampling_dt_seconds"]` (=5 here) and the HL source resamples the
reference to 5 s bars accordingly. The remaining cadence fidelity gap is the
**scanner evaluation interval**, which defaults to 60 s
(`HftbtRunConfig.scanner_interval_seconds=60`) and is what makes the sim sample
the market on a 60 s grid while the live engine evaluates sub-second. That is the
lever behind divergences #1 and #3. It can be probed today with
`--scanner-interval-seconds 5` (no code change).

## Reproduction

```bash
# corpus (≈3 GB; same as `make pull-data`)
aws s3 sync s3://hl-recorder-archive-819175935435/venue=hyperliquid/ \
  data/venue=hyperliquid/ --exclude '*' --include '*/hour=all/*' --size-only

# sim
export HLBT_HL_DATA_ROOT="$PWD/data" LOGURU_LEVEL=WARNING
uv run hl-bt run --strategy v1_late_resolution --data-source hl_hip4 --kind bucket \
  --config config/backtest/v1_hl_bucket_since0531.json \
  --out-dir data/sim/runs/v1_bucket_since0531 \
  --start 2026-05-31 --end 2026-06-08 --ref-source hl_perp --fee-taker 0.0 --slippage-bps 0

# live fills (read-only, on the engine box via SSM — see tools/dump_v1_bucket_fills.py)
```
