# HL all-days live-vs-sim fidelity, SHR-98 veto ACTIVE — both strategies, both kinds

**Date:** 2026-06-10
**Branch:** `agents/researcher-all-days-hl-live-vs-sim-with-shr-98-con-Ab2IQg` (off SHR-98 / PR #15)
**Window:** settlement dates **2026-06-06, 06-07, 06-08, 06-09** only. HL live config
has been stable since ~06-06 (binary `vlb900/dt5` deployed 06-05, bucket C3 rollback
06-06), so earlier days would confound today's live slot config against drifted live
fills. **The recorded corpus ends 06-09** — 06-10 is not recorded yet, so it is out of
scope.
**Scope:** ANALYSIS ONLY — no strategy/engine/config change, read-only SSM.

For the per-market method and live-mirror details this extends, see
[2026-06-09-hl-0608-live-vs-sim-both-strategies.md](2026-06-09-hl-0608-live-vs-sim-both-strategies.md).

---

## TL;DR

1. **The SHR-98 fleeting-fill veto IS active** in these runs (proven by an A/B on the
   exact market the task flagged — see below).
2. **It does what it says** — it removes fleeting marketable **entry** spikes (the
   canonical 06-08 v31 `#2230` 2×100 @0.90 phantom fills) — and **does not regress v1**
   (all 8 v1 cells are bit-identical veto-ON vs veto-OFF).
3. **But it barely moves overall fidelity.** Σ|sim−live| across the 16 cells is **$946
   (veto-ON) vs $960 (veto-OFF)** — a ~1.5% improvement. **12 of 16 cells are
   bit-identical ON/OFF.**
4. **The dominant residual is a different failure mode the veto does not address:**
   the sim's **stop-loss / exit fills against transient recorded-book dislocations**
   (verified best-bid crashes to 0.52 / 0.80 that revert up within ~0.6 s) that the
   live engine rode through. This, **amplified by per-run inventory isolation** (the
   sim has no shared cross-market inventory cap, so it over-enters), drives the big
   bucket divergences (−$138, −$229, −$383). All **unchanged by the veto.**
5. **v1 (buy-and-hold) tracks live within a few dollars on clean days** (06-08:
   sim +6.68 vs live +9.20). **v31 (theta churn) diverges materially on every bucket
   and on 06-09 binary.**

**Verdict:** SHR-98 is correct and safe (no v1 regression, kills the phantom entry
spikes it targets) but is a *small* lever on HL fidelity. The fidelity program's
open frontier for HL theta is **SHR-89** (execution: stop/exit fills into transient
book dislocations + latency) and **SHR-91** (shared cross-market inventory cap),
not the fleeting-entry veto.

---

## SHR-98 veto confirmation (the #2230 0-fill check)

The task's gate: the 06-08 v31 `#2230` bucket must no longer show the fleeting
`2×100 @0.90 / +$20` fills. A/B on that exact run (everything else identical):

| | fills on #2230 | leg PnL |
| :--- | :--- | ---: |
| **veto OFF** (`--ioc-fleeting-persistence-seconds 0`) | **2×100 @0.90** + 315 @0.97 → settle | **+$29.46** |
| **veto ON** (SHR-98 default 2.0 s) | 0.90 fills **vetoed**; 400+115 @0.97 → settle | **+$15.46** |

The fleeting **0.90** spike fills (the ~$20 the task named: 200 sh × (1.00−0.90)) are
removed; a *persistent* 0.97 near-settle entry survives (correctly — not fleeting).
The code is loaded and firing (`hftbt_runner.py:247/859/932`,
`ioc_fleeting_persistence_seconds=2.0`, `_is_fleeting_ask`/`_is_fleeting_bid` on both
entry and exit paths). The "stop and debug" trigger (0.90 fills still present) does
**not** apply.

---

## Method

- **Sim (live-faithful, SHR-99 single-config-source):**
  `hl-bt run --slot {v1,v31} --slot-class {binary,bucket} --kind {binary,bucket}
  --data-source hl_hip4 --ref-source hl_perp --ref-event mark --reference-ticks raw
  --scan-mode event --fee-taker 0.0 --slippage-bps 0`. `--slot` sources the EXACT
  live `config/strategy.yaml` decision config; `--slot-class`+`--kind` isolate each
  leg class with its live per-class params. **Single-process** (no `--workers`) per
  **SHR-100** — spawned workers would load *main* (no SHR-98) and drop config.
  `HLBT_HL_DATA_ROOT` pointed at the recorded corpus.
- **Veto A/B:** the same matrix re-run with `--ioc-fleeting-persistence-seconds 0`
  (`off_*` run dirs) to isolate the veto's effect.
- **Per-day window:** market settling date D 06:00 UTC → compare window
  **D-1 07:00 → D 07:00 UTC** (one daily market per window; next-day listing lag
  keeps them disjoint).
- **Live fills:** read-only venue `user_fills` for slots `v1`+`v31` via SSM
  (SSH blocked), filtered to the window legs. HL HIP-4 fees observed = **$0**.
  Settlement is a fill with `closedPnl` on HL, so window PnL = Σ(closedPnl − fee)
  **includes settlement** (per memory: trust venue closedPnl for HL).
- **Leg map** (binary = lowest #, confirmed by both slots' `coin_klass_map`):

  | settle | binary | buckets |
  | :-- | :-- | :-- |
  | 06-06 | #1590/#1591 | #1610/20/30 (+complements) |
  | 06-07 | #1640/#1641 | #1660/70/80 |
  | 06-08 | #2200/#2201 | #2220/30/40 |
  | 06-09 | #2250/#2251 | #2270/80/90 |

---

## Per-day table (live | sim veto-ON | sim veto-OFF)

`fll`=fills (excl. settlement leg), `ntl`=buy notional $, `pnl`=$ (incl. settlement).
`ΔON`=sim_ON−live. `vetoΔ`=sim_ON−sim_OFF (0 ⇒ veto inert on this cell).

| day | slot | kind | Lfll | Lntl | **Lpnl** | Sfll | Sntl | **ON** | **OFF** | Δ_ON | vetoΔ | dominant cause |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- |
| 0606 | v1 | binary | 0 | 0 | **+0.00** | 1 | 300 | +2.42 | +2.42 | +2.42 | 0 | phantom entry (live v1 didn't trade) |
| 0606 | v1 | bucket | 1 | 300 | **+5.34** | 15 | 558 | **−138.40** | −138.40 | −143.74 | 0 | **stop-out into 0.52 bid crash** (SHR-89) |
| 0606 | v31 | binary | 39 | 3300 | **−27.83** | 28 | 1475 | −55.52 | −54.28 | −27.69 | −1.24 | churn + inventory isolation (SHR-89/91) |
| 0606 | v31 | bucket | 11 | 998 | **−36.62** | 3 | 518 | **−229.66** | −229.66 | −193.04 | 0 | **stop-out into 0.52 bid crash** (SHR-89) |
| 0607 | v1 | binary | 0 | 0 | **+0.00** | 2 | 300 | +10.24 | +10.24 | +10.24 | 0 | phantom entry (live v1 didn't trade) |
| 0607 | v1 | bucket | 0 | 0 | **+0.00** | 2 | 300 | −95.92 | −95.92 | −95.92 | 0 | phantom entry + stop-out (SHR-91/89) |
| 0607 | v31 | binary | 21 | 1493 | **+76.90** | 4 | 474 | +81.73 | +81.73 | +4.83 | 0 | near-parity |
| 0607 | v31 | bucket | 48 | 1465 | **−133.30** | 30 | 2482 | **−383.44** | −383.44 | −250.14 | 0 | **over-entry + sell into 0.80** (SHR-91/89) |
| 0608 | v1 | binary | 2 | 300 | **+3.08** | 1 | 300 | +2.11 | +2.11 | −0.96 | 0 | parity (single-fill tick, SHR-79) |
| 0608 | v1 | bucket | 1 | 300 | **+6.12** | 1 | 300 | +4.57 | +4.57 | −1.55 | 0 | parity (single-fill tick, SHR-79) |
| 0608 | v31 | binary | 23 | 2436 | **+45.26** | 4 | 500 | +75.74 | +75.74 | +30.48 | 0 | under-cycles notional; over-prices win |
| 0608 | v31 | bucket | 0 | 0 | **+0.00** | 2 | 500 | **+15.46** | +29.46 | +15.46 | **+15.46→0** ✔ | **veto kills phantom 0.90 entry** (residual = SHR-91) |
| 0609 | v1 | binary | 2 | 299 | **+11.02** | 2 | 300 | +7.06 | +7.06 | −3.96 | 0 | single-fill tick (SHR-79) |
| 0609 | v1 | bucket | 2 | 280 | **+21.82** | 3 | 600 | −8.41 | −8.41 | −30.23 | 0 | over-entry + stop-out (SHR-91/89) |
| 0609 | v31 | binary | 17 | 1486 | **+5.32** | 11 | 1496 | **−106.00** | −103.23 | −111.32 | −2.77 | **stop/exit dislocations** (SHR-89) |
| 0609 | v31 | bucket | 8 | 585 | **+6.63** | 15 | 923 | −17.45 | +34.86 | −24.08 | −52.31 | over-entry; veto here *hurt* (sign flip) |

### Per (day, slot) totals

| day | slot | live | sim ON | Δ ON | sim OFF | Δ OFF |
| :-- | :-- | --: | --: | --: | --: | --: |
| 0606 | v1 | +5.34 | −135.98 | −141.32 | −135.98 | −141.32 |
| 0606 | v31 | −64.45 | −285.18 | −220.74 | −283.94 | −219.49 |
| 0607 | v1 | +0.00 | −85.69 | −85.69 | −85.69 | −85.69 |
| 0607 | v31 | −56.40 | −301.71 | −245.31 | −301.71 | −245.31 |
| 0608 | v1 | +9.20 | +6.68 | **−2.52** | +6.68 | −2.52 |
| 0608 | v31 | +45.26 | +91.20 | +45.94 | +105.20 | +59.94 |
| 0609 | v1 | +32.83 | −1.35 | −34.18 | −1.35 | −34.18 |
| 0609 | v31 | +11.95 | −123.45 | −135.40 | −68.37 | −80.32 |

**Grand:** LIVE **−16.25** · SIM_ON **−835.46** · SIM_OFF **−765.14**.
**Σ|sim−live|: ON 946.07 vs OFF 960.21** (lower = closer). Cells closer to live:
ON=2, OFF=2, **tie=12**.

---

## Does the SHR-98 veto bring sim closer to live?

**Yes, but marginally, and only where it was designed to.**

- **It removes the phantom fleeting *entry* fills it targets.** The one clean win is
  **06-08 v31 #2230**: veto-OFF books a +$29.46 (incl. the 2×100 @0.90 phantom);
  veto-ON books +$15.46 — closer to live's **$0** (live v31 never traded that bucket).
- **No v1 regression.** All 8 v1 cells are bit-identical ON/OFF — the veto never
  touched a v1 fill. ✔
- **Net effect is ~1.5%** (Σ|sim−live| 946 vs 960) because **12/16 cells don't change.**
  On 06-09 v31 bucket the veto even *hurt* (flipped +34.86 → −17.45, wrong sign vs
  live +6.63) — a small-sample artifact of vetoing one exit that happened to be on the
  profitable side.

The veto is **correct and safe**, but it is a precision tool for one specific
artifact, not the lever that closes HL theta fidelity.

---

## Why the big divergences remain (root cause, verified)

**1. Stop-loss / exit fills into transient recorded-book dislocations — the −$100s.**
The largest residuals are not phantom *entries* but phantom *exits*. Verified example,
**06-06 v1 bucket #1610** (sim −$138.40, live +$5.34, same winning leg):
- Sim buys 305 @0.982, then **stop-loss sells all 305 @0.523**, re-buys @0.993, settles
  @1.00 → −$138. Live simply **held #1610 to settle → +$5.34**.
- The recorded `#1610` book best-bid genuinely **crashed to 0.52332** (book is
  best-first; not a worst-first bug) and **recovered to 0.99 within ~1.4 s**. The sim,
  replaying ticks, catches the V-shaped crash; live (discrete scan + network latency)
  rode through it.
- **The crash is in the OUTCOME leg's own order book, not the reference.** At the
  crash instant the `#1610` book was `bid 0.52332 / ask 0.97965` — a **one-sided bid
  vacuum** (the ask, the fair-value ceiling, held at ~0.98; the spread blew from ~0.07
  to ~0.46 wide, then re-tightened to 0.983/0.985). The HL perp BTC **reference** mark
  at the same instant ticked **UP** (~59,820 → 60,866, through a ~4.4 s feed gap), i.e.
  the *opposite* direction — and `#1610` is the YES/favorite (settles 1.0), so the
  reference move would push fair value *toward 1.0*, not crash it. The bid vacuum is
  therefore **decoupled from (and contrary to) the reference** — a pure outcome-book
  liquidity flicker, not a repricing. v1's price stop (on the leg bid, ≈0.83) punched
  through and the IOC sold 305 sh into the 0.523 bid.
- **Why live didn't stop out — VERIFIED from engine logs: the live engine was DOWN
  (OOM crash-loop) during the exact 24 s window.** Live held the *identical* position
  (305 sh @0.98248, entered ~04:19 UTC, within 26 s of the sim's entry) — so this is
  not an entry-timing difference. Engine journald around the crash:
  `05:54:35 kernel oom-killer → Killed process (duckdb, rss 530MB) under
  hl-recorder-sync` → `05:54:37 hl-engine start-pre timed out, Terminating` →
  `05:54:43 Scheduled restart, restart counter at 23` → **`05:54:49 #1610 bid vacuum
  (engine not running)`** → `05:55:01 Started` → `05:55:08 reconcile_drift`. The engine
  was OOM-crash-looping (the known hourly-OOM pattern, fixed+deployed *later* on 06-06:
  +1 G swap + duckdb `memory_limit=256 MB`), so no scanner evaluated the stop; by
  recovery the book was back to ~0.98. **This confounds the −$138 cell:** the sim
  assumes a continuously-running engine, but live physically could not act. Had the
  engine been up, its known *"IOC stop walks down book"* behavior might have stopped
  out too — so the sim's stop-out is not necessarily *wrong*, just unobservable in this
  downtime-confounded live sample. → reliability gap (engine OOM availability), not only
  SHR-89.
- **The SHR-98 veto guards the wrong direction here.** `_is_fleeting_bid` vetoes a sell
  only if the bid retreats *further below* the fill price within the window. A bid that
  is *anomalously low and reverts up* is not caught — so stop-outs into transient
  crashes survive the veto (hence bit-identical ON/OFF). Same mechanism drives **06-07
  v31 bucket** (sells a large block @0.80 that reverts) and **06-06 v31 bucket** (−$229).
  → **SHR-89** (execution fidelity / latency / reject-refire). *Possible SHR-98
  extension: veto exits whose fill price is anomalously low vs the persistent bid
  (mean-reversion guard), not only continued-deterioration.*

**2. No shared cross-market inventory cap — sim over-enters.** The default `hl-bt run`
path simulates each (slot, kind) in **isolation** — no shared inventory ledger across
the binary + the three bucket legs, and the SHR-85 caps are not applied in the run
path (confirmed in the 06-09 doc). Live runs **one** ledger per slot
(`max_total_inventory ~1100`, `max_concurrent 5`). So the sim piles on notional live
had no room for (e.g. 06-07 v31 bucket: sim $2,482 vs live $1,465; 06-09 v31 bucket
sim $923 vs live $585), and bigger inventory means a stop-out walks deeper into the
book. → **SHR-91** (cross-market inventory/concurrency).

**3. Phantom v1 entries on 06-06 / 06-07.** Live **v1 did not trade** the 06-06 binary
or the entire 06-07 market, yet sim v1 entered (06-07 v1: +10.24 binary / −95.92
bucket, both phantom). Two non-exclusive causes: live v1's shared inventory was
consumed elsewhere (SHR-91), or live's real-time gates/IOC rejects differed from the
sim's gate evaluation on the recorded book (input/timing skew, possibly
**SHR-101** cross-question state). Unconfirmable without live decision diagnostics;
flagged.

**4. v31 binary churn under-capture (06-08 binary).** Live cycled $2,436 over 23 fills
(+45.26); sim cycled $500 over 4 fills (+75.74) — sim under-trades the count but
over-prices the win. → **SHR-89** (the same churn/cadence frontier the 06-09 doc flagged).

**5. v1 single-fill tick (06-08/06-09 binary, ±$1–4).** The near-parity v1 cells differ
only by which tick the lone partial-fill ladder snaps to. → **SHR-79**, bounded.

---

## Materially divergent cells (>~$5 or wrong sign) — residual map

| cell | Δ_ON | sign? | ticket |
| :-- | --: | :-- | :-- |
| 0606 v1 bucket | −143.74 | wrong sign | SHR-89 (stop-out dislocation) |
| 0606 v31 bucket | −193.04 | worse | SHR-89 + SHR-91 |
| 0606 v31 binary | −27.69 | worse | SHR-89/91 (churn+inventory) |
| 0607 v1 bucket | −95.92 | wrong sign | SHR-91 (phantom) + SHR-89 |
| 0607 v31 bucket | −250.14 | worse | SHR-91 (over-entry) + SHR-89 |
| 0607 v1 binary | +10.24 | wrong sign | SHR-91 (phantom entry) |
| 0608 v31 binary | +30.48 | over | SHR-89 (churn under-capture) |
| 0608 v31 bucket | +15.46 | over (residual) | SHR-91 (veto already removed entry phantom) |
| 0609 v31 binary | −111.32 | wrong sign | SHR-89 (stop/exit dislocations) |
| 0609 v1 bucket | −30.23 | wrong sign | SHR-91 + SHR-89 |
| 0609 v31 bucket | −24.08 | wrong sign | SHR-91 |

**Faithful cells (Δ < ~$3):** 06-08 v1 binary (−0.96), 06-08 v1 bucket (−1.55).
v1 on the two clean config-stable days (06-08) is the only place sim ≈ live.

---

## Caveats

- **Per-kind isolation is structural, not incidental.** Comparing live (one shared
  ledger across binary+bucket+all open markets) against sim (each kind run alone) is
  *inherently* biased toward sim over-trading. Absolute-PnL parity is not expected on
  churny theta until SHR-91 lands; the qualitative attribution above is robust.
- **Recorded-book dislocations.** The 0.52 / 0.80 transient bids are real recorded
  best-bids, but whether they were *executable* live (vs a 600 ms data flicker) is the
  open SHR-89 question. The sim treats them as executable; live's latency/scan often
  skipped them.
- **Live window PnL = Σ(closedPnl−fee)** over the window, settlement included. For v31
  (closes intraday) this is fill-driven; for v1 (holds) it includes the settlement leg.
- **Corpus ends 06-09**; 06-10 not recorded.
- **Engine OOM downtime confounds the live side (esp. 06-06).** The box was OOM
  crash-looping on the morning of 06-06 (restart counter 23; fixed+deployed later that
  day). At 05:54:37→05:55:01 the engine was DOWN — verified via journald — which is
  exactly why the 06-06 v1 #1610 stop didn't fire (no scanner running), and likely
  suppresses other 06-06/06-07 live decisions the sim makes. **The sim assumes 100%
  engine uptime; live did not have it.** Any 06-06/06-07 divergence may be partly
  availability, not strategy/execution fidelity. Treat those days' absolute Δ with
  caution; 06-08/06-09 (post-OOM-fix) are the cleaner comparison.
- **06-07 was a v31-only live day** (v1 had zero fills); 06-06/06-09 binary had no live
  v1 fills either — those Δ rows are phantom-entry comparisons, not execution gaps (and
  the missing live activity may itself be OOM downtime).

---

## Artifacts

- **Sim runs (veto-ON):** `data/sim/runs/m_{v1,v31}_{binary,bucket}_{0606,0607,0608,0609}/`
- **Sim runs (veto-OFF, A/B):** `data/sim/runs/off_{...}/`
- **Veto A/B check:** `data/sim/runs/v31_bucket_0608_vetocheck/` (ON) vs
  `data/sim/runs/v31_bucket_0608_vetoOFF/` (OFF)
- **Live fills (read-only SSM dump, committed):**
  `docs/research/2026-06-10-hl-live-fills-v1-v31-window.csv` (both slots, window legs,
  with `coin→klass` map and per-slot `DIAG` rows). `data/` is gitignored, so this
  non-reproducible artifact is checked in next to the doc; sim run dirs under
  `data/sim/runs/` are gitignored but reproducible via the Method commands above.
- **Dump script (read-only, SSM):** `tools/dump_hl_fills_all.py`
- **Comparison driver:** `tools/_compare_live_sim.py`
