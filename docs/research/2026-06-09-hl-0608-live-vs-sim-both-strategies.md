# HL 06-08 daily market — live-vs-sim, BOTH strategies (v1 + v31)

**Date:** 2026-06-09
**Market:** the one HL HIP-4 daily that settled **2026-06-08 06:00 UTC** — binary
leg `#2200` (priceBinary) + bucket legs `#2220/#2230/#2240` (priceBucket).
**Why now:** the 06-08 partition is now sealed to the recorded corpus (it was
*outside* the window in [docs/research/2026-06-08-v1-bucket-live-vs-sim.md](2026-06-08-v1-bucket-live-vs-sim.md),
which stopped at 06-07). This re-runs the comparison on the new day and extends
it from v1-bucket-only to **both live HL strategies on both tracks**.
**Scope:** ANALYSIS ONLY — no strategy/engine/config change.

---

## TL;DR

| Strategy | leg | coin | **live** | **sim** | Δ (sim−live) | dominant cause |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| **v1** | binary | #2200 | +3.08 | +2.60 | −0.48 | — (near parity) |
| **v1** | bucket | #2230 | +6.12 | +9.28 | **+3.16** | execution (single-fill tick) |
| **v1** | **total** | | **+9.20** | **+11.88** | **+2.68** | sim mildly optimistic |
| **v31** | binary | #2200 | +45.26 | +23.15 | **−22.11** | execution (churn under-captured) |
| **v31** | bucket | #2230 | **0** (not traded) | +10.00 | **+10.00** | cross-market cap not modeled |
| **v31** | **total** | | **+45.26** | **+33.15** | **−12.11** | two errors partially offset |

- **v1 ≈ live** (sim +$11.88 vs live +$9.20). Both legs held to the 06:00 settle
  and won; the only material gap is the bucket's single-shot fill landing on a
  cheaper tick than live's blended ladder.
- **v31 sim materially diverges** on this market: it *under-captures* the binary's
  intraday theta churn (−$22) and *over-trades* by phantom-entering a bucket the
  live engine had no inventory room for (+$10). The errors net to −$12.

**Reproduction (live-faithful):** `hl-bt run --strategy {v1_late_resolution,
v3_theta_harvester} --data-source hl_hip4 --kind {binary,bucket} --ref-source
hl_perp --fee-taker 0.0 --slippage-bps 0 --scanner-interval-seconds 2 --start
2026-06-06 --end 2026-06-09 --config <live-mirror>`. Live: read-only venue
`user_fills` for slots v1+v31 via SSM (SSH blocked), window 06-07 07:00→06-08
07:00 UTC. HL HIP-4 fees observed = $0, so fee model is a non-factor.

---

## What each strategy actually did on this market

**v1 (late_resolution) — pure buy-and-hold to settlement, both legs won.**

- Live `#2200` binary: 2 buys @0.9898 ($300 ntl) → settle @1.00 = **+3.08**.
- Live `#2230` bucket: 1 buy @0.98 ($300 ntl) → settle @1.00 = **+6.12**.
- Sim `#2200`: 10 fills (topup ladder), buy VWAP 0.9900, $257 ntl, held to settle
  → **+2.60**. Entered 06-08 04:04 (live 04:05) — essentially the same decision;
  marginally higher price + smaller fill ⇒ −$0.48. **This is parity.**
- Sim `#2230`: 2 buys @0.97 ($309 ntl), held to settle → **+9.28**. Sim's lone
  partial-fill ladder snapped to **0.97** vs live's blended **0.98** ⇒ +$3.16.
  This is the *same single-shot-fill sensitivity* (SHR-79) the 06-07 doc flagged:
  v1's edge is a favorite drifting to 1.0, so the entry price the one sim fill
  lands on swings the leg PnL by a few dollars in either direction.

**v31 (theta_harvester) — traded ONLY the binary live; round-tripped it intraday.**

- Live `#2200` binary: **23 fills**, $2,436 cumulative buy notional, multiple
  buy→sell cycles (0.857→0.921→0.944→0.975→…→0.993), fully closed *before*
  settlement → **+45.26**. Classic theta exit/re-entry harvesting.
- Sim `#2200` binary: **28 fills**, buy VWAP 0.9536, $1,500 cumulative buy
  notional, closed intraday (no settle row) → **+23.15**. Same sign, same shape,
  **~half the PnL** — the sim cycles less notional and catches fewer of the
  momentum swings the live engine's 0.2–2 s event-driven scan caught.
- Live `#2230` bucket: **never traded.** Sim `#2230` bucket: 1 buy @ entry near
  settle, held → **+10.00**. Sim-only.

---

## Why the divergence is there (mapped to the SHR-90 attribution buckets)

1. **Execution — v31 binary churn (−$22.11, the largest single residual).**
   Theta's PnL on a binary is the sum of many intraday round-trips, so it is
   dominated by *how faithfully the sim reproduces the exit/re-entry cadence*.
   Live's event-driven scan (`scan_min 0.2 s`) fires far more often than the
   sim's fixed 2 s tick, so live re-entered more times and cycled 1.6× the
   notional ($2,436 vs $1,500). This is exactly the **SHR-89** frontier (sim
   reject/re-fire + measured-latency model) — the residual that the fidelity
   program has *not* yet closed, and the one that matters most for theta.

2. **Execution — v1 bucket single-fill tick (+$3.16).** SHR-79: one partial-fill
   ladder vs live's smear across the book. Bounded by the tick spread; flips
   optimistic/pessimistic depending on which tick the fills snap to (here:
   optimistic, 0.97 vs 0.98).

3. **Cross-market inventory/concurrency NOT modeled — v31 bucket (+$10.00).**
   **Verified in code:** the default `hl-bt run` path simulates each question in
   an isolated `run_one_question` call (`hlanalysis/backtest/runner/hftbt_runner.py`,
   driven per-question by `parallel.py`); there is **no shared inventory ledger
   across simultaneously-open markets**. SHR-85's `halt_replay.py` caps
   (`max_total_inventory_usd` / `max_concurrent_positions`) are **not applied in
   the default run path** and, being per-run, would not share state across the
   binary and bucket sims anyway. Live v31 ran a *single* ledger
   (`max_total_inventory 1100`, `max_concurrent 5`) and was busy cycling the
   `#2200` binary, so it had no room for (and/or did not gate into) the `#2230`
   bucket. The sim, blind to that shared budget, enters both. ⇒ a phantom +$10
   the live desk could not have earned. (The specific live reason — full
   inventory vs gate veto — is unconfirmable without live decision diagnostics;
   the *code-level isolation* is confirmed regardless and is the general gap.)

4. **Input-skew ≈ 0.** v1 binary entered the same decision within seconds of live;
   no σ/gate divergence on this market (unlike #1510 in the 06-07 doc).

**Net read.** v1 is faithful (sim within ~$2–3, optimistic via bucket fill
timing). v31's sim is *not* trustworthy on a single theta market in two opposite
ways that partially cancel: it understates the binary churn and overstates by
phantom-trading a bucket. The fidelity-program levers that would close this are
**SHR-89** (churn/latency, the −$22) and a new **cross-market inventory** gap in
the run path (the +$10).

---

## Artifacts

- Sim runs: `data/sim/runs/{v1_binary_through0608, v31_binary_through0608,
  v1_bucket_livefaithful_2s_through0608, v31_bucket_through0608}/`.
- Live configs used: `config/backtest/v1_hl_bucket_live.json`,
  `config/backtest/v31_hl_bucket_live.json`; binary mirrors derived from
  `config/strategy.yaml` (v1 binary = bucket cfg w/ tte 2h, price_extreme_max
  0.99, exit_safety_d 1.0; v31 binary = base theta w/ vlb 900 / dt 5 / tte 12h,
  no bucket override).
- Live fills: read-only `user_fills` dump (both slots) via SSM, window 06-07→06-08.
