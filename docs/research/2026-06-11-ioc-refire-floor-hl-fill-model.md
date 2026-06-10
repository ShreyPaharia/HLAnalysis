# IOC-at-touch fill model for the HL backtest — re-fire latency floor (SHR-79 / SHR-89)

**Date:** 2026-06-11
**Branch:** `agents/ioc-at-touch-fill-model-for-hl-backtest-shr-79-shr-pVE-4X`
**Scope:** ANALYSIS + BACKTEST-ONLY. No engine/strategy/config/live change, no deploy.
Implements the fidelity fix pointed to by
[2026-06-10-hl-all-days-live-vs-sim-shr98.md](2026-06-10-hl-all-days-live-vs-sim-shr98.md).
Tickets: **SHR-79** (#23 sim fills full size at touch) + **SHR-89** (reject/re-submit +
measured-latency model).

---

## TL;DR

1. **Most of the prescribed IOC model already existed on `main`.** The depth-limited
   fill that walks the recorded book ladder (IOC remainder cancels), the
   latency-at-arrival fill, the intra-order multi-level walk, and the IOC
   marketability re-check all landed earlier (SHR-79/56/57 `e5a642a`, SHR-89
   `f4ea36d`, SHR-94 `e290c3a`). The 06-10 all-days runs already used them. That is
   why the sim's full-size dumps were **faithful to the recorded book** — the depth
   was genuinely there (verified below), not a missing depth cap.
2. **The one genuinely-missing lever is a minimum inter-order re-fire floor.** Live
   serializes one order per leg in flight; the event-driven sim re-fired at the 0.2 s
   scan cadence (5 Hz), over-churning persistently-wide bucket books. This PR adds
   `RunConfig.min_inter_order_seconds` (+ `--min-inter-order-seconds`), **measured at
   ~0.75 s** from the live fills, default `0.0` (legacy no-floor A/B arm).
3. **It is a correct but small lever.** Across the 16-cell all-days matrix
   Σ|sim−live| moves **960.21 (no floor) → 956.40 (floor 0.75 s)**. The
   no-floor arm **exactly reproduces** the documented analysis veto-OFF (960.21),
   validating the harness. The floor halves the #2230 fleeting phantom and nudges the
   churn cells, but the dominant bucket divergence is **SHR-91** (no shared inventory
   cap → over-entry) + the structural wide-book **doom-loop** (SHR-102 strategy
   defect), neither of which a fill-model floor addresses.
4. **#2230 fleeting veto (PR #15) verdict: FOLD the persistence insight, don't keep it
   standalone.** The floor halves #2230 (29.46 → 22.00) but leaves one 100@0.90
   fleeting fill; only a wall-clock persistence re-check kills it (→ 15.46). That
   persistence insight is real and complementary — it should be folded into the
   unified fill model (it fixes the SHR-94 latency-window bug), **not** shipped as a
   separate veto. This PR does **not** supersede PR #15; the two are complementary.

---

## The model as built

The faithful IOC-at-touch model is now four mechanisms; this PR adds the last one.

| # | mechanism | where | status |
| - | --------- | ----- | ------ |
| 1 | **Depth-limited fill** — IOC limit-at-touch fills only resting depth at-or-better than the limit, walking the recorded ladder; remainder cancels | `_build_asset` partial_fill_exchange + `record_fill_from_order` | pre-existing (SHR-79) |
| 2 | **Latency-at-arrival** — fill on `book(decision_ts + δ)` | `ConstantLatency`/`SampledLatency` | pre-existing (SHR-89) |
| 3 | **Intra-order level-walk is one instant** — multiple levels of one IOC fill simultaneously | hftbacktest queue model | pre-existing |
| 4 | **Cancel-remainder → re-fire, gated by a MIN inter-order latency floor** | `_inter_order_blocked` + `_route_enter`/`_route_exit` | **NEW (this PR)** |

**Re-fire floor (new).** After an order is dispatched to the venue for a leg (filled,
rejected, or fleeting-vetoed — any round-trip consumed), the leg's last-dispatch
timestamp is stamped. A subsequent enter/exit on that leg within
`min_inter_order_seconds` is suppressed (`n_refire_throttled++`); the position stays
as-is and the strategy re-evaluates on a later scan. Stop-loss is never throttled (the
stop must fire) but stamps the leg so a follow-on order respects the floor. The floor
is **bounded above** by the existing `scan_max_interval_seconds` (2.0 s idle backoff)
and the strategy re-fire is **liquidity-gated** by the existing depth-limited fill
(mechanism 1): the next clip only fills against whatever is hittable at the new limit.

**Measured floor value.** From `docs/research/2026-06-10-hl-live-fills-v1-v31-window.csv`,
grouping fills by `cloid` (distinct re-fired orders) on the same `(slot, symbol)` and
measuring the gap between consecutive distinct cloids:

```
ALL distinct-cloid gaps: n=120  MIN=0.734s  p5=0.832  p10=0.903  median=3.240
churny bucket #1670: ncloid=41  min_gap=0.734  p10=0.829  median=1.314
intra-cloid (same order) sub-fills: simultaneous (gap=0)
```

The hard floor is **~0.73 s** (round-trip + reconcile + one-order-in-flight
serialization); we bake **0.75 s** as the documented default for HL validation. Default
`0.0` (disabled) preserves bit-identical legacy behaviour and is the no-floor A/B arm.

---

## Test evidence (TDD)

New unit tests in `tests/unit/backtest/test_hftbt_runner.py` (written before the code):

- `test_run_config_min_inter_order_seconds_defaults_zero` — default 0.0 (back-compat).
- `_inter_order_blocked` boundary: disabled-when-0 / first-order / within-floor /
  at-or-after-floor (4 tests).
- `test_route_exit_throttled_within_floor_does_not_submit` — a re-fire within the
  floor never reaches the venue (no oid consumed, no fill, position unchanged,
  `n_refire_throttled==1`). *(task test c)*
- `test_route_exit_allowed_after_floor_fills_on_replenished_book` — full timeline:
  first exit fills → 0.2 s re-fire throttled → 0.8 s re-fire fills against replenished
  bid liquidity (real hftbacktest sell fill). *(task tests c + d)*
- `test_route_enter_throttled_within_floor_does_not_submit` — symmetric on entries.
- `test_route_exit_no_floor_allows_immediate_refire` — legacy A/B arm: floor=0 never
  throttles.
- `test_cli_min_inter_order_seconds_arg` — flag declared, defaults 0.0, wires into RunConfig.

The pre-existing depth-limit / latency / multi-level-walk behaviours (task tests a, b,
e, f) are already covered by `test_partial_fill_ioc_remainder_cancelled`,
`test_multi_level_walk_vwap_worse_than_touch`, the SHR-89 latency tests, etc.

**Full suite: 1539 passed.**

---

## All-days re-run (does the sim now match live?)

Single-process per **SHR-100** (spawned workers load `main` + drop config), from this
worktree, against the recorded corpus (`HLBT_HL_DATA_ROOT`). Same Method as the 06-10
doc: `--slot {v1,v31} --slot-class {binary,bucket} --kind {binary,bucket}
--data-source hl_hip4 --ref-source hl_perp --ref-event mark --reference-ticks raw
--scan-mode event --fee-taker 0.0 --slippage-bps 0`, with `--min-inter-order-seconds
0.75` (IOC) vs `0.0` (BASE). Driver: `tools/_run_ioc_matrix.sh`; compare:
`tools/_compare_ioc.py`. PnL is settlement-inclusive (`realized_pnl_at_settle`).

| arm | Σ\|sim−live\| | total sim PnL |
| --- | --: | --: |
| **BASE** (no floor) | **960.21** | −765.14 |
| **IOC** (floor 0.75 s) | **956.40** | −776.55 |
| **IOC + veto** (floor + PR #15) | **954.35** | −787.57 |
| *documented analysis* | *veto-OFF 960.21 / veto-ON 946.07* | |

**BASE = 960.21 reproduces the documented veto-OFF exactly** → the comparison is
trustworthy. The floor improves Σ by **−3.81** (small). Cells closer to live: IOC=3,
BASE=3, tie=10.

### Flagged cases — before/after

| case | live | BASE | IOC (floor) | what changed |
| ---- | --: | --: | --: | ------------ |
| **#2230** 0608 v31 bucket (fleeting phantom) | +0.00 | +29.46 | **+22.00** | floor throttles the 2nd 0.90 clip (0.545 s < 0.75 s) → 2×100 → **1×100**; closer to live |
| **#1670** 0607 v31 bucket (doom-loop) | −133.30 | −383.44 | −384.50 | barely moves; **fills 30 → 41** (toward live's 48) but the loss is structural spread-crossing, not cadence → SHR-91/SHR-102 |
| **#2200** 0608 v31 binary (churn cadence) | +45.26 | +75.74 | +75.37 | ~unchanged; sim **under**-fires (4 vs live 23) — the floor can only reduce fires, so it cannot close an under-firing gap |
| **#1610** 0606 v1 bucket (stop into vacuum) | +5.34 | −138.40 | −138.40 | **unchanged** — see below |

**#1610 is depth-backed + OOM-confounded, not a fill-model gap.** At the 0.523 crash the
recorded best bid held **953 sz at 0.52332 for 30+ s** (05:54:49 → 05:55:23, the known
06-06 OOM window). The depth-limited IOC therefore *correctly* fills 305@0.523 — the
depth was really there. Live didn't stop out because the **engine was OOM-down**
(verified in the 06-10 doc), not because of execution fidelity. No fill-model change can
or should move this cell; it is a reliability (engine-uptime) confound.

The floor slightly **hurts** two tight-binary churn cells (0606 / 0609 v31 binary,
≈ −2 to −2.4): live churns binaries fast (#1591: 39 fills, min gap 0.83 s) while the sim
already under-fires, so throttling widens the gap. Net Σ improves because the
wide-bucket gains outweigh the binary losses — but it confirms the floor is a
**targeted over-churn throttle**, not a universal fidelity win.

---

## #2230 displayed size + veto verdict (SHR-98 / PR #15)

**Measured displayed size at #2230's 0.90 ask.** The fleeting 0.90 ask appears in
**exactly two snapshots — 05:15:15.308 and 05:15:15.853 (0.545 s apart) — at 100 sz
each**, surrounded by a 0.97999 ask before and after (a ~1.1 s flicker, matching PR
#15's root-cause note). So the level was **not ≥200 thick**: each clip is already
depth-limited to 100 (faithful). The phantom is 2×100 = 200 *because the strategy
re-fired across the two snapshots*, not because of an over-fill.

**A/B on #2230 (v31 bucket 0608):**

| arm | 0.90 buys | cell PnL | vs live ($0) |
| --- | --- | --: | --: |
| BASE (no floor, no veto) | 2×100 @0.90 | +29.46 | 29.46 |
| **IOC (floor 0.75 s)** | **1×100 @0.90** | **+22.00** | 22.00 |
| IOC + veto (floor + PR #15) | 0×0.90 | +15.46 | 15.46 |

- The **floor alone halves the phantom** (throttles the 0.545 s re-fire) but cannot kill
  the *first* 100@0.90 — the floor never gates the first order on a leg.
- Only a **wall-clock persistence re-check** removes the first fleeting fill. That is the
  genuine fix in PR #15: SHR-94's marketability re-check keyed its veto window on the
  ~50 ms order latency, so on HL's burst-sampled book (next snapshot seconds away) it
  **never fired**; PR #15 re-keys it on wall-clock persistence.

**Verdict: FOLD the persistence insight into the fill model; do NOT keep a standalone
fleeting veto; this PR does NOT supersede PR #15.**

- The floor (this PR) and the wall-clock persistence re-check (PR #15) address **two
  different failure modes** — re-fire *churn* vs fleeting *entry levels* — and are
  complementary. Neither subsumes the other.
- But PR #15's value is the **wall-clock re-keying of the existing SHR-94 re-check**, not
  a new standalone "veto." The clean end state is **one** unified IOC fill model =
  depth-limit + latency-at-arrival + **wall-clock-persistence** marketability re-check +
  re-fire floor. Recommend rebasing PR #15's persistence fix onto this model and dropping
  the separate-veto framing (and re-evaluating the magnitude knob `ioc_fleeting_min_revert`,
  which contributed the 06-09 binary regressions for little gain).
- On the full matrix the standalone veto nets only **−2.05** on top of the floor
  (954.35 vs 956.40), driven entirely by #2230 and partly cancelled by binary
  regressions — too marginal to justify shipping as an independent mechanism, but the
  persistence *insight* is worth folding in.

---

## Residual gaps → ticket map

| residual | magnitude | ticket |
| -------- | --------- | ------ |
| Wide-bucket doom-loop (#1670 −383, #1610-bkt −229) — sim over-enters notional live had no room for; structural spread-crossing | dominant | **SHR-91** (shared inventory cap) + **SHR-102** (strategy spread gate) |
| #1610 stop into a depth-backed bid vacuum during engine OOM downtime | −138 (confounded) | engine OOM reliability (not a fill-model gap) |
| Tight-binary churn under-capture (sim 4 fills vs live 23) | +30 over | **SHR-89** churn/cadence — sim *under*-fires; the floor can't add fires |
| Fleeting *entry* levels (#2230 residual 100@0.90 after floor) | +6.5 | **SHR-98** persistence re-check (fold into fill model) |

**Bottom line.** The re-fire floor is the correct, measurable completion of the IOC-at-
touch fill model and removes the sim's faster-than-physically-possible re-fire churn. It
is a *small* fidelity lever on its own; the open HL-theta frontier remains **SHR-91**
(shared inventory cap) and **SHR-102** (the wide-bucket strategy doom-loop), with the
**SHR-98 wall-clock persistence re-check** folded into this same fill model for fleeting
entries.

## Artifacts

- Floor impl: `hlanalysis/backtest/runner/hftbt_runner.py` (`_inter_order_blocked`,
  `RunConfig.min_inter_order_seconds`), `result.py` (`n_refire_throttled`),
  `cli.py` (`--min-inter-order-seconds`).
- Matrix driver: `tools/_run_ioc_matrix.sh`; comparison: `tools/_compare_ioc.py`.
- Sim runs (gitignored, reproducible): `data/sim/runs/{base,ioc,iocveto}_*`.
- Live ground truth: `docs/research/2026-06-10-hl-live-fills-v1-v31-window.csv`.
