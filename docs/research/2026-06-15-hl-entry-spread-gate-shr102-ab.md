# SHR-102 — `entry_spread_gate` A/B on recorded HL data

**Date:** 2026-06-15 · **Scope:** ANALYSIS only (backtest-only; NO deploy, NO live-config change).
**Status:** **PARTIAL** — see *Coverage & compute wall* below. The full 40-day raw A/B did not
finish in the time-box; the verdict rests on the one completed binary A/B day, the mechanism, and
the raw-fidelity sibling result on the doom-loop day (SHR-102 exit-hold doc, `2026-06-12`).

**Lever under test:** `ThetaHarvesterParams.entry_spread_gate` (flag-gated, default `False`:
`hlanalysis/strategy/theta_params.py:82`; wired at `hlanalysis/strategy/theta_harvester.py:409`).
**Harness:** `tools/_entry_spread_gate_ab.py` — flips **only** `entry_spread_gate` on the deployed
v31 cells; raw `mark` ticks, event scan (0.2/2.0s), order latency 50ms, fee 0 (HL HIP-4 is
fee-free), slippage 0 — i.e. the live-faithful flags of the sibling fullsweep.

---

## What the boolean actually does (code-verified)

When `entry_spread_gate=True`, **after** the favorite leg is chosen (so it gates both fresh entries
*and* top-ups on that leg), the strategy computes from the **real** book:

```
live_half_spread = (ask − bid) / 2
mid              = (bid + ask) / 2
fair_edge        = chosen_p − mid − fee_entry
edge_budget      = fair_edge − edge_buffer
HOLD (skip, reason="entry_spread_too_wide")  iff  live_half_spread > edge_budget
```

Key properties:
- It is **fair/mid-referenced**, not limit-referenced — unlike the engine's `max_slippage_pct`
  clamp (which is limit-referenced and a no-op against a wide book; see
  `hl_shr98_verdict_2026_06_10`). This is the correct reference frame for the bucket doom-loop.
- **There is no separate scalar threshold knob.** The "threshold" is *dynamic*: it is the chosen
  leg's own fair edge minus the already-configured `edge_buffer` (0.02 on both v31 cells). So the
  gate auto-scales with each market; it does not need — and does not expose — a static
  `half_spread_max`. (Answers task item 5: a configurable threshold does **not** exist and is **not
  required**; the boolean is self-calibrating off `edge_buffer` + fair edge.)

## Why it *should* be ~no-op on the deployed config

The bucket doom-loop (buy@ask on a persistently-wide book → `exit_safety_d` IOC-dumps @bid →
re-enter@ask → repeat) is **already** neutralized in the deployed v31 bucket config by the
**exit-side** lever `exit_spread_hold: 0.04` (`config/strategy.yaml` `theta_overrides.priceBucket`,
shipped per `2026-06-12-hl-bucket-retune-exit-spread-hold.md`). With the position **held to
settlement** on wide books, the re-entry leg never fires — so an entry gate has nothing left to
suppress. The sibling doc measured exactly this on the churn day (06-07, raw fidelity, their
`tools/_bucket_retune_sweep.py`):

| config (06-07) | fills | PnL |
| :-- | --: | --: |
| base (esd=1.0) | 49 | −185.39 |
| **exit_spread_hold=0.04** | 23 | **+13.44** |
| **entry_spread_gate=true** | **49** | **−185.39  (bit-identical to base — INERT)** |
| entry_spread_gate + hold=0.04 | 23 | +13.44 (gate adds **nothing** over the hold) |

→ On the single most adversarial recorded bucket day, the **entry gate alone changes nothing**, and
it adds nothing on top of the deployed exit hold. The entry was never the problem; the *exit* round-trip was.

---

## Results (this study)

### Binary control — COMPLETED (1 day)
v31 priceBinary cell, 2026-06-14 (n=1 question, the only priceBinary that settled that day):

| arm | fills | PnL | hit | maxDD |
| :-- | --: | --: | --: | --: |
| base (`entry_spread_gate=false`) | 45 | $23.40 | 100% | $0.00 |
| gate (`entry_spread_gate=true`)  | 45 | $23.40 | 100% | $0.00 |
| **Δ** | **0** | **$0.00** | — | — |

**Bit-identical.** As expected: on a tight, fee-free binary book the live half-spread never exceeds
the fair-edge budget, so the gate never fires. Confirms the gate **does not harm tight markets** (it
is a pure no-op there).

### Bucket A/B — NOT COMPLETED by this harness in the time-box
The deployed-config bucket A/B (`bk_base` vs `bk_gate`, both with `exit_spread_hold=0.04`) and the
isolation A/B (`bk_nohold_base` vs `bk_nohold_gate`, exit hold zeroed) were defined and launched but
did not finish — see compute wall below. The **direct, raw-fidelity** answer for the bucket cell is
the sibling-doc 06-07 table above: **entry_spread_gate is inert.**

---

## Coverage & compute wall (honest accounting)

- **Corpus available:** 40 settlement days, 2026-05-07 .. 2026-06-15 (38 priceBucket + priceBinary
  question symbols).
- **Completed A/B days:** **1** (binary 06-14). **Underpowered** (n<15) — no split-half stability
  can be computed from this study's own runs.
- **Why it stalled:** the live-faithful arm uses event-scan (0.2s floor) over 24h HIP-4 markets. The
  per-question **sim loop** — not the data build (≈8s/question) — is the wall: an active book yields
  ~hundreds-of-thousands of strategy evaluations per question, so a single doom-loop bucket day's
  arm takes ~15–60 min, and `--no-cache` raw reference rebuilds add ~10–20 min/cell. 40 days × 6
  arms at that cost exceeded the time-box. (The sibling `*_fullsweep.py` paid this same cost; it is
  inherent to event-scan fidelity, not a harness bug — the harness was verified to flip only
  `entry_spread_gate` with bucket params matching the deployed C3 config.)
- The harness `tools/_entry_spread_gate_ab.py` runs all arms of a day **in one process**, reusing
  the per-question bundle memo so the DuckDB build is paid once per (day,kind); it is ready to
  complete the full corpus given a longer/over-night budget.

**Split-half sign stability:** N/A — **UNDERPOWERED** (1 completed A/B day). The one completed day
plus the raw 06-07 sibling point both show a **zero** delta; there is no sign to flip.

---

## Recommendation: **KEEP OFF**

`entry_spread_gate` is **inert on the deployed v31 config** and a **pure no-op on tight books**:
1. **Bucket (the only place it could bite):** the doom-loop is already fixed on the *exit* side
   (`exit_spread_hold=0.04`, live). The raw-fidelity 06-07 A/B shows the entry gate changes nothing
   standalone (49 fills/−$185.39 = base) and adds nothing on top of the exit hold. The dominant loss
   was the wide-book *re-entry round-trip*, which the exit hold already prevents.
2. **Binary (control):** bit-identical (Δ$0.00 / Δ0 fills, 06-14) — confirms no harm to tight markets.
3. **No threshold work needed:** the boolean is self-calibrating (fair edge − `edge_buffer`); there is
   no static threshold to tune (task item 5).

**Do not flip it on.** It is redundant with the deployed `exit_spread_hold` and carries a (small)
risk of skipping the *tradeable early-window* entries on bucket markets that are liquid before they
widen near settlement — i.e. a potential downside with no measured upside. Leave it `False`.

**If revisited:** complete the full-corpus bucket A/B over-night (the harness is ready); the bar to
overturn KEEP-OFF is a *positive* bucket PnL delta with split-half sign stability over n≥15 days —
which the mechanism predicts will not appear while `exit_spread_hold` is live.

## Reproduce
```bash
export HLBT_HL_DATA_ROOT=/abs/path/to/data            # main checkout's data/
uv run python tools/_entry_spread_gate_ab.py          # full corpus (long; event-scan wall)
uv run python tools/_entry_spread_gate_ab.py 5        # last 5 settlement days (smoke)
```
Base params are dumped from the live v31 slot via `backtest_params_from_slot` (priceBucket /
default), so the A/B is bit-identical to the deployed cell except for `entry_spread_gate`.
