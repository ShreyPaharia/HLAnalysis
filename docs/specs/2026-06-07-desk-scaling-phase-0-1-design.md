# Desk Scaling Roadmap — Phases 0 & 1 (Trust + Harden)

**Date:** 2026-06-07
**Author:** desk-manager review
**Status:** spec for implementation
**Scope:** Phase 0 (Trust PnL & risk visibility) and Phase 1 (Reliability
hardening). Phases 2–3 (sizing, hedge) and the deferred PM-multistrike epic are
out of scope for this document and get their own specs later.

---

## 1. Context & thesis

Four strategy slots run live with real orders: `v1` / `v31` on Hyperliquid
(HL) BTC binaries+buckets, `v1_pm` / `v31_pm` on Polymarket (PM) BTC daily
Up/Down. All on a single `t4g.micro` EC2 box (recorder + engine co-located, 1 GB
swap). HL true (settlement-inclusive) PnL is positive; PM slots are at burn-in
caps and barely trading.

**Thesis:** research is ahead of execution. The binding constraint on scaling is
not alpha or markets — it is that (a) we cannot yet *trust* reported PnL and
(b) the engine can lose money through operational failure (OOM, orphaned
orders, reject storms, stop-loss walking the book) faster than the thin edge
makes it. **We do not size up or add markets until the system is measurable and
reliable.** Phases 0 → 1 are that gate.

The Urgent crash-safety tier already landed (SHR-41 blocking REST, SHR-42 silent
ingest crash, SHR-43 dead-man's-switch, SHR-47 stop-loss in-flight guard, SHR-49
daily-loss uses true PnL, SHR-53 settlement PnL persisted, SHR-65 theta config
skew). This spec covers the *next* tier.

## 2. Decisions locked (from brainstorming)

- **Altitude:** desk roadmap — phased, gated, mapped to existing SHR tickets.
- **Capital trajectory:** $1k → ~$25k. Naked binaries acceptable at this size;
  hedge (Phase 3) is foundational-but-optional and out of scope here.
- **Research:** new alpha is **frozen** — no new PM series, no new v3.x variants,
  no sports/finance ports, no HL bucket retunes. The only research in later
  phases is re-validating the *existing* edge for safe sizing.
- **Infra:** **stay on the current `t4g.micro` box.** No instance split, no RAM
  bump. Reliability is solved in software within the 1 GB envelope — this makes
  memory discipline (Phase 1) load-bearing, not optional.

## 3. Non-goals (explicitly OUT of Phases 0 & 1)

- Any change to position-size, inventory, or daily-loss **caps** (caps move only
  in Phase 2, in gated tranches).
- New markets / PM series / strategy variants.
- Moving or resizing the EC2 instance.
- Live hedge execution (Phase 3).
- Sizing algorithm changes (Phase 2).
- Backtest/tuner work *except* what Phase 1 needs for sim/live parity (SHR-66);
  tuner-infra tickets (SHR-58 deterministic qidx, SHR-71 parallel-tune OOM) are
  Phase 2 prerequisites, tracked but not done here.

---

## 4. Phase 0 — Trust PnL & risk visibility

**Goal:** a single place where net PnL and open risk per slot are *trustworthy*,
reconciled to the venues. This is the gate for ever raising a cap. Motivated by
the live finding that fill-table PnL (−$122 / −$87) inverted the
settlement-inclusive truth (+$120 / +$291).

Several enablers already landed (SHR-49, SHR-53); Phase 0 is largely
**verify-they're-live + surface + reconcile**, not new accounting from scratch.

### Workstreams

- **W0.1 — Deploy observability tier-1.** `make engine-diag` (per-slot JSON
  snapshot: positions + true PnL, alias-filtered rejects, config fingerprint,
  feed health), `make engine-events Q=<idx>`, and JSON journald logs already
  exist as debug tooling but are **not deployed/operationalized**. Deliverable:
  deployed to EC2, wired into the runbook (DEPLOYMENT.md), one-command access
  documented. Design ref: `docs/specs/2026-06-05-engine-observability-tier1-design.md`.

- **W0.2 — Lock in & surface realized true PnL.** `engine-diag` already reports
  `true_pnl = realized_fills + settlement` per slot (no venue calls, by design).
  Add a regression test that pins SHR-49 (daily-loss gate reads
  settlement-inclusive PnL, no structurally-zero fallback) and SHR-53
  (settlement persisted, single close-path owner) so they cannot silently
  regress, and relabel the diag field so it is unambiguous that it excludes
  open-position MTM (MTM is added in W0.3, which has venue access).

- **W0.3 — Daily reconciliation report.** A module run on the box via SSM
  (same pattern as `engine-diag`; env-sourced for credentials) that, per slot,
  compares engine-DB realized true PnL + open positions against the venue's
  `exec_client.clearinghouse_state()` — `VenuePosition(qty, avg_entry,
  unrealized_pnl)` + `account_value_usd` + `positions_known`. Reports
  realized + open-MTM (= Σ venue `unrealized_pnl`) = total true PnL, flags
  position drift (qty mismatch, vanished, orphan) beyond tolerance, and **alerts
  via Telegram on drift**. Skips position reconciliation when
  `positions_known` is False (PM data-api flap) rather than treating an empty
  set as truth. Builds on `verify_position_state_recipe`.

- **W0.4 — Per-slot PnL attribution in the report.** The report breaks true PnL
  into realized-fills / settlement / open-MTM per slot so attribution is visible
  every day. (The exit-regret/leakage research metric is a Phase-2 concern, not
  a reconciliation primitive, and is out of scope here.)

### Phase 0 exit gate (all must hold)

- [ ] `engine-diag` + reconciliation report deployed and accessible via one
      command, documented in the runbook.
- [ ] Per-slot true PnL (fills + settlement + open MTM) reported, not fill-only.
- [ ] Engine-reported true PnL **reconciles to venue balances within tolerance
      for 7 consecutive days, across all 4 slots.**
- [ ] Drift beyond tolerance raises an alert (verified by an injected-drift test).

---

## 5. Phase 1 — Reliability hardening (on the current box)

**Goal:** the engine cannot lose money through operational failure faster than
edge makes it, **within the 1 GB `t4g.micro` envelope**. Each item maps to an
open SHR ticket. Because we are not adding RAM, the two memory tickets (W1.1,
W1.2) and the memory-guard (W1.9) are first-class, not cleanup.

### Workstreams

- **W1.1 — Bound `MarketState._questions` & scan set (SHR-44).** `_questions` is
  insert-only; settled questions are flagged, never evicted, and `Scanner.scan`
  iterates all of them every tick. On the 1 GB box this is a confirmed OOM path
  (kill 2026-05-29) and per-tick O(N) waste. Evict settled/expired questions
  after a retention window; bound the active scan set. **Highest priority on the
  current-box constraint.**

- **W1.2 — Cap recorder re-buffer on write failure (SHR-63).** `_flush_key`
  re-buffers all rows on any write exception with only a per-key cap and no
  global cap. On a persistent failure (disk full, EBS issue) buffered rows grow
  unbounded; the recorder is `OOMScoreAdjust=-500` (protected) and the engine
  `+500` (sacrificed first), so a recorder memory blow-up **OOM-kills the live
  engine**. Add a global buffer byte cap with drop-oldest + an alert. Critical
  because we share one box.

- **W1.3 — Reject circuit-breaker — VERIFY + regression-test (SHR-45).**
  Investigation found this is **already implemented** in `router.py` (a
  per-`(question_idx, side)` consecutive-reject counter with a threshold,
  pre-place suppression, trip log, and reset-on-fill). So the work is to pin it
  with a regression test, optionally bucket rejects by error-class in the
  `order_rejected` payload so `engine-diag` shows reason, and close the ticket —
  not to rebuild it.

- **W1.4 — Book the recovered lost-ACK fill (SHR-46, narrowed).** The
  pending-row-before-`place()` write and the reconcile fill-replay already
  exist. The remaining gap: the reconcile local-ghost branch marks the order
  `filled` and replays `Fill` rows but **does not upsert the `Position`**, so an
  unbooked fill leaves the position open forever → re-exit loop. Fix is to apply
  the venue-confirmed net delta to the `Position` table in that branch (HL
  apply-changes path only; PM stays alert-only). Especially important since we
  *expect* occasional OOM on the shared box until W1.1/W1.2 fully land.

- **W1.5 — Stop/exit IOC sweep within slippage budget + in-flight guard
  (SHR-48).** Stop/exit places a single IOC at top-of-book for full size; on
  thin HIP-4/PM books it partial-fills and the 1 Hz loop re-fires at the
  then-lower bid, walking the book down (stops out eventual settlement winners).
  Escalate to sweep within a slippage budget; guard against re-firing while an
  exit is in flight. (Complements the already-landed SHR-47 entry-side guard.)

- **W1.6 — PM `price_change` delta application (SHR-62).** `parse_price_change_message`
  packs a delta (changed levels only) into a `BookSnapshotEvent` as the *entire*
  book, so consumers see a near-empty 1–2 level phantom book between full frames
  → corrupt top-of-book and depth-walk slippage. Apply deltas against the
  maintained book instead of replacing it. Prerequisite for trustworthy PM
  fills and any future PM sizing.

- **W1.7 — Sim/live MarketState parity (SHR-66).** Engine and backtest
  MarketState diverge on windowing (live by COUNT vs backtest by TIME) and
  bucketing, changing σ / p_model / safety_d — especially across feed gaps (HL
  Tokyo POP outages). Converge the two on one windowing+bucketing definition
  (lean on the shared `hlanalysis/marketdata/` module). **This is the bridge to
  Phase 2:** sizing tuned in backtest is only safe if live σ matches.

- **W1.8 — Tests for the dangerous live paths (SHR-67).** The only e2e runtime
  test feeds a finite stream and asserts ENTRY+EXIT. Add tests for: order
  rejection path, stop-loss IOC→router chain, reconcile finding venue drift,
  restart with pre-existing DB+venue state, and feed disconnect/reconnect. TDD
  these alongside W1.3–W1.5 where possible.

- **W1.9 — Latching drawdown kill-switch + memory guard.** Today StaleDataHalt
  is a non-latching alert and never sets `slot.halted`. Add a **latching**
  desk-level halt that trips on the daily-loss cap and requires operator
  clearance to resume. Add a lightweight RSS watch that **gracefully self-halts**
  (flush DB, stop placing) before the kernel OOM-killer fires — the
  current-box safety net behind W1.1/W1.2.

### Phase 1 exit gate (all must hold)

- [ ] **7+ day continuous run with zero OOM kills** of recorder or engine.
- [ ] Zero unreconciled orphan positions over the run (W1.4 verified by an
      injected lost-ACK test + the daily reconciliation report from Phase 0).
- [ ] A simulated reject storm auto-halts the offending entry (W1.3) and a
      simulated daily-loss breach latches the kill-switch (W1.9).
- [ ] Dangerous-path test suite (W1.8) green in CI.
- [ ] Live vs backtest σ for the same window agree within tolerance (W1.7).

---

## 6. Risk limits during Phases 0 & 1

Unchanged from current live config — caps do **not** move until Phase 2:

| Slot | Per-pos cap | Inventory cap | Daily loss cap |
|------|-------------|---------------|----------------|
| v1 (HL) | $300 | $1,000 | $100 |
| v31 (HL) | $500 | $1,100 | $100 |
| v1_pm | $50 | $100 | $100 |
| v31_pm | $50 | $100 | $100 |

Defensive gates (`min_bid_notional`, `stop_loss`, `stale_data_halt`) stay on
even where they don't fire in backtest, per established policy — backtests don't
replay live adversarial microstructure.

## 7. Sequencing & dependencies

1. **Phase 0 first** (W0.1 → W0.2 → W0.3 → W0.4). The reconciliation report is a
   verification tool the Phase 1 gates depend on, so it lands first.
2. **Phase 1 memory tickets next** (W1.1, W1.2, W1.9) — they protect the box we
   are choosing not to upgrade; everything else runs more safely once they land.
3. **Then the execution-correctness tickets** (W1.3, W1.4, W1.5, W1.6),
   TDD'd with W1.8.
4. **W1.7 (sim/live parity)** can run in parallel; it gates Phase 2, not Phase 1
   operation, so it must merely be *done and verified*, not soak-tested.
5. The two 7-day soak gates (Phase 0 and Phase 1) can overlap — start the
   Phase 0 reconciliation soak as soon as W0.3 deploys, and let the Phase 1
   fixes land during it; the *final* clean 7-day window must include all fixes.

## 8. Testing strategy

- TDD for all behavior changes (W1.3–W1.6, W1.9), per repo convention.
- Validate against **recorded** inputs (binance_bbo spot + recorded PM book at
  dt=5), never synthetic/stale klines.
- Reconciliation and parity claims are evidence-gated: a fix is "done" only when
  the relevant gate checkbox is backed by a command output, not an assertion.
- Full suite stays green; no regression in the existing engine/backtest tests.

## 9. Open questions

- Reconciliation tolerance threshold (W0.3): start at a conservative absolute +
  relative band, tighten after observing a week of real drift. Decide the
  initial value during implementation from the first reconciliation samples.
