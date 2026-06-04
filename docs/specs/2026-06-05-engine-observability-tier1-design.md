# Engine Observability — Tier 1 (agent-debuggable over SSM)

**Date:** 2026-06-05
**Status:** Approved (design); implementation not started
**Epic:** SHR-30 (cleanup/refactor backlog)

## Problem

When a live issue happens on the EC2 box, the only access path is `aws ssm
send-command` (SSH blocked) — one-shot commands whose stdout is captured. An
agent (or operator) currently reconstructs "what is going on" by:

- tailing the last ~200 `journalctl` lines (prose + semi-structured `bus <kind>
  <json>` lines), and
- ad-hoc `sqlite3` queries against `state.db`, and
- `ls` of flag files (`restart_blocked`, `halt`).

Three classes of failure are slow or impossible to diagnose this way, all
attested by prior incidents:

1. **Silent non-action** — gates/vetoes/rejects vanish into prose. A funder
   mismatch rejected 208 orders over 21h with no aggregate signal
   (`pm_v1_funder_mismatch_2026_06_03`).
2. **State-vs-reality drift** — fill-table PnL misled vs. true settlement PnL
   (`hl_live_eval_2026_05_31`); config-vs-runtime skew ran `exit_safety_d=0.0`
   while YAML said `1.0` (`shr65_theta_config_skew_2026_06_04`).
3. **Feed/process health** — stale-data floods, halts, OOM, restart_blocked
   surfaced only as scattered lines.

Underlying all three: **truth must be reconstructed by grepping prose across two
systems**, over a slow one-shot pipe, often by a headless agent that can't open
an interactive session.

## Goal & non-goals

**Goal:** anything needed to diagnose a live issue is pullable as **structured
data in a single SSM round-trip**, and the engine's domain events are
**persisted in a queryable store** with bounded retention.

**Non-goals (this tier):**
- No self-hosted Grafana/Prometheus/Loki on the box.
- No instance resize (`t4g.small`).
- No new alerting channels (Telegram alerts already exist).

## What already exists (do not rebuild)

- `EventBus.publish` already logs every event to journald as
  `logger.info("bus {kind} {json}")` (`engine/event_bus.py:38`). Events are
  *visible* but only in rotating journald — not queryable by SQL.
- `risk_events.py` already models the full typed event set (`RiskVeto`,
  `OrderRejected`, `Entry`, `Exit`, `StopLossTriggered`, `FeedStale/Down/
  Recovered`, `EngineHeartbeat`, `PMStrikeMismatch`, …).
- `gate_decisions.jsonl` is already written next to `state.db`
  (`runtime.py:608`) — a gate-side decision trace exists, as a flat file.
- The bus is trivially subscribable (alerts + heartbeat already subscribe via
  `bus.subscribe()`, `runtime.py:511`).
- `state.db` already has `Settlement` + `settlement_pnl_since()` and
  `realized_pnl_since()` (`state.py`), so **true PnL = fills + settlement** is
  already computable — it's just not surfaced in one place.

The gaps are therefore narrow: **(1) queryable persistence, (2) clean JSON log
format, (3) a one-shot snapshot command.**

## Design

Three components, in dependency order.

### Component 1 — JSON log sink

Flip the loguru stderr sink in `engine/main.py` from prose to structured JSON
(`logger.add(sys.stderr, serialize=True, level=...)`). Bind stable context
(`alias`, `strategy`, `question_idx`, `cloid`) via `logger.bind(...)` at the
call sites that already have it, so each line carries queryable fields rather
than embedding them in the message string.

- **Interface:** every journald line is one JSON object with at least
  `time, level, message, extra.{alias,strategy,question_idx,cloid,event}`.
- **Depends on:** nothing. Pure sink change + opportunistic `bind` at hot paths
  (order lifecycle, scan decision, risk veto).
- **Keeps:** the existing `_InterceptHandler` (SDK/adapter logs still flow in;
  they simply lack the bound fields, which is fine).
- **Why:** makes journald deterministically filterable (`jq` over a single SSM
  pull) and is the prerequisite that makes any future Loki labels meaningful.
- **Risk/mitigation:** human readability drops at the console. Mitigation: keep
  `serialize=True` only for the deployed (journald) sink; a dev run can pass a
  flag for the pretty sink. Acceptable because the box is read by tools, not
  eyes.

### Component 2 — persisted `events` table (bounded)

Add an append-only `events` table to `state.db` and a single bus subscriber that
writes each published event to it. This reuses the existing bus fanout — one
more `bus.subscribe()` alongside alerts/heartbeat.

- **Schema:** `events(id INTEGER PK, ts_ns INTEGER, alias TEXT, kind TEXT,
  question_idx INTEGER NULL, reason TEXT NULL, payload_json TEXT)`. Indexed on
  `(alias, kind, ts_ns)` and `(question_idx, ts_ns)`.
- **Writer:** a new `_events_persist_loop(sub)` task in `runtime.py`, symmetric
  to the alerts loop. Pulls from its own bus queue, calls
  `StateDAL.append_event(...)`. Never blocks publish (bus already drops on slow
  consumers).
- **Retention (load-bearing on a 1 GiB box):** cap by row count and/or age.
  On each insert (or periodically) delete rows older than N days or beyond M
  rows. The unbounded-`_questions`→OOM scar (`hl_live_eval_2026_05_31`) is the
  reason this is non-optional. Retention bound is a config value.
- **Migration:** `0006_events.sql` following the existing
  `migrations/000N_*.sql` + `schema_migrations` pattern.
- **Why:** turns "why didn't slot X trade between T1 and T2?" and "how many
  rejects, of what kind?" into one SQL query instead of prose archaeology.
- **`gate_decisions.jsonl`:** leave it as-is for now (it already serves the
  gate trace). Optionally fold gate decisions into `events` later; out of scope
  for Tier 1 to avoid churn.

### Component 3 — `engine-diag` one-shot snapshot

A new CLI subcommand (e.g. `python -m hlanalysis.engine.diag` or a
`hl-engine diag` entry) that reads `state.db` (incl. the new `events` table) and
flag files, and prints **one JSON object** to stdout. No engine introspection /
no IPC — it reads the same on-disk state the engine writes, so it's safe to run
against a live engine over SSM.

Snapshot contents (per alias slot unless noted):
- `status`: running/paper/halted/blocked (from flag files + `Session_`).
- `positions`: open positions + `true_pnl = realized(fills) + settlement`.
- `open_orders`: live orders with age.
- `feed`: last heartbeat ts, events-ingested delta, stale/down state.
- `flags`: `restart_blocked`, `halt` presence + mtime.
- `rejects`: rolling counts by `kind`/`reason` from `events` over last N hours
  (this is where the "208×" surfaces as a number + one sample).
- `last_decision`: most recent decision/terminal-reason per slot.
- `config_fingerprint`: hash + key effective params per slot (surfaces
  config-vs-runtime skew like the theta `exit_safety_d` bug).

- **Make target:** `make engine-diag` wraps the SSM `send-command` +
  `get-command-invocation` round-trip (mirrors existing `engine-status` /
  `engine-logs` targets in the `Makefile`). Optionally
  `make engine-events Q=<idx>` for a single question's event trace.
- **Depends on:** Component 2 for `rejects`/`last_decision`; degrades
  gracefully (omits those sections) if the table is empty.
- **Why:** collapses the multi-call, multi-system reconstruction into one
  round-trip — the single highest-leverage item for agent debugging.

## Data flow

```
engine ──emits──► EventBus.publish ──► (a) journald JSON line  [Component 1]
                                   └──► (b) _events_persist_loop ──► state.db.events  [Component 2]

agent/operator ──SSM──► make engine-diag ──► reads state.db + flags ──► one JSON blob  [Component 3]
                  └────► make engine-logs ──► journalctl JSON | jq            [Component 1]
```

## Testing

- **Component 1:** unit-assert a sample log call emits parseable JSON with the
  bound fields; assert `_InterceptHandler` lines still appear.
- **Component 2:** publish each `risk_events` type through a real `EventBus` +
  in-memory/temp `state.db`, assert rows land with correct `kind`/`reason`/
  `question_idx`; assert retention prunes beyond the cap; assert publish never
  blocks when the persist consumer is slow (drop path).
- **Component 3:** seed a temp `state.db` (positions, fills, settlement, events)
  and assert the JSON snapshot has the expected sections and that
  `true_pnl == realized + settlement`; assert graceful degradation with an empty
  events table and with missing flag files.
- Follow TDD (existing suite is green at 728+); no live-venue calls in tests.

## Rollout

Backtest-irrelevant (engine-only). Ship behind the normal
`make deploy-engine` path. Components are independently deployable in order
1 → 2 → 3; each is useful alone (1 improves logs immediately; 2 makes events
queryable; 3 is the payoff). No config flip required to activate 1; 2 needs the
migration; 3 is a new command.

## Considered and rejected

- **Self-hosted Grafana + Prometheus + Loki on the t4g.micro.** ~400–600 MB
  contended working set + CPU-credit burn on a 1 GiB box already running
  recorder + engine; risks OOM-contention with the trading engine. Rejected:
  adds operational risk to *reduce* debugging time — net negative. Also wrong
  tool: the agent consumes structured data via commands, not dashboard pixels.
- **Upgrade to `t4g.small` to self-host the stack.** Only option that costs
  money *and* adds ops burden *and* still competes for resources, to host a
  dashboard the agent doesn't need. ~$6–8/mo incremental is trivial, but the
  justification (data-sovereignty / no log egress) doesn't currently apply.
  Rejected unless a hard no-SaaS constraint appears.

## Planned Phase 2 (opt-in, not now)

Once Tier 1 exists and logs are structured: ship JSON logs to **Grafana Cloud
(free tier)** via a light shipper (≈0 on-box RAM) and wire the **Grafana MCP**
so interactive agents can run LogQL/PromQL as tools. Strictly a layer on top of
Tier 1 — `engine-diag` over SSM remains the always-available fallback, because
interactively-authenticated MCP servers can be **absent in headless/cron runs**.
The live-debugging story must not depend on the MCP.
