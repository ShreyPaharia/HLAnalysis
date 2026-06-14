# Strategy-based slots + unified state DB — design

**Date:** 2026-06-15
**Status:** Approved design, pre-implementation
**Scope:** Live engine state model (`hlanalysis/engine/`). No strategy/decision-logic changes.

## 1. Problem & goals

Today a live-engine "slot" is keyed by `account_alias`, and each slot is its own
venue wallet (`config/deploy.yaml`: `v1`→`HL_ACCOUNT_ADDRESS`,
`v31`→`HL_ACCOUNT_ADDRESS_V31`, each PM slot its own `funder_address`). This
forces **1 strategy = 1 wallet** and gives each slot its own `state.db`. Two
consequences we want to remove:

1. **Can't run multiple strategies on one account.** Adding a strategy means
   provisioning and funding a new wallet, even when the new strategy trades
   different markets on a wallet we already control.
2. **Per-slot DBs duplicate the engine-wide `events` table 5×.** The persist
   loop writes every bus event into *every* slot's DB (~200 MB of identical
   events per slot; ~1 GB across the box). This was a contributor to the
   2026-06-14 root-disk-full outage.

**Goals:**
- **G1.** Slot identity is the *strategy*, not the account. Multiple strategy
  entries may point at the same `account`.
- **G2.** A single unified `state.db` for the whole engine.
- **G3.** Preserve today's correctness guarantees: venue-authoritative HL
  reconcile, per-strategy daily-loss gate, decision parity (sim≡live).

**Non-goals (this iteration):**
- Same-market overlap between two strategies on one wallet. We design the data
  model so this is a later *config* change, not another migration (see §3), but
  the disjoint-markets assumption holds for now.
- Merging wallets / changing how accounts/secrets are provisioned.
- Any change to strategy evaluation logic.

## 2. Key decisions (locked)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| DB boundary | **One unified `state.db`** (Approach 1) | Delivers G2 directly; the high-frequency `events` writer goes 5×→1× writes; reads are served from `CachedStateDAL` so write-lock contention is low. |
| Risk scope | **Per-strategy daily-loss caps** | Independent risk control; computable from a strategy's own symbols in the disjoint case. |
| Market overlap | **Disjoint now, overlap-extensible** | Lean on venue truth today; composite PKs make overlap a future config step. |
| Rollout | **Staged: unify keeping 1:1, then enable sharing** | De-risks a live-engine migration; Stage 0 is behavior-preserving and fully testable. |

## 3. Data model (Part A)

### 3.1 Slot identity & config

- Introduce `strategy_id` as the slot's unique key (e.g. `v31_pm_eth_ms`).
- `account` becomes an attribute referencing a wallet in `deploy.accounts`;
  **multiple strategy entries may share one `account`**.
- Each entry in `config/strategy.yaml` declares an explicit `strategy_id` and an
  `account`. (`account_alias` is the existing field; we keep the name or alias
  it to `account` — decided in the plan, no semantic change.)
- Cloid prefix is `hla-<strategy_id>-` (already effectively per-slot).

### 3.2 Unified schema

All per-slot tables live in one DB. Every row carries `account` + `strategy_id`.
Tables split into two attribution classes:

- **Cloid-keyed (already strategy-attributable):** `open_order` (already has
  `strategy_id`), `fill`, `trade_journal`. Add `account`; add `strategy_id`
  where missing.
- **`question_idx`-keyed (must become composite):** `position`, `seen_question`,
  `pm_strike`, `settlement`. Change PK `question_idx` → **`(strategy_id,
  question_idx)`**. `coin_klass` (PK `coin`) → `(strategy_id, coin)`.
- **`events`:** already has an `alias` column → repurpose as `strategy_id`;
  written **once** per event (not fanned out to N DBs).

In the disjoint-markets case a `question_idx` is owned by exactly one strategy,
so the composite PK never collides today — but it is what makes same-market
overlap a later config change rather than a schema migration.

## 4. Mechanics, migration, touchpoints (Part B)

### 4.1 Reconcile — group by account

- Reconcile loops become **per-account (wallet)**, not per-slot. For each
  account: fetch venue `clearinghouseState`/positions once, load all
  positions/orders `WHERE account = X` from the unified DB, attribute each venue
  position by `symbol` → its owning strategy (unique in the disjoint case), and
  flag an unclaimed venue symbol as an orphan (existing alert path).
- When 1:1 (current config), per-account ≡ per-slot → **reconcile behavior is
  unchanged**. HL stays venue-authoritative; PM stays local-ledger.

### 4.2 Per-strategy risk

- `realized_pnl_since` gains a `strategy_id` filter (sums that strategy's `fill`
  + `settlement` rows).
- HL wallet-level `closedPnl` is attributed to a strategy via
  symbol→question→`strategy_id` (clean in the disjoint case). PM is already
  cloid/strategy-tagged.
- Daily-loss gate evaluates per strategy.

### 4.3 DAL & caching

- One SQLite file (WAL, generous `busy_timeout`), one SQLAlchemy engine.
- Each slot gets a thin **strategy-scoped DAL view** that auto-injects its
  `strategy_id` into every read/write, so `scanner`/`router` call sites barely
  change.
- `CachedStateDAL` caches remain **per-strategy** (scoped by `strategy_id`);
  disjoint question spaces mean no cross-strategy cache-coherence concern. Reads
  stay in memory; only writes touch the shared file (low-rate, WAL-serialized in
  the single engine process).

### 4.4 Migration & staged rollout (highest risk)

- **Stage 0 — unify, keep 1:1 (behavior-preserving):**
  - Alembic revision: add `account`/`strategy_id` columns; rebuild the
    `question_idx`-keyed tables with composite PKs (SQLite needs table-recreate
    via Alembic **batch mode**).
  - One-time **offline merge script**: read the 5 live slot DBs, write their rows
    into one unified DB tagged with each source slot's `(account, strategy_id)`.
  - Cutover: archive all 5 DBs to S3 (existing pattern) → `systemctl stop
    hl-engine` → run merge → point engine at the unified DB → start →
    reconcile-on-start validates against venue truth.
  - No config/behavior change yet.
- **Stage 1 — enable sharing (config-only):** add a new strategy entry pointing
  at an existing `account`. No migration; new rows carry the new `strategy_id`.

### 4.5 Read-side touchpoints (all currently assume per-slot DBs)

- `scripts/sync-engine-to-s3.sh`: snapshot the single DB instead of looping
  `*/state.db`; `gate_decisions.jsonl` stays a per-strategy sibling file.
- `scripts/engine_events.py` (`make engine-events`): query the one DB filtered by
  `strategy_id`/question instead of globbing `*/state.db`.
- `hlanalysis/engine/reconcile_report.py` (`make reconcile-report`): per-strategy
  rows (filter by `strategy_id`), grouped by account for venue truth.
- `hlanalysis/engine/diag.py` (`make engine-diag`): alias filter → `strategy_id`
  filter on the single DB.
- `restart_blocked` / `halt` **flag files** stay per-strategy
  (`<root>/<strategy_id>/`) — they are not in the DB; only `state_db_path`
  collapses to one shared path. `DeployConfig.state_db_path_for` and the
  per-slot flag-path helpers are updated accordingly.

## 5. Testing

- **Unit:** composite-PK round-trip for each migrated table; strategy-scoped DAL
  isolation (writes under one `strategy_id` are invisible to another's reads);
  per-strategy `realized_pnl_since` attribution.
- **Migration:** 5 fixture slot DBs → 1 unified DB; assert row counts, tags, and
  no PK collisions; idempotent re-run.
- **Reconcile:** per-account grouping with 1:1 config reproduces current
  behavior; a two-strategy-one-account fixture (disjoint symbols) attributes
  positions correctly and flags orphans.
- **Parity gate:** `tests/unit/parity/test_ci_decision_parity_gate.py` must stay
  green — this change is state/observability only, no decision logic.
- **Full suite + `make parity-gate`** before any deploy.

## 6. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Live-DB merge loses/corrupts state (open positions exist) | Archive all 5 DBs to S3 first; offline merge with engine stopped; reconcile-on-start validates vs venue; keep the 5 originals until a clean soak. |
| Single-file blast radius | Daily S3 snapshot; HL recoverable from venue truth; PM from S3. |
| Write-lock contention on one file | WAL + `busy_timeout`; reads from cache; events writer goes 5×→1×; writes are small/low-rate. Re-evaluate under load before Stage 1. |
| SQLite composite-PK change | Alembic batch mode (table recreate + copy); covered by migration tests + the existing baseline/stamp machinery. |
| Per-strategy HL PnL attribution wrong if markets overlap | Out of scope this iteration; disjoint assumption documented; composite PK + `strategy_id` columns make the overlap-safe ledger a follow-up. |

## 7. Open items for the implementation plan

- Exact Alembic revision sequencing (column adds vs PK rebuilds; one revision or
  several) and batch-mode recreate ordering.
- Whether the strategy-scoped DAL is a wrapper over a shared `StateDAL` or
  `StateDAL` gains a mandatory `strategy_id` on its methods.
- `config/strategy.yaml` field naming (`account_alias` vs `account`) and the
  `deploy.yaml` twin.
- Cutover runbook (SSM commands, the merge-script invocation, rollback to the 5
  originals).
