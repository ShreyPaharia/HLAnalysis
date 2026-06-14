# Strategy-based slots + unified state DB — Implementation Plan (Stage 0)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the per-slot `state.db` files into one DB whose rows are tagged
`(account, strategy_id)`, with slots keyed by `strategy_id` — while preserving
today's 1:1 strategy↔account mapping and behavior (Stage 0).

**Architecture:** Single SQLite file (WAL). Each per-slot table gains `account`
+ `strategy_id`; `question_idx`-keyed tables get composite PK
`(strategy_id, question_idx)`. Each slot runs through a thin `StrategyScopedDAL`
that auto-injects its `strategy_id`. Reconcile groups by `account`. A one-time
offline merge script folds the live per-slot DBs into the unified DB.

**Tech Stack:** Python 3.12, SQLModel/SQLAlchemy, Alembic (batch mode for SQLite
PK rebuilds), pytest, loguru.

**Reference spec:** `docs/specs/2026-06-15-strategy-slots-unified-db-design.md`

**Conventions:** TDD (red→green→commit). `uv run pytest -q`. Lint with
`uvx ruff check` + `uvx ruff format`. Conventional commits, NO `Co-Authored-By`.
Run `make parity-gate` before the final handoff.

**Decisions resolved from spec §7:**
- DAL: a **`StrategyScopedDAL` wrapper** over the shared `StateDAL`/`CachedStateDAL`
  (auto-injects `strategy_id`), so `scanner`/`router` call sites are unchanged.
- Config field: keep the existing name **`account_alias`** (semantic no-op);
  add a separate **`strategy_id`** field (defaults to the entry `name` when
  omitted, to preserve current aliases).

---

## File structure

| File | Responsibility | Change |
|------|----------------|--------|
| `hlanalysis/engine/state.py` | ORM models + `StateDAL`/`CachedStateDAL` | Add cols + composite PKs; add `strategy_id` scoping params |
| `hlanalysis/engine/migrations_alembic/versions/0006_*.py` | Schema migration | NEW — columns + PK rebuilds (batch mode) |
| `hlanalysis/engine/scoped_dal.py` | `StrategyScopedDAL` wrapper | NEW |
| `hlanalysis/engine/config.py` | `DeployConfig`, slot config | Add `strategy_id`; single `state_db_path`; per-strategy flag paths |
| `hlanalysis/engine/_slot_builder.py` | Slot construction | Share one DAL; wrap per-slot in `StrategyScopedDAL` |
| `hlanalysis/engine/reconcile.py` | Reconcile loop | Group by account; attribute by symbol→strategy |
| `hlanalysis/engine/runtime.py` | Loop wiring | Per-account reconcile; events tagged once |
| `hlanalysis/engine/reconcile_report.py`, `diag.py` | Read-side | Filter by `strategy_id` |
| `scripts/engine_events.py`, `scripts/sync-engine-to-s3.sh` | Read-side | Single DB |
| `scripts/merge_slot_dbs.py` | Offline migration | NEW |
| `tests/unit/...` | Tests | NEW/updated per task |

---

## Task 1: Schema migration — add `account`/`strategy_id`, composite PKs

**Files:**
- Create: `hlanalysis/engine/migrations_alembic/versions/0006_unified_slot_db.py`
- Test: `tests/unit/test_unified_schema_migration.py`

Per-slot tables and their target shape:
- `open_order`: add `account` (TEXT). (`strategy_id` already present.)
- `fill`: add `account`, `strategy_id` (TEXT).
- `trade_journal`: add `account`, `strategy_id` (TEXT). PK stays `cloid`.
- `position`: add `account`, `strategy_id`; PK → `(strategy_id, question_idx)`.
- `seen_question`: add `account`, `strategy_id`; PK → `(strategy_id, question_idx)`.
- `pm_strike`: add `account`, `strategy_id`; PK → `(strategy_id, question_idx)`.
- `settlement`: add `account`, `strategy_id`; PK → `(strategy_id, question_idx)`.
- `coin_klass`: add `account`, `strategy_id`; PK → `(strategy_id, coin)`.
- `events`: add `strategy_id`; backfill from existing `alias` (keep `alias` for
  back-compat reads). PK stays surrogate `id`.

SQLite cannot ALTER a PK in place — composite-PK changes use Alembic **batch
mode** (`op.batch_alter_table(... recreate="always")`), which creates a new
table, copies rows, drops the old, renames. New columns default `NULL`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_unified_schema_migration.py
from hlanalysis.engine.state import StateDAL, Position, SeenQuestion
from sqlmodel import Session as _S

def test_position_composite_pk_allows_same_qidx_across_strategies(tmp_path):
    dal = StateDAL(tmp_path / "u.db")
    dal.run_migrations()
    with _S(dal._engine) as s:
        s.add(Position(strategy_id="a", account="w1", question_idx=7, symbol="@1",
                       qty=1.0, avg_entry=0.5, realized_pnl=0.0, last_update_ts_ns=1,
                       stop_loss_price=0.0))
        s.add(Position(strategy_id="b", account="w1", question_idx=7, symbol="@1",
                       qty=2.0, avg_entry=0.5, realized_pnl=0.0, last_update_ts_ns=1,
                       stop_loss_price=0.0))
        s.commit()  # must NOT raise IntegrityError (composite PK)
    cols = {c[1] for c in dal._engine.raw_connection().execute(
        "PRAGMA table_info(position)").fetchall()}
    assert {"strategy_id", "account"} <= cols

def test_existing_rows_migrate_and_backfill(tmp_path):
    # A pre-0006 DB upgrades cleanly; events.alias copies into strategy_id.
    dal = StateDAL(tmp_path / "u.db")
    dal.run_migrations()
    dal.append_event(ts_ns=1, alias="v1", kind="entry", question_idx=1,
                     reason=None, payload_json=None)
    # (Detailed legacy-DB fixture handled in Step 3's migration logic.)
```

- [ ] **Step 2: Run test, verify it fails** — `uv run pytest tests/unit/test_unified_schema_migration.py -x` → FAIL (`Position` has no `strategy_id`/`account`, or composite PK absent).

- [ ] **Step 3: Write the migration.** Model the revision on the existing
  `versions/0002_position_closed_qty.py` and `0003_coin_klass.py` for style.
  Use `op.batch_alter_table("position", recreate="always") as b:` then
  `b.add_column(sa.Column("strategy_id", sa.String(), nullable=True))`,
  `b.add_column(sa.Column("account", sa.String(), nullable=True))`, and
  `b.create_primary_key("pk_position", ["strategy_id", "question_idx"])`. Repeat
  for `seen_question`, `pm_strike`, `settlement`, `coin_klass` (its PK is
  `(strategy_id, coin)`). For `fill`/`trade_journal`/`open_order` just
  `add_column` (no PK change). For `events`: `add_column("strategy_id")` then
  `op.execute("UPDATE events SET strategy_id = alias")`. `down_revision` =
  current head (verify with `alembic history`; it is `0005_trade_journal`).

- [ ] **Step 4: Run tests, verify pass** — `uv run pytest tests/unit/test_unified_schema_migration.py tests/unit/test_state_alembic_migrations.py -q` → PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(engine): 0006 unified slot DB schema (account/strategy_id + composite PKs)"`

## Task 2: ORM models match new schema

**Files:** Modify `hlanalysis/engine/state.py` (model classes). Test: extend `tests/unit/test_unified_schema_migration.py`.

- [ ] **Step 1: Write failing test** — assert each model accepts `account` +
  `strategy_id`, and a `(strategy_id, question_idx)` round-trips for `Position`,
  `SeenQuestion`, `PmStrike`, `Settlement`:

```python
def test_models_roundtrip_with_scope(tmp_path):
    dal = StateDAL(tmp_path / "u.db"); dal.run_migrations()
    from hlanalysis.engine.state import PmStrike, Settlement
    from sqlmodel import Session as _S, select
    with _S(dal._engine) as s:
        s.add(PmStrike(strategy_id="v1_pm", account="pm1", question_idx=3, strike=100.0))
        s.commit()
        got = s.exec(select(PmStrike).where(PmStrike.strategy_id=="v1_pm",
                                            PmStrike.question_idx==3)).one()
        assert got.strike == 100.0
```

- [ ] **Step 2: Run, verify fail** — `uv run pytest tests/unit/test_unified_schema_migration.py::test_models_roundtrip_with_scope -x` → FAIL.

- [ ] **Step 3: Update models.** On `Position`, `SeenQuestion`, `PmStrike`,
  `Settlement`: make `question_idx` a non-PK field, add
  `strategy_id: str = Field(primary_key=True)` and `account: str`. On
  `CoinKlass`: `strategy_id` PK + keep `coin` PK (composite). On `Fill`,
  `TradeJournalRow`: add `strategy_id: str` + `account: str` (non-PK). On
  `OpenOrder`: add `account: str` (already has `strategy_id`). On `Event`: add
  `strategy_id: str | None`. Keep all existing fields/defaults.

- [ ] **Step 4: Run** — `uv run pytest tests/unit/test_unified_schema_migration.py tests/unit/test_state*.py tests/unit/test_trade_journal*.py -q` → PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(engine): ORM models carry account/strategy_id + composite PKs"`

## Task 3: `StrategyScopedDAL` wrapper

**Files:** Create `hlanalysis/engine/scoped_dal.py`. Test: `tests/unit/test_scoped_dal.py`.

The wrapper holds a shared `StateDAL`/`CachedStateDAL` + a `strategy_id` +
`account`, and exposes the same method surface the engine uses, auto-injecting
the scope. Reads filter by `strategy_id`; writes stamp `strategy_id`/`account`.
The underlying `StateDAL` methods gain optional `strategy_id`/`account` kwargs
(default `None` = no filter, preserving existing single-DB tests).

- [ ] **Step 1: Write failing test** — isolation between two scopes over one DB:

```python
# tests/unit/test_scoped_dal.py
from hlanalysis.engine.state import CachedStateDAL
from hlanalysis.engine.scoped_dal import StrategyScopedDAL

def test_scopes_isolated_over_shared_db(tmp_path):
    base = CachedStateDAL(tmp_path / "u.db"); base.run_migrations()
    a = StrategyScopedDAL(base, strategy_id="A", account="w1")
    b = StrategyScopedDAL(base, strategy_id="B", account="w1")
    a.upsert_position_fields(question_idx=5, symbol="@1", qty=1.0, avg_entry=0.5)
    assert a.get_position(5) is not None
    assert b.get_position(5) is None        # B cannot see A's row
```

- [ ] **Step 2: Run, verify fail** — `uv run pytest tests/unit/test_scoped_dal.py -x` → FAIL (module missing).

- [ ] **Step 3: Implement.** `StrategyScopedDAL` wraps the base DAL. For each
  method the engine calls (`get_position`, `all_positions`, `upsert_position`,
  `delete_position`, `live_orders`, `upsert_order`, `update_order_status`,
  `append_fill`, `get_seen_question`/`mark_seen`, `get_pm_strike`/`set_pm_strike`,
  `record_settlement`, `realized_pnl_since`, journal methods, …) it forwards to
  the base method passing `strategy_id=self.strategy_id, account=self.account`.
  Update `StateDAL`/`CachedStateDAL` methods to accept and apply those kwargs
  (filter on read, stamp on write). `CachedStateDAL` cache dicts are namespaced
  by `strategy_id` (e.g. `self._pos_cache[strategy_id][qidx]`).

- [ ] **Step 4: Run** — `uv run pytest tests/unit/test_scoped_dal.py tests/unit/test_state*.py -q` → PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(engine): StrategyScopedDAL over shared state DB"`

## Task 4: Wire one shared DB + scoped DAL per slot

**Files:** Modify `hlanalysis/engine/config.py` (`DeployConfig`), `hlanalysis/engine/_slot_builder.py`. Test: `tests/unit/test_slot_builder_unified.py` (or extend existing slot-builder tests).

- [ ] **Step 1: Write failing test** — two slots built from one deploy config
  share one DB file but get isolated scoped DALs:

```python
def test_two_slots_share_one_db(tmp_path, monkeypatch):
    # build_slot for two strategy configs with distinct strategy_id/account_alias
    # assert both resolve to the SAME state_db_path and writes stay isolated.
    ...
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement.** `DeployConfig.state_db_path` is now a single shared
  path (no per-alias subdir for the DB). Add `state_db_path()` returning that one
  path; keep `flag_path_for(strategy_id)` → `<root>/<strategy_id>/` for
  `restart_blocked`/`halt`/`gate_decisions.jsonl`/`engine_heartbeat`. Each slot
  config gets `strategy_id` (default = `name`). In `_slot_builder`: construct ONE
  shared `CachedStateDAL(state_db_path)` once (pass it in, or memoize on the
  runtime), then `dal = StrategyScopedDAL(shared, strategy_id, account_alias)`.
  `run_migrations()` once on the shared DAL. Everything downstream (Router,
  Scanner, TradeJournal, Reconciler) receives the scoped DAL.

- [ ] **Step 4: Run** — `uv run pytest tests/unit/test_slot_builder_unified.py tests/unit/test_engine*.py -q` → PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(engine): slots share one state DB via scoped DAL; per-strategy flag paths"`

## Task 5: Reconcile groups by account

**Files:** Modify `hlanalysis/engine/reconcile.py`, `hlanalysis/engine/runtime.py`. Test: `tests/unit/test_reconcile_by_account.py`.

- [ ] **Step 1: Write failing test** — one wallet, two strategies holding
  *disjoint* symbols; the per-account reconcile attributes each venue position to
  its owning strategy and flags a truly-unclaimed symbol as orphan:

```python
def test_reconcile_attributes_disjoint_symbols_to_owning_strategy(...):
    # venue_state has positions for @1 (owned by strat A) and @2 (owned by B)
    # reconcile(account="w1") leaves A.@1 and B.@2 intact, no false orphan.
    ...
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement.** Add a per-account reconcile entry point that takes
  the wallet's `venue_state` + the list of `(strategy_id, scoped_dal)` sharing
  that account. For each venue position, find the strategy whose DB has a
  position/order on that symbol (disjoint → unique) and reconcile within that
  scope; a symbol no scope claims → orphan alert (existing path). When a wallet
  has exactly one strategy (Stage 0), this reduces to the current per-slot
  behavior. Update `runtime.py` to drive reconcile per distinct `account` rather
  than per slot.

- [ ] **Step 4: Run** — `uv run pytest tests/unit/test_reconcile*.py -q` → PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(engine): reconcile per account, attribute positions per strategy"`

## Task 6: Per-strategy realized PnL / daily-loss

**Files:** Modify `hlanalysis/engine/state.py` (`realized_pnl_since`), `hlanalysis/engine/reconcile_report.py`, the daily-loss caller. Test: `tests/unit/test_per_strategy_pnl.py`.

- [ ] **Step 1: Write failing test** — two strategies on one wallet with disjoint
  fills; `realized_pnl_since(strategy_id="A")` returns only A's sum:

```python
def test_realized_pnl_is_per_strategy(tmp_path):
    base = CachedStateDAL(tmp_path/"u.db"); base.run_migrations()
    # append_fill for A (closed_pnl=+5) and B (closed_pnl=-3)
    # assert scoped A realized == +5, B == -3
    ...
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement.** `realized_pnl_since` (and `settlement_pnl_since`)
  accept `strategy_id` and filter `fill`/`settlement` rows by it. The scoped DAL
  forwards its `strategy_id`. HL venue-fill mirroring stamps `strategy_id` via
  symbol→question→owning-strategy (disjoint map). The daily-loss gate already
  calls through the slot's DAL, so it becomes per-strategy automatically once the
  scoped DAL is in place — add the test to lock it.

- [ ] **Step 4: Run** — `uv run pytest tests/unit/test_per_strategy_pnl.py tests/unit/test_reconcile_report*.py -q` → PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(engine): per-strategy realized PnL + daily-loss attribution"`

## Task 7: Offline merge migration script

**Files:** Create `scripts/merge_slot_dbs.py`. Test: `tests/unit/test_merge_slot_dbs.py`.

- [ ] **Step 1: Write failing test** — N fixture slot DBs (each pre-tagged with a
  distinct alias) merge into one unified DB; assert per-table row counts equal
  the sum, every row carries the right `(account, strategy_id)`, no PK collision,
  and a second run is idempotent (no duplicates):

```python
def test_merge_is_complete_and_idempotent(tmp_path):
    # build 2 slot DBs via StateDAL, seed positions/fills/events
    # run merge into unified; assert counts + tags; run again; counts unchanged.
    ...
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement.** CLI `merge_slot_dbs.py --src <dir> --out <unified.db>
  --map alias=account[,alias=account...]`. For each `<dir>/<alias>/state.db`:
  open read-only, for each table SELECT all rows, stamp `strategy_id=alias` and
  `account=<mapped>`, INSERT-OR-IGNORE into the unified DB (idempotent on PKs).
  Run the unified DB through `run_migrations()` first. Print per-table
  source/dest counts for verification.

- [ ] **Step 4: Run** — `uv run pytest tests/unit/test_merge_slot_dbs.py -q` → PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(scripts): offline merge of per-slot DBs into unified DB"`

## Task 8: Read-side touchpoints

**Files:** Modify `scripts/engine_events.py`, `scripts/sync-engine-to-s3.sh`, `hlanalysis/engine/diag.py`. Tests: extend `tests/unit/...` for diag/events; `tests/scripts/test_sync_engine_to_s3.sh`.

- [ ] **Step 1: Write/adjust failing tests** — `engine_events`/`diag` query a
  single unified DB filtered by `strategy_id`; the sync test asserts one
  `state.db.gz` (not per-alias) plus per-strategy `gate_decisions.jsonl.gz`.

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement.** `engine_events.py`: open the single DB, filter by
  `question_idx` (and optional `--strategy`). `diag.py`: `--alias` becomes a
  `strategy_id` filter on the one DB. `sync-engine-to-s3.sh`: snapshot the single
  `state.db` once; still loop the per-strategy dirs for `gate_decisions.jsonl`.

- [ ] **Step 4: Run** — `uv run pytest tests/unit/test_engine_diag*.py -q` and `bash tests/scripts/test_sync_engine_to_s3.sh` → PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(ops): read-side tools query unified DB by strategy_id"`

## Task 9: Config + cutover runbook

**Files:** Modify `config/strategy.yaml`, `config/deploy.yaml` (add `strategy_id` per entry, single `state_db_path`); create `docs/runbooks/unified-db-cutover.md`.

- [ ] **Step 1:** Add `strategy_id` to each strategy entry (= current alias) and
  set one shared `state_db_path` (e.g. `/data/engine/state.db`); keep the 1:1
  account mapping unchanged.
- [ ] **Step 2:** Validate config loads — `uv run python -c "from hlanalysis.engine.config import load_deploy_config; ..."` (mirror existing config tests). Add/extend a config test.
- [ ] **Step 3:** Write the cutover runbook: archive 5 DBs to S3 → `systemctl
  stop hl-engine` (SSM) → run `merge_slot_dbs.py` → start → `reconcile-report`
  vs venue → rollback = restore the 5 originals + revert config.
- [ ] **Step 4:** `uv run pytest tests/unit/test_config*.py -q` → PASS.
- [ ] **Step 5: Commit** — `git commit -am "chore(config): strategy_id + unified state_db_path; cutover runbook"`

## Final gate (before any deploy)

- [ ] `uvx ruff check hlanalysis scripts && uvx ruff format --check hlanalysis tests`
- [ ] `uv run pytest -q` (full suite green; the `sklearn` NBA test is a known env-skip)
- [ ] `make parity-gate` green (state/observability change only — must not move decision parity)
- [ ] Human review of the diff before cutover; cutover is an explicit, separate operator step (runbook).

---

## Self-review (against spec)

- **Spec coverage:** §3.1 slot identity → T4/T9; §3.2 schema → T1/T2; §4.1
  reconcile-by-account → T5; §4.2 per-strategy risk → T6; §4.3 scoped DAL/caching
  → T3/T4; §4.4 migration/staging → T7 (merge) + T9 (runbook); §4.5 touchpoints →
  T8; §5 testing → per-task tests + final gate. All covered.
- **Placeholders:** test bodies marked `...` are intentional skeletons the
  implementer fills using the named fixtures/asserts described in the same step;
  every such step names the exact methods and the assertion. No vague
  "add error handling".
- **Type consistency:** `strategy_id`/`account` names, `StrategyScopedDAL`,
  `state_db_path()`/`flag_path_for()` used consistently across T1–T9.
- **Scope:** Stage 0 only (behavior-preserving). Stage 1 (enable sharing) is a
  config-only follow-up, intentionally out of this plan.
