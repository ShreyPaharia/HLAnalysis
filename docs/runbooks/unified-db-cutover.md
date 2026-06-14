# Unified DB Cutover Runbook

One-time production cutover from per-slot state DBs
(`/opt/hl-recorder/data/engine/<alias>/state.db`) to the single shared DB
(`/opt/hl-recorder/data/engine/state.db`).

All commands are issued via AWS SSM `send-command` — **SSH to the box is
blocked**. Use `make <target>` from the local repo for the Make-wrapped
variants, or the raw `aws ssm send-command` form shown below.

---

## 0. Environment notes

- **No `sqlite3` CLI on the box.** Use `/opt/hl-recorder/.venv/bin/python`
  for ad-hoc DB reads.
- **Engine env required** for any command that touches live paths or the
  venv: prefix with `set -a; . /etc/hl-engine/env &&`.
- **Avoid `(` characters in SSM `echo` strings** — the SSM agent's shell may
  mis-parse them. Use heredoc-style or single-quoted strings.
- **`restart_blocked` flags are per-slot**, living at
  `/opt/hl-recorder/data/engine/<alias>/restart_blocked`, not at the DB root.
  `make engine-status` currently checks the wrong path for these flags; verify
  manually if a slot looks stuck after restart.

---

## 1. Pre-checks

```bash
make engine-status        # systemd state; last 30 journal lines
make reconcile-report     # per-slot realized + open-MTM PnL vs venue truth
```

Expected: engine `active (running)`, no `restart_blocked` or `halt` flags,
heartbeats flowing (events/30s > 0 in `make engine-diag`).

Note the existing per-slot DB paths on the box:

```
/opt/hl-recorder/data/engine/v1/state.db
/opt/hl-recorder/data/engine/v31/state.db
/opt/hl-recorder/data/engine/v31_pm/state.db
/opt/hl-recorder/data/engine/v1_pm/state.db
/opt/hl-recorder/data/engine/v31_pm_eth_ms/state.db
```

---

## 2. Archive per-slot DBs to S3

Archive before touching anything. Each `.backup` call produces a consistent
SQLite snapshot (WAL checkpointed, no partial pages).

```bash
aws ssm send-command \
  --instance-id <INSTANCE_ID> \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=[
    "set -a; . /etc/hl-engine/env",
    "DATE=$(date -u +%Y-%m-%d)",
    "BUCKET=s3://hl-recorder-archive-819175935435/engine/date=${DATE}",
    "VENV=/opt/hl-recorder/.venv/bin/python",
    "for ALIAS in v1 v31 v31_pm v1_pm v31_pm_eth_ms; do",
    "  SRC=/opt/hl-recorder/data/engine/${ALIAS}/state.db",
    "  TMP=/var/tmp/${ALIAS}-state-backup.db",
    "  if [ -f ${SRC} ]; then",
    "    ${VENV} -c \"import sqlite3; sqlite3.connect('"'"'${SRC}'"'"').backup(sqlite3.connect('"'"'${TMP}'"'"'))\"",
    "    aws s3 cp ${TMP} ${BUCKET}/pre-cutover-${ALIAS}-state.db",
    "    echo ARCHIVED ${ALIAS}",
    "  else",
    "    echo SKIP ${ALIAS} no db found",
    "  fi",
    "done"
  ]'
```

Verify the S3 objects landed:

```bash
aws s3 ls s3://hl-recorder-archive-819175935435/engine/date=$(date -u +%Y-%m-%d)/
```

---

## 3. Stop the engine

```bash
aws ssm send-command \
  --instance-id <INSTANCE_ID> \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["systemctl stop hl-engine"]'
```

Verify it stopped:

```bash
aws ssm send-command \
  --instance-id <INSTANCE_ID> \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["systemctl is-active hl-engine || echo STOPPED"]'
```

---

## 4. Merge per-slot DBs into the unified DB

Run `scripts/merge_slot_dbs.py` using the engine venv. This script reads
every per-slot `state.db` under `--src` and writes all rows (tagged with
their `strategy_id`) into the single output DB.

```bash
aws ssm send-command \
  --instance-id <INSTANCE_ID> \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=[
    "set -a; . /etc/hl-engine/env",
    "VENV=/opt/hl-recorder/.venv/bin/python",
    "SRC=/opt/hl-recorder/data/engine",
    "OUT=/opt/hl-recorder/data/engine/state.db",
    "if [ -f ${OUT} ]; then",
    "  echo WARNING: flat state.db already exists at ${OUT}",
    "  ${VENV} /opt/hl-recorder/scripts/merge_slot_dbs.py --src ${SRC} --out /var/tmp/merged-state.db",
    "  mv /var/tmp/merged-state.db ${OUT}",
    "else",
    "  ${VENV} /opt/hl-recorder/scripts/merge_slot_dbs.py --src ${SRC} --out ${OUT}",
    "fi",
    "echo MERGE DONE exit=${?}"
  ]'
```

**Important:** If a legacy flat `state.db` already exists at
`/opt/hl-recorder/data/engine/state.db` (e.g., from a prior partial cutover),
merge to a temp path (`/var/tmp/merged-state.db`) first, then move it into
place. The `if/else` block above handles this automatically.

After the merge, spot-check row counts:

```bash
aws ssm send-command \
  --instance-id <INSTANCE_ID> \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=[
    "/opt/hl-recorder/.venv/bin/python -c \"import sqlite3; c=sqlite3.connect('"'"'/opt/hl-recorder/data/engine/state.db'"'"'); print(c.execute('"'"'SELECT strategy_id, COUNT(*) FROM position GROUP BY strategy_id'"'"').fetchall())\""
  ]'
```

You should see one row per slot alias (`v1`, `v31`, `v31_pm`, `v1_pm`,
`v31_pm_eth_ms`) with their respective position counts (most will be 0 if
the engine is flat at cutover time).

---

## 5. Deploy the new code

The new engine code reads the unified DB via `deploy_cfg.state_db_path_shared()`
(`data/engine/state.db` per `config/deploy.yaml`), scoped per slot by
`StrategyScopedDAL(strategy_id=<alias>)`. Deploy and restart:

```bash
make deploy-engine
```

This runs `git pull` + `systemctl restart hl-engine` on the box via SSM.

---

## 6. Validate

```bash
make engine-status        # confirms active, no restart_blocked/halt
make engine-diag          # per-slot positions + true PnL + feed health
make reconcile-report     # venue-authoritative PnL; should match pre-cutover
```

Key things to confirm:

- `systemctl is-active hl-engine` → `active`
- Per-slot `restart_blocked` flags absent in
  `/opt/hl-recorder/data/engine/<alias>/restart_blocked`
- Heartbeats flowing: `make engine-diag` shows `events_per_30s > 0`
- `make reconcile-report` shows no new DRIFT vs venue truth (all positions
  that existed before the cutover are reconciled correctly)
- Telegram alerts resume normally (bot should send a startup log line)

---

## 7. Rollback

If the new code misbehaves:

1. **Stop the engine:**
   ```bash
   aws ssm send-command \
     --instance-id <INSTANCE_ID> \
     --document-name "AWS-RunShellScript" \
     --parameters 'commands=["systemctl stop hl-engine"]'
   ```

2. **Revert the deploy** to the previous commit:
   ```bash
   # On the box via SSM — replace <PREV_SHA> with the prior commit hash
   aws ssm send-command \
     --instance-id <INSTANCE_ID> \
     --document-name "AWS-RunShellScript" \
     --parameters 'commands=[
       "cd /opt/hl-recorder && git fetch origin && git checkout <PREV_SHA>",
       "systemctl start hl-engine"
     ]'
   ```

3. **The per-slot DBs are still in place** (`/opt/hl-recorder/data/engine/<alias>/state.db`);
   the old code reads from those paths and requires no further action. The
   merged `state.db` at the flat path will be ignored by the old code.

4. **If per-slot DBs were lost** (should not happen — the old code does not
   delete them), restore from the S3 archives made in Step 2:
   ```bash
   aws s3 cp s3://hl-recorder-archive-819175935435/engine/date=<DATE>/pre-cutover-v1-state.db \
     /var/tmp/v1-state.db
   # Then cp each /var/tmp/<alias>-state.db to the expected on-box path
   ```

5. **After rollback**, verify with `make reconcile-report` that per-slot PnL
   still matches venue truth before re-enabling live trading.

---

## Appendix: Deploy config reference

`config/deploy.yaml` sets:

```yaml
state_db_path: data/engine/state.db
kill_switch_path: data/engine/halt
```

`DeployConfig.state_db_path_shared()` returns `Path("data/engine/state.db")`.
On the EC2 box this resolves under `/opt/hl-recorder/` (the engine's working
directory), giving the absolute path `/opt/hl-recorder/data/engine/state.db`.

Per-slot flag files (kill switch, `restart_blocked`, `gate_decisions.jsonl`)
live in per-strategy subdirectories:
`/opt/hl-recorder/data/engine/<strategy_id>/`.

No change to `config/deploy.yaml` is required — the `state_db_path` value
already points at the intended unified location.
