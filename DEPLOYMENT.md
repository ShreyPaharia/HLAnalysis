# Deployment to AWS Tokyo

Infrastructure is ready to deploy the hl-recorder to AWS ap-northeast-1 (Tokyo) for lowest latency to Hyperliquid.

## What Was Created

The following files have been added for AWS deployment:

### Infrastructure (CDK)
- `deploy/cdk/main.go` - CDK app entry point
- `deploy/cdk/stack.go` - EC2, EBS, IAM, security group definitions
- `deploy/cdk/go.mod` - Go dependencies
- `deploy/cdk/cdk.json` - CDK configuration

### Deployment Scripts
- `scripts/deploy.sh` - Deploy code updates via SSM
- `Makefile` - Convenient deployment commands

### Documentation
- `deploy/README.md` - Full deployment guide

## Quick Start

### 1. Prerequisites

Install required tools:

```bash
# AWS CLI
# macOS: brew install awscli
# Configure with your credentials
aws configure

# AWS CDK CLI
npm install -g aws-cdk

# Go 1.22+ (for CDK)
# macOS: brew install go
```

Set environment variables:

```bash
export CDK_DEFAULT_REGION=ap-northeast-1
export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
```

### 2. Bootstrap CDK (one-time)

```bash
make cdk-bootstrap
```

### 3. Deploy Infrastructure

```bash
make cdk-deploy
```

This will:
- Create t4g.micro EC2 instance in Tokyo
- Attach 10GB EBS volume at `/data`
- Install Python 3.12, uv, and dependencies
- Clone this repository
- Start the recorder automatically

Wait ~5 minutes for deployment to complete.

### 4. Verify It's Running

```bash
# Check recorder status
make status

# View logs
make logs

# SSH into the instance
make ssh-ec2
```

## Cost Breakdown

- **EC2 t4g.micro**: $6.20/month (1GB RAM, ARM64)
- **EBS 10GB gp3**: $0.80/month
- **Total**: ~$7.00/month

## Architecture

```
┌─────────────────────────────────────────┐
│  AWS ap-northeast-1 (Tokyo)             │
│                                          │
│  ┌─────────────────────┐                │
│  │  EC2 t4g.micro      │                │
│  │  Amazon Linux 2023  │                │
│  │  ARM64/Graviton     │                │
│  │                     │                │
│  │  Python 3.12 + uv   │                │
│  │  systemd service    │                │
│  └─────────┬───────────┘                │
│            │                             │
│  ┌─────────▼───────────┐                │
│  │  EBS gp3 10GB       │                │
│  │  /data (Parquet)    │                │
│  └─────────────────────┘                │
│                                          │
└─────────────────────────────────────────┘
           │
           │ WebSocket
           ▼
  api.hyperliquid.xyz (~5ms)
  Binance APIs
```

## Daily Workflow

After initial deployment, to update code:

```bash
# Make changes to the code locally
git commit -am "Your changes"
git push origin main

# Deploy to EC2
make deploy
```

The deploy script will:
1. Pull latest code on EC2
2. Reinstall Python packages
3. Restart the service

## Monitoring

> **Access is SSM, not SSH.** Plain SSH to the box is blocked and an interactive
> shell should never be *assumed* (especially from automation/agents). The reliable,
> scriptable path is the non-interactive `make` targets below — each runs one
> `aws ssm send-command`. `make ssh-ec2` opens an `ssm start-session` for hands-on
> debugging only; don't depend on it in scripts. See CLAUDE.md → "Deploy is SSM-only"
> and "Ops & monitoring".

View recorder status and logs:

```bash
# Quick status check
make status

# Tail logs (last 100 lines)
make logs

# Interactive session
make ssh-ec2

# Once inside:
sudo systemctl status hl-recorder.service
tail -f /data/logs/recorder.log
journalctl -u hl-recorder.service -f
```

## Data Storage

Parquet files are written to `/data` with the following structure:

```
/data/
├── venue=hyperliquid/
│   ├── product_type=perp/
│   │   └── mechanism=clob/
│   │       └── event=trades/
│   │           └── symbol=BTC/
│   │               └── date=2026-05-06/
│   │                   └── hour=10/
│   │                       └── *.parquet
│   └── product_type=prediction_binary/
│       └── ...
└── venue=binance/
    └── ...
```

Check disk usage:

```bash
make ssh-ec2
df -h /data
du -sh /data/venue=*
```

### S3 Archival

Data is automatically archived to S3 (`hl-recorder-archive-<account>` in `ap-northeast-1`)
to prevent the 10 GB EBS volume from filling.

**On EC2:**

- `hl-recorder-sync.timer` runs hourly, pushes today + yesterday partitions to S3
- `hl-recorder-cleanup.timer` runs daily at 04:00 UTC: full sync, then deletes EBS
  partitions strictly older than `RETENTION_DAYS` (default `3`) that pass a
  per-partition file-count check
- Lifecycle: `Standard → Standard-IA (30d) → Deep Archive (90d)` keeps long-term
  cost near zero

Tune retention without redeploy:

```bash
make ssh-ec2
sudo systemctl edit hl-recorder-cleanup.service   # add Environment=RETENTION_DAYS=7
```

Inspect timer state and recent runs:

```bash
sudo systemctl list-timers | grep hl-recorder
sudo journalctl -u hl-recorder-sync.service -n 50
sudo journalctl -u hl-recorder-cleanup.service -n 50
```

**On your laptop:**

```bash
make pull-data        # incremental aws s3 sync s3://<bucket>/ ./data/
```

First pull is bandwidth-heavy (current data ~6 GB); subsequent pulls only
fetch new partitions. Egress to your laptop is free under the 100 GB/mo
AWS free tier.

Cost: ~$0.50/mo year 1, dropping to ~$0.10/mo steady-state once the lifecycle
moves bulk data to Deep Archive.

## Troubleshooting

### Deployment fails

Check CloudFormation:

```bash
aws cloudformation describe-stack-events \
  --stack-name HLRecorderStack \
  --max-items 20
```

### Service not starting

```bash
make ssh-ec2

# Check service status
sudo systemctl status hl-recorder.service

# View service logs
sudo journalctl -u hl-recorder.service -n 100

# Check user-data script (instance bootstrap)
sudo cat /var/log/cloud-init-output.log
```

### High latency

Verify you're in Tokyo:

```bash
make ssh-ec2
curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone
# Should show: ap-northeast-1a
```

Ping Hyperliquid:

```bash
ping -c 5 api.hyperliquid.xyz
# Should show ~5ms RTT
```

## Repository URL Configuration

The CDK stack clones from:

```
https://github.com/shreypaharia/HLAnalysis.git
```

If you fork this repo, update `repoURL` in `deploy/cdk/stack.go` line 73:

```go
repoURL := "https://github.com/YOUR_USERNAME/HLAnalysis.git"
```

Then redeploy:

```bash
make cdk-deploy
```

## Cleanup

To destroy all infrastructure:

```bash
make cdk-destroy
```

**WARNING**: This deletes the EC2 instance and creates an EBS snapshot. Download your data first if needed.

## Next Steps

1. **Deploy now**: Run `make cdk-bootstrap && make cdk-deploy`
2. **Monitor for 24h**: Check logs with `make logs`, verify data quality
3. **Tune retention if needed**: Default is `RETENTION_DAYS=3` on EBS; archived data lives in S3 indefinitely. Adjust via `sudo systemctl edit hl-recorder-cleanup.service` (see "S3 Archival" above).
4. **Pull data locally**: Run `make pull-data` to mirror the archive bucket to `./data/` for analytics
5. **Query data**: Use DuckDB to analyze recorded Parquet files

See `deploy/README.md` for more details.

## Engine — paper mode

Phase 1 default. The engine runs every code path end-to-end except the actual REST POST is logged-only.

```bash
uv run hl-engine \
  --strategy-config config/strategy.yaml \
  --deploy-config config/deploy.yaml \
  --symbols-config config/symbols.yaml
```

Required env vars (referenced by `config/deploy.yaml`):
- `HL_ACCOUNT_ADDRESS`, `HL_API_SECRET_KEY`
- `TG_BOT_TOKEN`, `TG_CHAT_ID`

Kill switch: `touch data/engine/halt` to stop new entries (existing positions held).
Restart drift: if the merge at startup detects ghost / orphan / position mismatch cases, the engine writes `data/engine/restart_blocked` and the scanner does not resume. Investigate, then `rm` the file and restart.

## Installing the engine on EC2

The recorder's user-data installs `hl-engine.service` automatically on a fresh CDK deploy. For an already-running box (or after editing the unit content / rotating SSM secrets), apply the change without a CDK redeploy:

```bash
# 1. Push your branch first — the installer pulls fresh code on the box.
git push origin <branch>

# 2. Pre-stage SSM secrets (one-time; SecureStrings encrypted with alias/aws/ssm).
aws ssm put-parameter --name /hl-engine/account-address --value '0x...' --type String --region ap-northeast-1
aws ssm put-parameter --name /hl-engine/api-secret-key  --value '0x...' --type SecureString --region ap-northeast-1
aws ssm put-parameter --name /hl-engine/tg-bot-token    --value '...'   --type SecureString --region ap-northeast-1
aws ssm put-parameter --name /hl-engine/tg-chat-id      --value '...'   --type String --region ap-northeast-1

# 3. Install the unit on the live box (idempotent — re-run anytime).
make install-engine-on-ec2

# 4. From here on, regular deploys work.
make deploy-engine
```

`scripts/install-engine-systemd.sh` writes `/etc/systemd/system/hl-engine.service` and runs `daemon-reload + enable --now`. The unit's `ExecStartPre=+/opt/hl-recorder/scripts/fetch-engine-secrets.sh` re-pulls SSM on every restart, so rotated secrets land via `make deploy-engine` — no installer re-run needed.

Confirm health:

```bash
make engine-status
```

Look for: `active (running)`, no `restart_blocked` flag, journal lines showing `paper_mode=true` on first start.

## Going live (Phase 1, Week 2)

1. Verify ≥ 1 observed allowlisted question's full lifecycle in paper mode.
2. Edit `config/strategy.yaml`: set `paper_mode: false`.
3. Confirm caps: `$100/position, $500 global, 5 concurrent, $200 daily-loss cap, 10% stop-loss`.
4. Restart engine; watch Telegram for first entry/exit.
5. If anything looks wrong: `touch data/engine/halt` immediately.

## Multi-account topology

The engine runs N `(strategy, account)` pairs concurrently in **one process**,
sharing one Hyperliquid WebSocket feed and one MarketState. Each pair is
otherwise fully isolated: separate signing key, state DB, risk gate, cloid
prefix, kill switch, and daily-loss cap.

### Design decisions

**Config schema = A** (one YAML, list of strategies). The fields across v1 and
v3.1 overlap heavily (allowlist matchers, defaults, global risk block), so a
single `strategies: [...]` list under `config/strategy.yaml` is more ergonomic
than juggling two files with a `--config A --config B` invocation.

**Process model = Y** (one process, N strategies). The t4g.micro has 1 GB RAM
plus 1 GB swap — running two engine processes would double the WS connection
count, SDK footprint, and Python interpreter overhead. The existing per-loop
`try/except` already provides per-strategy crash isolation; a stray exception
in `theta_harvester.evaluate()` is caught by `_scan_loop` and only that slot's
tick is dropped, never the WS feed or sibling strategies.

### Config layout

`config/strategy.yaml`:

```yaml
strategies:
  - name: late_resolution
    account_alias: v1                # → deploy.hl_accounts.v1
    strategy_type: late_resolution
    paper_mode: false
    allowlist: [...]
    defaults: {...}
    global: {...}

  - name: theta_harvester
    account_alias: v31
    strategy_type: theta_harvester
    paper_mode: false
    allowlist: [...]
    defaults: {...}
    global: {...}
    theta: { edge_max: 0.20, ... }   # v3.1-specific knobs
```

`config/deploy.yaml`:

```yaml
deploy:
  hl_accounts:
    v1:
      account_address: ${HL_ACCOUNT_ADDRESS}
      api_secret_key: ${HL_API_SECRET_KEY}
      base_url: https://api.hyperliquid.xyz
    v31:
      account_address: ${HL_ACCOUNT_ADDRESS_V31}
      api_secret_key: ${HL_API_SECRET_KEY_V31}
      base_url: https://api.hyperliquid.xyz
```

`.env.local` (or AWS SSM in prod):

```
HL_ACCOUNT_ADDRESS=0x...
HL_API_SECRET_KEY=0x...
HL_ACCOUNT_ADDRESS_V31=0x...
HL_API_SECRET_KEY_V31=0x...
```

### What gets namespaced per account

When `hl_accounts` has more than one entry (or any non-`default` alias), the
engine namespaces local artifacts under the alias:

| Path                              | Single-account default          | Multi-account                              |
| --------------------------------- | ------------------------------- | ------------------------------------------ |
| State DB                          | `data/engine/state.db`          | `data/engine/<alias>/state.db`             |
| Kill switch                       | `data/engine/halt`              | `data/engine/<alias>/halt`                 |
| Restart-drift block file          | `data/engine/restart_blocked`   | `data/engine/<alias>/restart_blocked`      |
| Cloid prefix (DB + venue)         | `hla-<uuid>`                    | `hla-<alias>-<hex>`                        |

Kill one strategy without touching the other:

```bash
touch data/engine/v1/halt        # only v1 halts; v31 keeps running
```

Clear restart drift on one strategy:

```bash
rm data/engine/v31/restart_blocked
```

### Migration from a single-account deployment

If you're upgrading an existing engine that used `data/engine/state.db`:

1. Stop the engine.
2. Decide your alias for the existing strategy (e.g. `v1`).
3. Move state into the new namespaced layout:
   ```bash
   mkdir -p data/engine/v1
   mv data/engine/state.db   data/engine/v1/state.db
   mv data/engine/state.db-wal  data/engine/v1/  2>/dev/null
   mv data/engine/state.db-shm  data/engine/v1/  2>/dev/null
   ```
4. Update `config/strategy.yaml` to the new `strategies:` list shape with
   `account_alias: v1`.
5. Restart. The reconcile loop will pick up live positions/orders that were
   placed before the migration, because the venue still has them.

Alternatively, set `account_alias: default` in `strategy.yaml` and
`hl_accounts.default:` in `deploy.yaml` — that keeps the legacy flat paths
(`data/engine/state.db`) and avoids the move. You lose alias-tagged cloids
on the venue but the engine still runs.

### Engine lifecycle in multi-account

| Event                               | Effect                                                                 |
| ----------------------------------- | ---------------------------------------------------------------------- |
| v1's daily_loss_cap breached        | v1 latches halted; v31 keeps trading. Engine exits when ALL halted.    |
| v1 hits `halt` kill switch          | Same — only v1 stops; v31 keeps trading.                               |
| v1's restart-drift gate trips       | v1's `_scan_loop` doesn't start; v31's does.                           |
| SIGINT / SIGTERM                    | Both slots cancelled; clean shutdown.                                  |
| New question (`QuestionMetaEvent`)  | One Telegram alert, marked seen in BOTH slots' DBs (per-alias rows).   |
| WS reconnect                        | One reconnect — shared across all slots.                               |

### Telegram alerts are alias-prefixed

Every bus event carries the originating slot's `account_alias`, and `AlertRules`
renders that as a bold tag at the start of each Telegram message:

```
[v1] 🟢 ENTRY
BTC > 80,000 on 2026-05-19 @ 12:00 UTC
YES (#30)
BUY 100 @ $0.96  (notional $96.00)
q=42  #30

[v31] 🟢 ENTRY
BTC > 80,000 on 2026-05-19 @ 12:00 UTC
YES (#30)
BUY 100 @ $0.96  (notional $96.00)
q=42  #30
```

Dedupe keys are also scoped per alias, so v1 and v31 emitting the same shape of
RiskVeto on the same symbol each get their own alert rather than collapsing
into one. `NewQuestion` is cross-slot — emitted once globally on first sight —
so it renders without an alias tag.

### Heartbeat output

The heartbeat line is now one-per-slot. Expect output like:

```
heartbeat alias=v1  events=12000 (+340) scans=120 (+12) decisions=3  | btc=$80300.42 questions=8 positions=1 live_orders=0
heartbeat alias=v31 events=12000 (+340) scans=120 (+12) decisions=1  | btc=$80300.42 questions=8 positions=0 live_orders=0 HALTED
```

`events` is shared (one WS feed); `scans` / `decisions` / positions / live
orders are per slot. `HALTED` is appended when the slot has latched its kill
switch or daily-loss halt.

### Capital and HL API rate limits

**Out of scope for the engine — flag for ops:**

- Capital allocation between accounts: the engine does NOT redistribute funds
  between v1 and v31. Each account's HL wallet must be funded independently
  (USDC deposit per wallet).
- HL rate limits: with two strategies on shared market data, the WS read load
  is unchanged. The write load (place/cancel) is up to 2× the single-account
  baseline — currently well below HL's documented limits for retail accounts.
  Monitor if order volume increases substantially.
- The engine's `_reconcile_loop` runs once per slot per `reconcile_interval_seconds`
  (default 60s). With 2 accounts that's 2 read calls/min to HL — negligible.

## Polymarket data

The recorder subscribes to Polymarket's CLOB WebSocket alongside HL and Binance
once a `polymarket` subscription block is added to `config/symbols.yaml`. The
PM adapter (`hlanalysis/adapters/polymarket.py`) drives two background tasks:

1. **Gamma poller** — every 60s, `GET https://gamma-api.polymarket.com/events?series_slug=<slug>&closed=false`,
   diff against the in-memory market set, emit a `QuestionMetaEvent` for each
   newly-listed market and a `SettlementEvent` for each newly-resolved one.
2. **CLOB WebSocket** — `wss://ws-subscriptions-clob.polymarket.com/ws/market`,
   subscribed via `{"type":"market","assets_ids":[...]}`. Frames are dispatched
   by `event_type` (`book` / `price_change` / `last_trade_price`) into the
   normalizers in `polymarket_normalize.py`. The receive loop is wrapped in an
   exponential-backoff retry (1s → cap 30s, reset on successful (re)connect).

### Smoke-test runbook

```bash
# 1. Trim a PM-only config (uses the existing match: filters)
cat > /tmp/symbols_pm_only.yaml <<'YAML'
subscriptions:
  - venue: polymarket
    product_type: prediction_binary
    mechanism: clob
    symbol: "*"
    channels: [trades, book]
    match:
      underlying: BTC
      class: priceBinary
      series_slug: btc-up-or-down-daily
YAML

# 2. Launch the recorder for ≥ 5 minutes
uv run python -m hlanalysis.recorder.main \
    --config /tmp/symbols_pm_only.yaml \
    --data-root data/recorder \
    --log-level INFO &
RECORDER_PID=$!
sleep 360
kill -INT $RECORDER_PID

# 3. Sanity-check the captured rows
uv run python -c "
import glob, pyarrow.parquet as pq
files = sorted(glob.glob('data/recorder/venue=polymarket/**/*.parquet', recursive=True))
print(f'parquet files: {len(files)}')
for p in files[-3:]:
    tbl = pq.ParquetFile(p).read()
    print(p, '→', tbl.num_rows, 'rows, schema:', tbl.schema.names[:6])
"
```

Expected on a healthy run:
- One or more parquet files under
  `data/recorder/venue=polymarket/product_type=prediction_binary/mechanism=clob/event={book_snapshot,trade}/symbol=<76-digit-id>/date=YYYY-MM-DD/hour=HH/`.
- `book_snapshot` rows contain `bid_px`/`bid_sz`/`ask_px`/`ask_sz` as float
  lists; `trade` rows contain `price`/`size`/`side`/`trade_id`.
- The `symbol` column is the raw PM ERC-1155 token ID (76-digit string),
  never coerced to int.

### When the BTC Up/Down series is dead-quiet

Polymarket only lists the **next** day's `btc-up-or-down-daily` market about
2.5h after the prior day settles. Between settlement and listing, Gamma will
return zero active markets and the adapter emits a single `HealthEvent
kind="no_active_markets"` then idles. This is expected. The Gamma poll loop
keeps running on a 60s cadence and the WS subscription kicks in automatically
once a new market appears.

### Failure modes seen during build-out

- **`websockets` 16.x lazy submodule loading** — `import websockets` alone
  does NOT make `websockets.exceptions` accessible. The adapter explicitly
  does `import websockets.exceptions`.
- **PyArrow dataset partition merge error** when reading partitioned files
  via `pq.read_table(path)` — pyarrow merges the in-file `venue` column
  against the `venue=...` hive partition value and errors on type mismatch.
  Read individual files via `pq.ParquetFile(path).read()` instead.

### Reconciliation report (Phase 0)

`make reconcile-report` — per-slot realized + open-MTM true PnL reconciled to the
venue, with position-drift detection. `JSON=1` for machine-readable output.
Exits nonzero on drift; sends a Telegram alert when `TELEGRAM_*` env is set.

**Drift kinds:**

| Kind | Meaning |
|------|---------|
| `qty_mismatch` | DB and venue agree the symbol is open but the qty differs beyond tolerance. |
| `vanished` | DB has an open position the venue doesn't recognise. |
| `orphan` | Venue has an open position the DB has no record of. |

Position reconciliation is skipped for a slot when `positions_known=False` (e.g. a
transient PM data-API flap) to avoid false `vanished` drift — PnL is still reported.

**Phase 0 exit gate — all must hold for 7 consecutive days, all 4 slots:**

- [ ] `engine-diag` + `reconcile-report` accessible via one command (documented above).
- [ ] Per-slot true PnL (realized + settlement) reported by `engine-diag`; total
      true PnL (+ open MTM) reported by `reconcile-report`.
- [ ] `reconcile-report` shows no unexplained drift beyond tolerance.
- [ ] An injected drift raises a Telegram alert (verified once).

### Daily desk report (Telegram)

`hl-daily-report.timer` runs `hl-daily-report.service` at **06:30 UTC daily**, which
runs `python -m hlanalysis.engine.reconcile_report --summary` and sends a single
Telegram message: per-strategy outcome-market PnL, fill count, account value, and
reconcile status, plus a desk total. Unit files are version-controlled at
`deploy/systemd/`. To (re)install on the box (e.g. after an instance rebuild),
via SSM as root:

```
cp deploy/systemd/hl-daily-report.{service,timer} /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now hl-daily-report.timer
systemctl list-timers hl-daily-report.timer   # verify next run
```

Manual send: `systemctl start hl-daily-report.service` (or run the module with
`--summary` directly). The report reads only (no DB writes); TG creds come from
`deploy.alerts.telegram` (env `TG_BOT_TOKEN`/`TG_CHAT_ID`).

### Daily engine S3 sync (SSM-free analysis)

`hl-engine-s3-sync.timer` runs `scripts/sync-engine-to-s3.sh` at **06:45 UTC
daily** (after settlement + the desk report), pushing a *consistent* snapshot of
each slot's operational data to the **same archive bucket** under a separate
`engine/` prefix. This replaces the slow per-run SSM `user_fills` dumps: analysts
pull it locally and query `state.db` directly.

Per slot alias under `/opt/hl-recorder/data/engine/<alias>/`, it uploads to
`s3://$BUCKET/engine/date=YYYY-MM-DD/<alias>/`:

- `state.db.gz` — a SQLite **`.backup`** snapshot (online backup API: folds the
  live WAL, committed rows only), NOT a `cp` (a raw copy of `.db`+`-wal` of a
  live-writing engine can be torn/corrupt-on-read). Stale `state.db.bak-*` and
  the live `-wal`/`-shm` are excluded by naming the file exactly.
- `gate_decisions.jsonl.gz` — the live decision log.
- `<alias>/traces/decision_trace.<SEAL>.jsonl.gz` — the **per-scan decision
  trace** used by `research/reconcile` (sim↔live). The engine appends to a plain
  `decision_trace.jsonl` (best-effort, never blocks trading); this sync **seals**
  it via an atomic rename (the engine re-opens the path each write, so it just
  recreates a fresh live file), gzips the sealed segment (~10× on the repetitive
  JSONL), and uploads it under the `date=` partition matching the segment's own
  **seal date** (so a reconcile can locate it by time window). Sealed segments
  older than `TRACE_LOCAL_RETENTION_DAYS` (default **2**) are then **pruned from
  the box once their S3 copy is confirmed** — bounding local disk while keeping
  recent reconciles fast (read straight off the box). `pull_live.py` reads these
  S3 segments and unions them with the live file, so reconciles work after the
  local copy is gone. (`SEAL_STAMP`/`TRACE_LOCAL_RETENTION_DAYS` are overridable
  env; see the script header.)
- `engine/log-filtered.gz` — journald `hl-engine` lines matching the signal
  regex (WARN/ERROR/halt/reject/FEED STALE/drift/…). KB/day, **not** the ~97
  MB/day raw 1 Hz PnL-poll + 429 + reject noise (piped `journalctl|grep|gzip`,
  never buffered). The top-level (un-namespaced) `state.db` lands under `_root/`.

The unit is niced (`Nice=19` / `IOSchedulingClass=idle`) so the nightly
`.backup`+gzip never contends with the engine on the t4g.micro. Idempotent —
re-running overwrites that day's `date=` prefix in place.

**Enable on the box** (operator, via SSM as root — files are version-controlled
at `deploy/systemd/`):

```
cp deploy/systemd/hl-engine-s3-sync.{service,timer} /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now hl-engine-s3-sync.timer
systemctl list-timers hl-engine-s3-sync.timer    # verify next run
systemctl start hl-engine-s3-sync.service        # optional: backfill today now
journalctl -u hl-engine-s3-sync.service -n 30 --no-pager
```

**Pull + inspect locally** (zero SSM):

```
scripts/pull-engine.sh                                   # -> ./data/engine/
scripts/inspect_engine_state.py ./data/engine --date 2026-06-10 --alias v1
```

Footprint: ~5–10 MB/day Tier 1 (5 slots × ~1–2 MB gzipped `state.db` + tiny gate
logs) + KB/day filtered logs → ~0.15–0.3 GB/month, **< $0.01/month** S3 storage
on top of the existing bucket; daily PUTs are negligible. Retention is optional
(the DBs are tiny); add an `engine/`-prefixed lifecycle expiry (e.g. 90 days) only
if desired. **Risk:** `state.db` contains live fills/positions — the bucket stays
private (no public access), same as the recorder archive.
